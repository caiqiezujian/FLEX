"""COMET22 text-service client; no local model or training GPU is used.

Contract: POST the FULL service URL with ordered srcs/mts/refs arrays. Accept
only a flat score array or an object containing that array under response_key.
Scores are raw wmt22-comet-da values in [0, 1], higher is better, not calibrated
accuracy probabilities. This client never truncates text or fabricates scores.
"""
from __future__ import annotations

import logging
import math
import time
from urllib.parse import urlsplit

try:
    import requests
except ImportError:
    requests = None

from .rewards import RewardError

LOG = logging.getLogger("flex.comet22")
DEFAULT_URL = "http://10.185.1.72:6668/22"
RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))


class COMET22ServiceError(RewardError):
    """Infrastructure failure, never a negative policy reward."""


def validate_comet22_settings(settings):
    if not isinstance(settings, dict):
        raise RewardError("reward.comet22 must be an object")
    allowed = {"backend", "service_url", "service_id", "model_id", "response_key",
               "timeout_seconds", "max_attempts"}
    if set(settings) - allowed:
        raise RewardError("Unsupported COMET22 setting; allowed fields: " + ", ".join(sorted(allowed)))
    normalized = {
        "backend": "service", "service_url": DEFAULT_URL,
        "service_id": "comet22-internal-6668", "model_id": "Unbabel/wmt22-comet-da",
        "response_key": "scores", "timeout_seconds": 300, "max_attempts": 2,
        **settings,
    }
    if normalized["backend"] != "service":
        raise RewardError("comet22.backend must be service")
    for name in ("service_url", "service_id", "model_id", "response_key"):
        value = normalized[name]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise RewardError(f"comet22.{name} must be nonempty text without surrounding whitespace")
    if normalized["response_key"] in ("system_score", "mean", "average", "score"):
        raise RewardError("comet22.response_key must identify per-candidate scores, not an aggregate")
    url = normalized["service_url"]
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError:
        raise RewardError("comet22.service_url must be a valid full HTTP(S) endpoint") from None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c.isspace() for c in url)):
        raise RewardError("comet22.service_url must be HTTP(S), without credentials/query/fragment")
    timeout = normalized["timeout_seconds"]
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 1 <= timeout <= 600):
        raise RewardError("comet22.timeout_seconds must be between 1 and 600")
    attempts = normalized["max_attempts"]
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise RewardError("comet22.max_attempts must be an integer from 1 to 3")
    return normalized


def ensure_comet22_ready(settings):
    """Configuration and dependency validation only; makes no HTTP request."""
    normalized = validate_comet22_settings(settings)
    if requests is None:
        raise RewardError("COMET22 HTTP client requires requests; install requirements.txt with python3")
    return normalized


def _parse_scores(body, expected, response_key="scores"):
    if isinstance(body, dict):
        if body.get("error"):
            raise COMET22ServiceError("COMET22 response reports an error; no reward assigned.")
        # Respect explicit service-side truncation reports. Absence of this
        # optional field does NOT prove that remote tokenization is untruncated.
        truncated = body.get("truncated", False)
        if isinstance(truncated, list):
            if len(truncated) != expected or any(type(value) is not bool for value in truncated):
                raise COMET22ServiceError("COMET22 truncation metadata is invalid; no reward assigned.")
            truncated = any(truncated)
        if type(truncated) is not bool or truncated:
            raise COMET22ServiceError("COMET22 reports truncated/ambiguous inputs; no reward assigned.")
        scores = body.get(response_key)
    else:
        scores = body
    if not isinstance(scores, list) or len(scores) != expected:
        raise COMET22ServiceError(
            "COMET22 must return a flat ordered score array or an object with the configured "
            "response_key array, one score per candidate; no reward assigned.")
    for value in scores:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise COMET22ServiceError(
                "COMET22 scores must be finite numeric values in [0, 1], not percentages or aggregates; "
                "no reward assigned.")
    return [float(value) for value in scores]


class ServiceCOMET22:
    """Reference-based COMET22 scoring; transport errors always fail closed."""

    def __init__(self, settings):
        self.settings = ensure_comet22_ready(settings)

    def score(self, source_text, candidate_text, reference_text):
        return self.score_batch(source_text, [candidate_text], reference_text)[0]

    def score_batch(self, source, hypotheses, reference):
        for name, text in (("source", source), ("reference", reference)):
            if not isinstance(text, str) or not text.strip():
                raise RewardError(f"COMET22 {name} must be nonempty text")
        if not isinstance(hypotheses, list):
            raise RewardError("COMET22 hypotheses must be a list of text strings")
        if any(not isinstance(text, str) or not text.strip() for text in hypotheses):
            raise RewardError("COMET22 hypotheses must contain nonempty text strings")
        if not hypotheses:
            return []
        payload = {"srcs": [source] * len(hypotheses), "mts": list(hypotheses),
                   "refs": [reference] * len(hypotheses)}
        settings = self.settings
        attempts = settings["max_attempts"]
        with requests.Session() as session:
            session.trust_env = False  # Private service: no proxies or .netrc.
            for attempt in range(1, attempts + 1):
                try:
                    response = session.post(
                        settings["service_url"], json=payload,
                        timeout=(10, settings["timeout_seconds"]), allow_redirects=False)
                except (requests.Timeout, requests.ConnectionError) as exc:
                    if attempt == attempts:
                        raise COMET22ServiceError(
                            f"COMET22 transport failed ({type(exc).__name__}) after {attempt} attempts; "
                            "no reward assigned.") from None
                    LOG.warning("COMET22 transient transport failure (%s), retry %d/%d",
                                type(exc).__name__, attempt + 1, attempts)
                except requests.RequestException as exc:
                    raise COMET22ServiceError(
                        f"COMET22 request failed ({type(exc).__name__}); no reward assigned.") from None
                else:
                    try:
                        status = response.status_code
                        if status == 200:
                            try:
                                body = response.json()
                            except ValueError:
                                raise COMET22ServiceError(
                                    "COMET22 returned invalid JSON; no reward assigned.") from None
                            return _parse_scores(body, len(hypotheses), settings["response_key"])
                        if status not in RETRYABLE_STATUS or attempt == attempts:
                            raise COMET22ServiceError(
                                f"COMET22 HTTP {status} after {attempt} attempt(s); no reward assigned.")
                        LOG.warning("COMET22 HTTP %d, retry %d/%d", status, attempt + 1, attempts)
                    finally:
                        response.close()
                time.sleep(float(attempt))
        raise COMET22ServiceError("COMET22 request ended without scores; no reward assigned.")
