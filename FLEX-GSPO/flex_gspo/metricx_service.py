"""MetricX-24 HTTP client: text batches in, ordered raw error scores out.

QE:        POST <base>/metricx24/qe  {srcs, mts}
Reference: POST <base>/metricx24     {srcs, mts, refs}
The service returns {scores: [...]}. No model weights are loaded here.
"""
from __future__ import annotations

import logging
import math
import time

try:
    import requests
except ImportError:
    requests = None

from .rewards import RewardError


LOG = logging.getLogger("flex.metricx")
REQUEST_TIMEOUT = (10, 180)
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1.0
RETRYABLE_STATUS = frozenset((429, 502, 503, 504))


class MetricXServiceError(RewardError):
    """Sanitized service failure; never contains texts or a response body."""


def ensure_service_ready(settings):
    """Validate configuration/dependency only, without contacting the endpoint."""
    from .metricx import validate_metricx_settings
    settings = validate_metricx_settings(settings)
    for key in ("service_url", "service_id"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise RewardError(f"Configure reward.metricx.{key} before service scoring.")
    if requests is None:
        raise RewardError("MetricX HTTP client requires requests; install requirements.txt with python3.")


def _parse_scores(response, expected):
    try:
        result = response.json()
    except ValueError:
        raise MetricXServiceError("MetricX returned invalid JSON; no reward assigned.") from None
    if not isinstance(result, dict) or not isinstance(result.get("scores"), list):
        raise MetricXServiceError("MetricX response must contain a scores array; no reward assigned.")
    scores = result["scores"]
    if len(scores) != expected:
        raise MetricXServiceError("MetricX score count does not match candidate count; no reward assigned.")
    for value in scores:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= 25 or not math.isfinite(value)):
            raise MetricXServiceError("MetricX scores must be finite numbers in [0, 25]; no reward assigned.")
    return [float(value) for value in scores]


def request_metricx_scores(
    source_text: str,
    candidate_texts: list[str],
    reference_text: str | None = None,
    *,
    mode: str = "qe",
    service_url: str = "",
    service_id: str = "",
) -> list[float]:
    """Score one source's complete translations, preserving their order.

    service_url is the base URL, not an endpoint. service_id is an operator's
    deployment label used by run provenance; this scores-only response cannot
    prove remote checkpoint identity or independently verify semantic order.
    The array-order contract is supplied by the service. Input text is neither
    stripped nor truncated. No references are sent in QE mode.
    """
    settings = {"backend": "service", "mode": mode,
                "service_url": service_url, "service_id": service_id}
    ensure_service_ready(settings)
    from .metricx import validate_metricx_text
    if not isinstance(source_text, str) or not source_text.strip():
        raise RewardError("MetricX source must be nonempty text")
    if not isinstance(candidate_texts, list):
        raise RewardError("MetricX candidates must be a list of text strings")
    for candidate in candidate_texts:
        validate_metricx_text(source_text, candidate, reference_text, mode)
    if not candidate_texts:
        return []
    payload = {"srcs": [source_text] * len(candidate_texts), "mts": list(candidate_texts)}
    if mode == "reference":
        payload["refs"] = [reference_text] * len(candidate_texts)
    endpoint = "/metricx24/qe" if mode == "qe" else "/metricx24"
    url = service_url.rstrip("/") + endpoint
    with requests.Session() as session:
        # This private endpoint must not inherit HTTP(S)_PROXY or .netrc auth.
        session.trust_env = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = None
            try:
                response = session.post(url, json=payload, timeout=REQUEST_TIMEOUT,
                                        allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == MAX_ATTEMPTS:
                    raise MetricXServiceError(
                        f"MetricX transport failed ({type(exc).__name__}) after {attempt} attempts; "
                        "no reward assigned.") from None
                LOG.warning("MetricX transient transport failure (%s), retry %s/%s",
                            type(exc).__name__, attempt + 1, MAX_ATTEMPTS)
            except requests.RequestException as exc:
                raise MetricXServiceError(
                    f"MetricX request failed ({type(exc).__name__}); no reward assigned.") from None
            else:
                try:
                    status = response.status_code
                    if status == 200:
                        return _parse_scores(response, len(candidate_texts))
                    if status not in RETRYABLE_STATUS or attempt == MAX_ATTEMPTS:
                        raise MetricXServiceError(
                            f"MetricX HTTP {status} after {attempt} attempt(s); no reward assigned.")
                    LOG.warning("MetricX HTTP %s, retry %s/%s", status, attempt + 1, MAX_ATTEMPTS)
                finally:
                    response.close()
            time.sleep(RETRY_DELAY_SECONDS)
    raise MetricXServiceError("MetricX request ended without scores; no reward assigned.")
