"""Service-only MetricX text scoring; no model weights or GPU are loaded here.

The service adapter returns one raw MetricX-24 error score in [0, 25] for each
candidate, in the same order. The separate deployment owns tokenizer/input-size
checks and model inference; this client does not truncate text or invent scores.
"""

import math
from urllib.parse import urlsplit

from .rewards import RewardError


def validate_metricx_settings(settings):
    """Check configuration only; no model imports or service requests."""
    if not isinstance(settings, dict):
        raise RewardError("reward.metricx must be an object")
    allowed = {"backend", "mode", "service_url", "service_id"}
    if set(settings) - allowed:
        raise RewardError("Unsupported MetricX setting; allowed fields: " + ", ".join(sorted(allowed)))
    normalized = dict(settings)
    normalized.setdefault("backend", "service")
    normalized.setdefault("mode", "qe")
    if normalized["backend"] != "service":
        raise RewardError("metricx.backend must be service")
    if normalized["mode"] not in ("qe", "reference"):
        raise RewardError("metricx.mode must be qe or reference")
    for key in ("service_url", "service_id"):
        normalized.setdefault(key, "")
        if not isinstance(normalized[key], str):
            raise RewardError(f"metricx.{key} must be a string (empty is allowed only for schema validation)")
        if normalized[key] != normalized[key].strip():
            raise RewardError(f"metricx.{key} must not contain surrounding whitespace")
    url = normalized["service_url"]
    if url:
        try:
            parsed = urlsplit(url)
            _ = parsed.port
        except ValueError as exc:
            raise RewardError("metricx.service_url must be a valid HTTP(S) URL") from exc
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or any(c.isspace() for c in url)):
            raise RewardError("metricx.service_url must be HTTP(S), without credentials/query/fragment")
    return normalized


def ensure_metricx_ready(config):
    """Check configuration and client dependency before GPU work; no requests."""
    if not isinstance(config, dict):
        raise RewardError("reward configuration must be an object")
    reward = config.get("reward", config)
    if not isinstance(reward, dict) or reward.get("provider") != "qalco":
        raise RewardError("reward.provider must be qalco")
    settings = validate_metricx_settings(reward.get("metricx"))
    from .metricx_service import ensure_service_ready
    ensure_service_ready(settings)


def build_metricx(settings):
    return ServiceMetricX(settings)


def validate_metricx_text(source, hypothesis, reference=None, mode="qe"):
    """Text validity only; service-side tokenization must never silently truncate."""
    for name, value in (("source", source), ("hypothesis", hypothesis)):
        if not isinstance(value, str) or not value.strip():
            raise RewardError(f"MetricX {name} must be nonempty text")
    if mode == "reference":
        if not isinstance(reference, str) or not reference.strip():
            raise RewardError("reference-mode MetricX requires reference text")
    elif mode != "qe":
        raise RewardError("MetricX mode must be qe or reference")


class ServiceMetricX:
    """Text-only score interface; no torch, tokenizer or model is loaded here."""

    def __init__(self, settings):
        self.settings = validate_metricx_settings(settings)
        from .metricx_service import ensure_service_ready
        ensure_service_ready(self.settings)

    def score(self, source_text, candidate_text, reference_text=None):
        """Single candidate -> one raw MetricX error score."""
        return self.score_batch(source_text, [candidate_text], reference_text)[0]

    def score_batch(self, source, hypotheses, reference=None):
        if not isinstance(hypotheses, list):
            raise RewardError("MetricX candidates must be a list of text strings")
        if not hypotheses:
            return []
        for text in hypotheses:
            validate_metricx_text(source, text, reference, self.settings["mode"])
        from . import metricx_service
        try:
            raw = metricx_service.request_metricx_scores(
                source, list(hypotheses),
                reference if self.settings["mode"] == "reference" else None,
                mode=self.settings["mode"], service_url=self.settings["service_url"],
                service_id=self.settings["service_id"])
        except metricx_service.MetricXServiceError:
            # The HTTP adapter has already removed bodies, texts and URL secrets.
            raise
        except Exception as exc:
            # Don't copy response bodies/URLs/credentials into training logs.
            raise RewardError(
                f"MetricX service scoring failed ({type(exc).__name__}); no reward assigned.") from None
        if not isinstance(raw, list) or len(raw) != len(hypotheses):
            raise RewardError("MetricX service must return one ordered score per candidate")
        for value in raw:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 25):
                raise RewardError("MetricX service scores must be finite numbers in [0, 25]")
        return [float(value) for value in raw]
