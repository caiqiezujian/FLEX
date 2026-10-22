"""QALCO reward entry points.

Quality comes from the separate COMET22 service in v2. MetricX v1 remains
available only to reproduce historical stopped-run diagnostics.
Infrastructure failures must never become bad-policy rewards or fake quality.
The ``qalco`` provider combines quality, latency and length cost into a
complete-trajectory reward without loading MetricX on the training GPUs.
"""


class RewardError(RuntimeError):
    """Infrastructure/config/scoring failure, not a model action failure."""


def _require_qalco(config):
    if not isinstance(config, dict):
        raise RewardError("reward configuration must be an object")
    settings = config.get("reward", config)
    if not isinstance(settings, dict) or settings.get("provider") != "qalco":
        raise RewardError("reward.provider must be qalco")


def validate_reward_setup(tasks, config):
    """Validate tasks and settings without loading a model or making requests."""
    _require_qalco(config)
    from .qalco import validate_tasks
    return validate_tasks(tasks, config)


def build_scorer(config):
    """Build the QALCO scorer; incomplete service configuration fails clearly."""
    _require_qalco(config)
    from .qalco import QalcoScorer
    return QalcoScorer(config)
