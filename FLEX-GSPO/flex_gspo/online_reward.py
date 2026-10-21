"""Text-only outcome reward adapter for a complete online audio trajectory.

SWIFT calls reward functions with the LAST assistant completion and batched
dataset columns. The last completion is not the translation: this adapter
requires the scheduler's complete ``rollout_infos.flex_trajectory`` instead.

No SWIFT, torch, local quality model, cached offline candidate library, or
offline calibration artifact is needed here. Framework-side GSPO computes
group advantages; this module returns raw, unnormalised trajectory rewards.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

from .qalco import QalcoScorer, validate_settings, validate_task
from .protocol import collect_translation
from .rewards import RewardError
from .rollout_sync import progress


LOG = logging.getLogger("flex.online_reward")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def load_reward_config(path=None):
    """Read reward-only configuration, with a path override for server runs."""
    config_path = Path(path or os.environ.get("FLEX_REWARD_CONFIG") or DEFAULT_CONFIG)
    try:
        with config_path.open("r", encoding="utf-8-sig") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RewardError(f"Cannot read reward config {config_path}: {type(exc).__name__}") from exc
    validate_settings(config)
    return config


def _object(value, name):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise RewardError(f"{name} must be a JSON object, not malformed JSON") from exc
    if not isinstance(value, dict):
        raise RewardError(f"{name} must be an object")
    return value


def _batch(value, size, name):
    # Never silently broadcast one source across unrelated candidates.
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise RewardError(f"{name} must have one item per completion ({size})")
    return value


def _trajectory(info, task):
    info = _object(info, "rollout_infos item")
    if "flex_trajectory" not in info:
        raise RewardError("Scheduler did not provide rollout_infos.flex_trajectory; refusing last-turn scoring")
    trajectory = dict(_object(info["flex_trajectory"], "flex_trajectory"))
    for flag in ("valid", "complete_input"):
        if type(trajectory.get(flag)) is not bool:
            raise RewardError(f"flex_trajectory.{flag} must be an explicit boolean")
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        raise RewardError("flex_trajectory.turns must be an array")
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise RewardError("flex_trajectory turn must be an object")
        if "response" not in turn or "audio_end_seconds" not in turn or "finish_reason" not in turn:
            raise RewardError("Scheduler turn must preserve response, audio_end_seconds and finish_reason")
        if not isinstance(turn["response"], str):
            raise RewardError("Scheduler turn.response must preserve the raw string")
        if turn["finish_reason"] not in ("stop", "eos", "length"):
            raise RewardError("Scheduler turn.finish_reason is not a supported generation result")
        end = turn["audio_end_seconds"]
        if (isinstance(end, bool) or not isinstance(end, (int, float))
                or not math.isfinite(end) or index >= len(task["chunks"])
                or abs(end - task["chunks"][index]["end_seconds"]) > 1 / 16000 + 1e-6):
            raise RewardError("Scheduler trajectory timing disagrees with task chunk schedule")
    trajectory["error"] = trajectory.get("invalid_reason") or trajectory.get("error")
    return trajectory


class OnlineQalcoReward:
    """Dependency-light delegate for a registered SWIFT ORM.

    ``flex_task`` and ``rollout_infos`` are lists matching ``completions``.
    Groups may be incomplete on a distributed worker. We batch the candidates
    that are present, but never compute group normalisation locally.
    """

    def __init__(self, config=None, *, scorer=None, log_dir=None):
        self.config = load_reward_config() if config is None else config
        self.settings = validate_settings(self.config)
        self.scorer = QalcoScorer(self.config) if scorer is None else scorer
        self.log_dir = log_dir if log_dir is not None else os.environ.get("FLEX_REWARD_LOG_DIR")
        self.last_records = []
        self._calls = 0
        self._log_lock = threading.Lock()
        LOG.warning(
            "QALCO online thresholds are exploratory: MetricX quality is a proxy, not accuracy; "
            "latency=%s is a source-clock scheduling proxy, not semantic or wall-clock delay. "
            "No offline candidate/calibration artifact is required.", self.settings["latency_metric"])

    def __call__(self, completions, flex_task=None, rollout_infos=None, **kwargs):
        if not isinstance(completions, (list, tuple)):
            raise RewardError("completions must be a batch")
        size = len(completions)
        self.last_records = []
        if not size:
            return []
        started = time.monotonic()
        progress(f"reward_begin local_candidates={size}")
        tasks = _batch(flex_task, size, "flex_task")
        infos = _batch(rollout_infos, size, "rollout_infos")
        identities = {}
        for name in ("request_id", "prompt_id"):
            values = kwargs.get(name)
            if values is not None:
                values = _batch(values, size, name)
                if any(value is not None and not isinstance(value, str) for value in values):
                    raise RewardError(f"{name} items must be strings or null")
                identities[name] = values
        groups = {}
        for index, (raw_task, info) in enumerate(zip(tasks, infos)):
            task = _object(raw_task, "flex_task item")
            validate_task(task)
            trajectory = _trajectory(info, task)
            # ID alone is insufficient: a corrupt manifest may reuse an ID
            # with different text/audio/timing. Full canonical data is safer.
            try:
                key = json.dumps(task, ensure_ascii=False, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise RewardError("flex_task contains non-JSON metadata") from exc
            group = groups.setdefault(key, {"task": task, "indices": [], "trajectories": []})
            group["indices"].append(index)
            group["trajectories"].append(trajectory)

        records = [None] * size
        for group in groups.values():
            scored = self.scorer.score_group(group["task"], group["trajectories"])
            if len(scored) != len(group["indices"]):
                raise RewardError("Trajectory scorer returned incorrect batch size")
            for index, trajectory, score in zip(group["indices"], group["trajectories"], scored):
                reward = score.get("reward")
                if (isinstance(reward, bool) or not isinstance(reward, (int, float))
                        or not math.isfinite(reward)):
                    raise RewardError("Trajectory scorer returned a non-finite reward")
                try:
                    translation = collect_translation(trajectory["turns"])
                except ValueError:
                    translation = None  # Preserve bad raw responses; never repair.
                records[index] = {
                    "task_id": group["task"].get("id"),
                    **{name: values[index] for name, values in identities.items()},
                    "candidate_index_in_local_batch": index,
                    "trajectory": copy.deepcopy(trajectory), "translation": translation, **score,
                }
        self.last_records = records
        self._calls += 1
        self._log_records(records, kwargs.get("trainer_state"))
        progress(f"reward_end local_candidates={size} elapsed={time.monotonic()-started:.1f}s")
        return [float(record["reward"]) for record in records]

    def _log_records(self, records, trainer_state):
        if not self.log_dir or not records:
            return
        rank = os.environ.get("RANK", "0")
        rank = "".join(c for c in rank if c.isalnum() or c in "-_") or "0"
        path = Path(self.log_dir) / f"rewards-rank{rank}-pid{os.getpid()}.jsonl"
        envelope = {"timestamp_unix": time.time(), "call": self._calls,
                    "global_step": getattr(trainer_state, "global_step", None)}
        try:
            with self._log_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps({**envelope, **record}, ensure_ascii=False,
                                                allow_nan=False) + "\n")
        except OSError as exc:
            raise RewardError(f"Cannot write reward diagnostics: {type(exc).__name__}") from exc
