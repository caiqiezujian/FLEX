"""Infrastructure validation shared by online rollout and outcome scoring.

The source clock never advances beyond the last audio sample. A single EOF
observation may produce a final delta, with its measured host elapsed time kept
separately. A scorer may charge an explicit deterministic EOF scheduling cost;
that cost is not this measurement and must not be called wall-clock latency.
"""
from __future__ import annotations

import math

from .rewards import RewardError

TIME_TOLERANCE = 1 / 16000 + 1e-6
TRAJECTORY_SCHEMA_VERSION = 2


def _finite_nonnegative(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise RewardError(f"{name} must be a finite nonnegative number")
    return float(value)


def validate_turn_schedule(task, turns):
    """Validate an audio prefix followed by at most one explicit EOF response.

    Missing ``kind`` means a legacy audio turn, never an inferred EOF turn.
    Prefixes are intentionally accepted: early protocol/length failures must be
    logged without being confused with corrupt infrastructure. The caller owns
    policy validity and, for v2 scoring, the requirement for a completed EOF.
    """
    if not isinstance(turns, list):
        raise RewardError("trajectory turns must be an array")
    chunks = task.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RewardError("task requires nonempty audio chunks")
    duration = _finite_nonnegative(task.get("duration_seconds"), "duration_seconds")
    count = len(chunks)
    if len(turns) > count + 1:
        raise RewardError("trajectory contains more than one EOF finalization turn")
    eof_elapsed = None
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise RewardError("trajectory turn must be an object")
        kind = turn.get("kind", "audio")
        expected_kind = "audio" if index < count else "eof"
        if kind != expected_kind:
            raise RewardError("trajectory turn kind disagrees with the audio/EOF schedule")
        end = _finite_nonnegative(turn.get("audio_end_seconds"), "turn.audio_end_seconds")
        expected_end = float(chunks[index]["end_seconds"]) if index < count else duration
        if abs(end - expected_end) > TIME_TOLERANCE:
            raise RewardError("trajectory timing disagrees with task chunk schedule")
        if kind == "eof":
            eof_elapsed = _finite_nonnegative(turn.get("eof_elapsed_seconds"), "eof_elapsed_seconds")
        elif "eof_elapsed_seconds" in turn:
            raise RewardError("eof_elapsed_seconds is only valid on the EOF turn")
    complete = len(turns) >= count
    eof_complete = len(turns) == count + 1
    return {
        "audio_turns": min(len(turns), count),
        "complete_input": complete,
        "eof_completed": eof_complete,
        "episode_complete": eof_complete,
        "eof_elapsed_seconds": eof_elapsed,
    }
