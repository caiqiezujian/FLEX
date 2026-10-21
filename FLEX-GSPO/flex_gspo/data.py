"""Validate causal task metadata shared by preparation and online rollout.

The Swift dataset keeps this object in the reward/environment-only ``flex_task``
column. Only the system prompt and released audio chunks enter the actor.
"""

from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = 1
MODEL_SAMPLE_RATE = 16000


class DataPreparationError(ValueError):
    """An existing input cannot be represented as a valid episode."""


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataPreparationError(f"{name} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DataPreparationError(f"{name} must be a finite positive number")
    return number


def _validate_languages(source_lang: Any, target_lang: Any) -> None:
    if (source_lang, target_lang) not in (("en", "zh"), ("zh", "en")):
        raise DataPreparationError("Languages must be opposite en/zh directions")


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataPreparationError(f"{name} must be a string or null")
    return value.strip() or None


def _validate_transcript_provenance(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DataPreparationError("transcript_provenance must be an object or null")
    if not isinstance(value.get("kind"), str) or not value["kind"].strip():
        raise DataPreparationError("transcript_provenance.kind must be a non-empty string")
    if "is_gold" in value and not isinstance(value["is_gold"], bool):
        raise DataPreparationError("transcript_provenance.is_gold must be boolean")
    if value["kind"] == "asr" and value.get("is_gold") is True:
        raise DataPreparationError("An ASR transcript must not be marked is_gold=true")
    if "segment_granularity" in value and (
        not isinstance(value["segment_granularity"], str) or not value["segment_granularity"].strip()
    ):
        raise DataPreparationError("transcript_provenance.segment_granularity must be a non-empty string")
    return dict(value)


def _validate_source_segments(value: Any, duration: float) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise DataPreparationError("source_segments must be a list or null")
    segments = []
    previous_end = 0.0
    for index, segment in enumerate(value):
        if not isinstance(segment, dict):
            raise DataPreparationError(f"source_segments[{index}] must be an object")
        start = segment.get("start_seconds")
        if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
            raise DataPreparationError(f"source_segments[{index}] has invalid start_seconds")
        end = _positive_finite(segment.get("end_seconds"), f"source_segments[{index}] end_seconds")
        if start + 1e-8 < previous_end or end <= start or end > duration:
            raise DataPreparationError(f"source_segments[{index}] must be ordered, non-overlapping, and within real duration")
        text = segment.get("text")
        if _optional_text(text, f"source_segments[{index}] text") is None:
            raise DataPreparationError(f"source_segments[{index}] text must be non-empty")
        # Preserve original word tokens, including leading spaces and punctuation.
        # Stripping each token destroys faithful prefix reconstruction (e.g. "I'm").
        segments.append({"start_seconds": float(start), "end_seconds": end, "text": text})
        previous_end = end
    return segments


def validate_task(task: Any) -> dict[str, Any]:
    """Validate timing and supervision; do not require files to exist on this host."""
    if not isinstance(task, dict):
        raise DataPreparationError("Task must be an object")
    if task.get("schema_version") != SCHEMA_VERSION or isinstance(task.get("schema_version"), bool):
        raise DataPreparationError(f"schema_version must be {SCHEMA_VERSION}")
    for name in ("id", "audio_path"):
        if not isinstance(task.get(name), str) or not task[name].strip():
            raise DataPreparationError(f"Task {name} must be a non-empty string")
    _validate_languages(task.get("source_lang"), task.get("target_lang"))
    duration = _positive_finite(task.get("duration_seconds"), "duration_seconds")
    if task.get("sample_rate") != MODEL_SAMPLE_RATE:
        raise DataPreparationError(f"sample_rate must be {MODEL_SAMPLE_RATE} (model input rate)")
    chunks = task.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise DataPreparationError("Task chunks must be a non-empty list")
    previous_end = 0.0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise DataPreparationError(f"Chunk {index} must be an object")
        start = chunk.get("start_seconds")
        if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
            raise DataPreparationError(f"Chunk {index} has invalid start_seconds")
        end = _positive_finite(chunk.get("end_seconds"), f"chunk {index} end_seconds")
        padded = _positive_finite(chunk.get("input_seconds"), f"chunk {index} input_seconds")
        if not math.isclose(start, previous_end, rel_tol=0, abs_tol=1e-8):
            raise DataPreparationError(f"Chunk {index} has a gap or overlap")
        if end <= start or end > duration:
            raise DataPreparationError(f"Chunk {index} must end after its start and within real duration")
        if padded + 1e-8 < end - start:
            raise DataPreparationError(f"Chunk {index} input_seconds is shorter than its real interval")
        previous_end = end
    if not math.isclose(previous_end, duration, rel_tol=0, abs_tol=1e-8):
        raise DataPreparationError("Chunks must cover the complete real audio duration")
    source = _optional_text(task.get("source_text"), "source_text")
    _optional_text(task.get("reference_text"), "reference_text")
    _validate_source_segments(task.get("source_segments"), duration)
    _validate_transcript_provenance(task.get("transcript_provenance"))
    expected_ready = bool(source)
    if not isinstance(task.get("quality_ready"), bool) or task["quality_ready"] != expected_ready:
        raise DataPreparationError("quality_ready must reflect non-empty source_text (reference_text is optional)")
    return task
