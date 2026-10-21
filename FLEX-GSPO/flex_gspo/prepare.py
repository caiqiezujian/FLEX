"""Build online-RL tasks from raw transcripts and untouched WAV files.

This module has no model, GPU, ffmpeg, or soundfile dependency. It scans WAV
headers and payload sizes, not sample values, and creates *time plans*, not
physical chunk files or fixed-policy trajectories. A disk-backed key index and
a bounded metadata worker queue keep memory independent of dataset row count.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .audio import encode_chunk_ref
from .data import validate_task
from .prompts import SYSTEM_PROMPT


LOGGER = logging.getLogger(__name__)
OUTPUT_NAMES = ("train.jsonl", "prepare_report.json", "rejected.jsonl", "duplicates.jsonl")


class PreparationError(ValueError):
    """A row or preparation request cannot safely become a training task."""


@dataclass(frozen=True)
class PrepareOptions:
    input_jsonl: str = "/data/flex/datasets/1019/en2zh_train_gspo_1.jsonl"
    audio_dir: str = "/data/flex/datasets/data_001"
    output_dir: str = "/data/flex/datasets/gspo_online_prepared"
    chunk_seconds: int = 1
    workers: int = 8
    progress_seconds: float = 5.0
    skip_invalid: bool = False
    limit: int | None = None
    max_duration_seconds: float | None = 40.0
    system_prompt_file: str | None = None


def _write_json_line(stream, value: Any) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def probe_wav(path: str | Path) -> dict[str, Any]:
    """Read RIFF/RF64 PCM/float WAV metadata and detect a truncated payload.

    Only header-sized reads are performed. This intentionally does not decode
    every sample and therefore does not claim to detect clipping or bad speech.
    Unknown/compressed WAV encodings are rejected rather than guessed.
    """
    path = Path(path)
    if not path.is_file():
        raise PreparationError(f"Audio file does not exist or is not a file: {path}")
    initial_stat = path.stat()
    file_size = initial_stat.st_size
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[8:] != b"WAVE" or header[:4] not in (b"RIFF", b"RF64"):
            raise PreparationError("Expected a little-endian RIFF/RF64 WAV file")
        rf64 = header[:4] == b"RF64"
        riff_size = struct.unpack_from("<I", header, 4)[0]
        if not rf64 and (riff_size < 4 or riff_size + 8 > file_size):
            raise PreparationError("Truncated or invalid RIFF container size")
        container_end = file_size if rf64 else riff_size + 8
        format_info = None
        data_size = None
        rf64_data_size = None
        offset = 12
        while offset + 8 <= container_end:
            stream.seek(offset)
            chunk_header = stream.read(8)
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            payload_offset = offset + 8
            if chunk_id == b"ds64":
                if not rf64 or chunk_size < 28 or payload_offset + chunk_size > file_size:
                    raise PreparationError("Invalid RF64 ds64 header")
                raw = stream.read(28)
                riff64, rf64_data_size, _sample_count, _table_length = struct.unpack("<QQQI", raw)
                if riff64 + 8 > file_size or riff64 < 4:
                    raise PreparationError("Truncated RF64 container")
                container_end = riff64 + 8
            elif chunk_size == 0xFFFFFFFF:
                if chunk_id != b"data" or rf64_data_size is None:
                    raise PreparationError("Unsupported or malformed extended WAV chunk size")
                chunk_size = rf64_data_size
            if payload_offset + chunk_size > container_end:
                raise PreparationError(f"Truncated WAV chunk {chunk_id!r}")
            if chunk_id == b"fmt ":
                if chunk_size < 16:
                    raise PreparationError("WAV fmt chunk is too short")
                raw = stream.read(min(chunk_size, 40))
                encoding, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", raw)
                if encoding == 0xFFFE:
                    if len(raw) < 40 or struct.unpack_from("<H", raw, 16)[0] < 22:
                        raise PreparationError("Invalid WAVE_FORMAT_EXTENSIBLE header")
                    # Standard subtype GUID: {format}-0000-0010-8000-00aa00389b71.
                    if raw[28:40] != bytes.fromhex("00001000800000aa00389b71"):
                        raise PreparationError("Unsupported extensible WAV subtype")
                    encoding = struct.unpack_from("<I", raw, 24)[0]
                if encoding not in (1, 3):
                    raise PreparationError(f"Unsupported compressed WAV encoding {encoding}; use PCM or IEEE float WAV")
                valid_bits = (8, 16, 24, 32) if encoding == 1 else (32, 64)
                if channels <= 0 or sample_rate <= 0 or bits not in valid_bits:
                    raise PreparationError("Invalid WAV channels, sample rate, or bit depth")
                if block_align != channels * (bits // 8) or byte_rate != sample_rate * block_align:
                    raise PreparationError("Inconsistent WAV block alignment or byte rate")
                format_info = {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bits_per_sample": bits,
                    "encoding": "PCM" if encoding == 1 else "IEEE_FLOAT",
                    "block_align": block_align,
                }
            elif chunk_id == b"data":
                if data_size is not None:
                    raise PreparationError("Multiple WAV data chunks are not supported")
                data_size = chunk_size
            offset = payload_offset + chunk_size + (chunk_size % 2)
        if format_info is None or data_size is None:
            raise PreparationError("WAV must contain fmt and data chunks")
        if data_size <= 0 or data_size % format_info["block_align"]:
            raise PreparationError("Empty WAV or incomplete sample frame")
        frame_count = data_size // format_info.pop("block_align")
        final_stat = path.stat()
        if (file_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise PreparationError("Audio changed while metadata was being read; retry after the write is complete")
        return {
            **format_info,
            "num_frames": frame_count,
            "duration_seconds": frame_count / format_info["sample_rate"],
            "file_size_bytes": file_size,
            "mtime_ns": initial_stat.st_mtime_ns,
            "validation": "wav_header_and_payload_bounds",
        }


def plan_chunks(duration_seconds: float, chunk_seconds: int = 1) -> list[dict[str, float]]:
    """Cover real audio exactly; silence padding never extends the source clock."""
    if isinstance(chunk_seconds, bool) or not isinstance(chunk_seconds, int) or not 1 <= chunk_seconds <= 8:
        raise PreparationError("chunk_seconds must be an integer from 1 to 8, matching SFT whole-second chunks")
    if isinstance(duration_seconds, bool) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise PreparationError("duration_seconds must be finite and positive")
    count = math.ceil(duration_seconds / chunk_seconds)
    return [
        {
            "start_seconds": float(index * chunk_seconds),
            "end_seconds": min(float((index + 1) * chunk_seconds), duration_seconds),
            "input_seconds": float(chunk_seconds),
        }
        for index in range(count)
    ]


def normalize_record(record: Any, audio_root: Path) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise PreparationError("Each nonempty JSONL line must contain an object")
    key = record.get("key")
    if not isinstance(key, str) or not key or key != key.strip():
        raise PreparationError("key must be a nonempty string without surrounding whitespace")
    if key in (".", "..") or any(char in key for char in ("/", "\\", "\x00")):
        raise PreparationError("key must be a filename stem, not a path")
    if record.get("lang") != "en":
        raise PreparationError("Only lang='en' English-to-Chinese records are supported in this first version")
    texts = {}
    for language in ("en", "zh"):
        entry = record.get(language)
        value = entry.get("txt") if isinstance(entry, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise PreparationError(f"{language}.txt must be a nonempty string")
        texts[language] = value.strip()
    audio_path = audio_root / f"{key}.wav"
    # A malicious key/drive prefix must never escape the explicitly chosen root.
    try:
        resolved_audio = audio_path.resolve()
        resolved_audio.relative_to(audio_root)
    except (ValueError, OSError) as exc:
        raise PreparationError("Audio path must resolve inside --audio-dir") from exc
    return {
        "id": key,
        "audio_path": str(resolved_audio),
        "source_text": texts["en"],
        "reference_text": texts["zh"],
    }


def build_row(record: dict[str, Any], *, chunk_seconds: int, max_duration_seconds: float | None,
              system_prompt: str) -> dict[str, Any]:
    metadata = probe_wav(record["audio_path"])
    duration = metadata["duration_seconds"]
    if max_duration_seconds is not None and duration > max_duration_seconds:
        raise PreparationError(f"Audio duration {duration:.3f}s exceeds --max-duration-seconds {max_duration_seconds}")
    task = {
        "schema_version": 1,
        **record,
        "source_lang": "en",
        "target_lang": "zh",
        "duration_seconds": duration,
        "audio_size_bytes": metadata["file_size_bytes"],
        "audio_mtime_ns": metadata["mtime_ns"],
        "sample_rate": 16000,
        "chunks": plan_chunks(duration, chunk_seconds),
        "quality_ready": True,
        "source_segments": None,
        "transcript_provenance": {"kind": "provided", "is_gold": False},
        "reference_provenance": {"kind": "machine_translation", "is_gold": False},
        "original_audio": metadata,
    }
    validate_task(task)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<audio>"},
        ],
        "audios": [encode_chunk_ref(task, 0)],
        # Keep metadata out of chat content and avoid Arrow heterogeneous structs.
        "flex_task": json.dumps(task, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    }


def _validate_options(options: PrepareOptions) -> tuple[Path, Path, Path, str]:
    input_path = Path(options.input_jsonl).expanduser().resolve()
    audio_root = Path(options.audio_dir).expanduser().resolve()
    output_root = Path(options.output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise PreparationError(f"Input JSONL does not exist: {input_path}")
    if not audio_root.is_dir():
        raise PreparationError(f"Audio directory does not exist: {audio_root}")
    if output_root == audio_root or output_root == input_path.parent:
        raise PreparationError("Use a dedicated output directory, separate from input JSONL/audio directories")
    if options.workers < 1 or options.workers > 256:
        raise PreparationError("workers must be between 1 and 256")
    if not math.isfinite(options.progress_seconds) or options.progress_seconds <= 0:
        raise PreparationError("progress_seconds must be finite and positive")
    if options.limit is not None and options.limit <= 0:
        raise PreparationError("limit must be positive")
    if options.max_duration_seconds is not None and (
        not math.isfinite(options.max_duration_seconds) or options.max_duration_seconds <= 0
    ):
        raise PreparationError("max_duration_seconds must be finite and positive")
    plan_chunks(1.0, options.chunk_seconds)
    system_prompt = SYSTEM_PROMPT
    if options.system_prompt_file:
        system_prompt = Path(options.system_prompt_file).read_text(encoding="utf-8-sig").strip()
        if system_prompt != SYSTEM_PROMPT:
            raise PreparationError("system_prompt_file must exactly match the preserved EN-to-ZH SFT prompt; "
                                   "the online scheduler does not accept an untrained prompt override")
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (output_root / name).exists()]
    if existing:
        raise PreparationError(f"Refusing to overwrite existing preparation outputs {existing}; choose a new --output-dir")
    return input_path, audio_root, output_root, system_prompt


def prepare_dataset(options: PrepareOptions) -> dict[str, Any]:
    """Stream, deduplicate, probe, and atomically publish a Swift task manifest.

    Invalid rows create a detailed rejection report. Unless ``skip_invalid`` is
    explicitly enabled, *no training manifest is published* when any row fails.
    The first eligible occurrence of a key wins; later duplicate keys are logged
    without requiring their audio to be probed again.
    """
    input_path, audio_root, output_root, system_prompt = _validate_options(options)
    lock_path = output_root / ".prepare.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PreparationError(f"Preparation lock already exists: {lock_path}; check for another running/interrupted process") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            _write_json_line(stream, {"pid": os.getpid(), "input_jsonl": str(input_path)})
        # Recheck after obtaining the lock so concurrent invocations cannot
        # overwrite outputs produced between the initial check and this point.
        input_path, audio_root, output_root, system_prompt = _validate_options(options)
        return _prepare_validated(options, input_path, audio_root, output_root, system_prompt)
    finally:
        lock_path.unlink(missing_ok=True)


def _prepare_validated(options: PrepareOptions, input_path: Path, audio_root: Path,
                       output_root: Path, system_prompt: str) -> dict[str, Any]:
    started = time.monotonic()
    last_log = started
    input_bytes = input_path.stat().st_size
    counters = {"physical_lines": 0, "rows_read": 0, "unique_rows": 0, "accepted_rows": 0,
                "rejected_rows": 0, "duplicate_rows": 0, "conflicting_duplicate_rows": 0,
                "blank_lines": 0, "bytes_read": 0, "audio_seconds": 0.0, "chunks": 0}
    LOGGER.info("Preparing online RL input: jsonl=%s audio_dir=%s output_dir=%s workers=%s chunk=%ss",
                input_path, audio_root, output_root, options.workers, options.chunk_seconds)
    LOGGER.info("No waveform copies, chunk files, model calls, or offline candidate trajectories will be created")

    def progress(force=False):
        nonlocal last_log
        now = time.monotonic()
        if force or now - last_log >= options.progress_seconds:
            percent = 100.0 * counters["bytes_read"] / max(input_bytes, 1)
            LOGGER.info("Prepare progress: read=%s unique=%s accepted=%s rejected=%s duplicates=%s "
                        "input=%.1f%% audio=%.3fh elapsed=%.1fs",
                        counters["rows_read"], counters["unique_rows"], counters["accepted_rows"],
                        counters["rejected_rows"], counters["duplicate_rows"], percent,
                        counters["audio_seconds"] / 3600, now - started)
            last_log = now

    # Own only this exact temporary directory; never modify source files.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_root) as temporary:
        stage = Path(temporary)
        connection = sqlite3.connect(stage / "keys.sqlite3")
        connection.execute("PRAGMA cache_size=-4096")
        connection.execute("CREATE TABLE seen (key TEXT PRIMARY KEY, line INTEGER NOT NULL, digest TEXT NOT NULL)")
        try:
            with (stage / "train.jsonl").open("w", encoding="utf-8", newline="\n") as manifest, \
                    (stage / "rejected.jsonl").open("w", encoding="utf-8", newline="\n") as rejected, \
                    (stage / "duplicates.jsonl").open("w", encoding="utf-8", newline="\n") as duplicates, \
                    concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
                queue = deque()

                def reject(line_number, key, error):
                    counters["rejected_rows"] += 1
                    _write_json_line(rejected, {"line": line_number, "key": key, "error": str(error)})
                    if counters["rejected_rows"] <= 5:
                        LOGGER.warning("Rejected line=%s key=%s: %s", line_number, key, error)

                def drain_one():
                    line_number, key, future = queue.popleft()
                    while True:
                        try:
                            row = future.result(timeout=options.progress_seconds)
                            break
                        except concurrent.futures.TimeoutError:
                            progress(force=True)
                        except (PreparationError, OSError, ValueError, struct.error) as exc:
                            reject(line_number, key, exc)
                            progress()
                            return
                    _write_json_line(manifest, row)
                    task = json.loads(row["flex_task"])
                    counters["accepted_rows"] += 1
                    counters["audio_seconds"] += task["duration_seconds"]
                    counters["chunks"] += len(task["chunks"])
                    progress()

                with input_path.open("rb") as source:
                    for line_number, raw_line in enumerate(source, 1):
                        counters["physical_lines"] = line_number
                        counters["bytes_read"] += len(raw_line)
                        if not raw_line.strip():
                            counters["blank_lines"] += 1
                            progress()
                            continue
                        counters["rows_read"] += 1
                        key = None
                        try:
                            # utf-8-sig accepts a BOM without loosening JSON validation.
                            parsed = json.loads(raw_line.decode("utf-8-sig"))
                            if isinstance(parsed, dict) and isinstance(parsed.get("key"), str):
                                key = parsed["key"]
                            record = normalize_record(parsed, audio_root)
                        except (ValueError, OSError, UnicodeError) as exc:
                            reject(line_number, key, exc)
                            progress()
                            continue
                        digest = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                        previous = connection.execute("SELECT line, digest FROM seen WHERE key=?", (record["id"],)).fetchone()
                        if previous is not None:
                            conflict = digest != previous[1]
                            counters["duplicate_rows"] += 1
                            counters["conflicting_duplicate_rows"] += int(conflict)
                            _write_json_line(duplicates, {"key": record["id"], "first_line": previous[0],
                                                         "duplicate_line": line_number, "conflicting_text": conflict})
                            progress()
                            continue
                        connection.execute("INSERT INTO seen VALUES (?, ?, ?)", (record["id"], line_number, digest))
                        counters["unique_rows"] += 1
                        if counters["unique_rows"] % 1000 == 0:
                            connection.commit()
                        future = executor.submit(build_row, record, chunk_seconds=options.chunk_seconds,
                                                 max_duration_seconds=options.max_duration_seconds,
                                                 system_prompt=system_prompt)
                        queue.append((line_number, record["id"], future))
                        if len(queue) >= max(1, options.workers * 4):
                            drain_one()
                        progress()
                        if options.limit is not None and counters["unique_rows"] >= options.limit:
                            break
                while queue:
                    drain_one()
                for stream in (manifest, rejected, duplicates):
                    stream.flush()
                    os.fsync(stream.fileno())
            published = counters["accepted_rows"] > 0 and (not counters["rejected_rows"] or options.skip_invalid)
            report = {
                "schema_version": 1,
                "status": "ready" if published else "failed",
                "manifest_published": published,
                "manifest_path": str(output_root / "train.jsonl") if published else None,
                "input_jsonl": str(input_path), "audio_dir": str(audio_root),
                "direction": "en2zh", "source_audio_modified": False,
                "offline_trajectories_required": False,
                "audio_validation": "header structure and declared PCM/float payload bounds, not full sample decoding",
                "sentence_alignment_available": False,
                "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                "chunk_seconds": options.chunk_seconds, "workers": options.workers,
                "skip_invalid": options.skip_invalid, "limit": options.limit,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                **counters,
            }
            with (stage / "prepare_report.json").open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Reports are available even on strict validation failure; the final
            # manifest appears only after it is complete and safe to consume.
            for name in ("rejected.jsonl", "duplicates.jsonl", "prepare_report.json"):
                os.replace(stage / name, output_root / name)
            if published:
                os.replace(stage / "train.jsonl", output_root / "train.jsonl")
            progress(force=True)
            if published:
                LOGGER.info("Ready: %s (%s tasks); original WAV files remain in %s",
                            output_root / "train.jsonl", counters["accepted_rows"], audio_root)
            else:
                LOGGER.error("No training manifest published. Inspect %s; fix input or explicitly use "
                             "--skip-invalid with a new output directory", output_root / "rejected.jsonl")
            return report
        finally:
            connection.close()
