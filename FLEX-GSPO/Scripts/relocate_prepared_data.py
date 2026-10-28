#!/usr/bin/env python3
"""Rebind prepared EN->ZH tasks to copied WAVs for a NEW training run.

Standard library only. Reads destination WAV headers, preserves the old task
order/text/timing, refreshes inventory timestamps, and rebuilds virtual refs.
Does not copy audio, rewrite source code, or migrate optimizer/run identities.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import time

if __package__:
    from ._bootstrap import PROJECT_ROOT
else:
    from _bootstrap import PROJECT_ROOT

from flex_gspo.audio import encode_chunk_ref
from flex_gspo.online import unpack_task
from flex_gspo.prepare import probe_wav
from flex_gspo.prompts import initial_messages


def relocate_row(row, audio_root):
    """Validate before replacing only the supported path/inventory fields."""
    if not isinstance(row, dict) or not isinstance(row.get("flex_task"), str):
        raise ValueError("Expected prepared train.jsonl with JSON-string flex_task; "
                         "for raw key/en/zh JSONL use 01_prepare.sh")
    task = unpack_task(row["flex_task"])
    if row.get("messages") != initial_messages() or row.get("audios") != [encode_chunk_ref(task, 0)]:
        raise ValueError("Original actor messages/first virtual chunk are inconsistent with the task")
    if row.get("images") or row.get("videos"):
        raise ValueError("Only audio-only prepared tasks are supported")
    key = task["id"]
    if key != key.strip() or key in (".", "..") or any(c in key for c in ("/", "\\", ":", "\0")):
        raise ValueError("Task id must be a filename stem for the flat key.wav directory")
    if PureWindowsPath(task["audio_path"]).name != key + ".wav":
        raise ValueError("Original audio filename must be exactly id.wav; refusing to guess its mapping")
    target = (audio_root / (key + ".wav")).resolve()
    try:
        target.relative_to(audio_root)
    except ValueError as exc:
        raise ValueError("Destination WAV resolves outside --audio-dir") from exc
    metadata = probe_wav(target)
    old_metadata = task.get("original_audio")
    if not isinstance(old_metadata, dict):
        raise ValueError("Missing original_audio inventory; rebuild with 01_prepare.sh")
    if task.get("audio_size_bytes") != metadata["file_size_bytes"]:
        raise ValueError("Destination file size differs from prepared inventory; recopy the original WAV")
    for name in ("sample_rate", "channels", "bits_per_sample", "encoding", "num_frames", "file_size_bytes"):
        if old_metadata.get(name) != metadata[name]:
            raise ValueError(f"Destination WAV metadata differs: {name}; recopy the original WAV")
    for previous in (task["duration_seconds"], old_metadata.get("duration_seconds")):
        if not isinstance(previous, (int, float)) or not math.isclose(
                previous, metadata["duration_seconds"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("Destination duration differs from the preserved chunk schedule")
    changed_path = task["audio_path"] != str(target)
    changed_mtime = task.get("audio_mtime_ns") != metadata["mtime_ns"]
    task["audio_path"] = str(target)
    task["audio_size_bytes"] = metadata["file_size_bytes"]
    task["audio_mtime_ns"] = metadata["mtime_ns"]
    task["original_audio"] = {**old_metadata, "mtime_ns": metadata["mtime_ns"]}
    result = dict(row)
    result["flex_task"] = json.dumps(task, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    result["audios"] = [encode_chunk_ref(task, 0)]
    return result, changed_path, changed_mtime


def relocate_dataset(input_jsonl, audio_dir, output_dir, *, workers=16, progress_seconds=5.0):
    source = Path(input_jsonl).expanduser().resolve()
    audio_root = Path(audio_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file() or not audio_root.is_dir():
        raise ValueError("--input-jsonl must exist and --audio-dir must be the copied WAV directory")
    if not 1 <= workers <= 128 or not math.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("workers must be 1..128 and progress-seconds must be finite and positive")
    if output == audio_root or audio_root in output.parents:
        raise ValueError("Use a dedicated output directory outside the WAV directory")
    # Exclusive directory creation is also the single-writer guard. An incomplete
    # run leaves its failure report; retry into another directory after correction.
    output.mkdir(parents=True, exist_ok=False)
    partial = output / "train.jsonl.partial"
    started = time.monotonic()
    last_log = started
    count = changed_paths = changed_mtimes = 0
    source_hash, target_hash = hashlib.sha256(), hashlib.sha256()
    source_stat = source.stat()
    current_line = 0
    pool = None
    pending = deque()
    print(f"[relocate] input={source}\n[relocate] destination_audio={audio_root}\n"
          f"[relocate] output={output} workers={workers}; WAV headers only, no GPU", flush=True)

    def progress(force=False):
        nonlocal last_log
        now = time.monotonic()
        if force or now - last_log >= progress_seconds:
            elapsed = now - started
            print(f"[relocate] validated={count} changed_paths={changed_paths} refreshed_mtimes={changed_mtimes} "
                  f"elapsed={elapsed:.1f}s rows_per_second={count / max(elapsed, 0.001):.1f}", flush=True)
            last_log = now

    def consume(writer):
        nonlocal count, changed_paths, changed_mtimes, current_line
        current_line, future = pending[0]
        while True:
            try:
                row, path_changed, mtime_changed = future.result(timeout=progress_seconds)
                break
            except TimeoutError:
                if future.done():
                    raise
                progress(force=True)
        pending.popleft()
        raw = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        writer.write(raw)
        target_hash.update(raw)
        count += 1
        changed_paths += int(path_changed)
        changed_mtimes += int(mtime_changed)
        progress()

    def process(raw, line_number):
        try:
            row = json.loads(raw.decode("utf-8-sig" if line_number == 1 else "utf-8"))
            return relocate_row(row, audio_root)
        except Exception as exc:
            raise ValueError(f"Input line {line_number}: {exc}") from exc

    try:
        pool = ThreadPoolExecutor(max_workers=workers)
        with source.open("rb") as reader, partial.open("xb") as writer:
            for line_number, raw in enumerate(reader, 1):
                source_hash.update(raw)
                if not raw.strip():
                    continue
                pending.append((line_number, pool.submit(process, raw, line_number)))
                if len(pending) >= workers * 2:
                    consume(writer)
            while pending:
                consume(writer)
        if not count:
            raise ValueError("No prepared tasks found; no train.jsonl published")
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError("Input JSONL changed during migration; use a completed, immutable manifest")
        report = {
            "schema_version": 1, "purpose": "new_run_prepared_audio_relocation",
            "input_jsonl": str(source), "audio_dir": str(audio_root), "output_jsonl": str(output / "train.jsonl"),
            "tasks": count, "changed_paths": changed_paths, "refreshed_mtimes": changed_mtimes,
            "input_sha256": source_hash.hexdigest(), "output_sha256": target_hash.hexdigest(),
            "elapsed_seconds": time.monotonic() - started,
            "validation": "destination WAV header, size, duration and original virtual-reference consistency",
            "note": "No audio payload checksum or semantic verification. Does not preserve exact-resume identity. "
                    "Paths must be visible under these same names inside the training container.",
        }
        (output / "relocate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        partial.replace(output / "train.jsonl")
        progress(force=True)
        print(f"[relocate] SUCCESS: {output / 'train.jsonl'}", flush=True)
        return report
    except BaseException as exc:
        for _, future in pending:
            future.cancel()
        if partial.exists():
            partial.unlink()
        (output / "relocate_failed.json").write_text(json.dumps({
            "error": str(exc), "validated_before_failure": count, "line": current_line,
            # A closed stdout or Ctrl-C after atomic publication must not label
            # an already complete manifest as absent. Never delete that output.
            "manifest_published": (output / "train.jsonl").is_file(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Existing prepared train.jsonl, not raw key/en/zh JSONL")
    parser.add_argument("--audio-dir", required=True, help="Copied flat directory containing every id.wav")
    parser.add_argument("--output-dir", required=True, help="New directory for train.jsonl and relocation report")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        relocate_dataset(args.input_jsonl, args.audio_dir, args.output_dir,
                         workers=args.workers, progress_seconds=args.progress_seconds)
    except (OSError, ValueError) as exc:
        print(f"[relocate] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
