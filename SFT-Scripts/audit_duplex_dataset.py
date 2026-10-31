#!/usr/bin/env python3
"""Audit every virtual-audio SFT row without changing the dataset or audio.

Default: inspect SoundFile headers and match the current duplex_audio.py boundary
checks. --decode additionally runs its real chunk reader (no SWIFT/model/GPU).
Reports contain counts and failed-row diagnostics, never copied training rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from duplex_audio import SAMPLE_RATE, chunk_wav_bytes, decode_chunk_ref


REASONS = {
    "invalid_json": "JSON解析失败",
    "invalid_schema": "消息或音频字段结构错误",
    "invalid_protocol": "WAIT/DIRECT/FLEX输出格式错误",
    "invalid_ref": "虚拟音频引用格式错误",
    "relative_audio_path": "音频路径不是当前系统的绝对路径",
    "missing_audio": "音频文件不存在",
    "audio_stat_error": "无法访问音频文件",
    "audio_header_error": "soundfile无法读取有效音频头",
    "source_changed": "音频属性与引用中的记录不一致",
    "source_path_mismatch": "同一条样本引用了不同源音频",
    "noncontiguous_turns": "音频轮次起点不符合连续输入时间线",
    "duration_mismatch": "audio_chunk_durations与引用不一致",
    "bounds_seconds_only": "秒数越界但帧区间合法，疑似时长舍入差异",
    "bounds_frames": "请求的采样帧越界或区间不足一个采样帧",
    "invalid_padding": "补零后的输入长度不符合实际读取规则",
    "decode_error": "实际分块读取/重采样失败",
}


def finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def extract_payload(row):
    """Accept both formats emitted by prepare_duplex_data.build_row."""
    if not isinstance(row, dict):
        raise ValueError("JSONL row must be an object")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a nonempty list of objects")
    if "audios" in row:
        refs = row["audios"]
        if not isinstance(refs, list) or not refs:
            raise ValueError("audios must be a nonempty list")
        if messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str):
            raise ValueError("Default format requires a string system message")
        turns = messages[1:]
        fmt = "string"
    else:
        if not isinstance(row.get("system"), str):
            raise ValueError("Missing audios; typed format requires a top-level system string")
        refs, turns, fmt = [], messages, "typed"
    if len(turns) % 2 or not turns:
        raise ValueError("Expected user/assistant pairs")
    outputs = []
    for i in range(0, len(turns), 2):
        user, assistant = turns[i:i + 2]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError(f"Invalid role order at turn {i // 2 + 1}")
        uc, ac = user.get("content"), assistant.get("content")
        if fmt == "typed":
            if not (isinstance(uc, list) and len(uc) == 1 and isinstance(uc[0], dict)
                    and uc[0].get("type") == "audio"):
                raise ValueError(f"Invalid typed audio at turn {i // 2 + 1}")
            if not (isinstance(ac, list) and len(ac) == 1 and isinstance(ac[0], dict)
                    and ac[0].get("type") == "text"):
                raise ValueError(f"Invalid typed text at turn {i // 2 + 1}")
            refs.append(uc[0].get("audio"))
            ac = ac[0].get("text")
        elif uc != "<audio>":
            raise ValueError(f"Expected <audio> at turn {i // 2 + 1}")
        if not isinstance(ac, str):
            raise ValueError(f"Assistant content must be a string at turn {i // 2 + 1}")
        outputs.append(ac)
    durations = row.get("audio_chunk_durations")
    if len(outputs) != len(refs) or not isinstance(durations, list) or len(durations) != len(refs):
        raise ValueError("Audio, duration and user/assistant turn counts must agree")
    return refs, durations, outputs, fmt


def inspect_audio(path):
    import soundfile as sf

    p = Path(path)
    if not p.is_absolute():
        return None, "relative_audio_path", "Path is not absolute in the auditing environment"
    try:
        stat = p.stat()
    except FileNotFoundError as exc:
        return None, "missing_audio", str(exc)
    except (OSError, ValueError) as exc:
        return None, "audio_stat_error", str(exc)
    try:
        # Same backend and frame count as the training reader; no ffprobe fallback.
        with sf.SoundFile(path) as stream:
            info = {"frames": len(stream), "sample_rate": stream.samplerate,
                    "channels": stream.channels, "audio_size_bytes": stat.st_size,
                    "audio_mtime_ns": stat.st_mtime_ns}
        if info["frames"] < 1 or info["sample_rate"] < 1 or info["channels"] < 1:
            raise ValueError("Empty audio or invalid audio header")
        info["duration_seconds"] = info["frames"] / info["sample_rate"]
        return info, None, None
    except (OSError, RuntimeError, ValueError) as exc:
        return None, "audio_header_error", str(exc)


def audit_line(line_number, raw, decode=False, max_details=10):
    errors, counts = [], Counter()
    result = {"line": line_number, "id": None, "row_format": None,
              "chunks_total": 0, "chunks_checked": 0, "chunks_failed": 0,
              "chunks_decoded": 0, "errors": errors, "issue_counts": counts}

    def issue(code, message, **details):
        counts[code] += 1
        if len(errors) < max_details:
            errors.append({"code": code, "message": message, **details})

    try:
        row = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        issue("invalid_json", str(exc))
        return result
    if isinstance(row, dict):
        result["id"] = str(row.get("id", row.get("key", "")))
    try:
        refs, durations, outputs, fmt = extract_payload(row)
    except ValueError as exc:
        issue("invalid_schema", str(exc))
        return result
    result["chunks_total"], result["row_format"] = len(refs), fmt
    metadata = {}  # At most the files in this row, not the whole corpus.
    source_path, previous_input_end = None, 0.0
    for index, (ref, duration, output) in enumerate(zip(refs, durations, outputs), 1):
        before = sum(counts.values())
        result["chunks_checked"] += 1
        valid_output = output == "<|WAIT|>" or any(
            output.startswith(tag) and output[len(tag):].strip()
            and not any(token in output[len(tag):] for token in ("<|WAIT|>", "<|DIRECT|>", "<|FLEX|>"))
            for tag in ("<|DIRECT|>", "<|FLEX|>")
        )
        if not valid_output:
            issue("invalid_protocol", "Expected WAIT alone or DIRECT/FLEX with new text", turn=index)
        try:
            data = decode_chunk_ref(ref)
        except (ValueError, TypeError, OverflowError) as exc:
            issue("invalid_ref", str(exc), turn=index)
            result["chunks_failed"] += 1
            previous_input_end = None
            continue
        path = data["path"]
        context = {"turn": index, "audio_path": path, "start_seconds": data["start"],
                   "end_seconds": data["end"], "input_seconds": data["input_seconds"]}
        if source_path is None:
            source_path = path
        elif path != source_path:
            issue("source_path_mismatch", "Different source audio within one session", **context)
        if previous_input_end is not None and abs(data["start"] - previous_input_end) > 1e-7:
            issue("noncontiguous_turns", f"Expected start={previous_input_end}", **context)
        previous_input_end = data["start"] + data["input_seconds"]
        if not finite_number(duration) or abs(duration - data["input_seconds"]) > 1e-7:
            issue("duration_mismatch", "Duration field disagrees with virtual reference", **context)
        if path not in metadata:
            metadata[path] = inspect_audio(path)
        info, code, message = metadata[path]
        if info is None:
            issue(code, message, **context)
        else:
            for attr in ("audio_size_bytes", "audio_mtime_ns"):
                if attr in data and data[attr] != info[attr]:
                    issue("source_changed", f"Stored {attr} differs from actual file", **context)
            sr, frames = info["sample_rate"], info["frames"]
            # Keep the strict reader behavior. A suspected rounding problem is
            # reported as FAILED until the reader/data is deliberately corrected.
            try:
                start_frame, end_frame = round(data["start"] * sr), round(data["end"] * sr)
                frames_bad = start_frame < 0 or end_frame > frames or end_frame <= start_frame
                seconds_bad = data["end"] > info["duration_seconds"] + 1e-8
                if frames_bad or seconds_bad:
                    issue("bounds_frames" if frames_bad else "bounds_seconds_only",
                          "Chunk bounds exceed the actual source audio", **context,
                          actual_duration_seconds=info["duration_seconds"], sample_rate=sr,
                          actual_frames=frames, start_frame=start_frame, end_frame=end_frame,
                          end_excess_seconds=data["end"] - info["duration_seconds"],
                          end_excess_frames=end_frame - frames,
                          start_at_or_after_eof=start_frame >= frames,
                          empty_frame_range=end_frame <= start_frame)
                actual_count = round((data["end"] - data["start"]) * SAMPLE_RATE)
                output_count = round(data["input_seconds"] * SAMPLE_RATE)
                if output_count < actual_count or output_count < 160:
                    issue("invalid_padding", "Invalid padded chunk length", **context)
            except (ValueError, OverflowError) as exc:
                issue("bounds_frames", f"Cannot convert interval to sample frames: {exc}", **context)
            if decode and sum(counts.values()) == before:
                try:
                    chunk_wav_bytes(ref)
                    result["chunks_decoded"] += 1
                except Exception as exc:
                    issue("decode_error", f"{type(exc).__name__}: {exc}", **context)
        if sum(counts.values()) > before:
            result["chunks_failed"] += 1
    return result


def run_audit(args):
    import soundfile as sf

    dataset = args.dataset.resolve(strict=True)
    if not dataset.is_file():
        raise ValueError(f"Not a JSONL file: {dataset}")
    initial_stat = dataset.stat()
    # Exclusive report directory: neither source data nor previous reports can be overwritten.
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    counts = Counter(rows_checked=0, rows_passed=0, rows_failed=0, blank_lines=0,
                     chunks_total=0, chunks_checked=0, chunks_failed=0, chunks_decoded=0,
                     rounding_suspect_only_rows=0)
    reason_rows, reason_occurrences, formats = Counter(), Counter(), Counter()
    scanned_lines, reached_eof = 0, False
    status, failure = "running", None
    last_progress, last_rows = began, 0
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = set()

    def progress(force=False):
        nonlocal last_progress, last_rows
        now = time.monotonic()
        if force or now - last_progress >= args.progress_seconds or counts["rows_checked"] - last_rows >= args.progress_every:
            elapsed = now - began
            rate = counts["rows_checked"] / max(elapsed, 0.001)
            print(f"[audit] checked={counts['rows_checked']:,} passed={counts['rows_passed']:,} "
                  f"failed={counts['rows_failed']:,} chunks={counts['chunks_checked']:,} "
                  f"pending={len(pending)} speed={rate:.1f} rows/s elapsed={elapsed:.1f}s", flush=True)
            last_progress, last_rows = now, counts["rows_checked"]

    print(f"[audit] Python: {sys.executable}\n[audit] input: {dataset}\n[audit] reports: {output}\n"
          f"[audit] mode={'decode' if args.decode else 'headers'} workers={args.workers}; "
          "source data is read-only", flush=True)
    try:
        with dataset.open("rb") as source, (output / "invalid.jsonl").open("x", encoding="utf-8") as invalid:
            iterator = enumerate(source, 1)
            while pending or not reached_eof:
                while not reached_eof and len(pending) < args.workers * 4:
                    try:
                        number, raw = next(iterator)
                    except StopIteration:
                        reached_eof = True
                        break
                    scanned_lines = number
                    if not raw.strip():
                        counts["blank_lines"] += 1
                        progress()
                        continue
                    pending.add(pool.submit(audit_line, number, raw, args.decode, args.max_details))
                if not pending:
                    continue
                done, pending = wait(pending, timeout=args.progress_seconds, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    counts["rows_checked"] += 1
                    for name in ("chunks_total", "chunks_checked", "chunks_failed", "chunks_decoded"):
                        counts[name] += result[name]
                    if result["row_format"]:
                        formats[result["row_format"]] += 1
                    reasons = result["issue_counts"]
                    if reasons:
                        counts["rows_failed"] += 1
                        reason_rows.update(reasons.keys())
                        reason_occurrences.update(reasons)
                        if set(reasons) == {"bounds_seconds_only"}:
                            counts["rounding_suspect_only_rows"] += 1
                        result["details_truncated"] = sum(reasons.values()) > len(result["errors"])
                        invalid.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    else:
                        counts["rows_passed"] += 1
                invalid.flush()
                progress()
        status = "completed"
    except KeyboardInterrupt:
        status, failure = "interrupted", "Stopped by user; counts cover completed rows only"
    except Exception as exc:
        status, failure = "error", f"{type(exc).__name__}: {exc}"
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True)
    try:
        final_stat = dataset.stat()
        unchanged = (initial_stat.st_size, initial_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns)
    except OSError:
        unchanged = False
    if not unchanged:
        status, failure = "error", "Dataset changed during audit; do not use this as a full audit"
    if status == "completed" and counts["rows_checked"] == 0:
        status, failure = "error", "Dataset contains no nonblank JSONL rows"
    progress(force=True)
    reader = Path(__file__).with_name("duplex_audio.py")
    summary = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "error": failure, "dataset": str(dataset),
        "mode": "decode" if args.decode else "headers", "workers": args.workers,
        "soundfile_version": sf.__version__,
        "reader_sha256": hashlib.sha256(reader.read_bytes()).hexdigest(),
        "full_dataset_checked": status == "completed" and reached_eof,
        "dataset_unchanged_during_audit": unchanged, "physical_lines_scanned": scanned_lines,
        **dict(counts),
        "pass_fraction": counts["rows_passed"] / counts["rows_checked"] if counts["rows_checked"] else None,
        "reason_row_counts": dict(reason_rows), "reason_occurrence_counts": dict(reason_occurrences),
        "row_format_counts": dict(formats),
        "mixed_content_formats_detected": len(formats) > 1,
        "reason_descriptions": {code: REASONS[code] for code in reason_rows},
        "elapsed_seconds": round(time.monotonic() - began, 3),
        "notes": [
            "One bad chunk makes its entire row fail; all decodable references are examined.",
            "reason_row_counts overlap when a row has several different problems.",
            "bounds_seconds_only is a rounding SUSPECT, not a proven cause or an automatic repair.",
            "Header mode does not decode PCM or prove audio integrity; --decode uses the actual chunk reader.",
            "No semantic/timestamp alignment, token-length, model/template, or GPU training verification.",
            "Typed and string rows are checked individually; mixing them in one dataset may break Arrow loading.",
            "Audio files must remain unchanged during audit and training. No original file is modified.",
            "invalid.jsonl is written in completion order; line is the original 1-based JSONL line number.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[audit] status={status}; total={counts['rows_checked']:,} "
          f"passed={counts['rows_passed']:,} failed={counts['rows_failed']:,}", flush=True)
    for code, count in reason_rows.most_common():
        print(f"[audit] {code}: {count:,} rows | {REASONS[code]}", flush=True)
    if len(formats) > 1:
        print("[audit] WARNING: mixed string/typed rows; row passes do not certify Arrow compatibility", flush=True)
    print(f"[audit] summary: {output / 'summary.json'}\n[audit] failed details: {output / 'invalid.jsonl'}", flush=True)
    if failure:
        print(f"[audit] ERROR: {failure}", file=sys.stderr, flush=True)
    return 130 if status == "interrupted" else 2 if status != "completed" else 1 if counts["rows_failed"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prepared virtual-audio training JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for reports; must not exist")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--decode", action="store_true", help="Also read/resample chunks with the existing training reader (slower)")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--progress-seconds", type=float, default=5)
    parser.add_argument("--max-details", type=int, default=10, help="Maximum detailed issues per failed row; all issues are counted")
    args = parser.parse_args(argv)
    if min(args.workers, args.progress_every, args.max_details) < 1 or not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        parser.error("workers, progress-every, progress-seconds and max-details must be positive")
    try:
        import soundfile  # Fail before creating reports if the dependency is absent.
        if args.decode:
            import numpy
            import scipy.signal
        return run_audit(args)
    except Exception as exc:
        print(f"[audit] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
