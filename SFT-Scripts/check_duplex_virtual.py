#!/usr/bin/env python3
"""Read-only sample check for virtual-audio SFT; no model weights or GPU job."""
import argparse
import importlib.metadata
import io
import json
from pathlib import Path
import sys

from duplex_audio import SAMPLE_RATE, chunk_wav_bytes, decode_chunk_ref


def check_row(row):
    import soundfile as sf

    refs = row.get("audios")
    durations = row.get("audio_chunk_durations")
    messages = row.get("messages")
    if not isinstance(refs, list) or not refs:
        raise ValueError("Expected default string-format row with nonempty audios")
    if not isinstance(durations, list) or len(durations) != len(refs):
        raise ValueError("audio_chunk_durations must match audios")
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError("Expected a system prompt and alternating user/assistant messages")
    if len(messages) != 1 + 2 * len(refs):
        raise ValueError("Message count does not match audio turns")
    previous_end, source_path = 0.0, None
    actual_samples = []
    for i, ref in enumerate(refs):
        data = decode_chunk_ref(ref)
        if source_path is None:
            source_path = data["path"]
        if data["path"] != source_path or abs(data["start"] - previous_end) > 1e-7:
            raise ValueError(f"Noncontiguous or mismatched source audio at turn {i}")
        if abs(float(durations[i]) - data["input_seconds"]) > 1e-7:
            raise ValueError(f"Duration mismatch at turn {i}")
        user, assistant = messages[1 + 2*i:3 + 2*i]
        if user.get("role") != "user" or user.get("content") != "<audio>":
            raise ValueError(f"Invalid audio user turn {i}")
        content = assistant.get("content")
        if assistant.get("role") != "assistant" or not isinstance(content, str):
            raise ValueError(f"Invalid assistant turn {i}")
        if content != "<|WAIT|>" and not any(
            content.startswith(tag) and content[len(tag):].strip()
            for tag in ("<|DIRECT|>", "<|FLEX|>")
        ):
            raise ValueError(f"Invalid protocol content at turn {i}")
        pcm, sr = sf.read(io.BytesIO(chunk_wav_bytes(ref)), dtype="float32")
        if sr != SAMPLE_RATE or pcm.ndim != 1 or len(pcm) != round(data["input_seconds"] * sr):
            raise ValueError(f"Invalid materialized waveform at turn {i}")
        actual_samples.append(len(pcm))
        previous_end = data["start"] + data["input_seconds"]
    return {"id": row.get("id"), "audio_turns": len(refs),
            "input_seconds": sum(durations), "sample_counts": actual_samples,
            "source_audio": source_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--rows", type=int, default=2, help="Number of leading nonempty rows to check")
    parser.add_argument("--framework", action="store_true", help="Also import the installed SWIFT template plugin")
    args = parser.parse_args(argv)
    if args.rows < 1:
        parser.error("--rows must be positive")
    try:
        print(f"[check] Python: {sys.executable}", flush=True)
        count = 0
        with args.dataset.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    summary = check_row(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
                count += 1
                print(json.dumps(summary, ensure_ascii=False), flush=True)
                if count >= args.rows:
                    break
        if count == 0:
            raise ValueError("Dataset contains no training rows")
        if args.framework:
            import duplex_sft_plugin
            from importlib.util import find_spec
            if find_spec("swift.cli._megatron.sft") is None:
                raise ValueError("Installed SWIFT has no swift.cli._megatron.sft entry point")
            print(f"[check] ms-swift={importlib.metadata.version('ms-swift')} "
                  f"template={duplex_sft_plugin._meta.template_type}", flush=True)
        print(json.dumps({"ok": True, "sampled_rows_checked": count,
                          "framework_import_checked": args.framework,
                          "full_dataset_checked": False,
                          "real_template_encoding_checked": False,
                          "gpu_training_verified": False}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"[check] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
