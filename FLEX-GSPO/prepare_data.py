#!/usr/bin/env python3
"""Prepare English-to-Chinese online-GSPO tasks without copying source WAVs."""

from __future__ import annotations

import argparse
import logging
import sys

from flex_gspo.prepare import PreparationError, PrepareOptions, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=PrepareOptions.input_jsonl, help="Raw JSONL: lang/key/en.txt/zh.txt")
    parser.add_argument("--audio-dir", default=PrepareOptions.audio_dir, help="Original WAV directory, filenames are key.wav")
    parser.add_argument("--output-dir", default=PrepareOptions.output_dir, help="Dedicated output directory; existing outputs are never overwritten")
    parser.add_argument("--chunk-seconds", type=int, default=1, help="Fixed causal chunk duration, integer 1..8 (default: 1)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent WAV-header probes, not model workers (default: 8)")
    parser.add_argument("--progress-seconds", type=float, default=5.0, help="Progress logging interval (default: 5)")
    parser.add_argument("--skip-invalid", action="store_true", help="Explicitly publish valid rows despite reported bad rows")
    parser.add_argument("--limit", type=int, help="Prepare only the first N distinct eligible keys for a smoke test")
    parser.add_argument("--max-duration-seconds", type=float, default=40.0,
                        help="Reject clips over this training budget (default: 40); increase context/turn limits together")
    parser.add_argument("--system-prompt-file", help="Optional UTF-8 prompt assertion: must exactly match the preserved SFT prompt")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        report = prepare_dataset(PrepareOptions(**vars(args)))
    except (PreparationError, OSError) as exc:
        logging.error("Preparation failed: %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.warning("Preparation interrupted. No partial training manifest is published; source audio is unchanged")
        return 130
    return 0 if report["manifest_published"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
