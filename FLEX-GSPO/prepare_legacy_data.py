#!/usr/bin/env python3
"""Prepare legacy key/en.txt/zh.txt data with the current SFT system prompt.

Inherited CLI: caiqiezujian/FLEX, main at
076008b3bb1ebe69b73cb5c75abd99307fdf82cb, FLEX-GSPO/prepare_data.py.
The legacy field/path contract is explicit; the current preparation core owns
validation, deduplication, progress, virtual chunks, and the shared SFT prompt.
Supply paths explicitly instead of inheriting the old server's defaults.
"""

from __future__ import annotations

import argparse
import logging
import sys

from flex_gspo.prepare import PreparationError, PrepareOptions, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", required=True,
                        help="Legacy JSONL: lang='en', key, en.txt, zh.txt")
    parser.add_argument("--audio-dir", required=True,
                        help="Original WAV directory; each key resolves to key.wav")
    parser.add_argument("--output-dir", required=True,
                        help="Dedicated output directory; existing preparation outputs are never overwritten")
    parser.add_argument("--chunk-seconds", type=int, default=1,
                        help="Fixed causal chunk duration, integer 1..8")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent WAV-header probes, not model workers")
    parser.add_argument("--progress-seconds", type=float, default=5.0,
                        help="Progress logging interval in seconds")
    parser.add_argument("--skip-invalid", action="store_true",
                        help="Publish valid rows despite reported invalid rows; originals are unchanged")
    parser.add_argument("--limit", type=int,
                        help="Prepare only the first N distinct eligible keys")
    parser.add_argument("--max-duration-seconds", type=float, default=40.0,
                        help="Reject clips over this budget; increase context/turn limits together")
    parser.add_argument("--system-prompt-file",
                        help="Optional UTF-8 assertion: must exactly match the current SFT concise strategy prompt")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    logging.info("Legacy data entry: key/en.txt/zh.txt + key.wav; using the CURRENT SFT concise strategy prompt")
    try:
        report = prepare_dataset(PrepareOptions(input_format="legacy", **vars(args)))
    except (PreparationError, OSError) as exc:
        logging.error("Preparation failed: %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.warning("Preparation interrupted. No partial training manifest is published; source audio is unchanged")
        return 130
    return 0 if report["manifest_published"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
