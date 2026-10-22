#!/usr/bin/env python3
"""Export existing FLEX logs to TensorBoard without touching training.

Serve the WHOLE training run directory to see native SWIFT loss metrics and
these additional reward/performance/text series together. Cursor checkpoints
avoid repeat imports on clean restarts; an observer crash between event flush
and cursor replacement may replay the last records (not transactional exactly-once).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import subprocess
import time

from flex_gspo.monitoring import Monitor, SourceLookup


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="Training output root (contains rewards/ and vN-.../)")
    p.add_argument("--watch", action="store_true", help="Continue reading appended logs; Ctrl-C stops only this observer")
    p.add_argument("--poll-seconds", type=float, default=10)
    p.add_argument("--max-samples", type=int, default=2, help="Maximum text examples per selected local callback (per rank/PID); original JSONL is unchanged")
    p.add_argument("--sample-every", type=int, default=10, help="Show candidate text at callback 1 and every N local callbacks; scalar summaries cover all callbacks")
    p.add_argument("--dataset", help="Optional prepared train.jsonl for source/reference display; indexed once without reading audio")
    p.add_argument("--gpu-ids", help="Optional physical nvidia-smi IDs, e.g. 0,1,2,3; reports whole-device samples, not process peaks")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.poll_seconds <= 3600 or not 0 <= args.max_samples <= 32 or args.sample_every < 1:
        raise SystemExit("poll-seconds must be 1..3600, max-samples 0..32, and sample-every positive")
    if args.gpu_ids and not re.fullmatch(r"\d+(,\d+)*", args.gpu_ids):
        raise SystemExit("gpu-ids must be comma-separated physical GPU integers")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    monitor = lookup = None
    try:
        # Verify the optional package before creating observer output/state.
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa: F401
        monitor = Monitor(args.run_dir, max_samples=args.max_samples, sample_every=args.sample_every)
        if args.dataset:
            lookup = SourceLookup(args.dataset, monitor.output)
            monitor.lookup = lookup
        logging.info("Observing %s; native loss remains in vN-.../runs; reward views are per-rank callbacks, not reconstructed G8 groups.", args.run_dir)
        logging.info("TensorBoard: python3 -m tensorboard.main --logdir %s --host 127.0.0.1 --port 6006", str(Path(args.run_dir).resolve()))
        if args.gpu_ids:
            logging.warning("GPU values are sampled whole-device usage, include other processes, and are NOT guaranteed training peak memory.")
        while True:
            imported = monitor.poll()
            logging.info("Imported new records=%d; saved text samples=%d", imported, monitor.state["samples"])
            if args.gpu_ids:
                try:
                    monitor.gpu_sample(args.gpu_ids)
                except (OSError, subprocess.SubprocessError) as exc:
                    logging.warning("GPU sample unavailable: %s", exc)
            if not args.watch:
                # Drain large logs in bounded chunks before one-shot exit.
                if monitor.last_progress:
                    continue
                break
            time.sleep(args.poll_seconds)
    except ImportError as exc:
        logging.error("Observer requires tensorboard in its own environment; training is unaffected. Missing import: %s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logging.error("Observer stopped; training is unaffected: %s", exc)
        return 2
    except KeyboardInterrupt:
        logging.info("Observer stopped; training continues independently.")
    finally:
        if monitor:
            monitor.close()
        if lookup:
            lookup.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
