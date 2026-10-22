#!/usr/bin/env python3
"""Re-score selected real rollout texts with COMET22, without an actor/GPU.

This is a deliberately nonrandom diagnostic sample, NOT a training dataset or
an estimate of population quality. Old logs remain untouched. Legacy rollouts
have no EOF drain; this tool never fabricates one or recomputes a v2 reward.
Running this script sends selected source/reference/translation text to the
configured scoring service; it does not read or upload audio.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import time

from flex_gspo.comet22 import ServiceCOMET22
from flex_gspo.degeneration import degeneration_report
from flex_gspo.protocol import collect_translation
from flex_gspo.rewards import RewardError


LOG = logging.getLogger("flex.reward_audit")
PID_NAME = re.compile(r"rewards-rank\d+-pid(\d+)\.jsonl$")


def _jsonl(path, *, allow_truncated_tail=False, diagnostics=None):
    with Path(path).open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig" if line_no == 1 else "utf-8"))
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
            except (UnicodeError, ValueError):
                if allow_truncated_tail and not raw.endswith(b"\n") and not handle.read(1):
                    if diagnostics is not None:
                        diagnostics["truncated_tail_lines"] += 1
                    LOG.warning("Skipped truncated final line: %s:%d", path, line_no)
                    return
                raise RewardError(f"Invalid JSON object at {path}:{line_no}") from None
            yield line_no, row


def _translation(record):
    turns = (record.get("trajectory") or {}).get("turns")
    try:
        if not isinstance(turns, list) or not turns:
            raise ValueError("missing trajectory turns")
        text = collect_translation(turns)
    except (TypeError, KeyError, ValueError, AttributeError):
        return None, False
    stored = record.get("translation")
    if stored is not None and stored != text:
        raise RewardError("Stored translation disagrees with append-only trajectory; refusing ambiguous scoring")
    return text, True


def select_candidates(run_dir, limit=100, pids=None):
    if type(limit) is not int or limit < 1:
        raise RewardError("--limit must be positive")
    root = Path(run_dir)
    log_root = root / "rewards" if (root / "rewards").is_dir() else root
    paths = sorted(log_root.glob("rewards-rank*-pid*.jsonl"))
    if pids is not None:
        wanted = set(pids)
        paths = [p for p in paths if PID_NAME.fullmatch(p.name) and PID_NAME.fullmatch(p.name).group(1) in wanted]
        found = {PID_NAME.fullmatch(p.name).group(1) for p in paths}
        if wanted - found:
            raise RewardError("Requested PID reward files were not all found")
    if not paths:
        raise RewardError("No rewards-rank*-pid*.jsonl files found under run-dir/rewards or run-dir")
    # Bound memory, while scanning all log rows so late failures are eligible.
    buckets = {key: [] for key in ("all_wait_or_empty", "possible_loop", "protocol_invalid", "other")}
    diagnostics = {"scanned_rows": 0, "truncated_tail_lines": 0, "files": [], "selection_counts": {}}
    for path in paths:
        initial = path.stat()
        diagnostics["files"].append(str(path.resolve()))
        for line_no, record in _jsonl(path, allow_truncated_tail=True, diagnostics=diagnostics):
            diagnostics["scanned_rows"] += 1
            if not isinstance(record.get("task_id"), str) or not record["task_id"]:
                raise RewardError(f"Reward record missing task_id at {path}:{line_no}")
            text, valid = _translation(record)
            if not valid:
                category = "protocol_invalid"
            elif not text.strip():
                category = "all_wait_or_empty"
            elif degeneration_report("", "", record["trajectory"]["turns"])["severe_repetition"]:
                # Selection only. Real source/reference exemptions are applied
                # AFTER joining the task metadata below.
                category = "possible_loop"
            else:
                category = "other"
            if len(buckets[category]) < limit:
                buckets[category].append({"log_file": str(path.resolve()), "log_line": line_no,
                                          "selection_category": category, "record": record})
            if diagnostics["scanned_rows"] % 10000 == 0:
                LOG.info("Reward scan: rows=%d", diagnostics["scanned_rows"])
        final = path.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise RewardError("Reward log changed during reading; audit a stopped run or immutable snapshot")
    selected = []
    # Round-robin categories, taking first examples in each category. Never
    # infer frequencies or complete global G8 groups from this sample.
    for index in range(limit):
        for category, items in buckets.items():
            if index < len(items):
                selected.append(items[index])
                if len(selected) == limit:
                    break
        if len(selected) == limit:
            break
    diagnostics["selection_counts"] = dict(Counter(item["selection_category"] for item in selected))
    return selected, diagnostics


def join_tasks(dataset, selected):
    wanted = {item["record"]["task_id"] for item in selected}
    found = {}
    for line_no, row in _jsonl(dataset):
        task = row.get("flex_task", row)
        if isinstance(task, str):
            try:
                task = json.loads(task)
            except ValueError:
                raise RewardError(f"Invalid flex_task JSON at dataset line {line_no}") from None
        if not isinstance(task, dict):
            raise RewardError(f"Invalid flex_task object at dataset line {line_no}")
        task_id = task.get("id")
        if task_id not in wanted:
            continue
        for key in ("source_text", "reference_text"):
            if not isinstance(task.get(key), str) or not task[key].strip():
                raise RewardError(f"Selected task {task_id} has missing {key}")
        if task_id in found and found[task_id] != task:
            raise RewardError(f"Dataset contains conflicting records for selected task {task_id}")
        found[task_id] = task
        if line_no % 10000 == 0:
            LOG.info("Dataset join: lines=%d matched=%d/%d", line_no, len(found), len(wanted))
    missing = wanted - set(found)
    if missing:
        raise RewardError(f"Dataset is missing {len(missing)} selected task IDs; verify matching dataset/run")
    return found


def run_audit(run_dir, dataset, output, config, *, limit=100, pids=None, compare_metricx=False,
              comet_scorer=None, metricx_scorer=None):
    output = Path(output).resolve()
    summary_path = output.with_name(output.name + ".summary.json")
    if output.exists() or summary_path.exists():
        raise RewardError("Audit output or summary already exists; choose a NEW output path")
    selected, selection = select_candidates(run_dir, limit, pids)
    if not selected:
        raise RewardError("No reward candidates available")
    tasks = join_tasks(dataset, selected)
    try:
        comet = comet_scorer or ServiceCOMET22(config["reward"]["comet22"])
    except (TypeError, KeyError):
        raise RewardError("Configure reward.comet22 for candidate auditing") from None
    metricx = None
    if compare_metricx:
        from flex_gspo.metricx import ServiceMetricX
        try:
            metricx = metricx_scorer or ServiceMetricX(config["reward"]["metricx"])
        except (TypeError, KeyError):
            raise RewardError("Configure reward.metricx before requesting --compare-metricx") from None
    started = time.time()
    summary = {
        "schema_version": 1, "status": "running", "created_unix": started,
        "selection": "First examples per diagnostic category, round-robin; NOT random or population statistics",
        "run_dir": str(Path(run_dir).resolve()), "dataset": str(Path(dataset).resolve()),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
        "selection_diagnostics": selection, "selected_candidates": len(selected), "written_candidates": 0,
        "quality_scored": 0, "new_full_rewards_computed": False,
        "note": "No synthetic EOF added, no legacy timing altered, no actor model/audio loaded. "
                "Metric values and loop diagnostics do not certify translation quality or training readiness.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve both outputs before requests. Exceptions preserve partial results
    # and an explicit failed summary rather than a false completed artifact.
    with output.open("x", encoding="utf-8") as handle:
        with summary_path.open("x", encoding="utf-8") as summary_handle:
            try:
                cached = {}
                for index, item in enumerate(selected, 1):
                    record = item["record"]
                    task = tasks[record["task_id"]]
                    text, protocol_valid = _translation(record)
                    guard = (degeneration_report(task["source_text"], task["reference_text"], record["trajectory"]["turns"])
                             if protocol_valid else None)
                    raw_comet = raw_metricx = None
                    if protocol_valid and text.strip():
                        key = (task["source_text"], text, task["reference_text"])
                        if key not in cached:
                            score = comet.score_batch(key[0], [key[1]], key[2])[0]
                            mx_score = metricx.score_batch(key[0], [key[1]], key[2])[0] if metricx else None
                            cached[key] = (score, mx_score)
                        raw_comet, raw_metricx = cached[key]
                        summary["quality_scored"] += 1
                    trajectory = record.get("trajectory") or {}
                    result = {
                        "schema_version": 1,
                        **{key: item[key] for key in ("log_file", "log_line", "selection_category")},
                        **{key: record.get(key) for key in ("task_id", "global_step", "call", "candidate_index_in_local_batch", "request_id", "prompt_id")},
                        "source_text": task["source_text"], "reference_text": task["reference_text"],
                        "duration_seconds": task.get("duration_seconds"), "translation": text,
                        "old_reward": record.get("reward"), "old_components": record.get("components"),
                        "protocol_valid": protocol_valid, "empty_translation": not text if protocol_valid else None,
                        "comet22_score": raw_comet, "metricx_error_rescored": raw_metricx,
                        "degeneration": guard, "eof_metadata_present": "eof_complete" in trajectory,
                        "new_full_reward": None,
                        "reward_not_recomputed_reason": "Text/guard audit only; do not fabricate missing EOF or reinterpret historical timing",
                    }
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    summary["written_candidates"] = index
                    LOG.info("Candidate audit %d/%d: id=%s category=%s comet=%s severe_loop=%s", index, len(selected),
                             task["id"], item["selection_category"], raw_comet,
                             guard["severe_repetition"] if guard else None)
                summary["status"] = "complete"
            except Exception as exc:
                summary["status"] = "failed"
                summary["error_type"] = type(exc).__name__
                raise
            finally:
                summary["elapsed_seconds"] = time.time() - started
                summary_handle.write(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="NEW diagnostic JSONL; never a training dataset")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "default.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--pids", help="Comma-separated PID suffixes from reward filenames, not host-namespace PIDs")
    parser.add_argument("--compare-metricx", action="store_true", help="Also send texts to configured MetricX service")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        pids = args.pids.split(",") if args.pids else None
        if pids is not None and any(not value.isdecimal() for value in pids):
            raise RewardError("--pids must be comma-separated numeric PID suffixes")
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        result = run_audit(args.run_dir, args.dataset, args.output, config, limit=args.limit,
                           pids=pids, compare_metricx=args.compare_metricx)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RewardError) as exc:
        message = str(exc) if isinstance(exc, RewardError) else type(exc).__name__
        LOG.error("Candidate audit failed: %s; inspect any partial output/summary", message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
