#!/usr/bin/env python3
"""Read a stopped FLEX run without GPU, audio decoding or MetricX requests.

Writes a NEW report directory, never edits training logs/checkpoints. Reward
windows are local callbacks, not reconstructed global GSPO groups. Examples
are diagnostic selections, not a random or independent evaluation set.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from flex_gspo.monitoring import number, reward_summary, slim_response
from flex_gspo.qalco import combine_reward, validate_settings


def rows(path, diagnostics):
    """Stream JSONL; explicitly count broken/truncated lines, never repair."""
    path = Path(path)
    initial = path.stat()
    diagnostics.update(path=str(path), bytes=initial.st_size, rows=0, bad_lines=0,
                       age_seconds=max(0, time.time() - initial.st_mtime))
    with path.open("rb") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig" if line_no == 1 else "utf-8"))
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
            except (UnicodeError, ValueError):
                diagnostics["bad_lines"] += 1
                continue
            diagnostics["rows"] += 1
            yield row
    final = path.stat()
    diagnostics["changed_during_read"] = (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns)


def stats(values):
    values = sorted(float(v) for v in values if number(v))
    if not values:
        return {"count": 0}
    return {"count": len(values), "mean": sum(values) / len(values),
            "min": values[0], "max": values[-1],
            "p50": values[(len(values)-1)//2], "p90": values[math.ceil(len(values)*0.9)-1]}


def native_window(records):
    keys = ("loss", "kl", "reward", "reward_std", "frac_reward_zero_std", "grad_norm",
            "clip_ratio/region_mean", "completions/mean_length", "learning_rate", "train_speed(s/it)")
    return {"rows": len(records), "iteration_range": [r.get("iteration") for r in records[::max(1, len(records)-1)]] if records else [],
            "metrics": {key: stats(r.get(key) for r in records) for key in keys},
            "zero_grad_steps": sum(r.get("grad_norm") == 0 for r in records)}


def compact(record):
    """Keep scalar/action evidence for windows; discard long hypothesis text."""
    traj = record.get("trajectory") or {}
    return {"reward": record.get("reward"), "components": record.get("components", {}),
            "trajectory": {"complete_input": traj.get("complete_input"), "turns": [
                {"response": slim_response(t.get("response", "")), "kind": t.get("kind", "audio")}
                for t in traj.get("turns", [])]}}


def example(record):
    traj = record.get("trajectory") or {}
    turns = traj.get("turns", [])
    return {key: record.get(key) for key in ("task_id", "global_step", "call", "candidate_index_in_local_batch", "reward", "translation", "components")} | {
        "complete_input": traj.get("complete_input"), "valid": traj.get("valid"),
        "invalid_reason": traj.get("invalid_reason") or traj.get("error"),
        "turn_count": len(turns), "final_audio_seconds": turns[-1].get("audio_end_seconds") if turns else None,
        "finish_reasons": dict(Counter(t.get("finish_reason", "missing") for t in turns)),
        "turns": [{key: t.get(key) for key in ("audio_end_seconds", "response", "finish_reason")} for t in turns]}


def reward_windows(groups):
    records = [r for group in groups for r in group[1]]
    return {"local_callbacks": len(groups), "step_range": [groups[0][0][1], groups[-1][0][1]] if groups else [],
            **(reward_summary(records) if records else {})}


def audit_rewards(path, window, settings, examples_limit):
    diagnostics, first, last = {}, [], deque(maxlen=window)
    totals, reasons, all_wait = Counter(), Counter(), 0
    selected, pairs, timing_pairs = [], [], []
    recent_invalid = deque(maxlen=examples_limit)
    current_key, group, seen_keys = None, [], set()
    repeated_callbacks = malformed_indices = 0
    formula_checked = formula_mismatches = uncheckable = 0
    quality_fail_delays = []
    same_text_timing_groups = 0

    def finish(key, batch):
        nonlocal repeated_callbacks, malformed_indices, same_text_timing_groups
        if not batch:
            return
        if key in seen_keys:
            repeated_callbacks += 1
        seen_keys.add(key)
        indices = [r.get("candidate_index_in_local_batch") for r in batch]
        if indices != list(range(len(batch))):
            malformed_indices += 1
        group_data = (key, [compact(r) for r in batch])
        if len(first) < window:
            first.append(group_data)
        last.append(group_data)
        by_task = {}
        for row in batch:
            by_task.setdefault(row.get("task_id"), []).append(row)
        for task_id, candidates in by_task.items():
            by_text = {}
            for row in candidates:
                if row.get("translation") and number((row.get("components") or {}).get("latency_seconds")):
                    by_text.setdefault(row["translation"], []).append(row)
            for text_rows in by_text.values():
                fast = min(text_rows, key=lambda r: r["components"]["latency_seconds"])
                slow = max(text_rows, key=lambda r: r["components"]["latency_seconds"])
                if slow["components"]["latency_seconds"] - fast["components"]["latency_seconds"] <= 0.01:
                    continue
                same_text_timing_groups += 1
                if len(timing_pairs) < examples_limit:
                    timing_pairs.append({"selection": "identical final text, different timing within local task callback",
                                         "file": path.name, "task_id": task_id,
                                         "faster": example(fast), "slower": example(slow)})
            usable = [r for r in candidates if number(r.get("reward"))]
            if len(usable) < 2 or len({r.get("translation") for r in usable}) < 2:
                continue
            low, high = min(usable, key=lambda r: r["reward"]), max(usable, key=lambda r: r["reward"])
            pairs.append({"selection": "largest local reward spread; NOT a complete global G8 claim",
                          "file": path.name, "task_id": task_id, "local_candidates": len(candidates),
                          "reward_spread": high["reward"]-low["reward"],
                          "lower": example(low), "higher": example(high)})
        pairs.sort(key=lambda pair: pair["reward_spread"], reverse=True)
        del pairs[examples_limit:]

    for record in rows(path, diagnostics):
        key = (record.get("call"), record.get("global_step"), record.get("timestamp_unix"))
        if group and key != current_key:
            finish(current_key, group)
            group = []
        current_key = key
        group.append(record)
        c = record.get("components") or {}
        tier = c.get("tier", "missing")
        totals[tier] += 1
        traj = record.get("trajectory") or {}
        turns = traj.get("turns", [])
        all_wait += bool(turns) and all(t.get("response", "").strip() == "<|WAIT|>" for t in turns)
        if tier == "invalid":
            reason = c.get("error") or c.get("reason") or traj.get("invalid_reason") or traj.get("error") or "unspecified"
            reasons[str(reason)] += 1
            if len(selected) < examples_limit:
                selected.append({"selection": "first invalid candidates (diagnostic, not random)",
                                 "file": path.name, **example(record)})
            recent_invalid.append({"selection": "latest invalid candidates (diagnostic, may overlap early selections)",
                                   "file": path.name, **example(record)})
        if tier == "quality_fail" and number(c.get("latency_seconds")):
            quality_fail_delays.append(c["latency_seconds"])
        if settings is not None:
            values = [c.get(k) for k in ("quality", "latency_seconds", "length_penalty")]
            if tier == "invalid":
                expected = -4.0
            elif all(number(v) for v in values):
                expected = combine_reward(*values, settings)["reward"]
            else:
                uncheckable += 1
                continue
            formula_checked += 1
            if not number(record.get("reward")) or not math.isclose(record["reward"], expected, rel_tol=1e-8, abs_tol=1e-7):
                formula_mismatches += 1
    finish(current_key, group)
    return {"source": diagnostics, "scope": "one rank/PID; local callback windows, not global GSPO groups",
            "early": reward_windows(first), "recent": reward_windows(list(last)),
            "windows_overlap": len(seen_keys) < 2*window, "all_candidates_tiers": dict(totals),
            "invalid_reasons": dict(reasons), "all_wait_candidates": all_wait,
            "quality_fail_latency_seconds": stats(quality_fail_delays),
            "same_text_different_timing_local_groups": same_text_timing_groups,
            "repeated_callback_keys": repeated_callbacks, "noncontiguous_candidate_index_callbacks": malformed_indices,
            "callback_completeness": "No end marker in legacy logs; a contiguous final prefix may still be partial.",
            "formula_recheck": {"enabled": settings is not None, "checked": formula_checked, "mismatches": formula_mismatches, "uncheckable": uncheckable,
                                "note": "Uses supplied current formula/config; hash match does not prove quality-model identity or semantic reward validity."}}, selected + list(recent_invalid) + timing_pairs, pairs


def load_settings(config_path, run_dir):
    if not config_path:
        return None, {"status": "not requested"}
    data = Path(config_path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    identities = []
    for path in sorted(run_dir.glob("launch_*.json")):
        item = json.loads(path.read_text(encoding="utf-8-sig"))
        identity = item.get("run_identity", {})
        identities.append((path.name, identity.get("reward_config_sha256")))
    matched = [name for name, old_hash in identities if old_hash == digest]
    if not matched or any(old_hash != digest for _, old_hash in identities):
        return None, {"status": "not rechecked: missing/mixed launch identities or changed config", "config_sha256": digest}
    settings = validate_settings(json.loads(data.decode("utf-8-sig")))
    return settings, {"status": "config hash matches launch identity", "launch_files": matched,
                      "settings": {key: settings[key] for key in ("quality_floor", "latency_metric", "latency_budget_seconds", "reference_length_ratio", "latency_weight", "length_weight")}}


def attach_sources(examples, dataset):
    wanted = {e.get("task_id") for e in examples}
    found, diagnostics = {}, {}
    for row in rows(Path(dataset), diagnostics):
        task = row.get("flex_task", row)
        task = json.loads(task) if isinstance(task, str) else task
        if not isinstance(task, dict) or task.get("id") not in wanted:
            continue
        item = {key: task.get(key) for key in ("source_text", "reference_text", "duration_seconds")}
        if task["id"] in found and found[task["id"]] != item:
            raise ValueError("Conflicting dataset ID: " + task["id"])
        found[task["id"]] = item
    for entry in examples:
        entry["source_metadata"] = found.get(entry.get("task_id"))
    return {**diagnostics, "requested_ids": len(wanted), "matched_ids": len(found)}


def checkpoint_inventory(root):
    """Metadata only: presence is NOT proof that optimizer state can resume."""
    result = []
    paths = set(root.glob("checkpoint-*")) | set(root.glob("v*/checkpoint-*"))
    for directory in sorted(p for p in paths if p.is_dir()):
        tracker = directory/"latest_checkpointed_iteration.txt"
        item = {"path": str(directory), "merged_export": directory.name.endswith("-merged"),
                "tracker_present": tracker.is_file(), "resumability_verified": False}
        if tracker.is_file():
            try:
                iteration = int(tracker.read_text(encoding="utf-8").strip())
                shard_dir = directory/f"iter_{iteration:07d}"
                item.update(iteration=iteration, common_pt_present=(shard_dir/"common.pt").is_file(),
                            nonmetadata_payload_present=any(p.is_file() and p.name not in {"common.pt", "metadata.json", ".metadata"}
                                                           and p.stat().st_size > 0 for p in shard_dir.rglob("*")))
            except ValueError:
                item["tracker_error"] = "not a positive integer iteration"
        result.append(item)
    return result


def launch_settings(root):
    selected = []
    allowed = ("group_size", "global_batch_size", "micro_batch_size", "steps_per_generation", "train_iters",
               "save_steps", "lr", "kl_beta", "max_length", "max_new_tokens", "vllm_memory", "vllm_max_num_seqs",
               "vllm_mm_processor_cache_gb", "recompute_granularity")
    for path in sorted(root.glob("launch_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        settings = doc.get("run_identity", {}).get("settings", {})
        # save_steps is intentionally excluded from run_identity; recover its actual argv value.
        argv = doc.get("argv", [])
        saved = {key: settings.get(key) for key in allowed}
        if "--save_steps" in argv and argv.index("--save_steps") + 1 < len(argv):
            saved["save_steps"] = argv[argv.index("--save_steps")+1]
        selected.append({"file": path.name, "settings": saved})
    return selected


def analyze(args):
    root = Path(args.run_dir).resolve()
    native = Path(args.native_log).resolve()
    if not root.is_dir() or not native.is_file() or root not in native.parents:
        raise ValueError("run-dir must exist and native-log must be a file inside it")
    if not 1 <= args.window <= 1000 or not 1 <= args.examples <= 20:
        raise ValueError("window must be 1..1000, examples must be 1..20")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise ValueError("Report output already exists; choose a NEW output directory (no overwrite)")
    reward_paths = sorted((root / "rewards").glob("rewards-rank*-pid*.jsonl"))
    if args.pids:
        pids = args.pids.split(",")
        if not all(p.isdigit() for p in pids):
            raise ValueError("pids must be comma-separated numeric LOG filename PIDs")
        reward_paths = [p for p in reward_paths if p.stem.rsplit("pid", 1)[-1] in pids]
        if {p.stem.rsplit("pid", 1)[-1] for p in reward_paths} != set(pids):
            raise ValueError("Some requested log PIDs were not found")
    ranks = [p.stem.split("-pid", 1)[0] for p in reward_paths]
    if len(ranks) != len(set(ranks)):
        raise ValueError("Multiple PID series per rank: use --pids to select this run, do not mix restarts")
    if not reward_paths:
        raise ValueError("No matching reward files")
    print("[audit] Reading training metrics; no model/API/audio access.", flush=True)
    nd = {}
    nr = [r for r in rows(native, nd) if "iteration" in r]
    ns = [r.get("iteration") for r in nr]
    report = {"schema_version": 1, "created_unix": time.time(),
              "scope": "Stopped-run diagnostics, NOT fixed-set evaluation or proof of translation improvement",
              "native": {"source": nd, "early": native_window(nr[:args.window]), "recent": native_window(nr[-args.window:]),
                         "latest": nr[-1] if nr else None, "duplicate_iterations": len(ns)-len(set(ns)),
                         "nonfinite_numeric_values": sum(isinstance(v, (int, float)) and not math.isfinite(v) for r in nr for v in r.values())},
              "rewards": [], "performance": [], "launch_settings": launch_settings(root),
              "checkpoint_inventory": checkpoint_inventory(root)}
    settings, report["reward_config"] = load_settings(args.reward_config, root)
    examples = []
    for path in reward_paths:
        print("[audit] Reading " + path.name, flush=True)
        summary, invalid, pairs = audit_rewards(path, args.window, settings, args.examples)
        report["rewards"].append(summary)
        examples.extend(invalid + pairs)
        perf = root / "performance" / path.name.replace("rewards-", "performance-")
        if perf.is_file():
            pd = {}
            values = list(rows(perf, pd))
            fields = ("generation_seconds", "weight_sync_seconds", "rollout_requests_seconds", "offload_to_cpu_seconds", "reload_to_gpu_seconds", "vllm_sleep_seconds")
            report["performance"].append({"source": pd, "scope": "per rank; host unsynchronized timings may overlap; do not sum ranks",
                "metrics": {key: stats(v.get(key) for v in values) for key in fields},
                "latest": values[-1] if values else None})
    if args.dataset:
        print("[audit] Looking up selected source/reference texts (no WAV reads).", flush=True)
        report["dataset_lookup"] = attach_sources(examples, args.dataset)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "examples.jsonl").open("x", encoding="utf-8") as f:
        for entry in examples:
            f.write(json.dumps(json_safe(entry), ensure_ascii=False, allow_nan=False) + "\n")
    print("[audit] Saved " + str(output / "summary.json"), flush=True)
    print("[audit] Saved " + str(output / "examples.jsonl") + " (contains source/reference and model text; inspect before sharing)", flush=True)
    return report


def json_safe(value):
    """Preserve non-finite evidence as a string; never emit invalid JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--native-log", required=True, help="Exact vN-.../logging.jsonl; do not guess newest run")
    p.add_argument("--output-dir", required=True, help="New diagnostic directory, never the original logs")
    p.add_argument("--pids", help="Select log-filename PIDs, not host PIDs, when a directory was reused")
    p.add_argument("--dataset", help="Optional train.jsonl, read once for selected source/reference texts")
    p.add_argument("--reward-config", help="Optional config; recheck formula only if hash matches launch identity")
    p.add_argument("--window", type=int, default=10, help="Native metric rows / local reward callbacks per window")
    p.add_argument("--examples", type=int, default=2, help="Max invalid examples and local candidate pairs per rank")
    return p


def main(argv=None):
    try:
        analyze(parser().parse_args(argv))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("[audit] ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
