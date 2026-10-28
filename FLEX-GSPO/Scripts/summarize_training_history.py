#!/usr/bin/env python3
"""Read ALL complete log lines for one launch; never load models/audio/services.

Standalone, standard library only. Native iteration N is matched to reward
global_step N-1 ONLY for the verified one-generation/one-update recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time

NATIVE = ("reward", "loss", "kl", "reward_std", "frac_reward_zero_std",
          "grad_norm", "learning_rate", "clip_ratio/region_mean",
          "completions/mean_length", "train_speed(s/it)")
PARTS = ("quality", "comet22_score", "latency_seconds", "quality_gate",
         "quality_deficit_penalty", "latency_cost_raw", "latency_penalty",
         "weighted_brevity_penalty", "weighted_repetition_penalty",
         "normalized_quality", "normalized_latency", "weighted_quality_signal", "weighted_latency_signal",
         "quality_group_std", "latency_group_std", "quality_normalization_scale", "latency_normalization_scale",
         "estimated_advantage", "utility_raw", "failure_reward")
FLAGS = ("protocol_valid", "empty_translation", "quality_pass", "feasible")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def dump(value):
    # Do not round small learning rates to zero.
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Stat:
    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.squares = 0.0
        self.low = self.high = None

    def add(self, value):
        if not finite(value):
            return
        self.n += 1
        self.total += value
        self.squares += value * value
        self.low = value if self.low is None else min(self.low, value)
        self.high = value if self.high is None else max(self.high, value)

    def merge(self, other):
        if not other.n:
            return
        self.n += other.n
        self.total += other.total
        self.squares += other.squares
        self.low = other.low if self.low is None else min(self.low, other.low)
        self.high = other.high if self.high is None else max(self.high, other.high)

    def result(self):
        if not self.n:
            return {"count": 0}
        mean = self.total / self.n
        return {"count": self.n, "mean": mean, "min": self.low, "max": self.high,
                "std": math.sqrt(max(0.0, self.squares / self.n - mean * mean))}


def snapshot(path, metadata):
    """Bounded file-size snapshot; append-only growth is ignored, not reread."""
    start = path.stat()
    meta = {"path": str(path), "snapshot_bytes": start.st_size,
            "age_seconds": max(0.0, time.time() - start.st_mtime),
            "complete_lines": 0, "bad_complete_lines": 0, "partial_tail_ignored": False}
    metadata.append(meta)
    last_progress = time.monotonic()
    with path.open("rb") as handle:
        line_no = 0
        while handle.tell() < start.st_size:
            raw = handle.readline(start.st_size - handle.tell())
            line_no += 1
            if not raw.endswith(b"\n"):
                meta["partial_tail_ignored"] = True
                break
            if not raw.strip():
                continue
            meta["complete_lines"] += 1
            try:
                row = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(row, dict):
                    raise ValueError("Expected JSON object")
            except (ValueError, UnicodeError):
                meta["bad_complete_lines"] += 1
                continue
            yield row
            if time.monotonic() - last_progress >= 5:
                print(f"[scan] {path.name}: lines={line_no}, bytes={handle.tell()}/{start.st_size}",
                      file=sys.stderr, flush=True)
                last_progress = time.monotonic()
    if path.stat().st_size < start.st_size:
        raise ValueError("Log shrank while reading: " + str(path))


def arg(plan, flag):
    values = plan.get("argv", [])
    return next((values[i + 1] for i, item in enumerate(values[:-1]) if item == flag), None)


def step(row):
    value = row.get("iteration", row.get("global_step"))
    if type(value) is int and value >= 0:
        return value
    match = re.fullmatch(r"\s*(\d+)(?:\s*/\s*\d+)?\s*", str(value))
    return int(match[1]) if match else None


def summarize_native(rows):
    metrics = {key: Stat() for key in NATIVE}
    bad = Counter()
    for row in rows:
        for key, value in metrics.items():
            item = row.get(key)
            value.add(item)
            if type(item) in (int, float) and not math.isfinite(item):
                bad[key] += 1
    present = {step(row) for row in rows}
    missing = sorted(set(range(min(present), max(present) + 1)) - present) if present else []
    return {"steps": len(rows), "range": [step(rows[0]), step(rows[-1])] if rows else [],
            "missing_steps_within_range": missing,
            "metrics": {key: value.result() for key, value in metrics.items()},
            "nonfinite": dict(bad), "zero_grad_steps": sum(r.get("grad_norm") == 0 for r in rows)}


class RewardStep:
    def __init__(self):
        self.metrics = {key: Stat() for key in ("reward", *PARTS)}
        self.flags = Counter()
        self.known = Counter()
        self.tiers = Counter()
        self.ranks = Counter()
        self.calls = defaultdict(set)
        self.groups = defaultdict(list)
        self.count = self.all_wait = self.eof_invalid = 0

    def add(self, row, rank):
        comp = row.get("components") or {}
        trajectory = row.get("trajectory") or {}
        turns = trajectory.get("turns") or []
        self.count += 1
        self.ranks[rank] += 1
        self.calls[rank].add((row.get("call"), row.get("timestamp_unix")))
        for key, stat in self.metrics.items():
            stat.add(row.get("reward") if key == "reward" else comp.get(key))
        for key in FLAGS:
            if type(comp.get(key)) is bool:
                self.known[key] += 1
                self.flags[key] += comp[key]
        self.tiers[comp.get("tier", "missing")] += 1
        all_wait = bool(turns) and all(isinstance(t, dict) and
                                     str(t.get("response", "")).strip() == "<|WAIT|>" for t in turns)
        self.all_wait += all_wait
        self.eof_invalid += bool(comp.get("protocol_valid") is False and turns
                                 and isinstance(turns[-1], dict) and turns[-1].get("kind") == "eof")
        self.groups[comp.get("global_group_id")].append({
            "index": comp.get("group_candidate_index"), "size": comp.get("group_size"),
            "finalized": comp.get("group_finalized"), "task": row.get("task_id"),
            "hash": row.get("task_hash"), "request": row.get("request_id"),
            "reward": row.get("reward"), "all_wait": all_wait})

    def problems(self, expected, group_size, world):
        bad = []
        if self.count != expected:
            bad.append(f"candidate_count={self.count}, expected={expected}")
        if set(self.ranks) != set(range(world)) or len(set(self.ranks.values())) != 1:
            bad.append("rank coverage/count mismatch")
        if any(len(values) != 1 for values in self.calls.values()):
            bad.append("multiple callbacks for one rank/step")
        if self.metrics["reward"].n != self.count:
            bad.append("nonfinite/missing reward")
        requests = []
        for gid, group in self.groups.items():
            indices = [r["index"] for r in group]
            if (not isinstance(gid, str) or not gid or len(group) != group_size
                    or any(type(i) is not int for i in indices) or set(indices) != set(range(group_size))
                    or any(r["size"] != group_size or r["finalized"] is not True for r in group)
                    or any(not isinstance(r["task"], str) or not r["task"]
                           or not isinstance(r["hash"], str) or not r["hash"] for r in group)
                    or len({(r["task"], r["hash"]) for r in group}) != 1):
                bad.append("incomplete/unfinalized/mixed G8 metadata")
                break
            requests.extend(r["request"] for r in group)
        if any(not isinstance(r, str) or not r for r in requests) or len(set(requests)) != len(requests):
            bad.append("missing/duplicate request identity")
        return bad


def summarize_rewards(items):
    metrics = {key: Stat() for key in ("reward", *PARTS)}
    flags, known, tiers = Counter(), Counter(), Counter()
    count = wait = eof = groups = zero_groups = wait_groups = 0
    for value in items:
        count += value.count
        wait += value.all_wait
        eof += value.eof_invalid
        flags.update(value.flags)
        known.update(value.known)
        tiers.update(value.tiers)
        for key in metrics:
            metrics[key].merge(value.metrics[key])
        for group in value.groups.values():
            groups += 1
            zero_groups += max(r["reward"] for r in group) == min(r["reward"] for r in group)
            wait_groups += all(r["all_wait"] for r in group)
    return {"covered_steps": len(items), "candidates": count, "complete_groups": groups,
            "zero_reward_variance_groups": zero_groups, "all_wait_groups": wait_groups,
            "all_wait_candidates": wait, "eof_protocol_invalid_candidates": eof,
            "tiers": dict(tiers), "metrics": {key: value.result() for key, value in metrics.items()},
            "flags": {key: {"known": known[key], "true": flags[key],
                            "fraction": flags[key] / known[key] if known[key] else None} for key in FLAGS}}


def analyze(args):
    root = Path(args.run_dir).resolve()
    if not root.is_dir():
        raise ValueError("Missing run directory: " + str(root))
    metadata, warnings = [], []
    plans = sorted(root.glob("launch_*.json"), key=lambda p: p.stat().st_mtime_ns)
    explicit_launch = getattr(args, "launch_file", None)
    launch = Path(explicit_launch).resolve() if explicit_launch else plans[-1] if plans else None
    if launch:
        try:
            launch.relative_to(root)
        except ValueError:
            raise ValueError("Launch file must be inside run_dir")
    plan = json.loads(launch.read_text(encoding="utf-8-sig")) if launch else {}
    paths = [Path(args.native_log).resolve()] if args.native_log else list(root.glob("*/logging.jsonl"))
    if not args.native_log and launch:
        paths = [p for p in paths if p.stat().st_mtime >= launch.stat().st_mtime - 2]
    if len(paths) != 1:
        raise ValueError("Need exactly one native log; use --native-log with the exact logging.jsonl")
    native = paths[0].resolve()
    try:
        native.relative_to(root)
    except ValueError:
        raise ValueError("Native log must be inside run_dir")
    if launch and native.stat().st_mtime < launch.stat().st_mtime - 2:
        raise ValueError("Native log predates selected launch; select the matching --launch-file and --native-log")
    if launch:
        warnings.append("Native logging.jsonl has no session_id: exact file selection and timestamps were checked, "
                        "but its session identity cannot be proven from native rows alone")
    rows, previous = [], -1
    for row in snapshot(native, metadata):
        current = step(row)
        if current is None:
            continue
        if current <= previous:
            raise ValueError(f"Duplicate/nonmonotonic native step {current}; select one uninterrupted log")
        previous = current
        rows.append(row)
    if not rows:
        raise ValueError("No completed native training steps found")
    completed = {step(row) for row in rows}
    rewards, excluded, ignored = {}, {}, Counter()
    session, run_id = plan.get("session_id"), plan.get("run_id")
    if not args.native_only:
        if not session or not run_id:
            raise ValueError("Full reward summary requires launch session_id/run_id; use --native-only otherwise")
        if arg(plan, "--num_iterations") != "1" or arg(plan, "--steps_per_generation") != "1":
            raise ValueError("Reward step alignment only supports num_iterations=1 and steps_per_generation=1")
        expected = int(arg(plan, "--global_batch_size"))
        group_size = int(arg(plan, "--num_generations"))
        world = int(arg(plan, "--nproc_per_node")) * int(arg(plan, "--nnodes") or 1)
        if min(expected, group_size, world) < 1 or expected % group_size:
            raise ValueError("Invalid launch batch/group/world configuration")
        for path in sorted((root / "rewards").glob("rewards-rank*-pid*.jsonl")):
            match = re.fullmatch(r"rewards-rank(\d+)-pid(\d+)\.jsonl", path.name)
            if not match:
                continue
            print("[scan] " + str(path), file=sys.stderr, flush=True)
            rank = int(match[1])
            for row in snapshot(path, metadata):
                if row.get("session_id") != session or row.get("run_id") != run_id:
                    ignored["other_session_or_run"] += 1
                    continue
                gs = row.get("global_step")
                if type(gs) is not int or gs + 1 not in completed:
                    ignored["not_in_completed_native_steps"] += 1
                    continue
                current = gs + 1
                if not isinstance(row.get("components"), dict) or not isinstance(row.get("trajectory"), dict):
                    excluded.setdefault(current, []).append("malformed reward structure")
                    continue
                rewards.setdefault(current, RewardStep()).add(row, rank)
        for current in sorted(completed):
            problems = rewards[current].problems(expected, group_size, world) if current in rewards else ["missing rewards"]
            if problems:
                excluded.setdefault(current, []).extend(problems)
        rewards = {s: value for s, value in rewards.items() if s not in excluded}
        if excluded:
            warnings.append("Incomplete/inconsistent reward steps excluded; native rows remain included")
    if any(meta["bad_complete_lines"] for meta in metadata):
        warnings.append("Malformed complete JSONL lines detected; inspect file_diagnostics before trusting coverage")
    differences = Stat()
    mismatches = []
    for row in rows:
        current = step(row)
        if current in rewards and finite(row.get("reward")):
            candidate_mean = rewards[current].metrics["reward"].result()["mean"]
            diff = candidate_mean - row["reward"]
            differences.add(abs(diff))
            if abs(diff) > 1e-5:
                mismatches.append({"step": current, "native_reward": row["reward"],
                                   "candidate_mean": candidate_mean, "difference": diff})
    if mismatches:
        warnings.append("Native reward differs from full-candidate arithmetic mean on some steps; "
                        "check identity/alignment/framework aggregation, not automatically a reward bug")
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(root), "native_log": str(native), "launch": str(launch) if launch else None,
              "session_id": session, "run_id": run_id,
              "launch_selection": "explicit" if explicit_launch else "latest_by_mtime",
              "scope": "All complete snapshot lines; training batches, not fixed validation evidence",
              "eof_counter_scope": "Protocol-invalid candidates whose final recorded turn is EOF; "
                                   "not a full per-turn parser audit proving EOF-only failure",
              "native": summarize_native(rows), "first_window": summarize_native(rows[:args.block_size]),
              "last_window": summarize_native(rows[-args.block_size:]), "blocks": [],
              "rewards": summarize_rewards(list(rewards.values())) if not args.native_only else None,
              "reward_reconciliation": {"absolute_difference": differences.result(), "tolerance": 1e-5,
                                        "mismatch_steps": len(mismatches), "first_mismatches": mismatches[:20]},
              "excluded_reward_steps": excluded, "ignored_reward_rows": dict(ignored),
              "file_diagnostics": metadata, "warnings": warnings}
    for offset in range(0, len(rows), args.block_size):
        block = rows[offset:offset + args.block_size]
        info = {"native": summarize_native(block)}
        if not args.native_only:
            info["rewards"] = summarize_rewards([rewards[step(r)] for r in block if step(r) in rewards])
        report["blocks"].append(info)
    return report, rows, rewards


def compact_native(value):
    return {"steps": value["steps"], "range": value["range"],
            "mean": {k: v.get("mean") for k, v in value["metrics"].items()},
            "reward_std_over_steps": value["metrics"]["reward"].get("std"),
            "nonfinite": value["nonfinite"], "missing_steps": value["missing_steps_within_range"]}


def compact_rewards(value):
    if value is None:
        return None
    count = value["candidates"]
    return {"covered_steps": value["covered_steps"], "candidates": count,
            "groups": value["complete_groups"], "zero_variance_groups": value["zero_reward_variance_groups"],
            "all_wait_groups": value["all_wait_groups"], "all_wait_candidates": value["all_wait_candidates"],
            "eof_protocol_invalid": value["eof_protocol_invalid_candidates"], "tiers": value["tiers"],
            "mean": {k: v.get("mean") for k, v in value["metrics"].items()},
            "quality_scored": value["metrics"]["quality"]["count"],
            "fractions": {k: v["fraction"] for k, v in value["flags"].items()},
            "all_wait_fraction": value["all_wait_candidates"] / count if count else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--native-log")
    parser.add_argument("--launch-file", help="Exact launch JSON for this session; must be inside run_dir")
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--output-dir", help="New directory; existing output is never overwritten")
    parser.add_argument("--native-only", action="store_true", help="Skip candidate logs; summarize native reward/loss only")
    args = parser.parse_args(argv)
    if args.block_size < 1:
        parser.error("block-size must be positive")
    try:
        report, rows, rewards = analyze(args)
        output = Path(args.output_dir) if args.output_dir else Path(args.run_dir) / "diagnostics" / (
            "history_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(dump(report) + "\n", encoding="utf-8")
        fields = ["iteration", *NATIVE, "elapsed_time", "remaining_time"]
        with (output / "all_steps.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        lines = ["全程训练摘要（未平滑；非固定验证集评估）", "身份: " + dump({k: report[k] for k in (
            "run_dir", "native_log", "session_id", "run_id")}), "全部训练步: " + dump(report["native"]),
                 "最初窗口: " + dump(report["first_window"]), "最近窗口: " + dump(report["last_window"])]
        if report["rewards"] is not None:
            lines.append("完整候选批次总计: " + dump(report["rewards"]))
        for block in report["blocks"]:
            lines.append("分段统计: " + dump(block))
        lines.extend(["未纳入奖励汇总的训练步: " + dump(report["excluded_reward_steps"]),
                      "忽略行: " + dump(report["ignored_reward_rows"]),
                      "文件读取检查: " + dump(report["file_diagnostics"]), "提示: " + dump(report["warnings"])])
        text = "\n".join(lines) + "\n"
        (output / "summary.txt").write_text(text, encoding="utf-8")
        brief = [lines[0], lines[1], "全部训练步: " + dump(compact_native(report["native"])),
                 "最初窗口: " + dump(compact_native(report["first_window"])),
                 "最近窗口: " + dump(compact_native(report["last_window"])),
                 "完整奖励总计: " + dump(compact_rewards(report["rewards"]))]
        for block in report["blocks"]:
            brief.append("分段: " + dump({"native": compact_native(block["native"]),
                                         "rewards": compact_rewards(block.get("rewards"))}))
        brief.extend(["原生与候选均值核对: " + dump(report["reward_reconciliation"]),
                      "未纳入奖励汇总的训练步: " + dump(report["excluded_reward_steps"]),
                      "提示: " + dump(report["warnings"])])
        compact_text = "\n".join(brief) + "\n"
        (output / "compact_summary.txt").write_text(compact_text, encoding="utf-8")
        print(compact_text)
        print("[summary] 请分享精简报告: " + str(output / "compact_summary.txt"))
        print("[summary] 报告: " + str(output / "summary.txt"))
        print("[summary] 每一步原始值: " + str(output / "all_steps.csv"))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("[summary] ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
