#!/usr/bin/env python3
"""Read-only evidence for a reward revision; standard library, no model/API/audio load.

Reads bounded snapshots of all completed steps, validates actual global groups,
and exports original candidate rows for offline replay. Does not change rewards.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Scripts import summarize_training_history as history
from flex_gspo.protocol import parse_response


def js(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def indexed_snapshot(path, inventory):
    size = path.stat().st_size
    meta = {"path": str(path), "snapshot_bytes": size, "partial_tail_ignored": False,
            "lines": 0, "bad_complete_lines": 0}
    inventory.append(meta)
    last = time.monotonic()
    with path.open("rb") as stream:
        while stream.tell() < size:
            offset = stream.tell()
            raw = stream.readline(size - offset)
            if not raw.endswith(b"\n"):
                meta["partial_tail_ignored"] = True
                break
            if not raw.strip():
                continue
            meta["lines"] += 1
            try:
                row = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(row, dict):
                    raise ValueError("JSON object required")
            except (ValueError, UnicodeError):
                meta["bad_complete_lines"] += 1
                raise ValueError(f"Corrupt complete JSONL line: {path}:{meta['lines']}")
            yield row, {"path": str(path), "offset": offset, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()}
            if time.monotonic() - last >= 5:
                print(f"[collect] {path.name}: {stream.tell()}/{size} bytes", flush=True)
                last = time.monotonic()
    if path.stat().st_size < size:
        raise ValueError("Log shrank during collection: " + str(path))


def inspect_turns(row):
    trajectory = row.get("trajectory") or {}
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        raise ValueError("Missing original trajectory turns")
    actions, failures = Counter(), []
    tiny = punctuated = ellipses = 0
    for number, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            raise ValueError("Malformed turn record")
        error = None
        try:
            action, delta = parse_response(turn.get("response"))
            actions[action] += 1
            if delta.strip():
                units = len(re.sub(r"\W", "", delta, flags=re.UNICODE))
                tiny += units <= 4
                punctuated += delta.rstrip().endswith(("。", ".", "!", "！", "?", "？"))
                ellipses += "…" in delta or "..." in delta
        except ValueError as exc:
            error = str(exc)
        finish = turn.get("finish_reason")
        if finish not in ("stop", "eos"):
            error = error or "finish_reason=" + str(finish)
        if error:
            failures.append({"turn": number, "kind": turn.get("kind"), "error": error})
    return {"actions": dict(actions), "turn_count": len(turns), "failure_turns": failures,
            "observed_all_wait": bool(turns) and actions["WAIT"] == len(turns),
            "eof_present": any(t.get("kind") == "eof" for t in turns),
            "tiny_text_turns_le4_alnum": tiny, "terminal_punctuation_turns": punctuated,
            "ellipsis_turns": ellipses}


class AuditStep(history.RewardStep):
    def add_indexed(self, row, rank, pointer):
        super().add(row, rank)
        comp = row["components"]
        item = self.groups[comp.get("global_group_id")][-1]
        observed = inspect_turns(row)
        item.update({"pointer": pointer, "observed": observed,
                     "duration_seconds": row.get("duration_seconds"),
                     "episode_complete": comp.get("protocol_valid") is True and not observed["failure_turns"] and all(
                         row["trajectory"].get(k) is True for k in ("complete_input", "episode_complete", "eof_completed")),
                     "translation_chars": len(row.get("translation") or ""),
                     "q": comp.get("quality"), "gated_latency": comp.get("latency_penalty"),
                     "latency_seconds": comp.get("latency_seconds"),
                     "protocol_valid": comp.get("protocol_valid"),
                     "empty": comp.get("empty_translation"), "tier": comp.get("tier"),
                     "brevity_cost": comp.get("weighted_brevity_penalty"),
                     "reward_parts": {k: comp.get(k) for k in history.PARTS}})


def describe_group(step, gid, candidates):
    q = [r["q"] for r in candidates if history.finite(r["q"])]
    lat = [r["gated_latency"] for r in candidates if history.finite(r["gated_latency"])]
    empty = sum(r["empty"] is True for r in candidates)
    waits = sum(r["observed"]["observed_all_wait"] and r["episode_complete"] is True for r in candidates)
    winner = max(candidates, key=lambda r: r["reward"])
    winner_q_gap = max(q) - winner["q"] if q and history.finite(winner["q"]) else None
    component_statistics = {}
    for name in history.PARTS:
        values = [r["reward_parts"][name] for r in candidates if history.finite(r["reward_parts"][name])]
        component_statistics[name] = {"count": len(values), "missing": len(candidates) - len(values),
            "range_observed": max(values) - min(values) if values else None,
            "std_population_full_group": statistics.pstdev(values) if len(values) == len(candidates) else None}
    return {"native_step": step, "global_group_id": gid, "task_id": candidates[0]["task"],
            "task_hash": candidates[0]["hash"], "component_statistics": component_statistics,
            "duration_seconds": candidates[0]["duration_seconds"],
            "all_wait_completed_count": waits, "empty_count": empty,
            "reward_mean": statistics.mean(r["reward"] for r in candidates),
            "reward_std_sample": statistics.stdev(r["reward"] for r in candidates),
            "quality_count": len(q), "quality_range": max(q) - min(q) if q else None,
            "quality_std_population": statistics.pstdev(q) if len(q) == len(candidates) else None,
            "gated_latency_count": len(lat),
            "gated_latency_std_population": statistics.pstdev(lat) if len(lat) == len(candidates) else None,
            "winner_quality_gap_from_best": winner_q_gap,
            "translation_chars_range": [min(r["translation_chars"] for r in candidates),
                                         max(r["translation_chars"] for r in candidates)],
            "protocol_invalid_count": sum(r["protocol_valid"] is False for r in candidates),
            "degenerate_count": sum(r["tier"] == "degenerate" for r in candidates)}


def select_groups(groups, limit):
    selected, used_tasks, used_keys = [], set(), set()
    quota = max(1, limit // 5)
    recent = sorted(groups, key=lambda g: (-g[0]["native_step"], g[0]["global_group_id"]))
    categories = [
        ("整组完成后全WAIT", lambda g: g["all_wait_completed_count"] == 8),
        ("部分候选空译", lambda g: 0 < g["empty_count"] < 8),
        ("协议错误或机械循环", lambda g: g["protocol_invalid_count"] or g["degenerate_count"]),
        ("全员有评分且质量接近、长度有差异", lambda g: g["quality_count"] == 8 and
         g["quality_range"] <= .03 and g["translation_chars_range"][1] - g["translation_chars_range"][0] >= 8),
    ]
    for label, predicate in categories:
        added = 0
        for desc, candidates in recent:
            key = (desc["native_step"], desc["global_group_id"])
            if predicate(desc) and desc["task_id"] not in used_tasks and key not in used_keys:
                selected.append((label, desc, candidates))
                used_tasks.add(desc["task_id"])
                used_keys.add(key)
                added += 1
                if added >= quota or len(selected) >= limit:
                    break
        if len(selected) >= limit:
            return selected
    # Reproducible hash selection across the whole observed run, not only recent steps.
    for desc, candidates in sorted(groups, key=lambda g: digest([g[0]["native_step"], g[0]["global_group_id"]])):
        if desc["task_id"] in used_tasks:
            continue
        selected.append(("全运行哈希补充抽样", desc, candidates))
        used_tasks.add(desc["task_id"])
        if len(selected) >= limit:
            break
    return selected


def load_tasks(path, wanted):
    found = {}
    print("[collect] Matching source/reference from manifest; audio is not read.", flush=True)
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("flex_task", row)
            task = json.loads(task) if isinstance(task, str) else task
            if task.get("id") in wanted:
                if task["id"] in found:
                    raise ValueError("Duplicate selected task ID in dataset: " + task["id"])
                found[task["id"]] = task
            if number % 50000 == 0:
                print(f"[collect] manifest lines={number} matched={len(found)}/{len(wanted)}", flush=True)
    if set(found) != wanted:
        raise ValueError("Missing selected task IDs: " + repr(sorted(wanted - set(found))))
    return found


def read_original(item):
    p = item["pointer"]
    with Path(p["path"]).open("rb") as stream:
        stream.seek(p["offset"])
        raw = stream.read(p["bytes"])
    if hashlib.sha256(raw).hexdigest() != p["sha256"]:
        raise ValueError("Selected log record changed during audit: " + p["path"])
    return json.loads(raw.decode("utf-8-sig"))


def distribution(values):
    values = sorted(v for v in values if history.finite(v))
    if not values:
        return {"count": 0}
    def quantile(p):
        position = (len(values) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"count": len(values), "min": values[0], "p25": quantile(.25),
            "median": quantile(.5), "p75": quantile(.75), "max": values[-1]}


def collect(args):
    run = Path(args.run_dir).resolve()
    launch, native = Path(args.launch_file).resolve(), Path(args.native_log).resolve()
    launch.relative_to(run)
    native.relative_to(run)
    plan = json.loads(launch.read_text(encoding="utf-8-sig"))
    config_path = getattr(args, "reward_config", None) or plan.get("environment", {}).get("FLEX_REWARD_CONFIG")
    config_evidence = {"path": config_path, "expected_sha256": plan.get("run_identity", {}).get("reward_config_sha256"),
                       "available": False, "matches_launch": None}
    if config_path and Path(config_path).is_file():
        config_bytes = Path(config_path).read_bytes()
        actual = hashlib.sha256(config_bytes).hexdigest()
        config_evidence.update(available=True, sha256=actual,
                               matches_launch=actual == config_evidence["expected_sha256"] if config_evidence["expected_sha256"] else None,
                               config=json.loads(config_bytes.decode("utf-8-sig")))
    if not plan.get("session_id") or not plan.get("run_id"):
        raise ValueError("Launch must contain session_id and run_id")
    expected = {"--num_generations": "8", "--steps_per_generation": "1", "--num_iterations": "1",
                "--reward_funcs": "qalco_online", "--reward_weights": "1"}
    if any(str(history.arg(plan, k)) != v for k, v in expected.items()):
        raise ValueError("Unsupported recipe: requires G8, one update/generation, scalar QALCO")
    scale_rewards = history.arg(plan, "--scale_rewards")
    if scale_rewards not in ("group", "none"):
        raise ValueError("Unsupported advantage scaling; expected group or none")
    if history.arg(plan, "--kl_in_reward") not in (None, "false", "False") or history.arg(plan, "--advantage_estimator") not in (None, "grpo"):
        raise ValueError("Cannot estimate group advantages for this recipe")
    if native.stat().st_mtime + 2 < launch.stat().st_mtime:
        raise ValueError("Native log predates launch; select matching files")
    batch = int(history.arg(plan, "--global_batch_size"))
    world = int(history.arg(plan, "--nproc_per_node")) * int(history.arg(plan, "--nnodes") or 1)
    if batch <= 0 or batch % 8 or world <= 0:
        raise ValueError("Invalid batch/world configuration")
    inventory, native_rows = [], []
    for row in history.snapshot(native, inventory):
        current = history.step(row)
        if current is None:
            continue
        if native_rows and current <= history.step(native_rows[-1]):
            raise ValueError("Native steps duplicate or decrease; choose one uninterrupted run")
        native_rows.append(row)
    if not native_rows or any(m.get("bad_complete_lines") for m in inventory):
        raise ValueError("Missing or corrupt native training steps")
    complete = {history.step(r) for r in native_rows}
    steps, ignored = {}, Counter()
    for path in sorted((run / "rewards").glob("rewards-rank*-pid*.jsonl")):
        match = re.fullmatch(r"rewards-rank(\d+)-pid\d+\.jsonl", path.name)
        if not match:
            continue
        rank = int(match[1])
        print("[collect] Scanning " + path.name, flush=True)
        for row, pointer in indexed_snapshot(path, inventory):
            if row.get("session_id") != plan["session_id"] or row.get("run_id") != plan["run_id"]:
                ignored["other_session_or_run"] += 1
                continue
            gs = row.get("global_step")
            if type(gs) is not int or gs + 1 not in complete:
                ignored["not_completed_native_step"] += 1
                continue
            if not isinstance(row.get("components"), dict) or not isinstance(row.get("trajectory"), dict):
                raise ValueError("Malformed reward structure")
            if (scale_rewards == "none") != (row.get("reward_version") == 4):
                raise ValueError("Reward version and launch advantage scaling disagree")
            steps.setdefault(gs + 1, AuditStep()).add_indexed(row, rank, pointer)
    excluded = {s: steps[s].problems(batch, 8, world) if s in steps else ["missing reward records"] for s in sorted(complete)}
    excluded = {s: v for s, v in excluded.items() if v}
    good = {s: obj for s, obj in steps.items() if s not in excluded}
    groups = [(describe_group(s, gid, rows), rows) for s, obj in sorted(good.items()) for gid, rows in sorted(obj.groups.items())]
    if not groups:
        raise ValueError("No validated complete G8 groups; inspect launch/log coverage")
    tasks, failure_kinds, failure_rounds, actions = defaultdict(Counter), Counter(), Counter(), Counter()
    task_hashes = {}
    observed = Counter()
    for desc, rows in groups:
        if desc["task_id"] in task_hashes and task_hashes[desc["task_id"]] != desc["task_hash"]:
            raise ValueError("Same task ID has different task hashes within this run: " + desc["task_id"])
        task_hashes[desc["task_id"]] = desc["task_hash"]
        task = tasks[desc["task_id"]]
        task["groups_seen"] += 1
        task["candidates_seen"] += 8
        task["empty_candidates"] += desc["empty_count"]
        task["all_wait_completed_candidates"] += desc["all_wait_completed_count"]
        task["all_wait_groups"] += desc["all_wait_completed_count"] == 8
        task["duration_seconds"] = desc["duration_seconds"]
        for row in rows:
            ob = row["observed"]
            actions.update(ob["actions"])
            for name in ("tiny_text_turns_le4_alnum", "terminal_punctuation_turns", "ellipsis_turns"):
                observed[name] += ob[name]
            observed["empty_after_eof"] += row["empty"] is True and ob["eof_present"]
            observed["empty_without_eof"] += row["empty"] is True and not ob["eof_present"]
            if ob["failure_turns"]:
                first = ob["failure_turns"][0]
                failure_kinds[str(first["kind"])] += 1
                failure_rounds[str(first["turn"])] += 1
    empty_tasks = sorted(({"task_id": tid, "task_hash": task_hashes[tid], **dict(v)} for tid, v in tasks.items() if v["empty_candidates"]),
                         key=lambda v: (-v["all_wait_groups"], -v["empty_candidates"], v["task_id"]))
    selected = select_groups(groups, args.groups)
    wanted = {desc["task_id"] for _, desc, _ in selected}
    sources = load_tasks(Path(args.dataset), wanted)
    exports = []
    for label, desc, items in selected:
        task = sources[desc["task_id"]]
        if any(digest(task) != item["hash"] for item in items):
            raise ValueError("Selected task differs from original training manifest: " + task["id"])
        exports.append({"selection": label, "group": desc, "task": task,
                        "candidates": [read_original(item) for item in sorted(items, key=lambda i: i["index"])]})
    mismatches = []
    for row in native_rows:
        s = history.step(row)
        if s in good and history.finite(row.get("reward")):
            mean = good[s].metrics["reward"].result()["mean"]
            if abs(mean - row["reward"]) > 1e-5:
                mismatches.append({"step": s, "native": row["reward"], "candidate_mean": mean})
    blocks = []
    for start in range(0, len(native_rows), args.block_size):
        chunk = native_rows[start:start + args.block_size]
        blocks.append({"native": history.compact_native(history.summarize_native(chunk)),
                       "rewards": history.compact_rewards(history.summarize_rewards([good[history.step(r)] for r in chunk if history.step(r) in good]))})
    report = {"schema_version": 1, "session_id": plan["session_id"], "run_id": plan["run_id"],
              "native_log": str(native), "latest_completed_step": max(complete), "advantage_scale": scale_rewards,
              "scope": "All completed native steps; only fully verified reward steps; training data, not fixed validation",
              "native": history.compact_native(history.summarize_native(native_rows)),
              "rewards": history.compact_rewards(history.summarize_rewards(list(good.values()))),
              "blocks": blocks, "excluded_reward_steps": excluded, "ignored_records": dict(ignored),
              "native_reward_mismatch_count": len(mismatches), "native_reward_first_mismatches": mismatches[:10],
              "all_wait_completed_groups": sum(g[0]["all_wait_completed_count"] == 8 for g in groups),
              "mixed_empty_groups": sum(0 < g[0]["empty_count"] < 8 for g in groups),
              "tasks_with_empty": len(empty_tasks), "top_empty_tasks": empty_tasks[:20],
              "tasks_seen_in_multiple_groups_with_all_wait": sum(t["all_wait_groups"] >= 2 for t in empty_tasks),
              "empty_candidate_audio_duration_seconds": distribution(row["duration_seconds"] for _, rows in groups for row in rows if row["empty"] is True),
              "all_wait_group_audio_duration_seconds": distribution(desc["duration_seconds"] for desc, _ in groups if desc["all_wait_completed_count"] == 8),
              "first_failure_kind": dict(failure_kinds), "first_failure_round": dict(failure_rounds),
              "observed_turn_statistics": dict(observed), "actions": dict(actions),
              "groups_with_all_8_quality_scores": sum(g[0]["quality_count"] == 8 for g in groups),
              "groups_quality_std_below_0_01": sum(g[0]["quality_std_population"] is not None and g[0]["quality_std_population"] < .01 for g in groups),
              "full_group_quality_std_distribution": distribution(desc["quality_std_population"] for desc, _ in groups),
              "full_group_gated_latency_std_distribution": distribution(desc["gated_latency_std_population"] for desc, _ in groups),
              "reward_config_evidence": {k: v for k, v in config_evidence.items() if k != "config"},
              "selected_groups": len(exports), "selected_candidates": len(exports) * 8,
              "selection_counts": dict(Counter(x["selection"] for x in exports)),
              "notes": ["No scores assigned to unscored empty/invalid candidates.",
                        "Legacy rewards.all_wait counts WAIT payloads; all_wait_completed_groups additionally requires valid finish reasons and completed input/EOF flags.",
                        "Small sentences, punctuation, COMET rank differences are audit cues, not proven errors.",
                        "Group advantages are offline estimates: center only for scale_rewards=none; sample std+1e-4 for group. Not captured GPU gradients.",
                        "Native rows lack session identity; explicit file and timestamp checks cannot fully prove native session.",
                        "Diagnostic sample is stratified; do not infer population rates from exported examples."]}
    out = Path(args.output_dir) if args.output_dir else run / "diagnostics" / ("reward_revision_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    with (out / "all_steps.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["native_step", *history.NATIVE, "verified_reward_step", "all_wait_groups", "all_wait_candidates"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in native_rows:
            s = history.step(row)
            value = {key: row.get(key) for key in history.NATIVE}
            value.update(native_step=s, verified_reward_step=s in good)
            if s in good:
                value.update(all_wait_groups=sum(all(r["observed"]["observed_all_wait"] and r["episode_complete"] is True for r in group) for group in good[s].groups.values()),
                             all_wait_candidates=sum(r["observed"]["observed_all_wait"] and r["episode_complete"] is True for group in good[s].groups.values() for r in group))
            writer.writerow(value)
    (out / "summary.json").write_text(js(report) + "\n", encoding="utf-8")
    (out / "reward_config_evidence.json").write_text(js(config_evidence) + "\n", encoding="utf-8")
    lines = ["奖励版本审查：全程统计与完整G8样例；未重新评分、未修改训练。"]
    lines += [key + ": " + js(value) for key, value in report.items() if key != "blocks"]
    lines += ["训练分段: " + js(block) for block in blocks]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "empty_tasks.jsonl").write_text("".join(js(v) + "\n" for v in empty_tasks), encoding="utf-8")
    (out / "group_statistics.jsonl").write_text("".join(js(desc) + "\n" for desc, _ in groups), encoding="utf-8")
    (out / "selected_groups.jsonl").write_text("".join(js(group) + "\n" for group in exports), encoding="utf-8")
    for offset in range(0, len(exports), 5):
        text = []
        for group in exports[offset:offset + 5]:
            desc = group["group"]
            text += ["\n音频组: " + js({"selection": group["selection"], **desc}),
                     "英文原文: " + js(group["task"].get("source_text")),
                     "机器参考: " + js(group["task"].get("reference_text"))]
            for rank, row in enumerate(sorted(group["candidates"], key=lambda r: (-r["reward"], r["components"]["group_candidate_index"])), 1):
                advantage = row["reward"] - desc["reward_mean"]
                if scale_rewards == "group":
                    advantage /= desc["reward_std_sample"] + 1e-4
                text += ["候选: " + js({"reward_rank": rank, "candidate_index": row["components"]["group_candidate_index"], "reward": row["reward"], "estimated_advantage": advantage}),
                         "奖励分项: " + js(row["components"]), "完整译文: " + js(row.get("translation")),
                         "轨迹状态: " + js({k: v for k, v in row["trajectory"].items() if k != "turns"})]
                text += ["逐轮输出: " + js({"audit_turn_number": i, **turn}) for i, turn in enumerate(row["trajectory"]["turns"], 1)]
        (out / f"review_part_{offset // 5 + 1:02d}.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out / "provenance.json").write_text(js({"launch": plan, "log_snapshots": inventory,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "exported_pointers": [item["pointer"] for _, _, rows in selected for item in rows]}) + "\n", encoding="utf-8")
    print(f"[collect] Done: {len(groups)} verified G8 groups; exported {len(exports) * 8} candidates.", flush=True)
    print("[collect] 先发 summary.txt，再发 review_part_*.txt；selected_groups.jsonl 留作奖励回放。", flush=True)
    print("[collect] OUTPUT: " + str(out.resolve()), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--native-log", required=True)
    p.add_argument("--launch-file", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--reward-config", help="Optional config path; otherwise read launch path and verify saved hash")
    p.add_argument("--groups", type=int, default=20)
    p.add_argument("--block-size", type=int, default=50)
    p.add_argument("--output-dir", help="New output directory; otherwise timestamped diagnostics directory")
    args = p.parse_args()
    if not 1 <= args.groups <= 100 or args.block_size < 1:
        p.error("--groups must be 1..100; --block-size must be positive")
    try:
        collect(args)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("[collect] ERROR: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
