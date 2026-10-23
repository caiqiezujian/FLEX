#!/usr/bin/env python3
"""Export recent same-audio candidate cohorts for a human brevity audit.

Read-only training-log snapshot. No model loading, WAV reads or service calls.
Complete translations are preserved; WAIT count is descriptive, not a reward.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import sys
import time

from collect_training_status import (DEFAULT_RUN, action, arg_value, finite, js,
    load_launch, read_tail, reward_groups, source_lookup, stats)
from flex_gspo.protocol import collect_translation
from flex_gspo.qalco import text_units


def candidate_summary(row, source_file):
    tr = row["trajectory"]
    audio = [t for t in tr["turns"] if t.get("kind") != "eof"]
    eof = [t for t in tr["turns"] if t.get("kind") == "eof"]
    counts = Counter(action(t) for t in audio)
    protocol_error = None
    try:
        emitted = collect_translation(tr["turns"])
    except (ValueError, TypeError, KeyError) as exc:
        emitted, protocol_error = None, str(exc)
    stored = row.get("translation")
    mismatch = stored != emitted if protocol_error is None else False
    # Never silently repair an invalid or inconsistent stored translation.
    text = stored if isinstance(stored, str) else emitted
    units = len(text_units(text)) if isinstance(text, str) else None
    c = row["components"]
    events = [{"turn": index, "kind": turn.get("kind", "audio"),
        "audio_end_seconds": turn.get("audio_end_seconds"), "response": turn.get("response")}
        for index, turn in enumerate(tr["turns"], 1) if action(turn) != "WAIT"
        or (isinstance(turn.get("response"), str) and turn["response"].strip() != "<|WAIT|>")]
    return {"source_file": source_file, "task_id": row.get("task_id"),
        "global_step_zero_based": row.get("global_step"), "call": row.get("call"),
        "candidate_index_in_local_batch": row.get("candidate_index_in_local_batch"),
        "request_id": row.get("request_id"), "prompt_id": row.get("prompt_id"),
        "duration_seconds": row.get("duration_seconds"), "reward": row.get("reward"),
        "comet22_score": c.get("comet22_score"), "latency_seconds": c.get("latency_seconds"),
        "old_length_penalty": c.get("length_penalty"), "tier": c.get("tier"),
        "quality_pass": c.get("quality_pass"), "protocol_valid": c.get("protocol_valid"),
        "empty_translation": c.get("empty_translation"), "severe_repetition": c.get("severe_repetition"),
        "complete_input": tr.get("complete_input"), "episode_complete": tr.get("episode_complete"),
        "audio_turns": len(audio), "WAIT": counts["WAIT"], "FLEX": counts["FLEX"],
        "DIRECT": counts["DIRECT"], "unrecognized": counts["UNRECOGNIZED"],
        "wait_fraction": counts["WAIT"]/len(audio) if audio else None,
        "eof_actions": [action(t) for t in eof], "text_units": units,
        "protocol_error": protocol_error or tr.get("error"), "stored_translation_mismatch": mismatch,
        "translation": text, "output_events": events}


def pearson(pairs):
    pairs = [(float(x), float(y)) for x, y in pairs if finite(x) and finite(y)]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs)/len(pairs)
    my = sum(y for _, y in pairs)/len(pairs)
    sx = sum((x-mx)**2 for x, _ in pairs)
    sy = sum((y-my)**2 for _, y in pairs)
    return sum((x-mx)*(y-my) for x, y in pairs)/math.sqrt(sx*sy) if sx and sy else None


def select_cohorts(observed, max_tasks, max_candidates):
    chosen, task_ids, used = [], set(), 0
    for cohort in sorted(observed.values(), key=lambda g: g["step"], reverse=True):
        rows = cohort["candidates"]
        if len(rows) < 2 or cohort["task_id"] in task_ids:
            continue
        # Keep all observed candidates for a selected task/round, never cherry
        # pick the shortest or lowest-WAIT candidate to meet the export budget.
        if used+len(rows) > max_candidates:
            continue
        chosen.append(cohort)
        used += len(rows)
        task_ids.add(cohort["task_id"])
        if len(chosen) >= max_tasks or used >= max_candidates:
            break
    return chosen


def collect(args):
    root = Path(args.run_dir).resolve()
    if not root.is_dir():
        raise ValueError("训练目录不存在: " + str(root))
    warnings = []
    plan, launch = load_launch(root, warnings)
    session, run_id = plan.get("session_id"), plan.get("run_id")
    if not session or not run_id:
        raise ValueError("需要本轮 launch 文件中的 session_id/run_id，避免混入旧轨迹。")
    now = time.time()
    metadata, observed = [], {}
    excluded, structural, seen = 0, 0, set()
    for path in sorted((root/"rewards").glob("rewards-rank*-pid*.jsonl")):
        rows, meta = read_tail(path, args.tail_mib*1024*1024, now)
        rows = [r for r in rows if r.get("session_id") == session and r.get("run_id") == run_id]
        if not rows:
            continue
        good = []
        for row in rows:
            tr = row.get("trajectory")
            if (type(row.get("global_step")) is not int or not isinstance(row.get("task_id"), str)
                    or not isinstance(row.get("components"), dict) or not isinstance(tr, dict)
                    or not isinstance(tr.get("turns"), list)
                    or any(not isinstance(t, dict) for t in tr["turns"])):
                structural += 1
            else:
                good.append(row)
        groups, dropped = reward_groups(good, native_step=None, window=args.scan_callbacks)
        excluded += dropped
        meta.update(selected_session_rows=len(good), local_callbacks_kept=len(groups))
        metadata.append(meta)
        for group in groups:
            for row in group["raw"]:
                identity = (path.name, row.get("call"), row.get("timestamp_unix"),
                            row.get("candidate_index_in_local_batch"))
                if identity in seen:
                    continue
                seen.add(identity)
                prompt = row.get("prompt_id")
                # Without prompt IDs these are same-task/same-step observed
                # cohorts, NOT certified GSPO group reconstruction.
                key = (row["global_step"], row["task_id"], prompt if isinstance(prompt, str) else None)
                cohort = observed.setdefault(key, {"step": row["global_step"], "task_id": row["task_id"],
                    "prompt_id": key[2], "candidates": []})
                cohort["candidates"].append(candidate_summary(row, path.name))
    selected = select_cohorts(observed, args.max_tasks, args.max_candidates)
    count = sum(len(g["candidates"]) for g in selected)
    if not count:
        raise ValueError("尚无同音频的已闭合多候选回调；请稍后重试或增加 --tail-mib / --scan-callbacks。")
    dataset = args.dataset or arg_value(plan, "--dataset")
    sources = source_lookup(dataset, {g["task_id"] for g in selected}, warnings)
    if len(sources) != len(selected):
        raise ValueError("部分样例未找到原文/参考，请检查 --dataset；不导出缺少比较依据的报告。")
    lines = []

    def emit(label, value):
        lines.append(label+": "+(value if isinstance(value, str) else js(value)))

    emit("FLEX 简洁性审查数据", {"created_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "run_dir": str(root), "session_id": session, "run_id": run_id,
        "task_cohorts": len(selected), "candidates": count,
        "requested_max_candidates": args.max_candidates,
        "selection": "最近若干不同音频；每组保留全部观测候选；不是随机样本或独立验证集",
        "grouping": "同task/同零基global_step；有prompt_id则再分组，无prompt_id不声称精确全局G8",
        "excluded_partial_callbacks": excluded, "malformed_rows_skipped": structural,
        "expected_group_size_from_launch": (plan.get("sampling") or {}).get("group_size")})
    emit("长度口径", "与训练text_units相同：汉字/假名逐字、拉丁词和数字串；不计标点。不是token数或语义信息量。")
    emit("审查目标", "检查少WAIT是否伴随更长、更松散的表达。WAIT、字数、相关系数都不直接代表好坏；先核对漏译和内容完整性。")
    for meta in metadata:
        emit("日志读取", meta)
    total_valid, all_rows = 0, []
    for index, cohort in enumerate(selected, 1):
        task = sources[cohort["task_id"]]
        reference = task.get("reference_text")
        if not isinstance(reference, str) or not reference.strip() or not isinstance(task.get("source_text"), str):
            raise ValueError("原文或参考缺失: "+cohort["task_id"])
        ref_units = len(text_units(reference))
        candidates = sorted(cohort["candidates"], key=lambda r: (
            r["WAIT"], r["source_file"], r["candidate_index_in_local_batch"] or 0))
        usable = [r for r in candidates if r["protocol_valid"] is True and r["complete_input"] is True
            and r["episode_complete"] is True and not r["stored_translation_mismatch"]
            and r["protocol_error"] is None and r["text_units"] is not None and r["text_units"] > 0]
        total_valid += len(usable)
        all_rows.extend(candidates)
        emit("音频组", {"group": index, "task_id": cohort["task_id"], "step_zero_based": cohort["step"],
            "prompt_id": cohort["prompt_id"], "observed_candidates": len(candidates),
            "reference_units": ref_units, "old_1_2_threshold": 1.2*ref_units,
            "nonempty_protocol_valid_complete": len(usable),
            "WAIT_vs_length_pearson_descriptive_only": pearson([(r["WAIT"], r["text_units"]) for r in usable]),
            "length_stats": stats(r["text_units"] for r in usable),
            "note": "按WAIT数升序展示，无质量或长度优选筛选。负相关只表示少WAIT常伴随更长，不能证明因果或冗余。"})
        emit("英文原文", task["source_text"])
        emit("中文机器参考", reference)
        for ci, row in enumerate(candidates, 1):
            emit("候选", {"candidate_in_report": ci, **{k: v for k, v in row.items()
                if k not in ("translation", "output_events")},
                "length_to_reference": row["text_units"]/ref_units if row["text_units"] is not None and ref_units else None})
            emit("完整译文", row["translation"] if row["translation"] is not None else "[协议错误，无法可靠拼接；见输出事件]")
            if args.include_increments:
                emit("逐次新增译文（无正常WAIT空轮）", row["output_events"])
    emit("导出汇总", {"candidates": count, "audio_groups": len(selected),
        "nonempty_protocol_valid_complete": total_valid, "tiers": dict(Counter(r["tier"] for r in all_rows)),
        "stored_translation_mismatches": sum(r["stored_translation_mismatch"] for r in all_rows),
        "note": "没有跨不同音频计算字数/WAIT总相关；不同音频时长和语速不可混为同一条件。"})
    if count < args.max_candidates:
        warnings.append("实际候选少于上限；为保留整组和不同音频，不拆组凑数。可增加 --max-tasks 或 --scan-callbacks。")
    for message in warnings:
        emit("需留意", message)
    return "\n".join(lines)+"\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=DEFAULT_RUN)
    p.add_argument("--dataset")
    p.add_argument("--max-candidates", type=int, default=160)
    p.add_argument("--max-tasks", type=int, default=25)
    p.add_argument("--scan-callbacks", type=int, default=60)
    p.add_argument("--tail-mib", type=int, default=32)
    p.add_argument("--include-increments", action="store_true", help="Include every non-WAIT increment without clipping")
    p.add_argument("--output", required=True, help="New UTF-8 text report; refuses overwrite")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not (2 <= args.max_candidates <= 1000 and 1 <= args.max_tasks <= 200
            and 1 <= args.scan_callbacks <= 500 and 1 <= args.tail_mib <= 128):
        raise SystemExit("参数范围: max-candidates2..1000, max-tasks1..200, scan-callbacks1..500, tail-mib1..128")
    try:
        output = Path(args.output)
        if output.exists():
            raise ValueError("报告已存在，请改用新文件名: "+str(output))
        print("[brevity] 读取本轮轨迹，不加载模型、不调用评分服务", file=sys.stderr, flush=True)
        report = collect(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(report)
        print("[brevity] 报告已保存: "+str(output), file=sys.stderr, flush=True)
        print(report, end="", flush=True)
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print("[brevity] ERROR: "+str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
