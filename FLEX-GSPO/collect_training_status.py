#!/usr/bin/env python3
"""Collect a bounded, text-only snapshot of a running FLEX training job.

Standard library only. Reads complete JSONL lines up to a fixed file size;
never imports the trainer, opens WAVs, loads weights or calls scoring services.
Reward summaries describe local callbacks, not reconstructed global G8 groups.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time


DEFAULT_RUN = "/data/yb/FLEX/outputs/gspo_comet22_g8_b32_3000_001"
NATIVE_KEYS = ("loss", "kl", "reward", "reward_std", "frac_reward_zero_std", "grad_norm",
               "clip_ratio/region_mean", "completions/mean_length", "learning_rate",
               "train_speed(s/it)")
REWARD_KEYS = ("quality", "comet22_score", "latency_seconds", "length_penalty")
FLAG_KEYS = ("protocol_valid", "empty_translation", "quality_pass", "latency_pass",
             "feasible", "severe_repetition")
PERF_KEYS = ("generation_seconds", "weight_sync_seconds", "rollout_requests_seconds",
             "offload_to_cpu_seconds", "reload_to_gpu_seconds", "vllm_sleep_seconds")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def js(value):
    return json.dumps(safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def stats(values):
    vals = sorted(float(v) for v in values if finite(v))
    if not vals:
        return {"count": 0}
    return {"count": len(vals), "mean": sum(vals) / len(vals), "min": vals[0],
            "max": vals[-1], "p50": vals[(len(vals)-1)//2],
            "p90": vals[math.ceil(len(vals)*.9)-1]}


def read_tail(path, max_bytes, now):
    """Return complete records from a bounded snapshot; ignore appending writes."""
    path = Path(path)
    before = path.stat()
    start = max(0, before.st_size - max_bytes)
    meta = {"path": str(path), "bytes_at_start": before.st_size,
            "age_seconds": max(0, now-before.st_mtime), "tail_limited": start > 0,
            "bad_complete_lines": 0, "partial_tail_ignored": False}
    with path.open("rb") as handle:
        if start:
            handle.seek(start-1)
            boundary = handle.read(1) == b"\n"
        else:
            boundary = True
        handle.seek(start)
        data = handle.read(before.st_size-start)
    if not boundary:
        cut = data.find(b"\n")
        data = data[cut+1:] if cut >= 0 else b""
        meta["leading_partial_line_ignored"] = True
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        data = data[:cut+1] if cut >= 0 else b""
        meta["partial_tail_ignored"] = True
    rows = []
    for raw in data.splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(row, dict):
                raise ValueError("Expected object")
        except (ValueError, UnicodeError):
            meta["bad_complete_lines"] += 1
            continue
        rows.append(row)
    after = path.stat()
    meta.update(rows_read=len(rows), changed_during_read=(before.st_size, before.st_mtime_ns)
                != (after.st_size, after.st_mtime_ns))
    return rows, meta


def step_of(row):
    value = row.get("iteration", row.get("global_step"))
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*(?:/\s*\d+)?\s*", value)
        if match:
            return int(match[1])
    return None


def elapsed_seconds(value):
    if finite(value):
        return float(value)
    if not isinstance(value, str):
        return None
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", value)
    if parts:
        return sum(float(n)*{"d": 86400, "h": 3600, "m": 60, "s": 1}[unit] for n, unit in parts)
    parts = value.split(":")
    if 2 <= len(parts) <= 3:
        try:
            return sum(float(part)*60**i for i, part in enumerate(reversed(parts)))
        except ValueError:
            pass
    return None


def arg_value(plan, flag):
    argv = plan.get("argv", [])
    if not isinstance(argv, list):
        return None
    for index, item in enumerate(argv[:-1]):
        if item == flag:
            return argv[index+1]
    return None


def load_launch(root, warnings):
    files = sorted(root.glob("launch_*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not files:
        warnings.append("没有 launch 身份记录；无法确认多个 PID 是否来自同次启动。")
        return {}, None
    # A partially written newest launch is not replaced silently by an old run.
    path = files[0]
    try:
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(plan, dict):
            raise ValueError("Expected object")
        return plan, path
    except (OSError, ValueError) as exc:
        warnings.append("最新 launch 无法读取: " + type(exc).__name__)
        return {}, path


def native_window(rows):
    return {"rows": len(rows), "step_range": [step_of(rows[0]), step_of(rows[-1])] if rows else [],
            "metrics": {key: stats(r.get(key) for r in rows) for key in NATIVE_KEYS},
            "zero_grad_steps": sum(r.get("grad_norm") == 0 for r in rows),
            "nonfinite_values": {key: sum(isinstance(r.get(key), float) and not math.isfinite(r[key])
                                          for r in rows) for key in NATIVE_KEYS
                                 if any(isinstance(r.get(key), float) and not math.isfinite(r[key]) for r in rows)}}


def report_speed(rows, total_steps):
    rates = []
    for left, right in zip(rows, rows[1:]):
        a, b = elapsed_seconds(left.get("elapsed_time")), elapsed_seconds(right.get("elapsed_time"))
        sa, sb = step_of(left), step_of(right)
        if a is not None and b is not None and sa is not None and sb is not None and b > a and sb > sa:
            rates.append((b-a)/(sb-sa))
    summary = stats(rates)
    result = {"recent_seconds_per_step_from_elapsed_deltas": summary,
              "note": "train_speed(s/it) 可能是累计均值；此处优先用近期 elapsed_time 差分估计。"}
    if rates and rows and type(total_steps) is int and total_steps > 0:
        left = max(0, total_steps-(step_of(rows[-1]) or 0))
        result.update(remaining_steps=left, estimated_remaining_hours=left*summary["mean"]/3600)
    return result


def compact(row):
    tr = row.get("trajectory") or {}
    turns = tr.get("turns") or []
    return {"reward": row.get("reward"), "components": row.get("components") or {},
            "duration_seconds": row.get("duration_seconds"),
            "complete_input": tr.get("complete_input"), "episode_complete": tr.get("episode_complete"),
            "eof_completed": tr.get("eof_completed"),
            "all_wait": bool(turns) and all(isinstance(t, dict) and
                isinstance(t.get("response"), str) and t["response"].strip() == "<|WAIT|>" for t in turns),
            "actions": Counter("eof/" + action(t) if t.get("kind") == "eof" else action(t)
                               for t in turns if isinstance(t, dict))}


def action(turn):
    response = turn.get("response")
    if not isinstance(response, str):
        return "UNRECOGNIZED"
    match = re.match(r"^<\|(WAIT|FLEX|DIRECT)\|>", response.strip())
    return match[1] if match else "UNRECOGNIZED"


def summarize_rewards(groups):
    rows = [row for group in groups for row in group["compact"]]
    result = {"local_callbacks": len(groups), "candidates": len(rows),
              "step_range": [groups[0]["step"], groups[-1]["step"]] if groups else [],
              "reward": stats(row.get("reward") for row in rows),
              "tiers": dict(Counter(row["components"].get("tier", "missing") for row in rows)),
              "all_wait_candidates": sum(row["all_wait"] for row in rows)}
    for key in REWARD_KEYS:
        result[key] = stats(row["components"].get(key) for row in rows)
    for key in FLAG_KEYS:
        values = [row["components"].get(key) for row in rows]
        known = [v for v in values if type(v) is bool]
        result[key] = {"known": len(known), "true": sum(known),
                       "fraction": sum(known)/len(known) if known else None}
    for key in ("complete_input", "episode_complete", "eof_completed"):
        values = [row[key] for row in rows if type(row[key]) is bool]
        result[key] = {"known": len(values), "true": sum(values)}
    counts = sum((row["actions"] for row in rows), Counter())
    for prefix in ("", "eof/"):
        total = sum(counts[prefix+a] for a in ("WAIT", "FLEX", "DIRECT", "UNRECOGNIZED"))
        result["eof_actions" if prefix else "audio_actions"] = {
            "count": total, **{a: counts[prefix+a]/total if total else None
                             for a in ("WAIT", "FLEX", "DIRECT", "UNRECOGNIZED")}}
    result["duration_buckets"] = {}
    for name, low, high in (("under_3s", 0, 3), ("3_to_10s", 3, 10), ("10s_or_more", 10, math.inf)):
        bucket = [r for r in rows if finite(r["duration_seconds"]) and low <= r["duration_seconds"] < high]
        result["duration_buckets"][name] = {"candidates": len(bucket),
            "all_wait": sum(r["all_wait"] for r in bucket),
            "empty": sum(r["components"].get("empty_translation") is True for r in bucket)}
    result["diagnostics"] = {key: stats((r["components"].get("diagnostics") or {}).get(key) for r in rows)
        for key in ("first_output_seconds", "max_no_text_gap_seconds", "eof_text_fraction", "eof_generation_seconds")}
    return result


def reward_groups(rows, native_step, window):
    groups = []
    for row in rows:
        key = (row.get("session_id"), row.get("call"), row.get("global_step"), row.get("timestamp_unix"))
        if not groups or groups[-1]["key"] != key:
            groups.append({"key": key, "step": row.get("global_step"), "raw": []})
        groups[-1]["raw"].append(row)
    usable, partial = deque(maxlen=window*2), 0
    for index, group in enumerate(groups):
        batch = group["raw"]
        indices = [r.get("candidate_index_in_local_batch") for r in batch]
        contiguous = indices == list(range(len(batch)))
        step = group["step"]
        # A later callback proves closure; a completed trainer step also fences
        # its preceding reward callback. Final unfenced prefixes are excluded.
        closed = index < len(groups)-1 or (type(native_step) is int and type(step) is int and step < native_step)
        if not contiguous or not closed:
            partial += 1
            continue
        group["compact"] = [compact(row) for row in batch]
        usable.append(group)
    return list(usable), partial


def choose_examples(groups, limit):
    rows = [row for group in groups for row in group["raw"]]
    rows = [row for row in rows if finite(row.get("reward"))]
    if not rows or not limit:
        return []
    # Compare two candidates of the same latest task where possible.
    latest_task = rows[-1].get("task_id")
    same = [r for r in rows if r.get("task_id") == latest_task
            and r.get("call") == rows[-1].get("call")]
    selected = [("latest_task_lower_reward", min(same, key=lambda r: r["reward"])),
                ("latest_task_higher_reward", max(same, key=lambda r: r["reward"]))]
    bad = next((r for r in reversed(rows) if (r.get("components") or {}).get("tier") in ("invalid", "degenerate")), None)
    if bad is not None:
        selected.append(("recent_invalid_or_repetition", bad))
    selected.extend(("recent_low_reward", r) for r in sorted(rows, key=lambda r: r["reward"]))
    result, seen = [], set()
    for why, row in selected:
        identity = (row.get("session_id"), row.get("call"), row.get("candidate_index_in_local_batch"), row.get("task_id"))
        if identity in seen:
            continue
        result.append((why, row))
        seen.add(identity)
        if len(result) >= limit:
            break
    return result


def source_lookup(dataset, wanted, warnings):
    found = {}
    bad_lines = 0
    if not wanted:
        return found
    if not dataset or not Path(dataset).is_file():
        warnings.append("未找到数据清单，样例不附原文/参考；可显式传 --dataset。")
        return found
    with Path(dataset).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task = row.get("flex_task", row)
                task = json.loads(task) if isinstance(task, str) else task
                if isinstance(task, dict) and task.get("id") in wanted:
                    found[task["id"]] = task
            except (ValueError, TypeError, AttributeError):
                bad_lines += 1
                if bad_lines <= 3:
                    warnings.append("数据清单含无法解析的行: " + str(index))
            if len(found) == len(wanted):
                break
            if index % 50000 == 0:
                print(f"[collect] 已扫描文本清单 {index} 行；不读取音频", file=sys.stderr, flush=True)
    missing = wanted-set(found)
    if bad_lines:
        warnings.append("本次扫描的数据清单无法解析行数: " + str(bad_lines))
    if missing:
        warnings.append("样例未匹配原文/参考的 ID 数: " + str(len(missing)))
    return found


def clipped(value, size):
    text = str(value) if value is not None else "[missing]"
    return text if len(text) <= size else text[:size] + " [仅报告展示截断]"


def collect(args):
    root = Path(args.run_dir).resolve()
    if not root.is_dir():
        raise ValueError("训练输出目录不存在: " + str(root))
    now, warnings, lines = time.time(), [], []

    def emit(label, value):
        lines.append(label + ": " + (value if isinstance(value, str) else js(value)))

    plan, launch_path = load_launch(root, warnings)
    if launch_path and not plan:
        raise ValueError("最新启动记录尚未写完或已损坏，请稍后重试；不会把旧运行当作本次运行。")
    if args.session_id and plan.get("session_id") and args.session_id != plan["session_id"]:
        raise ValueError("--session-id 与最新 launch 不同；请使用对应启动的独立 run_dir，避免混配原生日志。")
    session = args.session_id or plan.get("session_id")
    run_id = plan.get("run_id")
    launch_time = launch_path.stat().st_mtime if launch_path else None
    emit("FLEX 训练诊断快照", {"created_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "run_dir": str(root), "session_id": session, "run_id": run_id,
        "scope": "只读日志；局部回调统计；不是固定验证集效果评估"})
    emit("启动记录", {"path": str(launch_path) if launch_path else None,
        "sampling": plan.get("sampling"), "configuration": {flag: arg_value(plan, "--"+flag) for flag in
        ("train_iters", "save_steps", "global_batch_size", "micro_batch_size", "num_generations",
         "lr", "beta", "vllm_gpu_memory_utilization", "vllm_max_num_seqs", "vllm_mm_processor_cache_gb")}})
    config = args.reward_config or (plan.get("environment") or {}).get("FLEX_REWARD_CONFIG")
    if config and Path(config).is_file():
        raw = Path(config).read_bytes()
        cfg = json.loads(raw.decode("utf-8-sig")).get("reward", {})
        expected = (plan.get("run_identity") or {}).get("reward_config_sha256")
        emit("奖励配置核对", {"matches_launch_hash": hashlib.sha256(raw).hexdigest() == expected if expected else None,
            "settings": {k: cfg.get(k) for k in ("version", "quality_provider", "quality_floor", "latency_metric",
                "latency_budget_seconds", "eof_delay_seconds", "reference_length_ratio", "latency_weight", "length_weight")}})
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            warnings.append("当前配置与启动时 hash 不同，上述配置不能当作实际生效配置。")
    natives = [Path(args.native_log)] if args.native_log else sorted(root.glob("*/logging.jsonl"),
        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    if not args.native_log and launch_time:
        natives = [p for p in natives if p.stat().st_mtime >= launch_time-2]
    if not args.native_log and len(natives) > 1:
        warnings.append("多个原生日志都在本次启动后更新，不能自动判定：" + ", ".join(str(p) for p in natives[:5])
                        + "；请传 --native-log 指定。本次不使用这些步数确认奖励回调完成。")
        natives = []
    native_rows, native_step = [], None
    if natives:
        native_rows, meta = read_tail(natives[0], args.tail_mib*1024*1024, now)
        native_rows = [r for r in native_rows if step_of(r) is not None]
        # Preserve only the latest occurrence per step in this native file.
        native_rows = list({step_of(r): r for r in native_rows}.values())
        emit("训练日志来源", meta)
        if native_rows:
            native_step = step_of(native_rows[-1])
            emit("前一窗口", native_window(native_rows[-2*args.window:-args.window]))
            emit("最近窗口", native_window(native_rows[-args.window:]))
            for row in native_rows[-args.latest:]:
                emit("最近训练步", {k: row[k] for k in ("iteration", *NATIVE_KEYS, "elapsed_time",
                    "remaining_time", "memory(GiB)") if k in row})
            total = arg_value(plan, "--train_iters")
            match = re.search(r"/(\d+)\s*$", str(native_rows[-1].get("iteration", "")))
            total = int(total) if str(total).isdigit() else int(match[1]) if match else None
            emit("速度与预计剩余", report_speed(native_rows[-args.window:], total))
            if meta["age_seconds"] > 600:
                warnings.append("原生训练日志已超过10分钟未更新；需结合阶段日志判断当前是否仍推进。")
    if native_step is None:
        warnings.append("未找到本次启动已完成的训练步；可能仍初始化或首批 rollout 尚未结束。")
    all_examples, selected_pids = [], {}
    for path in sorted((root/"rewards").glob("rewards-rank*-pid*.jsonl")):
        if launch_time and path.stat().st_mtime < launch_time-2:
            continue
        rows, meta = read_tail(path, args.tail_mib*1024*1024, now)
        if session:
            rows = [r for r in rows if r.get("session_id") == session and (not run_id or r.get("run_id") == run_id)]
        else:
            ids = {r.get("session_id") for r in rows}
            if len(ids) > 1:
                warnings.append("跳过混合 session 文件；请显式传 --session-id: " + path.name)
                continue
        if not rows:
            continue
        malformed = sum(not isinstance(r.get("components"), dict)
            or not isinstance(r.get("trajectory"), dict)
            or not isinstance(r["trajectory"].get("turns"), list)
            or any(not isinstance(t, dict) for t in r["trajectory"]["turns"]) for r in rows)
        if malformed:
            warnings.append(f"跳过结构异常的奖励行 {malformed} 条: {path.name}")
            rows = [r for r in rows if isinstance(r.get("components"), dict)
                and isinstance(r.get("trajectory"), dict)
                and isinstance(r["trajectory"].get("turns"), list)
                and all(isinstance(t, dict) for t in r["trajectory"]["turns"])]
        if not rows:
            continue
        match = re.fullmatch(r"rewards-rank(\d+)-pid(\d+)\.jsonl", path.name)
        if match:
            selected_pids[(match[1], match[2])] = True
        groups, partial = reward_groups(rows, native_step, args.window)
        emit("奖励日志来源", {**meta, "selected_rows": len(rows), "excluded_partial_or_unfenced_callbacks": partial,
            "window_note": "仅完整行、从候选索引0开始且有后继回调或已完成训练步佐证；局部统计不能当作全局G8"})
        emit("前一奖励窗口 " + path.name, summarize_rewards(groups[-2*args.window:-args.window]))
        emit("最近奖励窗口 " + path.name, summarize_rewards(groups[-args.window:]))
        # Prefer rank0 for the same-task pair, then add at most one anomaly.
        for why, row in choose_examples(groups[-args.window:], args.examples):
            all_examples.append((path.name, why, row))
    if not selected_pids:
        warnings.append("尚未找到匹配本次 session 的奖励记录；不会借用旧 PID 的奖励。")
        # Still expose first-rollout progress, explicitly without session proof.
        shown_ranks = set()
        candidates = sorted((root/"performance").glob("phases-rank*-pid*.jsonl"),
                            key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for phase_path in candidates:
            match = re.fullmatch(r"phases-rank(\d+)-pid(\d+)\.jsonl", phase_path.name)
            if not match or match[1] in shown_ranks or len(shown_ranks) >= 4:
                continue
            if launch_time and phase_path.stat().st_mtime < launch_time-2:
                continue
            phase_rows, phase_meta = read_tail(phase_path, min(args.tail_mib, 2)*1024*1024, now)
            if launch_time:
                phase_rows = [r for r in phase_rows if finite(r.get("timestamp_unix"))
                              and r["timestamp_unix"] >= launch_time-2]
            if phase_rows:
                emit("未通过session对齐的阶段线索", {"file": phase_path.name,
                    "file_age_seconds": phase_meta["age_seconds"], "last_event": phase_rows[-1],
                    "note": "仅按启动时间筛选，尚无奖励身份可佐证；不能认定为本次训练。"})
                shown_ranks.add(match[1])
    for (rank, pid) in selected_pids:
        perf = root/"performance"/f"performance-rank{rank}-pid{pid}.jsonl"
        if perf.is_file():
            rows, meta = read_tail(perf, min(args.tail_mib, 2)*1024*1024, now)
            if launch_time:
                rows = [r for r in rows if finite(r.get("timestamp_unix")) and r["timestamp_unix"] >= launch_time-2]
            recent = rows[-args.window:]
            emit("阶段耗时 rank"+rank, {"source": meta, "pid": pid, "rows": len(recent),
                "metrics": {k: stats(r.get(k) for r in recent) for k in PERF_KEYS},
                "last": recent[-1] if recent else None,
                "note": "性能行无session字段，按已匹配奖励的rank/PID和启动时间关联；宿主机计时未同步CUDA，阶段不一定可直接相加。"})
        phases = root/"performance"/f"phases-rank{rank}-pid{pid}.jsonl"
        if phases.is_file():
            rows, meta = read_tail(phases, min(args.tail_mib, 2)*1024*1024, now)
            if launch_time:
                rows = [r for r in rows if finite(r.get("timestamp_unix")) and r["timestamp_unix"] >= launch_time-2]
            if rows:
                emit("最新阶段 rank"+rank, {"file_age_seconds": meta["age_seconds"], "last_event": rows[-1]})
    # Bound display volume across all ranks, while retaining an anomaly if one exists.
    examples = all_examples[:args.examples]
    bad = next((e for e in all_examples if e[1] == "recent_invalid_or_repetition"), None)
    if bad is not None and args.examples and bad not in examples:
        examples[-1:] = [bad]
    dataset = args.dataset or arg_value(plan, "--dataset")
    sources = source_lookup(dataset, {e[2].get("task_id") for e in examples}, warnings)
    for filename, why, row in examples:
        task = sources.get(row.get("task_id"), {})
        emit("译文样例", {"selection": why, "file": filename, **{k: row.get(k) for k in
            ("task_id", "global_step", "call", "reward", "duration_seconds")}, "components": row.get("components")})
        emit("英文原文", clipped(task.get("source_text"), args.text_chars))
        emit("中文机器参考", clipped(task.get("reference_text"), args.text_chars))
        emit("模型完整译文", clipped(row.get("translation"), args.text_chars))
        tr = row.get("trajectory") or {}
        turns = tr.get("turns") or []
        emit("轨迹状态", {k: tr.get(k) for k in ("valid", "complete_input", "episode_complete", "eof_completed", "error")})
        # Always preserve EOF in a truncated display.
        shown = list(enumerate(turns, 1))
        if len(shown) > args.max_turns:
            shown = shown[:args.max_turns-1] + shown[-1:]
            emit("轨迹展示提示", f"总计 {len(turns)} 轮，仅显示前 {args.max_turns-1} 轮和最后一轮。")
        for index, turn in shown:
            emit("逐轮", {"turn": index, "kind": turn.get("kind", "audio"), "audio_end_seconds": turn.get("audio_end_seconds"),
                "response": clipped(turn.get("response"), 350), "finish_reason": turn.get("finish_reason")})
    if args.gpu_ids:
        try:
            result = subprocess.run(["nvidia-smi", "--id="+args.gpu_ids,
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8, check=True)
            emit("当前整卡采样（包含其他进程，不是峰值）", result.stdout.strip())
        except (OSError, subprocess.SubprocessError) as exc:
            warnings.append("GPU采样不可用: " + type(exc).__name__)
    emit("读取说明", "每个文件按打开时的大小读取，末尾半行忽略；各文件不是同一时刻的原子快照。")
    emit("判断边界", "此报告用于检查训练数值、奖励行为与耗时。不同窗口任务不同，均值涨跌不能证明翻译提升。")
    for warning in warnings:
        emit("需留意", warning)
    return "\n".join(lines)+"\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=DEFAULT_RUN)
    p.add_argument("--dataset", help="Prepared train.jsonl; otherwise read from launch argv")
    p.add_argument("--native-log", help="Optional exact logging.jsonl path")
    p.add_argument("--reward-config", help="Read settings and compare with launch hash; no service calls")
    p.add_argument("--session-id", help="Explicit reward session; normally use the latest launch")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--latest", type=int, default=5)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--text-chars", type=int, default=1600)
    p.add_argument("--max-turns", type=int, default=14)
    p.add_argument("--tail-mib", type=int, default=16, help="Maximum bytes read per JSONL file, in MiB")
    p.add_argument("--gpu-ids", help="Optional physical GPU IDs, e.g. 0,1,2,3")
    p.add_argument("--output", help="Also save the report to a NEW text file; refuses overwrite")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not (1 <= args.window <= 100 and 1 <= args.latest <= 30 and 0 <= args.examples <= 8
            and 100 <= args.text_chars <= 10000 and 2 <= args.max_turns <= 100 and 1 <= args.tail_mib <= 128):
        raise SystemExit("参数超出范围；window=1..100, latest=1..30, examples=0..8, text-chars=100..10000, max-turns=2..100, tail-mib=1..128")
    if args.gpu_ids and not re.fullmatch(r"\d+(,\d+)*", args.gpu_ids):
        raise SystemExit("gpu-ids 必须是逗号分隔的非负整数")
    try:
        if args.output and Path(args.output).exists():
            raise ValueError("报告文件已存在，请换一个新文件名: " + args.output)
        print("[collect] 正在读取日志尾部；不会加载模型或调用评分服务", file=sys.stderr, flush=True)
        report = collect(args)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x", encoding="utf-8") as handle:
                handle.write(report)
            print("[collect] 报告已保存: " + str(out), file=sys.stderr, flush=True)
        print(report, end="", flush=True)
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print("[collect] ERROR: " + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
