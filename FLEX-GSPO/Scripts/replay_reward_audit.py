#!/usr/bin/env python3
"""Offline v3 reward replay from ordered collect_brevity_samples TXT exports.

No HTTP, audio reads, model loading, training or log mutation. COMET22 scores
are frozen observations. Only complete, scored, protocol-valid local G8 groups
can be replayed by this deliberately bounded audit tool. Unsupported/incomplete
groups are reported, never reduced to G7 or assigned fabricated quality scores.
Exported callbacks cannot certify the original distributed GSPO group boundary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__:
    from ._bootstrap import PROJECT_ROOT
else:
    from _bootstrap import PROJECT_ROOT

from flex_gspo.degeneration_v3 import degeneration_report_v3
from flex_gspo.gated_reward import combine_v3, finalize_group, validate_v3_settings
from flex_gspo.protocol import collect_translation, parse_response
from flex_gspo.qalco import emission_schedule, latency_statistics, text_units, validate_task
from flex_gspo.rewards import RewardError
from flex_gspo.trajectory import validate_turn_schedule


def number(value, name, minimum=None, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or minimum is not None and value < minimum
            or maximum is not None and value > maximum):
        raise ValueError(f"{name}: missing or invalid finite number")
    return float(value)


def read_exports(paths):
    """Files must be consecutive fragments; lines/JSON objects must be intact."""
    groups, current, candidate, reading_text = [], None, None, False
    for path in paths:
        with Path(path).open(encoding="utf-8-sig") as handle:
            for line_no, raw in enumerate(handle, 1):
                line = raw.rstrip("\r\n")
                if line.startswith("音频组: "):
                    current = json.loads(line.split(": ", 1)[1])
                    current["candidates"] = []
                    groups.append(current)
                    candidate, reading_text = None, False
                elif line.startswith(("英文原文: ", "中文机器参考: ")):
                    if current is None:
                        raise ValueError("Missing group header before source/reference")
                    key = "source_text" if line.startswith("英文原文: ") else "reference_text"
                    current[key] = line.split(": ", 1)[1]
                elif line.startswith("候选: "):
                    if current is None:
                        raise ValueError("Missing group header; supply preceding TXT fragment first")
                    candidate = json.loads(line.split(": ", 1)[1])
                    candidate.update(turns=[], export_location=f"{Path(path).resolve()}:{line_no}")
                    current["candidates"].append(candidate)
                    reading_text = False
                elif line.startswith("完整译文: "):
                    if candidate is None:
                        raise ValueError("Missing candidate header; supply preceding TXT fragment first")
                    candidate["translation"] = line.split(": ", 1)[1]
                    reading_text = True
                elif line.startswith("逐轮原始输出"):
                    reading_text = False
                elif line.startswith("逐次新增译文"):
                    raise ValueError("This older export omits WAIT turns; export all turns first")
                elif reading_text:
                    candidate["translation"] += "\n" + line
                elif line.startswith("{"):
                    turn = json.loads(line)
                    if "response" in turn:
                        if candidate is None:
                            raise ValueError("Turn before candidate header")
                        candidate["turns"].append(turn)
    if not groups:
        raise ValueError("No exported audio groups found")
    return groups


def replay_group(group, settings):
    rows = group["candidates"]
    if len(rows) != 8 or group.get("observed_candidates") != 8:
        raise ValueError(f"Expected all 8 candidates, received {len(rows)}")
    indices = [r.get("candidate_index_in_local_batch") for r in rows]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(8)):
        raise ValueError("Candidate indices must be exactly 0..7")
    requests = [r.get("request_id") for r in rows]
    if any(not isinstance(i, str) or not i for i in requests) or len(set(requests)) != 8:
        raise ValueError("Expected eight distinct nonempty request IDs")
    callback = rows[0].get("callback_identity")
    if not isinstance(callback, list) or len(callback) != 7:
        raise ValueError("Missing callback identity from full-turn exporter")
    if any(r.get("callback_identity") != callback for r in rows):
        raise ValueError("Mixed callbacks; cannot construct G8")
    if len({r.get("prompt_id") for r in rows}) != 1:
        raise ValueError("Mixed prompt IDs")
    duration = number(rows[0].get("duration_seconds"), "duration", minimum=0)
    audio_count = rows[0].get("audio_turns")
    if type(audio_count) is not int or audio_count < 1:
        raise ValueError("Missing audio turn count")
    for row in rows:
        if (row.get("task_id") != group.get("task_id")
                or row.get("global_step_zero_based") != group.get("step_zero_based")
                or row.get("duration_seconds") != duration or row.get("audio_turns") != audio_count):
            raise ValueError("Mixed task/step/duration/audio count")
        expected = [row.get("source_file"), callback[1], callback[2],
                    row.get("global_step_zero_based"), row.get("call"), callback[5], row.get("task_id")]
        if callback != expected or not callback[1] or not callback[2]:
            raise ValueError("Callback identity disagrees with candidate metadata")
        if len(row.get("turns", [])) != audio_count + 1:
            raise ValueError(f"Candidate {row.get('candidate_in_report')}: incomplete raw turns")
        if [t.get("turn") for t in row["turns"]] != list(range(1, audio_count + 2)):
            raise ValueError("Missing/duplicate/out-of-order exported turn ordinals")
        if any(row.get(flag) is not True for flag in ("protocol_valid", "complete_input", "episode_complete")):
            raise ValueError("This audit requires complete protocol-valid scored groups")
        if row.get("protocol_error") or row.get("stored_translation_mismatch"):
            raise ValueError("Export reports protocol or translation consistency error")
        number(row.get("comet22_score"), "COMET22 score", minimum=0, maximum=1)
        number(row.get("reward"), "observed old reward")

    task = {"id": group["task_id"], "source_lang": "en", "target_lang": "zh",
            "source_text": group.get("source_text"), "reference_text": group.get("reference_text"),
            "duration_seconds": duration,
            "chunks": [{"end_seconds": t.get("audio_end_seconds")}
                       for t in rows[0]["turns"][:audio_count]]}
    validate_task(task)
    items, details = [], []
    for row in rows:
        turns = row["turns"]
        schedule = validate_turn_schedule(task, turns)
        if not schedule["episode_complete"] or any(t.get("finish_reason") not in ("stop", "eos") for t in turns):
            raise ValueError("Incomplete episode or abnormal generation termination")
        if row.get("eof_actions") != [parse_response(turns[-1]["response"])[0]]:
            raise ValueError("EOF action summary mismatch")
        text = collect_translation(turns)
        if text != row.get("translation"):
            raise ValueError("Stored translation differs from raw append-only output")
        units = len(text_units(text))
        if units < 1 or units != row.get("text_units"):
            raise ValueError("Missing/incorrect measurable text unit count")
        _, times = emission_schedule(task, turns, settings["eof_delay_seconds"])
        statistics = latency_statistics(duration, times, len(text_units(task["reference_text"])),
                                         settings["eof_delay_seconds"])
        latency = statistics["mean_lag_seconds"]
        logged_latency = number(row.get("latency_seconds"), "logged latency", minimum=0)
        if not math.isclose(latency, logged_latency, rel_tol=1e-7, abs_tol=1e-6):
            raise ValueError(f"Recomputed latency {latency} differs from logged {logged_latency}")
        guard = degeneration_report_v3(task["source_text"], task["reference_text"], turns)
        quality, repeat = row["comet22_score"], guard["repetition_penalty"]
        eligible = not guard["severe_repetition"]
        provisional = combine_v3(quality, latency, 0.0, repeat, settings)["reward"] if eligible else -4.0
        items.append(dict(eligible=eligible, quality=quality, latency=latency,
                          units=units, repetition=repeat, reward=provisional))
        details.append({**row, "old_reward_observed": row["reward"],
                        "latency_recomputed": latency, "v3_repetition_diagnostics": guard})
    scored = finalize_group(items, settings)
    for row, score in zip(details, scored):
        row["v3"] = score
        row["old_rank"] = 1 + sum(r["old_reward_observed"] > row["old_reward_observed"] for r in details)
        row["v3_rank"] = 1 + sum(r["reward"] > score["reward"] for r in scored)
    old_top = [r["candidate_in_report"] for r in details if r["old_rank"] == 1]
    new_top = [r["candidate_in_report"] for r in details if r["v3_rank"] == 1]
    return {"group": group["group"], "task_id": group["task_id"],
            "step_zero_based": group["step_zero_based"], "source_text": task["source_text"],
            "reference_text": task["reference_text"], "old_top": old_top, "v3_top": new_top,
            "top_changed": set(old_top) != set(new_top), "candidates": details}


def report_text(result):
    lines = ["旧轨迹的 v3 奖励离线重算（不是 v3 训练后的输出）",
        f"已重算 {len(result['groups'])} 组 / {8 * len(result['groups'])} 条；整组跳过 {len(result['skipped_groups'])} 组。",
        "COMET22 冻结为日志分数；时延由原始轮次重算并核对；未调用服务、读取音频或运行模型。",
        "仅检查完整、已评分、协议有效的局部 G8；缺少全局 prompt/task 哈希，不认证分布式训练分组。",
        "两版奖励尺度不同，不能因 v3 奖励更大而声称质量提高；排名也不是人工质量标签。", ""]
    for group in result["groups"]:
        lines += [f"组 {group['group']} | {group['task_id']}",
            "英文: " + group["source_text"], "机器参考: " + group["reference_text"],
            f"最高奖励候选：旧版 {group['old_top']} -> v3 {group['v3_top']}",
            "候选 | Q | 时延秒 | 长度 | 旧奖励 | v3奖励 | 旧排名->新排名 | v3简洁扣分 | v3重复扣分"]
        for row in group["candidates"]:
            s = row["v3"]
            lines.append(f"{row['candidate_in_report']} | {s['quality']:.6f} | {s['latency']:.3f} | {s['units']} | "
                         f"{row['old_reward_observed']:.6f} | {s['reward']:.6f} | {row['old_rank']}->{row['v3_rank']} | "
                         f"{s['weighted_brevity_penalty']:.6f} | {s['weighted_repetition_penalty']:.6f}")
        s = group["candidates"][0]["v3"]
        lines.append(f"长度基准候选数={s['anchor_pool_size']}；长度基准={s['group_length_anchor']}")
        for row in group["candidates"]:
            lines += [f"候选 {row['candidate_in_report']} 完整译文: {row['translation']}"]
        lines.append("")
    for group in result["skipped_groups"]:
        lines.append("整组跳过: " + json.dumps(group, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-txt", nargs="+", required=True, help="TXT fragments, in original order")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "qalco_v3.json"))
    parser.add_argument("--output-dir", required=True, help="New directory; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        output = Path(args.output_dir)
        if output.exists():
            raise ValueError("Output directory already exists; use a new directory")
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        settings = validate_v3_settings(config["reward"])
        if (settings.get("version") != 3 or settings.get("quality_provider") != "comet22"
                or settings.get("latency_metric") != "mean_lag"):
            raise ValueError("This audit supports only v3 / COMET22 / mean_lag")
        number(settings.get("eof_delay_seconds"), "eof_delay_seconds", minimum=0)
        groups = read_exports(args.input_txt)
        result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "frozen_quality_local_G8_counterfactual_not_new_rollouts",
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "reward_settings": {k: v for k, v in settings.items() if k != "comet22"},
            "inputs": [{"path": str(Path(p).resolve()), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                       for p in args.input_txt], "groups": [], "skipped_groups": []}
        for group in groups:
            try:
                replayed = replay_group(group, settings)
            except (ValueError, KeyError, TypeError, RewardError) as exc:
                result["skipped_groups"].append({"group": group.get("group"),
                    "task_id": group.get("task_id"), "reason": str(exc)})
                print(f"[replay] SKIP group={group.get('group')}: {exc}", flush=True)
            else:
                result["groups"].append(replayed)
                print(f"[replay] OK group={group['group']} old_top={replayed['old_top']} v3_top={replayed['v3_top']}", flush=True)
        result["code_sha256"] = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), *[PROJECT_ROOT / "flex_gspo" / name
                for name in ("gated_reward.py", "qalco.py", "degeneration_v3.py", "degeneration.py", "protocol.py", "trajectory.py")]]}
        output.mkdir(parents=True, exist_ok=False)
        (output / "replay.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / "comparison.txt").write_text(report_text(result), encoding="utf-8")
        print(f"[replay] Saved: {output.resolve() / 'comparison.txt'}", flush=True)
        return 0 if result["groups"] else 2
    except (OSError, ValueError, KeyError, TypeError, RewardError) as exc:
        print(f"[replay] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
