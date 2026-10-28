#!/usr/bin/env python3
"""Replay verified complete G8 exports with frozen COMET22 and QALCO v4.

No HTTP, audio decoding, model inference, or training. Input is the unedited
selected_groups.jsonl produced by collect_reward_revision.py. Scores are frozen
observations, not semantic labels. This is a reward counterfactual, not a new
policy evaluation. Refuse partial groups, mixed runs, and fabricated scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

if __package__:
    from ._bootstrap import PROJECT_ROOT
else:
    from _bootstrap import PROJECT_ROOT

from flex_gspo.normalized_reward import finalize_group
from flex_gspo.protocol import collect_translation, parse_response
from flex_gspo.qalco import QalcoScorer, validate_settings, validate_task
from flex_gspo.rewards import RewardError


def digest(value, compact=False):
    options = {"separators": (",", ":")} if compact else {}
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, **options)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def finite(value, name, low=None, high=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or low is not None and value < low or high is not None and value > high):
        raise ValueError(f"{name} must be a finite number in its permitted range")
    return float(value)


def nonempty(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {name}")
    return value


def advantages(rewards, version):
    mean = statistics.fmean(rewards)
    scale = statistics.stdev(rewards) + 1e-4 if version == 3 else 1.0
    return [(r - mean) / scale for r in rewards]


class FrozenQuality:
    """One candidate's observed score; never borrows another row's score."""
    def __init__(self, task, text, score):
        self.task, self.text, self.score = task, text, score

    def score_batch(self, source, hypotheses, reference):
        if source != self.task["source_text"] or reference != self.task["reference_text"]:
            raise ValueError("Scoring text differs from hashed source/reference")
        if hypotheses != [self.text] or self.score is None:
            raise ValueError("Missing frozen COMET22 score for a recomputed eligible candidate")
        return [self.score]


def replay_group(export, config, source_location=None):
    if not isinstance(export, dict) or not isinstance(export.get("group"), dict):
        raise ValueError("Expected collect_reward_revision export object")
    task, description, rows = export.get("task"), export["group"], export.get("candidates")
    if not isinstance(task, dict):
        raise ValueError("Missing original task; a source/reference-only export is insufficient")
    validate_task(task)
    if not isinstance(rows, list) or len(rows) != 8 or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Exactly eight complete candidate records are required")
    components = [r.get("components") for r in rows]
    if any(not isinstance(c, dict) for c in components):
        raise ValueError("Missing candidate components")
    indices = [c.get("group_candidate_index") for c in components]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(8)):
        raise ValueError("Candidate indices must be unique and cover 0..7")
    rows = sorted(rows, key=lambda r: r["components"]["group_candidate_index"])
    task_hash = digest(task)
    group_id = nonempty(description.get("global_group_id"), "global_group_id")
    session = nonempty(rows[0].get("session_id"), "session_id")
    run = nonempty(rows[0].get("run_id"), "run_id")
    step = rows[0].get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("Missing nonnegative global_step")
    group_index = rows[0]["components"].get("global_group_index")
    if type(group_index) is not int or group_index < 0:
        raise ValueError("Missing global_group_index for group hash verification")
    prompt = rows[0]["components"].get("group_prompt_id")
    if not isinstance(prompt, str):
        raise ValueError("Missing group_prompt_id")
    if (description.get("task_id") != task.get("id") or description.get("task_hash") != task_hash
            or description.get("native_step") != step + 1):
        raise ValueError("Group description disagrees with task hash or completed step")
    requests, versions = [], []
    for row in rows:
        c = row["components"]
        version = row.get("reward_version", c.get("reward_version"))
        if (type(version) is not int or version not in (3, 4)
                or c.get("reward_version", version) != version):
            raise ValueError("Only identified v3/v4 source rewards are supported")
        versions.append(version)
        if (row.get("session_id") != session or row.get("run_id") != run
                or row.get("global_step") != step or row.get("task_hash") != task_hash
                or row.get("task_id") != task.get("id")):
            raise ValueError("Mixed run/session/step/task hash within G8")
        if (c.get("global_group_id") != group_id or c.get("group_size") != 8
                or c.get("group_finalized") is not True or c.get("global_group_index") != group_index
                or c.get("group_prompt_id") != prompt):
            raise ValueError("Mixed/unfinalized global group metadata")
        request = nonempty(c.get("group_request_id"), "group_request_id")
        if row.get("request_id") not in (None, request) or row.get("prompt_id") not in (None, prompt):
            raise ValueError("Request/prompt identity differs from finalized group")
        requests.append(request)
        finite(row.get("reward"), "old reward")
        if not math.isclose(finite(row.get("duration_seconds"), "duration", low=0),
                            task["duration_seconds"], rel_tol=0, abs_tol=1e-6):
            raise ValueError("Candidate duration differs from original task")
    if len(set(requests)) != 8 or len(set(versions)) != 1:
        raise ValueError("Duplicate request IDs or mixed reward versions")
    expected_gid = digest({"offset": group_index * 8, "task_hash": task_hash,
                           "prompt_id": prompt, "request_ids": requests}, compact=True)
    if group_id != expected_gid:
        raise ValueError("Global group hash does not match ordered request identities")
    settings = validate_settings(config)
    if settings["version"] != 4:
        raise ValueError("Target configuration must be QALCO v4")
    items, details, score_by_text = [], [], {}
    for row in rows:
        c, trajectory = row["components"], row.get("trajectory")
        if (not isinstance(trajectory, dict) or not isinstance(trajectory.get("turns"), list)
                or type(trajectory.get("valid")) is not bool):
            raise ValueError("Missing original raw trajectory")
        try:
            text = collect_translation(trajectory["turns"])
        except (ValueError, TypeError, KeyError):
            text = None
        if text != row.get("translation"):
            raise ValueError("Stored translation differs from append-only raw turns")
        q = c.get("quality")
        if q is not None:
            q = finite(q, "frozen COMET22", low=0, high=1)
            if c.get("quality_metric") != "comet22_reference":
                raise ValueError("Frozen quality is not identified as COMET22 reference scoring")
            if c.get("comet22_score") is not None and not math.isclose(
                    q, finite(c["comet22_score"], "comet22_score", low=0, high=1), abs_tol=1e-7):
                raise ValueError("Logged quality and COMET22 scores disagree")
            if text in score_by_text and not math.isclose(q, score_by_text[text], abs_tol=1e-6):
                raise ValueError("Identical translations have inconsistent frozen scores")
            score_by_text[text] = q
        scorer = QalcoScorer(config, quality_model=FrozenQuality(task, text, q))
        recomputed = scorer.score(task, trajectory)
        rc = recomputed["components"]
        if type(c.get("eligible")) is not bool or c["eligible"] != rc["eligible"]:
            raise ValueError("Logged eligibility disagrees with replayed protocol/completeness/repetition")
        # Missing Q remains missing; no rank-local candidate or model fallback.
        if rc["eligible"] and q is None:
            raise ValueError("Eligible candidate lacks a frozen quality score")
        items.append({"eligible": rc["eligible"], "quality": rc.get("quality"),
                      "latency": rc.get("latency_seconds"), "units": rc["emitted_units"],
                      "repetition": rc["repetition_penalty"], "reward": recomputed["reward"]})
        actions = []
        for turn in trajectory["turns"]:
            try:
                actions.append(parse_response(turn["response"])[0])
            except (ValueError, TypeError, KeyError):
                actions.append("INVALID")
        details.append({"index": c["group_candidate_index"], "request_id": row["request_id"] if row.get("request_id") else c["group_request_id"],
                        "translation": text, "quality": q, "old_reward": row["reward"],
                        "old_components": c, "recomputed_components": rc,
                        "empty_translation": text == "",
                        "observed_all_wait": bool(actions) and all(a == "WAIT" for a in actions),
                        "all_wait": bool(actions) and all(a == "WAIT" for a in actions)
                            and trajectory.get("episode_complete") is True
                            and trajectory.get("eof_completed") is True,
                        "turn_count": len(actions), "action_counts": {a: actions.count(a) for a in sorted(set(actions))},
                        "original_trajectory_pointer": {"input": source_location, "candidate_index": c["group_candidate_index"]}})
    new = finalize_group(items, settings)
    old_adv = advantages([r["reward"] for r in rows], versions[0])
    new_adv = advantages([r["reward"] for r in new], 4)
    for index, (detail, score) in enumerate(zip(details, new)):
        detail.update(old_advantage=old_adv[index], new_reward=score["reward"],
                      new_advantage=new_adv[index], v4=score,
                      old_rank=1 + sum(r["reward"] > detail["old_reward"] for r in rows),
                      new_rank=1 + sum(r["reward"] > score["reward"] for r in new))
    old_top = [r["index"] for r in details if r["old_rank"] == 1]
    new_top = [r["index"] for r in details if r["new_rank"] == 1]
    return {"session_id": session, "run_id": run, "source_reward_version": versions[0],
            "global_group_id": group_id, "global_step": step, "task_id": task["id"], "task_hash": task_hash,
            "source_text": task["source_text"], "reference_text": task["reference_text"],
            "old_top": old_top, "new_top": new_top, "top_changed": old_top != new_top,
            "candidates": details}


def summarize(groups, settings):
    candidates = [r for g in groups for r in g["candidates"]]
    low = [r for r in candidates if r["quality"] is not None and r["quality"] < settings["quality_floor"]]
    return {"groups": len(groups), "candidates": len(candidates),
            "top_changed_groups": sum(g["top_changed"] for g in groups),
            "any_rank_changed_groups": sum(any(r["old_rank"] != r["new_rank"] for r in g["candidates"]) for g in groups),
            "empty_candidates": sum(r["empty_translation"] for r in candidates),
            "all_wait_candidates": sum(r["all_wait"] for r in candidates),
            "all_invalid_groups": sum(all(not r["v4"]["eligible"] for r in g["candidates"]) for g in groups),
            "low_quality_candidates": len(low),
            "low_quality_positive_old_advantage": sum(r["old_advantage"] > 1e-12 for r in low),
            "low_quality_positive_new_advantage": sum(r["new_advantage"] > 1e-12 for r in low),
            "old_advantage_abs_max": max(abs(r["old_advantage"]) for r in candidates),
            "new_advantage_abs_max": max(abs(r["new_advantage"]) for r in candidates)}


def report_text(summary):
    lines = ["QALCO v4 冻结轨迹奖励回放", "",
             "未调用任何模型服务；未读取音频；没有生成新轨迹或执行训练。",
             "原始逐轮输出保存在输入 selected_groups.jsonl；groups.jsonl 提供文件、行号及候选序号指针。",
             "完整 G8 身份/hash 已校验；COMET22 分数冻结；协议、完整性、时延、重复由当前代码重算。",
             "旧 v3 优势 = (R-组均值)/(组样本标准差+1e-4)；v4 优势 = R-组均值。",
             "这是诊断性选择的组，不能代表全量分布；新旧奖励绝对大小不可用来判断翻译是否提升。",
             "排名变化不等于语义修复。COMET 漏判的否定范围、无依据内容仍需固定对照与人工核验。", ""]
    lines.extend(f"{key}: {value}" for key, value in summary["statistics"].items())
    lines += ["", "只需先发送本 summary.txt；候选全文/贡献见 groups.jsonl。"]
    return "\n".join(lines) + "\n"


def run(args):
    source, target, config_path = Path(args.input_jsonl), Path(args.output_dir), Path(args.config)
    if target.exists():
        raise ValueError("Output directory already exists; use a new path")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    settings = validate_settings(config)
    if settings["version"] != 4:
        raise ValueError("Target configuration must be QALCO v4")
    content = source.read_bytes()
    groups, identities, requests, group_ids = [], set(), set(), set()
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            group = replay_group(json.loads(line), config,
                                 {"path": str(source.resolve()), "line": line_number,
                                  "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()})
        except (ValueError, KeyError, TypeError, RewardError) as exc:
            raise ValueError(f"Input line {line_number}: {exc}") from exc
        identity = (group["session_id"], group["run_id"], group["source_reward_version"])
        identities.add(identity)
        ids = {r["request_id"] for r in group["candidates"]}
        if len(identities) != 1 or group["global_group_id"] in group_ids or requests & ids:
            raise ValueError("Mixed run/version or repeated group/request identity across input")
        requests.update(ids)
        group_ids.add(group["global_group_id"])
        groups.append(group)
        if len(groups) == 1 or len(groups) % 20 == 0:
            print(f"[replay] Verified and rescored groups={len(groups)} candidates={8 * len(groups)}", flush=True)
    if not groups:
        raise ValueError("Input contains no complete G8 groups")
    summary = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "kind": "frozen_comet_complete_G8_counterfactual_not_policy_evaluation",
               "session_id": groups[0]["session_id"], "run_id": groups[0]["run_id"],
               "source_reward_version": groups[0]["source_reward_version"], "target_reward_version": 4,
               "input": {"path": str(source.resolve()), "sha256": hashlib.sha256(content).hexdigest()},
               "config": {"path": str(config_path.resolve()), "sha256": hashlib.sha256(config_bytes).hexdigest()},
               "statistics": summarize(groups, settings),
               "notes": ["Frozen quality is a fallible proxy, not semantic truth.",
                         "No samples were skipped or shortened to partial groups.",
                         "The selected export is diagnostic, not a random validation population."]}
    code_files = [Path(__file__).resolve(), *[PROJECT_ROOT / "flex_gspo" / name for name in
                  ("qalco.py", "normalized_reward.py", "gated_reward.py", "degeneration_v3.py",
                   "degeneration.py", "protocol.py", "trajectory.py")]]
    summary["code_sha256"] = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}
    target.mkdir(parents=True, exist_ok=False)
    (target / "groups.jsonl").write_text("".join(json.dumps(g, ensure_ascii=False, allow_nan=False) + "\n" for g in groups), encoding="utf-8")
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (target / "summary.txt").write_text(report_text(summary), encoding="utf-8")
    print(f"[replay] Saved: {target.resolve() / 'summary.txt'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="selected_groups.jsonl from collect_reward_revision.py")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "qalco_v4.json"))
    parser.add_argument("--output-dir", required=True, help="New output directory")
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RewardError) as exc:
        print(f"[replay] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
