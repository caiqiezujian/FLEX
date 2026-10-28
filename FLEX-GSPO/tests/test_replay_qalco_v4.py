"""Complete-group frozen-quality replay, including invalid trajectories."""
import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Scripts import replay_qalco_v4 as replay
from flex_gspo.gated_reward import finalize_group as finalize_v3
from flex_gspo.normalized_reward import finalize_group as finalize_v4
from flex_gspo.protocol import collect_translation
from flex_gspo.qalco import QalcoScorer
from flex_gspo.rewards import RewardError


def config(version):
    return json.loads((ROOT / "configs" / f"qalco_v{version}.json").read_text(encoding="utf-8-sig"))


def export_group(*, empty_indices=(), invalid_indices=(), version=3):
    task = {"id": "synthetic-meeting", "source_lang": "en", "target_lang": "zh",
            "source_text": "We meet tomorrow to discuss the annual budget.",
            "reference_text": "我们明天开会讨论年度预算。", "duration_seconds": 2.0,
            "chunks": [{"end_seconds": 1.0}, {"end_seconds": 2.0}]}
    rows, items = [], []
    cfg = config(version)
    for index in range(8):
        text = "" if index in empty_indices else "我们明天开会讨论年度预算。" + "补充" * index
        trajectory = {"schema_version": 2, "valid": True, "complete_input": True,
                      "episode_complete": True, "eof_completed": True,
                      "turns": [
                          {"kind": "audio", "audio_end_seconds": 1.0, "finish_reason": "stop", "response": "<|WAIT|>"},
                          {"kind": "audio", "audio_end_seconds": 2.0, "finish_reason": "stop",
                           "response": "<|DIRECT|>" + text if text else "<|WAIT|>"},
                          {"kind": "eof", "audio_end_seconds": 2.0, "finish_reason": "stop",
                           "eof_elapsed_seconds": .1, "response": "BAD" if index in invalid_indices else "<|WAIT|>"}]}
        try:
            translation = collect_translation(trajectory["turns"])
        except ValueError:
            translation = None
        q = .74 + index * .02
        scorer = QalcoScorer(cfg, quality_model=replay.FrozenQuality(task, translation, q))
        result = scorer.score(task, trajectory)
        c = result["components"]
        items.append({"eligible": c["eligible"], "quality": c.get("quality"), "latency": c.get("latency_seconds"),
                      "units": c["emitted_units"], "repetition": c["repetition_penalty"], "reward": result["reward"]})
        rows.append({"session_id": "test-session", "run_id": "test-run", "global_step": 12,
                     "task_id": task["id"], "task_hash": replay.digest(task), "duration_seconds": 2.0,
                     "reward_version": version, "request_id": f"req-{index}", "prompt_id": "prompt-1",
                     "translation": translation, "trajectory": trajectory, **result})
    finalizer = finalize_v3 if version == 3 else finalize_v4
    final = finalizer(items, cfg["reward"])
    gid = replay.digest({"offset": 0, "task_hash": replay.digest(task), "prompt_id": "prompt-1",
                         "request_ids": [r["request_id"] for r in rows]}, compact=True)
    for index, (row, scores) in enumerate(zip(rows, final)):
        row["reward"] = scores["reward"]
        row["components"].update({k: v for k, v in scores.items() if k not in ("eligible", "quality", "latency", "units", "repetition", "reward")})
        row["components"].update(global_group_id=gid, group_candidate_index=index, group_size=8,
                                 group_finalized=True, global_group_index=0,
                                 group_prompt_id="prompt-1", group_request_id=row["request_id"])
    return {"selection": "synthetic-test", "task": task, "candidates": rows,
            "group": {"global_group_id": gid, "task_id": task["id"], "task_hash": replay.digest(task), "native_step": 13}}


class ReplayV4Test(unittest.TestCase):
    def setUp(self):
        self.config = config(4)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_export(self, groups):
        source = self.root / "selected_groups.jsonl"
        source.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in groups), encoding="utf-8")
        args = argparse.Namespace(input_jsonl=str(source), config=str(ROOT / "configs" / "qalco_v4.json"),
                                  output_dir=str(self.root / "replay"))
        with redirect_stdout(io.StringIO()):
            return replay.run(args)

    def test_complete_group_scores_and_advantages_recomputed_from_frozen_evidence(self):
        original = export_group()
        result = replay.replay_group(original, self.config, {"path": "original.jsonl", "line": 1})
        self.assertEqual(len(result["candidates"]), 8)
        self.assertAlmostEqual(sum(r["new_advantage"] for r in result["candidates"]), 0)
        old = [r["reward"] for r in original["candidates"]]
        expected = [(r - statistics.mean(old)) / (statistics.stdev(old) + 1e-4) for r in old]
        for row, advantage in zip(result["candidates"], expected):
            self.assertAlmostEqual(row["old_advantage"], advantage)
            self.assertEqual(row["quality"], original["candidates"][row["index"]]["components"]["quality"])
            self.assertEqual(row["turn_count"], 3)
            self.assertEqual(row["original_trajectory_pointer"]["input"]["line"], 1)

    def test_empty_and_protocol_errors_remain_missing_quality_not_zero(self):
        result = replay.replay_group(export_group(empty_indices=(0, 1), invalid_indices=(2,)), self.config)
        for row in result["candidates"][:3]:
            self.assertIsNone(row["quality"])
            self.assertIsNone(row["v4"]["quality"])
            self.assertFalse(row["v4"]["eligible"])
        self.assertEqual(result["candidates"][0]["new_reward"], result["candidates"][2]["new_reward"])
        summary = replay.summarize([result], self.config["reward"])
        self.assertEqual(summary["empty_candidates"], 2)

    def test_all_invalid_is_reported_and_has_zero_advantage(self):
        result = replay.replay_group(export_group(empty_indices=range(8)), self.config)
        self.assertTrue(all(r["new_advantage"] == 0 for r in result["candidates"]))
        self.assertTrue(all(r["old_advantage"] == 0 for r in result["candidates"]))
        self.assertEqual(replay.summarize([result], self.config["reward"])["all_invalid_groups"], 1)

    def test_v4_source_uses_center_only_not_sample_std(self):
        result = replay.replay_group(export_group(version=4), self.config)
        for row in result["candidates"]:
            self.assertAlmostEqual(row["old_advantage"], row["new_advantage"])
            self.assertAlmostEqual(row["old_reward"], row["new_reward"])

    def test_missing_scores_are_rejected_instead_of_borrowed_or_fabricated(self):
        group = export_group()
        group["candidates"][0]["components"]["quality"] = None
        group["candidates"][0]["components"]["comet22_score"] = None
        with self.assertRaisesRegex(ValueError, "Missing frozen"):
            replay.replay_group(group, self.config)

    def test_changed_task_duplicate_indices_and_mixed_sessions_rejected(self):
        for mutation in (lambda g: g["task"].update(source_text="A different source."),
                         lambda g: g["candidates"][0]["components"].update(group_candidate_index=1),
                         lambda g: g["candidates"][0].update(session_id="another-session"),
                         lambda g: g["candidates"][0].update(reward_version=4),
                         lambda g: g["candidates"][0]["components"].update(group_request_id="tampered")):
            with self.subTest(mutation=mutation):
                group = export_group()
                mutation(group)
                with self.assertRaises(ValueError):
                    replay.replay_group(group, self.config)

    def test_raw_translation_and_schedule_are_independently_checked(self):
        group = export_group()
        group["candidates"][0]["translation"] = "修改后译文"
        with self.assertRaisesRegex(ValueError, "append-only"):
            replay.replay_group(group, self.config)
        group = export_group()
        group["candidates"][0]["trajectory"]["turns"][1]["audio_end_seconds"] = 1.5
        with self.assertRaisesRegex(RewardError, "timing"):
            replay.replay_group(group, self.config)
        group = export_group()
        group["candidates"][0]["components"]["latency_seconds"] = 999
        result = replay.replay_group(group, self.config)
        self.assertNotEqual(result["candidates"][0]["v4"]["latency"], 999)

    def test_repeated_group_or_partial_group_refused_without_output(self):
        group = export_group()
        with self.assertRaisesRegex(ValueError, "repeated group"):
            self.run_export([group, copy.deepcopy(group)])
        self.assertFalse((self.root / "replay").exists())
        group["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "eight complete"):
            self.run_export([group])
        self.assertFalse((self.root / "replay").exists())

    def test_reports_include_integrity_and_raw_turn_pointer_without_remote_calls(self):
        summary = self.run_export([export_group(empty_indices=(0,))])
        self.assertEqual(summary["statistics"]["candidates"], 8)
        self.assertEqual(summary["statistics"]["empty_candidates"], 1)
        out = self.root / "replay"
        self.assertTrue((out / "summary.json").is_file())
        self.assertIn("没有生成新轨迹", (out / "summary.txt").read_text(encoding="utf-8"))
        group = json.loads((out / "groups.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(group["candidates"][0]["original_trajectory_pointer"]["input"]["line"], 1)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_export([export_group()])


if __name__ == "__main__":
    unittest.main()
