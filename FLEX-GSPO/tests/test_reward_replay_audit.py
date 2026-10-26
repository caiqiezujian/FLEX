"""Replay completeness and provenance failures must not fabricate a G8 reward."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Scripts import replay_reward_audit as replay
from flex_gspo.gated_reward import validate_v3_settings
from flex_gspo.qalco import emission_schedule, latency_statistics, text_units


class RewardReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.settings = validate_v3_settings({"version": 3, "quality_provider": "comet22",
            "latency_metric": "mean_lag", "eof_delay_seconds": .5})
        self.group = {"group": 1, "task_id": "test", "step_zero_based": 4,
            "observed_candidates": 8, "source_text": "The meeting will take place tomorrow.",
            "reference_text": "会议明天举行。", "candidates": []}
        turns = [{"turn": 1, "kind": "audio", "audio_end_seconds": 1., "finish_reason": "stop", "response": "<|WAIT|>"},
            {"turn": 2, "kind": "audio", "audio_end_seconds": 2., "finish_reason": "stop", "response": "<|FLEX|>会议"},
            {"turn": 3, "kind": "eof", "audio_end_seconds": 2., "finish_reason": "stop", "eof_elapsed_seconds": 100.,
             "response": "<|DIRECT|>明天举行。"}]
        task = {"duration_seconds": 2., "chunks": [{"end_seconds": 1.}, {"end_seconds": 2.}]}
        _, times = emission_schedule(task, turns, .5)
        lag = latency_statistics(2., times, len(text_units(self.group["reference_text"])), .5)["mean_lag_seconds"]
        for i in range(8):
            self.group["candidates"].append({"candidate_in_report": i+1,
                "candidate_index_in_local_batch": i, "request_id": f"request-{i}", "prompt_id": "",
                "source_file": "rewards-rank0-pid1.jsonl", "call": 5, "task_id": "test",
                "global_step_zero_based": 4, "duration_seconds": 2., "audio_turns": 2,
                "callback_identity": ["rewards-rank0-pid1.jsonl", "session", "run", 4, 5, 1., "test"],
                "protocol_valid": True, "complete_input": True, "episode_complete": True,
                "protocol_error": None, "stored_translation_mismatch": False,
                "eof_actions": ["DIRECT"], "comet22_score": .7 + i*.03, "reward": -.5,
                "latency_seconds": lag, "text_units": 6, "translation": "会议明天举行。",
                "tier": "quality_fail", "turns": copy.deepcopy(turns)})

    def export(self):
        g = self.group
        lines = ["音频组: " + json.dumps({k: v for k, v in g.items() if k != "candidates"}),
            "英文原文: " + g["source_text"], "中文机器参考: " + g["reference_text"]]
        for r in g["candidates"]:
            lines.extend(["候选: " + json.dumps({k: v for k, v in r.items() if k not in ("turns", "translation")}),
                "完整译文: " + r["translation"], "逐轮原始输出（包括 WAIT 和 EOF）:"])
            lines.extend(json.dumps(t, ensure_ascii=False) for t in r["turns"])
        return "\n".join(lines) + "\n"

    def test_current_eligibility_not_old_quality_tier_or_eof_wall_time(self):
        result = replay.replay_group(self.group, self.settings)
        self.assertEqual(result["v3_top"], [8])
        self.assertTrue(all(r["v3"]["eligible"] for r in result["candidates"]))
        self.assertEqual([r["v3"]["quality"] for r in result["candidates"]],
                         [r["comet22_score"] for r in self.group["candidates"]])
        self.assertLess(result["candidates"][0]["v3"]["latency"], 3.)

    def test_missing_single_raw_trace_rejects_entire_group(self):
        self.group["candidates"][-1]["turns"] = []
        with self.assertRaisesRegex(ValueError, "incomplete raw turns"):
            replay.replay_group(self.group, self.settings)

    def test_mixed_callbacks_or_duplicate_requests_rejected(self):
        bad = copy.deepcopy(self.group)
        bad["candidates"][1]["callback_identity"][4] += 1
        with self.assertRaisesRegex(ValueError, "Mixed callbacks"):
            replay.replay_group(bad, self.settings)
        self.group["candidates"][1]["request_id"] = "request-0"
        with self.assertRaisesRegex(ValueError, "distinct"):
            replay.replay_group(self.group, self.settings)

    def test_shared_schedule_and_raw_text_mismatch_rejected(self):
        bad = copy.deepcopy(self.group)
        bad["candidates"][1]["turns"][0]["audio_end_seconds"] = .5
        with self.assertRaisesRegex(replay.RewardError, "timing disagrees"):
            replay.replay_group(bad, self.settings)
        self.group["candidates"][1]["translation"] += "。"
        with self.assertRaisesRegex(ValueError, "differs"):
            replay.replay_group(self.group, self.settings)

    def test_split_files_offline_execution_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            text = self.export()
            split = text.index("完整译文: ", text.index('"candidate_in_report": 5'))
            paths = [root/"a.txt", root/"b.txt"]
            paths[0].write_text(text[:split], encoding="utf-8")
            paths[1].write_text(text[split:], encoding="utf-8")
            config = root/"v3.json"
            config.write_text(json.dumps({"reward": self.settings}), encoding="utf-8")
            output = root/"output"
            argv = ["--input-txt", *map(str, paths), "--config", str(config), "--output-dir", str(output)]
            before = [p.read_bytes() for p in paths]
            with patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(replay.main(argv), 0)
                self.assertEqual(replay.main(argv), 2)
            result = json.loads((output/"replay.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["groups"]), 1)
            self.assertEqual(len(result["groups"][0]["candidates"]), 8)
            self.assertEqual(result["groups"][0]["candidates"][4]["translation"], "会议明天举行。")
            self.assertEqual([p.read_bytes() for p in paths], before)


if __name__ == "__main__":
    unittest.main()
