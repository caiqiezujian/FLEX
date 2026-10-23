"""Brevity exports keep full observed cohorts and the actual emitted text."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import collect_brevity_samples as brevity


class CollectBrevitySamplesTests(unittest.TestCase):
    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")

    def row(self, index=0, call=1, task="a", prompt="prompt-a", session="current", text="会议将在周末举行。"):
        return {"session_id": session, "run_id": "run-current", "global_step": call-1,
            "call": call, "timestamp_unix": 1700000000.+call,
            "candidate_index_in_local_batch": index, "task_id": task, "prompt_id": prompt,
            "request_id": "request-"+str(index), "duration_seconds": 3., "reward": .7,
            "translation": text, "components": {"protocol_valid": True, "empty_translation": False,
                "severe_repetition": False, "quality_pass": True, "tier": "feasible",
                "comet22_score": .9, "latency_seconds": 1.5, "length_penalty": 0.},
            "trajectory": {"valid": True, "complete_input": True, "episode_complete": True,
                "turns": [{"kind": "audio", "response": "<|DIRECT|>"+text, "audio_end_seconds": 1.},
                          {"kind": "audio", "response": "<|WAIT|>", "audio_end_seconds": 2.},
                          {"kind": "audio", "response": "<|WAIT|>", "audio_end_seconds": 3.},
                          {"kind": "eof", "response": "<|WAIT|>", "audio_end_seconds": 3.}]}}

    def fixture(self, directory):
        root = Path(directory)/"run"
        root.mkdir()
        dataset = Path(directory)/"train.jsonl"
        self.write_rows(dataset, [{"flex_task": json.dumps({"id": task, "source_text": "We meet on the weekend: "+task,
            "reference_text": "我们将在周末开会。"}, ensure_ascii=False)} for task in ("a", "b")])
        (root/"launch_latest.json").write_text(json.dumps({"session_id": "current", "run_id": "run-current",
            "sampling": {"group_size": 8}, "argv": ["python3", "--dataset", str(dataset)]}), encoding="utf-8")
        rows = [self.row(session="old", text="OLD SESSION MUST BE EXCLUDED"),
            self.row(0), self.row(1), self.row(2, prompt="other-prompt"), self.row(3, prompt="other-prompt"),
            self.row(0, call=2, task="b", prompt="prompt-b"), self.row(1, call=2, task="b", prompt="prompt-b"),
            self.row(0, call=3, text="PENDING CALLBACK MUST BE EXCLUDED")]
        self.write_rows(root/"rewards"/"rewards-rank0-pid123.jsonl", rows)
        return root, dataset

    def header(self, report, label):
        prefix = label+": "
        return json.loads(next(line[len(prefix):] for line in report.splitlines() if line.startswith(prefix)))

    def test_complete_translation_and_every_increment_are_not_clipped(self):
        first, second, final = "完整译文"*1300, "然后继续翻译"*100, "最后结束。"
        row = self.row(text=first+second+final)
        row["trajectory"]["turns"] = [
            {"kind": "audio", "response": "<|FLEX|>"+first, "audio_end_seconds": 1.},
            {"kind": "audio", "response": "<|WAIT|>", "audio_end_seconds": 2.},
            {"kind": "audio", "response": "<|DIRECT|>"+second, "audio_end_seconds": 3.},
            {"kind": "eof", "response": "<|DIRECT|>"+final, "audio_end_seconds": 3.}]
        summary = brevity.candidate_summary(row, "rewards-rank0-pid123.jsonl")
        self.assertEqual(summary["translation"], first+second+final)
        self.assertEqual([event["response"] for event in summary["output_events"]],
                         ["<|FLEX|>"+first, "<|DIRECT|>"+second, "<|DIRECT|>"+final])
        self.assertFalse(summary["stored_translation_mismatch"])

    def test_wait_counts_exclude_eof_and_use_audio_denominator(self):
        summary = brevity.candidate_summary(self.row(), "test.jsonl")
        self.assertEqual(summary["audio_turns"], 3)
        self.assertEqual(summary["WAIT"], 2)
        self.assertEqual(summary["wait_fraction"], 2/3)
        self.assertEqual(summary["eof_actions"], ["WAIT"])
        self.assertEqual(len(summary["output_events"]), 1)

    def test_invalid_status_and_stored_mismatch_do_not_replace_original_output(self):
        row = self.row(text="保留的原始存储译文。")
        row["components"].update(protocol_valid=False, tier="invalid")
        row["trajectory"].update(valid=False, error="truncated output")
        row["trajectory"]["turns"][0]["response"] = "<|DIRECT|>不同的真实增量。"
        summary = brevity.candidate_summary(row, "test.jsonl")
        self.assertEqual(summary["translation"], "保留的原始存储译文。")
        self.assertTrue(summary["stored_translation_mismatch"])
        self.assertFalse(summary["protocol_valid"])
        self.assertEqual(summary["protocol_error"], "truncated output")
        self.assertEqual(summary["output_events"][0]["response"], "<|DIRECT|>不同的真实增量。")

    def test_budget_keeps_whole_cohorts_and_at_most_one_round_per_task(self):
        observed = {
            "new": {"step": 4, "task_id": "a", "candidates": [1, 2, 3]},
            "duplicate-task": {"step": 3, "task_id": "a", "candidates": [4, 5]},
            "too-large": {"step": 2, "task_id": "b", "candidates": [6, 7, 8]},
            "fits": {"step": 1, "task_id": "c", "candidates": [9, 10]}}
        chosen = brevity.select_cohorts(observed, max_tasks=10, max_candidates=5)
        self.assertEqual([g["task_id"] for g in chosen], ["a", "c"])
        self.assertEqual([len(g["candidates"]) for g in chosen], [3, 2])
        self.assertEqual(chosen[0]["candidates"], [1, 2, 3])

    def test_current_session_closed_callbacks_and_prompt_cohorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dataset = self.fixture(tmp)
            args = brevity.parser().parse_args([str(root), "--output", str(Path(tmp)/"report.txt"),
                "--max-candidates", "8", "--include-increments"])
            report = brevity.collect(args)
            header = self.header(report, "FLEX 简洁性审查数据")
            self.assertEqual(header["candidates"], 4)
            self.assertEqual(header["task_cohorts"], 2)
            self.assertEqual(header["excluded_partial_callbacks"], 1)
            self.assertNotIn("OLD SESSION MUST BE EXCLUDED", report)
            self.assertNotIn("PENDING CALLBACK MUST BE EXCLUDED", report)
            groups = [json.loads(line.split(": ", 1)[1]) for line in report.splitlines() if line.startswith("音频组: ")]
            self.assertEqual({g["task_id"] for g in groups}, {"a", "b"})
            self.assertTrue(all(g["observed_candidates"] == 2 for g in groups))
            self.assertIn("逐次新增译文（无正常WAIT空轮）", report)
            self.assertIn("英文原文: We meet on the weekend: a", report)

    def test_cli_utf8_report_matches_stdout_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dataset = self.fixture(tmp)
            output = Path(tmp)/"审查报告.txt"
            argv = [str(root), "--dataset", str(dataset), "--output", str(output), "--include-increments"]
            original = {p: p.read_bytes() for p in root.rglob("*.jsonl")}
            with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(brevity.main(argv), 0)
            report = output.read_text(encoding="utf-8")
            self.assertEqual(stdout.getvalue(), report)
            self.assertIn("完整译文: 会议将在周末举行。", report)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(brevity.main(argv), 2)
            self.assertEqual(output.read_text(encoding="utf-8"), report)
            self.assertEqual({p: p.read_bytes() for p in original}, original)


if __name__ == "__main__":
    unittest.main()
