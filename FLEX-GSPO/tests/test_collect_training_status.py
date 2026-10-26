"""Live diagnostic snapshots preserve identities, partial writes and numbers."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

from Scripts import collect_training_status as status


class CollectTrainingStatusTests(unittest.TestCase):
    def write_rows(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def candidate(self, call=1, index=0, session="current", reward=.6, text="会议九点开始。"):
        return {"run_id": "run-current", "session_id": session, "call": call,
                "global_step": call-1, "timestamp_unix": 1700000000.+call,
                "candidate_index_in_local_batch": index, "task_id": "task-1", "duration_seconds": 2.,
                "translation": text, "reward": reward,
                "components": {"quality": .9, "comet22_score": .9, "latency_seconds": 1.,
                    "length_penalty": 0., "protocol_valid": True, "empty_translation": False,
                    "quality_pass": True, "latency_pass": True, "feasible": True,
                    "severe_repetition": False, "tier": "feasible",
                    "diagnostics": {"eof_text_fraction": .1}},
                "trajectory": {"valid": True, "complete_input": True, "episode_complete": True,
                    "eof_completed": True, "turns": [
                        {"kind": "audio", "response": "<|DIRECT|>"+text, "audio_end_seconds": 1., "finish_reason": "stop"},
                        {"kind": "audio", "response": "<|WAIT|>", "audio_end_seconds": 2., "finish_reason": "stop"},
                        {"kind": "eof", "response": "<|WAIT|>", "audio_end_seconds": 2., "finish_reason": "stop"}]}}

    def fixture(self, directory):
        root = Path(directory)/"run"
        root.mkdir()
        dataset = Path(directory)/"train.jsonl"
        self.write_rows(dataset, [{"flex_task": json.dumps({"id": "task-1", "source_text": "The meeting starts at nine.",
            "reference_text": "会议九点开始。"}, ensure_ascii=False)}])
        config = Path(directory)/"config.json"
        config.write_text(json.dumps({"reward": {"version": 2, "quality_provider": "comet22",
            "quality_floor": .8, "latency_budget_seconds": 3.}}), encoding="utf-8")
        launch = root/"launch_20261022T000000.json"
        launch.write_text(json.dumps({"run_id": "run-current", "session_id": "current",
            "argv": ["python3", "--dataset", str(dataset), "--train_iters", "3000", "--save_steps", "500"],
            "sampling": {"group_size": 8}, "environment": {"FLEX_REWARD_CONFIG": str(config)},
            "run_identity": {"reward_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}}), encoding="utf-8")
        os.utime(launch, (time.time()-100, time.time()-100))
        native = root/"v0-current"/"logging.jsonl"
        self.write_rows(native, [{"iteration": "1/3000", "loss": .0001, "learning_rate": 5.5e-7,
                                 "elapsed_time": "1m 10s", "grad_norm": .3},
                                {"iteration": "2/3000", "loss": .0002, "learning_rate": 5.6e-7,
                                 "elapsed_time": "1m 20s", "grad_norm": .4}])
        rewards = root/"rewards"/"rewards-rank0-pid123.jsonl"
        self.write_rows(rewards, [self.candidate(session="old", reward=999., text="OLD SESSION MUST NOT APPEAR"),
            self.candidate(1, 0), self.candidate(1, 1, reward=.7),
            self.candidate(2, 0), self.candidate(2, 1, reward=.8),
            self.candidate(3, 0, reward=998., text="UNFINISHED CALLBACK MUST NOT APPEAR")])
        self.write_rows(root/"performance"/"performance-rank0-pid123.jsonl", [
            {"timestamp_unix": time.time(), "rank": "0", "pid": 123, "call": 2,
             "global_step": 1, "generation_seconds": 15., "offload_to_cpu_seconds": 2., "status": "ok"}])
        self.write_rows(root/"performance"/"phases-rank0-pid123.jsonl", [
            {"timestamp_unix": time.time(), "rank": "0", "pid": 123, "call": 3,
             "global_step": 2, "phase": "generation", "event": "begin"}])
        return root, dataset, config, native

    def report_item(self, report, label):
        prefix = label+": "
        return json.loads(next(line[len(prefix):] for line in report.splitlines() if line.startswith(prefix)))

    def test_tail_ignores_partial_utf8_line_and_partial_last_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"live.jsonl"
            first = (json.dumps({"text": "很长的旧记录"*20}, ensure_ascii=False)+"\n").encode()
            second = (json.dumps({"text": "保留完整中文"}, ensure_ascii=False)+"\n").encode()
            partial = '{"text":"尚未写完'.encode()
            path.write_bytes(first+second+partial)
            rows, meta = status.read_tail(path, len(second)+len(partial)+2, time.time())
            self.assertEqual(rows, [{"text": "保留完整中文"}])
            self.assertTrue(meta["tail_limited"])
            self.assertTrue(meta["leading_partial_line_ignored"])
            self.assertTrue(meta["partial_tail_ignored"])
            self.assertEqual(meta["bad_complete_lines"], 0)

    def test_tail_exact_newline_boundary_keeps_first_complete_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"live.jsonl"
            suffix = b'{"n":2}\n{"n":3}\n'
            path.write_bytes(b'{"n":1}\n'+suffix)
            rows, meta = status.read_tail(path, len(suffix), time.time())
            self.assertEqual(rows, [{"n": 2}, {"n": 3}])
            self.assertFalse(meta.get("leading_partial_line_ignored", False))

    def test_bad_complete_line_is_counted_and_not_confused_with_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"live.jsonl"
            path.write_bytes(b'{"n":1}\nnot json\n{"unfinished":')
            rows, meta = status.read_tail(path, 1000, time.time())
            self.assertEqual(rows, [{"n": 1}])
            self.assertEqual(meta["bad_complete_lines"], 1)
            self.assertTrue(meta["partial_tail_ignored"])

    def test_callback_fence_excludes_unfinished_last_and_tail_cut_first(self):
        rows = [self.candidate(1, 1), self.candidate(2, 0), self.candidate(2, 1), self.candidate(3, 0)]
        groups, excluded = status.reward_groups(rows, native_step=2, window=10)
        self.assertEqual([group["step"] for group in groups], [1])
        self.assertEqual(excluded, 2)
        groups, excluded = status.reward_groups(rows[-1:], native_step=3, window=10)
        self.assertEqual(len(groups), 1)
        self.assertEqual(excluded, 0)

    def test_numbers_preserve_small_lr_and_count_nonfinite_values(self):
        summary = status.native_window([{"iteration": "20/3000", "learning_rate": 5.5e-7,
                                         "loss": float("nan"), "kl": float("inf")}])
        self.assertEqual(summary["metrics"]["learning_rate"]["mean"], 5.5e-7)
        self.assertEqual(summary["metrics"]["loss"]["count"], 0)
        self.assertEqual(summary["nonfinite_values"], {"loss": 1, "kl": 1})
        encoded = status.js({"summary": summary, "raw": float("nan")})
        self.assertEqual(json.loads(encoded)["raw"], "nan")
        self.assertIn("5.5e-07", encoded)

    def test_speed_uses_elapsed_deltas_and_skips_reset_or_missing_values(self):
        rows = [{"iteration": "20/3000", "elapsed_time": "1h 0m 0s"},
                {"iteration": "21/3000", "elapsed_time": "1h 0m 12s"},
                {"iteration": "23/3000", "elapsed_time": "1h 0m 40s"},
                {"iteration": "24/3000", "elapsed_time": "0m 3s"}]
        result = status.report_speed(rows, 3000)
        self.assertEqual(result["recent_seconds_per_step_from_elapsed_deltas"]["count"], 2)
        self.assertEqual(result["recent_seconds_per_step_from_elapsed_deltas"]["mean"], 13.)
        self.assertAlmostEqual(result["estimated_remaining_hours"], (3000-24)*13/3600)
        self.assertEqual(status.elapsed_seconds("01:02:03"), 3723.)

    def test_missing_flags_are_unknown_and_eof_wait_is_separate_from_audio(self):
        row = self.candidate()
        row["components"].pop("severe_repetition")
        groups, _ = status.reward_groups([row], native_step=1, window=10)
        summary = status.summarize_rewards(groups)
        self.assertEqual(summary["severe_repetition"], {"known": 0, "true": 0, "fraction": None})
        self.assertEqual(summary["audio_actions"]["count"], 2)
        self.assertEqual(summary["audio_actions"]["WAIT"], .5)
        self.assertEqual(summary["eof_actions"]["WAIT"], 1.)
        self.assertEqual(summary["all_wait_candidates"], 0)

    def test_end_to_end_live_snapshot_filters_session_and_saves_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dataset, config, native = self.fixture(tmp)
            out = Path(tmp)/"report.txt"
            argv = [str(root), "--window", "1", "--max-turns", "2", "--examples", "2", "--output", str(out)]
            raw_before = {p: p.read_bytes() for p in root.rglob("*.jsonl")}
            with contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(status.main(argv), 0)
            report = out.read_text(encoding="utf-8")
            self.assertEqual(stdout.getvalue(), report)
            self.assertNotIn("OLD SESSION MUST NOT APPEAR", report)
            self.assertNotIn("UNFINISHED CALLBACK MUST NOT APPEAR", report)
            reward = self.report_item(report, "最近奖励窗口 rewards-rank0-pid123.jsonl")
            self.assertEqual(reward["candidates"], 2)
            self.assertAlmostEqual(reward["reward"]["mean"], .7)
            meta = self.report_item(report, "奖励日志来源")
            self.assertEqual(meta["selected_rows"], 5)
            self.assertEqual(meta["excluded_partial_or_unfenced_callbacks"], 1)
            self.assertTrue(self.report_item(report, "奖励配置核对")["matches_launch_hash"])
            self.assertEqual(self.report_item(report, "启动记录")["configuration"]["save_steps"], "500")
            self.assertEqual(self.report_item(report, "阶段耗时 rank0")["metrics"]["generation_seconds"]["mean"], 15.)
            self.assertIn("英文原文: The meeting starts at nine.", report)
            self.assertIn('"kind":"eof"', report)
            self.assertIn("仅显示前 1 轮和最后一轮", report)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(status.main(argv), 2)
            self.assertEqual(out.read_text(encoding="utf-8"), report)
            self.assertEqual({p: p.read_bytes() for p in raw_before}, raw_before)

    def test_explicit_mismatched_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _ = self.fixture(tmp)
            args = status.parser().parse_args([str(root), "--session-id", "old"])
            with self.assertRaisesRegex(ValueError, "session-id"):
                status.collect(args)

    def test_ambiguous_native_logs_do_not_prove_callback_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _ = self.fixture(tmp)
            self.write_rows(root/"v1"/"logging.jsonl", [{"iteration": "999/3000"}])
            args = status.parser().parse_args([str(root), "--examples", "0"])
            report = status.collect(args)
            self.assertIn("多个原生日志", report)
            self.assertNotIn("最近训练步:", report)
            meta = self.report_item(report, "奖励日志来源")
            self.assertEqual(meta["excluded_partial_or_unfenced_callbacks"], 1)

    def test_old_phase_appended_after_current_phase_does_not_replace_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _, _ = self.fixture(tmp)
            phase_path = root/"performance"/"phases-rank0-pid123.jsonl"
            with phase_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestamp_unix": time.time()-3600,
                                         "phase": "OLD_PHASE_MUST_NOT_APPEAR", "event": "begin"})+"\n")
            args = status.parser().parse_args([str(root), "--examples", "0"])
            report = status.collect(args)
            self.assertNotIn("OLD_PHASE_MUST_NOT_APPEAR", report)
            self.assertEqual(self.report_item(report, "最新阶段 rank0")["last_event"]["phase"], "generation")


if __name__ == "__main__":
    unittest.main()
