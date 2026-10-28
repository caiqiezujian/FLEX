import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "Scripts" / "summarize_training_history.py"
spec = importlib.util.spec_from_file_location("training_history", SCRIPT)
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.native = self.root / "v0-test" / "logging.jsonl"
        plan = {"session_id": "session-a", "run_id": "run-a", "argv": [
            "--num_iterations", "1", "--steps_per_generation", "1", "--global_batch_size", "32",
            "--num_generations", "8", "--nproc_per_node", "4", "--nnodes", "1"]}
        (self.root / "launch_001.json").write_text(json.dumps(plan), encoding="utf-8")
        write_rows(self.native, [{"iteration": f"{i}/3000", "reward": -1 + i / 10,
                                 "loss": .0001, "learning_rate": i * 1e-8} for i in (1, 2, 3)])
        self.args = argparse.Namespace(run_dir=str(self.root), native_log=str(self.native),
                                       native_only=False, block_size=2, output_dir=None)

    def reward(self, gs, rank, index, session="session-a"):
        return {"session_id": session, "run_id": "run-a", "global_step": gs,
                "call": gs + 1, "timestamp_unix": 1000 + gs, "task_id": f"task-{rank}",
                "task_hash": f"hash-{rank}", "request_id": f"request-{gs}-{rank}-{index}",
                "reward": -.5 + index / 10,
                "components": {"global_group_id": f"group-{gs}-{rank}",
                    "group_candidate_index": index, "group_size": 8, "group_finalized": True,
                    "tier": "quality_fail", "quality": .7, "quality_gate": 0,
                    "protocol_valid": True, "empty_translation": False,
                    "quality_pass": False, "feasible": False},
                "trajectory": {"turns": [{"kind": "audio", "response": "<|DIRECT|>你好"},
                                          {"kind": "eof", "response": "<|WAIT|>"}]}}

    def setup_rewards(self, missing=False):
        for rank in range(4):
            rows = [self.reward(gs, rank, i) for gs in range(4) for i in range(8)
                    if not (missing and gs == 1 and rank == 3 and i == 7)]
            rows += [self.reward(0, rank, 0, "old-session")]
            write_rows(self.root / "rewards" / f"rewards-rank{rank}-pid{rank + 10}.jsonl", rows)

    def test_all_steps_and_identity_alignment(self):
        self.setup_rewards()
        report, rows, rewards = history.analyze(self.args)
        self.assertEqual(report["native"]["steps"], 3)
        self.assertEqual(report["rewards"]["candidates"], 96)
        self.assertEqual(report["rewards"]["complete_groups"], 12)
        self.assertEqual(set(rewards), {1, 2, 3})
        self.assertEqual(report["ignored_reward_rows"]["not_in_completed_native_steps"], 32)
        self.assertEqual(report["ignored_reward_rows"]["other_session_or_run"], 4)
        self.assertGreater(report["native"]["metrics"]["learning_rate"]["mean"], 0)
        self.assertIn("e-08", history.dump(report))
        self.assertEqual(report["reward_reconciliation"]["mismatch_steps"], 3)

    def test_partial_and_corrupt_lines_are_reported(self):
        self.setup_rewards()
        path = self.root / "rewards" / "rewards-rank0-pid10.jsonl"
        with path.open("ab") as handle:
            handle.write(b"not-json\n{\"unfinished\":")
        report, _, _ = history.analyze(self.args)
        item = next(m for m in report["file_diagnostics"] if m["path"] == str(path))
        self.assertTrue(item["partial_tail_ignored"])
        self.assertEqual(item["bad_complete_lines"], 1)
        self.assertTrue(report["warnings"])

    def test_incomplete_g8_excluded_without_losing_native_step(self):
        self.setup_rewards(missing=True)
        report, _, rewards = history.analyze(self.args)
        self.assertEqual(report["native"]["steps"], 3)
        self.assertEqual(set(rewards), {1, 3})
        self.assertIn(2, report["excluded_reward_steps"])

    def test_duplicate_native_step_refused(self):
        with self.native.open("a", encoding="utf-8") as handle:
            handle.write('{"iteration":"3/3000","reward":0}\n')
        self.args.native_only = True
        with self.assertRaisesRegex(ValueError, "Duplicate/nonmonotonic"):
            history.analyze(self.args)

    def test_native_only_no_launch_needed(self):
        (self.root / "launch_001.json").unlink()
        self.args.native_only = True
        report, _, _ = history.analyze(self.args)
        self.assertIsNone(report["rewards"])

    def test_snapshot_ignores_appended_data(self):
        metadata = []
        iterator = history.snapshot(self.native, metadata)
        first = next(iterator)
        with self.native.open("a", encoding="utf-8") as handle:
            handle.write('{"iteration":"4/3000","reward":0}\n')
        self.assertEqual([history.step(first)] + [history.step(row) for row in iterator], [1, 2, 3])

    def test_alignment_rejects_changed_recipe(self):
        path = self.root / "launch_001.json"
        plan = json.loads(path.read_text())
        plan["argv"][1] = "2"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "alignment"):
            history.analyze(self.args)

    def test_old_native_is_not_combined_with_new_launch(self):
        launch_time = (self.root / "launch_001.json").stat().st_mtime
        os.utime(self.native, (launch_time - 100, launch_time - 100))
        with self.assertRaisesRegex(ValueError, "predates"):
            history.analyze(self.args)

    def test_explicit_launch_selects_original_session(self):
        self.setup_rewards()
        original = self.root / "launch_001.json"
        other = self.root / "launch_002.json"
        plan = json.loads(original.read_text())
        plan["session_id"] = "other-session"
        other.write_text(json.dumps(plan))
        self.args.launch_file = str(original)
        report, _, _ = history.analyze(self.args)
        self.assertEqual(report["session_id"], "session-a")
        self.assertEqual(report["launch_selection"], "explicit")
        self.assertEqual(report["rewards"]["candidates"], 96)

    def test_launch_outside_root_is_rejected(self):
        self.args.launch_file = str(self.root.parent / "outside.json")
        with self.assertRaisesRegex(ValueError, "inside run_dir"):
            history.analyze(self.args)


if __name__ == "__main__":
    unittest.main()
