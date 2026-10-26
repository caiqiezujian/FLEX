"""Offline candidate auditing never loads actor/audio or invents EOF rewards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from Scripts import audit_reward_candidates as audit
from flex_gspo.rewards import RewardError


class CandidateAuditTests(unittest.TestCase):
    @staticmethod
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def row(text, index=0):
        return {"task_id": "sample", "global_step": 1, "call": 2, "candidate_index_in_local_batch": index,
                "translation": text, "reward": -4 if not text else .8, "components": {"tier": "invalid" if not text else "feasible"},
                "trajectory": {"complete_input": True, "valid": True, "turns": [
                    {"response": "<|DIRECT|>" + text if text else "<|WAIT|>", "audio_end_seconds": 2, "finish_reason": "stop"}]}}

    def fixture(self, directory):
        root = Path(directory)
        log = root / "run" / "rewards" / "rewards-rank0-pid123.jsonl"
        dataset = root / "train.jsonl"
        self.write_rows(log, [self.row("你好。"), self.row("", 1), self.row("我喜欢它们。" * 13, 2)])
        self.write_rows(dataset, [{"flex_task": {"id": "sample", "source_text": "Hello.",
                                                   "reference_text": "你好。", "duration_seconds": 2}}])
        return root, log, dataset

    def test_real_texts_raw_scores_loop_guard_no_new_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            root, log, dataset = self.fixture(temp)
            before = log.read_bytes()
            comet = MagicMock()
            comet.score_batch.return_value = [.95]  # Guard must still catch high-scoring repetition.
            output = root / "audit.jsonl"
            report = audit.run_audit(root / "run", dataset, output, {}, comet_scorer=comet)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["quality_scored"], 2)
            self.assertEqual(report["written_candidates"], 3)
            self.assertEqual(comet.score_batch.call_count, 2)
            repeated = next(row for row in rows if row["selection_category"] == "possible_loop")
            self.assertTrue(repeated["degeneration"]["severe_repetition"])
            self.assertEqual(repeated["comet22_score"], .95)
            self.assertIsNone(repeated["new_full_reward"])
            self.assertFalse(repeated["eof_metadata_present"])
            self.assertEqual(log.read_bytes(), before)
            self.assertIsNone(next(row for row in rows if row["empty_translation"])["comet22_score"])

    def test_existing_outputs_fail_before_service(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, dataset = self.fixture(temp)
            output = root / "existing.jsonl"
            output.write_text("keep", encoding="utf-8")
            comet = MagicMock()
            with self.assertRaisesRegex(RewardError, "NEW"):
                audit.run_audit(root / "run", dataset, output, {}, comet_scorer=comet)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")
            comet.score_batch.assert_not_called()

    def test_service_failure_is_not_policy_reward_and_leaves_failed_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, dataset = self.fixture(temp)
            output = root / "audit.jsonl"
            comet = MagicMock()
            comet.score_batch.side_effect = RewardError("service unavailable")
            with self.assertRaises(RewardError):
                audit.run_audit(root / "run", dataset, output, {}, comet_scorer=comet)
            report = json.loads((root / "audit.jsonl.summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["quality_scored"], 0)
            self.assertFalse(report["new_full_rewards_computed"])

    def test_wrong_pid_missing_task_and_conflicting_task_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _, dataset = self.fixture(temp)
            with self.assertRaises(RewardError):
                audit.select_candidates(root / "run", pids=["999"])
            selected, _ = audit.select_candidates(root / "run", limit=2)
            self.assertEqual(len(selected), 2)
            self.write_rows(dataset, [{"flex_task": {"id": "other"}}])
            with self.assertRaisesRegex(RewardError, "missing"):
                audit.join_tasks(dataset, selected)
            self.write_rows(dataset, [{"id": "sample", "source_text": "A", "reference_text": "甲"},
                                      {"id": "sample", "source_text": "B", "reference_text": "乙"}])
            with self.assertRaisesRegex(RewardError, "conflicting"):
                audit.join_tasks(dataset, selected)

    def test_stored_text_disagreement_never_silently_repaired(self):
        row = self.row("你好")
        row["translation"] = "另一个译文"
        with self.assertRaisesRegex(RewardError, "disagrees"):
            audit._translation(row)


if __name__ == "__main__":
    unittest.main()
