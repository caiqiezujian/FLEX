import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flex_gspo.artifacts import ArtifactError
from test_artifacts import make_task, write_fixture
from score_candidates import run_scoring, scorer_identity


class TestOnlyScorer:
    """Fixture arithmetic, explicitly not translation-quality estimates."""
    def score_group(self, task, candidates):
        return [
            {"reward": float(index), "components": {
                "quality": 0.5 + index / 10, "laal_seconds": 1.5 - index / 10,
                "raw_laal_seconds": 1.5 - index / 10, "length_penalty": 0.0,
                "tier": "feasible" if index else "quality_fail", "feasible": bool(index),
            }} for index in range(len(candidates))
        ]


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.paths = write_fixture(self.root, [make_task("a"), make_task("b")], shards=2)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"reward": {"provider": "qalco", "quality_floor": 0.85}}))
        self.output = self.root / "calibration"
        self.factory_calls = 0

    def tearDown(self):
        self.temporary.cleanup()

    def factory(self, config):
        self.factory_calls += 1
        return TestOnlyScorer()

    def run_score(self, **kwargs):
        return run_scoring(self.manifest, [self.root], self.config, self.output,
                           scorer_factory=self.factory, **kwargs)

    def test_validate_only_does_not_load_scorer_or_write(self):
        report = self.run_score(validate_only=True)
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["selected_candidates"], 4)
        self.assertEqual(self.factory_calls, 0)
        self.assertFalse(self.output.exists())

    def test_all_candidates_preserved_with_provenance_and_summary(self):
        report = self.run_score()
        self.assertEqual(report["candidates"], 4)
        self.assertEqual(report["feasible_candidates"], 2)
        self.assertEqual(report["metrics"]["quality"]["count"], 4)
        self.assertAlmostEqual(report["metrics"]["quality"]["mean"], 0.55)
        self.assertFalse(report["offline_gspo_replay_ready"])
        self.assertEqual(report["provenance"]["quality_scoring"], "injected_test_scorer")
        rows = [json.loads(line) for line in (self.output / "scored.trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["id"] for row in rows], ["a", "b"])
        for row in rows:
            self.assertEqual(len(row["trajectories"]), 2)
            self.assertIn("score", row["trajectories"][0])
            self.assertIn("raw_token_ids", row["trajectories"][0]["turns"][0])
        self.assertEqual(json.loads((self.output / "run.json").read_text(encoding="utf-8"))["status"], "complete")

    def test_limit_scores_sorted_available_ids_without_best_only_filter(self):
        report = self.run_score(limit=1)
        self.assertEqual(report["available_groups"], 2)
        self.assertEqual(report["selected_groups"], 1)
        self.assertEqual(report["candidates"], 2)

    def test_existing_output_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "mine.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_score()
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.factory_calls, 0)

    def test_missing_audio_only_errors_when_requested(self):
        with self.assertRaisesRegex(ArtifactError, "audio file does not exist"):
            self.run_score(validate_only=True, check_audio=True)

    def test_model_failure_recorded_not_converted_to_low_reward(self):
        class Broken:
            def score_group(self, task, candidates):
                raise RuntimeError("fixture scorer failure")
        with self.assertRaisesRegex(RuntimeError, "fixture scorer failure"):
            run_scoring(self.manifest, [self.root], self.config, self.output,
                        scorer_factory=lambda config: Broken())
        record = json.loads((self.output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["completed_groups"], 0)
        self.assertFalse((self.output / "scored.trajectories.jsonl").exists())

    def test_partial_sampling_is_explicit_not_claimed_complete_dataset(self):
        self.paths[1].write_text("")
        report = self.run_score(validate_only=True)
        self.assertEqual(report["manifest_tasks_without_candidates"], 1)
        self.assertEqual(report["available_groups"], 1)

    def test_unsupported_provider_is_rejected(self):
        self.config.write_text(json.dumps({"reward": {"provider": "unsupported_provider"}}))
        with self.assertRaisesRegex(ArtifactError, "provider=qalco"):
            self.run_score()
        self.assertEqual(self.factory_calls, 0)

    def test_bad_score_missing_feasibility_aborts(self):
        class Bad:
            def score_group(self, task, candidates):
                return [{"reward": 1, "components": {}} for _ in candidates]
        with self.assertRaisesRegex(ArtifactError, "feasibility"):
            run_scoring(self.manifest, [self.root], self.config, self.output,
                        scorer_factory=lambda config: Bad())

    def test_production_preflight_records_declared_service_identity_without_loading_weights(self):
        config = {"reward": {"provider": "qalco", "metricx": {
            "backend": "service", "service_url": "http://localhost:9010/score",
            "service_id": "metricx24-qe-v1", "mode": "qe"}}}
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with patch("flex_gspo.rewards.validate_reward_setup") as validate, \
             patch("flex_gspo.rewards.build_scorer") as build:
            report = run_scoring(self.manifest, [self.root], self.config, validate_only=True)
        validate.assert_called_once()
        build.assert_not_called()
        identity = report["provenance"]["quality_scorer_identity"]
        self.assertIn("metricx.py", identity["details"]["implementation_sha256"])
        self.assertEqual(identity["details"]["service_id"], "metricx24-qe-v1")
        self.assertNotIn("model", identity["details"])
        self.assertEqual(identity["details"]["weight_identity_basis"],
                         "declared_service_identity_not_verified_remote_weights")
        original = scorer_identity(config)
        config["reward"]["metricx"]["service_id"] = "metricx24-qe-v2"
        self.assertNotEqual(original, scorer_identity(config))
        self.assertFalse(report["model_loaded"])


if __name__ == "__main__":
    unittest.main()
