import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="flex-gspo-preflight-")
        self.root = Path(self.temporary.name)
        self.model = self.root / "checkpoint"
        self.model.mkdir()
        # Shape/contents are deliberately not verified by the CPU path check.
        (self.model / "config.json").write_text("{}", encoding="utf-8")
        (self.model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"fixture-not-real-weights")
        self.audio = self.root / "audio.wav"
        self.audio.write_bytes(b"fixture-not-real-audio")
        self.task = {"schema_version": 1, "id": "one", "audio_path": str(self.audio),
                     "source_lang": "en", "target_lang": "zh", "sample_rate": 16000,
                     "duration_seconds": 2.5, "chunks": [
                         {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
                         {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
                         {"start_seconds": 2.0, "end_seconds": 2.5, "input_seconds": 1.0}],
                     "source_text": "A source sentence.", "reference_text": "一句源文。",
                     "quality_ready": True}
        self.manifest = self.root / "tasks.jsonl"
        self.config_file = self.root / "config.json"
        self.config = {"model_path": str(self.model), "manifest_path": str(self.manifest),
                       "output_dir": str(self.root / "out"), "rollout": {"group_size": 2},
                       "reward": {"provider": "qalco", "metricx": {"backend": "service"}}}
        self.write_fixtures()

    def tearDown(self):
        self.temporary.cleanup()

    def write_fixtures(self):
        self.manifest.write_text(json.dumps(self.task) + "\n", encoding="utf-8")
        self.config_file.write_text(json.dumps(self.config), encoding="utf-8")

    def test_training_preflight_without_model_or_network(self):
        with patch("flex_gspo.metricx_service.request_metricx_scores") as api, contextlib.redirect_stdout(io.StringIO()) as output:
            result = train.main(["--config", str(self.config_file), "--validate-only"])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["valid"])
        self.assertFalse(report["quality_service_ready"])
        api.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_audio_only_training_rejected_before_api(self):
        self.task.update(source_text=None, reference_text=None, quality_ready=False)
        self.write_fixtures()
        with patch("flex_gspo.metricx_service.request_metricx_scores") as api, self.assertRaises(ValueError):
            train.load_and_validate(self.config_file)
        api.assert_not_called()

    def test_source_and_reference_training_preflight(self):
        config, tasks = train.load_and_validate(self.config_file)
        self.assertEqual(len(tasks), 1)

    def test_reference_required_even_in_qe_for_length_budget(self):
        from flex_gspo.rewards import RewardError
        self.task.update(reference_text=None)
        self.write_fixtures()
        with self.assertRaisesRegex(RewardError, "reference_text"):
            train.load_and_validate(self.config_file)

    def test_old_output_not_overwritten(self):
        output = self.root / "out"
        output.mkdir()
        marker = output / "run.json"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(ValueError):
            train.load_and_validate(self.config_file)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_new_training_output_can_be_selected(self):
        output = self.root / "out"
        output.mkdir()
        (output / "run.json").write_text("previous run", encoding="utf-8")
        new_output = self.root / "new-training-run"
        config, tasks = train.load_and_validate(self.config_file, output_dir=str(new_output))
        self.assertEqual(config["output_dir"], str(new_output))

    def test_non_raw_sampling_rejected_for_training_consistency(self):
        self.config["rollout"]["temperature"] = .8
        self.write_fixtures()
        with self.assertRaises(ValueError):
            train.load_and_validate(self.config_file)

    def test_malformed_timeline_is_not_accepted(self):
        self.task["duration_seconds"] = float("nan")
        self.write_fixtures()
        with self.assertRaises(ValueError):
            train.load_and_validate(self.config_file)

    def test_nonfinite_and_negative_hyperparameters_rejected(self):
        for key, value in (("learning_rate", float("nan")), ("std_floor", -1), ("clip_low", 1)):
            with self.subTest(key=key):
                self.config["train"] = {key: value}
                self.write_fixtures()
                with self.assertRaises(ValueError):
                    train.load_and_validate(self.config_file)

    def test_no_weights_directory_is_not_a_checkpoint(self):
        (self.model / "model.safetensors").unlink()
        with self.assertRaises(ValueError):
            train.load_and_validate(self.config_file)

    def test_missing_quality_service_fails_before_actor_loading(self):
        from flex_gspo.rewards import RewardError
        from flex_gspo.trainer import run_training
        config, tasks = train.load_and_validate(self.config_file)
        with patch("flex_gspo.trainer.load_backend") as actor, self.assertRaises(RewardError):
            run_training(config, tasks)
        actor.assert_not_called()
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
