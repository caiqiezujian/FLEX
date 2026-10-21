"""External MetricX identity/readiness tests; no model loads or network requests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flex_gspo.rewards import RewardError
from flex_gspo.runtime import run_identity
from score_candidates import run_scoring, scorer_identity
from test_artifacts import write_fixture


def service_config():
    return {"reward": {"provider": "qalco", "metricx": {
        "backend": "service", "mode": "qe", "service_url": "", "service_id": "",
    }}}


class ServiceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest, self.paths = write_fixture(self.root)
        self.config = service_config()
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.output = self.root / "calibration"

    def tearDown(self):
        self.temporary.cleanup()

    def test_scorer_identity_needs_no_local_weights_and_hashes_service_hook(self):
        with patch("flex_gspo.runtime.directory_inventory", side_effect=AssertionError("no local weights")):
            identity = scorer_identity(self.config)
        details = identity["details"]
        self.assertEqual(details["backend"], "service")
        self.assertEqual(details["service_url"], "")
        self.assertIn("metricx_service.py", details["implementation_sha256"])
        self.assertNotIn("model", details)
        self.assertNotIn("tokenizer", details)
        self.assertFalse(details["metadata_and_tokenizer_assets_content_hashed"])
        self.assertIn("not_verified_remote_weights", details["weight_identity_basis"])

    def test_service_identity_changes_with_url_version_or_mode(self):
        original = scorer_identity(self.config)
        for name, value in (("service_url", "http://127.0.0.1:9900"),
                            ("service_id", "metricx-version-2"), ("mode", "reference")):
            changed = deepcopy(self.config)
            changed["reward"]["metricx"][name] = value
            self.assertNotEqual(original, scorer_identity(changed))

    def test_resume_identity_inventories_actor_but_not_service_model(self):
        actor = self.root / "actor"
        actor.mkdir()
        (actor / "config.json").write_text("{}", encoding="utf-8")
        config = {**self.config, "model_path": str(actor)}
        identity = run_identity(config, [{"id": "fixture"}], 4)
        models = identity["details"]["models"]
        self.assertEqual(set(models), {"actor", "metricx_service"})
        self.assertEqual(models["metricx_service"]["mode"], "qe")
        changed = deepcopy(config)
        changed["reward"]["metricx"]["service_id"] = "another-version"
        self.assertNotEqual(identity, run_identity(changed, [{"id": "fixture"}], 4))

    def test_preflight_reports_unconfigured_service_without_network_or_writes(self):
        with patch("flex_gspo.metricx_service.request_metricx_scores") as request, \
             patch("flex_gspo.rewards.build_scorer") as build:
            report = run_scoring(self.manifest, self.paths, self.config_path,
                                 self.output, validate_only=True)
        self.assertTrue(report["valid"])
        self.assertFalse(report["quality_service_ready"])
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["provenance"]["quality_scoring"], "service_metricx")
        self.assertFalse(self.output.exists())
        request.assert_not_called()
        build.assert_not_called()

    def test_actual_scoring_stops_before_creating_output_for_missing_configuration(self):
        with patch("flex_gspo.metricx_service.request_metricx_scores") as request, \
             patch("flex_gspo.rewards.build_scorer") as build:
            with self.assertRaisesRegex(RewardError, "service_url"):
                run_scoring(self.manifest, self.paths, self.config_path, self.output)
        self.assertFalse(self.output.exists())
        request.assert_not_called()
        build.assert_not_called()

    def test_configured_preflight_checks_readiness_without_connecting(self):
        self.config["reward"]["metricx"].update(
            service_url="http://127.0.0.1:9900", service_id="metricx-v1")
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with patch("flex_gspo.metricx_service.requests.Session") as session:
            report = run_scoring(self.manifest, self.paths, self.config_path,
                                 self.output, validate_only=True)
        self.assertTrue(report["quality_service_ready"])
        self.assertFalse(self.output.exists())
        session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
