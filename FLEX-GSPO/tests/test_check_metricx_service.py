"""The probe contacts only its configured text adapter, never actor/real GPU."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_metricx_service as probe_cli
from flex_gspo.rewards import RewardError


class ServiceProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = {"reward": {"provider": "qalco", "metricx": {
            "backend": "service", "mode": "qe", "service_url": "http://127.0.0.1:9010",
            "service_id": "metricx-test-deployment"}}}

    def test_default_mode_is_configured_qe_without_reference(self):
        with patch("flex_gspo.metricx_service.request_metricx_scores", return_value=[0.4, 0.3]) as call:
            result = probe_cli.probe(self.config)
        self.assertEqual(result["tested_modes"], ["qe"])
        self.assertTrue(result["service_connectivity_verified"])
        self.assertFalse(result["remote_model_identity_verified"])
        self.assertEqual(call.call_args.args[2], None)
        self.assertEqual(result["results"]["qe"]["qalco_quality"], [1 - 0.4 / 25, 1 - 0.3 / 25])

    def test_both_modes_use_separate_requests(self):
        with patch("flex_gspo.metricx_service.request_metricx_scores", side_effect=[[0.4, 0.3], [0.2, 0.1]]) as call:
            result = probe_cli.probe(self.config, "both")
        self.assertEqual(result["tested_modes"], ["qe", "reference"])
        self.assertEqual(call.call_count, 2)
        self.assertIsNone(call.call_args_list[0].args[2])
        self.assertEqual(call.call_args_list[1].args[2], probe_cli.REFERENCE)
        self.assertEqual(call.call_args_list[1].kwargs["mode"], "reference")
        self.assertEqual(result["results"]["reference"]["scores"], [0.2, 0.1])

    def test_failure_does_not_print_a_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.config), encoding="utf-8")
            output = io.StringIO()
            with patch("check_metricx_service.probe", side_effect=RewardError("no reward assigned")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(probe_cli.main(["--config", str(path)]), 1)
            self.assertEqual(output.getvalue(), "")

    def test_missing_service_config_fails_without_request(self):
        self.config["reward"]["metricx"]["service_url"] = ""
        with patch("flex_gspo.metricx_service.request_metricx_scores") as request:
            with self.assertRaisesRegex(RewardError, "service_url"):
                probe_cli.probe(self.config)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
