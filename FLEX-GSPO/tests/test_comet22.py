"""COMET22 strict transport contract; private services are never contacted."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

import check_comet22_service as probe_cli
from flex_gspo.comet22 import ServiceCOMET22, ensure_comet22_ready, validate_comet22_settings
from flex_gspo.rewards import RewardError


class COMET22Tests(unittest.TestCase):
    def setUp(self):
        self.session_patch = patch("flex_gspo.comet22.requests.Session")
        self.factory = self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.session = self.factory.return_value.__enter__.return_value
        self.sleep_patch = patch("flex_gspo.comet22.time.sleep")
        self.sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)
        self.settings = {"service_url": "http://127.0.0.1:6668/22", "service_id": "test-only"}

    @staticmethod
    def response(status=200, body=None):
        result = MagicMock()
        result.status_code = status
        result.json.return_value = {"scores": [0.9, 0.4]} if body is None else body
        result.text = "SECRET BODY"
        return result

    def score(self, **kwargs):
        return ServiceCOMET22(dict(self.settings, **kwargs)).score_batch(" Source. ", ["译文甲。", "译文乙。"], "参考。")

    def test_exact_endpoint_payload_reference_and_proxy(self):
        response = self.response()
        self.session.post.return_value = response
        self.assertEqual(self.score(), [0.9, 0.4])
        self.session.post.assert_called_once_with(
            "http://127.0.0.1:6668/22", json={"srcs": [" Source. "] * 2,
                "mts": ["译文甲。", "译文乙。"], "refs": ["参考。"] * 2},
            timeout=(10, 300), allow_redirects=False)
        self.assertIs(self.session.trust_env, False)
        response.close.assert_called_once()

    def test_flat_scores_and_explicit_custom_key(self):
        for body, config in (([0.2, 0.7], {}), ({"segment_scores": [0.2, 0.7]}, {"response_key": "segment_scores"})):
            with self.subTest(body=body):
                self.session.post.return_value = self.response(body=body)
                self.assertEqual(self.score(**config), [0.2, 0.7])

    def test_deployed_score_key_with_production_config(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default.json"
        settings = json.loads(config_path.read_text(encoding="utf-8"))["reward"]["comet22"]
        self.assertEqual(settings["response_key"], "score")
        scores = [0.9610479474067688, 0.86544269323349]
        self.session.post.return_value = self.response(body={"score": scores})
        scorer = ServiceCOMET22(settings)
        self.assertEqual(scorer.score_batch("Source", ["译文甲", "译文乙"], "参考"), scores)
        self.session.post.assert_called_once()

    def test_score_key_still_rejects_aggregates_bad_values_and_wrong_count(self):
        bodies = [{"score": 0.9}, {"score": [0.9]}, {"score": [0.9, 0.4, 0.2]},
                  {"score": [[0.9, 0.4]]}, {"scores": [0.9, 0.4]}]
        bodies += [{"score": [value, 0.1]} for value in
                   (True, None, "0.9", float("nan"), float("inf"), -0.1, 90, 1.01)]
        for body in bodies:
            with self.subTest(body=body):
                self.session.post.return_value = self.response(body=body)
                with self.assertRaisesRegex(RewardError, "no reward assigned"):
                    self.score(response_key="score")

    def test_explicit_score_key_does_not_guess_other_fields(self):
        self.session.post.return_value = self.response(
            body={"score": [0.9, 0.4], "scores": [0.1, 0.2], "system_score": 0.65})
        self.assertEqual(self.score(response_key="score"), [0.9, 0.4])

    def test_aggregate_not_mistaken_for_segment_scores(self):
        bodies = [0.9, {"system_score": 0.9}, {"scores": [[0.9, 0.4], 0.65]},
                  {"scores": [0.9]}, {"scores": [0.9, 0.4, 0.2]}, {"score": [0.9, 0.4]}]
        bodies += [{"scores": [value, 0.1]} for value in
                   (True, None, "0.9", float("nan"), float("inf"), -0.1, 90, 1.01)]
        for body in bodies:
            with self.subTest(body=body):
                self.session.post.reset_mock()
                response = self.response(body=body)
                self.session.post.return_value = response
                with self.assertRaises(RewardError) as caught:
                    self.score()
                self.assertIn("no reward assigned", str(caught.exception))
                self.assertNotIn("SECRET", str(caught.exception))
                self.session.post.assert_called_once()
                response.close.assert_called_once()
        self.sleep.assert_not_called()

    def test_truncation_or_reported_error_fails_closed(self):
        for metadata in ({"truncated": True}, {"truncated": [False, True]},
                         {"truncated": [False]}, {"truncated": "false"}, {"error": "SECRET"}):
            with self.subTest(metadata=metadata):
                self.session.post.return_value = self.response(body={"scores": [0.9, 0.4], **metadata})
                with self.assertRaises(RewardError) as caught:
                    self.score()
                self.assertNotIn("SECRET", str(caught.exception))
        self.session.post.return_value = self.response(body={"scores": [0.9, 0.4], "truncated": [False, False]})
        self.assertEqual(self.score(), [0.9, 0.4])

    def test_config_readiness_has_no_network(self):
        settings = ensure_comet22_ready(self.settings)
        self.assertEqual(settings["model_id"], "Unbabel/wmt22-comet-da")
        self.factory.assert_not_called()
        with patch("flex_gspo.comet22.requests", None):
            with self.assertRaisesRegex(RewardError, "requests"):
                ensure_comet22_ready(self.settings)

    def test_bad_settings_and_text_fail_before_request(self):
        bad = [{"service_url": ""}, {"service_url": "ftp://host"},
               {"service_url": "https://user:password@host/22"}, {"service_url": "http://host/22?q=secret"},
               {"response_key": "system_score"}, {"max_attempts": True}, {"max_attempts": 4},
               {"timeout_seconds": float("nan")}, {"timeout_seconds": 0}, {"unexpected": 1}]
        for override in bad:
            with self.subTest(override=override):
                with self.assertRaises(RewardError):
                    validate_comet22_settings(dict(self.settings, **override))
        for source, candidates, reference in (("", ["中文"], "参考"), ("source", [""], "参考"),
                                               ("source", ["中文"], ""), ("source", "中文", "参考")):
            with self.assertRaises(RewardError):
                ServiceCOMET22(self.settings).score_batch(source, candidates, reference)
        self.assertEqual(ServiceCOMET22(self.settings).score_batch("source", [], "参考"), [])
        self.factory.assert_not_called()

    def test_transient_http_and_transport_retry_once(self):
        first = self.response(503)
        self.session.post.side_effect = [first, self.response()]
        self.assertEqual(self.score(), [0.9, 0.4])
        first.close.assert_called_once()
        self.assertEqual(self.session.post.call_count, 2)
        self.sleep.assert_called_once_with(1.0)
        self.session.post.reset_mock()
        self.session.post.side_effect = [requests.ConnectionError("SECRET"), self.response()]
        self.assertEqual(self.score(), [0.9, 0.4])
        self.assertEqual(self.session.post.call_count, 2)

    def test_exhausted_retries_and_client_error_do_not_leak(self):
        self.session.post.side_effect = requests.Timeout("SECRET")
        with self.assertRaises(RewardError) as caught:
            self.score()
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertEqual(self.session.post.call_count, 2)
        self.session.post.reset_mock()
        self.session.post.side_effect = None
        self.session.post.return_value = self.response(422)
        with self.assertRaisesRegex(RewardError, "HTTP 422"):
            self.score()
        self.session.post.assert_called_once()

    def test_invalid_json_not_retried(self):
        response = self.response()
        response.json.side_effect = ValueError("SECRET")
        self.session.post.return_value = response
        with self.assertRaises(RewardError) as caught:
            self.score()
        self.assertNotIn("SECRET", str(caught.exception))
        self.session.post.assert_called_once()
        response.close.assert_called_once()


class COMET22ProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = {"reward": {"comet22": {"service_url": "http://127.0.0.1:6668/22"}}}
        self.case = {"id": "test", "source_text": "source", "reference_text": "参考",
                     "preferred": "好的译文", "dispreferred": ["错误译文", "重复译文"]}

    def test_success_has_separate_connectivity_quality_identity(self):
        scorer = MagicMock()
        scorer.score_batch.side_effect = [[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]
        result = probe_cli.probe(self.config, [self.case], scorer=scorer)
        self.assertTrue(result["service_connectivity_verified"])
        self.assertTrue(result["quality_regression_passed"])
        self.assertTrue(result["batch_order_consistency_passed"])
        self.assertFalse(result["remote_model_identity_verified"])
        self.assertFalse(result["remote_truncation_behavior_verified"])

    def test_reversed_order_failure_is_not_hidden(self):
        scorer = MagicMock()
        scorer.score_batch.side_effect = [[0.9, 0.5, 0.1], [0.9, 0.5, 0.1]]
        result = probe_cli.probe(self.config, [self.case], scorer=scorer)
        self.assertFalse(result["batch_order_consistency_passed"])

    def test_repetition_ranking_can_be_diagnostic_only(self):
        scorer = MagicMock()
        scorer.score_batch.side_effect = [[0.8, 0.2, 0.9], [0.9, 0.2, 0.8]]
        case = dict(self.case, required_dispreferred_indices=[0])
        result = probe_cli.probe(self.config, [case], scorer=scorer)
        self.assertTrue(result["quality_regression_passed"])
        self.assertFalse(result["results"][0]["all_paired_rankings_passed"])

    def test_required_quality_failure_cli_returns_nonzero_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            result = {"quality_regression_passed": False, "batch_order_consistency_passed": True}
            with patch("check_comet22_service.probe", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe_cli.main(["--config", str(config_path)]), 2)
                self.assertEqual(probe_cli.main(["--config", str(config_path), "--transport-only"]), 0)

    def test_invalid_case_ids_and_indices(self):
        for case in (dict(self.case, id=""), dict(self.case, required_dispreferred_indices=[]),
                     dict(self.case, required_dispreferred_indices=[True]), dict(self.case, required_dispreferred_indices=[2])):
            with self.assertRaises(RewardError):
                probe_cli.validate_cases([case])


if __name__ == "__main__":
    unittest.main()
