"""HTTP contract tests with mocked sessions; never contact the private endpoint."""
import unittest
from unittest.mock import MagicMock, patch

import requests

from flex_gspo.metricx_service import (ensure_service_ready,
                                      request_metricx_scores)
from flex_gspo.rewards import RewardError


class MetricXHTTPTests(unittest.TestCase):
    def setUp(self):
        self.session_patch = patch("flex_gspo.metricx_service.requests.Session")
        self.factory = self.session_patch.start()
        self.addCleanup(self.session_patch.stop)
        self.session = self.factory.return_value.__enter__.return_value
        self.sleep_patch = patch("flex_gspo.metricx_service.time.sleep")
        self.sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)
        self.settings = {"service_url": "http://127.0.0.1:16999/",
                         "service_id": "operator-declared-test-deployment"}

    @staticmethod
    def response(status=200, body=None):
        response = MagicMock()
        response.status_code = status
        response.json.return_value = {"scores": [0.4, 0.1]} if body is None else body
        response.text = "SECRET BODY"
        return response

    def score(self, **kwargs):
        return request_metricx_scores(" A source. ", ["译文甲。", "译文乙。"],
                                      **dict(self.settings, **kwargs))

    def test_qe_batch_payload_endpoint_order_and_no_proxy(self):
        response = self.response()
        self.session.post.return_value = response
        self.assertEqual(self.score(reference_text="不应该发送"), [0.4, 0.1])
        self.session.post.assert_called_once_with(
            "http://127.0.0.1:16999/metricx24/qe",
            json={"srcs": [" A source. "] * 2, "mts": ["译文甲。", "译文乙。"]},
            timeout=(10, 180), allow_redirects=False)
        self.assertIs(self.session.trust_env, False)
        response.close.assert_called_once()
        self.sleep.assert_not_called()

    def test_reference_batch_includes_matching_reference_positions(self):
        self.session.post.return_value = self.response(body={"scores": [25, 0]})
        self.assertEqual(self.score(mode="reference", reference_text=" 参考。 "), [25.0, 0.0])
        self.session.post.assert_called_once_with(
            "http://127.0.0.1:16999/metricx24",
            json={"srcs": [" A source. "] * 2, "mts": ["译文甲。", "译文乙。"],
                  "refs": [" 参考。 "] * 2}, timeout=(10, 180), allow_redirects=False)

    def test_readiness_only_validates_config_and_dependency(self):
        ensure_service_ready(dict(self.settings, backend="service", mode="qe"))
        self.factory.assert_not_called()
        with patch("flex_gspo.metricx_service.requests", None):
            with self.assertRaisesRegex(RewardError, "requests"):
                ensure_service_ready(dict(self.settings, backend="service", mode="qe"))

    def test_bad_text_or_configuration_never_sends_request(self):
        for source, candidates, extra in (
                ("", ["译文"], {}), ("English", "译文", {}),
                ("English", [""], {}), ("English", [None], {}),
                ("English", ["译文"], {"mode": "reference"}),
                ("English", ["译文"], {"mode": "wrong"}),
                ("English", ["译文"], {"service_url": ""}),
                ("English", ["译文"], {"service_id": ""})):
            with self.subTest(source=source, candidates=candidates, extra=extra):
                with self.assertRaises(RewardError):
                    request_metricx_scores(source, candidates, **dict(self.settings, **extra))
        self.factory.assert_not_called()

    def test_empty_candidate_batch_makes_no_request(self):
        self.assertEqual(request_metricx_scores("English", [], **self.settings), [])
        self.factory.assert_not_called()

    def test_invalid_json_shape_score_or_count_is_not_retried(self):
        invalid_bodies = [[], {"score": 1}, {"scores": "1"}, {"scores": []},
                          {"scores": [1]}, {"scores": [1, 2, 3]}]
        invalid_bodies += [{"scores": [value, 1]} for value in
                           (True, None, "2", float("nan"), float("inf"), -0.1, 25.1)]
        for body in invalid_bodies:
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

    def test_malformed_json_never_leaks_body_or_retries(self):
        response = self.response()
        response.json.side_effect = ValueError("SECRET JSON BODY")
        self.session.post.return_value = response
        with self.assertRaises(RewardError) as caught:
            self.score()
        self.assertNotIn("SECRET", str(caught.exception))
        self.session.post.assert_called_once()
        response.close.assert_called_once()
        self.sleep.assert_not_called()

    def test_retryable_statuses_retry_once_then_preserve_scores(self):
        for status in (429, 502, 503, 504):
            with self.subTest(status=status):
                self.session.post.reset_mock()
                self.sleep.reset_mock()
                first, second = self.response(status), self.response()
                self.session.post.side_effect = [first, second]
                self.assertEqual(self.score(), [0.4, 0.1])
                self.assertEqual(self.session.post.call_count, 2)
                self.assertEqual(self.session.post.call_args_list[0], self.session.post.call_args_list[1])
                self.sleep.assert_called_once_with(1.0)
                first.close.assert_called_once()
                second.close.assert_called_once()

    def test_retryable_transport_errors_retry_once(self):
        for error in (requests.Timeout, requests.ConnectionError):
            with self.subTest(error=error):
                self.session.post.reset_mock()
                self.sleep.reset_mock()
                self.session.post.side_effect = [error("SECRET URL"), self.response()]
                with self.assertLogs("flex.metricx", level="WARNING") as logs:
                    self.assertEqual(self.score(), [0.4, 0.1])
                self.assertEqual(self.session.post.call_count, 2)
                self.sleep.assert_called_once_with(1.0)
                self.assertNotIn("SECRET", " ".join(logs.output))

    def test_exhausted_transport_retries_stop_without_fake_scores(self):
        self.session.post.side_effect = requests.Timeout("SECRET URL")
        with self.assertLogs("flex.metricx", level="WARNING") as logs:
            with self.assertRaises(RewardError) as caught:
                self.score()
        self.assertEqual(self.session.post.call_count, 2)
        self.sleep.assert_called_once_with(1.0)
        self.assertIn("no reward assigned", str(caught.exception))
        self.assertNotIn("SECRET", str(caught.exception) + " ".join(logs.output))

    def test_exhausted_status_retries_stop_and_close_responses(self):
        first, second = self.response(503), self.response(503)
        self.session.post.side_effect = [first, second]
        with self.assertRaises(RewardError) as caught:
            self.score()
        self.assertEqual(self.session.post.call_count, 2)
        self.assertIn("503", str(caught.exception))
        self.assertNotIn("SECRET", str(caught.exception))
        self.sleep.assert_called_once_with(1.0)
        first.close.assert_called_once()
        second.close.assert_called_once()

    def test_client_errors_redirects_and_other_statuses_do_not_retry(self):
        for status in (201, 301, 302, 307, 400, 401, 403, 404, 422, 500):
            with self.subTest(status=status):
                self.session.post.reset_mock()
                response = self.response(status)
                self.session.post.return_value = response
                with self.assertRaises(RewardError) as caught:
                    self.score()
                self.assertIn(str(status), str(caught.exception))
                self.assertNotIn("SECRET", str(caught.exception))
                self.session.post.assert_called_once()
                response.close.assert_called_once()
        self.sleep.assert_not_called()

    def test_other_request_errors_are_sanitized_without_retry(self):
        self.session.post.side_effect = requests.RequestException("SECRET URL BODY")
        with self.assertRaises(RewardError) as caught:
            self.score()
        self.assertNotIn("SECRET", str(caught.exception))
        self.session.post.assert_called_once()
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
