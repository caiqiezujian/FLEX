import unittest
from unittest.mock import patch

from flex_gspo.metricx import (ServiceMetricX, build_metricx, ensure_metricx_ready,
                               validate_metricx_settings)
from flex_gspo.metricx_service import request_metricx_scores
from flex_gspo.rewards import RewardError, build_scorer


class MetricXServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = {"backend": "service", "mode": "qe",
                         "service_url": "http://127.0.0.1:9010",
                         "service_id": "metricx24-qe-test-version"}

    def ready_scorer(self):
        return ServiceMetricX(self.settings)

    def test_empty_config_can_be_schema_validated_without_local_models(self):
        settings = validate_metricx_settings({"backend": "service"})
        self.assertEqual(settings["service_url"], "")
        self.assertEqual(settings["service_id"], "")
        self.assertNotIn("model_path", settings)

    def test_unconfigured_client_never_returns_a_fake_score(self):
        with self.assertRaisesRegex(RewardError, "service_url"):
            request_metricx_scores("English", ["中文"])

    def test_missing_service_details_rejected_when_implementation_ready(self):
        for key in ("service_url", "service_id"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(RewardError, key):
                    ServiceMetricX(dict(self.settings, **{key: ""}))

    def test_no_local_model_loaded_by_service_factory(self):
        with patch("builtins.__import__", wraps=__import__) as imports:
            scorer = build_metricx(self.settings)
            self.assertIsInstance(scorer, ServiceMetricX)
        self.assertFalse(any(call.args[0].split(".")[0] in ("torch", "transformers")
                             for call in imports.call_args_list))

    def test_single_text_returns_one_numeric_score(self):
        scorer = self.ready_scorer()
        with patch("flex_gspo.metricx_service.request_metricx_scores", return_value=[2.5]) as request:
            self.assertEqual(scorer.score("A meeting.", "一场会议。"), 2.5)
        self.assertEqual(request.call_args.args, ("A meeting.", ["一场会议。"], None))

    def test_batch_order_preserved_and_qe_never_receives_reference(self):
        scorer = self.ready_scorer()
        with patch("flex_gspo.metricx_service.request_metricx_scores", return_value=[3, 1]) as request:
            self.assertEqual(scorer.score_batch("English", ["甲", "乙"], "参考"), [3.0, 1.0])
        self.assertEqual(request.call_args.args, ("English", ["甲", "乙"], None))
        self.assertEqual(request.call_args.kwargs["mode"], "qe")

    def test_reference_mode_passes_reference(self):
        self.settings["mode"] = "reference"
        scorer = self.ready_scorer()
        with patch("flex_gspo.metricx_service.request_metricx_scores", return_value=[1]) as request:
            self.assertEqual(scorer.score("English", "译文", "参考"), 1.0)
        self.assertEqual(request.call_args.args[2], "参考")
        with self.assertRaises(RewardError):
            scorer.score("English", "译文")

    def test_invalid_numeric_scores_and_batch_lengths_rejected(self):
        scorer = self.ready_scorer()
        for output in ([True], [None], ["2"], [float("nan")], [float("inf")],
                       [-0.1], [25.1], [], [1, 2], {"score": 2}):
            with self.subTest(output=output), patch(
                    "flex_gspo.metricx_service.request_metricx_scores", return_value=output):
                with self.assertRaises(RewardError):
                    scorer.score("English", "中文")

    def test_service_failure_never_becomes_bad_policy_reward(self):
        scorer = self.ready_scorer()
        with patch("flex_gspo.metricx_service.request_metricx_scores",
                   side_effect=TimeoutError("SECRET response body")):
            with self.assertRaises(RewardError) as caught:
                scorer.score("English", "中文")
        self.assertIn("no reward assigned", str(caught.exception))
        self.assertNotIn("SECRET", str(caught.exception))

    def test_invalid_text_does_not_reach_hook(self):
        scorer = self.ready_scorer()
        with patch("flex_gspo.metricx_service.request_metricx_scores") as request:
            for source, candidates in (("", ["中文"]), ("English", [""]), ("English", "中文")):
                with self.assertRaises(RewardError):
                    scorer.score_batch(source, candidates)
            self.assertEqual(scorer.score_batch("English", []), [])
            request.assert_not_called()

    def test_url_cannot_embed_authentication_or_query_secrets(self):
        for url in ("ftp://host/score", "http://user:secret@host/score",
                    "http://host/score?token=secret", "http://host/#secret",
                    "http://host:bad/score", "http://bad host/score"):
            with self.subTest(url=url), self.assertRaises(RewardError):
                validate_metricx_settings(dict(self.settings, service_url=url))

    def test_qalco_dispatches_to_service(self):
        config = {"reward": {"provider": "qalco", "metricx": self.settings}}
        self.assertIsInstance(build_scorer(config).quality_model, ServiceMetricX)

    def test_training_stops_before_actor_or_cuda_initialization(self):
        from flex_gspo.trainer import run_training
        config = {"reward": {"provider": "qalco", "metricx": dict(self.settings, service_url="")}}
        with patch("flex_gspo.trainer._run_training") as expensive:
            with self.assertRaisesRegex(RewardError, "service_url"):
                run_training(config, [])
            expensive.assert_not_called()

    def test_unsupported_reward_rejected_before_expensive_work(self):
        with self.assertRaisesRegex(RewardError, "provider"):
            ensure_metricx_ready({"reward": {"provider": "unsupported_provider"}})


if __name__ == "__main__":
    unittest.main()
