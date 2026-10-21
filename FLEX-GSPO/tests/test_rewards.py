"""Reward dispatch tests; detailed numerical behavior lives in test_qalco.py."""
import unittest
from unittest.mock import patch

from flex_gspo.rewards import RewardError, build_scorer, validate_reward_setup


class RewardDispatchTests(unittest.TestCase):
    def setUp(self):
        self.config = {"reward": {"provider": "qalco", "metricx": {"backend": "service"}}}

    def test_only_current_provider_is_allowed(self):
        for config in (None, [], {}, {"reward": None},
                       {"reward": {"provider": "unsupported_provider"}},
                       {"reward": {"provider": "mock"}}):
            with self.subTest(config=config):
                with self.assertRaises(RewardError):
                    build_scorer(config)
                with self.assertRaises(RewardError):
                    validate_reward_setup([], config)

    def test_preflight_delegates_without_calling_service(self):
        tasks = [{"id": "example"}]
        with patch("flex_gspo.qalco.validate_tasks", return_value={"checked": True}) as validate, \
             patch("flex_gspo.metricx_service.request_metricx_scores") as request:
            result = validate_reward_setup(tasks, self.config)
        self.assertEqual(result, {"checked": True})
        validate.assert_called_once_with(tasks, self.config)
        request.assert_not_called()

    def test_factory_has_no_alternative_or_fallback(self):
        with patch("flex_gspo.qalco.QalcoScorer", side_effect=RewardError("service unavailable")) as factory:
            with self.assertRaisesRegex(RewardError, "service unavailable"):
                build_scorer(self.config)
        factory.assert_called_once_with(self.config)

    def test_factory_accepts_reward_subsection(self):
        with patch("flex_gspo.qalco.QalcoScorer") as factory:
            result = build_scorer(self.config["reward"])
        self.assertIs(result, factory.return_value)


if __name__ == "__main__":
    unittest.main()
