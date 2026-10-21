import copy
import unittest
from unittest.mock import patch

from flex_gspo.metricx import ServiceMetricX
from flex_gspo.qalco import (QalcoScorer, combine_reward, emission_schedule,
                            latency_statistics, length_penalty, text_units,
                            validate_settings)
from flex_gspo.rewards import RewardError, build_scorer, validate_reward_setup


class FrozenQuality:
    def __init__(self, raw=2.0):
        self.raw = raw
        self.calls = []

    def score_batch(self, source, hypotheses, reference):
        self.calls.append((source, hypotheses, reference))
        return [self.raw] * len(hypotheses)


class QalcoTests(unittest.TestCase):
    def setUp(self):
        self.config = {"reward": {
            "provider": "qalco", "metricx": {
                "backend": "service", "mode": "qe",
                "service_url": "http://localhost:9010", "service_id": "metricx24-qe-test",
            },
            "quality_floor": .85, "latency_budget_seconds": 3.0,
            "latency_weight": .2, "length_weight": .1, "reference_length_ratio": 1.2,
        }}
        self.task = {"id": "one", "source_lang": "en", "target_lang": "zh",
                     "source_text": "We meet in Beijing.", "reference_text": "我们在北京开会。",
                     "duration_seconds": 4.0,
                     "chunks": [{"end_seconds": end} for end in (1., 2., 3., 4.)]}
        self.trajectory = {"valid": True, "complete_input": True, "turns": [
            {"response": "<|DIRECT|>我们", "audio_end_seconds": 1., "finish_reason": "stop"},
            {"response": "<|WAIT|>", "audio_end_seconds": 2., "finish_reason": "stop"},
            {"response": "<|FLEX|>在北京", "audio_end_seconds": 3., "finish_reason": "stop"},
            {"response": "<|DIRECT|>开会。", "audio_end_seconds": 4., "finish_reason": "eos"},
        ]}
        self.quality = FrozenQuality()
        self.scorer = QalcoScorer(self.config, quality_model=self.quality)

    def test_dispatch_and_preflight_are_lazy(self):
        with patch("flex_gspo.metricx_service.request_metricx_scores") as request:
            validate_reward_setup([self.task], self.config)
            scorer = build_scorer(self.config)
        self.assertIsInstance(scorer.quality_model, ServiceMetricX)
        request.assert_not_called()

    def test_semantic_whole_translation_and_deduplicated_quality(self):
        scores = self.scorer.score_group(self.task, [self.trajectory, copy.deepcopy(self.trajectory)])
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(self.quality.calls, [(self.task["source_text"], ["我们在北京开会。"],
                                               self.task["reference_text"])])
        self.assertAlmostEqual(scores[0]["components"]["quality"], .92)
        self.assertTrue(scores[0]["components"]["feasible"])

    def test_flex_direct_label_swap_reward_invariant(self):
        flipped = copy.deepcopy(self.trajectory)
        flipped["turns"][0]["response"] = "<|FLEX|>我们"
        self.assertEqual(self.scorer.score(self.task, self.trajectory),
                         self.scorer.score(self.task, flipped))

    def test_unit_tokenization_han_numbers_latin_and_punctuation(self):
        self.assertEqual([match.group() for match in text_units("AI模型 96，3.14 naïve！")],
                         ["AI", "模", "型", "96", "3.14", "naïve"])

    def test_split_word_completion_time(self):
        trajectory = copy.deepcopy(self.trajectory)
        for turn in trajectory["turns"]:
            turn["response"] = "<|WAIT|>"
        trajectory["turns"][0]["response"] = "<|FLEX|>inter"
        trajectory["turns"][2]["response"] = "<|DIRECT|>national"
        text, times = emission_schedule(self.task, trajectory["turns"])
        self.assertEqual(text, "international")
        self.assertEqual(times, [3.])

    def test_laal_exact_tau_excludes_rest_of_eof_batch(self):
        stats = latency_statistics(4., [1., 1., 3., 4., 4.], 4)
        self.assertEqual(stats["diagnostics"]["laal_tau"], 4)
        self.assertAlmostEqual(stats["raw_laal_seconds"], (1 + .2 + 1.4 + 1.6) / 4)
        self.assertEqual(stats["diagnostics"]["eof_text_fraction"], .4)

    def test_laal_uses_reference_when_longer(self):
        stats = latency_statistics(4., [1., 4.], 8)
        self.assertAlmostEqual(stats["raw_laal_seconds"], (1 + 3.5) / 2)

    def test_negative_laal_clamped_without_losing_raw(self):
        stats = latency_statistics(10., [1.] * 10, 10)
        self.assertLess(stats["raw_laal_seconds"], 0)
        self.assertEqual(stats["laal_seconds"], 0)

    def test_all_output_at_eof_is_not_zero_lag(self):
        stats = latency_statistics(30., [30.] * 100, 100)
        self.assertEqual(stats["laal_seconds"], 30)
        self.assertEqual(stats["diagnostics"]["laal_tau"], 1)

    def test_stall_diagnostic_includes_untranslated_tail_not_penalty(self):
        stats = latency_statistics(10., [1., 2.], 2)
        self.assertEqual(stats["diagnostics"]["max_no_text_gap_seconds"], 8)

    def test_length_penalty_has_no_shortness_bonus(self):
        self.assertEqual(length_penalty(1, 10, 1.2), 0)
        self.assertEqual(length_penalty(12, 10, 1.2), 0)
        self.assertAlmostEqual(length_penalty(18, 10, 1.2), .5)
        self.assertEqual(length_penalty(1000, 10, 1.2), 1)

    def test_three_tier_priority(self):
        settings = self.config["reward"]
        quality_bad = combine_reward(.849, 0, 0, settings)
        latency_bad = combine_reward(.99, 30, 0, settings)
        feasible = combine_reward(.85, 3, 1, settings)
        self.assertLess(quality_bad["reward"], latency_bad["reward"])
        self.assertLess(latency_bad["reward"], feasible["reward"])

    def test_worked_example_wait_only_if_it_improves(self):
        settings = self.config["reward"]
        a = combine_reward(.90, 1.6, .30, settings)["reward"]
        b = combine_reward(.93, 2.2, .05, settings)["reward"]
        c = combine_reward(.98, 6., 0., settings)["reward"]
        d = combine_reward(.90, 2.2, .30, settings)["reward"]
        self.assertGreater(b, a)
        self.assertGreater(a, d)
        self.assertGreater(d, c)
        self.assertAlmostEqual(b, .8294871794871794)

    def test_invalid_protocol_does_not_call_quality(self):
        self.trajectory["turns"][1]["response"] = "<|WAIT|>不要输出"
        score = self.scorer.score(self.task, self.trajectory)
        self.assertEqual(score["reward"], -4)
        self.assertFalse(score["components"]["protocol_valid"])
        self.assertEqual(self.quality.calls, [])

    def test_empty_translation_not_skipped_from_reward(self):
        for turn in self.trajectory["turns"]:
            turn["response"] = "<|WAIT|>"
        score = self.scorer.score(self.task, self.trajectory)
        self.assertEqual(score["reward"], -4)
        self.assertTrue(score["components"]["empty_translation"])
        self.assertTrue(score["components"]["protocol_valid"])
        self.assertEqual(self.quality.calls, [])

    def test_truncation_is_invalid(self):
        self.trajectory["turns"][-1]["finish_reason"] = "length"
        self.assertEqual(self.scorer.score(self.task, self.trajectory)["reward"], -4)

    def test_missing_reference_is_not_bad_policy(self):
        self.task["reference_text"] = None
        with self.assertRaises(RewardError):
            self.scorer.score(self.task, self.trajectory)

    def test_corrupt_timing_is_not_bad_policy(self):
        self.trajectory["turns"][0]["audio_end_seconds"] = 3.
        with self.assertRaises(RewardError):
            self.scorer.score(self.task, self.trajectory)

    def test_corrupt_model_score_is_not_bad_policy(self):
        for raw in (float("nan"), True, -1, 26):
            self.quality.raw = raw
            with self.subTest(raw=raw), self.assertRaises(RewardError):
                self.scorer.score(self.task, self.trajectory)

    def test_model_failure_propagates(self):
        def failure(*unused):
            raise RewardError("OOM")
        self.quality.score_batch = failure
        with self.assertRaisesRegex(RewardError, "OOM"):
            self.scorer.score(self.task, self.trajectory)

    def test_nonfinite_configuration_rejected(self):
        for name, value in (("quality_floor", 0), ("quality_floor", float("nan")),
                            ("latency_budget_seconds", 0), ("reference_length_ratio", .5),
                            ("latency_weight", True), ("length_weight", float("inf"))):
            settings = copy.deepcopy(self.config)
            settings["reward"][name] = value
            with self.subTest(name=name), self.assertRaises(RewardError):
                validate_settings(settings)

    def test_only_current_reward_fields_are_accepted(self):
        for key in ("unknown_option", "quality_floo", "extra_reward"):
            settings = copy.deepcopy(self.config)
            settings["reward"][key] = 0.5
            with self.subTest(key=key), self.assertRaisesRegex(RewardError, "Unsupported QALCO setting"):
                validate_settings(settings)

    def test_legacy_calibration_confirmation_is_rejected(self):
        for value in (True, False, "true", 1, None):
            settings = copy.deepcopy(self.config)
            settings["reward"]["calibration_acknowledged"] = value
            with self.subTest(value=value), self.assertRaisesRegex(RewardError, "Unsupported QALCO setting"):
                validate_settings(settings)

    def test_default_latency_is_monotonic_mean_lag(self):
        settings = validate_settings(self.config)
        self.assertEqual(settings["latency_metric"], "mean_lag")
        score = self.scorer.score(self.task, self.trajectory)
        self.assertEqual(score["components"]["latency_metric"], "source_clock_mean_positive_lag")

    def test_speech_laal_comparison_requires_explicit_selection(self):
        config = copy.deepcopy(self.config)
        config["reward"]["latency_metric"] = "speech_laal"
        scorer = QalcoScorer(config, quality_model=self.quality)
        components = scorer.score(self.task, self.trajectory)["components"]
        self.assertEqual(components["latency_metric"], "utterance_speech_laal")
        self.assertEqual(components["latency_seconds"], components["laal_seconds"])

    def test_mock_candidate_rejected(self):
        self.trajectory["mock"] = True
        with self.assertRaises(RewardError):
            self.scorer.score(self.task, self.trajectory)

    def test_source_translation_not_used_as_prefix_support_evidence(self):
        score = self.scorer.score(self.task, self.trajectory)
        self.assertFalse(score["components"]["prefix_support_checked"])


if __name__ == "__main__":
    unittest.main()
