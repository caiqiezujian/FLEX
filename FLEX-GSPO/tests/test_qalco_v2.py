"""Independent regression tests for COMET22/EOF QALCO v2 contracts."""
import copy
import math
import unittest

from flex_gspo.online_reward import load_reward_config
from flex_gspo.qalco import QalcoScorer, combine_reward, validate_settings
from flex_gspo.rewards import RewardError


class FixedQuality:
    def __init__(self, value=0.9, *, failure=None, malformed_batch=False):
        self.value = value
        self.failure = failure
        self.malformed_batch = malformed_batch
        self.calls = []

    def score_batch(self, source, hypotheses, reference):
        self.calls.append((source, hypotheses, reference))
        if self.failure:
            raise self.failure
        if self.malformed_batch:
            return None
        return [self.value for _ in hypotheses]


class QalcoV2Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_reward_config()
        self.settings = validate_settings(self.config)
        self.task = {
            "id": "v2-synthetic-meeting", "source_lang": "en", "target_lang": "zh",
            "source_text": "We are meeting in Beijing to discuss our annual budget.",
            "reference_text": "我们在北京开会，讨论年度预算。",
            "duration_seconds": 4.0,
            "chunks": [{"end_seconds": float(i)} for i in range(1, 5)],
        }
        self.quality = FixedQuality()
        self.scorer = QalcoScorer(self.config, quality_model=self.quality)

    @staticmethod
    def trajectory(audio=None, eof="<|WAIT|>"):
        if audio is None:
            audio = ["<|DIRECT|>我们", "<|FLEX|>在北京开会，",
                     "<|DIRECT|>讨论年度", "<|DIRECT|>预算。"]
        rows = [
            {"kind": "audio", "response": response,
             "audio_end_seconds": float(i + 1), "finish_reason": "stop"}
            for i, response in enumerate(audio)
        ]
        rows.append({"kind": "eof", "response": eof, "audio_end_seconds": 4.0,
                     "finish_reason": "stop", "eof_elapsed_seconds": 0.3})
        return {"schema_version": 2, "valid": True, "complete_input": True,
                "episode_complete": True, "eof_completed": True, "turns": rows}

    def test_default_is_comet22_v2_and_uses_raw_higher_better_quality(self):
        self.assertEqual(self.settings["version"], 2)
        self.assertEqual(self.settings["quality_provider"], "comet22")
        result = self.scorer.score(self.task, self.trajectory())
        self.assertEqual(result["components"]["quality"], 0.9)
        self.assertEqual(result["components"]["comet22_score"], 0.9)
        self.assertIsNone(result["components"]["metricx_error"])
        self.assertEqual(result["components"]["quality_metric"], "comet22_reference")
        self.assertEqual(self.quality.calls[0][1], ["我们在北京开会，讨论年度预算。"])

    def test_v2_rejects_tau_truncated_laal_as_training_objective(self):
        changed = copy.deepcopy(self.config)
        changed["reward"]["latency_metric"] = "speech_laal"
        with self.assertRaises(RewardError):
            validate_settings(changed)

    def test_v2_rejects_missing_or_other_trajectory_schema_even_when_invalid(self):
        for schema in (None, 1, 3, "2", True):
            for valid in (True, False):
                with self.subTest(schema=schema, valid=valid):
                    tr = self.trajectory()
                    tr["valid"] = valid
                    if schema is None:
                        tr.pop("schema_version")
                    else:
                        tr["schema_version"] = schema
                    with self.assertRaises(RewardError):
                        self.scorer.score(self.task, tr)

    def test_eof_wait_after_valid_translation_is_not_empty_failure(self):
        result = self.scorer.score(self.task, self.trajectory())
        self.assertFalse(result["components"]["empty_translation"])
        self.assertTrue(result["components"]["protocol_valid"])
        self.assertNotEqual(result["reward"], -4)
        self.assertFalse(result["components"]["diagnostics"]["eof_turn_text"])
        self.assertEqual(result["components"]["diagnostics"]["eof_generation_seconds"], 0.3)

    def test_all_wait_including_eof_is_failure_and_does_not_call_quality(self):
        result = self.scorer.score(self.task, self.trajectory(["<|WAIT|>"] * 4))
        self.assertEqual(result["reward"], -4)
        self.assertTrue(result["components"]["empty_translation"])
        self.quality.calls.clear()
        result = self.scorer.score(self.task, self.trajectory(["<|WAIT|>"] * 4))
        self.assertEqual(self.quality.calls, [])

    def test_finalization_can_recover_missing_short_translation(self):
        result = self.scorer.score(self.task, self.trajectory(
            ["<|WAIT|>"] * 4, "<|DIRECT|>我们在北京开会，讨论年度预算。"))
        self.assertNotEqual(result["reward"], -4)
        self.assertFalse(result["components"]["empty_translation"])
        self.assertTrue(result["components"]["diagnostics"]["eof_turn_text"])
        self.assertEqual(result["components"]["diagnostics"]["eof_text_fraction"], 1)

    def test_identical_translation_delayed_to_eof_costs_more(self):
        text = "我们在北京开会，讨论年度预算。"
        before = self.trajectory(["<|WAIT|>"] * 3 + ["<|DIRECT|>" + text])
        after = self.trajectory(["<|WAIT|>"] * 4, "<|DIRECT|>" + text)
        scored = self.scorer.score_group(self.task, [before, after])
        self.assertEqual(self.quality.calls[0][1], [text])
        self.assertGreater(scored[1]["components"]["latency_seconds"],
                           scored[0]["components"]["latency_seconds"])
        self.assertLess(scored[1]["reward"], scored[0]["reward"])

    def test_perfect_learned_quality_cannot_rescue_severe_loop(self):
        self.quality.value = 1.0
        repeated = self.trajectory(["<|DIRECT|>" + "我喜欢它们。" * 13] + ["<|WAIT|>"] * 3)
        result = self.scorer.score(self.task, repeated)
        self.assertEqual(result["reward"], -4)
        self.assertTrue(result["components"]["severe_repetition"])
        self.assertEqual(result["components"]["tier"], "degenerate")
        self.assertTrue(result["components"]["protocol_valid"])
        self.assertEqual(self.quality.calls, [])

    def test_ordinary_flex_anchors_still_get_quality_scored(self):
        tr = self.trajectory([
            "<|DIRECT|>需求增加了，", "<|FLEX|>增长来自海外市场，",
            "<|DIRECT|>增长速度有所放缓，", "<|FLEX|>但增长仍在继续。"])
        result = self.scorer.score(self.task, tr)
        self.assertFalse(result["components"]["severe_repetition"])
        self.assertEqual(len(self.quality.calls), 1)

    def test_wait_and_flex_do_not_receive_separate_tag_bonuses(self):
        original = self.trajectory()
        changed = copy.deepcopy(original)
        for row in changed["turns"]:
            row["response"] = row["response"].replace("<|FLEX|>", "<|DIRECT|>")
        self.assertEqual(self.scorer.score(self.task, original),
                         self.scorer.score(self.task, changed))

    def test_missing_finalization_is_not_silently_treated_as_completed(self):
        tr = self.trajectory()
        tr["turns"].pop()
        tr["episode_complete"] = False
        tr["eof_completed"] = False
        result = self.scorer.score(self.task, tr)
        self.assertEqual(result["reward"], -4)
        self.assertEqual(self.quality.calls, [])

    def test_truncated_generation_is_not_a_good_policy_candidate(self):
        tr = self.trajectory()
        tr["turns"][-1]["finish_reason"] = "length"
        self.assertEqual(self.scorer.score(self.task, tr)["reward"], -4)
        self.assertEqual(self.quality.calls, [])

    def test_scheduler_timing_corruption_raises_instead_of_assigning_bad_reward(self):
        tr = self.trajectory()
        tr["turns"][0]["audio_end_seconds"] = 3.0
        with self.assertRaises(RewardError):
            self.scorer.score(self.task, tr)

    def test_negative_eof_elapsed_is_infrastructure_failure(self):
        tr = self.trajectory()
        tr["turns"][-1]["eof_elapsed_seconds"] = -1
        with self.assertRaises(RewardError):
            self.scorer.score(self.task, tr)

    def test_invalid_policy_does_not_hide_corrupt_timing(self):
        tr = self.trajectory()
        tr["valid"] = False
        tr["turns"][0]["audio_end_seconds"] = 3.0
        with self.assertRaises(RewardError):
            self.scorer.score(self.task, tr)

    def test_completed_eof_requires_consistent_completion_flags(self):
        for flag in ("complete_input", "episode_complete", "eof_completed"):
            with self.subTest(flag=flag):
                tr = self.trajectory()
                tr[flag] = False
                with self.assertRaises(RewardError):
                    self.scorer.score(self.task, tr)

    def test_invalid_quality_values_fail_closed_not_fake_policy_rewards(self):
        for raw in (None, math.nan, math.inf, -0.1, 1.1, True, "0.9"):
            with self.subTest(raw=raw):
                scorer = QalcoScorer(self.config, quality_model=FixedQuality(raw))
                with self.assertRaises(RewardError):
                    scorer.score(self.task, self.trajectory())

    def test_missing_quality_batch_is_infrastructure_failure(self):
        scorer = QalcoScorer(self.config, quality_model=FixedQuality(malformed_batch=True))
        with self.assertRaises(RewardError):
            scorer.score(self.task, self.trajectory())

    def test_service_failure_propagates_without_reward_fallback(self):
        scorer = QalcoScorer(self.config, quality_model=FixedQuality(
            failure=RewardError("synthetic service failure")))
        with self.assertRaisesRegex(RewardError, "synthetic service failure"):
            scorer.score(self.task, self.trajectory())

    def test_below_quality_floor_waiting_still_has_cost(self):
        quality = self.settings["quality_floor"] / 2
        early = combine_reward(quality, 0.5, 0, self.settings)
        late = combine_reward(quality, 15, 0, self.settings)
        self.assertEqual(early["tier"], "quality_fail")
        self.assertEqual(late["tier"], "quality_fail")
        self.assertLess(late["reward"], early["reward"])

    def test_late_quality_pass_has_no_previous_six_second_clamp(self):
        values = [combine_reward(0.9, seconds, 0, self.settings)["reward"]
                  for seconds in (6.0, 9.0, 12.0, 30.0)]
        self.assertTrue(all(right < left for left, right in zip(values, values[1:])))

    def test_late_candidate_keeps_quality_and_length_discrimination(self):
        low_quality = combine_reward(0.85, 12, 0, self.settings)["reward"]
        high_quality = combine_reward(0.95, 12, 0, self.settings)["reward"]
        verbose = combine_reward(0.95, 12, 1, self.settings)["reward"]
        self.assertGreater(high_quality, low_quality)
        self.assertLess(verbose, high_quality)

    def test_tier_ordering_is_preserved_over_a_numeric_grid(self):
        groups = {"quality_fail": [], "latency_fail": [], "feasible": []}
        floor = self.settings["quality_floor"]
        for quality in (0, floor / 2, floor - 1e-9, floor, 0.9, 1):
            for delay in (0, 0.5, 3, 3.000001, 6, 1000, 1e12):
                for excess in (0, 0.5, 1):
                    result = combine_reward(quality, delay, excess, self.settings)
                    self.assertTrue(math.isfinite(result["reward"]))
                    groups[result["tier"]].append(result["reward"])
        self.assertGreater(min(groups["quality_fail"]), -4)
        self.assertLess(max(groups["quality_fail"]), min(groups["latency_fail"]))
        self.assertLess(max(groups["latency_fail"]), min(groups["feasible"]))


if __name__ == "__main__":
    unittest.main()
