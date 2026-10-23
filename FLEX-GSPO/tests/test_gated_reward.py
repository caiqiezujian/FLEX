"""Numerical and failure-boundary tests for the pure-Python v3 reward."""
import copy
import unittest

from flex_gspo.gated_reward import DEFAULT_V3, combine_v3, finalize_group, validate_v3_settings
from flex_gspo.rewards import RewardError


def item(quality=0.9, latency=1.0, units=30, repetition=0.0, eligible=True, **extra):
    return dict(eligible=eligible, quality=quality, latency=latency, units=units,
                repetition=repetition, reward=0.0, **extra)


class GatedRewardTests(unittest.TestCase):
    def setUp(self):
        self.settings = validate_v3_settings({"quality_floor": 0.8, "latency_budget_seconds": 3.0})

    def test_defaults_complete_without_mutation(self):
        original = {"quality_floor": 0.8, "comet22": {"response_key": "score"}, "v3": {}}
        before = copy.deepcopy(original)
        result = validate_v3_settings(original)
        self.assertEqual(original, before)
        self.assertEqual(result["v3"], DEFAULT_V3)
        self.assertEqual(result["latency_budget_seconds"], 3.0)
        result["v3"]["quality_gate_low"] = 0.1
        self.assertEqual(DEFAULT_V3["quality_gate_low"], 0.75)
        result["comet22"]["response_key"] = "changed"
        self.assertEqual(original["comet22"]["response_key"], "score")

    def test_bad_settings_fail_explicitly(self):
        bad = [None, [], {"v3": None}, {"v3": {"typo": 1}},
               {"quality_floor": True}, {"quality_floor": 0},
               {"latency_budget_seconds": 0}, {"latency_budget_seconds": float("inf")},
               {"v3": {"anchor_min_candidates": 2}}, {"v3": {"anchor_min_candidates": 3.0}},
               {"v3": {"quality_gate_low": 0.85}}, {"v3": {"quality_gate_high": 0.7}},
               {"v3": {"anchor_quantile": 1.1}}, {"v3": {"invalid_reward": -3}},
               {"v3": {"quality_deficit_weight": 20}},
               {"v3": {"brevity_weight": 0.11}}, {"v3": {"latency_cost_cap": 0}},
               {"v3": {"brevity_denominator_floor": 0}}]
        for key in DEFAULT_V3:
            for value in (True, float("nan"), float("inf"), -float("inf"), "0.1"):
                bad.append({"v3": {key: value}})
        for settings in bad:
            with self.subTest(settings=settings), self.assertRaises(RewardError):
                validate_v3_settings(settings)

    def test_gate_boundaries_and_continuity(self):
        for quality, gate in ((0.75, 0), (0.8, 0.5), (0.85, 1)):
            result = combine_v3(quality, 3, 0.5, 0.25, self.settings)
            self.assertAlmostEqual(result["quality_gate"], gate)
            self.assertAlmostEqual(result["latency_cost_raw"], 0.05)
            expected = quality - 2 * max(0, 0.8 - quality) - ((1 - gate) * 2 + gate * 0.05)
            expected -= 0.08 * gate * 0.5 + 0.2 * 0.25
            self.assertAlmostEqual(result["reward"], expected)
            left = combine_v3(quality - 1e-8, 3, 0.5, 0.25, self.settings)["reward"]
            right = combine_v3(quality + 1e-8, 3, 0.5, 0.25, self.settings)["reward"]
            self.assertLess(abs(right - left), 1e-6)

    def test_low_quality_cannot_gain_by_speed_or_shortness(self):
        for quality in (0, 0.6, 0.75):
            fast = combine_v3(quality, 0, 0, 0, self.settings)
            slow = combine_v3(quality, 200, 1, 0, self.settings)
            self.assertEqual(fast["reward"], slow["reward"])
            self.assertEqual(slow["weighted_brevity_penalty"], 0)
            self.assertEqual(slow["latency_penalty"], 2)
        worse = combine_v3(0.6, 0, 0, 0, self.settings)["reward"]
        better = combine_v3(0.7, 100, 1, 0, self.settings)["reward"]
        self.assertGreater(better, worse)

    def test_monotone_quality_and_invalid_lower_bound(self):
        for delay in (0, 3, 6, 100, 1e308):
            for brevity in (0, 0.5, 1):
                values = [combine_v3(i / 1000, delay, brevity, 1, self.settings)["reward"]
                          for i in range(1001)]
                self.assertTrue(all(b >= a - 1e-12 for a, b in zip(values, values[1:])))
                self.assertGreater(min(values), -4)

    def test_valid_negative_reward_and_diagnostics(self):
        result = combine_v3(0.8, 1, 0, 0, self.settings)
        self.assertLess(result["reward"], 0)
        self.assertEqual(result["tier"], "feasible")
        self.assertTrue(result["feasible"])
        self.assertTrue(result["quality_pass"])
        self.assertTrue(result["latency_pass"])
        over = combine_v3(0.9, 3.0001, 0, 0, self.settings)
        self.assertEqual(over["tier"], "latency_fail")
        self.assertGreater(over["reward"], 0)  # Diagnostic tier is not a reward cliff.

    def test_raw_cost_and_weighted_components(self):
        result = combine_v3(0.9, 6, 0.5, 0.4, self.settings)
        self.assertAlmostEqual(result["latency_cost_raw"], 0.3)
        self.assertAlmostEqual(result["weighted_brevity_penalty"], 0.04)
        self.assertAlmostEqual(result["weighted_repetition_penalty"], 0.08)
        self.assertEqual(result["length_penalty"], 0.5)
        self.assertEqual(result["repeat_penalty"], 0.4)
        self.assertAlmostEqual(result["reward"], 0.48)

    def test_missing_nonfinite_or_out_of_range_scalar_is_not_a_policy_error(self):
        base = [0.9, 1, 0, 0]
        for index in range(4):
            for bad in (None, True, "0.5", float("nan"), float("inf"), -0.1):
                args = list(base)
                args[index] = bad
                with self.subTest(index=index, bad=bad), self.assertRaises(RewardError):
                    combine_v3(*args, self.settings)
        for index in (0, 2, 3):
            args = list(base)
            args[index] = 1.1
            with self.assertRaises(RewardError):
                combine_v3(*args, self.settings)

    def test_pool_insufficient_disables_brevity(self):
        records = [item(quality=0.9, units=100), item(quality=0.89, units=20)]
        records += [item(quality=0.7, units=300) for _ in range(6)]
        result = finalize_group(records, self.settings)
        self.assertTrue(all(x["anchor_pool_size"] == 2 for x in result))
        self.assertTrue(all(x["group_length_anchor"] is None for x in result))
        self.assertTrue(all(x["length_penalty"] == 0 for x in result))

    def test_no_absolute_quality_pass_means_no_anchor(self):
        result = finalize_group([item(quality=0.77, units=n) for n in range(10, 18)], self.settings)
        self.assertTrue(all(x["anchor_pool_size"] == 0 for x in result))
        self.assertTrue(all(x["group_length_anchor"] is None for x in result))
        self.assertTrue(all(x["length_penalty"] == 0 for x in result))

    def test_low_quality_deletion_has_no_brevity_bonus_even_with_anchor(self):
        records = [item(quality=0.9, units=n) for n in (30, 31, 32)]
        records += [item(quality=0.7, units=1), item(quality=0.7, units=300)]
        result = finalize_group(records, self.settings)
        self.assertEqual(result[-2]["length_penalty"], 0)
        self.assertEqual(result[-1]["length_penalty"], 1)
        self.assertEqual(result[-2]["reward"], result[-1]["reward"])

    def test_linear_percentile_slack_and_floor_not_anchor_floor(self):
        records = [item(units=n) for n in (2, 4, 6, 8)]
        result = finalize_group(records, self.settings)
        self.assertEqual(result[0]["group_length_anchor"], 3.5)
        self.assertEqual(result[0]["group_length_slack"], 2)
        self.assertEqual(result[-1]["length_penalty"], (8 - 3.5 - 2) / 20)
        self.assertEqual(result[0]["length_penalty"], 0)

    def test_invalid_keeps_minus_four_and_cannot_poison_anchor(self):
        records = [item(units=n, identity=n) for n in (30, 40, 50)]
        records += [item(quality=1, latency=None, units=1, eligible=False, identity="invalid")]
        before = copy.deepcopy(records)
        result = finalize_group(records, self.settings)
        self.assertEqual(records, before)
        self.assertEqual([x["identity"] for x in result], [30, 40, 50, "invalid"])
        self.assertEqual(result[0]["group_length_anchor"], 35)
        self.assertEqual(result[-1]["reward"], -4)
        self.assertEqual(result[-1]["length_penalty"], 0)
        self.assertFalse(result[-1]["feasible"])
        all_invalid = finalize_group([item(quality=None, latency=None, units=0, eligible=False)] * 8,
                                     self.settings)
        self.assertTrue(all(x["reward"] == -4 and x["group_length_anchor"] is None for x in all_invalid))

    def test_group_strict_input_validation(self):
        bad_groups = [[], None, {}, [None]]
        for key in item():
            row = item()
            del row[key]
            bad_groups.append([row])
        for key, value in (("eligible", 1), ("quality", None), ("latency", float("nan")),
                           ("units", True), ("units", 1.0), ("units", 0),
                           ("repetition", float("inf")), ("reward", float("nan"))):
            row = item()
            row[key] = value
            bad_groups.append([row])
        bad_groups.append([item(eligible=False, quality=float("inf"))])
        for group in bad_groups:
            with self.subTest(group=group), self.assertRaises(RewardError):
                finalize_group(group, self.settings)

    def test_actual_group_eleven_ranking_and_values(self):
        values = [(0.834058940410614, 0.5461785249307478, 95),
                  (0.8695250749588013, 0.7102909297052153, 105),
                  (0.817269504070282, 0.8251856617647059, 102),
                  (0.8720796704292297, 1.698959072177822, 77),
                  (0.8956819772720337, 1.7681757406444907, 74),
                  (0.8885536789894104, 2.7384448058069384, 68),
                  (0.8377513289451599, 2.697954919175028, 69),
                  (0.8883181214332581, 3.1927502090301005, 69)]
        result = finalize_group([item(q, d, n) for q, d, n in values], self.settings)
        self.assertEqual(result[0]["anchor_pool_size"], 5)
        self.assertEqual(result[0]["group_length_anchor"], 69)
        self.assertAlmostEqual(result[1]["reward"], 0.8199477623622651, places=12)
        self.assertAlmostEqual(result[3]["reward"], 0.8384883235740921, places=12)
        self.assertAlmostEqual(result[4]["reward"], 0.8644152801453502, places=12)
        self.assertGreater(result[3]["reward"], result[1]["reward"])
        self.assertEqual(max(range(8), key=lambda i: result[i]["reward"]), 4)


if __name__ == "__main__":
    unittest.main()
