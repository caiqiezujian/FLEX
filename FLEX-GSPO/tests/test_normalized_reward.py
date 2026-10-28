"""Behavioral checks for bounded complete-group v4, including failure cases."""
import copy
import math
import random
import unittest

from flex_gspo.normalized_reward import DEFAULT_V4, finalize_group, validate_v4_settings
from flex_gspo.rewards import RewardError


def item(q=.9, d=1., units=50, p=0., eligible=True):
    return dict(eligible=eligible, quality=q, latency=d, units=units, repetition=p,
                reward=0. if eligible else -4.)


def failure():
    return item(None, None, 0, eligible=False)


class NormalizedRewardTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        original = {"v4": {"quality_std_floor": .04}}
        before = copy.deepcopy(original)
        self.assertEqual(validate_v4_settings(original)["v4"]["quality_std_floor"], .04)
        self.assertEqual(original, before)
        for key in DEFAULT_V4:
            for val in (True, float("nan"), float("inf"), -1, "0.1"):
                with self.subTest(key=key, val=val), self.assertRaises(RewardError):
                    validate_v4_settings({"v4": {key: val}})
        for options in ({"quality_std_floor": 0}, {"failure_margin": 0}, {"quality_gate_low": .79},
                        {"anchor_min_candidates": 2}, {"quality_gate_high": .8}, {"unknown": 0}):
            with self.assertRaises(RewardError):
                validate_v4_settings({"v4": options})

    def test_invariant_to_order_and_does_not_mutate_evidence(self):
        rows = [item(.8 + i * .01, i, 40 + i) for i in range(7)] + [failure()]
        before = copy.deepcopy(rows)
        actual = finalize_group(rows, {})
        reverse = finalize_group(list(reversed(rows)), {})
        self.assertEqual(rows, before)
        self.assertEqual(actual, list(reversed(reverse)))
        self.assertAlmostEqual(sum(r["estimated_advantage"] for r in actual), 0, places=12)

    def test_tiny_score_differences_are_not_unit_scale_advantages(self):
        rows = finalize_group([item(.9 + i * .000001) for i in range(8)], {})
        self.assertEqual(rows[0]["quality_normalization_scale"], .03)
        self.assertLess(max(abs(r["estimated_advantage"]) for r in rows), .001)

    def test_below_quality_floor_waiting_or_shortening_does_not_help(self):
        rows = finalize_group([item(.6, i * 5, 10 + 20 * i) for i in range(8)], {})
        self.assertTrue(all(r["quality_gate"] == 0 and r["latency_penalty"] == 2 for r in rows))
        self.assertTrue(all(r["estimated_advantage"] == 0 for r in rows))
        self.assertTrue(all(r["group_all_quality_fail"] for r in rows))

    def test_equal_quality_faster_and_nonredundant_rank_higher(self):
        rows = finalize_group([item(.9, 1), item(.9, 6)] + [item(.9, 1) for _ in range(6)], {})
        self.assertGreater(rows[0]["reward"], rows[1]["reward"])
        rows = finalize_group([item(.9, 1, 50)] * 6 + [item(.9, 1, 100), item(.9, 1, 50, .5)], {})
        self.assertGreater(rows[0]["reward"], rows[6]["reward"])
        self.assertGreater(rows[0]["reward"], rows[7]["reward"])

    def test_short_low_quality_cannot_set_brevity_anchor(self):
        rows = finalize_group([item(.9, units=100)] * 7 + [item(.1, units=1)], {})
        self.assertEqual(rows[0]["group_length_anchor"], 100)
        self.assertEqual(rows[0]["weighted_brevity_penalty"], 0)

    def test_all_wait_not_scored_and_has_zero_policy_advantage(self):
        rows = finalize_group([failure() for _ in range(8)], {})
        for row in rows:
            self.assertIsNone(row["quality"])
            self.assertIsNone(row["latency"])
            self.assertIsNone(row["normalized_quality"])
            self.assertTrue(row["group_all_invalid"])
            self.assertEqual(row["reward"], -.25)
            self.assertEqual(row["estimated_advantage"], 0)

    def test_mixed_failures_do_not_receive_fake_semantic_scores(self):
        rows = finalize_group([item(.5), item(.49), item(.47), item(.47)] + [failure()] * 4, {})
        worst_valid = min(r["reward"] for r in rows[:4])
        self.assertAlmostEqual(rows[4]["reward"], worst_valid - .25)
        self.assertTrue(all(r["estimated_advantage"] < 0 for r in rows[4:]))
        self.assertTrue(any(r["estimated_advantage"] < 0 for r in rows[:4]))
        self.assertTrue(rows[0]["low_quality_positive_advantage"])  # Still possible; never claim all bad are negative.
        self.assertIsNone(rows[4]["quality"])

    def test_single_eligible_among_failures_is_bounded_not_fake_normalized(self):
        rows = finalize_group([item(.2)] + [failure()] * 7, {})
        self.assertEqual(rows[0]["normalized_quality"], 0)
        self.assertAlmostEqual(rows[0]["estimated_advantage"], .25 * 7 / 8)
        self.assertTrue(rows[0]["low_quality_positive_advantage"])

    def test_identical_candidates_same_advantage_and_numeric_bound(self):
        rng = random.Random(42)
        for _ in range(100):
            inputs = [item(rng.random(), rng.random()*40, rng.randint(1, 200), rng.random()) for _ in range(7)] + [failure()]
            rows = finalize_group(inputs, {})
            self.assertTrue(all(math.isfinite(r["estimated_advantage"]) and abs(r["estimated_advantage"]) < 6 for r in rows))
        rows = finalize_group([item()] * 8, {})
        self.assertTrue(all(r["estimated_advantage"] == 0 for r in rows))

    def test_missing_score_is_infrastructure_error_not_policy_failure(self):
        for bad in (item(None), item(float("nan")), item(.8, -1), item(.8, 1, 0)):
            with self.assertRaises(RewardError):
                finalize_group([bad] + [item()] * 7, {})
        with self.assertRaises(RewardError):
            finalize_group([item()], {})

    def test_normalization_cannot_fix_misranked_quality_evidence(self):
        # A semantic scorer can prefer an actually wrong output. With these
        # observed Q/D values the scalar formula cannot infer the missing truth.
        rows = finalize_group([item(.883668, .612428), item(.880379, .878982)] + [item(.87, 1)] * 6, {})
        self.assertGreater(rows[0]["reward"], rows[1]["reward"])


if __name__ == "__main__":
    unittest.main()
