"""Numerical invariants for the MetricX objective; no network or GPU."""
import copy
import json
import math
from pathlib import Path
import statistics
import unittest

from flex_gspo.metricx_reward import finalize_group, provisional_v5, validate_v5_settings
from flex_gspo.qalco import validate_settings
from flex_gspo.rewards import RewardError


ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate_settings(json.loads((ROOT / "configs/qalco_v5.json").read_text(encoding="utf-8")))


def item(error=2., delay=2., units=100, repeat=0.):
    return dict(eligible=True, quality=1-error/25, metricx_error=error,
                latency=delay, units=units, repetition=repeat, reward=0.)


def failure():
    return dict(eligible=False, quality=None, metricx_error=None,
                latency=None, units=0, repetition=0., reward=-4.)


class MetricXRewardTests(unittest.TestCase):
    def test_raw_gate_boundary_and_direction(self):
        settings = config()
        for error, cost in ((4.99, 1.), (5., 1.), (math.nextafter(5., math.inf), 10.), (20., 10.)):
            got = provisional_v5(error, 1., 0., settings)
            self.assertEqual(got["latency_penalty"], cost)
            self.assertEqual(got["metricx_quality_reward"], -error)
        self.assertEqual(provisional_v5(1., 30., 0., settings)["latency_penalty"], 10.)

    def test_hpo_hinge_and_cost_have_same_normalized_signal(self):
        rows = [item(2., 1.), item(3., 3.), item(8., 0.1), item(1., 20.)]
        got = finalize_group(rows, config())
        hinge = [min(0., r["latency"]-10.) if r["metricx_error"] <= 5 else 0. for r in rows]
        mean, scale = statistics.fmean(hinge), max(statistics.pstdev(hinge), .5)
        expected = [max(-2., min(2., (v-mean)/scale)) for v in hinge]
        for row, expected_signal in zip(got, expected):
            self.assertAlmostEqual(row["normalized_latency"], expected_signal)

    def test_lower_error_wins_at_same_delay_and_length(self):
        got = finalize_group([item(e) for e in (1., 2., 3., 4., 6., 7., 8., 9.)], config())
        self.assertTrue(all(a["reward"] >= b["reward"] for a, b in zip(got, got[1:])))
        self.assertGreater(got[0]["reward"], got[-1]["reward"])

    def test_wait_cost_does_not_make_bad_quality_fast_candidate_better(self):
        got = finalize_group([item(2., 2.), item(2., 5.), item(8., .01), item(8., 9.)], config())
        self.assertGreater(got[0]["reward"], got[1]["reward"])
        self.assertAlmostEqual(got[2]["reward"], got[3]["reward"])
        self.assertGreater(got[1]["reward"], got[2]["reward"])

    def test_all_low_quality_has_quality_only_order(self):
        got = finalize_group([item(6.+i, .1+i) for i in range(8)], config())
        self.assertTrue(all(r["group_all_quality_fail"] for r in got))
        self.assertTrue(all(r["normalized_latency"] == 0. for r in got))
        self.assertTrue(all(r["weighted_brevity_penalty"] == 0. for r in got))
        self.assertGreater(got[0]["estimated_advantage"], 0.)
        self.assertTrue(got[0]["low_quality_positive_advantage"])

    def test_delay_weight_override_and_default_fallback_in_balanced_g8(self):
        # Same length/no repetition; only accuracy and delay trade off.
        rows = [item(2., 4.) for _ in range(4)] + [item(2.2, 2.) for _ in range(4)]
        settings = config()
        got = finalize_group(rows, settings)
        self.assertAlmostEqual(got[0]["reward"], -.3)
        self.assertAlmostEqual(got[4]["reward"], .3)
        self.assertLess(got[0]["estimated_advantage"], got[4]["estimated_advantage"])
        lower_delay = copy.deepcopy(settings)
        lower_delay["v5"]["latency_weight"] = .1
        alternative = finalize_group(rows, lower_delay)
        self.assertAlmostEqual(alternative[0]["reward"], .1)
        self.assertAlmostEqual(alternative[4]["reward"], -.1)
        # Missing override must use the restored 0.5 default.
        del settings["v5"]["latency_weight"]
        fallback = finalize_group(rows, settings)
        self.assertEqual([r["reward"] for r in got], [r["reward"] for r in fallback])

    def test_small_differences_not_inflated_by_zero_std(self):
        rows = [item(2.+i*1e-7, 2.+i*1e-7) for i in range(8)]
        got = finalize_group(rows, config())
        self.assertTrue(all(abs(r["reward"]) < 1e-5 for r in got))
        self.assertTrue(all(r["quality_normalization_scale"] == .5 for r in got))
        self.assertTrue(all(r["latency_normalization_scale"] == .5 for r in got))

    def test_failure_below_worst_real_score_without_fake_metricx(self):
        rows = [item(e, 1.+e/3) for e in (1., 2., 4., 6., 9.)] + [failure() for _ in range(3)]
        got = finalize_group(rows, config())
        baseline = min(r["reward"] for r in got[:5])-.25
        for r in got[5:]:
            self.assertEqual(r["reward"], baseline)
            self.assertIsNone(r["metricx_error"])
            self.assertIsNone(r["quality"])
            self.assertLess(r["estimated_advantage"], 0.)
        self.assertAlmostEqual(sum(r["estimated_advantage"] for r in got), 0.)

    def test_all_wait_or_identical_groups_have_no_fake_signal(self):
        for rows in ([failure() for _ in range(8)], [item() for _ in range(8)]):
            got = finalize_group(rows, config())
            self.assertTrue(all(r["estimated_advantage"] == 0. for r in got))

    def test_brevity_requires_three_near_best_quality_candidates(self):
        sparse = [item(1., units=100), item(1.1, units=120)] + [item(8., units=10) for _ in range(6)]
        self.assertIsNone(finalize_group(sparse, config())[0]["group_length_anchor"])
        dense = [item(1., units=100), item(1.1, units=110), item(1.2, units=130)] + [item(8., units=10) for _ in range(5)]
        got = finalize_group(dense, config())
        self.assertEqual(got[0]["group_length_anchor"], 105.)
        self.assertGreater(got[2]["weighted_brevity_penalty"], 0.)
        self.assertTrue(all(r["weighted_brevity_penalty"] == 0. for r in got[3:]))

    def test_repetition_penalty_and_input_immutability(self):
        rows = [item(repeat=0.), item(repeat=1.)]
        before = copy.deepcopy(rows)
        got = finalize_group(rows, config())
        self.assertAlmostEqual(got[0]["reward"]-got[1]["reward"], .2)
        self.assertEqual(rows, before)

    def test_config_prohibits_wrong_provider_qe_and_old_threshold(self):
        base = json.loads((ROOT / "configs/qalco_v5.json").read_text(encoding="utf-8"))
        mutations = [lambda r: r.update(quality_provider="comet22"),
                     lambda r: r["metricx"].update(mode="qe"),
                     lambda r: r.update(quality_floor=.85),
                     lambda r: r.update(v4={}),
                     lambda r: r["v5"].update(quality_std_floor=0),
                     lambda r: r["v5"].update(latency_cost_cap_seconds=0)]
        for mutate in mutations:
            trial = copy.deepcopy(base)
            mutate(trial["reward"])
            with self.assertRaises(RewardError):
                validate_settings(trial)

    def test_invalid_raw_metadata_is_never_silently_repaired(self):
        for bad in (None, True, -1., 26., math.inf, math.nan):
            row = item()
            row["metricx_error"] = bad
            with self.assertRaises(RewardError):
                finalize_group([row, item()], config())
        row = item()
        row["quality"] = .5
        with self.assertRaises(RewardError):
            finalize_group([row, item()], config())
        row = failure()
        row["metricx_error"] = 25.
        with self.assertRaises(RewardError):
            finalize_group([row, item()], config())

    def test_final_rewards_and_advantages_are_bounded(self):
        rows = [item(0., 0., 20), item(25., 100., 10000, 1.)] + [failure() for _ in range(6)]
        for row in finalize_group(rows, config()):
            self.assertTrue(math.isfinite(row["reward"]))
            self.assertLessEqual(abs(row["reward"]), 3.53)
            self.assertLessEqual(abs(row["estimated_advantage"]), 7.06)


if __name__ == "__main__":
    unittest.main()
