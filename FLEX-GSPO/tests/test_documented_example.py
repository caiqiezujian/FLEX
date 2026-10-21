"""Numerical method-document example; scores are hypothetical, not MetricX results."""
import unittest

from flex_gspo.qalco import QalcoScorer, latency_statistics
from flex_gspo.trainer import group_advantages


class DocumentedExampleTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "id": "synthetic-doc-only", "source_lang": "en", "target_lang": "zh",
            "source_text": "We are meeting in Beijing this weekend.",
            "reference_text": "我们周末在北京开会。", "duration_seconds": 6.0,
            "chunks": [{"end_seconds": float(i)} for i in range(1, 7)],
        }
        schedules = [
            {2: "我们开会，", 3: "地点在北京，", 4: "时间是这个周末。"},
            {1: "我们", 4: "周末在北京开会。"},
            {4: "我们开会，地点在北京，时间是这个周末。"},
            {6: "我们周末在北京开会。"},
            {4: "我们周末在上海开会。"},
        ]
        self.candidates = []
        for schedule in schedules:
            self.candidates.append({
                "valid": True, "complete_input": True, "final_wait": 6 not in schedule,
                "turns": [{"response": "<|DIRECT|>" + schedule[i] if i in schedule else "<|WAIT|>",
                           "audio_end_seconds": float(i), "finish_reason": "stop"}
                          for i in range(1, 7)],
            })
        class HypotheticalQuality:
            def score_batch(self, source, hypotheses, reference):
                # Used ONLY as a test fixture; production has no fake-score backend.
                errors = {
                    "我们开会，地点在北京，时间是这个周末。": 2.5,
                    "我们周末在北京开会。": 1.0,
                    "我们周末在上海开会。": 8.0,
                }
                if len(hypotheses) != 3:
                    raise AssertionError("Identical complete translations should share one score")
                return [errors[text] for text in hypotheses]
        config = {"reward": {"provider": "qalco", "metricx": {"backend": "service"},
                            "quality_floor": .85, "latency_budget_seconds": 3,
                            "reference_length_ratio": 1.2, "latency_weight": .2, "length_weight": .1}}
        self.scores = QalcoScorer(config, quality_model=HypotheticalQuality()).score_group(
            self.task, self.candidates)

    def test_document_table_matches_actual_reward_functions(self):
        expected = [
            (16, .90, .375, 0.4814814814814814, .8668091168091168),
            (9, .96, 2/3, 0, .935042735042735),
            (16, .90, 1.1875, 0.4814814814814814, .8251424501424501),
            (9, .96, 6, 0, -2),
            (9, .68, 4/3, 0, -2.2),
        ]
        for score, (units, quality, lag, cost, reward) in zip(self.scores, expected):
            with self.subTest(reward=reward):
                parts = score["components"]
                self.assertEqual(parts["diagnostics"]["emitted_units"], units)
                self.assertAlmostEqual(parts["quality"], quality)
                self.assertAlmostEqual(parts["laal_seconds"], lag)
                self.assertAlmostEqual(parts["length_penalty"], cost)
                self.assertAlmostEqual(score["reward"], reward)

    def test_advantages_are_relative_not_winner_only(self):
        advantages = group_advantages([score["reward"] for score in self.scores])
        expected = [.72397274, .76578656, .69843925, -1.03281889, -1.15537966]
        for actual, value in zip(advantages, expected):
            self.assertAlmostEqual(actual, value, places=7)
        self.assertGreater(advantages[2], 0)  # Worse than B, still above this group mean.

    def test_eof_cutoff_and_final_wait_interpretation(self):
        self.assertEqual(self.scores[3]["components"]["diagnostics"]["laal_tau"], 1)
        for index in (0, 1, 2, 4):
            self.assertTrue(self.candidates[index]["final_wait"])
            self.assertTrue(self.scores[index]["components"]["protocol_valid"])

    def test_documented_laal_eof_nonmonotonic_boundary(self):
        # This records a limitation, NOT desired policy behavior or a new reward.
        before = latency_statistics(6, [1, 1, 1, 4, 5, 6, 6, 6, 6, 6], 10)
        delayed = latency_statistics(6, [1, 1, 1, 6, 6, 6, 6, 6, 6, 6], 10)
        self.assertAlmostEqual(before["laal_seconds"], 1.5)
        self.assertAlmostEqual(delayed["laal_seconds"], 1.35)
        self.assertEqual(before["diagnostics"]["laal_tau"], 6)
        self.assertEqual(delayed["diagnostics"]["laal_tau"], 4)


if __name__ == "__main__":
    unittest.main()
