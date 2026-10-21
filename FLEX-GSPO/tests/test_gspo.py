import math
import unittest

from flex_gspo.trainer import (backward_trajectory, group_advantages,
                               gspo_coefficient)


class GSPOArithmeticTests(unittest.TestCase):
    def test_equal_group_has_no_advantage(self):
        self.assertEqual(group_advantages([1, 1]), [0, 0])

    def test_group_is_centered_and_ordered(self):
        values = group_advantages([-2, 0.5, 0.7])
        self.assertAlmostEqual(sum(values), 0)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_floor_does_not_amplify_tiny_noise_to_one(self):
        self.assertLess(abs(group_advantages([1, 1.000001], std_floor=0.05)[0]), 0.0001)

    def test_clipping_branches(self):
        self.assertEqual(gspo_coefficient(math.log(1.2), 1, .1, .1)[1], 0)
        self.assertEqual(gspo_coefficient(math.log(.8), -1, .1, .1)[1], 0)
        self.assertAlmostEqual(gspo_coefficient(math.log(.8), 1, .1, .1)[1], -.8)
        self.assertAlmostEqual(gspo_coefficient(math.log(1.2), -1, .1, .1)[1], 1.2)

    def test_coefficient_matches_finite_difference(self):
        for z in (-.2, -.05, 0, .05, .2):
            for advantage in (-1.3, 0, 0.7):
                with self.subTest(z=z, advantage=advantage):
                    step = 1e-7
                    left = gspo_coefficient(z - step, advantage, .1, .1)[0]
                    right = gspo_coefficient(z + step, advantage, .1, .1)[0]
                    self.assertAlmostEqual((right - left) / (2 * step),
                                           gspo_coefficient(z, advantage, .1, .1)[1], places=7)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            group_advantages([0, float("nan")])
        with self.assertRaises(ValueError):
            gspo_coefficient(float("inf"), 1, .1, .1)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional torch not installed")
class GSPOGradientTests(unittest.TestCase):
    def test_two_pass_matches_full_trajectory_autograd_with_unequal_turn_lengths(self):
        class TinyBackend:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))

            def score_turn(self, turn, reference=False):
                weights = torch.tensor(turn["weights"], dtype=torch.float64)
                logits = torch.stack((self.parameter * weights, -self.parameter * weights), dim=-1)
                labels = torch.tensor(turn["token_ids"])
                return torch.log_softmax(logits, -1).gather(-1, labels[:, None]).squeeze(-1)

        backend = TinyBackend()
        turns = [{"token_ids": [0], "weights": [1.]},
                 {"token_ids": [1, 0, 1, 0], "weights": [.5, 1.5, -.2, .7]}]
        for turn in turns:
            turn["old_logprobs"] = backend.score_turn(turn).detach().tolist()
        reference_scores = torch.cat([backend.score_turn(turn) for turn in turns])
        old = torch.tensor(sum((turn["old_logprobs"] for turn in turns), []), dtype=torch.float64)
        ratio = torch.exp((reference_scores - old).mean())
        advantage, group_size = 0.8, 2
        loss = -torch.minimum(ratio * advantage, torch.clamp(ratio, .9, 1.1) * advantage) / group_size
        loss.backward()
        expected = backend.parameter.grad.clone()
        backend.parameter.grad = None
        metrics = backward_trajectory(backend, {"turns": turns}, advantage,
                                      {"clip_low": .1, "clip_high": .1, "parity_tolerance": .01}, group_size)
        self.assertTrue(torch.allclose(backend.parameter.grad, expected, atol=1e-12))
        self.assertEqual(metrics["action_tokens"], 5)
        self.assertAlmostEqual(metrics["sequence_ratio"], 1.)


if __name__ == "__main__":
    unittest.main()
