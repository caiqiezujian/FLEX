"""Tiny CPU policy tests: real gradients/optimizer/files, never Omni/GPU validation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flex_gspo.runtime import read_resume_metadata, run_identity
from flex_gspo.trainer import _save_checkpoint, backward_trajectory, group_advantages

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional CPU torch unavailable")
class TinyPolicyIntegrationTests(unittest.TestCase):
    class Backend:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
            self.trainable = [self.parameter]

        def score_turn(self, turn, reference=False):
            parameter = self.parameter * 0 if reference else self.parameter
            logits = torch.stack((parameter, -parameter))
            probabilities = torch.log_softmax(logits, dim=0)
            return probabilities[torch.tensor(turn["token_ids"])]

        def rollout(self, choice):
            # Unequal per-turn lengths exercise whole-trajectory normalization.
            turns = [{"token_ids": [choice]}, {"token_ids": [choice, choice]}]
            for turn in turns:
                turn["old_logprobs"] = self.score_turn(turn).detach().tolist()
            return {"turns": turns}

        def save(self, path):
            root = Path(path)
            root.mkdir()
            (root / "adapter_config.json").write_text('{"fixture":"tiny CPU, not PEFT"}', encoding="utf-8")
            torch.save({"parameter": self.parameter.detach()}, root / "adapter_model.bin")

    def setUp(self):
        self.backend = self.Backend()
        self.settings = {"clip_low": .0003, "clip_high": .0004,
                         "parity_tolerance": 1e-8, "mean_logprob_tolerance": 1e-9}
        self.optimizer = torch.optim.AdamW(self.backend.trainable, lr=0.01, weight_decay=0)

    def update(self):
        group = [self.backend.rollout(0), self.backend.rollout(1)]
        advantages = group_advantages([1.0, -1.0])
        self.optimizer.zero_grad(set_to_none=True)
        for trajectory, advantage in zip(group, advantages):
            backward_trajectory(self.backend, trajectory, advantage, self.settings, len(group))
        gradient = self.backend.parameter.grad.item()
        self.optimizer.step()
        return group, gradient

    def test_real_cpu_optimizer_promotes_higher_reward_sequence(self):
        before = self.backend.parameter.item()
        _, gradient = self.update()
        self.assertLess(gradient, 0)
        self.assertGreater(self.backend.parameter.item(), before)
        # Second update requires fresh on-policy action probabilities.
        previous = self.backend.parameter.item()
        self.update()
        self.assertGreater(self.backend.parameter.item(), previous)

    def test_stale_sft_probabilities_rejected_before_update(self):
        group, _ = self.update()
        after = self.backend.parameter.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        with self.assertRaisesRegex(RuntimeError, "parity failed|logprob mismatch"):
            backward_trajectory(self.backend, group[0], 1, self.settings, 2)
        self.assertTrue(torch.equal(self.backend.parameter.detach(), after))
        self.assertIsNone(self.backend.parameter.grad)

    def test_checkpoint_integrity_and_optimizer_round_trip_on_cpu(self):
        self.update()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            identity = {"sha256": "tiny-test-only", "details": {}}
            # Only CUDA RNG acquisition is mocked; adapter/optimizer files and
            # SHA256 verification use the actual implementation.
            with patch("torch.cuda.get_rng_state", return_value=torch.get_rng_state()):
                _save_checkpoint(self.backend, self.optimizer, {}, output, 1, identity, 0, 1)
            checkpoint = output / "checkpoint-00000001"
            metadata = read_resume_metadata(checkpoint)
            self.assertEqual(metadata["step"], 1)
            self.assertEqual(json.loads((output / "latest.json").read_text(encoding="utf-8"))["step"], 1)
            state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
            replacement = self.Backend()
            replacement.parameter.data.copy_(self.backend.parameter.detach())
            replacement_optimizer = torch.optim.AdamW(replacement.trainable, lr=0.01, weight_decay=0)
            replacement_optimizer.load_state_dict(state["optimizer"])
            self.assertEqual(replacement_optimizer.state[replacement.parameter]["step"].item(), 1)
            with (checkpoint / "adapter" / "adapter_model.bin").open("ab") as handle:
                handle.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "integrity"):
                read_resume_metadata(checkpoint)


class ResumeIdentityIntegrationTests(unittest.TestCase):
    def test_only_schedule_extension_and_output_location_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor"
            actor.mkdir()
            (actor / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
            (actor / "model.safetensors").write_bytes(b"not real model weights")
            config = {"model_path": str(actor), "manifest_path": "/same/tasks.jsonl",
                      "output_dir": "/old/run", "seed": 42,
                      "train": {"max_updates": 2, "epochs": 1, "save_every": 1, "learning_rate": 1e-6},
                      "reward": {"provider": "qalco", "metricx": {
                          "backend": "service", "mode": "qe",
                          "service_url": "http://localhost:9000", "service_id": "test-service"}},
                      "rollout": {"group_size": 8}}
            audio = root / "fixture.wav"
            audio.write_bytes(b"tiny identity fixture, not actual audio")
            tasks = [{"id": "fixture", "audio_path": str(audio), "source_text": "fixed"}]
            original = run_identity(config, tasks, 4)
            extended = deepcopy(config)
            extended.update(output_dir="/new/run", resume_from="/old/checkpoint")
            extended["train"].update(max_updates=100, epochs=2, save_every=10)
            self.assertEqual(original, run_identity(extended, tasks, 4))
            extended["rollout"]["group_size"] = 4
            self.assertNotEqual(original, run_identity(extended, tasks, 4))
            self.assertNotEqual(original, run_identity(config, tasks, 1))
            changed_tasks = [dict(tasks[0], source_text="changed")]
            self.assertNotEqual(original, run_identity(config, changed_tasks, 4))


if __name__ == "__main__":
    unittest.main()
