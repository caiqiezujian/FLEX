import json
from pathlib import Path
import tempfile
import unittest

from flex_gspo.runtime import (atomic_json, canonical_digest, file_sha256, make_schedule,
                               read_resume_metadata, rollout_seed, run_identity)


class RuntimeTests(unittest.TestCase):
    def test_schedule_disjoint_equal_shards_reproducible(self):
        tasks = [{"id": str(i)} for i in range(13)]
        first = [make_schedule(tasks, 4, rank, epochs=2, seed=42) for rank in range(4)]
        self.assertEqual([len(x) for x in first], [6] * 4)
        for epoch in (0, 1):
            seen = [task["id"] for shard in first for ep, task in shard if ep == epoch]
            self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(first[0], make_schedule(tasks, 4, 0, epochs=2, seed=42))
        extended = make_schedule(tasks, 4, 0, epochs=3, seed=42)
        self.assertEqual(first[0], extended[:6])

    def test_seeds_include_task_rank_step_and_candidate(self):
        values = {rollout_seed(42, step, rank, candidate, "audio")
                  for step in range(2) for rank in range(4) for candidate in range(8)}
        self.assertEqual(len(values), 64)
        self.assertNotEqual(rollout_seed(42, 0, 0, 0, "audio"),
                            rollout_seed(42, 0, 0, 0, "another"))

    def test_checkpoint_integrity_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter").mkdir()
            (root / "adapter" / "adapter_config.json").write_text("{}")
            state = root / "training_state.pt"
            state.write_bytes(b"fixture")
            metadata = {"format_version": 2, "step": 3, "files_sha256": {
                "training_state.pt": file_sha256(state),
                "adapter/adapter_config.json": file_sha256(root / "adapter" / "adapter_config.json")}}
            atomic_json(root / "checkpoint.json", metadata)
            self.assertEqual(read_resume_metadata(root)["step"], 3)
            state.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "integrity"):
                read_resume_metadata(root)
            metadata["files_sha256"] = {"training_state.pt": file_sha256(state), "../escape": "bad"}
            atomic_json(root / "checkpoint.json", metadata)
            with self.assertRaises(ValueError):
                read_resume_metadata(root)

    def test_resume_identity_allows_only_operational_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fixture")
            config = {"model_path": str(root), "output_dir": "old", "seed": 42,
                      "train": {"max_updates": 2, "epochs": 1, "save_every": 1},
                      "rollout": {"group_size": 8}}
            tasks = [{"id": "one", "source_text": "source"}]
            before = run_identity(config, tasks, 4)
            config["train"].update(max_updates=100, epochs=2, save_every=25)
            config.update(output_dir="new", resume_from="old/checkpoint")
            self.assertEqual(before, run_identity(config, tasks, 4))
            config["rollout"]["group_size"] = 4
            self.assertNotEqual(before, run_identity(config, tasks, 4))
            self.assertNotEqual(before, run_identity(config, tasks, 1))

    def test_json_no_nan_and_order_invariant_digest(self):
        self.assertEqual(canonical_digest({"a": 1, "b": 2}), canonical_digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            canonical_digest({"a": float("nan")})

    def test_audio_replacement_changes_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fixture")
            audio = root / "audio.wav"
            audio.write_bytes(b"first audio")
            config = {"model_path": str(root), "output_dir": "output"}
            tasks = [{"id": "one", "audio_path": str(audio)}]
            before = run_identity(config, tasks, 1)
            audio.write_bytes(b"changed recording with different size")
            self.assertNotEqual(before, run_identity(config, tasks, 1))


if __name__ == "__main__":
    unittest.main()
