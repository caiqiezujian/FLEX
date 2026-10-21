from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import export_adapter
from flex_gspo.runtime import directory_inventory
from flex_gspo.trainer import reward_payload, score_shared_group


class ExportContractTests(unittest.TestCase):
    def test_intentionally_omitted_talker_only(self):
        export_adapter.validate_export_loading({"unexpected_keys": ["talker.weight", "code2wav.weight"]})
        for loading, include in (({"missing_keys": ["thinker.model.weight"]}, False),
                                 ({"unexpected_keys": ["random.weight"]}, False),
                                 ({"unexpected_keys": ["talker.weight"]}, True)):
            with self.assertRaises(RuntimeError):
                export_adapter.validate_export_loading(loading, include)

    def test_validate_never_loads_model_and_refuses_wrong_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, checkpoint = root / "base", root / "checkpoint"
            base.mkdir()
            (base / "config.json").write_text("{}")
            metadata = {"config": {"model_path": str(base)},
                        "identity": {"details": {"models": {"actor": directory_inventory(base)}}}}
            with patch("export_adapter.read_resume_metadata", return_value=metadata):
                self.assertEqual(export_adapter.validate_paths(base, checkpoint, root / "output")[0], base)
                with self.assertRaises(ValueError):
                    export_adapter.validate_paths(root / "wrong", checkpoint, root / "out")
                with self.assertRaises(ValueError):
                    export_adapter.validate_paths(base, checkpoint, base / "nested")


class SharedScorerTests(unittest.TestCase):
    def test_minimal_reward_payload_cannot_transfer_audio_or_logits(self):
        trajectory = {"id": "x", "valid": True, "translation": "翻译", "turns": [
            {"response": "<|DIRECT|>翻译", "audio_end_seconds": 1, "_replay": object(),
             "prompt_token_ids": [1, 2], "token_ids": [3], "old_logprobs": [-1]}]}
        result = reward_payload(trajectory)
        self.assertNotIn("_replay", result["turns"][0])
        self.assertNotIn("old_logprobs", result["turns"][0])
        self.assertEqual(result["turns"][0]["response"], "<|DIRECT|>翻译")

    def test_single_rank_shared_scorer_result(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional torch unavailable")
        class Scorer:
            def score_group(self, task, candidates):
                self.seen = candidates
                return [{"reward": .5, "components": {"feasible": True}} for _ in candidates]
        scorer = Scorer()
        result = score_shared_group(scorer, {"id": "x"}, [{"turns": [], "valid": False}], 0, 1)
        self.assertEqual(result[0]["reward"], .5)


if __name__ == "__main__":
    unittest.main()
