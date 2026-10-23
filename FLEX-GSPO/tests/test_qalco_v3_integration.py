"""CPU integration: frozen scorer -> provisional records -> final audit log."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from flex_gspo.gated_reward import finalize_group
from flex_gspo.online_reward import OnlineQalcoReward, load_reward_config
from flex_gspo.qalco import QalcoScorer, combine_reward, validate_settings
from flex_gspo.rewards import RewardError


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qalco_v3.json"


class Quality:
    def __init__(self):
        self.calls = []

    def score_batch(self, src, hypotheses, ref):
        self.calls.append(list(hypotheses))
        return [0.9] * len(hypotheses)


def task():
    return {"id": "synthetic-budget", "source_lang": "en", "target_lang": "zh",
            "source_text": "We meet in Beijing to discuss the annual budget.",
            "reference_text": "我们在北京开会，讨论年度预算。", "duration_seconds": 2.0,
            "chunks": [{"end_seconds": 1.0}, {"end_seconds": 2.0}]}


def trajectory(text="我们在北京开会，讨论年度预算。"):
    return {"schema_version": 2, "valid": True, "complete_input": True,
            "episode_complete": True, "eof_completed": True, "turns": [
                {"kind": "audio", "response": "<|WAIT|>", "finish_reason": "stop", "audio_end_seconds": 1.0},
                {"kind": "audio", "response": "<|DIRECT|>" + text if text else "<|WAIT|>",
                 "finish_reason": "stop", "audio_end_seconds": 2.0},
                {"kind": "eof", "response": "<|WAIT|>", "finish_reason": "stop", "audio_end_seconds": 2.0,
                 "eof_elapsed_seconds": 0.1}]}


class V3IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_reward_config(CONFIG)
        self.quality = Quality()
        self.scorer = QalcoScorer(self.config, quality_model=self.quality)

    def test_v3_rejects_obsolete_length_controls(self):
        for field in ("reference_length_ratio", "latency_weight", "length_weight"):
            config = copy.deepcopy(self.config)
            config["reward"][field] = 1.2
            with self.assertRaises(RewardError):
                validate_settings(config)
        with self.assertRaises(RewardError):
            combine_reward(0.9, 1.0, 0, self.scorer.settings)

    def test_complete_translation_scored_verbatim_and_never_locally_grouped(self):
        tr = trajectory()
        delegate = OnlineQalcoReward(self.config, scorer=self.scorer, log_dir="")
        delegate(["last-only"] * 8, flex_task=[task()] * 8,
                 rollout_infos=[{"flex_trajectory": tr}] * 8)
        self.assertEqual(self.quality.calls, [["我们在北京开会，讨论年度预算。"]])
        self.assertEqual(delegate._calls, 0)
        self.assertTrue(all(r["components"]["group_finalized"] is False for r in delegate.last_records))
        self.assertTrue(all(r["components"]["eligible"] for r in delegate.last_records))
        self.assertEqual(len(delegate.last_records[0]["task_hash"]), 64)

    def test_final_log_matches_final_reward_and_keeps_raw_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            delegate = OnlineQalcoReward(self.config, scorer=self.scorer, log_dir=directory)
            delegate(["last-only"] * 8, flex_task=[task()] * 8,
                     rollout_infos=[{"flex_trajectory": trajectory()}] * 8)
            self.assertEqual(list(Path(directory).glob("*.jsonl")), [])
            original = copy.deepcopy(delegate.last_records)
            items = [{"eligible": True, "quality": r["components"]["quality"],
                      "latency": r["components"]["latency_seconds"],
                      "units": r["components"]["emitted_units"], "repetition": 0.0, "reward": r["reward"]}
                     for r in original]
            computed = finalize_group(items, delegate.settings)
            final = copy.deepcopy(original)
            for r, result in zip(final, computed):
                r["reward"] = result["reward"]
                r["components"].update(result, group_finalized=True)
            delegate.commit_group_rewards(final, SimpleNamespace(global_step=7))
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["global_step"] == 7 and r["call"] == 1 for r in rows))
            self.assertEqual([r["reward"] for r in rows], [r["reward"] for r in final])
            self.assertEqual([r["trajectory"] for r in rows], [r["trajectory"] for r in original])
            with self.assertRaisesRegex(RewardError, "duplicate"):
                delegate.commit_group_rewards(final, SimpleNamespace(global_step=7))

    def test_invalid_and_empty_are_excluded_from_anchor(self):
        a = trajectory("")
        b = trajectory()
        b["turns"][0]["response"] = "<|WAIT|>invalid"
        for tr in (a, b):
            record = self.scorer.score(task(), tr)
            self.assertFalse(record["components"]["eligible"])
            self.assertEqual(record["reward"], -4)
        self.assertEqual(self.quality.calls, [])

    def test_v3_requires_eof_schema(self):
        tr = trajectory()
        tr["schema_version"] = 1
        with self.assertRaises(RewardError):
            self.scorer.score(task(), tr)

    def test_finalization_rejects_unfinalized_or_reordered_evidence(self):
        delegate = OnlineQalcoReward(self.config, scorer=self.scorer, log_dir="")
        delegate(["last-only"], flex_task=[task()], rollout_infos=[{"flex_trajectory": trajectory()}])
        with self.assertRaisesRegex(RewardError, "provisional"):
            delegate.commit_group_rewards(copy.deepcopy(delegate.last_records))
        with self.assertRaisesRegex(RewardError, "not finalized"):
            delegate(["last-only"], flex_task=[task()], rollout_infos=[{"flex_trajectory": trajectory()}])
        modified = copy.deepcopy(delegate.last_records)
        modified[0]["components"]["group_finalized"] = True
        modified[0]["trajectory"]["turns"] = []
        with self.assertRaisesRegex(RewardError, "evidence"):
            delegate.commit_group_rewards(modified)


if __name__ == "__main__":
    unittest.main()
