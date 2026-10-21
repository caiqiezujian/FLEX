import copy
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo.online_reward import OnlineQalcoReward, load_reward_config
from flex_gspo.qalco import QalcoScorer, combine_reward, latency_statistics
from flex_gspo.rewards import RewardError


class Quality:
    def __init__(self):
        self.calls = []

    def score_batch(self, source, hypotheses, reference):
        self.calls.append((source, list(hypotheses), reference))
        return [2.0] * len(hypotheses)


class OnlineRewardTests(unittest.TestCase):
    def setUp(self):
        self.config = {"reward": {
            "provider": "qalco", "latency_metric": "mean_lag", "quality_floor": .85,
            "latency_budget_seconds": 3, "reference_length_ratio": 1.2,
            "latency_weight": .2, "length_weight": .1,
            "metricx": {"backend": "service", "mode": "reference",
                        "service_url": "http://127.0.0.1:16999", "service_id": "test"}}}
        self.task = {"schema_version": 1, "id": "same-source", "source_lang": "en",
                     "target_lang": "zh", "source_text": "We meet in Beijing.",
                     "reference_text": "我们在北京开会。", "duration_seconds": 4.,
                     "chunks": [{"end_seconds": t} for t in (1., 2., 3., 4.)]}
        self.trajectory = {"valid": True, "complete_input": True, "turns": [
            {"response": "<|FLEX|>我们", "audio_end_seconds": 1., "finish_reason": "stop"},
            {"response": "<|WAIT|>", "audio_end_seconds": 2., "finish_reason": "stop"},
            {"response": "<|DIRECT|>在北京", "audio_end_seconds": 3., "finish_reason": "stop"},
            {"response": "<|DIRECT|>开会。", "audio_end_seconds": 4., "finish_reason": "stop"}]}
        self.quality = Quality()
        self.scorer = QalcoScorer(self.config, quality_model=self.quality)
        self.warning = patch("flex_gspo.online_reward.LOG.warning").start()
        self.addCleanup(patch.stopall)
        self.reward = OnlineQalcoReward(self.config, scorer=self.scorer, log_dir="")

    def call(self, trajectories=None, tasks=None):
        trajectories = trajectories if trajectories is not None else [self.trajectory]
        tasks = tasks if tasks is not None else [self.task] * len(trajectories)
        return self.reward(["THIS LAST COMPLETION MUST NOT BE SCORED"] * len(trajectories),
                           flex_task=[json.dumps(task) for task in tasks],
                           rollout_infos=[{"flex_trajectory": trajectory} for trajectory in trajectories])

    def test_scores_whole_translation_not_last_completion(self):
        self.call()
        self.assertEqual(self.quality.calls,
                         [(self.task["source_text"], ["我们在北京开会。"], self.task["reference_text"])])
        metrics = self.reward.last_records[0]["components"]
        self.assertEqual(metrics["metricx_error"], 2.)
        self.assertEqual(metrics["quality_metric"], "metricx24_reference")
        self.assertEqual(metrics["latency_metric"], "source_clock_mean_positive_lag")
        self.assertEqual(metrics["latency_seconds"], metrics["mean_lag_seconds"])

    def test_local_partial_group_not_normalized_to_zero(self):
        scores = self.call()
        direct = self.scorer.score(self.task, self.trajectory)["reward"]
        self.assertEqual(scores, [direct])
        self.assertGreater(direct, 0)

    def test_same_source_candidates_batched_and_deduplicated(self):
        alternative = copy.deepcopy(self.trajectory)
        alternative["turns"][-1]["response"] = "<|DIRECT|>举行会议。"
        self.call([self.trajectory, alternative, self.trajectory])
        self.assertEqual(len(self.quality.calls), 1)
        self.assertEqual(self.quality.calls[0][1], ["我们在北京开会。", "我们在北京举行会议。"])
        self.assertEqual(len(self.reward.last_records), 3)

    def test_same_id_different_tasks_not_combined(self):
        other = copy.deepcopy(self.task)
        other["source_text"] = "They meet in Beijing."
        self.call([self.trajectory] * 3, [self.task, other, self.task])
        self.assertEqual(len(self.quality.calls), 2)
        self.assertEqual([row["candidate_index_in_local_batch"] for row in self.reward.last_records], [0, 1, 2])

    def test_invalid_action_finite_reward_without_quality_api(self):
        self.trajectory["valid"] = False
        self.trajectory["invalid_reason"] = "WAIT unexpectedly carries text"
        self.assertEqual(self.call(), [-4.])
        self.assertFalse(self.quality.calls)
        self.assertIn("WAIT", self.reward.last_records[0]["components"]["error"])

    def test_empty_wait_episode_invalid_not_free_zero_latency(self):
        for turn in self.trajectory["turns"]:
            turn["response"] = "<|WAIT|>"
        self.assertEqual(self.call(), [-4.])
        self.assertFalse(self.quality.calls)

    def test_missing_rollout_info_is_infrastructure_failure(self):
        with self.assertRaisesRegex(RewardError, "refusing last-turn scoring"):
            self.reward(["<|DIRECT|>单轮。"], flex_task=[self.task], rollout_infos=[{}])

    def test_missing_scheduler_turn_field_is_infrastructure_failure(self):
        for field in ("response", "audio_end_seconds", "finish_reason"):
            trajectory = copy.deepcopy(self.trajectory)
            del trajectory["turns"][0][field]
            with self.subTest(field=field), self.assertRaises(RewardError):
                self.call([trajectory])

    def test_metadata_flags_must_be_explicit_booleans(self):
        self.trajectory["complete_input"] = "true"
        with self.assertRaises(RewardError):
            self.call()

    def test_corrupt_timing_is_not_bad_policy(self):
        self.trajectory["turns"][0]["audio_end_seconds"] = 2.
        with self.assertRaisesRegex(RewardError, "timing disagrees"):
            self.call()

    def test_incomplete_policy_episode_is_bad_policy(self):
        self.trajectory["turns"].pop()
        self.trajectory["complete_input"] = False
        self.assertEqual(self.call(), [-4.])

    def test_batch_shape_mismatch_is_not_silently_broadcast(self):
        with self.assertRaisesRegex(RewardError, "one item per completion"):
            self.reward(["a", "b"], flex_task=[self.task], rollout_infos=[{}])

    def test_empty_local_batch_needs_no_metadata_or_api(self):
        self.assertEqual(self.reward([]), [])
        self.assertEqual(self.reward.last_records, [])
        self.assertFalse(self.quality.calls)

    def test_request_and_prompt_ids_follow_input_order(self):
        self.reward(["final1", "final2"], flex_task=[self.task, self.task],
                    rollout_infos=[{"flex_trajectory": self.trajectory}] * 2,
                    request_id=["candidate-a", "candidate-b"], prompt_id=["prompt", "prompt"])
        self.assertEqual([r["request_id"] for r in self.reward.last_records],
                         ["candidate-a", "candidate-b"])
        self.assertEqual([r["prompt_id"] for r in self.reward.last_records], ["prompt", "prompt"])
        self.assertEqual(self.reward.last_records[0]["translation"], "我们在北京开会。")
        self.assertEqual(self.reward.last_records[0]["trajectory"]["turns"], self.trajectory["turns"])

    def test_bad_identity_batch_length_is_infrastructure_failure(self):
        with self.assertRaisesRegex(RewardError, "request_id"):
            self.reward(["final"], flex_task=[self.task],
                        rollout_infos=[{"flex_trajectory": self.trajectory}], request_id=[])

    def test_invalid_raw_outputs_are_logged_without_repair(self):
        self.trajectory["turns"][0]["response"] = "<|WAIT|>bad"
        self.call()
        self.assertEqual(self.reward.last_records[0]["trajectory"]["turns"][0]["response"], "<|WAIT|>bad")
        self.assertIsNone(self.reward.last_records[0]["translation"])

    def test_service_failure_propagates_without_invented_scores(self):
        with patch.object(self.quality, "score_batch", side_effect=RewardError("service unavailable")):
            with self.assertRaisesRegex(RewardError, "service unavailable"):
                self.call()

    def test_rank_diagnostic_log_does_not_include_source_text(self):
        with tempfile.TemporaryDirectory() as directory:
            self.reward.log_dir = directory
            self.reward(["final"], flex_task=[self.task],
                        rollout_infos=[{"flex_trajectory": self.trajectory}],
                        trainer_state=SimpleNamespace(global_step=7))
            paths = list(Path(directory).glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            raw = paths[0].read_text(encoding="utf-8")
            record = json.loads(raw)
            self.assertEqual(record["global_step"], 7)
            self.assertEqual(record["components"]["metricx_error"], 2.)
            self.assertNotIn(self.task["source_text"], raw)

    def test_default_config_is_reward_only_reference_mode(self):
        config = load_reward_config()
        self.assertEqual(set(config), {"reward"})
        self.assertEqual(config["reward"]["latency_metric"], "mean_lag")
        self.assertEqual(config["reward"]["metricx"]["mode"], "reference")
        self.assertNotIn("calibration_acknowledged", config["reward"])


class MonotonicLatencyTests(unittest.TestCase):
    def test_eof_cutoff_counterexample(self):
        early = latency_statistics(10., [1., 1., 1., 9., 10.], 5)
        later = latency_statistics(10., [1., 1., 1., 10., 10.], 5)
        # Legacy LAAL can improve solely because its tau cutoff changed.
        self.assertLess(later["laal_seconds"], early["laal_seconds"])
        self.assertAlmostEqual(early["mean_lag_seconds"], 1.2)
        self.assertAlmostEqual(later["mean_lag_seconds"], 1.4)

    def test_all_monotone_delays_never_improve_fixed_text_latency(self):
        schedules = list(itertools.combinations_with_replacement([1., 3., 4.], 4))
        for earlier in schedules:
            for later in schedules:
                if all(a <= b for a, b in zip(earlier, later)):
                    a = latency_statistics(4., earlier, 5)["mean_lag_seconds"]
                    b = latency_statistics(4., later, 5)["mean_lag_seconds"]
                    self.assertGreaterEqual(b + 1e-12, a)

    def test_same_quality_length_delaying_output_never_improves_reward(self):
        settings = {"quality_floor": .85, "latency_budget_seconds": 3.,
                    "latency_weight": .2, "length_weight": .1}
        a = latency_statistics(10., [1., 1., 1., 9., 10.], 5)["mean_lag_seconds"]
        b = latency_statistics(10., [1., 1., 1., 10., 10.], 5)["mean_lag_seconds"]
        self.assertGreater(combine_reward(.92, a, 0, settings)["reward"],
                           combine_reward(.92, b, 0, settings)["reward"])


if __name__ == "__main__":
    unittest.main()
