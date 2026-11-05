"""Raw MetricX observations must survive the distributed metadata boundary."""
import copy
import math
import unittest

from flex_gspo.group_reward import _local_metadata, _runtime_contract
from flex_gspo.monitoring import reward_summary
from flex_gspo.rewards import RewardError
from test_group_reward import _record, _sample, _task, _trainer


class MetricXV5MetadataTests(unittest.TestCase):
    def fixture(self, raw=5.0, *, eligible=True):
        task = _task("metricx-boundary")
        sample = _sample(task, "request-1", "prompt-1")
        record = _record(task, 0, sample.request_id, sample.prompt_id,
                         10 if eligible else 0, eligible=eligible)
        record["reward_version"] = 5
        record["components"].update(reward_version=5,
                                    metricx_error=raw if eligible else None,
                                    quality=1 - raw / 25 if eligible else None)
        return sample, record

    def test_raw_threshold_neighbors_are_not_recovered_from_quality(self):
        for raw in (math.nextafter(5.0, 0.0), 5.0, math.nextafter(5.0, math.inf)):
            with self.subTest(raw=raw):
                sample, record = self.fixture(raw)
                item = _local_metadata([sample], [record], version=5)[0]["item"]
                self.assertEqual(item["metricx_error"], raw)
                self.assertEqual(item["metricx_error"] <= 5.0, raw <= 5.0)

    def test_v4_metadata_shape_remains_unchanged(self):
        sample, record = self.fixture()
        record["reward_version"] = record["components"]["reward_version"] = 4
        item = _local_metadata([sample], [record], version=4)[0]["item"]
        self.assertNotIn("metricx_error", item)

    def test_missing_invalid_and_inconsistent_raw_scores_fail_closed(self):
        sample, record = self.fixture()
        for raw in (None, True, "5", -0.1, 25.1, float("nan"), float("inf"), 4.0):
            with self.subTest(raw=raw):
                changed = copy.deepcopy(record)
                changed["components"]["metricx_error"] = raw
                with self.assertRaises(RewardError):
                    _local_metadata([sample], [changed], version=5)

    def test_invalid_policy_keeps_both_quality_observations_missing(self):
        sample, record = self.fixture(eligible=False)
        item = _local_metadata([sample], [record], version=5)[0]["item"]
        self.assertIsNone(item["metricx_error"])
        self.assertIsNone(item["quality"])

    def test_v5_runtime_requires_center_only_and_loss_kl(self):
        trainer = _trainer(0, [], settings={"provider": "qalco", "version": 5})
        trainer.scale_rewards = "none"
        trainer.advantage_estimator = "grpo"
        trainer.kl_in_reward = False
        self.assertEqual(_runtime_contract(trainer)[1]["version"], 5)
        trainer.scale_rewards = "group"
        with self.assertRaises(RewardError):
            _runtime_contract(trainer)
        trainer.scale_rewards = "none"
        trainer.kl_in_reward = True
        with self.assertRaises(RewardError):
            _runtime_contract(trainer)

    def test_monitor_keeps_raw_error_and_negative_error_signal_separate(self):
        _, record = self.fixture(2.5)
        record["components"].update(metricx_quality_reward=-2.5,
                                    metricx_group_mean=3.0, metricx_group_std=0.5,
                                    metricx_error_threshold=5.0,
                                    latency_cost_cap_seconds=10.0)
        summary = reward_summary([record])
        self.assertEqual(summary["metricx_error/mean"], 2.5)
        self.assertEqual(summary["quality/mean"], 0.9)
        self.assertEqual(summary["metricx_quality_reward/mean"], -2.5)
        self.assertEqual(summary["metricx_group_std/mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
