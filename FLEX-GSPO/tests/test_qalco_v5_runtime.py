"""CPU-only v5 scorer, HTTP adapter, and complete G8 collective integration."""
import copy
import json
import math
from pathlib import Path
import statistics
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from flex_gspo.group_reward import is_group_scoring_active, run_group_reward_scoring
from flex_gspo.metricx import ServiceMetricX
from flex_gspo.online_reward import OnlineQalcoReward
from flex_gspo.qalco import QalcoScorer
from flex_gspo.rewards import RewardError
from test_group_reward import Collective, FakeTensor, _sample, _trainer
from test_qalco_v3_integration import task, trajectory


def config():
    return {"reward": {"provider": "qalco", "version": 5, "quality_provider": "metricx",
                       "metricx": {"backend": "service", "mode": "reference",
                                   "service_url": "http://127.0.0.1:16999",
                                   "service_id": "cpu-test"}}}


class V5RuntimeTests(unittest.TestCase):
    def run_case(self, *, empty_indices=(), invalid_indices=(), service_fail=None,
                 malformed_scores=None, wrong_scale=None, commit_fail=None, log_root=None):
        collective = Collective()
        source = task()
        translations = [f"�����ڱ�������Ԥ�㷽��{i}��" for i in range(8)]
        # Adjacent IEEE values straddle the exact raw-score gate. This catches
        # loss of raw evidence or reconstructing errors from rounded quality.
        raw_errors = [0., .5, 2., 5., math.nextafter(5., math.inf), 10., 20., 25.]
        scores = dict(zip(translations, raw_errors))
        trainers, batches, paths = [], [], []
        calls = [[] for _ in range(4)]
        for rank in range(4):
            class Quality:
                def __init__(self, rank):
                    self.rank = rank

                def score_batch(self, src, hypotheses, ref):
                    calls[self.rank].append((src, list(hypotheses), ref))
                    if self.rank == service_fail:
                        raise RewardError("MetricX unavailable; no substitute score")
                    if malformed_scores is not None and self.rank == malformed_scores[0]:
                        return malformed_scores[1]
                    return [scores[text] for text in hypotheses]

            adapter = OnlineQalcoReward(
                config(), scorer=QalcoScorer(config(), quality_model=Quality(rank)),
                log_dir=str(Path(log_root) / str(rank)) if log_root else "")
            trainer = _trainer(rank, [])
            trainer.num_generations = trainer.generation_batch_size = 8
            trainer.scale_rewards = "group" if rank == wrong_scale else "none"
            trainer.advantage_estimator = "grpo"
            trainer.kl_in_reward = False
            trainer.reward_funcs = [SimpleNamespace(scorer=adapter)]
            if rank == commit_fail:
                def broken_commit(*args, **kwargs):
                    raise OSError("local log device unavailable")
                adapter.commit_group_rewards = broken_commit
            trainers.append(trainer)
            indices = [rank * 2, rank * 2 + 1]
            batches.append([_sample(source, f"request-{i}", "metricx-full-g8") for i in indices])
            local_paths = []
            for index in indices:
                tr = trajectory("" if index in empty_indices else translations[index])
                tr["error"] = None
                if index in invalid_indices:
                    tr.update(valid=False, error="synthetic invalid protocol")
                local_paths.append(tr)
            paths.append(local_paths)
        original_paths = copy.deepcopy(paths)
        results, errors, contexts = {}, {}, {}

        def run(rank):
            def score(current, samples):
                self.assertTrue(is_group_scoring_active())
                adapter = current.reward_funcs[0].scorer
                result = adapter(["LAST TURN IS NOT THE TRANSLATION"] * 2,
                                 flex_task=[source] * 2,
                                 rollout_infos=[{"flex_trajectory": tr} for tr in paths[rank]],
                                 request_id=[s.request_id for s in samples],
                                 prompt_id=[s.prompt_id for s in samples])
                self.assertEqual(adapter._calls, 0)
                self.assertTrue(all(not r["components"]["group_finalized"] for r in adapter.last_records))
                if log_root:
                    self.assertEqual(list((Path(log_root) / str(rank)).glob("*.jsonl")), [])
                return FakeTensor([[value] for value in result])
            try:
                # Dispatch and finalization use the actual production module.
                results[rank] = run_group_reward_scoring(
                    trainers[rank], batches[rank], score, collective.for_rank(rank))
            except BaseException as exc:
                errors[rank] = exc
            finally:
                contexts[rank] = is_group_scoring_active()

        threads = [Thread(target=run, args=(rank,), daemon=True) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in threads), "A v5 peer skipped a collective")
        self.assertEqual(collective.counts, [2] * 4)
        self.assertEqual(contexts, {rank: False for rank in range(4)})
        self.assertEqual(paths, original_paths)
        return results, errors, trainers, calls, paths, collective, raw_errors

    def test_full_g8_uses_raw_errors_and_global_component_normalization(self):
        results, errors, trainers, calls, paths, collective, raw_errors = self.run_case()
        self.assertEqual(errors, {})
        rows = [row for trainer in trainers for row in trainer.reward_funcs[0].scorer.last_records]
        self.assertEqual(len({row["components"]["global_group_id"] for row in rows}), 1)
        self.assertEqual([row["components"]["group_candidate_index"] for row in rows], list(range(8)))
        quality_values = [-error for error in raw_errors]
        costs = [min(row["components"]["latency_seconds"], 10.) if error <= 5. else 10.
                 for row, error in zip(rows, raw_errors)]
        qm, lm = statistics.fmean(quality_values), statistics.fmean(costs)
        qscale, lscale = max(statistics.pstdev(quality_values), .5), max(statistics.pstdev(costs), .5)
        expected_rewards = []
        for index, (row, error) in enumerate(zip(rows, raw_errors)):
            components = row["components"]
            self.assertEqual(components["metricx_error"], error)
            self.assertEqual(components["metricx_quality_reward"], -error)
            self.assertEqual(components["quality"], 1. - error / 25.)
            self.assertIsNone(components["comet22_score"])
            self.assertEqual(components["quality_metric"], "metricx24_reference")
            self.assertEqual(components["quality_pass"], error <= 5.)
            self.assertEqual(components["latency_penalty"], costs[index])
            zq = max(-2., min(2., (quality_values[index] - qm) / qscale))
            zl = max(-2., min(2., (costs[index] - lm) / lscale))
            self.assertAlmostEqual(components["normalized_quality"], zq)
            self.assertAlmostEqual(components["normalized_latency"], zl)
            self.assertAlmostEqual(components["quality_group_mean"], qm)
            self.assertAlmostEqual(components["latency_group_mean"], lm)
            # All translations have equal lengths and no repetition.
            expected_rewards.append(zq - .5 * zl)
            self.assertAlmostEqual(row["reward"], expected_rewards[-1])
            self.assertEqual(components["group_size"], 8)
            self.assertEqual(components["group_eligible_count"], 8)
        average = statistics.fmean(expected_rewards)
        for row, expected in zip(rows, expected_rewards):
            self.assertAlmostEqual(row["components"]["estimated_advantage"], expected - average)
        for rank, trainer in enumerate(trainers):
            local = rows[rank * 2:rank * 2 + 2]
            self.assertEqual(results[rank].values, [[row["reward"]] for row in local])
            self.assertEqual([row["trajectory"] for row in local], paths[rank])
            self.assertEqual(trainer.reward_funcs[0].scorer._calls, 1)
            self.assertEqual(len(calls[rank]), 1)
            self.assertEqual(calls[rank][0][0], task()["source_text"])
            self.assertEqual(calls[rank][0][2], task()["reference_text"])
        metadata = [entry for envelope in collective.rounds[0].values() for entry in envelope["items"]]
        self.assertEqual(sorted(entry["item"]["metricx_error"] for entry in metadata), raw_errors)
        payloads = json.dumps(collective.payloads, ensure_ascii=False)
        for private in (task()["source_text"], task()["reference_text"], "�����ڱ�������Ԥ��", "LAST TURN"):
            self.assertNotIn(private, payloads)

    def test_mixed_invalid_candidates_keep_null_evidence_and_final_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            results, errors, trainers, calls, paths, _, _ = self.run_case(
                empty_indices={0}, invalid_indices={6}, log_root=directory)
            self.assertEqual(errors, {})
            rows = [row for trainer in trainers for row in trainer.reward_funcs[0].scorer.last_records]
            minimum = min(row["reward"] for row in rows if row["components"]["eligible"])
            for index, reason in ((0, "no measurable translation"), (6, "synthetic invalid protocol")):
                components = rows[index]["components"]
                self.assertAlmostEqual(rows[index]["reward"], minimum - .25)
                for field in ("quality", "metricx_error", "metricx_quality_reward", "latency_seconds",
                              "normalized_quality", "normalized_latency"):
                    self.assertIsNone(components[field])
                self.assertEqual(components["error"], reason)
                self.assertEqual(components["tier"], "invalid")
                self.assertEqual(components["group_eligible_count"], 6)
                self.assertLess(components["estimated_advantage"], 0.)
            self.assertAlmostEqual(sum(row["components"]["estimated_advantage"] for row in rows), 0.)
            self.assertEqual([len(call[0][1]) for call in calls], [1, 2, 2, 1])
            for rank, trainer in enumerate(trainers):
                log_files = list((Path(directory) / str(rank)).glob("*.jsonl"))
                self.assertEqual(len(log_files), 1)
                logged = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["reward"] for row in logged], [value[0] for value in results[rank].values])
                self.assertEqual([row["trajectory"] for row in logged], paths[rank])
                self.assertTrue(all(row["call"] == 1 and row["global_step"] == 3 for row in logged))
                self.assertTrue(all(row["components"]["group_finalized"] for row in logged))
                adapter = trainer.reward_funcs[0].scorer
                with self.assertRaisesRegex(RewardError, "duplicate"):
                    adapter.commit_group_rewards(adapter.last_records, trainer.state)

    def test_all_wait_skips_service_and_has_zero_advantages(self):
        results, errors, trainers, calls, *_ = self.run_case(empty_indices=set(range(8)))
        self.assertEqual(errors, {})
        self.assertEqual(calls, [[], [], [], []])
        for rank, trainer in enumerate(trainers):
            self.assertEqual(results[rank].values, [[-.25], [-.25]])
            for row in trainer.reward_funcs[0].scorer.last_records:
                self.assertEqual(row["components"]["estimated_advantage"], 0.)
                self.assertTrue(row["components"]["group_all_invalid"])
                self.assertIsNone(row["components"]["quality"])
                self.assertIsNone(row["components"]["metricx_error"])

    def test_scoring_and_commit_failures_reach_both_collectives_on_every_peer(self):
        for options in ({"service_fail": 2}, {"commit_fail": 1},
                        {"malformed_scores": (0, [3.])},
                        {"malformed_scores": (3, [25.1, 1.])}):
            with self.subTest(options=options):
                results, errors, *_ = self.run_case(**options)
                self.assertEqual(results, {})
                self.assertEqual(set(errors), {0, 1, 2, 3})
                self.assertTrue(all(isinstance(error, RewardError) for error in errors.values()))

    def test_second_standard_deviation_normalization_is_rejected_on_all_peers(self):
        results, errors, *_ = self.run_case(wrong_scale=3)
        self.assertEqual(results, {})
        self.assertEqual(set(errors), {0, 1, 2, 3})
        self.assertTrue(all(isinstance(error, RewardError) for error in errors.values()))


class V5ServiceIntegrationTests(unittest.TestCase):
    def test_real_service_adapter_scores_complete_reference_batch_and_deduplicates(self):
        paths = [trajectory("�����ڱ������ᣬ�������Ԥ�㡣"), trajectory("����������һ��Ĳ���Ԥ�㡣")]
        paths.append(copy.deepcopy(paths[0]))
        response = MagicMock(status_code=200)
        response.json.return_value = {"scores": [0., 25.]}
        with patch("flex_gspo.metricx_service.requests.Session") as factory, \
                patch("flex_gspo.comet22.ServiceCOMET22", side_effect=AssertionError("v5 must not build COMET")):
            session = factory.return_value.__enter__.return_value
            session.post.return_value = response
            scorer = QalcoScorer(config())
            self.assertIsInstance(scorer.quality_model, ServiceMetricX)
            factory.assert_not_called()
            adapter = OnlineQalcoReward(config(), scorer=scorer, log_dir="")
            rewards = adapter(["THIS LAST TURN MUST NOT BE SCORED"] * 3,
                              flex_task=[task()] * 3,
                              rollout_infos=[{"flex_trajectory": path} for path in paths])
            session.post.assert_called_once_with(
                "http://127.0.0.1:16999/metricx24",
                json={"srcs": [task()["source_text"]] * 2,
                      "mts": ["�����ڱ������ᣬ�������Ԥ�㡣", "����������һ��Ĳ���Ԥ�㡣"],
                      "refs": [task()["reference_text"]] * 2},
                timeout=(10, 180), allow_redirects=False)
            self.assertIs(session.trust_env, False)
            response.close.assert_called_once()
        self.assertEqual(rewards, [0., 0., 0.])
        self.assertEqual([row["components"]["metricx_error"] for row in adapter.last_records], [0., 25., 0.])
        self.assertEqual([row["components"]["quality"] for row in adapter.last_records], [1., 0., 1.])
        self.assertEqual([row["components"]["metricx_quality_reward"] for row in adapter.last_records], [0., -25., 0.])
        self.assertTrue(all(row["components"]["comet22_score"] is None for row in adapter.last_records))
        self.assertTrue(all(not row["components"]["group_finalized"] for row in adapter.last_records))
        self.assertEqual(adapter._calls, 0)
        with self.assertRaisesRegex(RewardError, "not finalized"):
            adapter(["ignored"], flex_task=[task()], rollout_infos=[{"flex_trajectory": paths[0]}])


if __name__ == "__main__":
    unittest.main()
