"""Actual v4 ORM and complete-G8 finalizer across four CPU collective peers."""
import copy
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
import unittest

from flex_gspo.group_reward import run_group_reward_scoring, validate_v4_advantage_source
from flex_gspo.online_reward import OnlineQalcoReward
from flex_gspo.qalco import QalcoScorer, combine_reward, validate_settings
from flex_gspo.rewards import RewardError
from test_group_reward import Collective, FakeTensor, _sample, _trainer, SWIFT
from test_qalco_v3_integration import task, trajectory


def config():
    return {"reward": {"provider": "qalco", "version": 4, "quality_provider": "comet22",
        "comet22": {"backend": "service", "service_url": "http://127.0.0.1:6668/22",
                    "service_id": "cpu-only", "response_key": "score"}}}


class V4RuntimeTests(unittest.TestCase):
    def run_case(self, *, empty_indices=(), service_fail=None, wrong_scale=None, commit_fail=None):
        collective = Collective()
        source = task()
        translations = [f"我们在北京讨论预算方案{i}。" for i in range(8)]
        scores = {text: .65 + index * .04 for index, text in enumerate(translations)}
        trainers, batches, paths, calls = [], [], [], [0] * 4
        for rank in range(4):
            class Quality:
                def __init__(self, rank):
                    self.rank = rank

                def score_batch(self, src, hypotheses, ref):
                    calls[self.rank] += 1
                    if self.rank == service_fail:
                        raise RewardError("service unavailable, no fake score")
                    return [scores[text] for text in hypotheses]

            adapter = OnlineQalcoReward(config(), scorer=QalcoScorer(config(), quality_model=Quality(rank)), log_dir="")
            trainer = _trainer(rank, [])
            trainer.num_generations = trainer.generation_batch_size = 8
            trainer.scale_rewards = "group" if rank == wrong_scale else "none"
            trainer.advantage_estimator = "grpo"
            trainer.kl_in_reward = False
            trainer.reward_funcs = [SimpleNamespace(scorer=adapter)]
            if rank == commit_fail:
                def broken_commit(*args, **kwargs):
                    raise OSError("disk unavailable")
                adapter.commit_group_rewards = broken_commit
            trainers.append(trainer)
            indices = [rank * 2, rank * 2 + 1]
            batches.append([_sample(source, f"request-{i}", "one-full-g8") for i in indices])
            paths.append([trajectory("" if i in empty_indices else translations[i]) for i in indices])
        results, errors = {}, {}

        def run(rank):
            def score(current, samples):
                adapter = current.reward_funcs[0].scorer
                result = adapter(["LAST TURN IS NOT THE TRANSLATION"] * 2, flex_task=[source] * 2,
                                 rollout_infos=[{"flex_trajectory": tr} for tr in paths[rank]],
                                 request_id=[s.request_id for s in samples],
                                 prompt_id=[s.prompt_id for s in samples])
                self.assertEqual(adapter._calls, 0)
                self.assertTrue(all(not r["components"]["group_finalized"] for r in adapter.last_records))
                return FakeTensor([[r] for r in result])
            try:
                # No injected finalizer: exercise version dispatch in production.
                results[rank] = run_group_reward_scoring(trainers[rank], batches[rank], score, collective.for_rank(rank))
            except BaseException as exc:
                errors[rank] = exc
        threads = [Thread(target=run, args=(r,), daemon=True) for r in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(collective.counts, [2] * 4)
        return results, errors, trainers, calls, paths

    def test_full_g8_cross_rank_mixed_empty_preserves_null_and_error(self):
        results, errors, trainers, calls, paths = self.run_case(empty_indices={0, 6})
        self.assertEqual(errors, {})
        rows = [row for t in trainers for row in t.reward_funcs[0].scorer.last_records]
        self.assertEqual(len({r["components"]["global_group_id"] for r in rows}), 1)
        self.assertEqual([r["components"]["group_candidate_index"] for r in rows], list(range(8)))
        self.assertAlmostEqual(sum(r["components"]["estimated_advantage"] for r in rows), 0.)
        lowest = min(r["reward"] for r in rows if r["components"]["eligible"])
        for index in (0, 6):
            c = rows[index]["components"]
            self.assertAlmostEqual(rows[index]["reward"], lowest - .25)
            self.assertIsNone(c["quality"])
            self.assertIsNone(c["latency_seconds"])
            self.assertIsNone(c["normalized_quality"])
            self.assertEqual(c["tier"], "invalid")
            self.assertEqual(c["error"], "no measurable translation")
            self.assertEqual(c["group_eligible_count"], 6)
            self.assertLess(c["estimated_advantage"], 0)
        for rank, trainer in enumerate(trainers):
            self.assertEqual(results[rank].values, [[r["reward"]] for r in rows[rank * 2:rank * 2 + 2]])
            self.assertEqual([r["trajectory"]["turns"] for r in trainer.reward_funcs[0].scorer.last_records],
                             [tr["turns"] for tr in paths[rank]])
            self.assertEqual(trainer.reward_funcs[0].scorer._calls, 1)

    def test_all_wait_no_service_calls_and_zero_advantage(self):
        _, errors, trainers, calls, _ = self.run_case(empty_indices=set(range(8)))
        self.assertEqual(errors, {})
        self.assertEqual(calls, [0] * 4)
        for trainer in trainers:
            for row in trainer.reward_funcs[0].scorer.last_records:
                self.assertEqual(row["reward"], -.25)
                self.assertEqual(row["components"]["estimated_advantage"], 0.)
                self.assertTrue(row["components"]["group_all_invalid"])
                self.assertIsNone(row["components"]["quality"])

    def test_service_and_commit_errors_finish_both_collectives(self):
        for kw in ({"service_fail": 2}, {"commit_fail": 1}):
            with self.subTest(kw=kw):
                results, errors, *_ = self.run_case(**kw)
                self.assertEqual(results, {})
                self.assertEqual(set(errors), {0, 1, 2, 3})
                self.assertTrue(all(isinstance(e, RewardError) for e in errors.values()))

    def test_extra_group_std_rejected_without_peer_hang(self):
        results, errors, *_ = self.run_case(wrong_scale=3)
        self.assertEqual(results, {})
        self.assertEqual(set(errors), {0, 1, 2, 3})

    def test_version_keys_and_historical_combiner_rejected(self):
        for key in ("v3", "reference_length_ratio", "latency_weight", "length_weight"):
            bad = config()
            bad["reward"][key] = {} if key == "v3" else 1.2
            with self.assertRaises(RewardError):
                validate_settings(bad)
        with self.assertRaises(RewardError):
            combine_reward(.9, 1., 0., validate_settings(config()))
        self.assertEqual(validate_settings(config())["version"], 4)

    @unittest.skipUnless((SWIFT / "swift/rl_core/advantage.py").exists(), "Audited SWIFT source unavailable")
    def test_center_only_source_contract_and_mutation(self):
        source = (SWIFT / "swift/rl_core/advantage.py").read_text(encoding="utf-8")
        self.assertFalse(validate_v4_advantage_source(source)["second_std_normalization"])
        changed = source.replace("std is not None and scale_rewards != 'none'", "std is not None")
        with self.assertRaises(RewardError):
            validate_v4_advantage_source(changed)


if __name__ == "__main__":
    unittest.main()
