"""CPU-only complete-group bridge and failure-participation regression tests."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from threading import Condition, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo.group_reward import (PATCH_ID, _local_metadata, finalize_global_metadata,
                                    is_group_scoring_active, run_group_reward_scoring,
                                    validate_group_reward_source)
from flex_gspo.rewards import RewardError


ROOT = Path(__file__).resolve().parents[1]
SWIFT = Path(os.environ.get("FLEX_SWIFT_SOURCE", ROOT.parent / "tmp" / "omni-rl-audit-swift-20261015"))


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]) if values else 1)

    def new_tensor(self, values):
        return FakeTensor(values)


class Collective:
    """Blocking two-round world collective; missing peers fail promptly."""
    def __init__(self, size=4):
        self.size = size
        self.condition = Condition()
        self.rounds = {}
        self.counts = [0] * size
        self.payloads = []

    def for_rank(self, rank):
        def gather(output, value):
            with self.condition:
                ordinal = self.counts[rank]
                self.counts[rank] += 1
                round_values = self.rounds.setdefault(ordinal, {})
                round_values[rank] = copy.deepcopy(value)
                self.payloads.append(copy.deepcopy(value))
                self.condition.notify_all()
                deadline = time.monotonic() + 4
                while len(round_values) != self.size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError("Collective timed out: a peer skipped the group reward boundary")
                    self.condition.wait(remaining)
                output[:] = [copy.deepcopy(round_values[index]) for index in range(self.size)]
        return SimpleNamespace(is_initialized=lambda: True, get_rank=lambda: rank,
                               get_world_size=lambda: self.size, all_gather_object=gather)


def _task(name):
    return {"id": name, "source_text": "Private source text never sent in the collective.",
            "reference_text": "Private reference never sent in the collective.", "audio_path": "/private.wav"}


def _sample(task, request, prompt):
    return SimpleNamespace(extra={"flex_task": json.dumps(task)}, request_id=request, prompt_id=prompt)


def _record(task, index, request, prompt, units, *, eligible=True):
    digest = hashlib.sha256(json.dumps(task, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode("utf-8")).hexdigest()
    return {"task_id": task["id"], "task_hash": digest,
            "candidate_index_in_local_batch": index, "request_id": request, "prompt_id": prompt,
            "trajectory": {"turns": [{"response": "Private raw translation"}]},
            "translation": "Private raw translation", "reward": 0.0 if eligible else -4.0,
            "components": {"eligible": eligible, "quality": .9 if eligible else None,
                           "latency_seconds": 1.0 if eligible else None, "emitted_units": units,
                           "repetition_penalty": 0.0, "protocol_valid": eligible,
                           "tier": "provisional" if eligible else "invalid",
                           "error": None if eligible else "original policy error"}}


def _settings():
    return {"provider": "qalco", "version": 3, "quality_floor": .8}


class Delegate:
    def __init__(self, records, settings=None, *, commit_fail=False):
        self.records = records
        self.settings = _settings() if settings is None else settings
        self.last_records = []
        self.committed = None
        self.commit_fail = commit_fail

    def commit_group_rewards(self, records, trainer_state):
        if self.commit_fail:
            raise OSError("Local log device unavailable")
        self.committed = records
        self.last_records = records


def _trainer(rank, records, *, size=4, settings=None, commit_fail=False):
    delegate = Delegate(records, settings, commit_fail=commit_fail)
    return SimpleNamespace(reward_funcs=[SimpleNamespace(scorer=delegate)],
                           reward_func_names=["QalcoOnlineORM"], reward_weights=[1.0],
                           dynamic_sample=False, use_gym_env=False, scale_rewards="group",
                           num_generations=4, generation_batch_size=8,
                           process_index=rank, world_size=size, state=SimpleNamespace(global_step=3))


def _finalizer(items, settings):
    minimum = min(item["units"] for item in items if item["eligible"]) if any(
        item["eligible"] for item in items) else None
    return [{**item, "reward": item["quality"] - .01 * (item["units"] - minimum)
             if item["eligible"] else item["reward"], "group_length_anchor": minimum}
            for item in items]


class GroupBridgeTests(unittest.TestCase):
    def run_case(self, *, mutation=None, score_fail=None, finalize_fail=None, commit_fail=None):
        collective = Collective()
        tasks = [_task("A"), _task("B")]
        trainers, batches = [], []
        for rank in range(4):
            task = tasks[rank // 2]
            prompt = "prompt-" + task["id"]
            samples = [_sample(task, f"r{rank}-{index}", prompt) for index in range(2)]
            records = [_record(task, index, sample.request_id, sample.prompt_id, 10 + rank * 2 + index)
                       for index, sample in enumerate(samples)]
            trainers.append(_trainer(rank, records, commit_fail=rank == commit_fail))
            batches.append(samples)
        if mutation:
            mutation(trainers, batches)
        results, errors = {}, {}
        after_context = {}

        def run(rank):
            trainer = trainers[rank]
            def score(current, samples):
                self.assertTrue(is_group_scoring_active())
                if rank == score_fail:
                    raise ValueError("Sensitive input must not enter metadata envelopes")
                current.reward_funcs[0].scorer.last_records = copy.deepcopy(
                    current.reward_funcs[0].scorer.records)
                return FakeTensor([[0.0] for sample in samples])
            def finalize(items, settings):
                if rank == finalize_fail:
                    raise ValueError("A rank-local finalizer defect")
                return _finalizer(items, settings)
            try:
                results[rank] = run_group_reward_scoring(
                    trainer, batches[rank], score, collective.for_rank(rank), finalizer=finalize)
            except BaseException as exc:
                errors[rank] = exc
            finally:
                after_context[rank] = is_group_scoring_active()
        threads = [Thread(target=run, args=(rank,), daemon=True) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in threads), "A group reward peer hung")
        self.assertEqual(collective.counts, [2, 2, 2, 2])
        self.assertEqual(after_context, {0: False, 1: False, 2: False, 3: False})
        return results, errors, trainers, collective

    def test_cross_rank_complete_groups_and_raw_evidence_unchanged(self):
        results, errors, trainers, collective = self.run_case()
        self.assertEqual(errors, {})
        self.assertEqual(results[0].values, [[.9], [.89]])
        self.assertEqual(results[1].values, [[.88], [.87]])
        # Group B gets its own anchor 14, not the group A anchor 10.
        self.assertEqual(results[2].values, [[.9], [.89]])
        for rank, trainer in enumerate(trainers):
            delegate = trainer.reward_funcs[0].scorer
            self.assertEqual(delegate.committed[0]["trajectory"], delegate.records[0]["trajectory"])
            self.assertEqual(delegate.committed[0]["translation"], delegate.records[0]["translation"])
            self.assertTrue(delegate.committed[0]["components"]["group_finalized"])
            self.assertEqual(delegate.committed[0]["components"]["group_candidate_index"], (rank % 2) * 2)
        transport = json.dumps(collective.payloads)
        for private in ("Private source", "Private reference", "Private raw", "/private.wav"):
            self.assertNotIn(private, transport)

    def test_local_service_failure_is_exchanged_by_every_rank(self):
        results, errors, trainers, collective = self.run_case(score_fail=1)
        self.assertEqual(results, {})
        self.assertEqual(set(errors), {0, 1, 2, 3})
        self.assertTrue(all(isinstance(error, RewardError) for error in errors.values()))
        self.assertNotIn("Sensitive input", json.dumps(collective.payloads))
        self.assertIsInstance(errors[1].__cause__, ValueError)

    def test_local_finalization_failure_reaches_status_collective(self):
        results, errors, _, _ = self.run_case(finalize_fail=2)
        self.assertEqual(results, {})
        self.assertEqual(set(errors), {0, 1, 2, 3})

    def test_local_commit_failure_reaches_status_collective(self):
        results, errors, _, _ = self.run_case(commit_fail=3)
        self.assertEqual(results, {})
        self.assertEqual(set(errors), {0, 1, 2, 3})

    def test_mismatched_prompt_instance_rejected(self):
        def mutate(trainers, batches):
            batches[1][0].prompt_id = "another-prompt"
            trainers[1].reward_funcs[0].scorer.records[0]["prompt_id"] = "another-prompt"
        _, errors, _, _ = self.run_case(mutation=mutate)
        self.assertEqual(len(errors), 4)

    def test_same_task_hash_does_not_merge_separate_contiguous_draws(self):
        def mutate(trainers, batches):
            for rank in (2, 3):
                for index, sample in enumerate(batches[rank]):
                    task = _task("A")
                    sample.extra["flex_task"] = json.dumps(task)
                    sample.prompt_id = "second-draw"
                    trainers[rank].reward_funcs[0].scorer.records[index] = _record(
                        task, index, sample.request_id, sample.prompt_id, 30 + rank + index)
        results, errors, trainers, _ = self.run_case(mutation=mutate)
        self.assertEqual(errors, {})
        self.assertEqual(results[2].values[0], [.9])
        first = trainers[0].reward_funcs[0].scorer.committed[0]["components"]
        second = trainers[2].reward_funcs[0].scorer.committed[0]["components"]
        self.assertNotEqual(first["global_group_id"], second["global_group_id"])
        self.assertEqual(second["group_length_anchor"], 32)

    def test_duplicate_request_ids_rejected_globally(self):
        def mutate(trainers, batches):
            batches[3][1].request_id = "r0-0"
            trainers[3].reward_funcs[0].scorer.records[1]["request_id"] = "r0-0"
        _, errors, _, _ = self.run_case(mutation=mutate)
        self.assertEqual(len(errors), 4)

    def test_rank_local_identity_conflict_still_collective(self):
        def mutate(trainers, batches):
            batches[1][1].request_id = "not-the-scored-candidate"
        _, errors, _, _ = self.run_case(mutation=mutate)
        self.assertEqual(len(errors), 4)

    def test_config_mismatch_and_dynamic_sampling_rejected_collectively(self):
        for field in ("config", "dynamic", "weights", "shape", "hash"):
            with self.subTest(field=field):
                def mutate(trainers, batches):
                    if field == "config":
                        trainers[1].reward_funcs[0].scorer.settings["quality_floor"] = .81
                    elif field == "dynamic":
                        trainers[1].dynamic_sample = True
                    elif field == "weights":
                        trainers[1].reward_weights = [.5]
                    elif field == "shape":
                        trainers[1].generation_batch_size = 12
                    else:
                        trainers[1].reward_funcs[0].scorer.records[0]["task_hash"] = "0" * 64
                _, errors, _, _ = self.run_case(mutation=mutate)
                self.assertEqual(len(errors), 4)

    def test_invalid_candidate_keeps_original_reward_and_policy_error(self):
        def mutate(trainers, batches):
            sample = batches[0][0]
            trainers[0].reward_funcs[0].scorer.records[0] = _record(
                _task("A"), 0, sample.request_id, sample.prompt_id, 0, eligible=False)
        results, errors, trainers, _ = self.run_case(mutation=mutate)
        self.assertEqual(errors, {})
        self.assertEqual(results[0].values[0], [-4.0])
        components = trainers[0].reward_funcs[0].scorer.committed[0]["components"]
        self.assertEqual(components["error"], "original policy error")
        self.assertEqual(components["tier"], "invalid")
        self.assertIsNone(components["quality"])

    def test_missing_record_ids_are_supplemented_from_samples(self):
        def mutate(trainers, batches):
            for trainer in trainers:
                for record in trainer.reward_funcs[0].scorer.records:
                    record.pop("prompt_id")
                    record.pop("request_id")
        _, errors, trainers, _ = self.run_case(mutation=mutate)
        self.assertEqual(errors, {})
        self.assertEqual(trainers[2].reward_funcs[0].scorer.committed[1]["request_id"], "r2-1")

    def test_full_task_identity_includes_more_than_id(self):
        def mutate(trainers, batches):
            changed = _task("A")
            changed["audio_path"] = "/different.wav"
            sample = batches[1][0]
            sample.extra["flex_task"] = json.dumps(changed)
            trainers[1].reward_funcs[0].scorer.records[0] = _record(
                changed, 0, sample.request_id, sample.prompt_id, 12)
        _, errors, _, _ = self.run_case(mutation=mutate)
        self.assertEqual(len(errors), 4)

    def test_unequal_local_counts_rejected_without_skipping_peers(self):
        def mutate(trainers, batches):
            batches[1].pop()
            trainers[1].reward_funcs[0].scorer.records.pop()
        _, errors, _, _ = self.run_case(mutation=mutate)
        self.assertEqual(len(errors), 4)

    def test_collective_itself_failure_never_attempts_second_collective(self):
        dist = Collective(size=1).for_rank(0)
        calls = []
        def broken(out, data):
            calls.append(data)
            raise OSError("Process group failed")
        dist.all_gather_object = broken
        sample = _sample(_task("A"), "id", "p")
        record = _record(_task("A"), 0, "id", "p", 5)
        trainer = _trainer(0, [record], size=1)
        def score(current, samples):
            current.reward_funcs[0].scorer.last_records = [record]
            return FakeTensor([[0]])
        with self.assertRaises(OSError):
            run_group_reward_scoring(trainer, [sample], score, dist, finalizer=_finalizer)
        self.assertEqual(len(calls), 1)


class SourceContractTests(unittest.TestCase):
    @unittest.skipUnless((SWIFT / "swift/megatron/trainers/grpo_trainer.py").exists(),
                         "Set FLEX_SWIFT_SOURCE to the audited checkout")
    def test_pinned_source_and_changed_gather(self):
        source = (SWIFT / "swift/megatron/trainers/grpo_trainer.py").read_text(encoding="utf-8")
        result = validate_group_reward_source(source)
        self.assertEqual(result["patch_id"], PATCH_ID)
        self.assertEqual(result["collectives_per_score"], 2)
        with self.assertRaises(RewardError):
            validate_group_reward_source(source.replace("gather(rewards_per_func)",
                                                        "gather(rewards_per_func, group=other)"))

    def test_no_heavy_dependency_imported_for_cpu_metadata_helpers(self):
        self.assertEqual(_local_metadata([], []), [])
        with self.assertRaises(RewardError):
            finalize_global_metadata([], _settings(), _finalizer)

    def test_real_online_adapter_task_hash_and_deferred_math_commit(self):
        from flex_gspo.gated_reward import finalize_group
        from flex_gspo.online_reward import OnlineQalcoReward
        from flex_gspo.qalco import QalcoScorer

        class Quality:
            def __init__(self):
                self.calls = 0

            def score_batch(self, source, hypotheses, reference):
                self.calls += 1
                return [.9] * len(hypotheses)

        config = {"reward": {"provider": "qalco", "version": 3, "quality_provider": "comet22",
                             "quality_floor": .8, "latency_budget_seconds": 3,
                             "comet22": {"backend": "service", "service_url": "http://127.0.0.1:6668/22",
                                         "service_id": "cpu-test", "response_key": "score"}}}
        task = {"id": "real-adapter", "source_lang": "en", "target_lang": "zh",
                "source_text": "Today we are meeting in Beijing.", "reference_text": "今天我们在北京开会。",
                "duration_seconds": 2., "chunks": [{"end_seconds": 1.}, {"end_seconds": 2.}]}
        translations = ["我们在北京开会。", "我们今天在北京开会。", "我们今天正在北京开会。", "我们今天正在北京举行会议。"]
        trajectories = [{"schema_version": 2, "valid": True, "complete_input": True,
                         "episode_complete": True, "eof_completed": True, "turns": [
                             {"kind": "audio", "response": "<|WAIT|>", "audio_end_seconds": 1.,
                              "finish_reason": "stop"},
                             {"kind": "audio", "response": "<|DIRECT|>" + translation,
                              "audio_end_seconds": 2., "finish_reason": "stop"},
                             {"kind": "eof", "response": "<|WAIT|>", "audio_end_seconds": 2.,
                              "finish_reason": "stop", "eof_elapsed_seconds": .01}]}
                        for translation in translations]
        quality = Quality()
        with patch("flex_gspo.online_reward.LOG.warning"):
            adapter = OnlineQalcoReward(config, scorer=QalcoScorer(config, quality_model=quality), log_dir="")
        samples = [_sample(task, f"candidate-{index}", "prompt-real") for index in range(4)]
        trainer = _trainer(0, [], size=1)
        trainer.generation_batch_size = trainer.num_generations = 4
        trainer.reward_funcs = [SimpleNamespace(scorer=adapter)]
        collective = Collective(size=1)

        def original(current, values):
            rewards = adapter(["last response is WAIT"] * 4, flex_task=[task] * 4,
                              rollout_infos=[{"flex_trajectory": trajectory} for trajectory in trajectories],
                              request_id=[sample.request_id for sample in values],
                              prompt_id=[sample.prompt_id for sample in values], trainer_state=current.state)
            self.assertEqual(adapter._calls, 0, "Provisional scores must not have been committed")
            self.assertTrue(all(not row["components"]["group_finalized"] for row in adapter.last_records))
            return FakeTensor([[reward] for reward in rewards])

        result = run_group_reward_scoring(trainer, samples, original, collective.for_rank(0), finalizer=finalize_group)
        self.assertEqual(result.shape, (4, 1))
        self.assertEqual(collective.counts, [2])
        self.assertEqual(quality.calls, 1)
        self.assertEqual(adapter._calls, 1)
        self.assertEqual([record["translation"] for record in adapter.last_records], translations)
        self.assertTrue(all(record["components"]["group_finalized"] for record in adapter.last_records))
        self.assertTrue(all(record["components"]["anchor_pool_size"] == 4 for record in adapter.last_records))
        self.assertEqual([record["trajectory"]["turns"] for record in adapter.last_records],
                         [trajectory["turns"] for trajectory in trajectories])

    def test_default_g8_b32_four_rank_real_adapter_and_finalizer(self):
        from flex_gspo.gated_reward import finalize_group
        from flex_gspo.online_reward import OnlineQalcoReward, load_reward_config
        from flex_gspo.qalco import QalcoScorer

        class Quality:
            def __init__(self):
                self.calls = 0

            def score_batch(self, source, hypotheses, reference):
                self.calls += 1
                return [.9] * len(hypotheses)

        config = load_reward_config()
        config["reward"].update(version=3, quality_provider="comet22")
        for legacy in ("reference_length_ratio", "latency_weight", "length_weight"):
            config["reward"].pop(legacy, None)
        options = ["我们在北京开会。", "我们今天在北京开会。", "今天我们正在北京开会。",
                   "今天我们在北京举行会议。", "今天我们在北京举行年度会议。",
                   "今天我们正在北京举行年度会议。", "今天我们正在北京举行年度预算会议。",
                   "今天我们在北京开会，讨论明年的预算、具体安排和需要解决的其他事项。"]
        tasks, trajectories, batches, trainers, qualities = [], [], [], [], []
        for rank in range(4):
            task = {"id": f"g8-task-{rank}", "source_lang": "en", "target_lang": "zh",
                    "source_text": "Today we meet in Beijing and discuss the annual budget.",
                    "reference_text": "今天我们在北京开会，讨论年度预算。", "duration_seconds": 2.,
                    "chunks": [{"end_seconds": 1.}, {"end_seconds": 2.}]}
            current = []
            for index in range(8):
                empty = rank == 0 or (rank == 1 and index < 2)
                current.append({"schema_version": 2, "valid": True, "complete_input": True,
                                "episode_complete": True, "eof_completed": True, "turns": [
                                    {"kind": "audio", "response": "<|WAIT|>", "audio_end_seconds": 1.,
                                     "finish_reason": "stop"},
                                    {"kind": "audio", "response": "<|WAIT|>" if empty else "<|DIRECT|>" + options[index],
                                     "audio_end_seconds": 2., "finish_reason": "stop"},
                                    {"kind": "eof", "response": "<|WAIT|>", "audio_end_seconds": 2.,
                                     "finish_reason": "stop", "eof_elapsed_seconds": .01}]})
            quality = Quality()
            with patch("flex_gspo.online_reward.LOG.warning"):
                adapter = OnlineQalcoReward(config, scorer=QalcoScorer(config, quality_model=quality), log_dir="")
            trainer = _trainer(rank, [])
            trainer.num_generations = 8
            trainer.generation_batch_size = 32
            trainer.reward_funcs = [SimpleNamespace(scorer=adapter)]
            tasks.append(task)
            trajectories.append(current)
            batches.append([_sample(task, f"g8-r{rank}-c{index}", f"g8-prompt-{rank}") for index in range(8)])
            trainers.append(trainer)
            qualities.append(quality)
        collective = Collective()
        results, errors, contexts = {}, {}, {}

        def run(rank):
            def original(current, samples):
                adapter = current.reward_funcs[0].scorer
                values = adapter(["LAST RESPONSE NOT FULL TRANSLATION"] * 8,
                                 flex_task=[tasks[rank]] * 8,
                                 rollout_infos=[{"flex_trajectory": value} for value in trajectories[rank]],
                                 request_id=[sample.request_id for sample in samples],
                                 prompt_id=[sample.prompt_id for sample in samples], trainer_state=current.state)
                self.assertEqual(adapter._calls, 0)
                return FakeTensor([[value] for value in values])
            try:
                results[rank] = run_group_reward_scoring(
                    trainers[rank], batches[rank], original, collective.for_rank(rank), finalizer=finalize_group)
            except BaseException as exc:
                errors[rank] = exc
            finally:
                contexts[rank] = is_group_scoring_active()
        threads = [Thread(target=run, args=(rank,), daemon=True) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, {})
        self.assertEqual(collective.counts, [2, 2, 2, 2])
        self.assertEqual(contexts, {0: False, 1: False, 2: False, 3: False})
        group_ids = set()
        for rank in range(4):
            adapter = trainers[rank].reward_funcs[0].scorer
            self.assertEqual(adapter._calls, 1)
            self.assertEqual(results[rank].values, [[record["reward"]] for record in adapter.last_records])
            self.assertEqual([record["trajectory"]["turns"] for record in adapter.last_records],
                             [trajectory["turns"] for trajectory in trajectories[rank]])
            group_ids.update(record["components"]["global_group_id"] for record in adapter.last_records)
            self.assertEqual({record["components"]["global_group_index"] for record in adapter.last_records}, {rank})
            self.assertTrue(all(record["components"]["group_size"] == 8 for record in adapter.last_records))
        self.assertEqual(len(group_ids), 4)
        self.assertEqual(qualities[0].calls, 0, "An all-empty group must not call quality scoring")
        self.assertEqual([record["reward"] for record in trainers[0].reward_funcs[0].scorer.last_records], [-4.] * 8)
        for record in trainers[0].reward_funcs[0].scorer.last_records:
            self.assertEqual(record["components"]["anchor_pool_size"], 0)
            self.assertEqual(record["components"]["length_penalty"], 0.)
            self.assertIsNone(record["components"]["quality"])
        partial = trainers[1].reward_funcs[0].scorer.last_records
        for record in partial[:2]:
            self.assertEqual(record["reward"], -4.)
            self.assertEqual(record["components"]["tier"], "invalid")
            self.assertEqual(record["components"]["error"], "no measurable translation")
            self.assertEqual(record["components"]["group_brevity_excess"], 0.)
            self.assertEqual(record["components"]["length_penalty"], 0.)
            self.assertEqual(record["components"]["anchor_pool_size"], 6)
            self.assertIsNone(record["components"]["latency_seconds"])
        self.assertGreater(trainers[2].reward_funcs[0].scorer.last_records[-1]["components"]["group_brevity_excess"], 0.)


if __name__ == "__main__":
    unittest.main()
