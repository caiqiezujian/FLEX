"""Run the actual upstream turn loop on four or eight simulated ranks.

The simulator fails promptly on a departed collective participant; it does not
need CUDA, NCCL, model weights or a distributed process group. The loop and the
token-history scheduler are compiled from the audited SWIFT source, not copied.
"""
from __future__ import annotations

from abc import ABC
import ast
import asyncio
import copy
import json
import os
from pathlib import Path
from threading import Condition, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo.audio import encode_chunk_ref
from flex_gspo.online import FlexStreamHooks
from flex_gspo.prompts import initial_messages
from flex_gspo.rollout_sync import make_completion_gather


ROOT = Path(__file__).resolve().parents[1]
SWIFT = Path(os.environ.get("FLEX_SWIFT_SOURCE", ROOT.parent / "tmp" / "omni-rl-audit-swift-20261015"))
WORLD = (0, 1, 2, 3)
OLD_GROUPS = ((0, 1), (0, 1), (2, 3), (2, 3))


def _compile_definitions(relative, names, namespace):
    path = SWIFT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Upstream contract changed: {relative}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


class CollectiveParticipationError(RuntimeError):
    pass


class Collectives:
    """Blocking all-gather semantics, with explicit collective-order checks."""

    def __init__(self, ranks=WORLD):
        self.condition = Condition()
        self.exited = set()
        self.counters = {}
        self.rounds = {}
        self.inference_calls = {rank: [] for rank in ranks}
        self.engine_requests = {rank: [] for rank in ranks}

    def gather(self, rank, group, kind, values):
        group = tuple(group)
        with self.condition:
            if rank not in group:
                raise AssertionError("Rank is not a collective participant")
            ordinal = self.counters.get((rank, group), 0)
            self.counters[rank, group] = ordinal + 1
            key = group, ordinal
            state = self.rounds.setdefault(key, {"kind": kind, "values": {}})
            if state["kind"] != kind:
                raise CollectiveParticipationError("Ranks entered different collectives at the same ordinal")
            state["values"][rank] = list(values)
            self.condition.notify_all()
            deadline = time.monotonic() + 5
            while len(state["values"]) != len(group):
                departed = set(group).intersection(self.exited).difference(state["values"])
                if departed:
                    raise CollectiveParticipationError(f"Collective {kind} still needs exited ranks {sorted(departed)}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CollectiveParticipationError(f"Collective {kind} timed out; arrived {sorted(state['values'])}")
                self.condition.wait(remaining)
            return [value for peer in group for value in state["values"][peer]]

    def exit(self, rank):
        with self.condition:
            self.exited.add(rank)
            self.condition.notify_all()


class Tokenizer:
    def decode(self, ids, **kwargs):
        pieces = {151676: "<|WAIT|>", 151645: "<|im_end|>", 3: "BAD"}
        return "".join(pieces[token] for token in ids)

    def encode(self, text, **kwargs):
        if text:
            raise AssertionError("No deterministic assistant prefix expected")
        return []


def _request(rank, turns):
    task = {
        "schema_version": 1, "id": f"rank-{rank}", "audio_path": "/not/read.wav", "sample_rate": 16000,
        "source_lang": "en", "target_lang": "zh", "source_text": "Hello.", "reference_text": "你好。",
        "quality_ready": True, "duration_seconds": float(turns),
        "chunks": [{"start_seconds": float(i), "end_seconds": float(i + 1), "input_seconds": 1.}
                   for i in range(turns)],
    }
    return SimpleNamespace(uuid=f"candidate-{rank}", data_dict={"flex_task": json.dumps(task)},
                           messages=initial_messages(), audios=[encode_chunk_ref(task, 0)],
                           images=[], videos=[], chat_template_kwargs={})


def _output(invalid=False, owner=None):
    ids = [3, 151645] if invalid else [151676, 151645]
    choice = SimpleNamespace(message=SimpleNamespace(content="BAD" if invalid else "<|WAIT|>"),
                             token_ids=ids, finish_reason="stop",
                             logprobs={"content": [{"logprob": -.1} for _ in ids]})
    return SimpleNamespace(response=SimpleNamespace(choices=[choice], owner=owner))


@unittest.skipUnless((SWIFT / "swift/rollout/agent_loop.py").exists(), "Set FLEX_SWIFT_SOURCE to the audited checkout")
class FourRankRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def remove_response(messages):
            if messages[-1]["role"] == "assistant":
                messages.pop()

        namespace = {"ABC": ABC, "deepcopy": copy.deepcopy, "copy": copy, "asyncio": asyncio,
                     "RolloutOutput": SimpleNamespace, "remove_response": remove_response,
                     "get_token_backed_response_ids": lambda value: value if isinstance(value, list) else None}
        _compile_definitions("swift/rollout/multi_turn.py", ["RolloutScheduler", "MultiTurnScheduler"], namespace)
        _compile_definitions("swift/rollout/agent_loop.py", ["extract_logprobs_from_choice", "_identity_gather",
                             "_run_multi_turn_impl"], namespace)
        cls.namespace = namespace
        # Exercise the real TP gather/slice implementation in the eight-rank
        # tests; a fake engine only supplies deterministic model responses.
        path = SWIFT / "swift/megatron/trainers/rollout_mixin.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mixin, = [node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "MegatronRolloutMixin"]
        method, = [node for node in mixin.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_colocate_rollout"]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
        cls.colocate_code = compile(module, str(path), "exec")

    def run_case(self, lengths, stop_groups=None, invalid_ranks=(), engine_groups=None):
        ranks = tuple(range(len(lengths)))
        actual_colocate = engine_groups is not None
        engine_groups = engine_groups or (ranks,) * len(ranks)
        collective = Collectives(ranks)
        results, errors = {}, {}
        scheduler_type = type("TestFlexScheduler", (FlexStreamHooks, self.namespace["MultiTurnScheduler"]), {})

        def run_rank(rank):
            loop = asyncio.new_event_loop()
            try:
                scheduler = scheduler_type(tokenizer=Tokenizer(),
                                           template=SimpleNamespace(_get_response_prefix=lambda _: ""), max_turns=8)
                requests = [_request(rank, lengths[rank])] if lengths[rank] else []
                loop.run_until_complete(scheduler.on_trajectory_start(requests))
                first_outputs = [_output(rank in invalid_ranks, owner=requests[0].uuid)] if requests else []
                engine_group = engine_groups[rank]

                def all_gather_object(output, value, group):
                    # Real object collectives serialize payloads. Copies keep
                    # other simulated ranks from mutating this rank's history.
                    kind = "input-lengths" if isinstance(value, int) else "input-requests"
                    output[:] = copy.deepcopy(collective.gather(rank, group, kind, [copy.deepcopy(value)]))

                distributed = SimpleNamespace(get_world_size=lambda group: len(group),
                                              get_rank=lambda group: group.index(rank),
                                              all_gather_object=all_gather_object)

                def engine_infer(infer_requests, request_config, use_tqdm):
                    owners = [request.uuid for request in infer_requests]
                    collective.engine_requests[rank].append(owners)
                    return [_output(owner=request.uuid) for request in infer_requests]

                trainer = SimpleNamespace(vllm_mode="colocate", vllm_tensor_parallel_size=len(engine_group),
                                          vllm_tp_group=engine_group, samples2requests=lambda values: values,
                                          engine=SimpleNamespace(infer=engine_infer),
                                          args=SimpleNamespace(router_replay_mode=None))
                namespace = {"torch": SimpleNamespace(distributed=distributed)}
                exec(self.colocate_code, namespace)

                def infer(batch, config):
                    collective.inference_calls[rank].append([len(req.audios) for req in batch])
                    if actual_colocate:
                        return namespace["_colocate_rollout"](trainer, batch, config)
                    collective.gather(rank, engine_group, "vllm-inference", [len(batch)])
                    return [_output(owner=request.uuid) for request in batch]

                if stop_groups is None:
                    # Exercise production group selection, not an idealized
                    # replacement predicate. Data partition groups remain OLD_GROUPS.
                    gather = make_completion_gather(
                        trainer,
                        lambda values, group: collective.gather(rank, group, "turn-finished", values),
                        distributed)
                else:
                    gather = lambda values: collective.gather(rank, stop_groups[rank], "turn-finished", values)
                results[rank] = self.namespace["_run_multi_turn_impl"](
                    loop, requests, first_outputs, scheduler, infer, SimpleNamespace(), max_turns=8,
                    gather_fn=gather)
            except BaseException as exc:
                errors[rank] = exc
            finally:
                loop.close()
                collective.exit(rank)

        template_stub = SimpleNamespace(StdTemplateInputs=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict("sys.modules", {"swift.template": template_stub}), patch("flex_gspo.rollout_sync.progress"):
            threads = [Thread(target=run_rank, args=(rank,), daemon=True) for rank in ranks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads), "Simulator must never leave hung threads")
        return collective, results, errors

    def assert_preserved(self, results, expected_audio_turns, invalid_ranks=()):
        for rank, audio_turns in enumerate(expected_audio_turns):
            if not audio_turns:
                self.assertEqual(results[rank], [])
                continue
            turns = audio_turns if rank in invalid_ranks else audio_turns + 1
            result, = results[rank]
            self.assertEqual(result.response.owner, f"candidate-{rank}")
            self.assertEqual(result.rollout_infos["num_turns"], turns)
            self.assertEqual(len(result.rollout_infos["audios"]), audio_turns)
            self.assertEqual(result.response_loss_mask, [[1, 1]] * turns)
            self.assertEqual(result.rollout_logprobs, [[-.1, -.1]] * turns)
            self.assertEqual(result.response_token_ids,
                             ([[3, 151645]] if rank in invalid_ranks else [[151676, 151645]] * turns))
            self.assertEqual(result.rollout_infos["flex_trajectory"]["valid"], rank not in invalid_ranks)
            self.assertEqual(result.rollout_infos["flex_trajectory"]["episode_complete"], rank not in invalid_ranks)

    def test_old_two_rank_termination_breaks_four_rank_inference(self):
        collective, results, errors = self.run_case([1, 1, 3, 4], OLD_GROUPS)
        self.assertEqual(set(results), {0, 1})
        self.assertEqual(set(errors), {2, 3})
        self.assertTrue(all(isinstance(error, CollectiveParticipationError) for error in errors.values()))
        self.assertEqual(collective.inference_calls[0], [[1], []])
        self.assertEqual(collective.inference_calls[1], [[1], []])

    def test_four_rank_termination_keeps_completed_ranks_participating(self):
        collective, results, errors = self.run_case([1, 1, 3, 4])
        self.assertEqual(errors, {})
        self.assert_preserved(results, [1, 1, 3, 4])
        self.assertEqual(collective.inference_calls[0], [[1], [], [], [], []])
        self.assertEqual(collective.inference_calls[1], [[1], [], [], [], []])
        self.assertEqual(collective.inference_calls[2], [[2], [3], [3], [], []])
        self.assertEqual(collective.inference_calls[3], [[2], [3], [4], [4], []])

    def test_rank_with_no_initial_requests_still_participates(self):
        collective, results, errors = self.run_case([0, 1, 3, 4])
        self.assertEqual(errors, {})
        self.assert_preserved(results, [0, 1, 3, 4])
        self.assertEqual(collective.inference_calls[0], [[], [], [], [], []])

    def test_invalid_early_stop_is_not_regenerated_or_dropped(self):
        collective, results, errors = self.run_case([4, 1, 3, 4], invalid_ranks={0})
        self.assertEqual(errors, {})
        self.assert_preserved(results, [1, 1, 3, 4], invalid_ranks={0})
        self.assertFalse(results[0][0].rollout_infos["flex_trajectory"]["complete_input"])
        self.assertEqual(collective.inference_calls[0], [[], [], [], [], []])

    def test_all_empty_ranks_exit_without_inference(self):
        collective, results, errors = self.run_case([0, 0, 0, 0])
        self.assertEqual(errors, {})
        self.assert_preserved(results, [0, 0, 0, 0])
        self.assertTrue(all(not calls for calls in collective.inference_calls.values()))

    def test_eight_ranks_two_tp4_engines_finish_independently_and_preserve_ownership(self):
        groups = ((0, 1, 2, 3),) * 4 + ((4, 5, 6, 7),) * 4
        lengths = [0, 1, 2, 1, 1, 3, 0, 5]
        collective, results, errors = self.run_case(lengths, engine_groups=groups)
        self.assertEqual(errors, {})
        self.assert_preserved(results, lengths)
        # The first engine finishes three rounds sooner. Its empty rank must
        # participate locally, without waiting for the other engine to finish.
        self.assertEqual([len(collective.inference_calls[rank]) for rank in range(8)], [3] * 4 + [6] * 4)
        self.assertEqual(collective.inference_calls[0], [[], [], []])
        self.assertEqual(collective.inference_calls[6], [[], [], [], [], [], []])
        for group in (groups[0], groups[4]):
            expected_owners = {f"candidate-{rank}" for rank in group}
            for rank in group:
                self.assertEqual(collective.engine_requests[rank], collective.engine_requests[group[0]])
                self.assertTrue(all(set(batch) <= expected_owners for batch in collective.engine_requests[rank]))
        self.assertEqual({group for group, _ in collective.rounds}, {groups[0], groups[4]})

    def test_eight_ranks_empty_engine_exits_while_other_engine_completes(self):
        groups = ((0, 1, 2, 3),) * 4 + ((4, 5, 6, 7),) * 4
        lengths = [0, 0, 0, 0, 1, 3, 0, 2]
        collective, results, errors = self.run_case(lengths, engine_groups=groups)
        self.assertEqual(errors, {})
        self.assert_preserved(results, lengths)
        for rank in range(4):
            self.assertEqual(collective.inference_calls[rank], [])
        self.assertEqual([len(collective.inference_calls[rank]) for rank in range(4, 8)], [4] * 4)

    def test_eight_ranks_invalid_candidate_does_not_drop_engine_collectives(self):
        groups = ((0, 1, 2, 3),) * 4 + ((4, 5, 6, 7),) * 4
        collective, results, errors = self.run_case([1, 2, 0, 1, 5, 1, 0, 4],
                                                   invalid_ranks={4}, engine_groups=groups)
        self.assertEqual(errors, {})
        self.assert_preserved(results, [1, 2, 0, 1, 1, 1, 0, 4], invalid_ranks={4})
        self.assertFalse(results[4][0].rollout_infos["flex_trajectory"]["complete_input"])
        self.assertEqual(collective.inference_calls[4], [[], [], [], [], []])


if __name__ == "__main__":
    unittest.main()
