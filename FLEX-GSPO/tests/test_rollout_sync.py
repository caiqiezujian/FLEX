"""Run the actual upstream turn loop on four simulated distributed ranks.

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

    def __init__(self):
        self.condition = Condition()
        self.exited = set()
        self.counters = {}
        self.rounds = {}
        self.inference_calls = {rank: [] for rank in WORLD}

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


def _output(invalid=False):
    ids = [3, 151645] if invalid else [151676, 151645]
    choice = SimpleNamespace(message=SimpleNamespace(content="BAD" if invalid else "<|WAIT|>"),
                             token_ids=ids, finish_reason="stop",
                             logprobs={"content": [{"logprob": -.1} for _ in ids]})
    return SimpleNamespace(response=SimpleNamespace(choices=[choice]))


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

    def run_case(self, lengths, stop_groups=None, invalid_ranks=()):
        collective = Collectives()
        results, errors = {}, {}
        scheduler_type = type("TestFlexScheduler", (FlexStreamHooks, self.namespace["MultiTurnScheduler"]), {})

        def run_rank(rank):
            loop = asyncio.new_event_loop()
            try:
                scheduler = scheduler_type(tokenizer=Tokenizer(),
                                           template=SimpleNamespace(_get_response_prefix=lambda _: ""), max_turns=8)
                requests = [_request(rank, lengths[rank])] if lengths[rank] else []
                loop.run_until_complete(scheduler.on_trajectory_start(requests))
                first_outputs = [_output(rank in invalid_ranks)] if requests else []

                def infer(batch, config):
                    collective.inference_calls[rank].append([len(req.audios) for req in batch])
                    collective.gather(rank, WORLD, "vllm-inference", [len(batch)])
                    return [_output() for _ in batch]

                if stop_groups is None:
                    # Exercise production group selection, not an idealized
                    # replacement predicate. Data partition groups remain OLD_GROUPS.
                    trainer = SimpleNamespace(vllm_mode="colocate", vllm_tensor_parallel_size=4,
                                              vllm_tp_group=WORLD)
                    distributed = SimpleNamespace(get_world_size=lambda group: len(group))
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
            threads = [Thread(target=run_rank, args=(rank,), daemon=True) for rank in WORLD]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads), "Simulator must never leave hung threads")
        return collective, results, errors

    def assert_preserved(self, results, expected_turns, invalid_ranks=()):
        for rank, turns in enumerate(expected_turns):
            if not turns:
                self.assertEqual(results[rank], [])
                continue
            result, = results[rank]
            self.assertEqual(result.rollout_infos["num_turns"], turns)
            self.assertEqual(len(result.rollout_infos["audios"]), turns)
            self.assertEqual(result.response_loss_mask, [[1, 1]] * turns)
            self.assertEqual(result.rollout_logprobs, [[-.1, -.1]] * turns)
            self.assertEqual(result.response_token_ids,
                             ([[3, 151645]] if rank in invalid_ranks else [[151676, 151645]] * turns))
            self.assertEqual(result.rollout_infos["flex_trajectory"]["valid"], rank not in invalid_ranks)

    def test_old_two_rank_termination_breaks_four_rank_inference(self):
        collective, results, errors = self.run_case([1, 1, 3, 4], OLD_GROUPS)
        self.assertEqual(set(results), {0, 1})
        self.assertEqual(set(errors), {2, 3})
        self.assertTrue(all(isinstance(error, CollectiveParticipationError) for error in errors.values()))
        self.assertEqual(collective.inference_calls[0], [[]])
        self.assertEqual(collective.inference_calls[1], [[]])

    def test_four_rank_termination_keeps_completed_ranks_participating(self):
        collective, results, errors = self.run_case([1, 1, 3, 4])
        self.assertEqual(errors, {})
        self.assert_preserved(results, [1, 1, 3, 4])
        self.assertEqual(collective.inference_calls[0], [[], [], [], []])
        self.assertEqual(collective.inference_calls[1], [[], [], [], []])
        self.assertEqual(collective.inference_calls[2], [[2], [3], [], []])
        self.assertEqual(collective.inference_calls[3], [[2], [3], [4], []])

    def test_rank_with_no_initial_requests_still_participates(self):
        collective, results, errors = self.run_case([0, 1, 3, 4])
        self.assertEqual(errors, {})
        self.assert_preserved(results, [0, 1, 3, 4])
        self.assertEqual(collective.inference_calls[0], [[], [], [], []])

    def test_invalid_early_stop_is_not_regenerated_or_dropped(self):
        collective, results, errors = self.run_case([4, 1, 3, 4], invalid_ranks={0})
        self.assertEqual(errors, {})
        self.assert_preserved(results, [1, 1, 3, 4], invalid_ranks={0})
        self.assertFalse(results[0][0].rollout_infos["flex_trajectory"]["complete_input"])
        self.assertEqual(collective.inference_calls[0], [[], [], [], []])

    def test_all_empty_ranks_exit_without_inference(self):
        collective, results, errors = self.run_case([0, 0, 0, 0])
        self.assertEqual(errors, {})
        self.assert_preserved(results, [0, 0, 0, 0])
        self.assertTrue(all(not calls for calls in collective.inference_calls.values()))


if __name__ == "__main__":
    unittest.main()
