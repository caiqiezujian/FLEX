"""Causal rollout invariants; no model weights or external services required."""

from __future__ import annotations

import ast
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from flex_gspo.audio import encode_chunk_ref
from flex_gspo.online import FlexStreamHooks, STATE_KEY, unpack_task
from flex_gspo.prompts import EOF_PROMPT, SYSTEM_PROMPT, initial_messages


def task_fixture():
    return {
        "schema_version": 1, "id": "test-audio", "audio_path": "/audio/test.wav",
        "source_lang": "en", "target_lang": "zh", "duration_seconds": 2.25,
        "sample_rate": 16000, "quality_ready": True,
        "source_text": "Confidential source transcript must not enter actor messages.",
        "reference_text": "评分参考不能进入模型输入。",
        "chunks": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
            {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
            {"start_seconds": 2.0, "end_seconds": 2.25, "input_seconds": 1.0},
        ],
    }


def request_fixture(task=None, uuid="candidate-a"):
    task = task or task_fixture()
    return SimpleNamespace(messages=initial_messages(), audios=[encode_chunk_ref(task, 0)],
                           data_dict={"flex_task": json.dumps(task, ensure_ascii=False)},
                           images=[], videos=[], uuid=uuid, chat_template_kwargs={})


def choice_fixture(text="<|WAIT|>", ids=None, finish_reason="stop"):
    ids = [151676, 151645] if ids is None else ids
    return SimpleNamespace(message=SimpleNamespace(content=text), token_ids=ids,
                           logprobs={"content": [{"logprob": -0.2} for _ in ids]},
                           finish_reason=finish_reason)


class Hooks(FlexStreamHooks):
    max_turns = 40


class OnlineHooksTests(unittest.TestCase):
    def setUp(self):
        self.hooks = Hooks()
        self.request = request_fixture()
        asyncio.run(self.hooks.on_trajectory_start([self.request]))

    def observe(self, text="<|WAIT|>", turn=1, finish_reason="stop"):
        choice = choice_fixture(text, finish_reason=finish_reason)
        self.request.messages.append({"role": "assistant", "content": text})
        result = asyncio.run(self.hooks.on_turn_end(self.request, choice, turn))
        return choice, result

    def test_source_reference_stay_out_of_actor_messages(self):
        messages = json.dumps(self.request.messages, ensure_ascii=False)
        task = task_fixture()
        self.assertNotIn(task["source_text"], messages)
        self.assertNotIn(task["reference_text"], messages)
        self.assertEqual(len(self.request.audios), 1)
        self.assertEqual(self.request.audios, [encode_chunk_ref(task, 0)])

    def test_wait_retains_history_and_advances_exactly_one_new_audio_chunk(self):
        choice, result = self.observe()
        self.assertFalse(result["done"])
        self.assertEqual(result["rollout_infos"]["audios"], [encode_chunk_ref(task_fixture(), 0)])
        self.hooks.step(self.request, choice, 1)
        self.assertEqual(self.request.messages[-2:], [
            {"role": "assistant", "content": "<|WAIT|>"}, {"role": "user", "content": "<audio>"},
        ])
        self.assertEqual(self.request.audios, [encode_chunk_ref(task_fixture(), i) for i in range(2)])
        # Previous turn metadata must remain a snapshot, not a shared list.
        self.assertEqual(len(result["rollout_infos"]["audios"]), 1)

    def test_final_wait_is_valid_and_complete_not_a_forced_translation(self):
        for turn in range(1, 5):
            choice, result = self.observe(turn=turn)
            if turn < 4:
                self.hooks.step(self.request, choice, turn)
        trajectory = result["rollout_infos"]["flex_trajectory"]
        self.assertTrue(result["done"])
        self.assertTrue(trajectory["valid"])
        self.assertTrue(trajectory["complete_input"])
        self.assertTrue(trajectory["episode_complete"])
        self.assertTrue(trajectory["eof_completed"])
        self.assertEqual(trajectory["turns"][-1]["kind"], "eof")
        self.assertGreaterEqual(trajectory["eof_elapsed_seconds"], 0)
        self.assertEqual(trajectory["turns"][-1]["response"], "<|WAIT|>")
        self.assertEqual(trajectory["turns"][-1]["audio_end_seconds"], 2.25)
        self.assertEqual(len(result["rollout_infos"]["audios"]), 3)
        with self.assertRaises(RuntimeError):
            self.hooks.step(self.request, choice, 4)

    def test_flex_then_direct_are_preserved_without_sentence_repair(self):
        outputs = ("<|FLEX|>这场会议", "<|DIRECT|>将在周末", "<|DIRECT|>举行。")
        for turn, text in enumerate(outputs, 1):
            choice, result = self.observe(text, turn)
            if turn < 3:
                self.hooks.step(self.request, choice, turn)
        self.assertEqual([t["response"] for t in result["rollout_infos"]["flex_trajectory"]["turns"]], list(outputs))

    def test_invalid_protocol_stops_with_original_text_and_no_repair(self):
        for text in ("translation without a tag", "<|WAIT|>不能带译文", "<|FLEX|>", "<|DIRECT|>甲<|FLEX|>乙"):
            with self.subTest(text=text):
                request = request_fixture()
                asyncio.run(self.hooks.on_trajectory_start([request]))
                result = asyncio.run(self.hooks.on_turn_end(request, choice_fixture(text), 1))
                self.assertTrue(result["done"])
                trajectory = result["rollout_infos"]["flex_trajectory"]
                self.assertFalse(trajectory["valid"])
                self.assertFalse(trajectory["complete_input"])
                self.assertEqual(trajectory["turns"][0]["response"], text)

    def test_length_truncation_is_invalid_even_with_a_valid_tag(self):
        _, result = self.observe("<|DIRECT|>看似正常。", finish_reason="length")
        self.assertTrue(result["done"])
        self.assertFalse(result["rollout_infos"]["flex_trajectory"]["valid"])
        self.assertIn("limit", result["rollout_infos"]["flex_trajectory"]["invalid_reason"])

    def test_exact_token_ids_and_complete_finite_logprobs_are_required(self):
        variants = []
        for ids in ([], None, [True], ["151676"], (151676,)):
            candidate = choice_fixture()
            candidate.token_ids = ids
            variants.append(candidate)
        for values in (None, {}, {"content": []}, {"content": [{"logprob": -0.2}]},
                       {"content": [{"logprob": float("nan")}, {"logprob": 0.0}]},
                       {"content": [{"logprob": True}, {"logprob": 0.0}]}):
            candidate = choice_fixture()
            candidate.logprobs = values
            variants.append(candidate)
        for candidate in variants:
            with self.subTest(candidate=candidate):
                with self.assertRaises(RuntimeError):
                    asyncio.run(self.hooks.on_turn_end(self.request, candidate, 1))
                self.assertEqual(self.request.data_dict[STATE_KEY]["turns"], [])

    def test_candidates_with_shared_initial_objects_get_independent_histories(self):
        first, second = request_fixture(), request_fixture(uuid="candidate-b")
        second.messages, second.audios, second.data_dict = first.messages, first.audios, first.data_dict
        asyncio.run(self.hooks.on_trajectory_start([first, second]))
        first.messages.append({"role": "assistant", "content": "<|WAIT|>"})
        result = asyncio.run(self.hooks.on_turn_end(first, choice_fixture(), 1))
        self.hooks.step(first, choice_fixture(), 1)
        self.assertFalse(result["done"])
        self.assertEqual(second.messages, initial_messages())
        self.assertEqual(len(second.audios), 1)
        self.assertEqual(second.data_dict[STATE_KEY]["turns"], [])

    def test_initial_full_audio_or_future_chunks_are_rejected(self):
        for audios in ([task_fixture()["audio_path"]], [encode_chunk_ref(task_fixture(), 1)],
                       [encode_chunk_ref(task_fixture(), 0), encode_chunk_ref(task_fixture(), 1)]):
            request = request_fixture()
            request.audios = audios
            with self.assertRaisesRegex(ValueError, "first virtual chunk"):
                asyncio.run(self.hooks.on_trajectory_start([request]))

    def test_context_and_task_contract_fail_before_first_inference(self):
        self.hooks.max_turns = 2
        with self.assertRaisesRegex(ValueError, "needs 4 turns"):
            asyncio.run(self.hooks.on_trajectory_start([request_fixture()]))
        self.hooks.max_turns = 40
        request = request_fixture()
        request.messages[0]["content"] = "another instruction"
        with self.assertRaisesRegex(ValueError, "SFT prompt"):
            asyncio.run(self.hooks.on_trajectory_start([request]))
        request = request_fixture()
        request.data_dict = {}
        with self.assertRaisesRegex(ValueError, "flex_task"):
            asyncio.run(self.hooks.on_trajectory_start([request]))

    def test_turn_order_and_audio_order_are_enforced(self):
        with self.assertRaises(RuntimeError):
            asyncio.run(self.hooks.on_turn_end(self.request, choice_fixture(), 2))
        self.request.audios[0] = encode_chunk_ref(task_fixture(), 1)
        with self.assertRaisesRegex(RuntimeError, "causal chunk plan"):
            asyncio.run(self.hooks.on_turn_end(self.request, choice_fixture(), 1))

    def test_unpack_copies_input_and_rejects_missing_reference(self):
        original = task_fixture()
        unpacked = unpack_task(original)
        unpacked["chunks"][0]["end_seconds"] = 0.5
        self.assertEqual(original["chunks"][0]["end_seconds"], 1.0)
        original["reference_text"] = None
        with self.assertRaisesRegex(ValueError, "reference"):
            unpack_task(original)

    def test_prompt_equals_original_sft_literal_when_source_is_available(self):
        source = Path(__file__).resolve().parents[2] / "Scripts" / "prepare_duplex_data.py"
        if not source.is_file():
            self.skipTest("Original SFT script not included in this standalone checkout")
        module = ast.parse(source.read_text(encoding="utf-8"))
        expression = next(node.value for node in module.body if isinstance(node, ast.Assign)
                          and any(isinstance(name, ast.Name) and name.id == "PROTOCOL_SYSTEM_PROMPT_TEMPLATE"
                                  for name in node.targets))
        self.assertEqual(SYSTEM_PROMPT, ast.literal_eval(expression).format(target_lang="中文"))


def _find_swift_source():
    explicit = os.environ.get("FLEX_SWIFT_SOURCE")
    if explicit:
        return Path(explicit)
    local_audit = Path(__file__).resolve().parents[2] / "tmp" / "omni-rl-audit-swift-20261015"
    if local_audit.is_dir():
        return local_audit
    spec = importlib.util.find_spec("swift")
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parents[1]
    return None


SWIFT_SOURCE = _find_swift_source()


@unittest.skipUnless(SWIFT_SOURCE, "Set FLEX_SWIFT_SOURCE to test the audited upstream rollout driver")
class UpstreamRolloutDriverContractTests(unittest.TestCase):
    """Execute real upstream function bodies, not a rewritten lookalike loop.

    Imports needing CUDA are omitted via AST extraction. Protocol containers and
    tokenizer are tiny test doubles; therefore this is a CPU contract test, not
    a claim that vLLM/Megatron model training has run.
    """

    def test_upstream_driver_preserves_all_turn_tokens_masks_logprobs_and_audio(self):
        import abc

        class Output:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        def ids_or_none(value):
            return value if isinstance(value, list) else None

        def remove_response(messages):
            if messages and messages[-1]["role"] == "assistant":
                messages.pop()

        namespace = {"asyncio": asyncio, "ABC": abc.ABC, "deepcopy": deepcopy,
                     "get_token_backed_response_ids": ids_or_none, "remove_response": remove_response,
                     "RolloutOutput": Output}

        def load_nodes(relative_path, names):
            source_path = SWIFT_SOURCE / relative_path
            if not source_path.is_file():
                self.skipTest(f"Upstream source file not available: {source_path}")
            source = ast.parse(source_path.read_text(encoding="utf-8"))
            nodes = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name in names]
            if {node.name for node in nodes} != set(names):
                self.fail(f"Upstream contract changed in {source_path}")
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            program = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
            exec(compile(program, str(source_path), "exec"), namespace)

        load_nodes("swift/rollout/multi_turn.py", ["RolloutScheduler"])
        load_nodes("swift/rollout/agent_loop.py", ["_identity_gather", "extract_logprobs_from_choice",
                                                  "run_multi_turn", "_run_multi_turn_impl"])
        plans = {
            "candidate-a": [("<|WAIT|>", [151676, 151645]),
                            ("<|FLEX|>会议", [151678, 201, 151645]),
                            ("<|DIRECT|>开始。", [151677, 202, 151645]),
                            ("<|WAIT|>", [151676, 151645])],
            "candidate-b": [("bad-tag-output", [203, 151645])],
        }
        token_to_text = {tuple(ids): text for plan in plans.values() for text, ids in plan}

        class Tokenizer:
            def decode(self, ids, **kwargs):
                return token_to_text[tuple(ids)]

        class Scheduler(FlexStreamHooks, namespace["RolloutScheduler"]):
            pass

        scheduler = Scheduler(max_turns=40, tokenizer=Tokenizer())
        requests = [request_fixture(uuid=key) for key in plans]
        asyncio.run(scheduler.on_trajectory_start(requests))
        calls = []

        def rollout_fn(active_requests, config):
            calls.append([request.uuid for request in active_requests])
            outputs = []
            for request in active_requests:
                turn = len(request.data_dict[STATE_KEY]["turns"])
                if turn:
                    assistants = [message for message in request.messages if message["role"] == "assistant"]
                    self.assertTrue(all(isinstance(message["content"], list) for message in assistants))
                if turn == 3:
                    self.assertEqual(request.messages[-1], {"role": "user", "content": EOF_PROMPT})
                    self.assertEqual(len(request.audios), 3)
                text, ids = plans[request.uuid][turn]
                choice = choice_fixture(text, ids)
                outputs.append(Output(response=SimpleNamespace(choices=[choice])))
            return outputs

        first = rollout_fn(requests, None)
        outputs = namespace["run_multi_turn"](requests, first, scheduler, rollout_fn, None, max_turns=40)
        first_output, invalid_output = outputs
        self.assertEqual(first_output.response_token_ids, [ids for _, ids in plans["candidate-a"]])
        self.assertEqual(first_output.response_loss_mask, [[1, 1], [1, 1, 1], [1, 1, 1], [1, 1]])
        self.assertEqual(first_output.rollout_logprobs, [[-0.2, -0.2], [-0.2] * 3, [-0.2] * 3, [-0.2] * 2])
        self.assertEqual(first_output.rollout_infos["audios"], [encode_chunk_ref(task_fixture(), i) for i in range(3)])
        self.assertEqual(first_output.rollout_infos["num_turns"], 4)
        self.assertTrue(first_output.rollout_infos["flex_trajectory"]["complete_input"])
        self.assertTrue(first_output.rollout_infos["flex_trajectory"]["episode_complete"])
        self.assertFalse(invalid_output.rollout_infos["flex_trajectory"]["valid"])
        self.assertEqual(invalid_output.rollout_infos["num_turns"], 1)
        self.assertEqual(calls[:3], [["candidate-a", "candidate-b"], ["candidate-a"], ["candidate-a"]])


if __name__ == "__main__":
    unittest.main()
