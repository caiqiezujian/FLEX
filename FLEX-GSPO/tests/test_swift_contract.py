"""Run the real audited upstream multi-turn driver with tiny CPU test doubles.

No copied driver implementation: functions/classes are compiled from the local
upstream source, excluding heavy import statements. This checks scheduler and
sample contracts, NOT a CUDA model forward or Megatron optimizer update.
Set FLEX_SWIFT_SOURCE to run these checks on your installed source checkout.
"""
import ast
import asyncio
from abc import ABC
import copy
from dataclasses import dataclass, field, fields
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo.audio import encode_chunk_ref
from flex_gspo.online import FlexStreamHooks
from flex_gspo.prompts import initial_messages


ROOT = Path(__file__).resolve().parents[1]
SWIFT = Path(os.environ.get("FLEX_SWIFT_SOURCE", ROOT.parent / "tmp" / "omni-rl-audit-swift-20261015"))


def compile_definitions(relative, names, namespace):
    path = SWIFT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Upstream contract changed: {relative}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


class Tokenizer:
    def decode(self, ids, **kwargs):
        pieces = {151676: "<|WAIT|>", 151677: "<|DIRECT|>", 151678: "<|FLEX|>",
                  151645: "<|im_end|>", 1: "你好", 2: "世界。", 3: "BAD"}
        return "".join(pieces[token] for token in ids)

    def encode(self, value, **kwargs):
        if value:
            raise AssertionError("No deterministic response prefix expected")
        return []


class Output(SimpleNamespace):
    pass


def task():
    return {"schema_version": 1, "id": "test", "audio_path": "/no/file/read.wav", "sample_rate": 16000,
            "source_lang": "en", "target_lang": "zh", "source_text": "Hello world.",
            "reference_text": "你好世界。", "quality_ready": True, "duration_seconds": 3.,
            "chunks": [{"start_seconds": float(i), "end_seconds": float(i + 1), "input_seconds": 1.}
                       for i in range(3)]}


def request(identifier):
    data = task()
    return SimpleNamespace(uuid=identifier, data_dict={"flex_task": json.dumps(data)},
                           messages=initial_messages(), audios=[encode_chunk_ref(data, 0)], images=[], videos=[],
                           chat_template_kwargs={})


def output(ids, text):
    choice = SimpleNamespace(message=SimpleNamespace(content=text), token_ids=ids, finish_reason="stop",
                             logprobs={"content": [{"logprob": -.1} for _ in ids]})
    return Output(response=SimpleNamespace(choices=[choice]))


@unittest.skipUnless((SWIFT / "swift/rollout/agent_loop.py").exists(), "Set FLEX_SWIFT_SOURCE to the audited checkout")
class ActualUpstreamContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def remove_response(messages):
            if messages[-1]["role"] == "assistant":
                messages.pop()

        ns = {"ABC": ABC, "deepcopy": copy.deepcopy, "copy": copy, "asyncio": asyncio,
              "RolloutOutput": Output, "remove_response": remove_response,
              "get_token_backed_response_ids": lambda content: content if isinstance(content, list) else None}
        compile_definitions("swift/rollout/multi_turn.py", ["RolloutScheduler", "MultiTurnScheduler"], ns)
        compile_definitions("swift/rollout/agent_loop.py", ["extract_logprobs_from_choice", "_identity_gather",
                            "run_multi_turn", "_run_multi_turn_impl"], ns)
        cls.ns = ns

    def test_real_driver_preserves_all_turn_ids_masks_logprobs_and_audio(self):
        scheduler_type = type("TestScheduler", (FlexStreamHooks, self.ns["MultiTurnScheduler"]), {})
        scheduler = scheduler_type(tokenizer=Tokenizer(), template=SimpleNamespace(_get_response_prefix=lambda _: ""),
                                   max_turns=3)
        requests = [request("normal"), request("bad")]
        asyncio.run(scheduler.on_trajectory_start(requests))
        first = [output([151676, 151645], "<|WAIT|>"), output([3, 151645], "BAD")]
        seen = []

        def generate(batch, config):
            seen.append([len(r.audios) for r in batch])
            results = []
            for req in batch:
                self.assertEqual(req.uuid, "normal")
                self.assertNotIn("Hello world", str(req.messages))
                if len(req.audios) == 2:
                    self.assertEqual(req.messages[2]["content"], [151676, 151645])
                    results.append(output([151678, 1, 151645], "<|FLEX|>你好"))
                else:
                    self.assertEqual(len(req.audios), 3)
                    self.assertEqual(req.messages[4]["content"], [151678, 1, 151645])
                    results.append(output([151677, 2, 151645], "<|DIRECT|>世界。"))
            return results

        with patch.dict("sys.modules", {"swift.template": SimpleNamespace(StdTemplateInputs=lambda **kw: SimpleNamespace(**kw))}):
            outputs = self.ns["run_multi_turn"](requests, first, scheduler, generate, SimpleNamespace(), max_turns=3)
        normal, bad = outputs
        self.assertEqual(normal.response_token_ids, [[151676, 151645], [151678, 1, 151645], [151677, 2, 151645]])
        self.assertEqual(normal.response_loss_mask, [[1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertEqual([len(lp) for lp in normal.rollout_logprobs], [2, 3, 3])
        self.assertEqual(len(normal.rollout_infos["audios"]), 3)
        self.assertTrue(normal.rollout_infos["flex_trajectory"]["complete_input"])
        self.assertFalse(bad.rollout_infos["flex_trajectory"]["valid"])
        self.assertEqual(bad.response_loss_mask, [[1, 1]])  # Invalid policy output is NOT hidden.
        self.assertEqual(seen, [[2], [3], []])

        # Run actual upstream sample merge: it MUST transfer all observed audio.
        ns = {"dataclass": dataclass, "field": field, "fields": fields, "copy": copy,
              "StandardKeys": ["messages", "images", "audios", "videos", "tools", "objects"],
              "_MULTIMODAL_KEYS": ("images", "videos", "audios")}
        compile_definitions("swift/rl_core/data.py", ["OnPolicySample"], ns)
        sample = ns["OnPolicySample"].from_row({"messages": initial_messages(), "audios": [encode_chunk_ref(task(), 0)],
                                               "flex_task": json.dumps(task())})
        sample.apply_rollout_output(rollout_output=normal)
        self.assertEqual(len(sample.audios), 3)
        self.assertEqual(sample.response_loss_mask[0], [1, 1])
        self.assertNotIn("flex_task", sample.to_template_dict())
        self.assertIn("flex_task", sample.to_reward_row())


if __name__ == "__main__":
    unittest.main()
