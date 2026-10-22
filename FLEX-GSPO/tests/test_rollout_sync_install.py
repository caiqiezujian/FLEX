"""Exercise the installed wrappers through the real upstream generation method.

Only CUDA/vLLM operations and the multi-turn driver's internals are stubbed.
The generation method is compiled from the audited SWIFT checkout, so its
module-level run_multi_turn lookup must actually hit the installed wrapper.
"""
from __future__ import annotations

import ast
import asyncio
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

from flex_gspo.online import FlexStreamHooks
from flex_gspo import rollout_sync


ROOT = Path(__file__).resolve().parents[1]
SWIFT = Path(os.environ.get("FLEX_SWIFT_SOURCE", ROOT.parent / "tmp" / "omni-rl-audit-swift-20261015"))
SOURCE = SWIFT / "swift/megatron/trainers/rollout_mixin.py"


class PlainScheduler:
    async def on_trajectory_start(self, requests):
        self.started_requests = list(requests)


class FlexScheduler(FlexStreamHooks, PlainScheduler):
    async def on_trajectory_start(self, requests):
        # Audio/task validation is tested separately. Here we isolate wiring.
        await PlainScheduler.on_trajectory_start(self, requests)


@unittest.skipUnless(SOURCE.exists(), "Set FLEX_SWIFT_SOURCE to the audited checkout")
class RolloutSyncInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")

    def make_module(self):
        module = ModuleType("test_swift_rollout_mixin")
        module.state = SimpleNamespace(gathers=[], continuation_requests=[], driver_error=None,
                                       driver_gather=None, predicate_result=None, driver_calls=0)
        state = module.state
        training_group = object()
        engine_group = object()
        sizes = {training_group: 2, engine_group: 4}

        def gather_object(values, group):
            state.gathers.append((list(values), group))
            return list(values) * sizes[group]

        def driver(requests, first_turn_outputs, scheduler, rollout_fn, request_config,
                   max_turns=None, gather_fn=None):
            state.driver_calls += 1
            state.driver_gather = gather_fn
            state.predicate_result = gather_fn([True])
            if state.driver_error is not None:
                raise state.driver_error
            # A completed local rank must still pass the empty batch upstream.
            self.assertEqual(rollout_fn([], request_config), [])
            return first_turn_outputs

        module.__dict__.update({
            "gather_object": gather_object,
            "run_multi_turn": driver,
            "invoke_async_hook": asyncio.run,
            "nullcontext": nullcontext,
            "inspect": inspect,
            "torch": SimpleNamespace(distributed=SimpleNamespace(
                get_world_size=lambda group: sizes[group])),
        })
        tree = ast.parse(self.source)
        upstream_class = next(node for node in tree.body
                              if isinstance(node, ast.ClassDef) and node.name == "MegatronRolloutMixin")
        generate = next(node for node in upstream_class.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_generate_completions")
        generate.decorator_list = []
        compiled_class = ast.ClassDef(name="MegatronRolloutMixin", bases=[], keywords=[],
                                      body=[generate], decorator_list=[])
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        compiled = ast.fix_missing_locations(ast.Module(body=[future, compiled_class], type_ignores=[]))
        exec(compile(compiled, str(SOURCE), "exec"), module.__dict__)
        module.training_group = training_group
        module.engine_group = engine_group
        return module

    def trainer(self, module, *, size=4, mode="colocate", flex=True):
        trainer = module.MegatronRolloutMixin()
        trainer.vllm_mode = mode
        trainer.vllm_tensor_parallel_size = size
        if size > 1:
            trainer.vllm_tp_group = module.engine_group
        trainer.multi_turn_scheduler = FlexScheduler() if flex else PlainScheduler()
        trainer.enable_server_multi_turn = False
        trainer.enable_offload = False
        trainer.args = SimpleNamespace(sleep_level=0, max_turns=64)
        trainer._step = trainer._last_loaded_step = 0
        trainer._preprocess_inputs = lambda samples: samples
        trainer.samples2requests = lambda samples: samples
        trainer._get_request_config = lambda: "test-config"
        trainer._get_rollout_group = lambda: module.training_group

        def rollout(requests, config):
            self.assertEqual(config, "test-config")
            module.state.continuation_requests.append(list(requests))
            return list(requests)

        trainer._rollout_requests = rollout
        trainer._postprocess_rollout_outputs = lambda samples, outputs: outputs
        return trainer

    def install(self, module, source=None):
        with patch.object(rollout_sync.inspect, "getsource", return_value=source or self.source), \
                patch.object(rollout_sync, "progress"):
            rollout_sync.install_rollout_sync_patch(module)

    def run_generate(self, trainer):
        with patch.object(rollout_sync, "progress"):
            return trainer._generate_completions(["first-output"])

    def test_actual_method_uses_installed_driver_and_engine_group(self):
        module = self.make_module()
        original_driver = module.run_multi_turn
        original_generate = module.MegatronRolloutMixin._generate_completions
        trainer = self.trainer(module)
        original_partition_method = trainer._get_rollout_group
        self.install(module)
        self.assertIsNot(module.run_multi_turn, original_driver)
        self.assertIsNot(module.MegatronRolloutMixin._generate_completions, original_generate)
        self.assertEqual(self.run_generate(trainer), ["first-output"])
        self.assertEqual(module.state.driver_calls, 1)
        self.assertEqual(module.state.gathers, [([True], module.engine_group)])
        self.assertEqual(module.state.predicate_result, [True] * 4)
        self.assertEqual(module.state.continuation_requests, [["first-output"], []])
        self.assertIs(trainer._get_rollout_group, original_partition_method)
        self.assertIsNone(rollout_sync._ACTIVE_TRAINER.get())

    def test_tp1_needs_no_engine_group_and_no_distributed_completion_gather(self):
        module = self.make_module()
        trainer = self.trainer(module, size=1)
        self.assertFalse(hasattr(trainer, "vllm_tp_group"))
        self.install(module)
        self.assertEqual(self.run_generate(trainer), ["first-output"])
        self.assertEqual(module.state.gathers, [])
        self.assertEqual(module.state.predicate_result, [True])
        self.assertEqual(module.state.continuation_requests[-1], [])

    def test_installer_is_idempotent(self):
        module = self.make_module()
        self.install(module)
        installed_driver = module.run_multi_turn
        installed_generate = module.MegatronRolloutMixin._generate_completions
        # A repeated installation must not even inspect source or double-wrap.
        with patch.object(rollout_sync.inspect, "getsource", side_effect=AssertionError("read twice")):
            rollout_sync.install_rollout_sync_patch(module)
        self.assertIs(module.run_multi_turn, installed_driver)
        self.assertIs(module.MegatronRolloutMixin._generate_completions, installed_generate)
        self.assertEqual(self.run_generate(self.trainer(module)), ["first-output"])
        self.assertEqual(module.state.driver_calls, 1)

    def test_optional_telemetry_preserves_real_upstream_generation_collectives(self):
        module = self.make_module()
        trainer = self.trainer(module)
        # A class-level method models upstream, rather than the other tests'
        # deliberately instance-level stub, so the telemetry wrapper can bind.
        module.MegatronRolloutMixin._rollout_requests = lambda self, requests, config: list(requests)
        del trainer._rollout_requests
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"FLEX_PERF_LOG_DIR": directory}):
            self.install(module)
            self.assertEqual(self.run_generate(trainer), ["first-output"])
            self.assertEqual(module.state.gathers, [([True], module.engine_group)])
            paths = list(Path(directory).glob("performance-*.jsonl"))
            self.assertEqual(len(paths), 1)
            row = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(row["rollout_requests_count"], 2)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["local_trajectories"], 1)

    def test_original_exception_is_preserved_and_context_is_reset(self):
        module = self.make_module()
        trainer = self.trainer(module)
        self.install(module)
        error = RuntimeError("original rollout failure")
        module.state.driver_error = error
        sentinel = object()
        token = rollout_sync._ACTIVE_TRAINER.set(sentinel)
        try:
            with self.assertRaises(RuntimeError) as caught:
                self.run_generate(trainer)
            self.assertIs(caught.exception, error)
            self.assertIs(rollout_sync._ACTIVE_TRAINER.get(), sentinel)
        finally:
            rollout_sync._ACTIVE_TRAINER.reset(token)
        self.assertIsNone(rollout_sync._ACTIVE_TRAINER.get())

    def test_server_and_non_flex_preserve_upstream_completion_scope(self):
        for mode, flex in (("server", True), ("colocate", False)):
            with self.subTest(mode=mode, flex=flex):
                module = self.make_module()
                trainer = self.trainer(module, mode=mode, flex=flex)
                self.install(module)
                self.assertEqual(self.run_generate(trainer), ["first-output"])
                self.assertEqual(module.state.gathers, [([True], module.training_group)])
                self.assertEqual(module.state.predicate_result, [True, True])
                self.assertIsNone(rollout_sync._ACTIVE_TRAINER.get())

    def test_direct_driver_call_without_active_trainer_preserves_gather_object(self):
        module = self.make_module()
        self.install(module)

        def original_gather(values):
            return values

        self.assertEqual(module.run_multi_turn(
            requests=[], first_turn_outputs=[], scheduler=PlainScheduler(),
            rollout_fn=lambda requests, config: [], request_config=None,
            gather_fn=original_gather), [])
        self.assertIs(module.state.driver_gather, original_gather)

    def test_changed_source_contract_fails_without_installing_partial_patch(self):
        module = self.make_module()
        original_driver = module.run_multi_turn
        original_generate = module.MegatronRolloutMixin._generate_completions
        changed = self.source.replace("outputs = run_multi_turn(", "outputs = other_driver(")
        self.assertNotEqual(changed, self.source)
        with self.assertRaisesRegex(RuntimeError, "call contract changed"):
            self.install(module, changed)
        self.assertIs(module.run_multi_turn, original_driver)
        self.assertIs(module.MegatronRolloutMixin._generate_completions, original_generate)
        self.assertFalse(hasattr(module, "_flex_rollout_sync_patch"))


if __name__ == "__main__":
    unittest.main()
