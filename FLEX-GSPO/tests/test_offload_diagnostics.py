"""Optional offload diagnostics preserve the upstream call and its failures."""
from __future__ import annotations

import ast
from contextlib import redirect_stderr
from functools import wraps
from io import StringIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo import offload_diagnostics as diagnostics


ROOT = Path(__file__).resolve().parents[1]
SWIFT = Path(os.environ.get("FLEX_SWIFT_SOURCE", ROOT.parent / "tmp" / "omni-rl-audit-swift-20261015"))


class FakeStorage:
    def __init__(self, size=64, identity=1234):
        self.size_bytes = size
        self._cdata = identity

    def nbytes(self):
        return self.size_bytes


class FakeParam:
    def __init__(self, storage=None, requires_grad=False):
        self.device = SimpleNamespace(type="cuda", index=3)
        self.dtype = "torch.bfloat16"
        self.shape = (4, 8)
        self.requires_grad = requires_grad
        self._storage = storage or FakeStorage()

    @property
    def data(self):
        return self

    def stride(self):
        return (8, 1)

    def storage_offset(self):
        return 0

    def element_size(self):
        return 2

    def numel(self):
        return 32

    def untyped_storage(self):
        return self._storage


class FakeCuda:
    def __init__(self, failure_at=None, error=None):
        self.calls = []
        self.failure_at = failure_at
        self.error = error

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 3

    def synchronize(self, device=None):
        self.calls.append(device)
        if len(self.calls) == self.failure_at:
            raise self.error


def make_models(param=None):
    param = FakeParam() if param is None else param
    module = SimpleNamespace(named_parameters=lambda: iter([("frozen.weight", param)]))
    return [SimpleNamespace(module=module, named_parameters=module.named_parameters,
                            buffers=[], expert_parallel_buffers=[])], param


def records(stderr):
    prefix = "[FLEX offload] "
    return [json.loads(line[len(prefix):]) for line in stderr.getvalue().splitlines()
            if line.startswith(prefix)]


class OffloadDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"FLEX_DEBUG_OFFLOAD": "1", "RANK": "3"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @staticmethod
    def module(original, cuda=None):
        return SimpleNamespace(torch=SimpleNamespace(cuda=cuda or FakeCuda()),
                               offload_megatron_model_to_cpu=original)

    def test_disabled_does_not_import_framework_or_patch(self):
        def original(models):
            return models
        module = self.module(original)
        for environment in ({}, {"FLEX_DEBUG_OFFLOAD": "0"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch("importlib.import_module", side_effect=AssertionError("framework import forbidden")):
                diagnostics.install_offload_diagnostics()
                diagnostics.install_offload_diagnostics(module)
        self.assertIs(module.offload_megatron_model_to_cpu, original)
        self.assertEqual(module.torch.cuda.calls, [])

    def test_enabled_installer_is_idempotent_and_preserves_call(self):
        calls = []
        result = object()

        def original(models):
            calls.append(models)
            return result

        models, param = make_models()
        before = dict(param.__dict__)
        module = self.module(original)
        stderr = StringIO()
        with redirect_stderr(stderr):
            diagnostics.install_offload_diagnostics(module)
            wrapped = module.offload_megatron_model_to_cpu
            diagnostics.install_offload_diagnostics(module)
            self.assertIs(module.offload_megatron_model_to_cpu, wrapped)
            self.assertIs(wrapped(models), result)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], models)
        self.assertEqual(param.__dict__, before)
        self.assertEqual(len(module.torch.cuda.calls), 2)
        self.assertIs(wrapped.__wrapped__, original)
        self.assertIn("first_success", {row["phase"] for row in records(stderr)})

    def test_entry_sync_error_is_preserved_and_prevents_original_call(self):
        calls = []

        def original(models):
            calls.append(models)

        error = RuntimeError("prior CUDA operation failed")
        module = self.module(original, FakeCuda(failure_at=1, error=error))
        stderr = StringIO()
        with redirect_stderr(stderr):
            diagnostics.install_offload_diagnostics(module)
            with self.assertRaises(RuntimeError) as caught:
                module.offload_megatron_model_to_cpu(make_models()[0])
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, [])
        self.assertIn("entry_sync_error", {row["phase"] for row in records(stderr)})

    def test_failing_parameter_reports_name_and_before_after_storage(self):
        error = RuntimeError("CUDA invalid argument on frozen parameter")
        models, failing_param = make_models()

        def original(models):
            for name, param in models[0].module.named_parameters():
                param._storage.size_bytes = 0
                raise error

        module = self.module(original)
        stderr = StringIO()
        with redirect_stderr(stderr):
            diagnostics.install_offload_diagnostics(module)
            with self.assertRaises(RuntimeError) as caught:
                module.offload_megatron_model_to_cpu(models)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(module.torch.cuda.calls), 1)
        row, = [row for row in records(stderr) if row["phase"] == "offload_error"]
        parameter = row["site"]["param"]
        self.assertEqual(parameter["before"]["name"], "model[0].frozen.weight")
        self.assertEqual(parameter["before"]["storage_nbytes"], 64)
        self.assertEqual(parameter["current"]["storage_nbytes"], 0)
        self.assertEqual(parameter["before"]["storage_identity"], parameter["current"]["storage_identity"])
        self.assertEqual(failing_param.untyped_storage().nbytes(), 0)

    def test_exit_sync_error_has_its_own_phase_and_preserves_exception(self):
        calls = []

        def original(models):
            calls.append(models)
            return 42

        error = RuntimeError("asynchronous transfer failed")
        module = self.module(original, FakeCuda(failure_at=2, error=error))
        stderr = StringIO()
        with redirect_stderr(stderr):
            diagnostics.install_offload_diagnostics(module)
            with self.assertRaises(RuntimeError) as caught:
                module.offload_megatron_model_to_cpu(make_models()[0])
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)
        self.assertIn("exit_sync_error", {row["phase"] for row in records(stderr)})

    def test_metadata_failure_does_not_replace_upstream_error(self):
        class BrokenParam(FakeParam):
            def untyped_storage(self):
                raise ValueError("metadata is unavailable")

        error = RuntimeError("original offload failure")
        models, param = make_models(BrokenParam())

        def original(models):
            for name, param in models[0].module.named_parameters():
                raise error

        module = self.module(original)
        with redirect_stderr(StringIO()):
            diagnostics.install_offload_diagnostics(module)
            with self.assertRaises(RuntimeError) as caught:
                module.offload_megatron_model_to_cpu(models)
        self.assertIs(caught.exception, error)

    def test_snapshot_exception_does_not_replace_upstream_error(self):
        error = RuntimeError("original error after snapshot failed")

        def original(models):
            raise error

        module = self.module(original)
        stderr = StringIO()
        with redirect_stderr(stderr), patch.object(diagnostics, "_snapshot", side_effect=ValueError("snapshot failed")):
            diagnostics.install_offload_diagnostics(module)
            with self.assertRaises(RuntimeError) as caught:
                module.offload_megatron_model_to_cpu(make_models()[0])
        self.assertIs(caught.exception, error)
        row, = [row for row in records(stderr) if row["phase"] == "offload_error"]
        self.assertEqual(row["summary"]["metadata_error"], "ValueError")

    def test_decorated_upstream_preserves_keyword_call_and_original_traceback(self):
        error = RuntimeError("copy failed inside decorated upstream")
        seen = []

        def original(models):
            for name, param in models[0].module.named_parameters():
                raise error

        @wraps(original)
        def decorated(*args, **kwargs):
            seen.append((args, kwargs))
            return original(*args, **kwargs)

        models, param = make_models()
        module = self.module(decorated)
        stderr = StringIO()
        with redirect_stderr(stderr):
            diagnostics.install_offload_diagnostics(module)
            with self.assertRaises(RuntimeError) as caught:
                module.offload_megatron_model_to_cpu(models=models)
        self.assertIs(caught.exception, error)
        self.assertEqual(seen[0][0], ())
        self.assertIs(seen[0][1]["models"], models)
        row, = [row for row in records(stderr) if row["phase"] == "offload_error"]
        self.assertEqual(row["site"]["param"]["before"]["name"], "model[0].frozen.weight")


class OffloadDiagnosticWiringTests(unittest.TestCase):
    def test_scheduler_installs_diagnostics_before_rollout_engine_creation(self):
        tree = ast.parse((ROOT / "swift_plugin.py").read_text(encoding="utf-8"))
        scheduler, = [node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "FlexStreamScheduler"]
        init, = [node for node in scheduler.body if isinstance(node, ast.FunctionDef)
                 and node.name == "__init__"]
        installers = [node for node in ast.walk(init) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "install_offload_diagnostics"]
        self.assertEqual(len(installers), 1)

    @unittest.skipUnless((SWIFT / "swift/megatron/trainers/rollout_mixin.py").exists(),
                         "Set FLEX_SWIFT_SOURCE to the audited checkout")
    def test_audited_scheduler_initialization_precedes_first_offload_context(self):
        path = SWIFT / "swift/megatron/trainers/rollout_mixin.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls, = [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "MegatronRolloutMixin"]
        init, = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_init_rollout_engine"]
        calls = [node for node in ast.walk(init) if isinstance(node, ast.Call)]
        scheduler_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                           and node.func.attr == "_prepare_scheduler"]
        self.assertEqual(len(scheduler_calls), 1)
        contexts = [node for node in ast.walk(init) if isinstance(node, ast.With)
                    and any(isinstance(item.context_expr, ast.Call)
                            and isinstance(item.context_expr.func, ast.Name)
                            and item.context_expr.func.id == "context" for item in node.items)]
        self.assertTrue(contexts)
        self.assertLess(scheduler_calls[0].lineno, min(node.lineno for node in contexts))


if __name__ == "__main__":
    unittest.main()
