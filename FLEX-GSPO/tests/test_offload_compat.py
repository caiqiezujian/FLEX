"""Exercise the audited SWIFT offload/reload code with CPU tensor doubles.

No CUDA or SWIFT import is needed. Set FLEX_SWIFT_SOURCE to the audited source
checkout to run these tests; the original storage-owning functions are compiled
from that checkout rather than reproduced in this test.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flex_gspo.offload_compat import (
    PATCH_ID, build_offload_function, install_offload_compat, validate_offload_source,
)


ROOT = Path(__file__).resolve().parents[1]
SWIFT = Path(os.environ.get(
    "FLEX_SWIFT_SOURCE", ROOT.parents[1] / "tmp" / "omni-rl-audit-swift-20261015"))
SOURCE = SWIFT / "swift/megatron/trainers/utils.py"


@dataclass(frozen=True)
class Device:
    type: str
    index: int | None = None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


def device(value):
    if isinstance(value, Device):
        return value
    name, _, index = str(value).partition(":")
    return Device(name, int(index) if index else None)


class Storage:
    def __init__(self, values, events, name):
        self.values, self.events, self.name = list(values), events, name

    def size(self):
        return len(self.values)

    def resize_(self, size):
        self.events.append(("resize", self.name, size))
        self.values[:] = (self.values + [0] * size)[:size]


class Tensor:
    def __init__(self, values, events, name, *, target="cuda:0", dtype="bf16", shape=None,
                 storage=None):
        self.events, self.name = events, name
        self.device, self.dtype = device(target), dtype
        self._storage = storage if storage is not None else Storage(values, events, name)
        self.shape = tuple(shape) if shape is not None else (len(values),)
        self.copy_error = None

    @property
    def data(self):
        return self

    def storage(self):
        return self._storage

    def size(self):
        return self.shape

    def copy_(self, source, non_blocking=False):
        self.events.append(("copy", self.name, source.name, non_blocking))
        if self.copy_error is not None:
            raise self.copy_error
        self._storage.values[:] = source.storage().values
        return self

    def to(self, target, non_blocking=False):
        self.events.append(("to", self.name, str(target), non_blocking))
        return Tensor(self._storage.values, self.events, self.name, target=target,
                      dtype=self.dtype, shape=self.shape)

    def zero_(self):
        self.events.append(("zero", self.name))
        self._storage.values[:] = [0] * self._storage.size()
        return self


class Parameter:
    def __init__(self, data, requires_grad=True, grad=None):
        self.data, self.requires_grad, self.grad = data, requires_grad, grad

    @property
    def device(self):
        return self.data.device


class Model:
    def __init__(self, parameters):
        self.items = parameters

    def parameters(self):
        return iter(self.items.values())

    def named_parameters(self):
        return iter(self.items.items())

    def buffers(self):
        return iter(())


class DDP(Model):
    def __init__(self, parameters, buffers, expert_buffers):
        super().__init__(parameters)
        self.module = Model(parameters)
        self.buffers, self.expert_parallel_buffers = buffers, expert_buffers


class Torch:
    def __init__(self):
        self.events = []
        self.pinned_error = self.pageable_error = None
        self.sync_errors = {}
        self.sync_count = 0
        self.cuda = SimpleNamespace(is_available=lambda: True, synchronize=self.synchronize)

    def no_grad(self):
        return lambda function: function

    def synchronize(self, target=None):
        self.sync_count += 1
        self.events.append(("sync", str(target)))
        if self.sync_count in self.sync_errors:
            raise self.sync_errors[self.sync_count]

    def empty(self, shape, *, dtype, device, pin_memory):
        self.events.append(("allocate", tuple(shape), dtype, device, pin_memory))
        error = self.pinned_error if pin_memory else self.pageable_error
        if error is not None:
            raise error
        size = 1
        for dim in shape:
            size *= dim
        return Tensor([0] * size, self.events, "host", target=device, dtype=dtype, shape=shape)


def compile_upstream(torch):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"offload_megatron_model_to_cpu", "load_megatron_model_to_gpu"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise AssertionError("Audited SWIFT functions are missing")
    namespace = {
        "torch": torch, "DDP": DDP,
        "gc": SimpleNamespace(collect=lambda: torch.events.append(("gc",))),
        "empty_cache": lambda: torch.events.append(("empty_cache",)),
        "get_current_device": lambda: "cuda:0",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["offload_megatron_model_to_cpu"], namespace["load_megatron_model_to_gpu"]


@unittest.skipUnless(SOURCE.is_file(), "Set FLEX_SWIFT_SOURCE to the audited SWIFT checkout")
class OffloadCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.torch = Torch()
        self.original, self.reload = compile_upstream(self.torch)
        self.offload = build_offload_function(self.original, torch_module=self.torch)

    def make_model(self):
        events = self.torch.events
        dense = SimpleNamespace(param_data=Tensor([1, 2, 3], events, "dense"),
                                grad_data=Tensor([7, 8, 9], events, "dense_grad", dtype="fp32"))
        expert = SimpleNamespace(param_data=Tensor([4, 5], events, "expert", target="cuda:1"),
                                 grad_data=Tensor([6, 7], events, "expert_grad", target="cuda:1", dtype="fp32"))
        dense_view = Tensor([], events, "train_dense", shape=(3,), storage=dense.param_data.storage())
        expert_view = Tensor([], events, "train_expert", target="cuda:1", shape=(2,),
                             storage=expert.param_data.storage())
        frozen = Parameter(Tensor([10, 20], events, "frozen"), requires_grad=False)
        model = DDP({"dense": Parameter(dense_view), "expert": Parameter(expert_view),
                     "frozen": frozen}, [dense], [expert])
        return model, dense, expert, frozen

    def test_audited_contract_accepted_and_changed_copy_policy_rejected(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertEqual(validate_offload_source(source)["patch_id"], PATCH_ID)
        changed = source.replace("buffer.param_data.cpu_data.copy_(buffer.param_data.data, non_blocking=False)",
                                 "buffer.param_data.cpu_data.copy_(buffer.param_data.data, non_blocking=True)")
        with self.assertRaisesRegex(RuntimeError, "implementation changed"):
            validate_offload_source(changed)

    def test_already_disabled_pinning_and_blocking_manual_patch_remain_compatible(self):
        source = inspect.getsource(self.original).replace("pin_memory=True", "pin_memory=False")
        for blocking in (False, True):
            with self.subTest(already_blocking=blocking):
                changed = source.replace(".to('cpu', non_blocking=True)",
                                         ".to('cpu', non_blocking=False)") if blocking else source
                self.assertTrue(validate_offload_source(changed)["call_site_supported"])
                with patch("flex_gspo.offload_compat.inspect.getsource", return_value=changed):
                    offload = build_offload_function(self.original, torch_module=self.torch)
                model, dense, expert, frozen = self.make_model()
                self.torch.events.clear()
                offload([model])
                self.assertTrue(all(not e[-1] for e in self.torch.events if e[0] in ("allocate", "to")))
                self.reload([model])
                self.assertEqual(dense.param_data.storage().values, [1, 2, 3])
                self.assertEqual(expert.param_data.storage().values, [4, 5])
                self.assertEqual(frozen.data.storage().values, [10, 20])

    def test_ddp_dense_expert_trainable_views_survive_offload_and_upstream_reload(self):
        model, dense, expert, frozen = self.make_model()
        before_views = [model.items[name].data for name in ("dense", "expert")]
        self.assertIsNone(self.offload([model]))
        self.assertEqual(self.torch.events[:2], [("sync", "cuda:0"), ("sync", "cuda:1")])
        for buffer, values in ((dense, [1, 2, 3]), (expert, [4, 5])):
            self.assertEqual(buffer.param_data.storage().size(), 0)
            self.assertEqual(buffer.param_data.cpu_data.storage().values, values)
            self.assertEqual(buffer.param_data_size, len(values))
            self.assertEqual(buffer.grad_data.storage().size(), 0)
            copied = next(i for i, event in enumerate(self.torch.events)
                          if event == ("copy", "host", buffer.param_data.name, False))
            freed = self.torch.events.index(("resize", buffer.param_data.name, 0))
            self.assertLess(copied, freed)
        self.assertEqual(frozen.device.type, "cpu")
        self.reload([model])
        for name, view, values in zip(("dense", "expert"), before_views, ([1, 2, 3], [4, 5])):
            self.assertIs(model.items[name].data, view)
            self.assertEqual(view.storage().values, values)
        self.assertEqual(dense.grad_data.storage().values, [0, 0, 0])
        self.assertEqual(expert.grad_data.storage().values, [0, 0])
        self.assertEqual(frozen.device.type, "cuda")
        self.assertEqual(frozen.data.storage().values, [10, 20])

    def test_pageable_buffer_preserves_multidimensional_shape_dtype_and_reuse(self):
        tensor = Tensor([1, 2, 3, 4], self.torch.events, "matrix", dtype="float32", shape=(2, 2))
        buffer = SimpleNamespace(param_data=tensor, grad_data=Tensor([1] * 4, self.torch.events, "grad"))
        model = DDP({"matrix": Parameter(tensor)}, [buffer], [])
        self.offload([model])
        self.assertEqual(self.torch.events[:2], [
            ("sync", "cuda:0"), ("allocate", (2, 2), "float32", "cpu", False)])
        host = tensor.cpu_data
        self.assertEqual(host.shape, (2, 2))
        self.assertEqual(host.dtype, "float32")
        self.assertEqual(host.storage().values, [1, 2, 3, 4])
        self.reload([model])
        tensor.storage().values[:] = [5, 6, 7, 8]
        self.offload([model])
        self.assertIs(tensor.cpu_data, host)
        self.assertEqual(host.storage().values, [5, 6, 7, 8])
        self.assertEqual(len([e for e in self.torch.events if e[0] == "allocate"]), 1)
        self.reload([model])
        self.assertEqual(tensor.storage().values, [5, 6, 7, 8])

    def test_dense_and_expert_buffers_never_attempt_pinned_allocation(self):
        model, dense, expert, _ = self.make_model()
        self.torch.pinned_error = AssertionError("Pinned allocation must not be attempted")
        self.offload([model])
        allocations = [event for event in self.torch.events if event[0] == "allocate"]
        self.assertEqual([event[-1] for event in allocations], [False, False])
        self.reload([model])
        self.assertEqual(dense.param_data.storage().values, [1, 2, 3])
        self.assertEqual(expert.param_data.storage().values, [4, 5])

    def test_allocator_errors_propagate_without_retry_copy_or_storage_free(self):
        for message in ("CUDA error: invalid argument", "CUDA out of memory",
                        "CUDA error: an illegal memory access was encountered", "CPU allocator exhausted"):
            with self.subTest(message=message):
                model, dense, _, _ = self.make_model()
                self.torch.events.clear()
                original_error = RuntimeError(message)
                self.torch.pageable_error = original_error
                with self.assertRaises(RuntimeError) as caught:
                    self.offload([model])
                self.assertIs(caught.exception, original_error)
                self.assertEqual(dense.param_data.storage().values, [1, 2, 3])
                allocations = [e for e in self.torch.events if e[0] == "allocate"]
                self.assertEqual(len(allocations), 1)
                self.assertFalse(allocations[0][-1])
                self.assertFalse(any(e[0] in ("copy", "resize", "to") for e in self.torch.events))

    def test_entry_sync_failure_precedes_all_copy_storage_and_parameter_mutations(self):
        model, dense, _, frozen = self.make_model()
        original_error = RuntimeError("pending device error")
        self.torch.sync_errors[2] = original_error
        with self.assertRaises(RuntimeError) as caught:
            self.offload([model])
        self.assertIs(caught.exception, original_error)
        self.assertEqual(self.torch.events, [("sync", "cuda:0"), ("sync", "cuda:1")])
        self.assertEqual(dense.param_data.storage().values, [1, 2, 3])
        self.assertEqual(frozen.device.type, "cuda")

    def test_frozen_and_reference_parameters_and_gradients_copy_blocking(self):
        model, _, _, _ = self.make_model()
        reference_param = Parameter(Tensor([31, 32], self.torch.events, "reference"),
                                    grad=Tensor([41, 42], self.torch.events, "reference_grad"))
        reference = Model({"reference": reference_param})
        self.offload([model, reference])
        transfers = [e for e in self.torch.events if e[0] == "to"]
        self.assertEqual(transfers, [("to", "frozen", "cpu", False),
                                    ("to", "reference", "cpu", False),
                                    ("to", "reference_grad", "cpu", False)])
        self.reload([reference])
        self.assertEqual(reference_param.device.type, "cuda")
        self.assertEqual(reference_param.data.storage().values, [31, 32])
        self.assertEqual(reference_param.grad.storage().values, [41, 42])

    def test_reused_host_shape_dtype_invariants_still_fail_without_reallocation(self):
        for field, changed in (("shape", (999,)), ("dtype", "bad-dtype")):
            with self.subTest(field=field):
                model, dense, _, _ = self.make_model()
                self.offload([model])
                self.reload([model])
                setattr(dense.param_data.cpu_data, field, changed)
                self.torch.events.clear()
                with self.assertRaisesRegex(AssertionError, "reallocating would reintroduce"):
                    self.offload([model])
                self.assertFalse(any(e[0] in ("allocate", "copy", "resize") for e in self.torch.events))

    def test_copy_failure_propagates_without_free_or_new_allocation(self):
        model, dense, _, _ = self.make_model()
        self.offload([model])
        self.reload([model])
        original_error = RuntimeError("CUDA error: invalid argument")
        dense.param_data.cpu_data.copy_error = original_error
        self.torch.events.clear()
        with self.assertRaises(RuntimeError) as caught:
            self.offload([model])
        self.assertIs(caught.exception, original_error)
        self.assertEqual(dense.param_data.storage().values, [1, 2, 3])
        self.assertFalse(any(e[0] in ("allocate", "resize") for e in self.torch.events))

    def test_install_is_idempotent_and_leaves_original_and_torch_allocator_untouched(self):
        module = SimpleNamespace(offload_megatron_model_to_cpu=self.original)
        allocator = self.torch.empty.__func__
        install_offload_compat(module)
        replacement = module.offload_megatron_model_to_cpu
        self.assertIsNot(replacement, self.original)
        self.assertEqual(replacement._flex_offload_compat, PATCH_ID)
        install_offload_compat(module)
        self.assertIs(module.offload_megatron_model_to_cpu, replacement)
        self.assertFalse(hasattr(self.original, "_flex_offload_compat"))
        self.assertIs(self.torch.empty.__func__, allocator)


if __name__ == "__main__":
    unittest.main()
