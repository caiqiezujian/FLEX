"""Model-offload compatibility for the audited Megatron-SWIFT implementation.

Keep the upstream DDP storage ownership, buffer reuse and reload protocol.
Only the pinned host allocation and model D2H transfer policy are adapted.
This is a runtime fix, not diagnostic instrumentation; no site-packages file
or global torch API is modified. Optimizer offload remains upstream-owned.
"""
from __future__ import annotations

import ast
from functools import wraps
import importlib
import inspect
import textwrap


PATCH_ID = "flex-model-offload-pageable-v1"


def _constant(node, value):
    return isinstance(node, ast.Constant) and node.value == value


def _offload_ast(source):
    tree = ast.parse(textwrap.dedent(source))
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "offload_megatron_model_to_cpu"), None)
    if function is None or [a.arg for a in function.args.args] != ["models"]:
        raise RuntimeError("SWIFT model-offload signature changed; review host-memory compatibility")
    allocations, transfers, copies, frees = [], [], [], []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        if (node.func.attr == "empty" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"
                and _constant(keywords.get("device"), "cpu")
                and isinstance(keywords.get("pin_memory"), ast.Constant)
                and type(keywords["pin_memory"].value) is bool):
            allocations.append(node)
        if (node.func.attr == "to" and node.args and _constant(node.args[0], "cpu")
                and isinstance(keywords.get("non_blocking"), ast.Constant)
                and type(keywords["non_blocking"].value) is bool):
            transfers.append(node)
        if (node.func.attr == "copy_" and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "cpu_data"
                and _constant(keywords.get("non_blocking"), False)):
            copies.append(node)
        if node.func.attr == "resize_" and node.args and _constant(node.args[0], 0):
            frees.append(node)
    if (len(allocations), len(transfers), len(copies), len(frees)) != (1, 3, 1, 2):
        raise RuntimeError("SWIFT model-offload implementation changed; review host-memory compatibility")
    if not copies[0].lineno < min(n.lineno for n in frees):
        raise RuntimeError("SWIFT offload must finish D2H copying before freeing GPU storage")
    return function, allocations, transfers


def validate_offload_source(source):
    _offload_ast(source)
    return {"patch_id": PATCH_ID, "call_site_supported": True,
            "model_cpu_buffer": "pageable (pin_memory=False)",
            "model_d2h": "blocking; synchronize source devices before offload",
            "optimizer_offload": "unchanged"}


def _synchronize_sources(models, torch_module):
    if not torch_module.cuda.is_available():
        return
    devices = set()
    for model in models:
        for parameter in model.parameters():
            if parameter.device.type == "cuda":
                devices.add(parameter.device)
        for name in ("buffers", "expert_parallel_buffers"):
            buffers = getattr(model, name, ())
            if callable(buffers):  # Ordinary torch Module.buffers is a method.
                continue
            for buffer in buffers:
                for field in ("param_data", "grad_data"):
                    tensor = getattr(buffer, field, None)
                    if tensor is not None and tensor.device.type == "cuda":
                        devices.add(tensor.device)
    if devices:
        for device in sorted(devices, key=str):
            torch_module.cuda.synchronize(device)
    else:
        torch_module.cuda.synchronize()


def build_offload_function(original, torch_module=None):
    raw = inspect.unwrap(original)
    function, allocations, transfers = _offload_ast(inspect.getsource(raw))
    # Compile only this function using its original global bindings. Preserve
    # the full upstream buffer/parameter bookkeeping, not a second implementation.
    next(kw for kw in allocations[0].keywords if kw.arg == "pin_memory").value = ast.Constant(False)
    for transfer in transfers:
        next(kw for kw in transfer.keywords if kw.arg == "non_blocking").value = ast.Constant(False)
    function.decorator_list = []
    namespace = dict(raw.__globals__)
    torch_module = torch_module if torch_module is not None else namespace["torch"]
    namespace["torch"] = torch_module
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(tree, "<flex-model-offload-compat>", "exec"), namespace)
    adapted = torch_module.no_grad()(namespace[function.name])

    @wraps(original)
    def offload(models):
        models = list(models)
        _synchronize_sources(models, torch_module)
        return adapted(models)

    offload._flex_offload_compat = PATCH_ID
    return offload


def install_offload_compat(module=None):
    """Install at FLEX scheduler construction, before first engine offload.

    Only replace rollout_mixin's imported model-offload binding. Training
    utilities, optimizer functions and the global torch allocator stay intact.
    """
    if module is None:
        module = importlib.import_module("swift.megatron.trainers.rollout_mixin")
    original = module.offload_megatron_model_to_cpu
    if getattr(original, "_flex_offload_compat", None) == PATCH_ID:
        return
    module.offload_megatron_model_to_cpu = build_offload_function(original)
