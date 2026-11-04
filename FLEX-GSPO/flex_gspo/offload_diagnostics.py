"""Opt-in, read-only diagnostics around SWIFT's existing model offload call.

The original copy/storage algorithm is unchanged. Reported sites locate where
an exception surfaced; they do not establish its CUDA root cause.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
import sys


PATCH_ID = "flex-offload-diagnostics-v1"


def _emit(phase, **fields):
    try:
        print("[FLEX offload] " + json.dumps({"phase": phase, "rank": os.environ.get("RANK", "?"),
                                             **fields}, ensure_ascii=False, default=str),
              file=sys.stderr, flush=True)
    except Exception:
        pass  # A diagnostic write must not replace the original training error.


def _tensor_metadata(tensor, name, storage_names):
    result = {"name": name}
    try:
        result.update(requires_grad=bool(tensor.requires_grad), shape=list(tensor.shape),
                      stride=list(tensor.stride()), device=str(tensor.device), dtype=str(tensor.dtype),
                      numel=tensor.numel(), storage_offset=tensor.storage_offset())
        storage = tensor.untyped_storage()
        result.update(storage_nbytes=storage.nbytes(), storage_identity=int(storage._cdata))
        names = storage_names.get((result["device"], result["storage_identity"]), [])
        result.update(shared_ddp_buffer=bool(names), ddp_buffers=names[:4])
    except Exception as exc:
        result["metadata_error"] = type(exc).__name__
    return result


def _snapshot(models):
    entries, storage_names, errors = {}, {}, []
    # Do not consume an iterator that the original offload function still needs.
    if not isinstance(models, (list, tuple)):
        return entries, storage_names, {"metadata_error": "models is not a list/tuple"}
    parameter_count = frozen_count = shared_count = 0
    for chunk_index, chunk in enumerate(models):
        try:
            for field in ("buffers", "expert_parallel_buffers"):
                buffers = getattr(chunk, field, None)
                if not isinstance(buffers, (list, tuple)):
                    continue
                for index, buffer in enumerate(buffers):
                    tensor = buffer.param_data
                    name = f"model[{chunk_index}].{field}[{index}].param_data"
                    metadata = _tensor_metadata(tensor, name, {})
                    entries[id(buffer)] = entries[id(tensor)] = metadata
                    if "storage_identity" in metadata:
                        key = (metadata["device"], metadata["storage_identity"])
                        storage_names.setdefault(key, []).append(name)
        except Exception as exc:
            errors.append(type(exc).__name__)
    for chunk_index, chunk in enumerate(models):
        try:
            for name, param in chunk.named_parameters():
                metadata = _tensor_metadata(param, f"model[{chunk_index}].{name}", storage_names)
                entries[id(param)] = metadata
                parameter_count += 1
                frozen_count += metadata.get("requires_grad") is False
                shared_count += bool(metadata.get("shared_ddp_buffer"))
        except Exception as exc:
            errors.append(type(exc).__name__)
    return entries, storage_names, {"parameters": parameter_count, "frozen_parameters": frozen_count,
                                    "parameters_sharing_ddp_storage": shared_count,
                                    "ddp_storages": len(storage_names), "metadata_errors": errors[:3]}


def _failure_site(exc, source_function, entries, storage_names):
    site = {"note": "Exception reporting site, not proof of the originating CUDA operation."}
    try:
        traceback = exc.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code is source_function.__code__:
                site.update(file=frame.f_code.co_filename, line=traceback.tb_lineno,
                            function=frame.f_code.co_name)
                for key in ("param", "buffer"):
                    value = frame.f_locals.get(key)
                    if value is None:
                        continue
                    before = entries.get(id(value))
                    tensor = value if key == "param" else getattr(value, "param_data", None)
                    if tensor is not None:
                        name = before.get("name", key) if before else key
                        site[key] = {"before": before,
                                     "current": _tensor_metadata(tensor, name, storage_names)}
            traceback = traceback.tb_next
    except Exception as diagnostic_error:
        site["metadata_error"] = type(diagnostic_error).__name__
    return site


def install_offload_diagnostics(module=None):
    """Patch only the imported rollout binding, and only on explicit opt-in."""
    if os.environ.get("FLEX_DEBUG_OFFLOAD") != "1":
        return
    if module is None:
        module = importlib.import_module("swift.megatron.trainers.rollout_mixin")
    if getattr(module, "_flex_offload_diagnostics_patch", None) == PATCH_ID:
        return
    original = getattr(module, "offload_megatron_model_to_cpu", None)
    try:
        signature = inspect.signature(original)
        parameter, = signature.parameters.values()
        if parameter.name != "models" or parameter.kind not in (
                inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise ValueError("expected offload_megatron_model_to_cpu(models)")
        source_function = inspect.unwrap(original)
        source_lines, line = inspect.getsourcelines(source_function)
        source = {"file": inspect.getsourcefile(source_function), "line": line,
                  "sha256": hashlib.sha256("".join(source_lines).encode("utf-8")).hexdigest()}
        cuda = module.torch.cuda
        if not callable(cuda.synchronize) or not callable(cuda.current_device):
            raise ValueError("CUDA diagnostic hooks are unavailable")
    except Exception as exc:
        raise RuntimeError("FLEX offload diagnostics require readable SWIFT source and the original "
                           "offload_megatron_model_to_cpu(models) contract") from exc
    calls = 0
    success_reported = False

    @wraps(original)
    def diagnostic_offload(*args, **kwargs):
        nonlocal calls, success_reported
        models = signature.bind(*args, **kwargs).arguments["models"]
        calls += 1
        context = {"call": calls, "source": source,
                   "allocator": {name: os.environ.get(name) for name in
                                 ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}}
        # Synchronize before metadata/copies so pending earlier CUDA failures
        # are not mistaken for a defect in this offload invocation.
        try:
            device = cuda.current_device()
            context["current_device"] = device
            cuda.synchronize(device)
        except BaseException as exc:
            _emit("entry_sync_error", **context, error_type=type(exc).__name__)
            raise
        try:
            entries, storage_names, summary = _snapshot(models)
        except Exception as exc:
            entries, storage_names, summary = {}, {}, {"metadata_error": type(exc).__name__}
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _emit("offload_error", **context, summary=summary, error_type=type(exc).__name__,
                  site=_failure_site(exc, source_function, entries, storage_names))
            raise
        try:
            cuda.synchronize(device)
        except BaseException as exc:
            _emit("exit_sync_error", **context, summary=summary, error_type=type(exc).__name__)
            raise
        if not success_reported:
            success_reported = True
            _emit("first_success", **context, summary=summary)
        return result

    module.offload_megatron_model_to_cpu = diagnostic_offload
    module._flex_offload_diagnostics_patch = PATCH_ID
    _emit("install", patch_id=PATCH_ID, source=source,
          note="Opt-in entry/exit synchronization and metadata only; original offload algorithm unchanged.")
