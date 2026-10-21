"""Scoped fix for Megatron TP2 / vLLM TP4 multi-turn termination.

Only FLEX's colocated completion predicate is replaced. The upstream driver,
data partitioning, tokens, masks, logprobs, empty-request collectives and GSPO
loss remain upstream-owned. No installed framework file is rewritten.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
from functools import wraps
import importlib
import inspect
import os
import time


PATCH_ID = "flex-vllm-group-termination-v1"
_ACTIVE_TRAINER = ContextVar("flex_rollout_trainer", default=None)


def progress(message):
    # SWIFT configures its own logger, not necessarily the Python root logger.
    # Print bounded metadata only, never audio/reference/translation contents.
    print(f"[FLEX rollout][rank={os.environ.get('RANK', '?')}] {message}", flush=True)


def log_every():
    value = int(os.environ.get("FLEX_ROLLOUT_LOG_EVERY", "5"))
    if value <= 0:
        raise ValueError("FLEX_ROLLOUT_LOG_EVERY must be positive")
    return value


def validate_rollout_sync_source(source):
    """CPU preflight: reject a changed call site instead of silently missing it."""
    tree = ast.parse(source)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "MegatronRolloutMixin"), None)
    if cls is None:
        raise RuntimeError("SWIFT has no MegatronRolloutMixin")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    generate = methods.get("_generate_completions")
    colocate = methods.get("_colocate_rollout")
    if generate is None or colocate is None:
        raise RuntimeError("SWIFT rollout method contract changed; review FLEX synchronization patch")
    calls = [n for n in ast.walk(generate) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "run_multi_turn"]
    if len(calls) != 1 or not {"scheduler", "rollout_fn", "gather_fn"}.issubset(
            {kw.arg for kw in calls[0].keywords}):
        raise RuntimeError("SWIFT run_multi_turn call contract changed; refusing an unpatched multi-turn run")
    has_tp_collective = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "all_gather_object"
        and any(kw.arg == "group" and isinstance(kw.value, ast.Attribute)
                and isinstance(kw.value.value, ast.Name) and kw.value.value.id == "self"
                and kw.value.attr == "vllm_tp_group" for kw in n.keywords)
        for n in ast.walk(colocate))
    if not has_tp_collective:
        raise RuntimeError("SWIFT vLLM collective contract changed; review FLEX synchronization patch")
    return {"patch_id": PATCH_ID, "call_site_supported": True,
            "completion_scope": "actual vLLM TP group; local predicate for TP1",
            "data_partition_scope": "unchanged Megatron rollout group"}


def make_completion_gather(trainer, gather_object, distributed):
    """Bind to the already-created engine's group, NOT the Megatron data group.

    Explicit dependencies also allow CPU tests to exercise the production
    selection logic with deterministic four-rank collective test doubles.
    """
    if trainer.vllm_mode != "colocate":
        raise RuntimeError("FLEX completion-group override is colocate-only")
    size = int(trainer.vllm_tensor_parallel_size)
    if size < 1:
        raise RuntimeError("Invalid vLLM tensor parallel size")
    group = None
    if size > 1:
        group = getattr(trainer, "vllm_tp_group", None)
        if group is None:
            raise RuntimeError("vLLM TP group is not initialized; refusing unsafe multi-turn rollout")
        actual = distributed.get_world_size(group=group)
        if actual != size:
            raise RuntimeError(f"vLLM completion group size mismatch: expected={size}, actual={actual}")
    interval = log_every()
    round_number = 0
    previous_local = None
    progress(f"sync_ready patch={PATCH_ID} completion_group_size={size} "
             "data_partition=unchanged empty_requests=participate")

    def completion_gather(values):
        nonlocal round_number, previous_local
        if not isinstance(values, list) or len(values) != 1 or type(values[0]) is not bool:
            raise RuntimeError("SWIFT completion predicate must be one boolean per rank")
        round_number += 1
        local_active = values[0]
        report = round_number == 1 or round_number % interval == 0 or local_active != previous_local
        previous_local = local_active
        if report:
            progress(f"completion_check_begin round={round_number} local_active={local_active}")
        # Never skip this gather merely because THIS rank is finished.
        result = list(values) if size == 1 else gather_object(values, group=group)
        if not isinstance(result, list) or len(result) != size or any(type(v) is not bool for v in result):
            raise RuntimeError("Completion gather did not return one boolean per vLLM rank")
        if report or not any(result):
            progress(f"completion_check_end round={round_number} active_ranks={sum(result)}/{size} "
                     f"all_done={not any(result)}")
        return result

    return completion_gather


def install_rollout_sync_patch(module=None):
    """Install once, when the FLEX scheduler is constructed by the trainer.

    Delaying import until scheduler construction avoids importing a half-built
    trainer while SWIFT is initially importing external_plugins. No CUDA group
    is read here. Group lookup happens only inside an active rollout.
    """
    if module is None:
        module = importlib.import_module("swift.megatron.trainers.rollout_mixin")
    if getattr(module, "_flex_rollout_sync_patch", None) == PATCH_ID:
        return
    validate_rollout_sync_source(inspect.getsource(module))
    original_driver = module.run_multi_turn
    driver_signature = inspect.signature(original_driver)
    required = {"requests", "scheduler", "rollout_fn", "gather_fn"}
    if not required.issubset(driver_signature.parameters):
        raise RuntimeError("SWIFT multi-turn driver signature changed; cannot safely install FLEX patch")
    original_generate = module.MegatronRolloutMixin._generate_completions
    from .online import FlexStreamHooks

    @wraps(original_generate)
    def generate_with_context(trainer, *args, **kwargs):
        applies = (getattr(trainer, "vllm_mode", None) == "colocate"
                   and isinstance(getattr(trainer, "multi_turn_scheduler", None), FlexStreamHooks))
        if not applies:
            return original_generate(trainer, *args, **kwargs)
        token = _ACTIVE_TRAINER.set(trainer)
        started = time.monotonic()
        progress("generation_begin (weight sync, wake-up, first inference, then multi-turn)")
        try:
            result = original_generate(trainer, *args, **kwargs)
            progress(f"generation_end local_trajectories={len(result)} elapsed={time.monotonic()-started:.1f}s")
            return result
        except BaseException:
            progress("generation_failed; original exception follows")
            raise
        finally:
            _ACTIVE_TRAINER.reset(token)

    @wraps(original_driver)
    def synchronized_driver(*args, **kwargs):
        trainer = _ACTIVE_TRAINER.get()
        if trainer is None:
            # Other schedulers and non-Megatron/server paths remain untouched.
            return original_driver(*args, **kwargs)
        bound = driver_signature.bind(*args, **kwargs)
        if bound.arguments["scheduler"] is not trainer.multi_turn_scheduler:
            raise RuntimeError("FLEX rollout scheduler/context mismatch")
        bound.arguments["gather_fn"] = make_completion_gather(trainer, module.gather_object, module.torch.distributed)
        original_rollout = bound.arguments["rollout_fn"]
        interval = log_every()
        continuation = 0

        def traced_rollout(requests, config):
            nonlocal continuation
            continuation += 1
            report = continuation == 1 or continuation % interval == 0 or not requests
            start = time.monotonic()
            if report:
                progress(f"continuation_begin call={continuation} local_requests={len(requests)}")
            # Empty local batches MUST still enter the upstream collective.
            outputs = original_rollout(requests, config)
            if report:
                progress(f"continuation_end call={continuation} local_outputs={len(outputs)} "
                         f"elapsed={time.monotonic()-start:.1f}s")
            return outputs

        bound.arguments["rollout_fn"] = traced_rollout
        return original_driver(*bound.args, **bound.kwargs)

    module.run_multi_turn = synchronized_driver
    module.MegatronRolloutMixin._generate_completions = generate_with_context
    module._flex_rollout_sync_patch = PATCH_ID
    progress(f"patch_installed id={PATCH_ID} scope=FLEX-colocate-only framework_files=unchanged")
