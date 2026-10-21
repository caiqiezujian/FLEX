"""Whole-trajectory GSPO with a memory-bounded, exactly factored backward pass.

For N assistant action tokens across every turn, z=mean(log p-new - log p-old),
s=exp(z), L=-min(s*A, clip(s,1-e_low,1+e_high)*A). First recompute z without
gradients. Let c=dL/dz at this unchanged policy. A second pass backpropagates
c * sum(log p-new for this turn)/N per turn. Summing gives c*grad(z), exactly
grad(L) (away from the usual clipping kink). c is detached, and NO optimizer
step occurs between passes. This is NOT independently clipping per turn.

All ranks own complete models and independent audio groups. Only adapter
gradients are averaged once per update, avoiding per-turn DDP collectives.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from .backend import load_backend, public_trajectory
from .runtime import (Progress, atomic_json, file_sha256, make_schedule,
                      read_resume_metadata, rollout_seed, run_identity)

LOG = logging.getLogger("flex.gspo")


def gspo_coefficient(mean_log_ratio: float, advantage: float, clip_low: float, clip_high: float):
    """Return loss, detached d(loss)/d(mean-log-ratio), and sequence ratio."""
    if not all(math.isfinite(v) for v in (mean_log_ratio, advantage, clip_low, clip_high)):
        raise ValueError("non-finite GSPO input")
    try:
        ratio = math.exp(mean_log_ratio)
    except OverflowError as exc:
        raise ValueError("nonfinite GSPO importance ratio") from exc
    clipped = min(1.0 + clip_high, max(1.0 - clip_low, ratio))
    active = not ((advantage > 0 and ratio > 1.0 + clip_high)
                  or (advantage < 0 and ratio < 1.0 - clip_low))
    return -min(ratio * advantage, clipped * advantage), -advantage * ratio if active else 0.0, ratio


def group_advantages(rewards, mode="group_std", std_floor=0.05):
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("group needs at least two finite rewards")
    mean = sum(rewards) / len(rewards)
    centered = [r - mean for r in rewards]
    if mode == "center":
        return centered
    if mode != "group_std" or std_floor < 0:
        raise ValueError("advantage_normalization must be center or group_std with nonnegative floor")
    std = math.sqrt(sum(x * x for x in centered) / (len(centered) - 1))
    return [x / max(std, std_floor, 1e-6) for x in centered]


def backward_trajectory(backend, trajectory, advantage, train_config, group_size):
    """Two-pass sequence GSPO; invalid sampled trajectories are not dropped."""
    import torch
    turns = trajectory["turns"]
    count = sum(len(turn["token_ids"]) for turn in turns)
    if count == 0:
        # No model action exists to penalize (e.g. context rejected at turn 0).
        raise ValueError("trajectory contains no actions; reduce chunk/context budget before training")
    total_difference, max_difference, first_pass = 0.0, 0.0, []
    with torch.no_grad():
        for turn in turns:
            new = backend.score_turn(turn)
            old = torch.tensor(turn["old_logprobs"], device=new.device, dtype=new.dtype)
            difference = new - old
            first_pass.append(new.detach().cpu())
            total_difference += float(difference.sum())
            max_difference = max(max_difference, float(difference.abs().max()))
    # On-policy, one update per rollout: material mismatch is a bug, not something
    # to conceal with importance weighting. BF16 cached/full passes can differ.
    tolerance = float(train_config.get("parity_tolerance", 0.05))
    if max_difference > tolerance:
        raise RuntimeError(f"rollout/recompute logprob parity failed: max={max_difference:.6f} > {tolerance}")
    mean_tolerance = float(train_config.get("mean_logprob_tolerance", min(
        float(train_config.get("clip_low", .0003)), float(train_config.get("clip_high", .0004))) / 4))
    if abs(total_difference / count) > mean_tolerance:
        raise RuntimeError(f"mean rollout/recompute logprob mismatch {total_difference / count:.8g} exceeds {mean_tolerance}; it would contaminate the narrow GSPO clipping interval")
    loss, coefficient, ratio = gspo_coefficient(
        total_difference / count, advantage,
        float(train_config.get("clip_low", 0.0003)), float(train_config.get("clip_high", 0.0004)),
    )
    beta = float(train_config.get("kl_beta", 0.0))
    reference_kl_sum, second_difference, second_max = 0.0, 0.0, 0.0
    for turn, first in zip(turns, first_pass):
        reference = None
        if beta:
            with torch.no_grad():
                reference = backend.score_turn(turn, reference=True).detach()
        new = backend.score_turn(turn)
        difference = new.detach() - first.to(new.device)
        second_difference += float(difference.sum())
        second_max = max(second_max, float(difference.abs().max()))
        local_loss = coefficient * new.sum() / count
        if beta:
            # Sampled nonnegative k3 estimator, not exact full-vocabulary KL.
            log_ratio = reference - new
            kl = torch.exp(log_ratio) - log_ratio - 1.0
            reference_kl_sum += float(kl.detach().sum())
            local_loss = local_loss + beta * kl.sum() / count
        (local_loss / group_size).backward()
    if second_max > tolerance or abs(second_difference / count) > mean_tolerance:
        raise RuntimeError("no-grad versus gradient-enabled recomputation differs; factored gradient is not trustworthy and no optimizer step was applied")
    return {"gspo_loss": loss, "sequence_ratio": ratio, "advantage": advantage,
            "action_tokens": count, "max_logprob_difference": max_difference,
            "mean_logprob_difference": total_difference / count,
            "recompute_mean_logprob_difference": second_difference / count,
            "sampled_reference_kl": reference_kl_sum / count}


def synchronize_gradients(parameters, world_size):
    import torch
    import torch.distributed as distributed
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if world_size > 1:
            distributed.all_reduce(parameter.grad, op=distributed.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def _collective_phase(function, label, world_size):
    """All ranks report Python failures before entering the next phase.

    A killed process or failed CUDA context still requires torchrun's fail-fast
    handling; no application-level helper can guarantee recovery from those.
    """
    import torch.distributed as distributed
    error, result = None, None
    try:
        result = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        LOG.exception("%s failed", label)
    failures = [error]
    if world_size > 1:
        failures = [None] * world_size
        distributed.all_gather_object(failures, error)
    if any(failures):
        raise RuntimeError(f"{label} aborted across ranks: {failures}")
    return result


def reward_payload(trajectory):
    """Only scoring data crosses ranks, not prompts, logprobs or audio tensors."""
    result = {key: trajectory[key] for key in
              ("id", "valid", "error", "translation", "complete_input", "final_wait", "mock")
              if key in trajectory}
    result["turns"] = [{key: turn[key] for key in
                        ("turn_id", "response", "tag", "delta", "finish_reason",
                         "audio_start_seconds", "audio_end_seconds", "input_seconds", "committed")
                        if key in turn} for turn in trajectory["turns"]]
    return result


def score_shared_group(scorer, task, trajectories, rank, world_size):
    """Rank 0 calls the quality service and computes QALCO for all audio groups."""
    import torch.distributed as distributed
    local = (task, [reward_payload(t) for t in trajectories])
    payloads = [local]
    if world_size > 1:
        payloads = [None] * world_size
        distributed.all_gather_object(payloads, local)
    def score():
        if rank != 0:
            return None
        all_scores = []
        for source_task, candidates in payloads:
            evaluations = scorer.score_group(source_task, candidates)
            if len(evaluations) != len(candidates):
                raise ValueError("scorer must return exactly one evaluation per candidate")
            if not all(math.isfinite(float(e["reward"])) for e in evaluations):
                raise ValueError("scorer returned nonfinite reward")
            all_scores.append(evaluations)
        return all_scores
    evaluations = _collective_phase(score, "shared QALCO reward scoring", world_size)
    if world_size > 1:
        objects = [evaluations]
        distributed.broadcast_object_list(objects, src=0)
        evaluations = objects[0]
    return evaluations[rank]


def _save_checkpoint(backend, optimizer, config, output, step, identity, rank, world_size):
    import torch
    import torch.distributed as distributed

    local_rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
                 "cuda": torch.cuda.get_rng_state()}
    rng_states = [local_rng]
    if world_size > 1:
        rng_states = [None] * world_size
        distributed.all_gather_object(rng_states, local_rng)
    def save():
        if rank != 0:
            return
        target = output / f"checkpoint-{step:08d}"
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        # Failed saves remain as visibly incomplete private directories; never
        # publish latest.json until weights AND optimizer/RNG state are durable.
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-writing-", dir=str(output)))
        backend.save(str(temporary / "adapter"))
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "rng_states": rng_states},
                   temporary / "training_state.pt")
        hashes = {p.relative_to(temporary).as_posix(): file_sha256(p)
                  for p in sorted(temporary.rglob("*")) if p.is_file()}
        atomic_json(temporary / "checkpoint.json",
                    {"format_version": 2, "step": step, "identity": identity,
                     "config": config, "files_sha256": hashes})
        os.replace(temporary, target)
        atomic_json(output / "latest.json", {"step": step, "checkpoint": str(target.resolve())})
        LOG.info("Checkpoint saved: %s", target)
    _collective_phase(save, "checkpoint save", world_size)


def _run_training(config, tasks):
    import torch
    import torch.distributed as distributed
    from .rewards import build_scorer

    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; --validate-only performs CPU configuration checks")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        distributed.init_process_group(
            "nccl", timeout=timedelta(seconds=int(config.get("train", {}).get("distributed_timeout_seconds", 3600))))
    output = Path(config["output_dir"])
    identity = _collective_phase(lambda: run_identity(config, tasks, world_size),
                                 "run identity", world_size)
    resume = config.get("resume_from")
    resume_metadata = None
    if resume:
        resume_metadata = _collective_phase(lambda: read_resume_metadata(resume), "resume integrity", world_size)
        if resume_metadata["identity"] != identity:
            raise ValueError("resume identity changed: tasks/model/reward/algorithm/code/world_size must match; "
                             "only total epochs/max_updates and output/log/save settings may change")
    start_step = resume_metadata["step"] if resume_metadata else 0
    train_config, rollout_config = config.get("train", {}), config.get("rollout", {})
    seed = int(config.get("seed", 42))
    schedule = make_schedule(tasks, world_size, rank,
                             epochs=int(train_config.get("epochs", 1)), seed=seed,
                             shuffle=bool(train_config.get("shuffle", True)))
    maximum = int(train_config.get("max_updates", 2))
    schedule = schedule[:maximum]
    if start_step >= len(schedule):
        raise ValueError("checkpoint already covers the requested schedule; increase epochs/max_updates")
    if rank == 0 and len(tasks) % world_size:
        LOG.warning("Each epoch omits %s shuffled tail tasks for equal rank updates",
                    len(tasks) % world_size)
    def reserve():
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            if any(output.iterdir()):
                raise FileExistsError("output_dir must be new/empty, including when resuming")
            with (output / "run.json").open("x", encoding="utf-8") as handle:
                json.dump({"world_size": world_size,
                           "identity": identity, "config": config, "start_step": start_step},
                          handle, ensure_ascii=False, allow_nan=False, indent=2)
    _collective_phase(reserve, "output reservation", world_size)
    log_path = output / f"train.rank{rank}.jsonl"
    with Progress(float(train_config.get("log_interval_seconds", 15)), rank) as progress:
        device = torch.device("cuda", local_rank)
        random.seed(seed)
        torch.manual_seed(seed)  # identical new adapters before broadcast
        backend_config = dict(config, _progress_callback=progress.status)
        if resume:
            backend_config["resume_adapter_path"] = str(Path(resume) / "adapter")
        progress.status("loading local Thinker and attention LoRA")
        backend = _collective_phase(lambda: load_backend(backend_config, device),
                                    "model loading", world_size)
        if world_size > 1:
            for parameter in backend.trainable:
                distributed.broadcast(parameter.data, src=0)
        scorer = _collective_phase(lambda: build_scorer(config) if rank == 0 else None,
                                    "shared scorer setup", world_size)
        optimizer = torch.optim.AdamW(backend.trainable,
                                     lr=float(train_config.get("learning_rate", 1e-6)),
                                     weight_decay=float(train_config.get("weight_decay", 0.0)))
        if resume:
            def restore():
                # Only load trusted, locally generated checkpoints: torch
                # optimizer/RNG serialization uses pickle.
                state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
                if state["step"] != start_step or len(state["rng_states"]) != world_size:
                    raise ValueError("checkpoint training state/metadata disagreement")
                optimizer.load_state_dict(state["optimizer"])
                rng = state["rng_states"][rank]
                random.setstate(rng["python"])
                torch.set_rng_state(rng["torch"])
                torch.cuda.set_rng_state(rng["cuda"], device)
            _collective_phase(restore, "optimizer and RNG restore", world_size)
        group_size = int(rollout_config.get("group_size", 8))
        LOG.info("rank=%s model ready trainable=%s groups=%s G=%s start_step=%s",
                 rank, backend.trainable_count, len(schedule), group_size, start_step)
        save_every = int(train_config.get("save_every", 100))
        last_saved = -1
        with log_path.open("x", encoding="utf-8") as log:
            for step, (epoch, task) in enumerate(schedule, 1):
                if step <= start_step:
                    continue
                started = time.monotonic()
                def sample_group():
                    trajectories = []
                    for index in range(group_size):
                        progress.status(f"step={step}/{len(schedule)} id={task['id']} candidate={index+1}/{group_size}")
                        trajectories.append(backend.rollout(
                            task, index, rollout_seed(seed, step, rank, index, task["id"])))
                        LOG.info("rank=%s step=%s candidate=%s/%s valid=%s turns=%s",
                                 rank, step, index + 1, group_size, trajectories[-1]["valid"],
                                 len(trajectories[-1]["turns"]))
                    return trajectories
                trajectories = _collective_phase(sample_group, "on-policy rollout", world_size)
                progress.status(f"step={step} quality service and trajectory reward scoring, {group_size} candidates")
                evaluations = score_shared_group(scorer, task, trajectories, rank, world_size)
                rewards = [float(evaluation["reward"]) for evaluation in evaluations]
                advantages = group_advantages(rewards, train_config.get("advantage_normalization", "group_std"),
                                              float(train_config.get("std_floor", 0.05)))
                optimizer.zero_grad(set_to_none=True)
                def backward_group():
                    metrics = []
                    for index, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
                        progress.status(f"step={step} sequence GSPO backward {index+1}/{group_size}")
                        metrics.append(backward_trajectory(backend, trajectory, advantage, train_config, group_size))
                    return metrics
                metrics = _collective_phase(backward_group, "gradient computation", world_size)
                progress.status(f"step={step} synchronizing adapter gradients")
                synchronize_gradients(backend.trainable, world_size)
                norm = torch.nn.utils.clip_grad_norm_(backend.trainable, float(train_config.get("max_grad_norm", 1)))
                if not torch.isfinite(norm):
                    raise RuntimeError("nonfinite gradient norm; optimizer update aborted")
                optimizer.step()
                feasible = sum(bool(e.get("components", {}).get("feasible", False)) for e in evaluations)
                def write_logs():
                    for trajectory, evaluation, metric in zip(trajectories, evaluations, metrics):
                        record = {"step": step, "epoch": epoch + 1, "rank": rank,
                                  "gradient_norm": float(norm), "step_seconds": time.monotonic() - started,
                                  "group_feasible_fraction": feasible / group_size,
                                  "advantage_normalization": train_config.get("advantage_normalization", "group_std"),
                                  "std_floor": float(train_config.get("std_floor", .05)),
                                  "algorithm_variant": "whole-trajectory-GSPO-one-update",
                                  "trajectory": public_trajectory(trajectory), "score": evaluation, "metrics": metric}
                        log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                _collective_phase(write_logs, "training logs", world_size)
                LOG.info("rank=%s update=%s/%s id=%s rewards=%s feasible=%s/%s grad=%.6g elapsed=%.1fs",
                         rank, step, len(schedule), task["id"], [round(r, 5) for r in rewards],
                         feasible, group_size, float(norm), time.monotonic() - started)
                # Release replay tensors before checkpointing / next rollout.
                del trajectories
                backend.clear_audio_cache()
                if step % save_every == 0:
                    progress.status(f"step={step} saving recoverable checkpoint")
                    _save_checkpoint(backend, optimizer, config, output, step, identity, rank, world_size)
                    last_saved = step
        if last_saved != len(schedule):
            _save_checkpoint(backend, optimizer, config, output, len(schedule), identity, rank, world_size)
        if rank == 0:
            atomic_json(output / "complete.json", {"last_step": len(schedule),
                        "start_step": start_step, "world_size": world_size, "identity_sha256": identity["sha256"]})
    return {"rank": rank, "log_path": str(log_path), "groups_processed": len(schedule) - start_step}


def run_training(config, tasks):
    """Always release our process group; torchrun terminates peer ranks on failure."""
    from .metricx import ensure_metricx_ready
    ensure_metricx_ready(config)
    try:
        return _run_training(config, tasks)
    finally:
        import torch.distributed as distributed
        if distributed.is_initialized():
            distributed.destroy_process_group()
