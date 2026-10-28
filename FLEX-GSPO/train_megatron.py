#!/usr/bin/env python3
"""Launch fresh multi-turn GSPO through Megatron-SWIFT, not static trajectory replay.

--dry-run imports no CUDA/framework modules and writes nothing. Arguments are
passed as a subprocess argv list, never executed through a shell.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
AUDITED_SWIFT_COMMIT = "f15bb91439d756d4a366b51956d72d1f088b7af6"
V4_QUALITY_CASES = ROOT / "configs" / "quality_regressions_v4.jsonl"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("model", help="Local merged SFT HF checkpoint, including its trained tokenizer")
    p.add_argument("dataset", help="Prepared train.jsonl, NOT trajectories.jsonl")
    p.add_argument("output", help="New run directory, or existing run with --resume-from")
    p.add_argument("--reward-config", default=str(ROOT / "configs" / "default.json"))
    p.add_argument("--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=16)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--steps-per-generation", type=int, default=1)
    p.add_argument("--train-iters", type=int, default=3000)
    p.add_argument("--max-turns", type=int, default=65, help="Includes one EOF text round; 65 allows 64 audio chunks")
    p.add_argument("--max-chunk-seconds", type=float, default=2.0, help="vLLM audio profiling length cap; must cover prepared input_seconds")
    p.add_argument("--image-slots", type=int, choices=(0, 1), default=1,
                   help="Keep one image profiling slot for accepted Omni initialization compatibility; requests stay audio-only")
    p.add_argument("--max-length", type=int, default=8192, help="Rollout prompt budget; vLLM total budget adds max-new-tokens, not a final training truncation cap")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Per audio turn, not entire trajectory")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--vllm-memory", type=float, default=0.40)
    p.add_argument("--vllm-max-num-seqs", type=int, default=16)
    p.add_argument("--vllm-mm-processor-cache-gb", type=float, default=0.0,
                   help="0 disables multimodal processor/transport caches to bypass the observed vLLM 0.19.1 cache-clear hang; KV prefix caching stays enabled")
    p.add_argument("--recompute-granularity", choices=("full", "selective"), default="full",
                   help="Keep full for the accepted recipe; selective is an opt-in speed/memory experiment")
    p.add_argument("--performance-log", choices=("on", "off"), default="off",
                   help="Optional CPU wall-clock phase JSONL; no CUDA synchronization or new collectives")
    p.add_argument("--master-port", type=int, default=29581)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-num-proc", type=int, default=4)
    p.add_argument("--rollout-log-every", type=int, default=5,
                   help="Log multi-turn collective/generation progress every N rounds (also first, idle, and final rounds)")
    p.add_argument("--resume-from", help="Native Megatron adapter checkpoint including optimizer/RNG; not HF merged weights")
    p.add_argument("--swift-source", help="Optional local audited ms-swift source checkout")
    p.add_argument("--dry-run", action="store_true", help="Print only: no file writes, scoring, or GPU use")
    return p


def validate_options(a: argparse.Namespace) -> None:
    gpu_ids = [x.strip() for x in a.gpu_ids.split(",") if x.strip()]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("This recipe requires four distinct GPU IDs, e.g. --gpu-ids 0,1,2,3")
    for name in ("group_size", "global_batch_size", "micro_batch_size", "steps_per_generation", "train_iters",
                 "max_turns", "max_length", "max_new_tokens", "lora_rank", "save_steps", "vllm_max_num_seqs", "dataset_num_proc", "rollout_log_every"):
        if getattr(a, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if a.group_size < 2:
        raise ValueError("GSPO requires at least two candidates per group")
    generation_batch = a.global_batch_size * a.steps_per_generation
    if generation_batch % a.group_size:
        raise ValueError("global-batch-size * steps-per-generation must be divisible by group-size")
    # TP2/PP1/CP1 on four cards => DP2. EP is not a further divisor of DP.
    if (generation_batch // a.group_size) % (2 * a.micro_batch_size):
        raise ValueError("Prompt count per generation must be divisible by DP(2) * micro-batch-size")
    if a.global_batch_size % (2 * a.micro_batch_size):
        raise ValueError("global-batch-size must be divisible by DP(2) * micro-batch-size")
    if generation_batch < 4:
        raise ValueError("Four ranks require at least four trajectories per generation batch")
    if not 0 < a.vllm_memory < 0.9:
        raise ValueError("--vllm-memory must be between 0 and 0.9; GPUs are shared with training")
    if not all(math.isfinite(v) for v in (a.lr, a.temperature, a.kl_beta, a.lora_alpha, a.vllm_memory,
                                        a.max_chunk_seconds, a.vllm_mm_processor_cache_gb)):
        raise ValueError("All floating-point arguments must be finite (no NaN/Infinity)")
    if a.vllm_mm_processor_cache_gb < 0:
        raise ValueError("--vllm-mm-processor-cache-gb must be nonnegative; use 0 to disable multimodal caches")
    if not 0.1 <= a.max_chunk_seconds <= 8:
        raise ValueError("--max-chunk-seconds must be in 0.1..8 seconds")
    if a.lr <= 0 or a.temperature <= 0 or a.kl_beta < 0 or a.lora_alpha <= 0:
        raise ValueError("lr, temperature, lora-alpha must be positive; kl-beta must be nonnegative")
    if a.max_length <= a.max_new_tokens:
        raise ValueError("max-length must exceed per-turn max-new-tokens")
    if not 1024 <= a.master_port <= 65535:
        raise ValueError("master-port must be in 1024..65535")


def load_settings(a: argparse.Namespace) -> dict:
    from flex_gspo.online_reward import load_reward_config
    from flex_gspo.qalco import validate_settings
    return validate_settings(load_reward_config(a.reward_config))


def build_command(a: argparse.Namespace, *, reward_settings=None) -> list[str]:
    """Flags checked against the audited upstream implementation."""
    validate_options(a)
    settings = load_settings(a) if reward_settings is None else reward_settings
    # v4 normalizes quality and gated delay inside complete prompt groups.
    # Upstream still subtracts the group mean; a second std division would
    # amplify near-ties again and undo the configured component scale floors.
    scale_rewards = "none" if settings["version"] == 4 else "group"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4", "--nnodes", "1",
           "--master_port", str(a.master_port), "--module", "swift.cli._megatron.rlhf"]
    pairs = {
        "rlhf_type": "grpo", "model": a.model, "model_type": "qwen3_omni_moe",
        "template": "flex_qwen3_omni", "dataset": a.dataset, "output_dir": a.output,
        "external_plugins": str(ROOT / "swift_plugin.py"), "multi_turn_scheduler": "flex_stream",
        "reward_funcs": "qalco_online", "reward_weights": "1", "remove_unused_columns": "false",
        "load_from_cache_file": "false", "split_dataset_ratio": "0", "dataset_shuffle": "true",
        "train_iters": a.train_iters, "global_batch_size": a.global_batch_size,
        "micro_batch_size": a.micro_batch_size, "num_generations": a.group_size,
        "steps_per_generation": a.steps_per_generation, "num_iterations": 1,
        "tuner_type": "lora", "lora_rank": a.lora_rank, "lora_alpha": a.lora_alpha,
        "target_modules": "all-linear", "freeze_llm": "false", "freeze_vit": "true",
        "freeze_aligner": "true", "bf16": "true", "bridge_backend": "mcore-bridge",
        "tensor_model_parallel_size": "2", "expert_model_parallel_size": "2",
        "pipeline_model_parallel_size": "1", "context_parallel_size": "1",
        "sequence_parallel": "true", "padding_free": "false",
        "moe_grouped_gemm": "true", "moe_permute_fusion": "true", "moe_aux_loss_coeff": "0",
        "recompute_granularity": a.recompute_granularity,
        "gradient_accumulation_fusion": "false", "attention_backend": "flash",
        "lr": a.lr, "min_lr": a.lr * 0.1, "lr_warmup_fraction": "0.03", "clip_grad": "1",
        "beta": a.kl_beta, "importance_sampling_level": "sequence", "epsilon": "0.0003",
        "epsilon_high": "0.0004", "loss_type": "grpo", "scale_rewards": scale_rewards,
        "dynamic_sample": "false", "overlong_filter": "false", "truncation_strategy": "delete",
        "max_length": a.max_length, "max_completion_length": a.max_new_tokens,
        "completion_length_limit_scope": "per_round", "max_turns": a.max_turns,
        "temperature": a.temperature, "top_p": "1", "top_k": "-1", "repetition_penalty": "1",
        "use_vllm": "true", "vllm_mode": "colocate", "vllm_tensor_parallel_size": "4",
        "vllm_enable_lora": "false", "vllm_gpu_memory_utilization": a.vllm_memory,
        "vllm_max_model_len": a.max_length + a.max_new_tokens,
        "vllm_max_num_seqs": a.vllm_max_num_seqs, "vllm_enforce_eager": "true",
        "vllm_limit_mm_per_prompt": json.dumps({"audio": {"count": a.max_turns, "length": round(a.max_chunk_seconds * 16000)},
                                                "image": a.image_slots, "video": 0}, separators=(",", ":")),
        "vllm_enable_prefix_caching": "true", "vllm_mm_processor_cache_gb": a.vllm_mm_processor_cache_gb,
        "sleep_level": "2", "offload_model": "true", "offload_optimizer": "true", "offload_bridge": "false",
        "save_safetensors": "true", "merge_lora": "true", "save_steps": a.save_steps,
        "no_save_optim": "false", "no_save_rng": "false", "save_total_limit": "3",
        "finetune": "false" if a.resume_from else "true", "no_load_optim": "false", "no_load_rng": "false",
        # Upstream completion logging re-encodes messages without their audios.
        # Keep it disabled for this audio-only recipe. Our complete trajectory
        # and reward JSONL records remain enabled independently.
        "logging_steps": "1", "log_completions": "false", "report_to": "tensorboard",
        "dataset_num_proc": a.dataset_num_proc, "dataloader_num_workers": "0",
        "dataloader_persistent_workers": "false", "seed": a.seed,
    }
    # Selective recomputation rejects recompute_method in the audited framework.
    # Keep the original full-mode flags exactly; never pass them in selective mode.
    if a.recompute_granularity == "full":
        pairs.update({"recompute_method": "uniform", "recompute_num_layers": "1"})
    if settings["version"] == 4:
        pairs.update({"advantage_estimator": "grpo", "kl_in_reward": "false"})
    if a.resume_from:
        pairs["mcore_adapter"] = a.resume_from
    for key, value in pairs.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def build_environment(a: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": a.gpu_ids, "ENABLE_AUDIO_OUTPUT": "0", "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1", "TOKENIZERS_PARALLELISM": "false",
                "FLEX_REWARD_CONFIG": str(Path(a.reward_config).resolve()),
                "FLEX_REWARD_LOG_DIR": str(Path(a.output).resolve() / "rewards"),
                "FLEX_ROLLOUT_LOG_EVERY": str(a.rollout_log_every)})
    # Explicit off must override an inherited experiment setting.
    if a.performance_log == "on":
        env["FLEX_PERF_LOG_DIR"] = str(Path(a.output).resolve() / "performance")
    else:
        env.pop("FLEX_PERF_LOG_DIR", None)
    paths = [str(ROOT)]
    if a.swift_source:
        paths.insert(0, str(Path(a.swift_source).resolve()))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # Do not force expandable_segments=True: incompatible with some CuMem pools.
    return env


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_local_inputs(a: argparse.Namespace) -> dict:
    for name, path in (("model", a.model), ("dataset", a.dataset), ("reward config", a.reward_config),
                       ("plugin", ROOT / "swift_plugin.py")):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if not Path(a.model).is_dir() or not Path(a.dataset).is_file():
        raise ValueError("model must be a checkpoint directory and dataset must be a JSONL file")
    with Path(a.reward_config).open(encoding="utf-8-sig") as f:
        json.load(f)
    from flex_gspo.audio import encode_chunk_ref
    from flex_gspo.online import unpack_task
    from flex_gspo.prompts import initial_messages
    count, digest = 0, hashlib.sha256()
    print("[preflight] Validating and hashing task manifest (metadata only; no audio decode)...", flush=True)
    with Path(a.dataset).open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw.decode("utf-8-sig" if line_number == 1 else "utf-8"))
            if not isinstance(row, dict) or not isinstance(row.get("flex_task"), str):
                raise ValueError(f"Line {line_number}: expected prepared train.jsonl with JSON-string flex_task")
            task = unpack_task(row["flex_task"])
            if row.get("messages") != initial_messages() or row.get("audios") != [encode_chunk_ref(task, 0)]:
                raise ValueError(f"Line {line_number}: initial actor input differs from SFT prompt + first virtual audio chunk")
            if row.get("images") or row.get("videos"):
                raise ValueError(f"Line {line_number}: audio-only training cannot contain images/videos")
            if len(task["chunks"]) + 1 > a.max_turns:
                raise ValueError(f"Line {line_number}: task needs {len(task['chunks']) + 1} turns including EOF but --max-turns={a.max_turns}")
            if any(chunk["input_seconds"] > a.max_chunk_seconds for chunk in task["chunks"]):
                raise ValueError(f"Line {line_number}: audio chunk exceeds --max-chunk-seconds={a.max_chunk_seconds}")
            count += 1
            if count % 10000 == 0:
                print(f"[preflight] Validated tasks={count}", flush=True)
    minimum = a.global_batch_size * a.steps_per_generation // a.group_size
    if count < minimum:
        raise ValueError(f"Only {count} tasks; this batch configuration requires at least {minimum}")
    print(f"[preflight] Manifest passed: tasks={count}", flush=True)
    if a.swift_source and not (Path(a.swift_source) / "swift" / "cli" / "_megatron" / "rlhf.py").is_file():
        raise ValueError("--swift-source is not an ms-swift checkout with Megatron RLHF")
    if a.resume_from and not Path(a.resume_from).is_dir():
        raise FileNotFoundError(f"Native resume checkpoint missing: {a.resume_from}")
    if a.resume_from:
        resume = Path(a.resume_from)
        tracker = resume / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise ValueError("--resume-from must point to a native Megatron checkpoint containing latest_checkpointed_iteration.txt")
        iteration = int(tracker.read_text().strip())
        if iteration <= 0 or not (resume / f"iter_{iteration:07d}" / "common.pt").is_file():
            raise ValueError("Native checkpoint is incomplete (missing positive iteration or common.pt)")
        # Swift also copies tracker/common.pt/args.json into the HF -merged
        # export. Those files alone do NOT include trainable/optimizer shards.
        checkpoint_data = resume / f"iter_{iteration:07d}"
        metadata_only = {"common.pt", "metadata.json", ".metadata"}
        has_payload = any(path.is_file() and path.name not in metadata_only and path.stat().st_size > 0
                          for path in checkpoint_data.rglob("*"))
        if resume.name.endswith("-merged") or not has_payload:
            raise ValueError("Resume requires native sharded model/optimizer payloads, not checkpoint-N-merged "
                             "or a directory containing only tracker/common.pt metadata")
        checkpoint_args = resume / "args.json"
        if not checkpoint_args.is_file():
            raise ValueError("Resume checkpoint is missing args.json; cannot verify optimizer/RNG availability")
        saved = json.loads(checkpoint_args.read_text(encoding="utf-8"))
        if saved.get("no_save_optim", True) or saved.get("no_save_rng", True):
            raise ValueError("Checkpoint did not save optimizer/RNG: use accepted SFT weights to start a NEW run instead")
        if a.train_iters <= iteration:
            raise ValueError(f"--train-iters must exceed restored iteration {iteration}")
    output = Path(a.output)
    if output.exists() and any(output.iterdir()) and not a.resume_from:
        raise ValueError("Output directory is nonempty. Choose a new directory or use --resume-from; nothing overwritten.")
    return {"sha256": digest.hexdigest(), "tasks": count}


def run_identity(a: argparse.Namespace, manifest: dict) -> dict:
    """Bind resume to data, reward, model metadata and training/code settings.

    The 60+ GB model is not rehashed. Its path, index/config/tokenizer hashes,
    and all safetensors file sizes/mtimes are recorded; this is a metadata
    fingerprint, not a cryptographic content proof of all parameter values.
    """
    model = Path(a.model).resolve()
    metadata_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                      "added_tokens.json", "model.safetensors.index.json", "preprocessor_config.json")
    model_metadata = {name: hash_file(model / name) for name in metadata_files if (model / name).is_file()}
    weights = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in sorted(model.glob("*.safetensors"))}
    ignored = {"output", "dataset", "reward_config", "resume_from", "dry_run", "gpu_ids", "master_port",
               "save_steps", "dataset_num_proc", "rollout_log_every", "performance_log"}
    # Include train_iters: changing the LR decay horizon mid-run is not exact resume.
    settings = {k: v for k, v in vars(a).items() if k not in ignored}
    source_files = [ROOT / "swift_plugin.py", ROOT / "train_megatron.py", *sorted((ROOT / "flex_gspo").glob("*.py"))]
    identity = {"schema": 1, "framework_commit": AUDITED_SWIFT_COMMIT, "model_path": str(model),
            "model_metadata_sha256": model_metadata, "weight_file_metadata": weights,
            "dataset_sha256": manifest["sha256"], "dataset_tasks": manifest["tasks"],
            "reward_config_sha256": hash_file(Path(a.reward_config)), "settings": settings,
            "code_sha256": {str(p.relative_to(ROOT)): hash_file(p) for p in source_files}}
    if load_settings(a)["version"] == 4:
        identity["quality_regression_cases_sha256"] = hash_file(V4_QUALITY_CASES)
    return identity


def verify_resume_identity(a: argparse.Namespace, identity: dict) -> None:
    if not a.resume_from:
        return
    # Swift may put its checkpoint below output/vN-.../checkpoint-N.
    resume = Path(a.resume_from).resolve()
    roots = [Path(a.output).resolve(), *list(resume.parents)[:3]]
    manifests = {p for directory in roots for p in directory.glob("launch_*.json")}
    prior = []
    for path in sorted(manifests, reverse=True):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "run_identity" in document:
            prior.append(document["run_identity"])
    if not prior:
        raise ValueError("Cannot verify resume: no launch_*.json run identity beside the original checkpoint/run")
    if not any(old == identity for old in prior):
        raise ValueError("Resume rejected: model/data/reward/code or training settings differ from the original run. "
                         "Restore the original configuration, including train-iters/LR horizon, or start a new run.")


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        settings = load_settings(a)
        cmd, env = build_command(a, reward_settings=settings), build_environment(a)
        if a.vllm_mm_processor_cache_gb > 0:
            print("[launch] WARNING: Multimodal caches are enabled explicitly. This removes the default "
                  "bypass for the observed vLLM 0.19.1 cache-clear hang; verify an upstream fix and a "
                  "multi-step sleep/wake test before a long run. KV prefix caching is independent.",
                  file=sys.stderr, flush=True)
        plan = {"audited_swift_commit": AUDITED_SWIFT_COMMIT, "argv": cmd,
                "reward_settings": settings,
                "quality_preflight_cases": str(V4_QUALITY_CASES) if settings["version"] == 4 else "built-in basic cases",
                "reward_normalization": {
                    "component_normalization": settings["version"] == 4,
                    "upstream_scale_rewards": "none" if settings["version"] == 4 else "group",
                    "upstream_group_mean_baseline": True,
                    "note": "v4: component scaling then group mean centering only; no second std division. "
                            "Normalized reward means are not translation-quality trends."},
                "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "ENABLE_AUDIO_OUTPUT", "FLEX_REWARD_CONFIG",
                                                     "FLEX_REWARD_LOG_DIR", "FLEX_ROLLOUT_LOG_EVERY", "PYTHONPATH")},
                "rollout_sync": "FLEX plugin: completion predicate uses actual vLLM TP group; Megatron data groups unchanged",
                "sampling": {"group_size": a.group_size, "generation_batch_size": a.global_batch_size * a.steps_per_generation,
                             "fresh_prompts_per_generation": a.global_batch_size * a.steps_per_generation // a.group_size},
                "performance": {"phase_logging": a.performance_log,
                                "log_dir": env.get("FLEX_PERF_LOG_DIR"),
                                "vllm_max_num_seqs": a.vllm_max_num_seqs,
                                "vllm_mm_processor_cache_gb": a.vllm_mm_processor_cache_gb,
                                "micro_batch_size": a.micro_batch_size,
                                "recompute_granularity": a.recompute_granularity,
                                "note": "Candidate configuration, not a measured speedup; compare steady-state throughput and peak VRAM."},
                "note": "Source-audited recipe; CPU checks do not certify GPU training or translation quality."}
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        print("[launch] " + shlex.join(cmd), flush=True)
        if a.dry_run:
            return 0
        manifest = validate_local_inputs(a)
        plan["run_identity"] = run_identity(a, manifest)
        verify_resume_identity(a, plan["run_identity"])
        if settings["version"] not in (2, 3, 4):
            raise ValueError("New training requires QALCO v2/v3/v4 / COMET22 / EOF. v1 is retained for historical audit only.")
        # Fail before GPU initialization; check both transport and basic semantic ordering.
        probe_cmd = [sys.executable, str(ROOT / "Scripts" / "check_comet22_service.py"), "--config", a.reward_config]
        if settings["version"] == 4:
            probe_cmd.extend(["--cases-jsonl", str(V4_QUALITY_CASES)])
        subprocess.run(probe_cmd, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / "Scripts" / "check_environment.py"), "--model", a.model,
                        "--require-training", "--required-gpus", "4"], env=env, check=True)
        output = Path(a.output)
        output.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        env["FLEX_RUN_ID"] = hashlib.sha256(json.dumps(plan["run_identity"], sort_keys=True).encode()).hexdigest()
        env["FLEX_SESSION_ID"] = uuid.uuid4().hex
        plan.update(run_id=env["FLEX_RUN_ID"], session_id=env["FLEX_SESSION_ID"])
        with (output / f"launch_{stamp}.json").open("x", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        print("[launch] Starting four ranks; fresh online rollouts, no offline candidate file.", flush=True)
        return subprocess.run(cmd, env=env, check=False).returncode
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[launch] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
