#!/usr/bin/env python3
"""Read-only capabilities/checkpoint checks; no installs/downloads/model loading."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

if __package__:
    from ._bootstrap import PROJECT_ROOT
else:
    from _bootstrap import PROJECT_ROOT

AUDITED_SWIFT_COMMIT = "f15bb91439d756d4a366b51956d72d1f088b7af6"
PROTOCOL = ("<|WAIT|>", "<|DIRECT|>", "<|FLEX|>")
CAPABILITIES = {
    "swift/megatron/trainers/rollout_mixin.py": ("run_multi_turn(", "on_trajectory_start(", "_prepare_vllm_engine"),
    "swift/megatron/trainers/grpo_trainer.py": ("importance_sampling_level", "seq_level_log_weights"),
    "swift/rl_core/data.py": ("def apply_rollout_output", "rollout_output.rollout_infos", "_MULTIMODAL_KEYS"),
    "swift/rollout/multi_turn.py": ("def get_response_token_data", "def on_turn_end", "def on_trajectory_start"),
    "swift/template/templates/qwen.py": ("class Qwen3OmniTemplate", "trim_audio_to_hop_length"),
    "swift/cli/_megatron/rlhf.py": ("megatron_rlhf_main",),
}


def source_capabilities(root: Path) -> dict:
    results = {}
    for relative, needles in CAPABILITIES.items():
        path = root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        results[relative] = {needle: needle in text for needle in needles}
    return results


def inspect_checkpoint(model: Path) -> dict:
    """Inspect tokenizer and safetensors headers on CPU, not parameter tensors."""
    config_path = model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing checkpoint config.json: {model}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_omni_moe":
        raise ValueError(f"Expected qwen3_omni_moe merged SFT checkpoint; got {config.get('model_type')!r}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    ids = {}
    for token in PROTOCOL:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] == tokenizer.unk_token_id:
            raise ValueError(f"{token} is not an existing single token: {encoded}. Restore the trained SFT tokenizer; "
                             "do NOT add random tokens during RL.")
        if tokenizer.decode(encoded, skip_special_tokens=False) != token:
            raise ValueError(f"Protocol token does not round-trip: {token}")
        ids[token] = encoded[0]
    if len(set(ids.values())) != 3:
        raise ValueError("Protocol token IDs are not distinct")
    index_path = model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"] if index_path.is_file() else {}
    if weight_map:
        missing = sorted({name for name in weight_map.values() if not (model / name).is_file()})
        if missing:
            raise ValueError(f"Missing weight shards: {missing[:8]}")
    from safetensors import safe_open
    shapes = {}
    for key in ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight"):
        shard = model / weight_map.get(key, "model.safetensors")
        tied = config.get("thinker_config", {}).get("text_config", {}).get("tie_word_embeddings", False)
        if key.endswith("lm_head.weight") and tied and key not in weight_map and weight_map:
            continue
        if not shard.is_file():
            raise ValueError(f"Cannot find safetensors shard for {key}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                if key.endswith("lm_head.weight") and tied:
                    continue
                raise ValueError(f"Expected weight key missing: {key}")
            shape = list(handle.get_slice(key).get_shape())
        if len(shape) != 2 or max(ids.values()) >= shape[0] or len(tokenizer) > shape[0]:
            raise ValueError(f"Tokenizer/weight vocabulary mismatch: {key}={shape}, tokenizer={len(tokenizer)}, protocol={ids}")
        shapes[key] = shape
    return {"model_type": config["model_type"], "protocol_token_ids": ids, "tokenizer_length": len(tokenizer),
            "weight_shapes": shapes, "note": "Shapes cannot prove protocol rows were trained; preserve the accepted SFT checkpoint."}


def inventory(model: str | None = None, require_training: bool = False, required_gpus: int = 4, cpu_only: bool = False) -> dict:
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": {}, "errors": [], "warnings": [], "audited_swift_commit": AUDITED_SWIFT_COMMIT}
    if require_training and sys.version_info < (3, 10):
        result["errors"].append("Python >= 3.10 is required")
    packages = ("ms-swift", "mcore-bridge", "megatron-core", "transformer-engine", "torch", "transformers", "vllm",
                "peft", "trl", "accelerate", "qwen-omni-utils", "decord", "flash-attn", "numpy", "soundfile", "scipy", "requests", "safetensors")
    for name in packages:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
            if require_training:
                result["errors"].append(f"Missing package: {name}")
    if require_training:
        try:
            from packaging.version import Version
            minimums = {"mcore-bridge": "1.6.3", "megatron-core": "0.16", "peft": "0.15",
                        "transformers": "4.57.0.dev0", "qwen-omni-utils": "0.0.9"}
            for name, minimum in minimums.items():
                installed = result["packages"].get(name)
                if installed and Version(installed) < Version(minimum):
                    result["errors"].append(f"Audited recipe requires {name}>={minimum}; found {installed}")
        except ImportError:
            result["errors"].append("Missing packaging (install requirements.txt)")
    try:
        spec = importlib.util.find_spec("swift")
        if spec is None or spec.origin is None:
            raise RuntimeError("ms-swift is not importable")
        root = Path(spec.origin).resolve().parent.parent
        result["swift_source"] = str(root)
        result["capabilities"] = source_capabilities(root)
        from flex_gspo.rollout_sync import validate_rollout_sync_source
        result["flex_rollout_sync"] = validate_rollout_sync_source(
            (root / "swift/megatron/trainers/rollout_mixin.py").read_text(encoding="utf-8"))
        if os.environ.get("FLEX_REWARD_CONFIG"):
            from flex_gspo.online_reward import load_reward_config
            config = load_reward_config()
            if config["reward"].get("version") in (3, 4):
                from flex_gspo.group_reward import validate_group_reward_source
                result["flex_group_reward"] = validate_group_reward_source(
                    (root / "swift/megatron/trainers/grpo_trainer.py").read_text(encoding="utf-8"))
                if config["reward"].get("version") == 4:
                    from flex_gspo.group_reward import validate_v4_advantage_source
                    result["flex_v4_advantage"] = validate_v4_advantage_source(
                        (root / "swift/rl_core/advantage.py").read_text(encoding="utf-8"))
        absent = [f"{file}: {needle}" for file, checks in result["capabilities"].items() for needle, ok in checks.items() if not ok]
        if absent:
            result["errors"].append("Installed SWIFT lacks audited multi-turn/GSPO/audio APIs: " + "; ".join(absent))
        if (root / ".git").exists() and shutil.which("git"):
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
            result["swift_commit"] = head.stdout.strip()
            if result["swift_commit"] != AUDITED_SWIFT_COMMIT:
                result["warnings"].append("SWIFT source differs from the audited commit; capability checks do not guarantee GPU compatibility.")
    except Exception as exc:
        result["errors"].append(f"Framework check failed: {type(exc).__name__}: {exc}")
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        memory = {}
        for line in meminfo.read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value, *_ = line.split()
                memory[key.rstrip(":")] = round(int(value) / 1024**2, 2)
        result["cpu_memory_gib"] = memory
        if memory.get("MemAvailable", 0) < 128:
            result["warnings"].append("Less than 128 GiB CPU RAM available. Four-rank offload/conversion can exceed this; inspect RAM before scaling.")
    if shutil.which("nvidia-smi"):
        try:
            run = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20, check=False)
            result["nvidia_smi"] = run.stdout.strip() or run.stderr.strip()
        except Exception as exc:
            result["warnings"].append(f"nvidia-smi failed: {exc}")
    if not cpu_only:
        try:
            import torch
            result["torch_cuda_build"] = torch.version.cuda
            result["visible_cuda_devices"] = torch.cuda.device_count()
            if require_training and result["visible_cuda_devices"] < required_gpus:
                result["errors"].append(f"Need {required_gpus} visible CUDA GPUs, found {result['visible_cuda_devices']}")
            if require_training:
                # Exercise compiled ABI compatibility without constructing model/engine.
                import transformer_engine.pytorch  # noqa: F401
                import megatron.core  # noqa: F401
                import mcore_bridge  # noqa: F401
                import vllm  # noqa: F401
                import inspect
                from vllm import EngineArgs
                parameters = inspect.signature(EngineArgs).parameters
                missing = [key for key in ("enable_sleep_mode", "limit_mm_per_prompt", "distributed_executor_backend", "tensor_parallel_size")
                           if key not in parameters]
                if missing:
                    result["errors"].append(f"vLLM lacks required EngineArgs capabilities: {missing}")
        except Exception as exc:
            result["errors"].append(f"CUDA/framework import failed: {type(exc).__name__}: {exc}")
    elif require_training:
        result["warnings"].append("--cpu-only skipped CUDA/compiled-extension validation")
    if model:
        try:
            result["checkpoint"] = inspect_checkpoint(Path(model))
        except Exception as exc:
            result["errors"].append(f"Checkpoint check failed: {type(exc).__name__}: {exc}")
    result["note"] = "Read-only checks; no downloads, model loading, training or MetricX scoring. A real multi-turn GPU smoke run is still required."
    result["ok"] = not result["errors"]
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--require-training", action="store_true")
    p.add_argument("--required-gpus", type=int, default=4)
    p.add_argument("--cpu-only", action="store_true")
    a = p.parse_args()
    data = inventory(a.model, a.require_training, a.required_gpus, a.cpu_only)
    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
    return 0 if data["ok"] or not a.require_training else 2


if __name__ == "__main__":
    raise SystemExit(main())
