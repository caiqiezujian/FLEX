#!/usr/bin/env python3
"""Validate or train a FLEX GSPO attention LoRA adapter with QALCO rewards."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

from flex_gspo.runtime import read_resume_metadata


def load_and_validate(config_path, *, limit=None, output_dir=None,
                      overrides=None, resume_from=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for section, name, value in overrides or []:
        if value is not None:
            if section:
                config.setdefault(section, {})[name] = value
            else:
                config[name] = value
    if output_dir is not None:
        config["output_dir"] = output_dir
    if resume_from is not None:
        config["resume_from"] = str(Path(resume_from).resolve())
    if config.get("resume_from"):
        read_resume_metadata(config["resume_from"])
    for key in ("model_path", "manifest_path", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"config requires a nonempty {key}")
    model_path = Path(config["model_path"])
    if not model_path.is_dir():
        raise ValueError("model_path must be an existing local merged SFT checkpoint directory")
    for filename in ("config.json", "tokenizer_config.json"):
        if not (model_path / filename).is_file():
            raise ValueError(f"merged checkpoint missing {filename}")
    if not (any(model_path.glob("*.safetensors")) or any(model_path.glob("pytorch_model*.bin"))):
        raise ValueError("merged checkpoint contains no model weight files")
    output = Path(config["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output_dir must be absent or empty; resume writes into a NEW output directory")
    from flex_gspo.artifacts import load_training_tasks
    tasks = load_training_tasks(config["manifest_path"])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("empty manifest selection")
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("FLEX_PREFLIGHT_WORLD_SIZE", "1")))
    if len(tasks) < world_size:
        raise ValueError("training selection needs at least one complete audio task per rank")
    for task in tasks:
        if not Path(task["audio_path"]).is_file():
            raise ValueError(f"{task['id']}: audio file does not exist")
        for chunk in task["chunks"]:
            if float(chunk["input_seconds"]) > 8:
                raise ValueError("model input chunk exceeds the audited 8 second limit")
    rollout = config.get("rollout", {})
    if float(rollout.get("temperature", 1)) != 1 or float(rollout.get("top_p", 1)) != 1 or int(rollout.get("top_k", 0)) not in (0, -1):
        raise ValueError("v1 requires temperature=1, top_p=1, top_k=0/-1 for exact sampled-policy logprobs")
    if type(rollout.get("group_size", 8)) is not int or rollout.get("group_size", 8) < 2:
        raise ValueError("group_size must be >= 2")
    if not 0 < int(rollout.get("max_new_tokens", 128)) < int(rollout.get("max_context_tokens", 4096)):
        raise ValueError("require 0 < max_new_tokens < max_context_tokens")
    training = config.get("train", {})
    model = config.get("model", {})
    for key, default in (("lora_rank", 8), ("lora_alpha", 32)):
        if type(model.get(key, default)) is not int or model.get(key, default) < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    for key, default in (("learning_rate", 1e-6), ("max_grad_norm", 1), ("parity_tolerance", 0.05)):
        value = float(training.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"train.{key} must be finite and positive")
    for key, default in (("clip_low", .0003), ("clip_high", .0004)):
        if not 0 < float(training.get(key, default)) < 1:
            raise ValueError(f"train.{key} must be between 0 and 1")
    if not math.isfinite(float(training.get("kl_beta", 0))) or float(training.get("kl_beta", 0)) < 0:
        raise ValueError("kl_beta must be finite and nonnegative")
    if not math.isfinite(float(training.get("std_floor", .05))) or float(training.get("std_floor", .05)) < 0:
        raise ValueError("std_floor must be finite and nonnegative")
    if training.get("advantage_normalization", "group_std") not in ("center", "group_std"):
        raise ValueError("advantage_normalization must be center or group_std")
    for name, default in (("max_updates", 2), ("epochs", 1), ("save_every", 100),
                          ("distributed_timeout_seconds", 3600)):
        if type(training.get(name, default)) is not int or training.get(name, default) < 1:
            raise ValueError(f"train.{name} must be a positive integer")
    interval = training.get("log_interval_seconds", 15)
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("log_interval_seconds must be positive and finite")
    if type(training.get("shuffle", True)) is not bool:
        raise ValueError("train.shuffle must be boolean")
    if not math.isfinite(float(training.get("weight_decay", 0))) or float(training.get("weight_decay", 0)) < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if "mean_logprob_tolerance" in training and not 0 < float(training["mean_logprob_tolerance"]) < min(float(training.get("clip_low", .0003)), float(training.get("clip_high", .0004))):
        raise ValueError("mean_logprob_tolerance must be positive and smaller than both GSPO clip widths")
    calibration = config.get("reward", {}).get("calibration_acknowledged", False)
    if type(calibration) is not bool:
        raise ValueError("reward.calibration_acknowledged must be a boolean, not a string")
    for task in tasks:
        if not isinstance(task.get("source_text"), str) or not task["source_text"].strip():
            raise ValueError(f"{task['id']}: training quality reward requires source_text")
    from flex_gspo.rewards import validate_reward_setup
    validate_reward_setup(tasks, config)
    return config, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON config; paths are resolved against current directory")
    parser.add_argument("--validate-only", action="store_true", help="check paths/config/data; no model or training")
    parser.add_argument("--limit", type=int, help="select at most this many manifest tasks globally")
    parser.add_argument("--output-dir", help="override output_dir; each training run requires a new directory")
    parser.add_argument("--model-path", help="local merged SFT checkpoint")
    parser.add_argument("--tasks", help="existing FLEX-GSPO-Data train.tasks.jsonl; not trajectories.jsonl")
    parser.add_argument("--group-size", type=int, help="online candidates per audio, default config G=8")
    parser.add_argument("--max-updates", type=int, help="total updates including resumed steps")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-from", help="completed checkpoint directory; output-dir must still be NEW")
    parser.add_argument("--allow-uncalibrated-reward", action="store_true",
                        help="explicit smoke-test override; does not make reward thresholds validated")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Checking training paths, manifests and configuration (no model loaded yet)")
    config, tasks = load_and_validate(
        args.config, limit=args.limit, output_dir=args.output_dir,
        resume_from=args.resume_from,
        overrides=[(None, "model_path", args.model_path), (None, "manifest_path", args.tasks),
                   ("rollout", "group_size", args.group_size), ("train", "max_updates", args.max_updates),
                   ("train", "epochs", args.epochs)])
    if args.validate_only:
        from flex_gspo.metricx import ensure_metricx_ready
        from flex_gspo.rewards import RewardError
        try:
            ensure_metricx_ready(config)
            service_ready = True
        except RewardError:
            service_ready = False
        print(json.dumps({"valid": True, "tasks": len(tasks),
                          "quality_service_ready": service_ready,
                          "service_connectivity_verified": False,
                          "reward_calibration_acknowledged": config.get("reward", {}).get("calibration_acknowledged", False),
                          "warning": "configuration validation does not prove GPU memory, reward calibration or model/runtime compatibility"}))
        return 0
    if not config["reward"].get("calibration_acknowledged", False):
        if not args.allow_uncalibrated_reward:
            raise ValueError("Reward thresholds are illustrative, not calibrated. Score/check existing candidates first, "
                             "then set reward.calibration_acknowledged=true; for an engineering smoke test only, "
                             "pass --allow-uncalibrated-reward --max-updates 2.")
        logging.warning("UNCALIBRATED REWARD: engineering smoke test only, not a validated research run")
        config["allow_uncalibrated_reward"] = True
    from flex_gspo.trainer import run_training
    result = run_training(config, tasks)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
