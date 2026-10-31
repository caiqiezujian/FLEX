#!/usr/bin/env python3
"""Export one SFT PEFT adapter plus its original HF base as a standalone model.

Uses the ms-swift 3.12.1 export interface. No training dataset or MCore conversion
is needed. Preserves inputs and publishes the final directory only after checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

PROTOCOL = ["<|WAIT|>", "<|DIRECT|>", "<|FLEX|>"]
VOCAB_KEYS = ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight")


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def weight_inventory(directory, prefix):
    """Read safetensors shapes and check sharded indexes without loading weights."""
    from safetensors import safe_open

    directory = Path(directory)
    index_path = directory / f"{prefix}.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map: {index_path}")
        names = sorted(set(weight_map.values()))
        if any(not isinstance(name, str) or Path(name).name != name for name in names):
            raise ValueError(f"Shard names must be local filenames: {index_path}")
        files = [directory / name for name in names]
    elif (directory / f"{prefix}.safetensors").is_file():
        files = [directory / f"{prefix}.safetensors"]
    else:
        raise ValueError(f"No {prefix}.safetensors or its shard index in {directory}")
    inventory = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in inventory:
                    raise ValueError(f"Duplicate tensor across shards: {key}")
                inventory[key] = {"file": str(path), "shape": list(handle.get_slice(key).get_shape())}
                if weight_map is not None and weight_map.get(key) != path.name:
                    raise ValueError(f"Tensor/index disagreement: {key}")
    if weight_map is not None and set(weight_map) != set(inventory):
        raise ValueError(f"Index references missing tensors: {index_path}")
    if not inventory:
        raise ValueError(f"Empty model weights: {directory}")
    return inventory


def find_adapter_module(inventory, target):
    matches = []
    for key in inventory:
        normalized = key.replace(".modules_to_save.default", "").replace(".modules_to_save", "")
        if normalized.endswith(target):
            matches.append(key)
    if len(matches) != 1:
        raise ValueError(f"Expected one saved full module for {target}; found {matches}. "
                         "Use the PEFT checkpoint saved with modules_to_save word_embeddings output_layer.")
    return matches[0]


def protocol_ids(tokenizer):
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in PROTOCOL}
    for token, token_id in ids.items():
        if not isinstance(token_id, int) or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Missing protocol token: {token}")
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"Protocol token is split into multiple tokens: {token}")
        if token not in tokenizer.all_special_tokens:
            raise ValueError(f"Protocol token is not registered as special: {token}")
    if len(set(ids.values())) != len(PROTOCOL):
        raise ValueError("Protocol token IDs must be distinct")
    return ids


def preflight(base, adapter, tokens_file):
    from transformers import AutoTokenizer

    config = read_json(base / "config.json")
    if config.get("model_type") != "qwen3_omni_moe":
        raise ValueError("--base must be the original Hugging Face Qwen3-Omni model directory, not MCore")
    if config.get("quantization_config") or config.get("thinker_config", {}).get("quantization_config"):
        raise ValueError("Use the original unquantized HF base for this merge")
    tokens = tokens_file.read_text(encoding="utf-8-sig").split()
    if tokens != PROTOCOL:
        raise ValueError(f"Special-token file must contain these tokens in order: {PROTOCOL}")
    adapter_config = read_json(adapter / "adapter_config.json")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("Expected a PEFT LoRA adapter_config.json, not a merged/MCore checkpoint")
    training_args = read_json(adapter / "args.json")
    trained_tokens = training_args.get("new_special_tokens")
    if trained_tokens != tokens:
        raise ValueError("Checkpoint args.json new_special_tokens does not match the protocol token list. "
                         "Check the actual training token order; do not guess or edit it just to bypass this check.")
    # Explicitly rebuild the same token additions from the original base.
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    ids = protocol_ids(tokenizer)
    base_weights = weight_inventory(base, "model")
    adapter_weights = weight_inventory(adapter, "adapter_model")
    if not any("lora_A" in key for key in adapter_weights) or not any("lora_B" in key for key in adapter_weights):
        raise ValueError("Adapter has no LoRA A/B weights")
    expected_vocab = config["thinker_config"]["text_config"]["vocab_size"]
    if expected_vocab < len(tokenizer):
        expected_vocab = ((len(tokenizer) + 127) // 128) * 128
    modules = {}
    for target in VOCAB_KEYS:
        if target not in base_weights:
            raise ValueError(f"Original base is missing {target}")
        key = find_adapter_module(adapter_weights, target)
        shape = adapter_weights[key]["shape"]
        base_shape = base_weights[target]["shape"]
        if len(shape) != 2 or shape != [expected_vocab, base_shape[1]]:
            raise ValueError(f"Saved module {key} has shape {shape}; HF loader expects "
                             f"{[expected_vocab, base_shape[1]]}. Do not discard trained vocabulary rows.")
        if max(ids.values()) >= shape[0]:
            raise ValueError(f"Protocol IDs do not fit saved module {key}")
        modules[target] = {"key": key, **adapter_weights[key]}
    return {"protocol_token_ids": ids, "tokenizer_length": len(tokenizer),
            "expected_vocab_size": expected_vocab, "saved_modules": modules,
            "base_tensor_shapes": {key: value["shape"] for key, value in base_weights.items()},
            "training_base_recorded": training_args.get("model"),
            "training_args_tokens_verified": True}


def verify_export(output, expected):
    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(output), local_files_only=True, trust_remote_code=False)
    if protocol_ids(tokenizer) != expected["protocol_token_ids"] or len(tokenizer) != expected["tokenizer_length"]:
        raise ValueError("Exported tokenizer differs from the expected training protocol tokenizer")
    processor = AutoProcessor.from_pretrained(str(output), local_files_only=True, trust_remote_code=False)
    if protocol_ids(processor.tokenizer) != expected["protocol_token_ids"]:
        raise ValueError("Exported processor tokenizer protocol mismatch")
    config = read_json(output / "config.json")
    if config.get("model_type") != "qwen3_omni_moe":
        raise ValueError("Unexpected exported model_type")
    if config["thinker_config"]["text_config"]["vocab_size"] != expected["expected_vocab_size"]:
        raise ValueError("Exported config vocabulary size does not match trained modules")
    inventory = weight_inventory(output, "model")
    if any("lora_A" in key or "lora_B" in key or "modules_to_save" in key for key in inventory):
        raise ValueError("Export still contains unmerged adapter tensor names")
    # ENABLE_AUDIO_OUTPUT=0 exports the Thinker for speech-to-text translation.
    # Its text, audio and vision modules must all still be present.
    for key, shape in expected["base_tensor_shapes"].items():
        if not key.startswith("thinker."):
            continue
        wanted = expected["saved_modules"][key]["shape"] if key in VOCAB_KEYS else shape
        if key not in inventory or inventory[key]["shape"] != wanted:
            raise ValueError(f"Export missing/mismatched Thinker tensor: {key}")
    checked = []
    for target, saved in expected["saved_modules"].items():
        exported = inventory[target]
        with safe_open(saved["file"], framework="pt", device="cpu") as src, \
                safe_open(exported["file"], framework="pt", device="cpu") as dst:
            for token_id in expected["protocol_token_ids"].values():
                actual = dst.get_slice(target)[token_id:token_id + 1]
                reference = src.get_slice(saved["key"])[token_id:token_id + 1].to(actual.dtype)
                if not torch.isfinite(actual).all() or not torch.equal(reference, actual):
                    raise ValueError(f"Saved protocol row was not preserved: {target}, token_id={token_id}")
        checked.append(target)
    return {"protocol_token_ids": expected["protocol_token_ids"], "tokenizer_length": len(tokenizer),
            "vocab_size": expected["expected_vocab_size"], "tensor_count": len(inventory),
            "trained_protocol_rows_preserved_in": checked, "processor_load_verified": True,
            "inference_verified": False}


def export_command(args, staging):
    return [sys.executable, "-u", "-m", "swift.cli.export",
            "--model", str(args.base), "--adapters", str(args.adapter),
            "--model_type", "qwen3_omni", "--template", "qwen3_omni",
            "--load_args", "false", "--train_type", "lora",
            "--new_special_tokens", str(args.special_tokens), "--merge_lora", "true",
            "--torch_dtype", "bfloat16", "--device_map", args.device_map,
            "--safe_serialization", "true", "--max_shard_size", "5GB",
            "--output_dir", str(staging)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Original unquantized HF Qwen3-Omni directory (NOT MCore)")
    parser.add_argument("adapter", type=Path, help="One checkpoint-N with adapter_config.json and adapter weights")
    parser.add_argument("output", type=Path, help="New directory for the merged full model")
    parser.add_argument("--gpu-ids", default="0", help="Free GPU IDs for this single-process export, e.g. 0 or 0,1")
    parser.add_argument("--device-map", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--special-tokens", type=Path, default=Path(__file__).with_name("duplex_protocol_tokens.txt"))
    parser.add_argument("--check-only", action="store_true", help="Check inputs, tokens and shapes without loading full weights")
    args = parser.parse_args(argv)
    staging = None
    try:
        for name in ("base", "adapter", "special_tokens"):
            setattr(args, name, getattr(args, name).resolve(strict=True))
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(f"Output must not already exist: {args.output}")
        for root in (args.base, args.adapter):
            if root == args.output or root in args.output.parents:
                raise ValueError("Output must not be inside the original base/adapter directory")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids if args.device_map == "auto" else ""
        os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        version = importlib.metadata.version("ms-swift")
        print(f"[merge] Python: {sys.executable}\n[merge] ms-swift: {version}", flush=True)
        if version != "3.12.1":
            print("[merge] NOTE: this recipe was source-audited for ms-swift 3.12.1; use your SFT environment.", flush=True)
        print("[merge] Checking local HF base, adapter tensors and protocol tokens...", flush=True)
        expected = preflight(args.base, args.adapter, args.special_tokens)
        print(json.dumps({"input_checks_passed": True, "protocol_token_ids": expected["protocol_token_ids"],
                          "vocab_size": expected["expected_vocab_size"],
                          "training_base_recorded": expected["training_base_recorded"]}, ensure_ascii=False), flush=True)
        if args.check_only:
            print("[merge] Check-only complete; no model was merged or written.", flush=True)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.output.with_name(args.output.name + ".incomplete-" + uuid.uuid4().hex[:8])
        command = export_command(args, staging)
        print("[merge] " + shlex.join(command), flush=True)
        started = time.monotonic()
        # Run as one process; device_map can distribute HF weights across visible GPUs.
        child_env = dict(os.environ)
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "NPROC_PER_NODE", "NNODES"):
            child_env.pop(key, None)
        subprocess.run(command, env=child_env, check=True)
        print("[merge] Export finished; verifying saved full weights and trained protocol rows...", flush=True)
        verified = verify_export(staging, expected)
        report = {"created_utc": datetime.now(timezone.utc).isoformat(), "base": str(args.base),
                  "adapter": str(args.adapter), "output": str(args.output), "ms_swift": version,
                  "elapsed_seconds": round(time.monotonic() - started, 2), **verified}
        (staging / "flex_merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.output.exists():
            raise FileExistsError(f"Output appeared during export; leaving staging intact: {args.output}")
        staging.rename(args.output)
        print(f"[merge] SUCCESS: {args.output}", flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"[merge] Interrupted. Partial output, if any: {staging}", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"[merge] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if staging is not None:
            print(f"[merge] Partial/unverified export retained at: {staging}; do not use it for inference.", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
