#!/usr/bin/env python3
"""Merge only the new RL LoRA into its exact full SFT base for existing inference."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import tempfile

from flex_gspo.runtime import Progress, atomic_json, read_resume_metadata


def validate_export_loading(loading, include_talker=False):
    """Only intentionally omitted output heads may appear as unused weights."""
    if any(loading.get(name) for name in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Export cannot initialize missing model weights: {loading}")
    allowed = () if include_talker else ("talker.", "code2wav.", "code_predictor.")
    unexpected = [key for key in loading.get("unexpected_keys", [])
                  if not allowed or not key.startswith(allowed)]
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint tensors during export: {unexpected[:20]}")


def validate_paths(base_model, checkpoint, output):
    base, checkpoint, output = (Path(p).resolve() for p in (base_model, checkpoint, output))
    metadata = read_resume_metadata(checkpoint)
    expected = Path(metadata["config"]["model_path"]).resolve()
    if base != expected:
        raise ValueError("Use the exact merged SFT base recorded in checkpoint.json, not the original pretrained model")
    from flex_gspo.runtime import directory_inventory
    if directory_inventory(base) != metadata["identity"]["details"]["models"]["actor"]:
        raise ValueError("SFT base inventory changed after training; export refused")
    if output.exists():
        raise FileExistsError("export output must not exist; no overwrite")
    if base == output or base in output.parents or checkpoint == output or checkpoint in output.parents:
        raise ValueError("export output must be outside input model/checkpoint directories")
    return base, checkpoint, output, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True, help="completed checkpoint-N directory, not adapter/ alone")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", help="cpu (requires substantial RAM) or an explicit cuda device")
    parser.add_argument("--include-talker", action="store_true",
                        help="preserve original speech-output modules; requires complete original weights")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base, checkpoint, output, metadata = validate_paths(args.base_model, args.checkpoint, args.output_dir)
    if args.validate_only:
        print(json.dumps({"valid": True, "step": metadata["step"], "weights_loaded": False}))
        return 0
    import torch
    import transformers
    from peft import PeftModel
    from transformers import (Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration,
                              Qwen3OmniMoeProcessor)
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Use the isolated training environment (transformers==4.57.1)")
    with Progress() as progress:
        progress.status("loading local SFT checkpoint for export; CPU RAM can exceed model file size")
        # Keep the outer Omni config and thinker.* prefixes expected by the
        # existing vLLM loader; a bare Thinker.save_pretrained is not equivalent.
        outer_config = Qwen3OmniMoeConfig.from_pretrained(str(base), local_files_only=True)
        outer_config.enable_audio_output = bool(args.include_talker)
        model, loading = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(base), config=outer_config, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": args.device}, low_cpu_mem_usage=True, attn_implementation="sdpa",
            output_loading_info=True)
        validate_export_loading(loading, args.include_talker)
        processor = Qwen3OmniMoeProcessor.from_pretrained(str(base), local_files_only=True)
        for token, expected in (("<|WAIT|>", 151676), ("<|DIRECT|>", 151677), ("<|FLEX|>", 151678)):
            if processor.tokenizer.encode(token, add_special_tokens=False) != [expected]:
                raise ValueError("SFT protocol tokenizer mismatch; export never repairs/re-registers tokens")
        progress.status("merging trained Thinker attention LoRA")
        adapted = PeftModel.from_pretrained(model.thinker, str(checkpoint / "adapter"),
                                            is_trainable=False, local_files_only=True)
        model.thinker = adapted.merge_and_unload(safe_merge=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".flex-export-", dir=str(output.parent)))
        progress.status(f"saving merged checkpoint to temporary directory {temporary}")
        model.save_pretrained(str(temporary), safe_serialization=True, max_shard_size="5GB")
        processor.save_pretrained(str(temporary))
        atomic_json(temporary / "qalco_export.json",
                    {"base_model": str(base), "checkpoint": str(checkpoint), "step": metadata["step"],
                     "training_identity": metadata["identity"]["sha256"],
                     "includes_talker": bool(args.include_talker),
                     "outer_omni_config_preserved": True,
                     "gpu_inference_verified": False})
        if output.exists():
            raise FileExistsError(f"output appeared during export; completed files retained at {temporary}")
        os.rename(temporary, output)
    print(json.dumps({"exported": str(output), "inference_validation_required": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
