#!/usr/bin/env python3
"""Small GPU forward/backward smoke checks for the optional attention fallback.

No model checkpoint, training data, downloads, optimizer or distributed job.
Passing this is not certification of the full SFT/MoE/checkpoint-save path.
"""
import argparse
import importlib.metadata
import os
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", default="0", help="One idle physical GPU, e.g. 6")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    os.environ["NVTE_MASKED_SOFTMAX_FUSION"] = "0"
    try:
        print(f"[attention-check] Python={sys.executable}; physical GPU={args.gpu_id}", flush=True)
        for package in ("torch", "transformers", "transformer-engine", "ms-swift", "flash-attn"):
            print(f"[attention-check] {package}={importlib.metadata.version(package)}", flush=True)
        import torch
        from duplex_attention_compat import install_attention_compat
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig, Qwen3OmniMoeVisionEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioAttention, Qwen3OmniMoeVisionAttention,
        )

        device, dtype = "cuda:0", torch.bfloat16
        torch.cuda.set_device(0)
        print(f"[attention-check] GPU={torch.cuda.get_device_name(0)}; torch CUDA={torch.version.cuda}", flush=True)
        install_attention_compat()
        cu = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)

        def check_grad(label, output, source):
            output.float().square().mean().backward()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all() or source.grad is None or not torch.isfinite(source.grad).all():
                raise RuntimeError(f"{label}: nonfinite output/gradient or missing gradient")
            print(f"[attention-check] PASS: {label} forward/backward", flush=True)

        print("[attention-check] BEGIN: Omni audio attention", flush=True)
        audio_config = Qwen3OmniMoeAudioEncoderConfig(d_model=128, encoder_attention_heads=2, attention_dropout=0.0)
        audio_config._attn_implementation = "sdpa"
        audio = Qwen3OmniMoeAudioAttention(audio_config).to(device=device, dtype=dtype)
        x = torch.randn(12, 128, device=device, dtype=dtype, requires_grad=True)
        y = audio(x, cu_seqlens=cu)
        with torch.no_grad():
            changed = x.detach().clone()
            changed[5:] += 10
            torch.testing.assert_close(y[:5], audio(changed, cu_seqlens=cu)[:5], rtol=0, atol=0)
        check_grad("segmented Omni audio (future segment isolated)", y, x)

        print("[attention-check] BEGIN: Omni vision attention", flush=True)
        vision_config = Qwen3OmniMoeVisionEncoderConfig(hidden_size=128, num_heads=2)
        vision_config._attn_implementation = "sdpa"
        vision = Qwen3OmniMoeVisionAttention(vision_config).to(device=device, dtype=dtype)
        x = torch.randn(12, 128, device=device, dtype=dtype, requires_grad=True)
        pos = (torch.ones(12, 64, device=device, dtype=dtype), torch.zeros(12, 64, device=device, dtype=dtype))
        y = vision(x, cu_seqlens=cu, position_embeddings=pos)
        check_grad("Omni vision math SDPA", y, x)

        print("[attention-check] BEGIN: Transformer Engine unfused attention", flush=True)
        from transformer_engine.pytorch.attention import DotProductAttention
        attention = DotProductAttention(num_attention_heads=2, kv_channels=64,
                                        attention_dropout=0.0, qkv_format="sbhd", attn_mask_type="causal")
        q, k, v = [torch.randn(8, 1, 2, 64, device=device, dtype=dtype, requires_grad=True) for _ in range(3)]
        result = attention(q, k, v)
        check_grad("Transformer Engine unfused", result, q)
        print("[attention-check] ALL PASSED. Next: a short real SFT run; full training/save not verified.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("[attention-check] FAILED: preserve this traceback; do not start full SFT yet.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
