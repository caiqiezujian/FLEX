"""Opt-in attention fallback for the existing SWIFT 3.12.1 Omni SFT job.

Uses native Transformers projections and PyTorch math SDPA. Audio segments are
processed separately: older Omni SDPA paths can otherwise ignore cu_seqlens.
No parameters or checkpoint keys are added/removed by these forward wrappers.
"""
from functools import wraps
import inspect


def _patch_qwen_loader(qwen_module):
    """SWIFT 3.12.1 drops attn_impl for non-flash MCore backends; fix this alias.

    Patch only the Qwen HF loader module, never import Megatron in the dataset
    preflight process. Other Qwen model types keep their existing settings.
    """
    original = qwen_module.get_model_tokenizer_with_flash_attn
    if getattr(original, "_flex_omni_sdpa_loader", False):
        return

    @wraps(original)
    def load(*args, **kwargs):
        config = kwargs.get("model_config")
        if getattr(config, "model_type", None) == "qwen3_omni_moe":
            kwargs["attn_impl"] = "sdpa"
            print("[FLEX attention] Loading Omni HF towers with attn_impl=sdpa", flush=True)
        return original(*args, **kwargs)

    load._flex_omni_sdpa_loader = True
    qwen_module.get_model_tokenizer_with_flash_attn = load


def install_swift_loader_compat():
    from swift.llm.model.model import qwen
    _patch_qwen_loader(qwen)


def _require_sdpa(module):
    mode = getattr(module.config, "_attn_implementation", None)
    if mode != "sdpa":
        raise RuntimeError(f"FLEX compatibility mode requires HF attn_impl=sdpa, got {mode!r}. "
                           "Launch with --compat-old-driver; do not override --attn_impl.")


def _audio_boundaries(hidden_states, cu_seqlens):
    import torch

    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("Expected nonempty flattened Omni audio [tokens, channels]")
    if (cu_seqlens is None or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2
            or cu_seqlens.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Audio attention requires integer cu_seqlens; cannot ignore segment boundaries")
    boundaries = cu_seqlens.detach().cpu().tolist()
    if (boundaries[0] != 0 or boundaries[-1] != hidden_states.shape[0]
            or any(b <= a for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("Audio cu_seqlens must strictly increase from 0 to the token count")
    return boundaries


def _slice_audio_mask(mask, start, end, total):
    if mask is None:
        return None
    if mask.ndim < 2 or mask.shape[-1] != total or mask.shape[-2] not in (1, total):
        raise ValueError("Unsupported audio attention mask shape; refusing to drop the mask")
    if mask.shape[-2] == 1:
        return mask[..., :, start:end]
    return mask[..., start:end, start:end]


def install_attention_compat():
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioAttention, Qwen3OmniMoeVisionAttention,
    )

    audio_cls, vision_cls = Qwen3OmniMoeAudioAttention, Qwen3OmniMoeVisionAttention
    if getattr(audio_cls.forward, "_flex_math_segments", False):
        if not getattr(vision_cls.forward, "_flex_math_segments", False):
            raise RuntimeError("Incomplete FLEX attention patch; restart in a fresh process")
        return
    original_audio, original_vision = audio_cls.forward, vision_cls.forward
    for original in (original_audio, original_vision):
        if not {"hidden_states", "cu_seqlens"}.issubset(inspect.signature(original).parameters):
            raise RuntimeError("Installed Omni attention API differs from the audited Transformers 4.57.x path")

    @wraps(original_audio)
    def audio_forward(self, hidden_states, cu_seqlens=None, attention_mask=None, **kwargs):
        _require_sdpa(self)
        boundaries = _audio_boundaries(hidden_states, cu_seqlens)
        outputs = []
        with sdpa_kernel(SDPBackend.MATH):
            for start, end in zip(boundaries, boundaries[1:]):
                local_cu = cu_seqlens.new_tensor([0, end - start])
                mask = _slice_audio_mask(attention_mask, start, end, hidden_states.shape[0])
                outputs.append(original_audio(self, hidden_states[start:end], cu_seqlens=local_cu,
                                              attention_mask=mask, **kwargs))
        return torch.cat(outputs, dim=0)

    @wraps(original_vision)
    def vision_forward(self, *args, **kwargs):
        _require_sdpa(self)
        # The native VisionAttention SDPA path already splits by cu_seqlens.
        with sdpa_kernel(SDPBackend.MATH):
            return original_vision(self, *args, **kwargs)

    audio_forward._flex_math_segments = True
    vision_forward._flex_math_segments = True
    audio_cls.forward = audio_forward
    vision_cls.forward = vision_forward
    print("[FLEX attention] compatibility enabled: Omni audio=segmented math SDPA; "
          "vision=native segmented math SDPA. Megatron must use unfused with packing/padding_free=false.", flush=True)
