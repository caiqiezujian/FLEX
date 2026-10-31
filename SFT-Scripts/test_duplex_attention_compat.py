"""CPU semantic checks for the optional segmented fallback, not GPU validation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig, Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention, Qwen3OmniMoeVisionAttention,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duplex_attention_compat import install_attention_compat, _patch_qwen_loader


@pytest.fixture
def attention_models(monkeypatch):
    original_audio = Qwen3OmniMoeAudioAttention.forward
    original_vision = Qwen3OmniMoeVisionAttention.forward
    # Let monkeypatch restore both classes after each test, including failures.
    monkeypatch.setattr(Qwen3OmniMoeAudioAttention, "forward", original_audio)
    monkeypatch.setattr(Qwen3OmniMoeVisionAttention, "forward", original_vision)
    install_attention_compat()
    audio_config = Qwen3OmniMoeAudioEncoderConfig(d_model=16, encoder_attention_heads=2, attention_dropout=0.0)
    audio_config._attn_implementation = "sdpa"
    vision_config = Qwen3OmniMoeVisionEncoderConfig(hidden_size=16, num_heads=2)
    vision_config._attn_implementation = "sdpa"
    return Qwen3OmniMoeAudioAttention(audio_config), Qwen3OmniMoeVisionAttention(vision_config), original_audio


def test_audio_matches_separate_calls_and_future_has_zero_influence(attention_models):
    audio, _, original = attention_models
    x = torch.randn(7, 16, requires_grad=True)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    keys = set(audio.state_dict())
    result = audio(x, cu)
    expected = torch.cat([original(audio, x[:3], torch.tensor([0, 3], dtype=torch.int32)),
                          original(audio, x[3:], torch.tensor([0, 4], dtype=torch.int32))])
    torch.testing.assert_close(result, expected)
    changed = x.detach().clone()
    changed[3:] += 100
    torch.testing.assert_close(result[:3], audio(changed, cu)[:3], rtol=0, atol=0)
    result[:3].square().sum().backward()
    assert torch.count_nonzero(x.grad[3:]) == 0
    assert torch.count_nonzero(x.grad[:3]) > 0
    assert set(audio.state_dict()) == keys


def test_audio_respects_masks_and_preserves_gradients(attention_models):
    audio, _, original = attention_models
    x = torch.randn(7, 16, requires_grad=True)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    mask = torch.zeros(1, 1, 7, 7)
    mask[..., 0] = float("-inf")
    result = audio(x, cu, attention_mask=mask)
    expected = torch.cat([original(audio, x[:3], torch.tensor([0, 3], dtype=torch.int32), mask[..., :3, :3]),
                          original(audio, x[3:], torch.tensor([0, 4], dtype=torch.int32), mask[..., 3:, 3:])])
    torch.testing.assert_close(result, expected)
    actual_grad = torch.autograd.grad(result.square().sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("cu", [None, [1, 7], [0, 0, 7], [0, 8], [0, 4, 3, 7]])
def test_invalid_boundaries_fail_without_silent_cross_segment_attention(attention_models, cu):
    audio, _, _ = attention_models
    value = None if cu is None else torch.tensor(cu, dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens"):
        audio(torch.randn(7, 16), value)


def test_bad_backend_is_rejected_before_flash_kernel(attention_models):
    audio, _, _ = attention_models
    audio.config._attn_implementation = "flash_attention_2"
    with pytest.raises(RuntimeError, match="requires HF"):
        audio(torch.randn(7, 16), torch.tensor([0, 7], dtype=torch.int32))


def test_vision_segments_and_gradients(attention_models):
    _, vision, _ = attention_models
    x = torch.randn(7, 16, requires_grad=True)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    pos = (torch.ones(7, 8), torch.zeros(7, 8))
    output = vision(x, cu, position_embeddings=pos)
    changed = x.detach().clone()
    changed[3:] += 100
    torch.testing.assert_close(output[:3], vision(changed, cu, position_embeddings=pos)[:3], rtol=0, atol=0)
    output[:3].square().sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad[3:]) == 0


def test_install_is_idempotent_and_restores_sdpa_flags_after_forward(attention_models):
    audio, _, _ = attention_models
    first = type(audio).forward
    install_attention_compat()
    assert first is type(audio).forward
    before = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
              torch.backends.cuda.math_sdp_enabled())
    audio(torch.randn(3, 16), torch.tensor([0, 3], dtype=torch.int32))
    after = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
             torch.backends.cuda.math_sdp_enabled())
    assert before == after


def test_loader_forces_only_omni_sdpa_even_when_megatron_drops_option():
    calls = []
    module = SimpleNamespace(get_model_tokenizer_with_flash_attn=lambda *a, **kw: calls.append(kw))
    _patch_qwen_loader(module)
    wrapped = module.get_model_tokenizer_with_flash_attn
    _patch_qwen_loader(module)
    assert wrapped is module.get_model_tokenizer_with_flash_attn
    module.get_model_tokenizer_with_flash_attn(model_config=SimpleNamespace(model_type="qwen3_omni_moe"))
    module.get_model_tokenizer_with_flash_attn(model_config=SimpleNamespace(model_type="qwen2"), attn_impl="flash_attn")
    assert calls[0]["attn_impl"] == "sdpa"
    assert calls[1]["attn_impl"] == "flash_attn"
