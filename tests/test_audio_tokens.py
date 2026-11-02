"""No weights or vLLM import: exercise guarded count reuse and exact fallbacks."""

from types import SimpleNamespace
import sys
import weakref

import numpy as np
import pytest

from flex_web.audio_tokens import AudioTokenCounter


def native_type(module, name, **attrs):
    cls = type(name, (), {'__module__': module})
    instance = cls()
    instance.__dict__.update(attrs)
    return instance


@pytest.fixture
def native_counter(monkeypatch):
    """Only exercise the guard contract; this is not a GPU/processor benchmark."""
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='4.57.1'))
    feature = native_type('transformers.models.whisper.feature_extraction_whisper',
                          'WhisperFeatureExtractor', sampling_rate=16000, hop_length=160,
                          n_fft=400, n_samples=480000, chunk_length=30,
                          nb_max_frames=3000, feature_size=128, padding_side='right',
                          padding_value=0.0, return_attention_mask=False, dither=0.0)
    hf = native_type('transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe',
                     'Qwen3OmniMoeProcessor', feature_extractor=feature,
                     audio_token='<|audio_pad|>')
    mm = native_type('vllm.model_executor.models.qwen3_omni_moe_thinker',
                     'Qwen3OmniMoeThinkerMultiModalProcessor',
                     info=SimpleNamespace(get_hf_processor=lambda: hf))
    preprocessor = SimpleNamespace(
        _get_mm_processor=lambda: mm,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen3_omni_moe'),
                                     mm_processor_kwargs=None))
    return preprocessor, hf, feature


def build(native_counter, max_entries=64, version='0.11.1'):
    return AudioTokenCounter.for_preprocessor(native_counter[0],
                                              vllm_version=version,
                                              max_entries=max_entries)


def test_each_length_is_measured_then_reused_across_signal_values(native_counter):
    cache = build(native_counter)
    calls = []

    def exact(audio):
        calls.append(len(audio))
        # Deliberately unusual counts: cache must use the exact counter, not a
        # hard-coded 13 tokens/second estimate or a rounded duration key.
        return {16000: 12, 32000: 25, 48000: 38, 16001: 13}[len(audio)]

    for size, value in ((16000, 12), (32000, 25), (48000, 38), (16001, 13)):
        assert cache.count(np.zeros(size, dtype=np.float32), exact) == (value, False)
        assert cache.count(np.ones(size, dtype=np.float32), exact) == (value, True)
    assert calls == [16000, 32000, 48000, 16001]


def test_lru_eviction_clear_and_no_waveform_retention(native_counter):
    cache = build(native_counter, max_entries=2)
    calls = []

    def exact(audio):
        calls.append(len(audio))
        return len(audio)

    audio = np.zeros(16000, dtype=np.float32)
    ref = weakref.ref(audio)
    cache.count(audio, exact)
    del audio
    assert ref() is None
    cache.count(np.zeros(32000, dtype=np.float32), exact)
    cache.count(np.zeros(16000, dtype=np.float32), exact)  # make 32000 oldest
    cache.count(np.zeros(48000, dtype=np.float32), exact)
    assert len(cache._counts) == 2
    assert cache.count(np.zeros(32000, dtype=np.float32), exact)[1] is False
    cache.clear()
    assert not cache._counts
    assert cache.mode == 'length_lru'
    assert cache.count(np.zeros(32000, dtype=np.float32), exact)[1] is False


@pytest.mark.parametrize('version', ['0.11.0', '0.11.1.dev1', '0.19.1'])
def test_unknown_vllm_keeps_exact_counting(native_counter, version):
    cache = build(native_counter, version=version)
    calls = []
    for _ in range(2):
        assert cache.count(np.zeros(16000, dtype=np.float32),
                           lambda a: calls.append(len(a)) or 12) == (12, False)
    assert len(calls) == 2 and cache.mode == 'exact'


@pytest.mark.parametrize('version', ['4.57.0', '4.57.2', '4.58.0', None])
def test_unverified_transformers_keeps_exact(native_counter, monkeypatch, version):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__=version))
    assert build(native_counter).mode == 'exact'


@pytest.mark.parametrize('change', ['feature_type', 'model_type', 'overrides',
                                  'rate', 'hop', 'placeholder', 'custom_mm'])
def test_unverified_configuration_disables_memoization(native_counter, change):
    preprocessor, hf, feature = native_counter
    if change == 'feature_type':
        hf.feature_extractor = SimpleNamespace(**feature.__dict__)
    elif change == 'model_type':
        preprocessor.model_config.hf_config.model_type = 'qwen2_5_omni'
    elif change == 'overrides':
        preprocessor.model_config.mm_processor_kwargs = {'truncation': True}
    elif change == 'rate':
        feature.sampling_rate = 8000
    elif change == 'hop':
        feature.hop_length = 320
    elif change == 'placeholder':
        hf.audio_token = '<other>'
    elif change == 'custom_mm':
        preprocessor._get_mm_processor = lambda: SimpleNamespace(info=SimpleNamespace(
            get_hf_processor=lambda: hf))
    assert build(native_counter).mode == 'exact'


def test_configuration_change_and_explicit_disable_are_permanent(native_counter):
    cache = build(native_counter)
    audio = np.zeros(16000, dtype=np.float32)
    assert cache.count(audio, lambda a: 12) == (12, False)
    native_counter[2].n_samples = 960000
    assert cache.count(audio, lambda a: 20) == (20, False)
    native_counter[2].n_samples = 480000
    assert cache.count(audio, lambda a: 21) == (21, False)
    assert cache.mode == 'exact' and not cache._counts
    fresh = build(native_counter)
    fresh.count(audio, lambda a: 12)
    fresh.disable('runtime expanded prompt mismatch')
    assert fresh.count(audio, lambda a: 18) == (18, False)
    assert fresh.reason == 'runtime expanded prompt mismatch'


def test_disabled_and_unavailable_preprocessor_are_safe(native_counter):
    assert build(native_counter, max_entries=0).mode == 'exact'
    missing = AudioTokenCounter.for_preprocessor(object(), vllm_version='0.11.1')
    assert missing.mode == 'exact'


@pytest.mark.parametrize('audio', [np.zeros(10, dtype=np.float64),
                                  np.zeros((1, 10), dtype=np.float32),
                                  np.zeros(0, dtype=np.float32), [0.0] * 10])
def test_unexpected_audio_layout_bypasses_cache(native_counter, audio):
    cache = build(native_counter)
    calls = []
    for _ in range(2):
        assert cache.count(audio, lambda a: calls.append(1) or 3) == (3, False)
    assert len(calls) == 2 and not cache._counts


def test_counter_failure_is_not_cached(native_counter):
    cache = build(native_counter)
    audio = np.zeros(16000, dtype=np.float32)

    def failing(audio):
        raise RuntimeError('bad preprocessing')

    with pytest.raises(RuntimeError, match='bad preprocessing'):
        cache.count(audio, failing)
    assert cache.count(audio, lambda a: 12) == (12, False)


@pytest.mark.parametrize('value', [-1, 1.5, True, '12'])
def test_invalid_exact_result_not_cached(native_counter, value):
    cache = build(native_counter)
    with pytest.raises(ValueError, match='non-negative integer'):
        cache.count(np.zeros(16000, dtype=np.float32), lambda a: value)
    assert not cache._counts


@pytest.mark.parametrize('size', [-1, 1025, True, 1.5])
def test_invalid_size_rejected(size):
    with pytest.raises(ValueError, match='0..1024'):
        AudioTokenCounter(size)


def test_native_whisper_length_invariance_without_model_download():
    """Optional CPU check against the actual audited HF implementation.

    A processor is constructed from its class only: no checkpoint, tokenizer,
    network download, vLLM or GPU is involved.  Missing/incompatible optional
    dependencies are reported as a skip rather than changed by the test.
    """
    try:
        import transformers
        if transformers.__version__ != '4.57.1':
            pytest.skip('native length check requires Transformers 4.57.1')
        from transformers import WhisperFeatureExtractor
        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
            _get_feat_extract_output_lengths,
        )
    except ImportError as exc:
        pytest.skip(f'optional Transformers processor unavailable: {exc}')

    feature = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                      hop_length=160)
    rng = np.random.default_rng(24)
    for size in (16000, 32000, 48000, 15999, 16001, 7999):
        padding = (-size) % feature.hop_length
        silence = np.pad(np.zeros(size, dtype=np.float32), (0, padding))
        noise = np.pad(rng.uniform(-0.5, 0.5, size).astype(np.float32), (0, padding))
        # vLLM 0.11.1 pre-pads each clip before passing it to Whisper.  Include
        # a longer unrelated item to verify padding/batching does not change
        # valid-frame or audio-placeholder counts for the original clip.
        longer = np.zeros(64000, dtype=np.float32)
        single = feature([silence], sampling_rate=16000, padding=True,
                         truncation=False, return_attention_mask=True,
                         return_tensors='np')
        batch = feature([silence, noise, longer], sampling_rate=16000, padding=True,
                        truncation=False, return_attention_mask=True,
                        return_tensors='np')
        single_frames = single['attention_mask'].sum(-1)
        batch_frames = batch['attention_mask'].sum(-1)
        expected_frames = (size + padding) // 160
        assert int(single_frames[0]) == expected_frames
        assert batch_frames[:2].tolist() == [expected_frames, expected_frames]
        single_count = int(_get_feat_extract_output_lengths(single_frames)[0])
        batch_counts = _get_feat_extract_output_lengths(batch_frames)
        assert batch_counts[:2].tolist() == [single_count, single_count]
