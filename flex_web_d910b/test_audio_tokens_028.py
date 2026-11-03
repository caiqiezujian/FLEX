"""Contract checks for 0.28's guarded counter; no real processor or NPU used."""

from types import SimpleNamespace
import sys
import weakref

import numpy as np
import pytest

from flex_web_d910b.audio_tokens import AudioTokenCounter


def native_type(module, name, **attrs):
    value = type(name, (), {'__module__': module})()
    value.__dict__.update(attrs)
    return value


def fake_tokenizer(token_id=151646):
    tokenizer = SimpleNamespace(token_id=token_id, init_kwargs={})
    tokenizer.convert_tokens_to_ids = lambda _: tokenizer.token_id
    tokenizer.encode = lambda text, add_special_tokens=False: [tokenizer.token_id]
    return tokenizer


@pytest.fixture(params=['5.5.3', '5.6.2'])
def native028(monkeypatch, request):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__=request.param))
    feature = native_type('transformers.models.whisper.feature_extraction_whisper',
                          'WhisperFeatureExtractor', sampling_rate=16000, hop_length=160,
                          n_fft=400, n_samples=480000, chunk_length=30,
                          nb_max_frames=3000, feature_size=128, padding_side='right',
                          padding_value=0.0, return_attention_mask=False, dither=0.0)
    tokenizer = fake_tokenizer()
    hf = native_type('transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe',
                     'Qwen3OmniMoeProcessor', feature_extractor=feature,
                     audio_token='<|audio_pad|>', tokenizer=tokenizer)
    mm = native_type('vllm.model_executor.models.qwen3_omni_moe_thinker',
                     'Qwen3OmniMoeThinkerMultiModalProcessor',
                     info=SimpleNamespace(get_hf_processor=lambda: hf))
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen3_omni_moe'),
                             mm_processor_kwargs=None,
                             multimodal_config=SimpleNamespace(mm_processor_kwargs=None))
    prep = SimpleNamespace(renderer=SimpleNamespace(mm_processor=mm, tokenizer=tokenizer), model_config=config)
    return prep, hf, feature


@pytest.mark.parametrize('key', ['sampling_rate', 'raw_speech', 'padding', 'max_length',
                                 'truncation', 'pad_to_multiple_of',
                                 'return_attention_mask', 'return_tensors'])
def test_tokenizer_audio_kwarg_override_forces_exact(native028, key):
    prep, hf, _ = native028
    hf.tokenizer.init_kwargs[key] = True
    counter = AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64)
    assert counter.mode == 'exact'


def test_tokenizer_audio_kwarg_mutation_disables_existing_cache(native028):
    prep, hf, _ = native028
    counter = AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64)
    audio = np.zeros(16000, dtype=np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    hf.tokenizer.init_kwargs['truncation'] = True
    assert counter.count(audio, lambda _: 17) == (17, False)
    assert counter.mode == 'exact' and not counter._counts


def test_custom_feature_audio_kwarg_override_forces_exact(native028):
    prep, hf, feature = native028
    feature.valid_kwargs = type('Extra', (), {'__annotations__': {'future_audio_flag': bool}})
    hf.tokenizer.init_kwargs['future_audio_flag'] = True
    assert AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64).mode == 'exact'


def test_custom_audio_argument_schema_forces_exact_even_without_override(native028):
    prep, _, feature = native028
    feature.valid_kwargs = type('Extra', (), {'__annotations__': {'future_audio_flag': bool}})
    assert AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64).mode == 'exact'


def test_tokenizer_unrelated_config_preserves_cache(native028):
    prep, hf, _ = native028
    hf.tokenizer.init_kwargs['model_max_length'] = 8192
    counter = AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64)
    assert counter.mode == 'length_lru'


def test_missing_tokenizer_init_kwargs_forces_exact(native028):
    prep, hf, _ = native028
    del hf.tokenizer.init_kwargs
    assert AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64).mode == 'exact'


def test_audio_kwarg_schema_change_disables_existing_cache(native028):
    prep, hf, feature = native028
    hf.tokenizer.init_kwargs['future_audio_flag'] = True
    counter = AudioTokenCounter.for_preprocessor(prep, vllm_version='0.28.0', max_entries=64)
    audio = np.zeros(16000, dtype=np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    feature.valid_kwargs = type('Extra', (), {'__annotations__': {'future_audio_flag': bool}})
    assert counter.count(audio, lambda _: 17) == (17, False)
    assert counter.mode == 'exact' and not counter._counts


def build(native028, size=64, version='0.28.0'):
    return AudioTokenCounter.for_preprocessor(native028[0], vllm_version=version, max_entries=size)


def test_same_exact_sample_length_reuses_count_not_audio(native028):
    counter = build(native028)
    assert counter.mode == 'length_lru'
    calls = []

    def measure(audio):
        calls.append(len(audio))
        return len(audio) // 1234  # Deliberately not a tokens-per-second guess.

    rng = np.random.default_rng(28)
    for length in (7999, 8000, 8001, 15999, 16000, 16001, 32000, 48000):
        silence = np.zeros(length, dtype=np.float32)
        noise = rng.uniform(-1, 1, size=length).astype(np.float32)
        ref = weakref.ref(silence)
        expected = length // 1234
        assert counter.count(silence, measure) == (expected, False)
        del silence
        assert ref() is None
        assert counter.count(noise, measure) == (expected, True)
    assert calls == [7999, 8000, 8001, 15999, 16000, 16001, 32000, 48000]


def test_lru_and_explicit_disable(native028):
    counter = build(native028, size=2)
    measure = lambda a: len(a)
    chunk = lambda n: np.zeros(n, np.float32)
    assert counter.count(chunk(16000), measure) == (16000, False)
    assert counter.count(chunk(32000), measure) == (32000, False)
    assert counter.count(chunk(16000), measure) == (16000, True)
    assert counter.count(chunk(48000), measure) == (48000, False)
    assert counter.count(chunk(32000), measure) == (32000, False)
    counter.disable('prompt count mismatch')
    assert counter.count(chunk(32000), measure) == (32000, False)
    assert counter.mode == 'exact' and not counter._counts


def test_distinct_tokenizers_with_equal_layout_can_reuse(native028):
    prep, hf, _ = native028
    hf.tokenizer = fake_tokenizer(prep.renderer.tokenizer.token_id)
    assert hf.tokenizer is not prep.renderer.tokenizer
    counter = build(native028)
    assert counter.mode == 'length_lru'
    audio = np.zeros(16000, np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    assert counter.count(audio, lambda _: 99) == (12, True)


def test_distinct_tokenizers_with_different_ids_use_exact(native028):
    prep, hf, _ = native028
    hf.tokenizer = fake_tokenizer(prep.renderer.tokenizer.token_id + 1)
    assert build(native028).mode == 'exact'


@pytest.mark.parametrize('which', ['processor', 'renderer'])
def test_one_tokenizer_id_change_disables_cache(native028, which):
    prep, hf, _ = native028
    hf.tokenizer = fake_tokenizer(prep.renderer.tokenizer.token_id)
    counter = build(native028)
    audio = np.zeros(16000, np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    (hf.tokenizer if which == 'processor' else prep.renderer.tokenizer).token_id += 1
    assert counter.count(audio, lambda _: 21) == (21, False)
    assert counter.mode == 'exact' and not counter._counts


def test_both_tokenizers_changed_to_same_new_id_still_invalidates(native028):
    prep, hf, _ = native028
    hf.tokenizer = fake_tokenizer(prep.renderer.tokenizer.token_id)
    counter = build(native028)
    audio = np.zeros(16000, np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    hf.tokenizer.token_id += 1
    prep.renderer.tokenizer.token_id += 1
    assert counter.count(audio, lambda _: 21) == (21, False)
    assert counter.mode == 'exact' and not counter._counts


def test_tokenizer_encoding_mutation_invalidates_existing_entry(native028):
    prep, hf, _ = native028
    hf.tokenizer = fake_tokenizer(prep.renderer.tokenizer.token_id)
    counter = build(native028)
    audio = np.zeros(16000, np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    hf.tokenizer.encode = lambda *args, **kwargs: [151646, 151646]
    assert counter.count(audio, lambda _: 21) == (21, False)
    assert counter.mode == 'exact' and not counter._counts


@pytest.mark.parametrize('which', ['processor', 'renderer'])
@pytest.mark.parametrize('encoded', [[], [151646, 151646], [999]])
def test_non_single_token_layout_falls_back_to_exact(native028, which, encoded):
    prep, hf, _ = native028
    hf.tokenizer = fake_tokenizer(prep.renderer.tokenizer.token_id)
    tokenizer = hf.tokenizer if which == 'processor' else prep.renderer.tokenizer
    tokenizer.encode = lambda *args, **kwargs: encoded
    assert build(native028).mode == 'exact'


@pytest.mark.parametrize('vllm_version', ['0.27.0', '0.28.0rc1', '0.28.1', '0.29.0'])
def test_unverified_vllm_is_exact(native028, vllm_version):
    assert build(native028, version=vllm_version).mode == 'exact'


@pytest.mark.parametrize('transformers_version', ['4.57.1', '5.5.2', '5.5.4', '5.6.1', '5.6.3', '6.0.0', None])
def test_unverified_transformers_is_exact(native028, monkeypatch, transformers_version):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__=transformers_version))
    assert build(native028).mode == 'exact'


@pytest.mark.parametrize('change', ['missing_renderer', 'unknown_processor', 'unknown_hf',
                                  'unknown_feature', 'wrong_model', 'wrong_rate', 'wrong_hop',
                                  'wrong_token', 'config_overrides', 'multimodal_overrides'])
def test_unverified_processor_or_configuration_is_exact(native028, change):
    prep, hf, feature = native028
    if change == 'missing_renderer': prep.renderer = None
    elif change == 'unknown_processor': prep.renderer.mm_processor = SimpleNamespace(info=SimpleNamespace(get_hf_processor=lambda: hf))
    elif change == 'unknown_hf': prep.renderer.mm_processor.info.get_hf_processor = lambda: SimpleNamespace(feature_extractor=feature)
    elif change == 'unknown_feature': hf.feature_extractor = SimpleNamespace(**feature.__dict__)
    elif change == 'wrong_model': prep.model_config.hf_config.model_type = 'qwen2_5_omni'
    elif change == 'wrong_rate': feature.sampling_rate = 8000
    elif change == 'wrong_hop': feature.hop_length = 320
    elif change == 'wrong_token': hf.audio_token = '<other>'
    elif change == 'config_overrides': prep.model_config.mm_processor_kwargs = {'truncation': True}
    elif change == 'multimodal_overrides': prep.model_config.multimodal_config.mm_processor_kwargs = {'padding': False}
    assert build(native028).mode == 'exact'


@pytest.mark.parametrize('change', ['feature', 'renderer', 'processor', 'model_config', 'overrides', 'tokenizer', 'token_id'])
def test_configuration_mutation_invalidates_cached_counts(native028, change):
    prep, _, feature = native028
    counter = build(native028)
    audio = np.zeros(16000, np.float32)
    assert counter.count(audio, lambda _: 12) == (12, False)
    if change == 'feature': feature.n_samples += 1
    elif change == 'renderer': prep.renderer = SimpleNamespace(mm_processor=prep.renderer.mm_processor)
    elif change == 'processor': prep.renderer.mm_processor = object()
    elif change == 'model_config': prep.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen3_omni_moe'))
    elif change == 'overrides': prep.model_config.multimodal_config.mm_processor_kwargs = {'truncation': True}
    elif change == 'tokenizer': prep.renderer.tokenizer = object()
    elif change == 'token_id': prep.renderer.tokenizer.token_id += 1
    assert counter.count(audio, lambda _: 21) == (21, False)
    assert counter.mode == 'exact' and not counter._counts


def test_disabled_or_failed_count_does_not_reuse(native028):
    assert build(native028, size=0).mode == 'exact'
    counter = build(native028)
    audio = np.zeros(16000, np.float32)
    with pytest.raises(RuntimeError):
        counter.count(audio, lambda _: (_ for _ in ()).throw(RuntimeError('failed')))
    assert counter.count(audio, lambda _: 12) == (12, False)


def test_original_011_compatibility_still_available(native028, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(__version__='4.57.1'))
    current = native028[0]
    legacy = SimpleNamespace(model_config=current.model_config,
                             _get_mm_processor=lambda: current.renderer.mm_processor)
    counter = AudioTokenCounter.for_preprocessor(legacy, vllm_version='0.11.1', max_entries=2)
    assert counter.mode == 'length_lru'
    assert counter.count(np.zeros(16000, np.float32), lambda _: 12) == (12, False)
    assert counter.count(np.ones(16000, np.float32), lambda _: 99) == (12, True)


@pytest.mark.parametrize('audio', [np.zeros(10, np.float64), np.zeros((1, 10), np.float32), [], np.zeros(0, np.float32)])
def test_unusual_audio_bypasses_cache(native028, audio):
    counter = build(native028)
    assert counter.count(audio, lambda _: 7) == (7, False)
    assert counter.count(audio, lambda _: 7) == (7, False)
    assert not counter._counts


@pytest.mark.parametrize('expected_version', ['5.5.3', '5.6.2'])
def test_optional_hf_real_extractor_frame_counts_are_content_independent(expected_version):
    """Optional CPU check; does not replace actual vLLM 0.28/NPU validation."""
    try:
        import transformers
        if transformers.__version__ != expected_version:
            pytest.skip(f'requires Transformers {expected_version}')
        from transformers import WhisperFeatureExtractor
    except ImportError as exc:
        pytest.skip(f'optional processor unavailable: {exc}')
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160)
    rng = np.random.default_rng(553)
    for length in (7999, 8000, 8001, 15999, 16000, 16001, 32000, 48000):
        silence = np.zeros(length, np.float32)
        noise = rng.uniform(-0.5, 0.5, length).astype(np.float32)
        padded_silence = np.pad(silence, (0, (-length) % 160))
        padded_noise = np.pad(noise, (0, (-length) % 160))
        features = extractor([padded_silence, padded_noise], sampling_rate=16000,
                             padding=True, truncation=False, return_attention_mask=True,
                             return_tensors='np')
        counts = features['attention_mask'].sum(-1).tolist()
        assert counts[0] == counts[1]
