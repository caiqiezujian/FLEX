"""Bounded memoization of *measured* audio placeholder expansion lengths.

This is not an audio-feature cache and never stores a waveform.  The first
occurrence of each exact sample length is still counted by the independent
vLLM preprocessor.  The optimization is restricted to the native, audited
vLLM 0.11.1 / Transformers 4.57.1 and the experimental vLLM 0.28.0 /
Transformers 5.5.3 or 5.6.2 native Qwen3-Omni audio paths. Other installations
keep the original exact counter, and generation must still check its actual
expanded token count before committing a response.

In that path, audio is padded to the feature extractor's hop length; audio
placeholder counts derive from frame lengths / attention-mask lengths, not
signal values.  We deliberately do not replace that behavior with a guessed
tokens-per-second formula.  Primary implementation references:
https://github.com/vllm-project/vllm/blob/v0.11.1/vllm/model_executor/models/qwen3_omni_moe_thinker.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen3_omni_moe_thinker.py
https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
https://github.com/huggingface/transformers/blob/v5.6.2/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
"""

from collections import OrderedDict
from collections.abc import Mapping
from numbers import Integral

import numpy as np


_MM_TYPE = ('vllm.model_executor.models.qwen3_omni_moe_thinker',
            'Qwen3OmniMoeThinkerMultiModalProcessor')
_HF_TYPE = ('transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe',
            'Qwen3OmniMoeProcessor')
_FEATURE_TYPE = ('transformers.models.whisper.feature_extraction_whisper',
                 'WhisperFeatureExtractor')
_FEATURE_FIELDS = ('sampling_rate', 'hop_length', 'n_fft', 'n_samples',
                   'chunk_length', 'nb_max_frames', 'feature_size',
                   'padding_side', 'padding_value', 'return_attention_mask',
                   'dither')
# Transformers 5.5.3/5.6.2 AudioKwargs: ProcessorMixin._merge_kwargs may
# override audio defaults from tokenizer.init_kwargs before the call.
_AUDIO_TOKENIZER_KWARGS = frozenset({
    'sampling_rate', 'raw_speech', 'padding', 'max_length', 'truncation',
    'pad_to_multiple_of', 'return_attention_mask', 'return_tensors',
})


def _type_id(value):
    cls = type(value)
    return cls.__module__, cls.__qualname__


class AudioTokenCounter:
    """Per-engine, serial-worker-only LRU of exact length -> expansion counts."""

    def __init__(self, max_entries=64):
        if (isinstance(max_entries, bool) or not isinstance(max_entries, int)
                or not 0 <= max_entries <= 1024):
            raise ValueError('audio token cache size must be an integer in 0..1024')
        self.max_entries = max_entries
        self.mode = 'exact'
        self.reason = 'processor has not been verified'
        self._counts = OrderedDict()
        self._signature_reader = None
        self._signature = None

    @classmethod
    def for_preprocessor(cls, preprocessor, *, vllm_version, max_entries=64):
        counter = cls(max_entries)
        if not max_entries:
            counter.reason = 'disabled by configuration'
            return counter
        supported_versions = {'0.11.1': frozenset({'4.57.1'}),
                              '0.28.0': frozenset({'5.5.3', '5.6.2'})}
        expected_transformers = supported_versions.get(vllm_version)
        if expected_transformers is None:
            counter.reason = 'unverified vLLM version'
            return counter
        try:
            import transformers

            transformers_version = getattr(transformers, '__version__', None)
            if transformers_version not in expected_transformers:
                counter.reason = 'unverified Transformers version'
                return counter
            if vllm_version == '0.28.0':
                renderer = preprocessor.renderer
                mm_processor = renderer.mm_processor
                def current_processor():
                    if preprocessor.renderer is not renderer:
                        raise ValueError('multimodal renderer changed')
                    return renderer.mm_processor
            else:
                mm_processor = preprocessor._get_mm_processor()
                def current_processor():
                    return preprocessor._get_mm_processor()
            hf_processor = mm_processor.info.get_hf_processor()
            feature = hf_processor.feature_extractor
            original_model_config = preprocessor.model_config
            original_hf_config = original_model_config.hf_config
            original_mm_config = getattr(original_model_config, 'multimodal_config', None)
            if vllm_version == '0.28.0':
                tokenizer = hf_processor.tokenizer
                # vLLM may give the processor a deep-copied tokenizer; compare
                # placeholder IDs while tracking each tokenizer independently.
                renderer_tokenizer = renderer.tokenizer

            def signature():
                # Reject replacement processors and user overrides, rather than
                # reusing counts across different preprocessing semantics.
                if current_processor() is not mm_processor:
                    raise ValueError('multimodal processor changed')
                if (_type_id(mm_processor) != _MM_TYPE
                        or _type_id(hf_processor) != _HF_TYPE
                        or _type_id(feature) != _FEATURE_TYPE
                        or hf_processor.feature_extractor is not feature):
                    raise ValueError('unverified audio processor classes')
                model_config = preprocessor.model_config
                if model_config is not original_model_config or model_config.hf_config is not original_hf_config:
                    raise ValueError('model configuration changed')
                if vllm_version == '0.28.0' and mm_processor.info.get_hf_processor() is not hf_processor:
                    raise ValueError('Hugging Face processor changed')
                if vllm_version == '0.28.0':
                    if (renderer.tokenizer is not renderer_tokenizer or hf_processor.tokenizer is not tokenizer
                            or getattr(model_config, 'multimodal_config', None) is not original_mm_config):
                        raise ValueError('tokenizer or multimodal configuration changed')
                    init_kwargs = getattr(tokenizer, 'init_kwargs', None)
                    if not isinstance(init_kwargs, Mapping):
                        raise ValueError('tokenizer initialization arguments unavailable')
                    valid_kwargs = getattr(feature, 'valid_kwargs', None)
                    if valid_kwargs is not None:
                        raise ValueError('custom audio feature argument schema')
                    if set(init_kwargs) & _AUDIO_TOKENIZER_KWARGS:
                        raise ValueError('tokenizer initialization overrides audio parameters')
                    token_id = tokenizer.convert_tokens_to_ids(hf_processor.audio_token)
                    if isinstance(token_id, bool) or not isinstance(token_id, Integral) or token_id < 0:
                        raise ValueError('invalid audio placeholder token ID')
                    if renderer_tokenizer.convert_tokens_to_ids(hf_processor.audio_token) != token_id:
                        raise ValueError('audio placeholder token IDs differ')
                    for candidate in (tokenizer, renderer_tokenizer):
                        if candidate.encode(hf_processor.audio_token, add_special_tokens=False) != [token_id]:
                            raise ValueError('audio placeholder is not a single matching token')
                else:
                    token_id = None
                if model_config.hf_config.model_type != 'qwen3_omni_moe':
                    raise ValueError('unverified model type')
                if getattr(model_config, 'mm_processor_kwargs', None):
                    raise ValueError('custom multimodal processor arguments')
                multimodal = getattr(model_config, 'multimodal_config', None)
                if getattr(multimodal, 'mm_processor_kwargs', None):
                    raise ValueError('custom multimodal processor arguments')
                if feature.sampling_rate != 16000 or feature.hop_length != 160:
                    raise ValueError('unverified audio sampling configuration')
                if getattr(hf_processor, 'audio_token', None) != '<|audio_pad|>':
                    raise ValueError('unverified audio placeholder')
                values = tuple(getattr(feature, key, None) for key in _FEATURE_FIELDS)
                if any(not isinstance(value, (str, int, float, bool, type(None)))
                       for value in values):
                    raise ValueError('non-scalar feature configuration')
                return (values, int(token_id)) if token_id is not None else values

            counter._signature = signature()
            counter._signature_reader = signature
            counter.mode = 'length_lru'
            counter.reason = f'native vLLM {vllm_version} / Transformers {transformers_version} Qwen3-Omni'
        except Exception as exc:
            # The optional optimization must never prevent a previously valid
            # installation from starting.  Exact preprocessing remains intact.
            counter.disable(f'processor verification unavailable: {type(exc).__name__}')
        return counter

    def count(self, audio, exact_count):
        """Return (expansion count, cache hit); exact_count accepts the waveform.

        The caller validates/resamples the audio and supplies one mono float32
        ndarray.  Unexpected layouts bypass memoization; exact_count retains
        its existing validation and failure behavior.
        """
        if self.mode == 'length_lru':
            try:
                if self._signature_reader() != self._signature:
                    self.disable('audio preprocessing configuration changed')
            except Exception:
                self.disable('audio preprocessing configuration changed')
        eligible = (self.mode == 'length_lru'
                    and isinstance(audio, np.ndarray) and audio.ndim == 1
                    and audio.dtype == np.dtype('float32') and audio.size > 0)
        key = len(audio) if eligible else None
        if eligible and key in self._counts:
            value = self._counts.pop(key)
            self._counts[key] = value
            return value, True
        value = exact_count(audio)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError('audio expansion counter must return a non-negative integer')
        value = int(value)
        if eligible:
            self._counts[key] = value
            while len(self._counts) > self.max_entries:
                self._counts.popitem(last=False)
        return value, False

    def clear(self):
        self._counts.clear()

    def disable(self, reason):
        """Permanently return to exact counting, e.g. after a runtime mismatch."""
        self.clear()
        self.mode = 'exact'
        self.reason = str(reason)
        self._signature_reader = None
        self._signature = None
