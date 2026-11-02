"""Bounded memoization of *measured* audio placeholder expansion lengths.

This is not an audio-feature cache and never stores a waveform.  The first
occurrence of each exact sample length is still counted by the independent
vLLM preprocessor.  The optimization is restricted to the native, audited
vLLM 0.11.1 / Transformers 4.57.1 Qwen3-Omni audio path.  Other installations
keep the original exact counter, and generation must still check its actual
expanded token count before committing a response.

In that path, audio is padded to the feature extractor's hop length; audio
placeholder counts derive from frame lengths / attention-mask lengths, not
signal values.  We deliberately do not replace that behavior with a guessed
tokens-per-second formula.  Primary implementation references:
https://github.com/vllm-project/vllm/blob/v0.11.1/vllm/model_executor/models/qwen3_omni_moe_thinker.py
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py
"""

from collections import OrderedDict
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
        if vllm_version != '0.11.1':
            counter.reason = 'unverified vLLM version'
            return counter
        try:
            import transformers

            if getattr(transformers, '__version__', None) != '4.57.1':
                counter.reason = 'unverified Transformers version'
                return counter
            mm_processor = preprocessor._get_mm_processor()
            hf_processor = mm_processor.info.get_hf_processor()
            feature = hf_processor.feature_extractor

            def signature():
                # Reject replacement processors and user overrides, rather than
                # reusing counts across different preprocessing semantics.
                if preprocessor._get_mm_processor() is not mm_processor:
                    raise ValueError('multimodal processor changed')
                if (_type_id(mm_processor) != _MM_TYPE
                        or _type_id(hf_processor) != _HF_TYPE
                        or _type_id(feature) != _FEATURE_TYPE
                        or hf_processor.feature_extractor is not feature):
                    raise ValueError('unverified audio processor classes')
                model_config = preprocessor.model_config
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
                return values

            counter._signature = signature()
            counter._signature_reader = signature
            counter.mode = 'length_lru'
            counter.reason = 'native vLLM 0.11.1 / Transformers 4.57.1 Qwen3-Omni'
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
