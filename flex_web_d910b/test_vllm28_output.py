"""Safety checks for accepting expanded prompt lengths from vLLM output."""
from types import SimpleNamespace

import numpy as np
import pytest

from flex_web_d910b import model


@pytest.mark.parametrize('value', [-1.0, float('inf'), float('nan')])
def test_invalid_cpu_offload_budget_is_rejected(value):
    with pytest.raises(ValueError):
        model.ModelOptions(cpu_offload_gb=value)


@pytest.mark.parametrize('actual_ids,error', [
    (None, 'prompt_token_ids'),
    ([1, 2], 'expansion mismatch'),
])
def test_missing_or_mismatched_prompt_count_never_creates_pending_turn(actual_ids, error):
    counter = model.AudioTokenCounter(0)
    engine = SimpleNamespace(
        backend='vllm', _closed=False, stop_ids=[],
        tok=SimpleNamespace(decode=lambda *args, **kwargs: '<|WAIT|>'),
        _audio_token_counter=counter, metadata={})
    session = object.__new__(model.ModelSession)
    session.engine, session._closed, session._pending = engine, False, None
    prepared = (np.zeros(10, np.float32), 0, None, None, 3, 'audio-id', {})
    output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[99], finish_reason='stop')],
                             prompt_token_ids=actual_ids)

    with pytest.raises(RuntimeError, match=error):
        session._finish_generation(prepared, output)
    assert session._pending is None
    assert engine.metadata['audio_count_mode'] == 'exact'


def test_matching_prompt_count_allows_pending_turn():
    engine = SimpleNamespace(
        backend='vllm', _closed=False, stop_ids=[],
        tok=SimpleNamespace(decode=lambda *args, **kwargs: '<|WAIT|>'))
    session = object.__new__(model.ModelSession)
    session.engine, session._closed, session._pending = engine, False, None
    prepared = (np.zeros(10, np.float32), 0, None, None, 3, 'audio-id', {})
    output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[99], finish_reason='stop')],
                             prompt_token_ids=[1, 2, 3])
    generation = session._finish_generation(prepared, output)
    assert generation.reply == '<|WAIT|>'
    assert generation.prompt_tokens == 3
    assert session._pending[0] is generation
