"""Accelerated multi-hour context tests; no GPU, sleeps, network, or ASR claims."""

import weakref

import numpy as np
import pytest

from flex_web.model import ContextLimit, Generation, ModelEngine, ModelOptions, SAMPLE_RATE, parse_reply


@pytest.fixture(autouse=True)
def no_mock_delay(monkeypatch):
    monkeypatch.setattr('flex_web.model.time.sleep', lambda _: None)
    monkeypatch.setattr('flex_web.model.LOG.info', lambda *args, **kwargs: None)


@pytest.mark.parametrize('chunk_seconds,turn_count', [(1, 21600), (3, 7200)])
def test_six_hour_session_bounds_pcm_tokens_and_turn_records(chunk_seconds, turn_count):
    options = ModelOptions(max_audio_turns=32, history_window_seconds=60,
                           history_retain_seconds=30)
    engine = ModelEngine(options, backend='mock')
    session = engine.new_session('en')
    audio = np.zeros(SAMPLE_RATE * chunk_seconds, np.float32)
    first_snapshot = None
    last_rollovers = 0
    tags = ('<|FLEX|>', '<|WAIT|>', '<|DIRECT|>')
    for index in range(turn_count):
        generation = session.generate(audio)
        assert parse_reply(generation.reply)[0] == tags[index % 3]
        assert generation.prompt_tokens + options.max_tokens <= options.max_model_len
        session.commit(audio, generation)
        if index == 0:
            first_snapshot = weakref.ref(session.audios[0])
        metadata = session.context_metadata
        assert metadata['total_committed_turns'] == index + 1
        assert metadata['retained_turns'] == len(session._turns) == len(session.audios)
        assert metadata['retained_turns'] <= options.max_audio_turns
        assert metadata['retained_audio_seconds'] <= options.history_window_seconds
        assert metadata['retained_context_tokens'] <= options.max_model_len
        assert sum(item.nbytes for item in session.audios) <= 60 * SAMPLE_RATE * 4
        assert len(session.history) <= options.max_model_len
        assert metadata['rollovers'] >= last_rollovers
        last_rollovers = metadata['rollovers']
    assert metadata['total_committed_audio_seconds'] == 6 * 3600
    assert metadata['dropped_turns'] + metadata['retained_turns'] == turn_count
    assert metadata['dropped_audio_seconds'] + metadata['retained_audio_seconds'] == 6 * 3600
    assert metadata['rollovers'] > 100
    assert first_snapshot() is None
    session.close()
    assert not session.audios and not session._turns and not session.history
    assert session.context_metadata['retained_context_tokens'] == 0
    assert session.context_metadata['total_committed_turns'] == turn_count


def test_audio_window_boundary_and_atomic_rollover():
    engine = ModelEngine(ModelOptions(history_window_seconds=4, history_retain_seconds=2), 'mock')
    session = engine.new_session('zh')
    audio = np.zeros(SAMPLE_RATE, np.float32)
    for _ in range(4):
        generation = session.generate(audio)
        session.commit(audio, generation)
    assert session.context_metadata['rollovers'] == 0
    generation = session.generate(audio)
    assert session.context_metadata['retained_turns'] == 4
    with pytest.raises(RuntimeError, match='previous generation'):
        session.generate(audio)
    session.commit(audio, generation)
    metadata = session.context_metadata
    assert metadata['rollovers'] == 1
    assert metadata['retained_turns'] == 2
    assert metadata['retained_audio_seconds'] == 2
    assert metadata['dropped_turns'] == 3
    assert metadata['last_rollover']['reasons'] == ['audio_window']
    metadata['last_rollover']['reasons'].append('external mutation')
    assert session.context_metadata['last_rollover']['reasons'] == ['audio_window']


def test_long_wait_history_is_bounded_without_inventing_translation(monkeypatch):
    engine = ModelEngine(ModelOptions(history_window_seconds=6, history_retain_seconds=3), 'mock')
    session = engine.new_session('en')
    original = session._mock_generate

    def wait_only(audio, plan, extra):
        generated = original(audio, plan, extra)
        return Generation('<|WAIT|>', list(b'<|WAIT|>'), generated.prompt_tokens, 'stop', extra)

    monkeypatch.setattr(session, '_mock_generate', wait_only)
    audio = np.zeros(SAMPLE_RATE, np.float32)
    for _ in range(1000):
        generation = session.generate(audio)
        assert parse_reply(generation.reply) == ('<|WAIT|>', '')
        session.commit(audio, generation)
    assert session.context_metadata['total_committed_turns'] == 1000
    assert session.context_metadata['retained_turns'] <= 6
    assert session.context_metadata['rollovers'] > 100
    assert all(turn.token_ids == tuple(b'<|WAIT|>') for turn in session._turns)
    # Bounds are memory guarantees only: WAIT does not prove source coverage.


def test_single_chunk_exceeds_window_is_rejected_without_clearing():
    engine = ModelEngine(ModelOptions(history_window_seconds=2, history_retain_seconds=1), 'mock')
    session = engine.new_session('en')
    audio = np.zeros(SAMPLE_RATE, np.float32)
    generation = session.generate(audio)
    session.commit(audio, generation)
    metadata = session.context_metadata
    with pytest.raises(ContextLimit, match='incoming audio exceeds'):
        session.generate(np.zeros(3 * SAMPLE_RATE, np.float32))
    assert session.context_metadata == metadata


def test_short_turn_limit_does_not_restart_mock_action_sequence():
    engine = ModelEngine(ModelOptions(max_audio_turns=1), 'mock')
    session = engine.new_session('en')
    audio = np.zeros(SAMPLE_RATE, np.float32)
    tags = []
    for _ in range(7):
        generation = session.generate(audio)
        tags.append(parse_reply(generation.reply)[0])
        session.commit(audio, generation)
    assert tags == ['<|FLEX|>', '<|WAIT|>', '<|DIRECT|>'] * 2 + ['<|FLEX|>']
    assert session.context_metadata['rollovers'] == 6
    assert len(session.audios) == 1
