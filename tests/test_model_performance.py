"""Input identity and timing invariants; fake vLLM, no GPU speed claims."""

from dataclasses import replace
import json

import numpy as np
import pytest

from flex_web.model import ModelBackendError, ModelEngine, ModelOptions
from test_model import fake_transformers, fake_vllm
from test_model_batch import batch_engine, audio


def test_audio_ids_survive_reuse_and_rollover_with_matching_snapshots(fake_vllm):
    options, observed, _ = fake_vllm
    engine = ModelEngine(replace(options, max_audio_turns=4))
    session = engine.new_session('en')
    ids = []
    for index in range(5):
        source = np.full(16000, index, np.float32)
        result = session.generate(source)
        request = observed.requests[-1]
        sent_ids = request['multi_modal_uuids']['audio']
        sent_audio = request['multi_modal_data']['audio']
        assert len(sent_ids) == len(sent_audio)
        assert all(not item.flags.writeable for item in sent_audio)
        ids.append(sent_ids[-1])
        assert len(set(ids)) == len(ids)
        assert sent_ids == (ids if index < 4 else ids[3:])
        assert [int(item[0]) for item in sent_audio] == (list(range(index + 1))
                                                       if index < 4 else [3, 4])
        session.commit(source, result)
        source[:] = -1
    assert [turn.audio_id for turn in session._turns] == ids[3:]
    assert session.context_metadata['rollovers'] == 1
    session.close()
    assert not session._turns and not session.audios


def test_failed_attempt_does_not_reuse_id_for_different_audio(fake_vllm, monkeypatch):
    options, observed, _ = fake_vllm
    engine = ModelEngine(options)
    session = engine.new_session('zh')
    real_generate = engine.llm.generate
    failed = []

    def fail(requests, *args, **kwargs):
        failed.append(requests[0]['multi_modal_uuids']['audio'][-1])
        raise RuntimeError('fake backend failure')

    monkeypatch.setattr(engine.llm, 'generate', fail)
    with pytest.raises(ModelBackendError):
        session.generate(np.zeros(16000, np.float32))
    assert session._pending is None and not session._turns
    monkeypatch.setattr(engine.llm, 'generate', real_generate)
    source = np.ones(16000, np.float32)
    result = session.generate(source)
    fresh = observed.requests[-1]['multi_modal_uuids']['audio'][-1]
    assert fresh not in failed
    session.commit(source, result)
    assert session._turns[0].audio_id == fresh


def test_batch_sessions_and_new_sessions_never_share_audio_identity(batch_engine):
    engine, observed, _ = batch_engine
    sessions = [engine.new_session('en') for _ in range(4)]
    same_audio = audio(0)
    first = engine.generate_batch([(session, same_audio) for session in sessions])
    ids = [request['multi_modal_uuids']['audio'][0] for request in observed.batches[-1]]
    assert len(set(ids)) == 4
    for session, result in zip(sessions, first):
        session.commit(same_audio, result)
        session.close()
    new_session = engine.new_session('en')
    result = new_session.generate(same_audio)
    new_id = observed.batches[-1][0]['multi_modal_uuids']['audio'][0]
    assert new_id not in ids
    new_session.commit(same_audio, result)


def test_timing_fields_are_per_request_and_batch_call_is_shared(batch_engine):
    engine, _, _ = batch_engine
    sessions = [engine.new_session('en'), engine.new_session('zh')]
    results = engine.generate_batch([(session, audio(index)) for index, session in enumerate(sessions)])
    assert results[0].timings is not results[1].timings
    assert results[0].timings['engine_generate_s'] == results[1].timings['engine_generate_s']
    for result in results:
        timing = result.timings
        assert timing['request_prepare_s'] >= timing['audio_count_s'] >= 0
        assert timing['engine_generate_s'] >= 0 and timing['response_decode_s'] >= 0
        assert isinstance(timing['audio_count_cache_hit'], bool)
        assert isinstance(timing['audio_count_mode'], str)
        json.dumps(timing, allow_nan=False)


def test_cuda_graph_option_reaches_engine_without_changing_default(fake_vllm):
    options, observed, _ = fake_vllm
    assert ModelOptions().enforce_eager is True
    engine = ModelEngine(replace(options, enforce_eager=False))
    assert observed.llms[-1].kwargs['enforce_eager'] is False
    assert engine.metadata['enforce_eager'] is False


def enable_count_memo_for_fake_backend(engine):
    # Production eligibility is covered separately in test_audio_tokens. The
    # fake backend has a known length-only expansion rule; exercise integration
    # without pretending it is the real audited processor implementation.
    counter = engine._audio_token_counter
    counter.mode = 'length_lru'
    counter.reason = 'test-only deterministic processor'
    counter._signature = ()
    counter._signature_reader = lambda: ()


def test_count_memo_reduces_preprocessing_and_preserves_exact_requests(fake_vllm):
    options, observed, _ = fake_vllm
    sources = [np.full(size, index / 10, np.float32)
               for index, size in enumerate((16000, 16000, 32000, 16000, 8000))]
    records = []
    for enabled in (False, True):
        engine = ModelEngine(replace(options, max_audio_turns=3))
        if enabled:
            enable_count_memo_for_fake_backend(engine)
        session = engine.new_session('en')
        counted = len(observed.count_requests)
        record = []
        for source in sources:
            result = session.generate(source)
            record.append((observed.requests[-1]['prompt_token_ids'], result.prompt_tokens,
                           result.token_ids, result.reply))
            session.commit(source, result)
        assert len(observed.count_requests) - counted == (3 if enabled else 5)
        assert result.timings['audio_count_mode'] == ('length_lru' if enabled else 'exact')
        records.append(record)
        session.close()
        engine.close()
        assert not engine._audio_token_counter._counts
    assert records[0] == records[1]


def test_runtime_mismatch_disables_memo_without_committing(fake_vllm):
    options, observed, _ = fake_vllm
    engine = ModelEngine(options)
    enable_count_memo_for_fake_backend(engine)
    session = engine.new_session('zh')
    source = np.zeros(16000, np.float32)
    first = session.generate(source)
    session.commit(source, first)
    history = list(session.history)
    observed.mismatch = 1
    with pytest.raises(RuntimeError, match='context expansion mismatch'):
        session.generate(source)
    assert session._pending is None and session.history == history
    assert session.total_committed_turns == 1
    assert len(observed.count_requests) == 1  # failed request used memoized count
    assert engine._audio_token_counter.mode == 'exact'
    assert not engine._audio_token_counter._counts
    assert engine.metadata['audio_count_mode'] == 'exact'
    observed.mismatch = 0
    recovered = session.generate(source)
    assert len(observed.count_requests) == 2
    assert recovered.timings['audio_count_cache_hit'] is False
    session.commit(source, recovered)


@pytest.mark.parametrize('changes', [
    {'audio_token_cache_size': -1}, {'audio_token_cache_size': 1025},
    {'audio_token_cache_size': True}, {'audio_token_cache_size': 1.5},
    {'enforce_eager': 'false'},
])
def test_performance_options_are_validated(changes):
    with pytest.raises(ValueError):
        ModelOptions(**changes)
