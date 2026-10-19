"""Bounded multi-session batches with mocked vLLM; no GPU throughput claims."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from flex_web.model import (
    ContextLimit, Generation, ModelBackendError, ModelEngine, ModelOptions, ProtocolError,
)
from test_model import fake_transformers, fake_vllm  # shared CPU-only backend fixtures


@pytest.fixture
def batch_engine(fake_vllm, monkeypatch):
    options, observed, tok = fake_vllm
    engine = ModelEngine(replace(options, max_batch_size=4))
    observed.batches = []
    observed.malformed = set()
    observed.bad_expansion = set()
    observed.failure = None
    pad = tok.convert_tokens_to_ids('<|audio_pad|>')

    def generate(requests, sampling, use_tqdm=False):
        assert use_tqdm is False
        observed.batches.append(requests)
        if observed.failure is not None:
            raise observed.failure
        outputs = []
        for index, request in enumerate(requests):
            arrays = iter(request['multi_modal_data']['audio'])
            expanded = []
            for token in request['prompt_token_ids']:
                expanded.extend([pad] * max(1, len(next(arrays)) // 1600)
                                if token == pad else [token])
            if index in observed.bad_expansion:
                expanded.append(999)
            last = request['multi_modal_data']['audio'][-1]
            marker = int(last[0])
            prompt = tok.decode(request['prompt_token_ids'])
            lang = 'zh' if '翻译为中文' in prompt else 'en'
            completion = SimpleNamespace(
                token_ids=tok.encode(f'<|DIRECT|> {lang}-speaker-{marker}<|im_end|>'),
                finish_reason='stop')
            outputs.append(SimpleNamespace(prompt_token_ids=expanded,
                                           outputs=[] if index in observed.malformed else [completion]))
        return outputs

    monkeypatch.setattr(engine.llm, 'generate', generate)
    yield engine, observed, tok
    engine.close()


def audio(marker, samples=16000):
    return np.full(samples, marker, np.float32)


def test_four_private_bilingual_sessions_in_one_vllm_call(batch_engine):
    engine, observed, tok = batch_engine
    sessions = [engine.new_session(lang) for lang in ('zh', 'en', 'zh', 'en')]
    first_audio = [audio(index + 1) for index in range(4)]
    first = engine.generate_batch(list(zip(sessions, first_audio)))
    assert len(observed.batches) == 1
    assert len(observed.batches[0]) == 4
    assert observed.llms[0].kwargs['max_num_seqs'] == 4
    assert engine.metadata['max_batch_size'] == 4
    for index, (session, source, generation) in enumerate(zip(sessions, first_audio, first)):
        assert isinstance(generation, Generation)
        assert generation.reply == f'<|DIRECT|> {session.target_lang}-speaker-{index + 1}'
        assert session.total_committed_turns == 0
        session.commit(source, generation)
        source[:] = 99  # committed buffers must not alias producers or peers
    second_audio = [audio(index + 11, 32000) for index in range(4)]
    second = engine.generate_batch(list(zip(sessions, second_audio)))
    assert len(observed.batches) == 2
    for index, (session, source, generation, request) in enumerate(
            zip(sessions, second_audio, second, observed.batches[1])):
        assert [sample[0] for sample in request['multi_modal_data']['audio']] == [index + 1, index + 11]
        prompt = tok.decode(request['prompt_token_ids'])
        assert first[index].reply in prompt
        assert all(other.reply not in prompt for offset, other in enumerate(first) if offset != index)
        assert generation.prompt_tokens == len(request['prompt_token_ids']) + 9 + 19
        assert generation.token_ids == tok.encode(generation.reply)
        session.commit(source, generation)
        assert session.total_committed_turns == 2
        assert session.committed_audio_extra_tokens == 28
    sessions[0].close()
    assert not sessions[0].history
    assert all(session.total_committed_turns == 2 and len(session.audios) == 2
               for session in sessions[1:])


def test_bad_audio_preparation_only_skips_that_item(batch_engine):
    engine, observed, _ = batch_engine
    sessions = [engine.new_session('en') for _ in range(4)]
    sources = [audio(1), np.array([np.nan]), audio(3), audio(4)]
    results = engine.generate_batch(list(zip(sessions, sources)))
    assert isinstance(results[1], ValueError)
    assert not isinstance(results[1], ModelBackendError)
    assert len(observed.batches) == 1 and len(observed.batches[0]) == 3
    assert [value.reply for value in results if isinstance(value, Generation)] == [
        '<|DIRECT|> en-speaker-1', '<|DIRECT|> en-speaker-3', '<|DIRECT|> en-speaker-4']
    assert sessions[1]._pending is None and not sessions[1].audios


@pytest.mark.parametrize('failure', ['malformed', 'bad_expansion', 'decode'])
def test_one_output_failure_does_not_pollute_other_session(batch_engine, monkeypatch, failure):
    engine, observed, tok = batch_engine
    sessions = [engine.new_session('en') for _ in range(4)]
    if failure == 'decode':
        original = tok.decode

        def decode(ids, **kwargs):
            decoded = original(ids, **kwargs)
            if decoded == '<|DIRECT|> en-speaker-2':
                raise ValueError('decoder failure')
            return decoded

        monkeypatch.setattr(tok, 'decode', decode)
    else:
        getattr(observed, failure).add(1)
    sources = [audio(i + 1) for i in range(4)]
    results = engine.generate_batch(list(zip(sessions, sources)))
    assert isinstance(results[1], Exception)
    assert not isinstance(results[1], ModelBackendError)
    assert sessions[1]._pending is None
    for index in (0, 2, 3):
        assert isinstance(results[index], Generation)
        sessions[index].commit(sources[index], results[index])
        assert sessions[index].total_committed_turns == 1
    assert sessions[1].total_committed_turns == 0


def test_preprocessor_failure_is_private(batch_engine, monkeypatch):
    engine, observed, _ = batch_engine
    original = engine.audio_expansion

    def expansion(source):
        if source[0] == 2:
            raise ValueError('invalid audio feature shape')
        return original(source)

    monkeypatch.setattr(engine, 'audio_expansion', expansion)
    sessions = [engine.new_session('zh') for _ in range(3)]
    result = engine.generate_batch([(session, audio(i + 1)) for i, session in enumerate(sessions)])
    assert isinstance(result[1], ValueError)
    assert not isinstance(result[1], ModelBackendError)
    assert len(observed.batches[0]) == 2
    assert result[0].reply.endswith('speaker-1') and result[2].reply.endswith('speaker-3')


def test_duplicate_foreign_and_pending_sessions_never_enter_batch(batch_engine):
    engine, observed, _ = batch_engine
    duplicate, valid = engine.new_session('en'), engine.new_session('zh')
    foreign = ModelEngine(backend='mock').new_session('en')
    results = engine.generate_batch([(duplicate, audio(1)), (valid, audio(2)),
                                     (duplicate, audio(3)), (foreign, audio(4))])
    assert all(isinstance(results[index], ValueError) for index in (0, 2, 3))
    assert len(observed.batches[0]) == 1
    assert duplicate._pending is None and foreign._pending is None
    assert isinstance(results[1], Generation)
    pending = engine.generate_batch([(valid, audio(2))])[0]
    assert isinstance(pending, RuntimeError) and 'previous generation' in str(pending)
    assert len(observed.batches) == 1


@pytest.mark.parametrize('failure', ['backend', 'count'])
def test_batch_failure_does_not_apply_planned_rollovers(batch_engine, monkeypatch, failure):
    engine, observed, _ = batch_engine
    engine.options = replace(engine.options, max_audio_turns=1)
    sessions = [engine.new_session('en') for _ in range(4)]
    items = [(session, audio(index + 1)) for index, session in enumerate(sessions)]
    for (session, source), generation in zip(items, engine.generate_batch(items)):
        session.commit(source, generation)
    old_states = [(list(session.history), session.context_metadata) for session in sessions]
    if failure == 'backend':
        observed.failure = RuntimeError('whole batch failure')
    else:
        monkeypatch.setattr(engine.llm, 'generate', lambda *args, **kwargs: [])
    results = engine.generate_batch(items)
    assert all(isinstance(result, ModelBackendError) for result in results)
    assert all(isinstance(result.__cause__, RuntimeError) for result in results)
    if failure == 'backend':
        assert all(result.__cause__ is observed.failure for result in results)
    else:
        assert all('output count' in str(result.__cause__) for result in results)
    for session, (history, metadata) in zip(sessions, old_states):
        assert session.history == history and session.context_metadata == metadata
        assert session._pending is None


def test_shared_failure_does_not_upgrade_already_rejected_input(batch_engine):
    engine, observed, _ = batch_engine
    sessions = [engine.new_session('en') for _ in range(4)]
    observed.failure = ValueError('backend cannot produce the requested batch')
    results = engine.generate_batch([(sessions[0], audio(1)),
                                     (sessions[1], np.array([np.nan])),
                                     (sessions[2], audio(3)), (sessions[3], audio(4))])
    assert isinstance(results[1], ValueError)
    assert not isinstance(results[1], ModelBackendError)
    for index in (0, 2, 3):
        assert isinstance(results[index], ModelBackendError)
        assert results[index].__cause__ is observed.failure
    assert all(session._pending is None and not session.audios for session in sessions)


def test_single_session_interface_raises_shared_failure_with_original_cause(batch_engine):
    engine, observed, _ = batch_engine
    session = engine.new_session('en')
    observed.failure = RuntimeError('GPU engine failure')
    with pytest.raises(ModelBackendError, match='GPU engine failure') as raised:
        session.generate(audio(1))
    assert raised.value.__cause__ is observed.failure
    assert session._pending is None and session.total_committed_turns == 0


def test_stop_budget_failure_does_not_block_another_session(batch_engine):
    engine, observed, _ = batch_engine
    engine.options = replace(engine.options, max_audio_turns=1, context_policy='stop')
    full, empty = engine.new_session('en'), engine.new_session('zh')
    first = full.generate(audio(1))
    full.commit(audio(1), first)
    results = engine.generate_batch([(full, audio(1)), (empty, audio(2))])
    assert isinstance(results[0], ContextLimit)
    assert isinstance(results[1], Generation)
    assert len(observed.batches[-1]) == 1
    assert full.total_committed_turns == 1 and full._pending is None


def test_cancel_after_batch_discards_only_canceled_pending(batch_engine):
    engine, _, _ = batch_engine
    first, second = engine.new_session('en'), engine.new_session('zh')
    results = engine.generate_batch([(first, audio(1)), (second, audio(2))])
    first.close()
    with pytest.raises(RuntimeError, match='closed'):
        first.commit(audio(1), results[0])
    second.commit(audio(2), results[1])
    assert second.total_committed_turns == 1
    assert first.total_committed_turns == 0 and first._pending is None


def test_batch_preserves_existing_protocol_commit_guard(batch_engine, monkeypatch):
    engine, _, _ = batch_engine
    one, two = engine.new_session('en'), engine.new_session('zh')
    generated = engine.generate_batch([(one, audio(1)), (two, audio(2))])
    generated[0].reply = 'unprefixed invalid reply'
    with pytest.raises(ProtocolError):
        one.commit(audio(1), generated[0])
    two.commit(audio(2), generated[1])
    assert not one.history[len(one._system_tokens):] and not one.audios
    assert two.total_committed_turns == 1


def test_mock_batch_is_private_and_respects_maximum(monkeypatch):
    monkeypatch.setattr('flex_web.model.time.sleep', lambda _: None)
    engine = ModelEngine(ModelOptions(max_batch_size=4), backend='mock')
    sessions = [engine.new_session(lang) for lang in ('en', 'zh', 'en', 'zh')]
    sources = [audio(i + 1) for i in range(4)]
    for stage in ('<|FLEX|>', '<|WAIT|>', '<|DIRECT|>'):
        results = engine.generate_batch(list(zip(sessions, sources)))
        for session, source, generation in zip(sessions, sources, results):
            assert generation.reply.startswith(stage)
            session.commit(source, generation)
    assert all(session.total_committed_turns == 3 for session in sessions)
    too_many = engine.generate_batch([(sessions[0], audio(1))] * 5)
    assert len(too_many) == 5 and all(isinstance(value, ValueError) for value in too_many)
    assert engine.generate_batch([]) == []
    engine.close()
    assert isinstance(engine.generate_batch([(sessions[0], audio(1))])[0], RuntimeError)


@pytest.mark.parametrize('size', [0, -1, True, 1.5])
def test_batch_size_requires_positive_integer(size):
    with pytest.raises(ValueError, match='max_batch_size'):
        ModelOptions(max_batch_size=size)
