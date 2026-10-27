"""CPU-only protocol, lifecycle, and failure tests for the browser server.

The deterministic test engine is not a speech-recognition/translation model.
These tests never load a checkpoint or call an external service.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web import server


RATE = 16000


class DeterministicSession:
    def __init__(self, engine, language):
        self.engine = engine
        self.language = language
        self.generated = []
        self.committed = []

    def generate(self, audio):
        self.generated.append(audio.copy())
        self.engine.entered.set()
        if self.engine.block:
            if not self.engine.release.wait(5):
                raise RuntimeError('test did not release the model worker')
        index = len(self.generated) - 1
        if self.engine.fail_at == index:
            raise server.ContextLimit('test context limit')
        replies = self.engine.replies
        return SimpleNamespace(reply=replies[index % len(replies)],
                               finish_reason=self.engine.finish_reason,
                               prompt_tokens=100 + index,
                               token_ids=[10 + index])

    def commit(self, audio, result):
        self.committed.append((audio.copy(), result))

    def close(self):
        pass


class DeterministicEngine:
    def __init__(self, **options):
        self.sessions = []
        self.replies = options.get('replies', ['<|WAIT|>', '<|FLEX|>A safe beginning',
                                               '<|DIRECT|> and an ending.'])
        self.finish_reason = options.get('finish_reason', 'stop')
        self.fail_at = options.get('fail_at')
        self.block = options.get('block', False)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def new_session(self, language):
        session = DeterministicSession(self, language)
        self.sessions.append(session)
        return session

    def warmup(self, seconds=1):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    def make(engine_options=None, **settings):
        engine = DeterministicEngine(**(engine_options or {}))
        monkeypatch.setattr(server, 'ModelEngine', lambda *args, **kwargs: engine)
        configuration = server.Settings(backend='mock', log_dir=tmp_path / 'logs', **settings)
        return server.create_app(configuration), engine
    return make


def start(**changes):
    data = dict(type='start', source_lang='en', target_lang='zh', chunk_seconds=1,
                adaptive=False, source_kind='file', source_name='test.wav', media_offset_s=0)
    data.update(changes)
    return data


def receive_until(ws, event_type):
    events = []
    for _ in range(200):
        event = ws.receive_json()
        events.append(event)
        if event['type'] == event_type:
            return events
    raise AssertionError(f'No {event_type!r} event in 200 messages')


def connect_ready(ws, **changes):
    ws.send_json(start(**changes))
    events = receive_until(ws, 'ready')
    assert not any(e['type'] == 'error' for e in events)
    return events[-1]


def frame(sequence, offset, pcm):
    return struct.pack('<II', sequence, offset) + np.asarray(pcm, dtype='<i2').tobytes()


def send_audio(ws, pcm, *, sequence=0, offset=0, size=4000):
    for begin in range(0, len(pcm), size):
        piece = pcm[begin:begin + size]
        ws.send_bytes(frame(sequence, offset, piece))
        sequence += 1
        offset += len(piece)
    return sequence, offset


def test_health_config_and_security_headers(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        response = client.get('/api/config')
        assert response.status_code == 200
        config = response.json()
        assert config['backend'] == 'mock' and config['status'] == 'ready'
        assert config['directions'] == ['en-zh', 'zh-en']
        assert config['sample_rate'] == RATE and not config['busy']
        assert response.headers['x-content-type-options'] == 'nosniff'
        assert "frame-ancestors 'none'" in response.headers['content-security-policy']
        health = client.get('/api/health')
        assert health.status_code == 200
        payload = health.json()
        engine_health = payload.pop('engine_health')
        assert engine_health['state'] == 'ready' and engine_health['accepting']
        assert payload == dict(status='ready', backend='mock', busy=False,
                               active_sessions=0, max_sessions=4, available_sessions=4)


@pytest.mark.parametrize('source,target', [('en', 'zh'), ('zh', 'en')])
def test_both_checkpoint_directions_have_isolated_sessions(app_factory, source, target):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws, source_lang=source, target_lang=target)
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['reason'] == 'finished' and done['translation'] == ''
        assert engine.sessions[0].language == target
        assert engine.sessions[0].generated == []


def test_web_direction_switch_selects_real_session_prompt(tmp_path, monkeypatch):
    # Use the actual ModelSession with the mock backend: verify Web routing and
    # session state together without loading weights or claiming translation quality.
    engine = server.ModelEngine(backend='mock')
    sessions = []
    new_session = engine.new_session

    def record_session(target):
        session = new_session(target)
        sessions.append(session)
        return session

    monkeypatch.setattr(engine, 'new_session', record_session)
    monkeypatch.setattr(server, 'ModelEngine', lambda *args, **kwargs: engine)
    app = server.create_app(server.Settings(backend='mock', log_dir=tmp_path / 'logs'))
    with TestClient(app) as client:
        # Startup warmup creates its own disposable session before any client.
        sessions.clear()
        for source, target, prompt_language in [('en', 'zh', '中文'), ('zh', 'en', 'English')]:
            with client.websocket_connect('/ws') as ws:
                ready = connect_ready(ws, source_kind='microphone', source_lang=source, target_lang=target)
                assert ready['config']['source_lang'] == source
                assert ready['config']['target_lang'] == target
                session = sessions[-1]
                assert session.system_prompt.startswith(
                    f'你是一个同声传译专家，负责将用户音频中的语音同声翻译为{prompt_language}。')
                assert session.total_committed_turns == 0
                assert session.audios == []
                send_audio(ws, np.zeros(RATE, dtype=np.int16))
                ws.send_json({'type': 'finish'})
                assert receive_until(ws, 'done')[-1]['reason'] == 'finished'
        assert sessions[0] is not sessions[1]
        assert sessions[0].system_prompt != sessions[1].system_prompt


@pytest.mark.parametrize('changes,code', [
    ({'source_lang': 'en', 'target_lang': 'en'}, 'direction'),
    ({'source_lang': 'ja'}, 'direction'),
    ({'source_kind': 'upload'}, 'bad_start'),
    ({'chunk_seconds': True}, 'bad_start'),
    ({'chunk_seconds': '1'}, 'bad_start'),
    ({'chunk_seconds': .2}, 'bad_start'),
    ({'media_offset_s': -1}, 'bad_start'),
    ({'adaptive': 'false'}, 'bad_start'),
])
def test_invalid_start_never_allocates_model_session(app_factory, changes, code):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json(start(**changes))
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == code
        assert not any(e['type'] == 'ready' for e in events)
        assert not engine.sessions


def test_binary_audio_cannot_start_a_session(app_factory):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_bytes(frame(0, 0, [1, 2]))
        events = receive_until(ws, 'done')
        assert events[0]['code'] == 'bad_start'
        assert not engine.sessions


def test_access_token_is_required_and_not_exposed_by_config_or_log(app_factory, tmp_path):
    secret = 'private-test-secret'
    app, _ = app_factory(token=secret)
    with TestClient(app) as client:
        assert client.get('/api/config').json()['auth_required'] is True
        assert secret not in client.get('/api/config').text
        with client.websocket_connect('/ws') as ws:
            ws.send_json(start())
            assert receive_until(ws, 'done')[0]['code'] == 'unauthorized'
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, token=secret)
            ws.send_json({'type': 'finish'})
            receive_until(ws, 'done')
    for path in (tmp_path / 'logs').glob('*.jsonl'):
        assert secret not in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('origin', ['https://attacker.example', 'http://[broken'])
def test_cross_origin_socket_rejected_before_accept(app_factory, origin):
    app, engine = app_factory()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect('/ws', headers={'origin': origin}):
                pass
        assert exc.value.code == 1008
        assert not engine.sessions


@pytest.mark.parametrize('origin,allowed', [('http://testserver', ()),
                                         ('https://translation.example', ('https://translation.example',))])
def test_same_or_explicitly_allowed_origin(app_factory, origin, allowed):
    app, _ = app_factory(allowed_origins=allowed)
    with TestClient(app) as client, client.websocket_connect('/ws', headers={'origin': origin}) as ws:
        connect_ready(ws)
        ws.send_json({'type': 'finish'})
        assert receive_until(ws, 'done')[-1]['reason'] == 'finished'


def test_only_one_client_owns_single_gpu(app_factory):
    app, engine = app_factory(max_sessions=1)
    with TestClient(app) as client, client.websocket_connect('/ws') as first:
        connect_ready(first)
        assert client.get('/api/health').json()['busy']
        with client.websocket_connect('/ws') as second:
            second.send_json(start())
            assert receive_until(second, 'done')[0]['code'] == 'busy'
        assert len(engine.sessions) == 1
        first.send_json({'type': 'finish'})
        receive_until(first, 'done')


@pytest.mark.parametrize('packet,code', [
    (b'', 'bad_frame'),
    (struct.pack('<II', 0, 0), 'bad_frame'),
    (struct.pack('<II', 0, 0) + b'\x01', 'bad_frame'),
    (frame(0, 0, [0] * 8001), 'bad_frame'),
    (frame(1, 0, [0]), 'sequence_gap'),
    (frame(0, 1, [0]), 'sequence_gap'),
], ids=['empty', 'header_only', 'odd_bytes', 'oversize', 'sequence', 'offset'])
def test_invalid_pcm_packet_stops_without_translation(app_factory, packet, code):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        ws.send_bytes(packet)
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == code
        assert events[-1]['processed_audio_s'] == 0
        assert not engine.sessions[0].committed


def test_eof_tail_has_exact_source_coverage_and_safe_padding(app_factory, tmp_path):
    app, engine = app_factory()
    original = ((np.arange(17600) * 13) % 30000 - 15000).astype(np.int16)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, media_offset_s=37.5)
        send_audio(ws, original)
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        turns = [event for event in events if event['type'] == 'turn']
        assert [t['audio_seconds'] for t in turns] == pytest.approx([1, .1])
        assert [t['source_start_s'] for t in turns] == pytest.approx([0, 1])
        assert [t['source_end_s'] for t in turns] == pytest.approx([1, 1.1])
        assert all(t['media_offset_s'] == 37.5 for t in turns)
        assert turns[-1]['model_audio_seconds'] == 1
        done = events[-1]
        assert done['received_audio_s'] == done['processed_audio_s'] == pytest.approx(1.1)
        assert done['unprocessed_audio_s'] == 0 and done['pending_audio_s'] == 0
        combined = np.concatenate([audio[:round(turn['audio_seconds'] * RATE)]
                                   for (audio, _), turn in zip(engine.sessions[0].committed, turns)])
        np.testing.assert_array_equal(combined, original.astype(np.float32) / 32768.0)
    journal = tmp_path / 'logs' / f"{ready['session_id']}.jsonl"
    saved = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
    assert saved[0]['type'] == 'ready' and saved[-1]['type'] == 'done'
    assert len([event for event in saved if event['type'] == 'turn']) == 2


def test_wait_flex_and_direct_preserve_append_only_deltas(app_factory):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(3 * RATE, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        turns = [event for event in events if event['type'] == 'turn']
        assert [t['tag'] for t in turns] == ['WAIT', 'FLEX', 'DIRECT']
        assert [t['delta'] for t in turns] == ['', 'A safe beginning', ' and an ending.']
        assert events[-1]['translation'] == 'A safe beginning and an ending.'
        assert len(engine.sessions[0].committed) == 3  # WAIT is committed once, never requeued.


def test_partial_chunk_waits_for_arrival_or_eof_without_fake_model_turn(app_factory):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(RATE // 4, dtype=np.int16))
        ws.send_json({'type': 'ping'})
        receive_until(ws, 'pong')
        assert not engine.sessions[0].generated
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        turns = [event for event in events if event['type'] == 'turn']
        assert len(turns) == 1
        assert turns[0]['source_end_s'] == .25
        assert len(engine.sessions[0].committed) == 1


@pytest.mark.parametrize('control', ['{invalid', '[]', '{"type":"unknown"}', 'x' * 4097],
                         ids=['bad_json', 'array', 'unknown', 'too_large'])
def test_invalid_control_is_reported_as_client_error(app_factory, control):
    app, _ = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        ws.send_text(control)
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == 'bad_control'


@pytest.mark.parametrize('reply,finish', [('<|DIRECT|>partial', 'length'),
                                       ('No protocol tag', 'stop'),
                                       ('<|WAIT|>hidden translation', 'stop')])
def test_invalid_or_truncated_model_output_is_never_committed(app_factory, reply, finish):
    app, engine = app_factory(engine_options=dict(replies=[reply], finish_reason=finish))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(RATE, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == 'model_protocol'
        assert not any(e['type'] == 'turn' for e in events)
        assert events[-1]['translation'] == ''
        assert events[-1]['unprocessed_audio_s'] == 1
        assert not engine.sessions[0].committed


def test_context_limit_is_explicit_and_preserves_committed_prefix(app_factory):
    app, engine = app_factory(engine_options=dict(replies=['<|FLEX|>Committed'], fail_at=1))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(2 * RATE, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == 'context_limit'
        assert events[-1]['translation'] == 'Committed'
        assert events[-1]['processed_audio_s'] == 1
        assert events[-1]['unprocessed_audio_s'] == 1
        assert len(engine.sessions[0].committed) == 1


def test_cancel_during_generation_retains_gpu_slot_until_worker_exits(app_factory):
    app, engine = app_factory(engine_options=dict(block=True), max_sessions=1)
    with TestClient(app) as client, client.websocket_connect('/ws') as first:
        connect_ready(first)
        send_audio(first, np.zeros(RATE, dtype=np.int16))
        assert engine.entered.wait(2)
        first.send_json({'type': 'cancel'})
        with client.websocket_connect('/ws') as second:
            second.send_json(start())
            assert receive_until(second, 'done')[0]['code'] == 'busy'
        engine.release.set()
        events = receive_until(first, 'done')
        assert events[-1]['reason'] == 'cancelled'
        assert not any(e['type'] == 'turn' for e in events)
        assert not engine.sessions[0].committed


def test_disconnect_during_generation_does_not_release_gpu_early(app_factory):
    app, engine = app_factory(engine_options=dict(block=True), max_sessions=1)
    with TestClient(app) as client, client.websocket_connect('/ws') as first:
        connect_ready(first)
        send_audio(first, np.zeros(RATE, dtype=np.int16))
        assert engine.entered.wait(2)
        first.close()
        with client.websocket_connect('/ws') as second:
            second.send_json(start())
            assert receive_until(second, 'done')[0]['code'] == 'busy'
        engine.release.set()

        async def wait_released():
            for _ in range(100):
                if not app.state.active_sessions:
                    return
                await asyncio.sleep(.01)
            raise AssertionError('GPU session slot was not released after generation completed')

        client.portal.call(wait_released)
        assert not engine.sessions[0].committed
        with client.websocket_connect('/ws') as third:
            connect_ready(third)
            third.send_json({'type': 'finish'})
            assert receive_until(third, 'done')[-1]['reason'] == 'finished'


def test_builtin_mock_engine_exercises_real_adapter_without_gpu(tmp_path):
    app = server.create_app(server.Settings(backend='mock', log_dir=tmp_path))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_lang='zh', target_lang='en')
        assert ready['backend'] == 'mock'
        send_audio(ws, np.zeros(3 * RATE, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        turns = [event for event in events if event['type'] == 'turn']
        assert [turn['tag'] for turn in turns] == ['FLEX', 'WAIT', 'DIRECT']
        assert turns[1]['delta'] == ''
        assert turns[2]['delta'].startswith(' ')
        assert 'not a translation' in events[-1]['translation']


def test_disconnect_after_finish_cancels_draining_before_commit(app_factory):
    app, engine = app_factory(engine_options=dict(block=True))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(RATE, dtype=np.int16))
        assert engine.entered.wait(2)
        ws.send_json({'type': 'finish'})
        ws.close()

        async def let_disconnect_arrive():
            await asyncio.sleep(.05)
        client.portal.call(let_disconnect_arrive)
        engine.release.set()

        async def wait_released():
            for _ in range(100):
                if not app.state.active_sessions:
                    return
                await asyncio.sleep(.01)
            raise AssertionError('GPU slot not released after EOF disconnect')

        client.portal.call(wait_released)
        assert not engine.sessions[0].committed


def test_audio_after_finish_is_rejected_during_drain(app_factory, monkeypatch):
    original_push = server.PCMQueue.push
    rejected = threading.Event()

    def observe_push(queue, packet):
        if queue.closed:
            rejected.set()
        return original_push(queue, packet)

    monkeypatch.setattr(server.PCMQueue, 'push', observe_push)
    app, engine = app_factory(engine_options=dict(block=True))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        sequence, offset = send_audio(ws, np.zeros(RATE, dtype=np.int16))
        assert engine.entered.wait(2)
        ws.send_json({'type': 'finish'})
        ws.send_bytes(frame(sequence, offset, [0]))
        assert rejected.wait(2)
        engine.release.set()
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == 'input_closed'
        assert not engine.sessions[0].committed


def test_eof_wait_is_diagnostic_not_a_fake_flush_turn(app_factory):
    app, engine = app_factory(engine_options=dict(replies=['<|WAIT|>']))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(RATE, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        turns = [event for event in events if event['type'] == 'turn']
        assert len(turns) == 1 and len(engine.sessions[0].generated) == 1
        assert events[-1]['last_tag'] == 'WAIT' and events[-1]['final_wait'] is True
        assert events[-1]['translation'] == ''


def test_warmup_failure_closes_engine_and_executor(app_factory):
    app, engine = app_factory()
    lifecycle_threads = []

    def warmup(seconds):
        lifecycle_threads.append(threading.get_ident())
        raise RuntimeError('simulated warmup OOM')

    def close():
        lifecycle_threads.append(threading.get_ident())
        engine.closed = True

    engine.warmup, engine.close = warmup, close
    with pytest.raises(RuntimeError, match='simulated warmup OOM'):
        with TestClient(app):
            pass
    assert engine.closed
    assert lifecycle_threads[0] == lifecycle_threads[1] != threading.get_ident()
    assert app.state.status == 'stopped' and app.state.executor._shutdown


def test_engine_constructor_failure_still_shuts_down_executor(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('simulated checkpoint load failure')
    monkeypatch.setattr(server, 'ModelEngine', fail)
    app = server.create_app(server.Settings(backend='mock', log_dir=tmp_path))
    with pytest.raises(RuntimeError, match='simulated checkpoint load failure'):
        with TestClient(app):
            pass
    assert app.state.status == 'stopped' and app.state.executor._shutdown


@pytest.mark.parametrize('fail_event', ['ready', 'status', 'turn'])
def test_journal_write_failure_is_visible_and_gpu_is_released(app_factory, monkeypatch, fail_event):
    original_log = server.SessionLog
    instances = []

    class FailingLog:
        def __init__(self, directory, session_id):
            self.closed = False
            instances.append(self)

        def write(self, event):
            if event['type'] in (fail_event, 'error', 'done'):
                raise OSError('simulated disk full')

        def close(self):
            self.closed = True

    monkeypatch.setattr(server, 'SessionLog', FailingLog)
    app, engine = app_factory(engine_options=dict(replies=['<|DIRECT|>Validated']))
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            ws.send_json(start())
            if fail_event != 'ready':
                receive_until(ws, 'ready')
                send_audio(ws, np.zeros(RATE, dtype=np.int16))
                ws.send_json({'type': 'finish'})
            events = receive_until(ws, 'done')
            assert next(e for e in events if e['type'] == 'error')['code'] == 'server_error'
            assert events[-1]['reason'] == 'server_error'
            if fail_event != 'turn':
                assert not engine.sessions[0].committed
        assert instances[0].closed
        assert not client.get('/api/health').json()['busy']
        monkeypatch.setattr(server, 'SessionLog', original_log)
        with client.websocket_connect('/ws') as second:
            connect_ready(second)
            second.send_json({'type': 'finish'})
            assert receive_until(second, 'done')[-1]['reason'] == 'finished'


def test_journal_creation_failure_does_not_lock_gpu(app_factory, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError('simulated read-only log directory')
    monkeypatch.setattr(server, 'SessionLog', fail)
    app, engine = app_factory()
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            ws.send_json(start())
            events = receive_until(ws, 'done')
            assert events[0]['code'] == 'server_error'
            assert not engine.sessions
        assert not client.get('/api/health').json()['busy']


def test_journal_close_failure_does_not_interrupt_socket_cleanup(app_factory, monkeypatch):
    class CloseFailingLog:
        def __init__(self, *args):
            pass

        def write(self, event):
            pass

        def close(self):
            raise OSError('simulated close flush failure')

    monkeypatch.setattr(server, 'SessionLog', CloseFailingLog)
    app, _ = app_factory()
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws)
            ws.send_json({'type': 'finish'})
            assert receive_until(ws, 'done')[-1]['reason'] == 'finished'
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert not client.get('/api/health').json()['busy']


def test_queue_backpressure_keeps_accepted_samples_and_reports_rejection():
    async def scenario():
        queue = server.PCMQueue(max_pending_seconds=1, max_session_seconds=10)
        queue.push(frame(0, 0, np.arange(8000)))
        queue.push(frame(1, 8000, np.arange(8000, 16000)))
        with pytest.raises(server.ClientError) as exc:
            queue.push(frame(2, 16000, [42]))
        assert exc.value.code == 'backlog_limit'
        assert queue.received == queue.pending == 16000
        assert queue.sequence == 2
        queue.finish()
        offset, audio, _ = await queue.take(16000, 48000, True)
        assert offset == 0
        np.testing.assert_array_equal(audio, np.arange(16000).astype(np.float32) / 32768)
        assert await queue.take(16000, 48000, True) is None
    asyncio.run(scenario())


def test_adaptive_queue_combines_only_arrived_contiguous_samples():
    async def scenario():
        queue = server.PCMQueue(max_pending_seconds=10, max_session_seconds=20)
        for i in range(7):
            queue.push(frame(i, 8000 * i, np.full(8000, i)))
        offset, audio, _ = await queue.take(16000, 48000, True)
        assert offset == 0 and len(audio) == 48000
        assert queue.pending == 8000 and queue.taken == 48000
        queue.finish()
        offset, tail, _ = await queue.take(16000, 48000, True)
        assert offset == 48000 and len(tail) == 8000
        np.testing.assert_array_equal(tail, np.full(8000, 6 / 32768, dtype=np.float32))
    asyncio.run(scenario())


def test_duration_limit_is_checked_before_accepting_packet():
    async def scenario():
        queue = server.PCMQueue(max_pending_seconds=2, max_session_seconds=.5)
        queue.push(frame(0, 0, np.zeros(8000)))
        with pytest.raises(server.ClientError) as exc:
            queue.push(frame(1, 8000, [0]))
        assert exc.value.code == 'session_limit'
        assert queue.received == 8000
    asyncio.run(scenario())


@pytest.mark.parametrize('settings', [dict(chunk_seconds=.1),
                                     dict(max_pending_seconds=1, max_chunk_seconds=3),
                                     dict(max_session_seconds=float('nan')),
                                     dict(directions=('en-ja',)),
                                     dict(backend='unsupported')])
def test_invalid_server_configuration_fails_at_startup(settings):
    values = dict(backend='mock', log_dir=None)
    values.update(settings)
    with pytest.raises(ValueError):
        server.create_app(server.Settings(**values))
