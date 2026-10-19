"""Four independent WebSocket clients sharing one synthetic model, no GPU."""
import asyncio
from contextlib import ExitStack

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flex_web import server
from test_media import wav_bytes
from test_server import app_factory, connect_ready, receive_until, send_audio, start


async def wait_state(app, session_id, expected):
    for _ in range(200):
        if app.state.scheduler.state(session_id) == expected:
            return
        await asyncio.sleep(.005)
    raise AssertionError(f'{session_id} never reached scheduler state {expected}')


async def wait_count(app, count):
    for _ in range(200):
        if len(app.state.active_sessions) == count:
            return
        await asyncio.sleep(.005)
    raise AssertionError(f'session count never reached {count}')


def install_batch_audit(engine, bad_session=None):
    batches = []

    def generate_batch(items):
        indices = [engine.sessions.index(session) for session, _ in items]
        batches.append(indices)
        results = []
        for index, (session, audio) in zip(indices, items):
            try:
                result = session.generate(audio)
                result.reply = ('INVALID' if index == bad_session else
                                f'<|DIRECT|>{session.language}:{index}:{len(session.generated)};')
                results.append(result)
            except Exception as exc:
                results.append(exc)
        return results

    engine.generate_batch = generate_batch
    return batches


@pytest.mark.parametrize('source_kind', ['microphone', 'file'])
def test_four_clients_have_independent_audio_history_exports_and_capacity(app_factory, source_kind):
    app, engine = app_factory(batch_wait_ms=100)
    batches = install_batch_audit(engine)
    with TestClient(app) as client, ExitStack() as stack:
        sockets, ready, audio = [], [], []
        for i in range(4):
            samples = np.full(32000, 1000 * (i + 1), dtype=np.int16)
            audio.append(samples)
            extra = {}
            if source_kind == 'file':
                response = client.post('/api/media?name=synthetic.wav', content=wav_bytes(samples))
                assert response.status_code == 200
                extra['media_id'] = response.json()['media_id']
            ws = stack.enter_context(client.websocket_connect('/ws'))
            sockets.append(ws)
            language = 'zh' if i % 2 == 0 else 'en'
            ready.append(connect_ready(ws, source_kind=source_kind, target_lang=language,
                                       source_lang='en' if language == 'zh' else 'zh', **extra))
            assert ready[-1]['active_sessions'] == i + 1
            assert ready[-1]['max_sessions'] == ready[-1]['max_batch_size'] == 4
            assert ready[-1]['available_sessions'] == 3 - i
        ids = [event['session_id'] for event in ready]
        assert len(set(ids)) == 4
        assert client.get('/api/health').json()['busy']
        with client.websocket_connect('/ws') as fifth:
            fifth.send_json(start())
            rejected = receive_until(fifth, 'done')
            assert rejected[0]['code'] == 'busy'
            assert rejected[0]['active_sessions'] == 4 and rejected[0]['available_sessions'] == 0
        assert len(engine.sessions) == 4
        for i, ws in enumerate(sockets):
            if source_kind == 'file':
                ws.send_json({'type': 'finish', 'position_s': 2, 'ended': True})
            else:
                send_audio(ws, audio[i])
                ws.send_json({'type': 'finish'})
        for i, ws in enumerate(sockets):
            events = receive_until(ws, 'done')
            turns = [event for event in events if event['type'] == 'turn']
            assert len(turns) == 2
            assert all(event['session_id'] == ids[i] for event in events)
            assert all(event['scheduler_wait_s'] >= 0 and event['service_s'] >= 0 for event in turns)
            assert all(1 <= event['batch_size'] <= 4 for event in turns)
            assert events[-1]['reason'] == 'finished'
            language = 'zh' if i % 2 == 0 else 'en'
            expected = f'{language}:{i}:1;{language}:{i}:2;'
            assert events[-1]['translation'] == expected
            exported = client.get(f'/api/sessions/{ids[i]}/export').json()
            assert exported['complete'] and exported['translation'] == expected
            assert exported['session_id'] == ids[i]
            assert len(exported['turns']) == 2
            actual = np.concatenate(engine.sessions[i].generated)
            np.testing.assert_array_equal(actual, audio[i].astype(np.float32) / 32768)
            assert len(engine.sessions[i].committed) == 2
        client.portal.call(wait_count, app, 0)
        assert not client.get('/api/health').json()['busy']
        assert sorted(index for batch in batches for index in batch) == [0, 0, 1, 1, 2, 2, 3, 3]
        assert any(len(batch) > 1 for batch in batches)
        assert not app.state.media.uploads


def test_one_bad_protocol_session_does_not_stop_other_three(app_factory):
    app, engine = app_factory(batch_wait_ms=100)
    install_batch_audit(engine, bad_session=1)
    with TestClient(app) as client, ExitStack() as stack:
        sockets = [stack.enter_context(client.websocket_connect('/ws')) for _ in range(4)]
        ready = [connect_ready(ws, source_kind='microphone') for ws in sockets]
        for ws in sockets:
            send_audio(ws, np.zeros(16000, np.int16))
            ws.send_json({'type': 'finish'})
        for i, ws in enumerate(sockets):
            events = receive_until(ws, 'done')
            assert events[-1]['reason'] == ('model_protocol' if i == 1 else 'finished')
            assert len(engine.sessions[i].committed) == (0 if i == 1 else 1)
            snapshot = client.get(f"/api/sessions/{ready[i]['session_id']}/export").json()
            assert snapshot['complete'] is (i != 1)
        client.portal.call(wait_count, app, 0)


def test_four_real_mock_adapter_sessions_roll_context_independently(tmp_path):
    options = server.ModelOptions(history_window_seconds=4, history_retain_seconds=2)
    app = server.create_app(server.Settings(backend='mock', log_dir=None, temp_dir=tmp_path,
                                           model_options=options, batch_wait_ms=0))
    with TestClient(app) as client, ExitStack() as stack:
        sockets = [stack.enter_context(client.websocket_connect('/ws')) for _ in range(4)]
        ready = [connect_ready(ws, source_kind='microphone', source_lang='en' if i % 2 == 0 else 'zh',
                               target_lang='zh' if i % 2 == 0 else 'en') for i, ws in enumerate(sockets)]
        sequences = [0] * 4
        for turn in range(9):
            for i, ws in enumerate(sockets):
                sequences[i], _ = send_audio(ws, np.full(16000, i + 1, np.int16),
                                             sequence=sequences[i], offset=turn * 16000)
            for i, ws in enumerate(sockets):
                event = receive_until(ws, 'turn')[-1]
                assert event['session_id'] == ready[i]['session_id']
                assert event['turn_id'] == turn
                assert event['tag'] == ('FLEX', 'WAIT', 'DIRECT')[turn % 3]
                context = event['context']
                assert context['total_committed_turns'] == turn + 1
                assert context['total_committed_audio_seconds'] == turn + 1
                assert context['retained_audio_seconds'] <= 4
                if turn >= 4:
                    assert context['rollovers'] >= 1
        for i, ws in enumerate(sockets):
            ws.send_json({'type': 'finish'})
            assert receive_until(ws, 'done')[-1]['turn_count'] == 9
            export = client.get(f"/api/sessions/{ready[i]['session_id']}/export").json()
            assert export['turn_count'] == 9
            assert ('【演示】' in export['translation']) is (i % 2 == 0)
    assert not list(tmp_path.iterdir())


def test_cancel_queued_client_frees_only_its_slot_without_waiting_for_gpu(app_factory):
    app, engine = app_factory(engine_options={'block': True}, max_batch_size=1, batch_wait_ms=0)
    with TestClient(app) as client, ExitStack() as stack:
        sockets = [stack.enter_context(client.websocket_connect('/ws')) for _ in range(4)]
        ready = [connect_ready(ws, source_kind='microphone') for ws in sockets]
        send_audio(sockets[0], np.zeros(16000, np.int16))
        assert engine.entered.wait(2)
        send_audio(sockets[1], np.zeros(16000, np.int16))
        client.portal.call(wait_state, app, ready[1]['session_id'], 'queued')
        sockets[1].send_json({'type': 'cancel'})
        cancelled = receive_until(sockets[1], 'done')[-1]
        assert cancelled['reason'] == 'cancelled' and cancelled['processed_audio_s'] == 0
        client.portal.call(wait_count, app, 3)
        assert ready[0]['session_id'] in app.state.active_sessions
        assert not engine.sessions[1].generated and not engine.sessions[1].committed
        with client.websocket_connect('/ws') as replacement:
            connect_ready(replacement)
            replacement.send_json({'type': 'finish'})
            assert receive_until(replacement, 'done')[-1]['reason'] == 'finished'
        engine.release.set()
        for i in (0, 2, 3):
            sockets[i].send_json({'type': 'finish'})
            assert receive_until(sockets[i], 'done')[-1]['reason'] == 'finished'
        client.portal.call(wait_count, app, 0)


def test_cancel_running_client_keeps_its_slot_until_shared_gpu_call_returns(app_factory):
    app, engine = app_factory(engine_options={'block': True}, max_batch_size=1, batch_wait_ms=0)
    with TestClient(app) as client, ExitStack() as stack:
        sockets = [stack.enter_context(client.websocket_connect('/ws')) for _ in range(4)]
        ready = [connect_ready(ws, source_kind='microphone') for ws in sockets]
        send_audio(sockets[0], np.zeros(16000, np.int16))
        assert engine.entered.wait(2)
        sockets[0].send_json({'type': 'cancel'})
        with client.websocket_connect('/ws') as fifth:
            fifth.send_json(start())
            assert receive_until(fifth, 'done')[0]['code'] == 'busy'
        assert ready[0]['session_id'] in app.state.active_sessions
        engine.release.set()
        assert receive_until(sockets[0], 'done')[-1]['reason'] == 'cancelled'
        client.portal.call(wait_count, app, 3)
        assert not engine.sessions[0].committed
        with client.websocket_connect('/ws') as replacement:
            connect_ready(replacement)
            replacement.send_json({'type': 'finish'})
            assert receive_until(replacement, 'done')[-1]['reason'] == 'finished'
        for ws in sockets[1:]:
            ws.send_json({'type': 'finish'})
            assert receive_until(ws, 'done')[-1]['reason'] == 'finished'
        client.portal.call(wait_count, app, 0)


@pytest.mark.parametrize('settings', [
    {'max_sessions': 0}, {'max_sessions': True}, {'max_sessions': 1.5},
    {'max_batch_size': 0}, {'max_batch_size': 5}, {'max_batch_size': True},
    {'batch_wait_ms': -1}, {'batch_wait_ms': 101}, {'batch_wait_ms': float('nan')},
])
def test_invalid_concurrency_limits_fail_before_model_load(settings):
    with pytest.raises(ValueError):
        server.create_app(server.Settings(backend='mock', **settings))


def test_explicit_serial_batch_mode_still_admits_four_sessions(app_factory):
    app, _ = app_factory(max_batch_size=1)
    with TestClient(app) as client:
        config = client.get('/api/config').json()
        assert config['max_sessions'] == 4 and config['max_batch_size'] == 1
        assert app.state.model_options.max_batch_size == 1
