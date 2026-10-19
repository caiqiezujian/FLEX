"""Runtime status is independent of translation strategy; synthetic CPU only."""
import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flex_web import server
from test_server import app_factory, connect_ready, receive_until, send_audio, start


async def wait_health(app, state):
    for _ in range(500):
        snapshot = app.state.scheduler.health()
        if snapshot['state'] == state:
            return snapshot
        await asyncio.sleep(.005)
    raise AssertionError(f'engine health never became {state}')


def test_processing_monitor_does_not_call_wait_or_one_running_chunk_slow():
    clock = [0.0]
    monitor = server.ProcessingMonitor(1, clock=lambda: clock[0])
    for _ in range(1000):
        clock[0] += 1
        assert monitor.update(1)['state'] == 'normal'
        assert monitor.update(0)['state'] == 'normal'
    assert monitor.started is None


def test_processing_monitor_requires_sustained_backlog_and_recovers_with_hysteresis():
    clock = [0.0]
    monitor = server.ProcessingMonitor(1, clock=lambda: clock[0])
    assert monitor.update(4)['state'] == 'normal'
    clock[0] = 4.9
    assert monitor.update(4)['state'] == 'normal'
    clock[0] = 5
    assert monitor.update(4)['state'] == 'slow'
    assert monitor.update(2)['state'] == 'slow'
    assert monitor.update(1.5)['state'] == 'normal'
    clock[0] = 10
    assert monitor.update(5)['state'] == 'normal'
    clock[0] = 12
    assert monitor.update(2)['state'] == 'normal'
    clock[0] = 14
    assert monitor.update(5)['state'] == 'normal'
    clock[0] = 18
    assert monitor.update(5)['state'] == 'normal'
    clock[0] = 19
    assert monitor.update(5)['state'] == 'slow'


def test_processing_monitor_threshold_accounts_for_three_second_chunks():
    clock = [0.0]
    monitor = server.ProcessingMonitor(3, clock=lambda: clock[0])
    for _ in range(20):
        clock[0] += 1
        assert monitor.update(3)['state'] == 'normal'
    assert monitor.update(6)['state'] == 'normal'
    clock[0] += 5
    assert monitor.update(6)['state'] == 'slow'
    assert monitor.update(3)['state'] == 'normal'


@pytest.mark.parametrize('options', [
    {'slow_inference_seconds': 0}, {'slow_inference_seconds': True},
    {'slow_inference_seconds': float('nan')}, {'slow_inference_seconds': '5'},
    {'stalled_inference_seconds': float('inf')}, {'stalled_inference_seconds': -1},
    {'slow_inference_seconds': 5, 'stalled_inference_seconds': 5},
    {'slow_inference_seconds': 10, 'stalled_inference_seconds': 5},
])
def test_runtime_thresholds_are_validated_before_loading_model(options):
    with pytest.raises(ValueError):
        server.Settings(backend='mock', **options).validate()


def test_stalled_worker_keeps_ownership_rejects_queued_and_new_work_then_recovers(app_factory):
    app, engine = app_factory(engine_options={'block': True}, max_batch_size=1,
                              batch_wait_ms=0, slow_inference_seconds=.02,
                              stalled_inference_seconds=.4)
    with TestClient(app) as client, client.websocket_connect('/ws') as first, client.websocket_connect('/ws') as queued:
        a = connect_ready(first, source_kind='microphone')
        b = connect_ready(queued, source_kind='microphone')
        try:
            send_audio(first, np.zeros(16000, dtype=np.int16))
            first.send_json({'type': 'finish'})
            assert engine.entered.wait(2)
            send_audio(queued, np.zeros(16000, dtype=np.int16))
            queued.send_json({'type': 'finish'})
            slow = client.portal.call(wait_health, app, 'slow')
            assert slow['accepting']
            assert not engine.sessions[0].committed
            client.portal.call(wait_health, app, 'stalled')
            status = client.get('/api/health')
            assert status.status_code == 503
            assert status.json()['engine_health']['state'] == 'stalled'
            assert not status.json()['engine_health']['accepting']
            assert client.get('/api/config').status_code == 200
            assert client.get('/api/config').json()['engine_health']['state'] == 'stalled'
            events = receive_until(queued, 'done')
            assert events[-1]['reason'] == 'engine_stalled'
            assert not engine.sessions[1].generated
            assert a['session_id'] in app.state.active_sessions
            assert app.state.scheduler.state(a['session_id']) == 'running'
            assert not engine.sessions[0].committed
            saved = client.get('/api/sessions/' + b['session_id'] + '/export').json()
            assert not saved['complete'] and saved['translation'] == ''
            with client.websocket_connect('/ws') as rejected:
                rejected.send_json(start())
                failure = receive_until(rejected, 'done')
                assert failure[0]['code'] == 'engine_stalled'
                assert not any(x['type'] == 'ready' for x in failure)
            # Existing session receives status in the same page, not fake EOF.
            for _ in range(5):
                event = receive_until(first, 'status')[-1]
                if event['engine_health']['state'] == 'stalled':
                    break
            else:
                raise AssertionError('missing stalled status on running session')
        finally:
            engine.release.set()
        done = receive_until(first, 'done')[-1]
        assert done['reason'] == 'finished' and done['last_tag'] == 'WAIT'
        assert done['processing_health']['state'] == 'normal'
        assert len(engine.sessions[0].committed) == 1
        assert client.get('/api/health').status_code == 200
        assert client.get('/api/health').json()['engine_health']['state'] == 'ready'
        with client.websocket_connect('/ws') as next_session:
            connect_ready(next_session)
            next_session.send_json({'type': 'finish'})
            assert receive_until(next_session, 'done')[-1]['reason'] == 'finished'


def test_shared_backend_failure_is_visible_and_retains_previous_translation(app_factory):
    app, engine = app_factory(engine_options={'replies': ['<|DIRECT|> safe prefix']}, batch_wait_ms=0)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_kind='microphone')
        sequence, offset = send_audio(ws, np.zeros(16000, dtype=np.int16))
        assert receive_until(ws, 'turn')[-1]['delta'] == ' safe prefix'

        def failed_batch(items):
            raise RuntimeError('synthetic shared GPU backend failure')

        engine.generate_batch = failed_batch
        send_audio(ws, np.zeros(16000, dtype=np.int16), sequence=sequence, offset=offset)
        ws.send_json({'type': 'finish'})
        events = receive_until(ws, 'done')
        assert events[-1]['reason'] == 'engine_failed'
        assert not any(event['type'] == 'turn' for event in events)
        assert len(engine.sessions[0].committed) == 1
        health = client.get('/api/health')
        assert health.status_code == 503
        assert health.json()['engine_health']['state'] == 'failed'
        exported = client.get('/api/sessions/' + ready['session_id'] + '/export').json()
        assert exported['translation'] == ' safe prefix' and not exported['complete']
        with client.websocket_connect('/ws') as rejected:
            rejected.send_json(start())
            assert receive_until(rejected, 'done')[0]['code'] == 'engine_failed'
        assert len(engine.sessions) == 1


def test_private_protocol_failure_does_not_poison_shared_model(app_factory):
    app, engine = app_factory(engine_options={'replies': ['bad protocol']}, batch_wait_ms=0)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws)
        send_audio(ws, np.zeros(16000, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        assert receive_until(ws, 'done')[-1]['reason'] == 'model_protocol'
        assert client.get('/api/health').status_code == 200
        engine.replies = ['<|DIRECT|> new session works']
        with client.websocket_connect('/ws') as next_session:
            connect_ready(next_session)
            send_audio(next_session, np.zeros(16000, dtype=np.int16))
            next_session.send_json({'type': 'finish'})
            done = receive_until(next_session, 'done')[-1]
            assert done['reason'] == 'finished'
            assert done['translation'] == ' new session works'


def test_repeated_wait_is_healthy_with_no_new_translation(app_factory):
    app, engine = app_factory(engine_options={'replies': ['<|WAIT|>']}, batch_wait_ms=0)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        connect_ready(ws, source_kind='microphone')
        sequence = offset = 0
        for turn in range(100):
            sequence, offset = send_audio(ws, np.zeros(16000, dtype=np.int16),
                                          sequence=sequence, offset=offset)
            event = receive_until(ws, 'turn')[-1]
            assert event['tag'] == 'WAIT' and event['delta'] == ''
            assert event['engine_health']['state'] == 'ready'
            assert event['processing_health']['state'] == 'normal'
            assert event['processed_audio_s'] == turn + 1
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['reason'] == 'finished' and done['final_wait']
        assert done['translation'] == '' and len(engine.sessions[0].committed) == 100
        assert client.get('/api/health').status_code == 200


def test_input_backlog_error_is_reported_before_running_worker_finishes(app_factory):
    app, engine = app_factory(engine_options={'block': True}, max_sessions=1,
                              max_pending_seconds=3, batch_wait_ms=0)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_kind='microphone')
        sequence, offset = send_audio(ws, np.zeros(16000, dtype=np.int16))
        try:
            assert engine.entered.wait(2)
            send_audio(ws, np.zeros(52000, dtype=np.int16), sequence=sequence, offset=offset)
            error = receive_until(ws, 'error')[-1]
            assert error['code'] == 'backlog_limit'
            assert error['in_flight']
            assert error['engine_health']['state'] == 'ready'
            assert ready['session_id'] in app.state.active_sessions
            assert app.state.scheduler.state(ready['session_id']) == 'running'
            assert not engine.sessions[0].committed
            with client.websocket_connect('/ws') as new_client:
                new_client.send_json(start())
                assert receive_until(new_client, 'done')[0]['code'] == 'busy'
        finally:
            engine.release.set()
        events = receive_until(ws, 'done')
        assert events[-1]['reason'] == 'backlog_limit'
        assert not any(event['type'] in ('turn', 'error') for event in events)
        assert not engine.sessions[0].committed
