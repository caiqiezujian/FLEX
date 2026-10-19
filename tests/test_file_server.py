"""Uploaded file HTTP + WebSocket integration; no model/GPU is required."""
import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from test_media import wav_bytes
from test_server import app_factory, connect_ready, receive_until, start


def upload(client, samples=17600, **kwargs):
    response = client.post('/api/media?name=sample.wav',
                           content=wav_bytes(np.arange(samples, dtype=np.int16)), **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_auth_origin_and_config(app_factory):
    app, _ = app_factory(token='example-secret')
    with TestClient(app) as client:
        config = client.get('/api/config').json()
        assert config['file_mode'] == 'uploaded'
        assert config['max_upload_bytes'] == 4 * 1024 * 1024 * 1024
        assert client.post('/api/media', content=b'123').status_code == 401
        assert client.post('/api/media', content=b'123',
                           headers={'Authorization': 'Bearer wrong-token'}).status_code == 401
        auth = {'Authorization': 'Bearer example-secret'}
        assert client.post('/api/media', content=b'123',
                           headers={**auth, 'Origin': 'http://attacker.example'}).status_code == 403
        item = upload(client, headers={**auth, 'Origin': 'http://testserver'})
        assert 'example-secret' not in json.dumps(item)
        assert client.delete('/api/media/' + item['media_id']).status_code == 401
        assert client.delete('/api/media/' + item['media_id'],
                             headers={'Authorization': 'Bearer wrong-token'}).status_code == 401
        assert client.delete('/api/media/' + item['media_id'], headers=auth).status_code == 200
        assert not app.state.media.uploads


@pytest.mark.parametrize('token', ['', 'synthetic-intranet-test-token'])
@pytest.mark.parametrize('secure', [False, True])
def test_intranet_upload_ws_and_delete_keep_optional_auth_and_origin_checks(app_factory, token, secure):
    app, engine = app_factory(token=token)
    origin = f'{"https" if secure else "http"}://10.0.0.8:18889'
    socket_url = f'{"wss" if secure else "ws"}://10.0.0.8:18889/ws'
    headers = {'Origin': origin}
    start_auth = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
        start_auth['token'] = token
    with TestClient(app, base_url=origin) as client:
        assert client.get('/api/config').json()['auth_required'] is bool(token)
        hostile = {**headers, 'Origin': 'http://unrelated.example'}
        assert client.post('/api/media', content=b'123', headers=hostile).status_code == 403
        mismatched_scheme = {**headers, 'Origin': f'{"http" if secure else "https"}://10.0.0.8:18889'}
        assert client.post('/api/media', content=b'123', headers=mismatched_scheme).status_code == 403
        item = upload(client, samples=16000, headers=headers)
        assert client.delete('/api/media/' + item['media_id'], headers=hostile).status_code == 403
        assert item['media_id'] in app.state.media.uploads
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(socket_url, headers=hostile):
                pass
        assert rejected.value.code == 1008
        assert not engine.sessions
        with client.websocket_connect(socket_url, headers={'Origin': origin}) as ws:
            connect_ready(ws, media_id=item['media_id'], **start_auth)
            ws.send_json(dict(type='finish', position_s=1))
            done = receive_until(ws, 'done')[-1]
            assert done['reason'] == 'finished' and done['processed_audio_s'] == 1
        assert not app.state.media.uploads
        unused = upload(client, samples=160, headers=headers)
        assert client.delete('/api/media/' + unused['media_id'], headers=headers).status_code == 200
        assert not app.state.media.uploads


@pytest.mark.parametrize('content,code', [(b'', 'empty_upload'), (b'no audio', 'unsupported_media')])
def test_invalid_upload_cleans_temporary_files(app_factory, content, code):
    app, engine = app_factory()
    with TestClient(app) as client:
        response = client.post('/api/media', content=content)
        assert response.status_code == 400
        assert response.json()['detail']['code'] == code
        assert not app.state.media.uploads and not list(app.state.media.root.iterdir())
        assert not engine.sessions


def test_upload_size_and_duration_are_checked_before_model(app_factory):
    app, engine = app_factory(max_upload_bytes=10000, max_session_seconds=.1)
    with TestClient(app) as client:
        response = client.post('/api/media', content=b'x' * 10001)
        assert response.status_code == 413
        response = client.post('/api/media', content=wav_bytes(np.zeros(1601)))
        assert response.status_code == 400
        assert response.json()['detail']['code'] == 'media_duration'
        assert not app.state.media.uploads and not engine.sessions


def test_file_progress_pause_tail_coverage_and_automatic_cleanup(app_factory):
    app, engine = app_factory()
    with TestClient(app) as client:
        item = upload(client)
        assert item['samples'] == 17600
        directory = app.state.media.uploads[item['media_id']].directory
        with client.websocket_connect('/ws') as ws:
            ready = connect_ready(ws, media_id=item['media_id'], adaptive=True)
            assert ready['file_mode'] == 'uploaded' and ready['duration_s'] == 1.1
            assert ready['media_id'] == item['media_id']
            assert not ready['config']['adaptive']
            ws.send_json(dict(type='progress', position_s=.5))
            ws.send_json(dict(type='ping'))
            receive_until(ws, 'pong')
            assert not engine.sessions[0].generated
            ws.send_json(dict(type='progress', position_s=1))
            first = receive_until(ws, 'turn')[-1]
            assert first['source_end_s'] == 1
            ws.send_json(dict(type='progress', position_s=1))  # pause/duplicate is safe
            ws.send_json(dict(type='ping'))
            receive_until(ws, 'pong')
            assert len(engine.sessions[0].generated) == 1
            ws.send_json(dict(type='finish', position_s=1.1))
            events = receive_until(ws, 'done')
            tail = next(e for e in events if e['type'] == 'turn')
            assert tail['audio_seconds'] == .1 and tail['model_audio_seconds'] == 1
            assert events[-1]['processed_audio_s'] == events[-1]['received_audio_s'] == 1.1
            original = np.arange(17600, dtype=np.int16).astype(np.float32) / 32768
            actual = np.concatenate([engine.sessions[0].generated[0], engine.sessions[0].generated[1][:1600]])
            np.testing.assert_array_equal(actual, original)
        assert not app.state.media.uploads and not directory.exists()


def test_delayed_file_progress_exceeds_live_backlog_without_failure(app_factory):
    app, engine = app_factory(max_pending_seconds=3)
    with TestClient(app) as client:
        item = upload(client, samples=160000)
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, media_id=item['media_id'])
            ws.send_json(dict(type='finish', position_s=10))
            events = receive_until(ws, 'done')
            turns = [e for e in events if e['type'] == 'turn']
            assert len(turns) == 10 and all(e['audio_seconds'] == 1 for e in turns)
            assert events[-1]['reason'] == 'finished'
            assert events[-1]['processed_audio_s'] == 10
        assert not app.state.media.uploads


@pytest.mark.parametrize('control,code', [
    ({'type': 'progress', 'position_s': -1}, 'backward_seek'),
    ({'type': 'progress', 'position_s': '2'}, 'bad_progress'),
    ({'type': 'finish'}, 'bad_progress'),
])
def test_bad_progress_has_actionable_error_and_cleans_upload(app_factory, control, code):
    app, engine = app_factory()
    with TestClient(app) as client:
        item = upload(client)
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, media_id=item['media_id'])
            ws.send_json(control)
            events = receive_until(ws, 'done')
            assert next(e for e in events if e['type'] == 'error')['code'] == code
            assert not engine.sessions[0].generated
        assert not app.state.media.uploads


def test_file_offset_and_manual_finish_only_release_observed_interval(app_factory):
    app, engine = app_factory()
    with TestClient(app) as client:
        item = upload(client, samples=32000)
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, media_id=item['media_id'], media_offset_s=.5)
            ws.send_json(dict(type='finish', position_s=1.25))
            events = receive_until(ws, 'done')
            turn = next(e for e in events if e['type'] == 'turn')
            assert turn['source_start_s'] == 0 and turn['source_end_s'] == .75
            assert turn['media_offset_s'] == .5
            expected = np.arange(32000, dtype=np.int16)[8000:20000].astype(np.float32) / 32768
            np.testing.assert_array_equal(engine.sessions[0].generated[0][:12000], expected)


def test_file_cancel_retains_gpu_and_file_until_generation_exits(app_factory):
    app, engine = app_factory(engine_options=dict(block=True))
    with TestClient(app) as client:
        item = upload(client)
        directory = app.state.media.uploads[item['media_id']].directory
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, media_id=item['media_id'])
            ws.send_json(dict(type='progress', position_s=1))
            assert engine.entered.wait(2)
            ws.send_json(dict(type='cancel'))
            with client.websocket_connect('/ws') as second:
                second.send_json(start())
                assert receive_until(second, 'done')[0]['code'] == 'busy'
            assert directory.exists()
            assert client.delete('/api/media/' + item['media_id']).status_code == 409
            engine.release.set()
            events = receive_until(ws, 'done')
            assert events[-1]['reason'] == 'cancelled'
            assert not engine.sessions[0].committed
        assert not directory.exists() and not app.state.media.uploads


def test_uploaded_file_rejects_binary_capture(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        item = upload(client)
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, media_id=item['media_id'])
            ws.send_bytes(b'1234567890')
            assert receive_until(ws, 'done')[0]['code'] == 'bad_frame'


def test_rejected_protocol_output_is_saved_for_diagnosis(app_factory, tmp_path):
    app, engine = app_factory(engine_options=dict(replies=['Missing protocol tag']))
    with TestClient(app) as client:
        item = upload(client)
        with client.websocket_connect('/ws') as ws:
            ready = connect_ready(ws, media_id=item['media_id'])
            ws.send_json(dict(type='finish', position_s=1))
            events = receive_until(ws, 'done')
            assert events[-1]['reason'] == 'model_protocol'
            assert not engine.sessions[0].committed
        saved = [json.loads(line) for line in (tmp_path / 'logs' / (ready['session_id'] + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        rejected = next(e for e in saved if e['type'] == 'rejected_turn')
        assert rejected['response'] == 'Missing protocol tag' and rejected['token_ids'] == [10]


def test_unclaimed_uploads_are_removed_at_shutdown(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        item = upload(client)
        directory = app.state.media.uploads[item['media_id']].directory
        root = app.state.media.root
        assert directory.exists()
    assert not directory.exists() and not root.exists()


def test_file_disconnect_keeps_resources_until_inflight_model_finishes(app_factory):
    app, engine = app_factory(engine_options=dict(block=True))
    with TestClient(app) as client:
        item = upload(client)
        directory = app.state.media.uploads[item['media_id']].directory
        with client.websocket_connect('/ws') as ws:
            connect_ready(ws, media_id=item['media_id'])
            ws.send_json(dict(type='progress', position_s=1))
            assert engine.entered.wait(2)
            ws.close()
            async def tick():
                await asyncio.sleep(.05)
            client.portal.call(tick)
            assert directory.exists() and app.state.active
            engine.release.set()
            async def released():
                for _ in range(100):
                    if app.state.active is None:
                        return
                    await asyncio.sleep(.01)
                raise AssertionError('admission not released')
            client.portal.call(released)
        assert not directory.exists() and not engine.sessions[0].committed
