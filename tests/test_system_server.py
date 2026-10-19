"""System/tab audio uses the existing ordered PCM stream, not screen video."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_server import app_factory, connect_ready, receive_until, send_audio, start


def test_system_audio_capability_and_browser_policy(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client:
        response = client.get('/api/config')
        assert response.json()['input_sources'] == ['file', 'system']
        assert 'display-capture=(self)' in response.headers['permissions-policy']
        assert 'camera=()' in response.headers['permissions-policy']
        assert 'microphone=(self)' in response.headers['permissions-policy']


@pytest.mark.parametrize('chunk', [1, 2, 3])
@pytest.mark.parametrize('include_microphone', [False, True])
def test_system_audio_preserves_chunks_history_and_export(app_factory, chunk, include_microphone):
    app, engine = app_factory()
    pcm = np.arange(6 * 16000, dtype=np.int16)
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            ready = connect_ready(ws, source_kind='system', source_name='computer audio',
                                  chunk_seconds=chunk, include_microphone=include_microphone)
            assert ready['config']['source_kind'] == 'system'
            assert ready['config']['include_microphone'] is include_microphone
            send_audio(ws, pcm)
            ws.send_json({'type': 'finish'})
            events = receive_until(ws, 'done')
            assert not any(e['type'] == 'error' for e in events)
            done = events[-1]
            assert done['reason'] == 'finished'
            assert done['received_audio_s'] == done['processed_audio_s'] == 6
            turns = [e for e in events if e['type'] == 'turn']
            assert len(turns) == 6 // chunk
            assert [len(audio) for audio in engine.sessions[0].generated] == [chunk * 16000] * (6 // chunk)
            assert len(engine.sessions[0].committed) == len(turns)
            np.testing.assert_array_equal(np.concatenate(engine.sessions[0].generated),
                                          pcm.astype(np.float32) / 32768)
        result = client.get(f"/api/sessions/{ready['session_id']}/export?format=json").json()
        assert result['source_kind'] == 'system'
        assert result['include_microphone'] is include_microphone
        assert result['translation'] == ''.join(e['delta'] for e in turns)


@pytest.mark.parametrize('changes', [
    {'include_microphone': 'true'},
    {'include_microphone': 1},
    {'include_microphone': None},
    {'media_offset_s': 1},
    {'media_id': 'a' * 32},
    {'source_kind': 'file', 'include_microphone': True},
    {'source_kind': 'microphone', 'include_microphone': True},
])
def test_invalid_system_metadata_rejected_before_model(app_factory, changes):
    app, engine = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        options = {'source_kind': 'system', **changes}
        ws.send_json(start(**options))
        events = receive_until(ws, 'done')
        assert next(e for e in events if e['type'] == 'error')['code'] == 'bad_start'
        assert not any(e['type'] == 'ready' for e in events)
        assert engine.sessions == []


def test_legacy_microphone_clients_still_work(app_factory):
    app, _ = app_factory()
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_kind='microphone', source_name='legacy client')
        assert ready['config']['source_kind'] == 'microphone'
        ws.send_json({'type': 'finish'})
        assert receive_until(ws, 'done')[-1]['reason'] == 'finished'
