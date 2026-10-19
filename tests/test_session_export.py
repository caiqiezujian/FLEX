import json

import numpy as np
from fastapi.testclient import TestClient

from test_server import app_factory, connect_ready, receive_until, send_audio


def test_exports_are_full_snapshots_with_optional_auth(app_factory):
    app, _ = app_factory(token='test-token')
    auth = {'Authorization': 'Bearer test-token'}
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, token='test-token')
        url = '/api/sessions/' + ready['session_id'] + '/export'
        send_audio(ws, np.zeros(32000, dtype=np.int16))
        receive_until(ws, 'turn')
        receive_until(ws, 'turn')
        assert client.get(url).status_code == 401
        assert client.get(url, headers={**auth, 'Origin': 'https://unrelated.example'}).status_code == 403
        snapshot = client.get(url, headers=auth).json()
        assert snapshot['translation'] == 'A safe beginning'
        assert not snapshot['complete'] and len(snapshot['turns']) == 2
        assert 'test-token' not in json.dumps(snapshot)
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['turn_count'] == 2 and done['results_available']
        text = client.get(url + '?format=txt', headers=auth)
        assert text.text == 'A safe beginning'
        assert 'attachment;' in text.headers['content-disposition']
        assert client.get(url + '?format=xml', headers=auth).status_code == 400
        assert client.get('/api/sessions/missing/export', headers=auth).status_code == 404


def test_long_done_does_not_repeat_whole_transcript_over_websocket(app_factory):
    delta = '字' * 6000
    app, _ = app_factory(engine_options=dict(replies=['<|DIRECT|>' + delta]))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws)
        send_audio(ws, np.zeros(48000, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['translation_chars'] == 18000 and done['translation_truncated']
        assert 'translation' not in done
        result = client.get('/api/sessions/' + ready['session_id'] + '/export').json()
        assert result['translation'] == delta * 3 and len(result['turns']) == 3


def test_no_log_still_allows_export_and_temp_files_are_cleaned(tmp_path):
    from flex_web.server import Settings, create_app
    app = create_app(Settings(backend='mock', log_dir=None, temp_dir=tmp_path))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws)
        send_audio(ws, np.zeros(16000, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        receive_until(ws, 'done')
        response = client.get('/api/sessions/' + ready['session_id'] + '/export?format=txt')
        assert response.status_code == 200 and response.text
    assert not list(tmp_path.iterdir())


def test_result_limit_stops_explicitly_and_preserves_exportable_prefix(app_factory):
    app, _ = app_factory(max_result_bytes=2048, engine_options=dict(replies=['<|DIRECT|>' + 'x' * 2000]))
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws)
        send_audio(ws, np.zeros(16000, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['reason'] == 'result_limit'
        exported = client.get('/api/sessions/' + ready['session_id'] + '/export').json()
        assert exported['translation'] == '' and exported['turns'] == []
