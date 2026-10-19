"""Independent lifecycle regressions for long sessions and partial exports.

All audio and model outputs here are synthetic; no checkpoint is loaded.
"""
import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flex_web.results import ResultStore
from flex_web.server import PCMQueue, RATE, Settings
from test_server import app_factory, connect_ready, frame, receive_until, send_audio


async def wait_for_release(app):
    for _ in range(100):
        if app.state.active is None:
            return
        await asyncio.sleep(.01)
    raise AssertionError('session resources were not released')


def test_one_microphone_session_outlives_old_duration_and_turn_limits(app_factory):
    app, engine = app_factory(engine_options={'replies': ['<|DIRECT|> x']})
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_kind='microphone')
        url = '/api/sessions/' + ready['session_id'] + '/export'
        sequence = offset = 0
        for index in range(301):
            sequence, offset = send_audio(ws, np.zeros(RATE, dtype=np.int16),
                                          sequence=sequence, offset=offset)
            event = receive_until(ws, 'turn')[-1]
            assert event['turn_id'] == index
            assert event['source_start_s'] == index
            assert event['source_end_s'] == index + 1
            if index == 199:
                snapshot = client.get(url).json()
                assert not snapshot['complete']
                assert snapshot['translation'] == ' x' * 200
                assert len(snapshot['turns']) == 200
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['reason'] == 'finished'
        assert done['received_audio_s'] == done['processed_audio_s'] == 301
        assert done['unprocessed_audio_s'] == 0
        assert len(engine.sessions) == 1 and len(engine.sessions[0].committed) == 301
        client.portal.call(wait_for_release, app)
        exported = client.get(url).json()
        assert exported['complete']
        assert exported['translation'] == ' x' * 301
        assert exported['turn_count'] == len(exported['turns']) == 301


def test_cancelled_inflight_turn_is_absent_from_export(app_factory):
    app, engine = app_factory(engine_options={'replies': ['<|DIRECT|> first', '<|DIRECT|> uncommitted']})
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_kind='microphone')
        sequence, offset = send_audio(ws, np.zeros(RATE, dtype=np.int16))
        receive_until(ws, 'turn')
        engine.block = True
        engine.entered.clear()
        send_audio(ws, np.zeros(RATE, dtype=np.int16), sequence=sequence, offset=offset)
        assert engine.entered.wait(2)
        ws.send_json({'type': 'cancel'})
        async def process_cancel():
            await asyncio.sleep(.03)
        client.portal.call(process_cancel)
        assert app.state.active == ready['session_id']
        engine.release.set()
        done = receive_until(ws, 'done')[-1]
        assert done['reason'] == 'cancelled'
        client.portal.call(wait_for_release, app)
        exported = client.get('/api/sessions/' + ready['session_id'] + '/export').json()
        assert not exported['complete']
        assert exported['translation'] == ' first'
        assert len(exported['turns']) == len(engine.sessions[0].committed) == 1
        assert exported['done']['unprocessed_audio_s'] == 1


def test_protocol_failure_exports_only_valid_prefix_and_is_not_complete(app_factory):
    app, _ = app_factory(engine_options={'replies': ['<|FLEX|> first', 'bad protocol']})
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ready = connect_ready(ws, source_kind='microphone')
        send_audio(ws, np.zeros(2 * RATE, dtype=np.int16))
        ws.send_json({'type': 'finish'})
        done = receive_until(ws, 'done')[-1]
        assert done['reason'] == 'model_protocol'
        client.portal.call(wait_for_release, app)
        exported = client.get('/api/sessions/' + ready['session_id'] + '/export').json()
        assert exported['translation'] == ' first'
        assert not exported['complete'] and len(exported['turns']) == 1
        assert exported['done']['reason'] == 'model_protocol'


def test_result_io_failure_rolls_back_text_and_event_to_same_prefix(tmp_path):
    store = ResultStore(temp_dir=tmp_path)
    try:
        item = store.create('partial-write')
        item.write({'type': 'turn', 'turn_id': 0, 'delta': '安全前缀'})
        expected_event_bytes, expected_text_bytes = item.event_bytes, item.text_bytes
        original_text = item.text

        class FailOnce:
            failed = False

            def write(self, data):
                if not self.failed:
                    self.failed = True
                    original_text.write(data[:4])  # Including an incomplete UTF-8 codepoint.
                    raise OSError('synthetic full disk')
                return original_text.write(data)

            def __getattr__(self, name):
                return getattr(original_text, name)

        item.text = FailOnce()
        with pytest.raises(OSError, match='synthetic full disk'):
            item.write({'type': 'turn', 'turn_id': 1, 'delta': '不能保存'})
        assert item.event_bytes == expected_event_bytes
        assert item.text_bytes == expected_text_bytes
        assert item.turn_count == 1
        exported = json.loads(b''.join(item.snapshot().iter_json()))
        assert exported['translation'] == '安全前缀'
        assert len(exported['turns']) == 1 and len(exported['events']) == 1
        item.write({'type': 'turn', 'turn_id': 2, 'delta': '继续'})
        exported = json.loads(b''.join(item.snapshot().iter_json()))
        assert exported['translation'] == '安全前缀继续'
        assert [turn['turn_id'] for turn in exported['turns']] == [0, 2]
    finally:
        store.close()


def test_microphone_offsets_stay_global_at_six_hours_and_limit_is_explicit():
    async def scenario():
        queue = PCMQueue(max_pending_seconds=12, max_session_seconds=8 * 3600)
        # Fast-forward counters without storing or replaying six hours of PCM.
        old_samples = 6 * 3600 * RATE
        queue.received = queue.taken = old_samples
        queue.sequence = 216000
        queue.push(frame(queue.sequence, old_samples, np.zeros(8000, dtype=np.int16)))
        queue.finish()
        start, audio, _ = await queue.take(RATE, 3 * RATE, True)
        assert start == old_samples and len(audio) == 8000
        assert queue.received == queue.taken == old_samples + 8000
        assert not queue.parts and queue.pending == 0
    asyncio.run(scenario())
    Settings(backend='mock', max_session_seconds=8 * 3600).validate()
    with pytest.raises(ValueError, match='sample-offset'):
        Settings(backend='mock', max_session_seconds=75 * 3600).validate()
