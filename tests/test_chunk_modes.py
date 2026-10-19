"""End-to-end checks for selectable file chunks and live backlog adaptation.

The fake engine records its actual audio inputs: choosing a duration must affect
model calls, not only the UI or the reported settings. No GPU is required.
"""
import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flex_web.server import PCMQueue, RATE
from test_file_server import upload
from test_server import app_factory, connect_ready, frame, receive_until


@pytest.mark.parametrize('chunk_seconds', [1, 2, 3])
def test_uploaded_file_chunk_choice_controls_model_inputs_and_exact_tail(
        app_factory, chunk_seconds):
    # More than the live queue cap is released at once after the first turn.
    # Uploaded files must retain fixed boundaries even with adaptive=True.
    total_samples = 118000  # 7.375 seconds, including a non-integer final tail.
    base = chunk_seconds * RATE
    app, engine = app_factory(max_pending_seconds=3)
    with TestClient(app) as client:
        item = upload(client, samples=total_samples)
        with client.websocket_connect('/ws') as ws:
            ready = connect_ready(ws, media_id=item['media_id'],
                                  chunk_seconds=chunk_seconds, adaptive=True)
            assert ready['config']['chunk_seconds'] == chunk_seconds
            assert ready['config']['adaptive'] is False

            # A partial prefix is not a complete inference chunk. A pong
            # establishes that the server handled the preceding progress event.
            ws.send_json(dict(type='progress', position_s=chunk_seconds - .25))
            ws.send_json(dict(type='ping'))
            receive_until(ws, 'pong')
            assert engine.sessions[0].generated == []

            ws.send_json(dict(type='progress', position_s=chunk_seconds))
            first = receive_until(ws, 'turn')[-1]
            assert first['audio_seconds'] == chunk_seconds
            assert len(engine.sessions[0].generated[0]) == base

            # Delayed browser progress can release a large prefix in one go.
            # The file-backed queue must not merge it or trigger live backlog
            # overflow; finish is the only reason a short final chunk is sent.
            ws.send_json(dict(type='finish', position_s=total_samples / RATE))
            remaining = receive_until(ws, 'done')
            turns = [first] + [event for event in remaining if event['type'] == 'turn']
            expected_sizes = [min(base, total_samples - offset)
                              for offset in range(0, total_samples, base)]
            assert len(turns) == len(expected_sizes)
            assert [turn['audio_seconds'] for turn in turns] == pytest.approx(
                [size / RATE for size in expected_sizes])

            cursor, recorded = 0, []
            for turn, audio, size in zip(turns, engine.sessions[0].generated,
                                         expected_sizes):
                assert turn['source_start_s'] == pytest.approx(cursor / RATE)
                assert turn['source_end_s'] == pytest.approx((cursor + size) / RATE)
                # The model's minimum 1-second input padding is not part of the
                # source timeline and may never replace or duplicate samples.
                assert len(audio) == max(RATE, size)
                recorded.append(audio[:size])
                assert np.count_nonzero(audio[size:]) == 0
                cursor += size
            assert cursor == total_samples
            expected_audio = np.arange(total_samples, dtype=np.int16).astype(np.float32) / 32768
            np.testing.assert_array_equal(np.concatenate(recorded), expected_audio)
            done = remaining[-1]
            assert done['reason'] == 'finished'
            assert done['received_audio_s'] == done['processed_audio_s'] == total_samples / RATE
        assert not app.state.media.uploads


def push_samples(queue, pcm):
    """Send legal <=500ms transport frames independently of model chunk size."""
    for offset in range(0, len(pcm), RATE // 2):
        queue.push(frame(queue.sequence, queue.received, pcm[offset:offset + RATE // 2]))


@pytest.mark.parametrize('base_seconds,pending_seconds,adaptive,first_seconds', [
    (1, 2.6, True, 2),
    (1, 2.6, False, 1),
    (1, 4.25, True, 3),
    (1, 4.25, False, 1),
    (2, 2.6, True, 2),
    (2, 2.6, False, 2),
    (2, 4, True, 3),
    (2, 4, False, 2),
    (2, 6.25, True, 3),
    (2, 6.25, False, 2),
    (3, 7.25, True, 3),
    (3, 7.25, False, 3),
])
def test_live_queue_adapts_only_contiguous_arrived_samples_and_respects_cap(
        base_seconds, pending_seconds, adaptive, first_seconds):
    async def scenario():
        queue = PCMQueue(max_pending_seconds=12, max_session_seconds=30)
        source = np.arange(round(pending_seconds * RATE), dtype=np.int16)
        push_samples(queue, source)
        queue.finish()
        base, maximum = base_seconds * RATE, 3 * RATE
        outputs, cursor = [], 0
        while queue.pending:
            pending = queue.pending
            expected_size = min(pending, base)
            if adaptive and pending >= base:
                expected_size = min(maximum, pending // base * base)
            start, audio, _ = await queue.take(base, maximum, adaptive)
            assert start == cursor
            assert len(audio) == expected_size
            if not outputs:
                assert len(audio) == first_seconds * RATE
            outputs.append(audio)
            cursor += len(audio)
        assert await queue.take(base, maximum, adaptive) is None
        assert queue.received == queue.taken == cursor == len(source)
        np.testing.assert_array_equal(np.concatenate(outputs),
                                      source.astype(np.float32) / 32768)

    asyncio.run(scenario())


@pytest.mark.parametrize('base_seconds', [1, 2, 3])
@pytest.mark.parametrize('adaptive', [False, True])
def test_live_queue_waits_for_full_base_chunk_before_generating(base_seconds, adaptive):
    async def scenario():
        queue = PCMQueue(max_pending_seconds=12, max_session_seconds=30)
        base = base_seconds * RATE
        source = np.arange(base, dtype=np.int16)
        push_samples(queue, source[:-RATE // 4])
        pending_take = asyncio.create_task(queue.take(base, 3 * RATE, adaptive))
        await asyncio.sleep(0)
        assert not pending_take.done()
        push_samples(queue, source[-RATE // 4:])
        start, audio, _ = await asyncio.wait_for(pending_take, timeout=1)
        assert start == 0 and len(audio) == base
        np.testing.assert_array_equal(audio, source.astype(np.float32) / 32768)
        queue.finish()
        assert await queue.take(base, 3 * RATE, adaptive) is None

    asyncio.run(scenario())
