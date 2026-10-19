"""CPU-only file storage, decoding and prefix gating regressions."""
import asyncio
import io
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web import media


def wav_bytes(samples, *, rate=16000, channels=1):
    output = io.BytesIO()
    with wave.open(output, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(samples, dtype='<i2').tobytes())
    return output.getvalue()


@pytest.fixture
def store():
    value = media.MediaStore()
    yield value
    value.close()


def make_upload(store, samples):
    item = store.reserve('../../not-a-path.wav')
    item.raw_path.write_bytes(wav_bytes(samples))
    media.decode_upload(item, 10)
    item.ready = True
    return item


def test_native_wav_is_exact_and_does_not_require_ffmpeg(store, monkeypatch):
    monkeypatch.setattr(media.shutil, 'which', lambda _: None)
    samples = np.arange(17600, dtype=np.int16)
    item = make_upload(store, samples)
    assert item.samples == len(samples)
    assert item.directory.parent == store.root
    assert item.pcm_path.read_bytes() == samples.astype('<i2').tobytes()
    assert item.public()['duration_s'] == 1.1


def test_native_wav_duration_empty_and_truncation_fail(store):
    item = store.reserve('long.wav')
    item.raw_path.write_bytes(wav_bytes(np.zeros(16001)))
    with pytest.raises(media.MediaError) as failure:
        media.decode_upload(item, 1)
    assert failure.value.code == 'media_duration'
    item = store.reserve('empty.wav')
    item.raw_path.write_bytes(wav_bytes([]))
    with pytest.raises(media.MediaError) as failure:
        media.decode_upload(item, 1)
    assert failure.value.code == 'empty_audio'
    item = store.reserve('truncated.wav')
    item.raw_path.write_bytes(wav_bytes(np.zeros(20))[:-10])
    with pytest.raises(media.MediaError) as failure:
        media.decode_upload(item, 1)
    assert failure.value.code == 'invalid_media'


@pytest.mark.parametrize('data', [b'#EXTM3U\nhttps://example.com/secret',
                                  b'ffconcat version 1.0\nfile /etc/passwd',
                                  b'<MPD><BaseURL>http://localhost</BaseURL></MPD>',
                                  b'random text'])
def test_playlists_and_non_media_never_reach_ffmpeg(store, monkeypatch, data):
    def forbidden(*args, **kwargs):
        raise AssertionError('unsafe input reached FFmpeg')
    monkeypatch.setattr(media.subprocess, 'run', forbidden)
    item = store.reserve('fake.mp4')
    item.raw_path.write_bytes(data)
    with pytest.raises(media.MediaError) as failure:
        media.decode_upload(item, 3)
    assert failure.value.code == 'unsupported_media'


def test_ffmpeg_command_is_bounded_and_preserves_audio_timestamps(store, monkeypatch):
    monkeypatch.setattr(media.shutil, 'which', lambda _: '/usr/bin/ffmpeg')
    command = []
    def fake_run(args, **kwargs):
        command.extend(args)
        assert kwargs['timeout'] == 60
        Path(args[-1]).write_bytes(b'\0\0' * 500)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(media.subprocess, 'run', fake_run)
    item = store.reserve('attacker-selected-name.mp4')
    item.raw_path.write_bytes(b'\0\0\0\x18ftypisom' + b'\0' * 30)
    media.decode_upload(item, 3)
    assert command[command.index('-protocol_whitelist') + 1] == 'file,pipe'
    assert command[command.index('-f') + 1] == 'mov'
    assert 'aresample=async=1:first_pts=0' in command
    assert command[command.index('-fs') + 1] == str((3 * 16000 + 1) * 2)
    assert command[command.index('-i') + 1] == str(item.raw_path)
    assert item.samples == 500


def test_ffmpeg_missing_is_actionable(store, monkeypatch):
    monkeypatch.setattr(media.shutil, 'which', lambda _: None)
    item = store.reserve('compressed.mp3')
    item.raw_path.write_bytes(b'ID3' + b'\0' * 30)
    with pytest.raises(media.MediaError) as failure:
        media.decode_upload(item, 3)
    assert failure.value.code == 'ffmpeg_missing'
    assert '16 kHz mono PCM16 WAV' in str(failure.value)


def test_slots_single_claim_and_expiry_are_bounded(monkeypatch):
    store = media.MediaStore(max_slots=1, ttl_seconds=10)
    try:
        item = store.reserve('one.wav')
        with pytest.raises(media.MediaError) as failure:
            store.reserve('two.wav')
        assert failure.value.code == 'upload_busy'
        item.ready = True
        store.claim(item.media_id)
        with pytest.raises(media.MediaError) as failure:
            store.claim(item.media_id)
        assert failure.value.code == 'media_in_use'
        with pytest.raises(media.MediaError):
            store.discard(item.media_id)
        monkeypatch.setattr(media.time, 'monotonic', lambda: item.created + 100)
        store.expire()
        assert item.directory.exists()  # A slow active session never expires.
        store.discard(item.media_id, allow_claimed=True)
        assert not item.directory.exists()
        another = store.reserve('two.wav')
        another.ready = True
        another.created -= 11
        store.expire()
        assert not another.directory.exists() and not store.uploads
    finally:
        root = store.root
        store.close()
    assert not root.exists()


def test_file_queue_reads_only_released_prefix_and_exact_tail(store):
    async def scenario():
        samples = np.arange(36001, dtype=np.int16)
        item = make_upload(store, samples)
        queue = media.FileQueue(item, .5)
        try:
            pending = asyncio.create_task(queue.take(16000, 48000, True))
            await asyncio.sleep(0)
            assert not pending.done() and queue.handle.tell() == 16000
            queue.advance(.75)
            await asyncio.sleep(0)
            assert not pending.done()  # Partial chunk stays gated until finish.
            queue.advance(1.5)
            start, audio, _ = await pending
            assert start == 0 and len(audio) == 16000
            np.testing.assert_array_equal(audio, samples[8000:24000] / np.float32(32768))
            queue.advance(item.samples / media.RATE)
            queue.finish()
            start, tail, _ = await queue.take(16000, 48000, True)
            assert start == 16000 and len(tail) == 12001
            np.testing.assert_array_equal(tail, samples[24000:] / np.float32(32768))
            assert await queue.take(16000, 48000, True) is None
            assert queue.received == queue.taken == 28001
        finally:
            queue.close()
    asyncio.run(scenario())


def test_delayed_progress_does_not_drop_combine_or_overflow(store):
    async def scenario():
        item = make_upload(store, np.arange(160000, dtype=np.int16))
        queue = media.FileQueue(item)
        try:
            queue.advance(10)  # Browser blocked for many base chunks.
            queue.finish()
            turns = []
            while True:
                block = await queue.take(16000, 48000, True)
                if block is None:
                    break
                turns.append(block)
            assert len(turns) == 10
            assert [block[0] for block in turns] == list(range(0, 160000, 16000))
            assert all(len(block[1]) == 16000 for block in turns)
        finally:
            queue.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('position', [None, '1', True, float('nan'), float('inf'), -1, 5])
def test_bad_progress_cannot_release_samples(store, position):
    async def scenario():
        item = make_upload(store, np.zeros(16000))
        queue = media.FileQueue(item)
        try:
            with pytest.raises(media.MediaError):
                queue.advance(position)
            assert queue.received == queue.taken == 0
        finally:
            queue.close()
    asyncio.run(scenario())


def test_manual_finish_does_not_promote_near_eof_to_whole_file(store):
    async def scenario():
        item = make_upload(store, np.zeros(16000))
        queue = media.FileQueue(item)
        try:
            queue.advance(.999)
            queue.finish()
            _, audio, _ = await queue.take(16000, 16000, False)
            assert len(audio) == 15984
            assert queue.received == 15984
        finally:
            queue.close()
    asyncio.run(scenario())


def test_chunk_availability_time_is_not_overwritten_by_newer_progress(store, monkeypatch):
    async def scenario():
        item = make_upload(store, np.zeros(64000))
        queue = media.FileQueue(item)
        try:
            clock = [100.0]
            monkeypatch.setattr(media.time, 'monotonic', lambda: clock[0])
            queue.advance(2)
            clock[0] = 105.0
            queue.advance(3)
            # Older queued chunks remain available since 100, not since 105.
            assert (await queue.take(16000, 48000, True))[2] == 100.0
            assert (await queue.take(16000, 48000, True))[2] == 100.0
            assert (await queue.take(16000, 48000, True))[2] == 105.0
            clock[0] = 106.0
            queue.advance(3.1)
            clock[0] = 110.0
            queue.finish()
            assert (await queue.take(16000, 48000, True))[2] == 106.0
        finally:
            queue.close()
    asyncio.run(scenario())


def test_progress_frequency_does_not_create_unbounded_availability_records(store):
    async def scenario():
        item = make_upload(store, np.zeros(16000))
        queue = media.FileQueue(item)
        try:
            for sample in range(1, 16001):
                queue.advance(sample / 16000)
            assert len(queue.availability) == 1
            assert queue.received == 16000
        finally:
            queue.close()
    asyncio.run(scenario())
