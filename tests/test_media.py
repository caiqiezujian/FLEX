"""CPU-only file storage, decoding and prefix gating regressions."""
import asyncio
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
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


@pytest.mark.parametrize('max_seconds', [3, 8 * 60 * 60])
def test_ffmpeg_command_is_bounded_and_preserves_audio_timestamps(store, monkeypatch, max_seconds):
    monkeypatch.setattr(media.shutil, 'which', lambda _: '/usr/bin/ffmpeg')
    command = []
    def fake_run(args, **kwargs):
        command.extend(args)
        assert kwargs['timeout'] == 1800
        Path(args[-1]).write_bytes(b'\0\0' * 500)
        return type('Result', (), {'returncode': 0})()
    monkeypatch.setattr(media.subprocess, 'run', fake_run)
    item = store.reserve('attacker-selected-name.mp4')
    item.raw_path.write_bytes(b'\0\0\0\x18ftypisom' + b'\0' * 30)
    media.decode_upload(item, max_seconds)
    assert command[command.index('-protocol_whitelist') + 1] == 'file,pipe'
    assert command[command.index('-f') + 1] == 'mov'
    assert 'aresample=async=1:first_pts=0' in command
    assert command[command.index('-fs') + 1] == str((max_seconds * 16000 + 1) * 2)
    assert command[command.index('-t') + 1] == str(max_seconds + 1)
    assert command[command.index('-i') + 1] == str(item.raw_path)
    assert item.samples == 500


def test_media_store_uses_configured_temporary_disk(tmp_path):
    parent = tmp_path / 'large-recordings'
    parent.mkdir()
    store = media.MediaStore(temp_dir=parent)
    try:
        root = store.root
        assert root.parent == parent
        item = store.reserve('meeting.mp4')
        assert item.directory.is_dir()
    finally:
        store.close()
    assert parent.is_dir() and not root.exists()
    assert media.MAX_UPLOAD_BYTES == 4 * 1024 ** 3


def test_configured_decoder_timeout_is_forwarded_and_reported(store, monkeypatch):
    monkeypatch.setattr(media.shutil, 'which', lambda _: '/usr/bin/ffmpeg')
    def timed_out(args, **kwargs):
        assert kwargs['timeout'] == 37
        raise media.subprocess.TimeoutExpired(args, kwargs['timeout'])
    monkeypatch.setattr(media.subprocess, 'run', timed_out)
    item = store.reserve('meeting.mp3')
    item.raw_path.write_bytes(b'ID3' + b'\0' * 30)
    with pytest.raises(media.MediaError) as failure:
        media.decode_upload(item, 8 * 60 * 60, timeout=37)
    assert failure.value.code == 'decode_timeout'


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


def test_hours_of_progress_use_one_timestamp_range_without_loading_audio(monkeypatch):
    # A virtual sparse PCM file has the metadata of an eight-hour recording,
    # but materializes only the individual chunks requested by FileQueue.
    class SparsePCM(io.BytesIO):
        def __init__(self, length):
            super().__init__()
            self.length = length
            self.read_sizes = []

        def read(self, size=-1):
            assert 0 <= size <= media.RATE * 2
            self.read_sizes.append(size)
            size = min(size, self.length - self.tell())
            self.seek(size, io.SEEK_CUR)
            return b'\0' * size

    async def scenario():
        total = 8 * 60 * 60 * media.RATE
        handle = SparsePCM(total * 2)
        item = SimpleNamespace(samples=total, pcm_path=SimpleNamespace(open=lambda mode: handle))
        clock = [100.0]
        monkeypatch.setattr(media.time, 'monotonic', lambda: clock[0])
        queue = media.FileQueue(item)
        try:
            queue.advance(4 * 60 * 60)
            assert len(queue.availability) == 1
            assert not handle.read_sizes  # Releasing a prefix never reads it.
            clock[0] = 200.0
            queue.advance(8 * 60 * 60)
            assert len(queue.availability) == 2
            assert queue.received == queue.pending == total
            for index in range(4):
                start, audio, arrived = await queue.take(media.RATE, media.RATE * 3, True)
                assert start == index * media.RATE and len(audio) == media.RATE
                assert arrived == 100.0
                assert np.count_nonzero(audio) == 0
            assert queue.availability[0] == (5 * media.RATE, 4 * 60 * 60 * media.RATE, 100.0)
            assert len(queue.availability) == 2
            assert sum(handle.read_sizes) == 4 * media.RATE * 2
        finally:
            queue.close()
        assert handle.closed
    asyncio.run(scenario())


def test_range_timestamps_remain_exact_across_two_second_chunks_and_tail(store, monkeypatch):
    async def scenario():
        item = make_upload(store, np.zeros(9 * media.RATE))
        queue = media.FileQueue(item, chunk_samples=2 * media.RATE)
        clock = [10.0]
        monkeypatch.setattr(media.time, 'monotonic', lambda: clock[0])
        try:
            queue.advance(6.5)
            assert len(queue.availability) == 1
            clock[0] = 20.0
            queue.advance(8.5)
            queue.finish()
            chunks = []
            while (chunk := await queue.take(2 * media.RATE, 3 * media.RATE, False)) is not None:
                chunks.append((chunk[0], len(chunk[1]), chunk[2]))
            assert [stamp for _, _, stamp in chunks] == [10, 10, 10, 20, 20]
            assert [length for _, length, _ in chunks] == [32000, 32000, 32000, 32000, 8000]
            assert not queue.availability
        finally:
            queue.close()
    asyncio.run(scenario())


@pytest.fixture
def real_wmv(tmp_path):
    executable = shutil.which('ffmpeg')
    if not executable or not shutil.which('ffprobe'):
        pytest.skip('Real WMV integration requires FFmpeg and ffprobe')
    path = tmp_path / 'synthetic.wmv'
    subprocess.run([executable, '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-f', 'lavfi', '-i', 'color=c=blue:s=160x96:r=10:d=1.2',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100:duration=1.2',
                    '-c:v', 'wmv2', '-c:a', 'wmav2', '-f', 'asf', str(path)],
                   check=True, timeout=30)
    return path


def test_real_wmv_transcodes_same_playback_timeline_to_pcm(store, real_wmv):
    item = store.reserve('synthetic.wmv')
    original = real_wmv.read_bytes()
    item.raw_path.write_bytes(original)
    assert media._container(item.raw_path) == 'asf'
    media.decode_upload(item, 3, timeout=30)
    assert item.raw_path.read_bytes() == original
    assert item.transcoded and item.samples > 0
    assert media._container(item.playback_path) == 'mov'
    public = item.public()
    assert public['playback_url'] == f'/api/media/{item.media_id}/playback'
    assert public['playback_type'] == 'video/mp4' and public['transcoded']
    assert item.playback_secret not in json.dumps(public)
    expected = item.directory / 'independent.pcm'
    subprocess.run([shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin',
                    '-i', str(item.playback_path), '-map', '0:a:0', '-vn',
                    '-af', 'aresample=async=1:first_pts=0', '-ac', '1', '-ar', '16000',
                    '-f', 's16le', '-acodec', 'pcm_s16le', str(expected)], check=True, timeout=30)
    assert expected.read_bytes() == item.pcm_path.read_bytes()


def test_real_wmv_duration_limit_is_not_silently_truncated(store, real_wmv):
    item = store.reserve('synthetic.wmv')
    item.raw_path.write_bytes(real_wmv.read_bytes())
    with pytest.raises(media.MediaError) as error:
        media.decode_upload(item, .5, timeout=30)
    assert error.value.code == 'media_duration'
    assert not item.transcoded and not item.samples


def _fake_wmv(store, monkeypatch):
    item = store.reserve('test.wmv')
    item.raw_path.write_bytes(bytes.fromhex('3026b2758e66cf11a6d900aa0062ce6c') + b'\0' * 32)
    monkeypatch.setattr(media.shutil, 'which', lambda name: '/usr/bin/' + str(name))
    return item


def test_wmv_missing_encoder_is_actionable(store, monkeypatch):
    item = _fake_wmv(store, monkeypatch)
    def rejected(args, **kwargs):
        assert 'libx264' in args and '-n' in args and '-shortest' not in args
        assert args[args.index('-f') + 1] == 'asf'
        assert args[args.index('-protocol_whitelist') + 1] == 'file,pipe'
        kwargs['stderr'].write(b"Unknown encoder 'libx264'")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(media.subprocess, 'run', rejected)
    with pytest.raises(media.MediaError) as error:
        media.decode_upload(item, 3)
    assert error.value.code == 'transcode_encoder' and 'libx264' in str(error.value)
    assert not item.transcoded


def test_wmv_timeout_and_disk_budget_are_explicit(store, monkeypatch):
    item = _fake_wmv(store, monkeypatch)
    def oversized(args, **kwargs):
        assert args[args.index('-fs') + 1] == '1024'
        assert 0 < kwargs['timeout'] <= 7
        item.playback_path.write_bytes(b'\0' * 1024)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(media.subprocess, 'run', oversized)
    with pytest.raises(media.MediaError) as error:
        media.decode_upload(item, 3, timeout=7, max_playback_bytes=1024)
    assert error.value.code == 'playback_size'
    def timed_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    monkeypatch.setattr(media.subprocess, 'run', timed_out)
    with pytest.raises(media.MediaError) as error:
        media.decode_upload(item, 3, timeout=7)
    assert error.value.code == 'decode_timeout'


def test_wmv_probe_required_and_bad_timeline_rejected(store, monkeypatch):
    item = _fake_wmv(store, monkeypatch)
    monkeypatch.setattr(media.shutil, 'which', lambda name: None if name == 'ffprobe' else '/absent/ffmpeg')
    with pytest.raises(media.MediaError) as error:
        media.decode_upload(item, 3)
    assert error.value.code == 'ffprobe_missing'
    monkeypatch.setattr(media.shutil, 'which', lambda name: '/usr/bin/' + str(name))
    def prepared(args, **kwargs):
        if '-show_entries' in args:
            kwargs['stdout'].write(json.dumps(dict(format=dict(duration='2.0'), streams=[
                dict(codec_type='video', codec_name='h264'), dict(codec_type='audio', codec_name='aac')])).encode())
        elif 'libx264' in args:
            item.playback_path.write_bytes(b'mp4')
        else:
            assert args[args.index('-i') + 1] == str(item.playback_path)
            item.pcm_path.write_bytes(b'\0\0' * media.RATE)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(media.subprocess, 'run', prepared)
    with pytest.raises(media.MediaError) as error:
        media.decode_upload(item, 3)
    assert error.value.code == 'media_timeline' and not item.transcoded


def test_playback_readers_defer_cleanup_without_accepting_new_claims(store):
    item = make_upload(store, np.zeros(16000))
    item.transcoded = True
    item.playback_path.write_bytes(b'video')
    handle = store.open_playback(item)
    store.discard(item.media_id)
    assert item.directory.exists() and handle.read() == b'video'
    with pytest.raises(media.MediaError) as error:
        store.playback(item.media_id)
    assert error.value.code == 'media_missing'
    with pytest.raises(media.MediaError):
        store.claim(item.media_id)
    store.close_playback(item, handle)
    assert not item.directory.exists() and item.media_id not in store.uploads
