"""Optional real-decoder tests using synthetic media, never user recordings."""
import shutil
import subprocess

import numpy as np
import pytest

from flex_web.media import MediaError, MediaStore, decode_upload


FFMPEG = shutil.which('ffmpeg')
pytestmark = pytest.mark.skipif(FFMPEG is None, reason='real FFmpeg is not installed')


def generate(arguments, output):
    subprocess.run([FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'error',
                    *arguments, str(output)], check=True, timeout=30,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


@pytest.mark.parametrize('extension,codec', [
    ('wav', 'pcm_s16le'), ('flac', 'flac'), ('mp3', 'libmp3lame'), ('m4a', 'aac'),
])
def test_real_decoder_common_formats_keep_head_and_short_tail(tmp_path, extension, codec):
    source = tmp_path / f'synthetic.{extension}'
    generate(['-f', 'lavfi', '-i', 'sine=frequency=997:sample_rate=48000:duration=1.375',
              '-ac', '2', '-c:a', codec], source)
    store = MediaStore()
    try:
        item = store.reserve(source.name)
        shutil.copyfile(source, item.raw_path)
        decode_upload(item, 3, ffmpeg=FFMPEG)
        decoded = np.fromfile(item.pcm_path, dtype='<i2').astype(np.float32) / 32768
        # Lossless durations are exact; AAC may retain codec padding at EOF.
        if extension in ('wav', 'flac', 'mp3'):
            assert len(decoded) == 22000
        else:
            assert 22000 <= len(decoded) <= 22000 + 1024
        assert np.sqrt(np.mean(decoded[:1600] ** 2)) > .02
        assert np.sqrt(np.mean(decoded[20400:22000] ** 2)) > .02
    finally:
        store.close()


def test_real_mp4_delayed_audio_is_not_moved_ahead_of_video(tmp_path):
    source = tmp_path / 'delayed-track.mp4'
    generate(['-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=25:d=2',
              '-f', 'lavfi', '-i', 'sine=frequency=997:sample_rate=48000:duration=1.5',
              '-filter_complex', '[1:a]asetpts=PTS+0.5/TB[a]',
              '-map', '0:v', '-map', '[a]', '-c:v', 'mpeg4', '-c:a', 'aac', '-t', '2'], source)
    store = MediaStore()
    try:
        item = store.reserve(source.name)
        shutil.copyfile(source, item.raw_path)
        decode_upload(item, 3, ffmpeg=FFMPEG)
        decoded = np.fromfile(item.pcm_path, dtype='<i2').astype(np.float32) / 32768
        assert 1.95 <= len(decoded) / 16000 <= 2.05
        assert np.max(np.abs(decoded[:6400])) < .001  # The initial 0.4 s is still silence.
        assert np.sqrt(np.mean(decoded[9600:12800] ** 2)) > .02
    finally:
        store.close()


def test_real_decoder_overlong_file_fails_not_silently_truncated(tmp_path):
    source = tmp_path / 'long.flac'
    generate(['-f', 'lavfi', '-i', 'sine=frequency=997:sample_rate=48000:duration=2',
              '-c:a', 'flac'], source)
    store = MediaStore()
    try:
        item = store.reserve(source.name)
        shutil.copyfile(source, item.raw_path)
        with pytest.raises(MediaError) as caught:
            decode_upload(item, 1, ffmpeg=FFMPEG)
        assert caught.value.code == 'media_duration'
        assert item.pcm_path.stat().st_size < 16000 * 2 * 3
    finally:
        store.close()
