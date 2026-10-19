"""Bounded temporary uploads and file-backed, prefix-only audio delivery.

Decoding a file ahead of playback does not expose it to the model: FileQueue
only reads the prefix released by the browser's monotonically advancing clock.
No uploaded name is ever interpreted as a filesystem path or FFmpeg option.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
import wave

import numpy as np

RATE = 16000
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024


class MediaError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass
class Upload:
    media_id: str
    directory: Path
    source_name: str
    created: float
    samples: int = 0
    claimed: bool = False
    ready: bool = False

    @property
    def raw_path(self):
        return self.directory / 'upload.bin'

    @property
    def pcm_path(self):
        return self.directory / 'audio.pcm'

    def public(self):
        return dict(media_id=self.media_id, duration_s=self.samples / RATE,
                    samples=self.samples, sample_rate=RATE, source_name=self.source_name)


class MediaStore:
    """Event-loop-owned reservations; decoding is delegated to bounded workers."""
    def __init__(self, max_slots=4, ttl_seconds=600, *, temp_dir=None):
        # Long recordings belong on a configured disk, not necessarily /tmp
        # (which can be a small RAM-backed filesystem on deployment hosts).
        self.temporary = tempfile.TemporaryDirectory(prefix='flex-media-', dir=temp_dir)
        self.root = Path(self.temporary.name)
        self.uploads = {}
        self.max_slots, self.ttl_seconds = max_slots, ttl_seconds

    def expire(self):
        now = time.monotonic()
        for key, item in list(self.uploads.items()):
            if item.ready and not item.claimed and now - item.created > self.ttl_seconds:
                self.discard(key)

    def reserve(self, name):
        self.expire()
        if len(self.uploads) >= self.max_slots:
            raise MediaError('upload_busy', 'Too many pending files. Finish or discard an earlier upload, then retry.', 429)
        key = uuid.uuid4().hex
        directory = self.root / key
        directory.mkdir()
        item = Upload(key, directory, str(name or 'media')[:256], time.monotonic())
        self.uploads[key] = item
        return item

    def claim(self, key):
        self.expire()
        if not isinstance(key, str) or key not in self.uploads:
            raise MediaError('media_missing', 'Uploaded file expired or was removed. Upload it again.', 404)
        item = self.uploads[key]
        if not item.ready:
            raise MediaError('media_not_ready', 'File is still uploading or decoding.', 409)
        if item.claimed:
            raise MediaError('media_in_use', 'This uploaded file already belongs to a session.', 409)
        item.claimed = True
        return item

    def discard(self, key, *, allow_claimed=False):
        item = self.uploads.get(key)
        if item is None:
            return
        if item.claimed and not allow_claimed:
            raise MediaError('media_in_use', 'Cancel the active session before discarding its file.', 409)
        shutil.rmtree(item.directory)
        self.uploads.pop(key, None)

    def close(self):
        self.uploads.clear()
        self.temporary.cleanup()


def _container(path):
    """Use magic, not a client extension; never let FFmpeg probe a playlist."""
    with path.open('rb') as handle:
        header = handle.read(64)
    if header[:4] in (b'RIFF', b'RF64') and header[8:12] == b'WAVE':
        return 'wav'
    if header[:4] == b'fLaC':
        return 'flac'
    if header[:4] == b'OggS':
        return 'ogg'
    if header[:4] == b'\x1aE\xdf\xa3':
        return 'matroska'
    if header[4:8] in (b'ftyp', b'moov', b'mdat', b'wide', b'free'):
        return 'mov'
    if header[:4] == b'RIFF' and header[8:12] == b'AVI ':
        return 'avi'
    if header[:4] == b'FORM' and header[8:12] in (b'AIFF', b'AIFC'):
        return 'aiff'
    if header[:3] == b'ID3':
        return 'mp3'
    if len(header) > 1 and header[0] == 255 and header[1] & 0xf6 == 0xf0:
        return 'aac'
    if len(header) > 1 and header[0] == 255 and header[1] & 0xe0 == 0xe0:
        return 'mp3'
    raise MediaError('unsupported_media', 'Unsupported or invalid media. Choose WAV, MP3, MP4/M4A, WebM/MKV, OGG, FLAC, AAC, AIFF or AVI; playlists are not accepted.')


def decode_upload(item, max_seconds, timeout=1800, *, ffmpeg=None):
    """Write bounded 16k mono PCM16; reject overlong or empty media explicitly."""
    limit = round(max_seconds * RATE)
    container = _container(item.raw_path)
    native = False
    if container == 'wav':
        try:
            with wave.open(str(item.raw_path), 'rb') as source:
                native = (source.getframerate() == RATE and source.getnchannels() == 1
                          and source.getsampwidth() == 2 and source.getcomptype() == 'NONE')
                if native:
                    count = source.getnframes()
                    if count > limit:
                        raise MediaError('media_duration', f'File exceeds the {max_seconds:g}-second session limit. Choose a shorter segment or increase --max-session-seconds.')
                    copied = 0
                    with item.pcm_path.open('xb') as output:
                        while True:
                            block = source.readframes(RATE)
                            if not block:
                                break
                            output.write(block)
                            copied += len(block)
                    if copied != count * 2:
                        raise MediaError('invalid_media', 'WAV data is truncated; upload the complete file.')
        except (wave.Error, EOFError):
            # The stdlib wave reader does not support float/extensible WAV in
            # every Python version; let the restricted WAV decoder handle it.
            native = False
    if not native:
        executable = shutil.which(str(ffmpeg)) if ffmpeg else shutil.which('ffmpeg')
        if executable is None:
            raise MediaError('ffmpeg_missing', 'This format needs FFmpeg on the server. Install ffmpeg, or upload a 16 kHz mono PCM16 WAV.', 503)
        # Select a single safe demuxer before opening the uploaded data. No concat,
        # HLS, DASH or URL probing; mov external data references default disabled.
        command = [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-xerror',
                   '-protocol_whitelist', 'file,pipe', '-f', container]
        if container == 'mov':
            command += ['-enable_drefs', '0', '-use_absolute_path', '0']
        command += ['-threads', '1', '-i', str(item.raw_path), '-map', '0:a:0', '-vn', '-sn', '-dn',
                   '-filter_threads', '1',
                   '-af', 'aresample=async=1:first_pts=0', '-ac', '1', '-ar', str(RATE),
                   '-t', str(max_seconds + 1), '-fs', str((limit + 1) * 2),
                   '-f', 's16le', '-acodec', 'pcm_s16le', str(item.pcm_path)]
        try:
            # Error output goes to a bounded-by-input disk file, not unbounded RAM.
            with (item.directory / 'decode.stderr').open('wb') as error:
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=error,
                                        timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise MediaError('decode_timeout', 'Media decoding timed out. Try a shorter file or convert it to PCM16 WAV.', 422) from exc
        if result.returncode:
            raise MediaError('decode_failed', 'Unable to decode an audio track. Check the file is complete and contains audio; try PCM16 WAV.', 422)
    length = item.pcm_path.stat().st_size
    if not length or length % 2:
        raise MediaError('empty_audio', 'The uploaded file contains no usable audio.', 422)
    if length // 2 > limit:
        raise MediaError('media_duration', f'File exceeds the {max_seconds:g}-second session limit. Choose a shorter segment or increase --max-session-seconds.')
    item.samples = length // 2
    return item.samples


class FileQueue:
    """PCMQueue-compatible view over released file samples, without RAM backlog."""
    def __init__(self, item, offset_seconds=0, chunk_samples=RATE):
        self.item = item
        self.offset = math.floor(offset_seconds * RATE + 1e-7)
        if self.offset < 0 or self.offset >= item.samples:
            raise MediaError('media_offset', 'Playback start must be before the end of the uploaded audio.')
        self.total = item.samples - self.offset
        self.received = self.pending = self.taken = 0
        self.closed = False
        self.changed = asyncio.Event()
        self.released_at = time.monotonic()
        self.chunk_samples = chunk_samples
        self.next_boundary = chunk_samples
        self.availability = deque()
        self.handle = item.pcm_path.open('rb')
        self.handle.seek(self.offset * 2)

    def push(self, packet):
        raise MediaError('bad_frame', 'Uploaded-file sessions use progress messages, not browser audio capture.')

    def advance(self, position_seconds):
        if self.closed:
            raise MediaError('input_closed', 'Playback progress arrived after finish.')
        if isinstance(position_seconds, bool) or not isinstance(position_seconds, (int, float)) or not math.isfinite(position_seconds):
            raise MediaError('bad_progress', 'position_s must be a finite number.')
        absolute = math.floor(position_seconds * RATE + 1e-7)
        # Container/browser floating point clocks can differ by a few samples at
        # EOF. Clamp an overshoot only; never round an unheard tail up to EOF.
        if absolute > self.item.samples + 16:
            raise MediaError('bad_progress', 'Playback position exceeds decoded audio duration.')
        available = min(absolute, self.item.samples) - self.offset
        if available < self.received:
            raise MediaError('backward_seek', 'Backward seeking is not supported in an append-only session. Start a new session at that position.')
        if available > self.received:
            released = time.monotonic()
            # A delayed progress update can release hours of disk-backed audio.
            # Store its fixed-chunk boundaries as one inclusive range instead
            # of allocating one tuple per chunk. Later updates must not replace
            # the original availability time of an older queued chunk.
            if self.next_boundary <= available:
                last = available // self.chunk_samples * self.chunk_samples
                if (self.availability and self.availability[-1][2] == released
                        and self.availability[-1][1] + self.chunk_samples == self.next_boundary):
                    first, _, stamp = self.availability[-1]
                    self.availability[-1] = (first, last, stamp)
                else:
                    self.availability.append((self.next_boundary, last, released))
                self.next_boundary = last + self.chunk_samples
            self.pending += available - self.received
            self.received = available
            self.released_at = released
            self.changed.set()

    def finish(self):
        self.closed = True
        self.changed.set()

    async def take(self, base, maximum, adaptive):
        while self.pending < base and not self.closed:
            self.changed.clear()
            await self.changed.wait()
        if not self.pending:
            return None
        # Fixed chunks keep file results independent of browser callback delays.
        size = min(self.pending, base)
        data = self.handle.read(size * 2)
        if len(data) != size * 2:
            raise MediaError('media_read', 'Uploaded audio became unavailable; no samples were skipped.', 500)
        start = self.taken
        self.taken += size
        self.pending -= size
        arrived = self.released_at
        while self.availability and self.availability[0][0] <= self.taken:
            first, last, arrived = self.availability[0]
            if last <= self.taken:
                self.availability.popleft()
            else:
                first += ((self.taken - first) // self.chunk_samples + 1) * self.chunk_samples
                self.availability[0] = (first, last, arrived)
                break
        audio = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
        return start, audio, arrived

    def close(self):
        self.handle.close()
