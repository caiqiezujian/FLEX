"""Disk-backed session results; bounded RAM and streaming, consistent exports.

These temporary files exist even with --no-log. They are not a durable recovery
database: completed sessions expire and shutdown removes this process's files.
Only directories created by this store are ever removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time


class ResultLimit(RuntimeError):
    pass


@dataclass
class Snapshot:
    events: object
    text: object
    event_bytes: int
    text_bytes: int
    metadata: dict
    complete: bool

    def close(self):
        self.events.close()
        self.text.close()

    def iter_text(self):
        try:
            remaining = self.text_bytes
            while remaining:
                block = self.text.read(min(65536, remaining))
                if not block:
                    raise OSError('Session result became unavailable')
                remaining -= len(block)
                yield block
        finally:
            self.close()

    def iter_json(self):
        """Stream JSON without materializing hours of events or text in RAM."""
        import codecs
        encode = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        try:
            header = dict(schema_version=2, **self.metadata, complete=self.complete,
                          snapshot=True, timing_note='Source intervals are input intervals, not word alignment or semantic latency.')
            yield encode(header)[:-1] + b',"translation":"'
            decoder = codecs.getincrementaldecoder('utf-8')()
            remaining = self.text_bytes
            while remaining:
                block = self.text.read(min(65536, remaining))
                if not block:
                    raise OSError('Session result became unavailable')
                remaining -= len(block)
                piece = decoder.decode(block, final=not remaining)
                yield encode(piece)[1:-1]
            yield b'","turns":['
            first = True
            remaining = self.event_bytes
            while remaining:
                line = self.events.readline(remaining)
                if not line:
                    raise OSError('Session events became unavailable')
                remaining -= len(line)
                event = json.loads(line)
                if event['type'] == 'turn':
                    if not first:
                        yield b','
                    yield line.rstrip(b'\r\n')
                    first = False
            yield b'],"events":['
            self.events.seek(0)
            remaining, first = self.event_bytes, True
            while remaining:
                line = self.events.readline(remaining)
                if not line:
                    raise OSError('Session events became unavailable')
                remaining -= len(line)
                if not first:
                    yield b','
                yield line.rstrip(b'\r\n')
                first = False
            yield b']}'
        finally:
            self.close()


class SessionResult:
    def __init__(self, directory, session_id, max_bytes):
        self.directory, self.session_id = directory, session_id
        self.max_bytes = max_bytes
        self.event_bytes = self.text_bytes = self.translation_chars = self.turn_count = 0
        self.metadata = {'session_id': session_id, 'error': None}
        self.finished_at = None
        self.events = (directory / 'events.jsonl').open('xb')
        try:
            self.text = (directory / 'translation.txt').open('xb')
        except BaseException:
            self.events.close()
            raise

    def write(self, event):
        line = (json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
        delta = event.get('delta', '') if event['type'] == 'turn' else ''
        encoded = delta.encode('utf-8')
        if self.event_bytes + self.text_bytes + len(line) + len(encoded) > self.max_bytes:
            raise ResultLimit('Session result storage budget reached; export results and start a new session, or increase --max-result-mb.')
        # Publish byte counts only after both files are flushed, so downloads
        # see complete lines and the text corresponding to precisely those turns.
        try:
            self.events.write(line)
            self.events.flush()
            self.text.write(encoded)
            self.text.flush()
        except OSError:
            self.events.seek(self.event_bytes)
            self.events.truncate()
            self.text.seek(self.text_bytes)
            self.text.truncate()
            raise
        self.event_bytes += len(line)
        self.text_bytes += len(encoded)
        self.translation_chars += len(delta)
        if event['type'] == 'turn':
            self.turn_count += 1
        if event['type'] == 'ready':
            config = event.get('config', {})
            self.metadata.update(backend=event.get('backend'), started_at=event.get('utc'),
                                 source_language=config.get('source_lang'), target_language=config.get('target_lang'),
                                 source_kind=config.get('source_kind'), source_name=config.get('source_name'),
                                 capture_mode='server-file' if config.get('media_id') else 'worklet',
                                 media_offset_s=config.get('media_offset_s', 0), chunk_seconds=config.get('chunk_seconds'))
        if event['type'] == 'error':
            self.metadata['error'] = event
        if event['type'] == 'done':
            self.metadata['done'] = event

    def finish(self):
        if self.finished_at is None:
            self.finished_at = time.monotonic()
            try:
                self.events.close()
            finally:
                self.text.close()

    def snapshot(self):
        events = (self.directory / 'events.jsonl').open('rb')
        try:
            text = (self.directory / 'translation.txt').open('rb')
        except BaseException:
            events.close()
            raise
        done = self.metadata.get('done', {})
        complete = (done.get('reason') == 'finished'
                    and done.get('turn_count') == self.turn_count
                    and done.get('translation_chars') == self.translation_chars)
        return Snapshot(events, text, self.event_bytes, self.text_bytes,
                        {**self.metadata, 'translation_chars': self.translation_chars, 'turn_count': self.turn_count,
                         'session_finished': self.finished_at is not None}, complete)


class ResultStore:
    def __init__(self, *, max_sessions=8, ttl_seconds=86400, max_bytes=256 * 1024 * 1024, temp_dir=None):
        self.temporary = tempfile.TemporaryDirectory(prefix='flex-results-', dir=temp_dir)
        self.root = Path(self.temporary.name)
        self.items = {}
        self.max_sessions, self.ttl_seconds, self.max_bytes = max_sessions, ttl_seconds, max_bytes

    def _discard(self, key):
        item = self.items[key]
        if item.finished_at is None:
            return False
        try:
            shutil.rmtree(item.directory)
        except OSError:  # A Windows download may still have its snapshot open.
            return False
        del self.items[key]
        return True

    def expire(self):
        now = time.monotonic()
        for key, item in list(self.items.items()):
            if item.finished_at is not None and now - item.finished_at > self.ttl_seconds:
                self._discard(key)

    def create(self, session_id):
        self.expire()
        if len(self.items) >= self.max_sessions:
            for key, item in list(self.items.items()):
                if self._discard(key):
                    break
        if len(self.items) >= self.max_sessions:
            raise ResultLimit('Result storage is busy; finish an active export and retry.')
        directory = self.root / session_id
        directory.mkdir()
        try:
            item = SessionResult(directory, session_id, self.max_bytes)
        except BaseException:
            shutil.rmtree(directory)
            raise
        self.items[session_id] = item
        return item

    def close(self):
        for item in self.items.values():
            item.finish()
        self.items.clear()
        self.temporary.cleanup()
