"""Single-GPU WebSocket application. Browser capture and inference run independently.

Only complete, protocol-valid generations are committed. Several independent
sessions share one model through a bounded scheduler. A disconnected client
never frees its admission slot until its in-flight generation has finished.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import functools
import hmac
import json
import logging
import math
import os
from pathlib import Path
import struct
import time
from urllib.parse import urlsplit
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles

from .model import ContextLimit, ModelBackendError, ModelEngine, ModelOptions, ProtocolError, SUPPORTED_DIRECTIONS, parse_reply
from .media import FileQueue, MAX_UPLOAD_BYTES, MediaError, MediaStore, decode_upload
from .tls import TLSConfigError, prepare_local_tls, validate_tls_files
from .results import ResultLimit, ResultStore
from .scheduler import InferenceScheduler, SchedulerCancelled, SchedulerUnavailable

RATE = 16000
STATIC = Path(__file__).resolve().parent.parent / 'static'
LOG = logging.getLogger('flex_web')


@dataclass
class Settings:
    backend: str = 'vllm'
    model: str = ''
    log_dir: Path | None = field(default_factory=lambda: Path('logs'))
    directions: tuple[str, ...] = SUPPORTED_DIRECTIONS
    chunk_seconds: float = 1.0
    max_chunk_seconds: float = 3.0
    max_pending_seconds: float = 12.0
    max_session_seconds: float = 28800.0
    max_sessions: int = 4
    max_batch_size: int | None = None
    batch_wait_ms: float = 10.0
    slow_inference_seconds: float = 5.0
    stalled_inference_seconds: float = 60.0
    token: str = ''
    allowed_origins: tuple[str, ...] = ()
    idle_timeout_seconds: float = 300.0
    model_options: ModelOptions | None = None
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    max_uploads: int = 4
    upload_timeout_seconds: float = 3600.0
    decode_timeout_seconds: float = 1800.0
    temp_dir: Path | None = None
    max_result_bytes: int = 256 * 1024 * 1024
    result_retention_hours: float = 24.0
    ffmpeg: str | None = None

    @property
    def batch_size(self):
        return self.max_sessions if self.max_batch_size is None else self.max_batch_size

    def validate(self):
        if self.backend not in ('mock', 'vllm'):
            raise ValueError('backend must be mock or vllm')
        if self.backend == 'vllm' and not self.model:
            raise ValueError('--model is required for vllm (use a merged local checkpoint)')
        if type(self.max_sessions) is not int or self.max_sessions < 1:
            raise ValueError('max_sessions must be a positive integer')
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= self.max_sessions:
            raise ValueError('require 1 <= max_batch_size <= max_sessions')
        if (isinstance(self.batch_wait_ms, bool) or not math.isfinite(self.batch_wait_ms)
                or not 0 <= self.batch_wait_ms <= 100):
            raise ValueError('batch_wait_ms must be finite and between 0 and 100')
        for value in (self.slow_inference_seconds, self.stalled_inference_seconds):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('inference health thresholds must be finite and positive')
        if self.stalled_inference_seconds <= self.slow_inference_seconds:
            raise ValueError('stalled_inference_seconds must exceed slow_inference_seconds')
        if not self.directions or any(x not in SUPPORTED_DIRECTIONS for x in self.directions):
            raise ValueError('directions must contain en-zh and/or zh-en')
        numeric = (self.chunk_seconds, self.max_chunk_seconds, self.max_pending_seconds,
                   self.max_session_seconds, self.idle_timeout_seconds)
        if not all(math.isfinite(v) and v > 0 for v in numeric):
            raise ValueError('time limits must be finite and positive')
        if not .5 <= self.chunk_seconds <= self.max_chunk_seconds <= 6:
            raise ValueError('require 0.5 <= chunk_seconds <= max_chunk_seconds <= 6')
        if self.max_pending_seconds < self.max_chunk_seconds:
            raise ValueError('max_pending_seconds must be >= max_chunk_seconds')
        if self.max_session_seconds * RATE > 2**32 - 1:
            raise ValueError('max_session_seconds exceeds the PCM transport sample-offset range (about 74 hours)')
        if (type(self.max_upload_bytes) is not int or self.max_upload_bytes <= 0
                or type(self.max_uploads) is not int or self.max_uploads <= 0
                or not math.isfinite(self.upload_timeout_seconds) or self.upload_timeout_seconds <= 0):
            raise ValueError('upload limits must be finite and positive')
        if (not math.isfinite(self.decode_timeout_seconds) or self.decode_timeout_seconds <= 0
                or not math.isfinite(self.result_retention_hours) or self.result_retention_hours <= 0
                or type(self.max_result_bytes) is not int or self.max_result_bytes <= 0):
            raise ValueError('decode/result limits must be finite and positive')


class ClientError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ProcessingMonitor:
    """Warn on sustained unprocessed audio, never on WAIT or missing text.

    Includes the current queued/running turn. A transient burst is not a fault;
    recovery has hysteresis to keep status from flashing at the threshold.
    This monitor never changes audio, playback, chunking, or model behavior.
    """
    def __init__(self, chunk_seconds, *, sustain_seconds=5.0, clock=None):
        self.threshold = max(3.0, chunk_seconds * 2)
        self.recovery_threshold = self.threshold / 2
        self.sustain_seconds = sustain_seconds
        self.clock = clock or time.monotonic
        self.started = None
        self.slow = False

    def update(self, unprocessed_seconds):
        now = self.clock()
        if unprocessed_seconds <= self.recovery_threshold:
            self.started = None
            self.slow = False
        elif unprocessed_seconds >= self.threshold:
            if self.started is None:
                self.started = now
            if now - self.started >= self.sustain_seconds:
                self.slow = True
        elif not self.slow:
            # Before warning, require a continuous interval over the threshold.
            self.started = None
        return dict(state='slow' if self.slow else 'normal',
                    message='处理速度暂时落后，正在处理积压音频。' if self.slow else '',
                    backlog_seconds=max(0.0, unprocessed_seconds),
                    sustained_seconds=max(0.0, now - self.started) if self.started is not None else 0.0)


class PCMQueue:
    """Event-loop-owned FIFO, measured in samples, without unbounded Queue.put()."""
    def __init__(self, max_pending_seconds, max_session_seconds):
        self.parts = deque()
        self.pending = self.received = self.taken = self.sequence = 0
        self.capacity = round(max_pending_seconds * RATE)
        self.limit = round(max_session_seconds * RATE)
        self.closed = False
        self.changed = asyncio.Event()

    def push(self, packet):
        if self.closed:
            raise ClientError('input_closed', 'Audio arrived after finish.')
        # Transport frames are at most 500ms; inference chunks are independently sized.
        if len(packet) < 10 or (len(packet) - 8) % 2 or len(packet) > 8 + RATE:
            raise ClientError('bad_frame', 'Expected an 8-byte header plus 1..8000 PCM16 samples.')
        sequence, start = struct.unpack_from('<II', packet)
        if sequence != self.sequence or start != self.received:
            raise ClientError('sequence_gap', 'Audio sequence/sample offset is discontinuous; start a new session.')
        audio = np.frombuffer(packet, dtype='<i2', offset=8).astype(np.float32) / 32768.0
        if self.received + len(audio) > self.limit:
            raise ClientError('session_limit', 'Configured total session duration reached. Increase --max-session-seconds for a longer session; this is separate from the rolling model context.')
        if self.pending + len(audio) > self.capacity:
            raise ClientError('backlog_limit', 'Inference cannot keep up: pending audio limit reached. No queued audio was silently dropped; this session is stopping.')
        self.parts.append((audio, time.monotonic()))
        self.pending += len(audio)
        self.received += len(audio)
        self.sequence += 1
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
        size = min(self.pending, base)
        if adaptive and self.pending >= base:
            size = min(maximum, self.pending // base * base)
        output, remaining, last_received = [], size, 0.0
        while remaining:
            block, received_at = self.parts.popleft()
            count = min(remaining, len(block))
            output.append(block[:count])
            last_received = received_at
            if count < len(block):
                self.parts.appendleft((block[count:], received_at))
            remaining -= count
        start = self.taken
        self.taken += size
        self.pending -= size
        return start, np.concatenate(output), last_received


class SessionLog:
    def __init__(self, directory, session_id):
        self.handle = None
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            # Exclusive UUID filename, never derived from a client file name/path.
            self.handle = (directory / f'{session_id}.jsonl').open('x', encoding='utf-8')

    def write(self, event):
        if self.handle:
            self.handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()


def validate_start(data, settings):
    if not isinstance(data, dict) or data.get('type') != 'start':
        raise ClientError('bad_start', 'First message must be a start JSON object.')
    if settings.token and not hmac.compare_digest(str(data.get('token', '')).encode('utf-8'), settings.token.encode('utf-8')):
        raise ClientError('unauthorized', 'Invalid access token.')
    source, target = data.get('source_lang'), data.get('target_lang')
    if f'{source}-{target}' not in settings.directions:
        raise ClientError('direction', 'This model is not configured for that translation direction.')
    chunk = data.get('chunk_seconds', settings.chunk_seconds)
    offset = data.get('media_offset_s', 0)
    for name, value in (('chunk_seconds', chunk), ('media_offset_s', offset)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ClientError('bad_start', f'{name} must be a finite number.')
    if not .5 <= chunk <= settings.max_chunk_seconds or offset < 0:
        raise ClientError('bad_start', 'Invalid chunk duration or media offset.')
    if not isinstance(data.get('adaptive', True), bool):
        raise ClientError('bad_start', 'adaptive must be a boolean.')
    if data.get('source_kind') not in ('file', 'microphone', 'system'):
        raise ClientError('bad_start', 'source_kind must be file, system or microphone.')
    include_microphone = data.get('include_microphone', False)
    if not isinstance(include_microphone, bool):
        raise ClientError('bad_start', 'include_microphone must be a boolean.')
    if include_microphone and data['source_kind'] != 'system':
        raise ClientError('bad_start', 'include_microphone is only supported with system audio.')
    if data['source_kind'] == 'system' and offset != 0:
        raise ClientError('bad_start', 'System audio starts at sample zero, without a file offset.')
    media_id = data.get('media_id')
    if media_id is not None and (data['source_kind'] != 'file' or not isinstance(media_id, str)
                                 or len(media_id) != 32 or any(c not in '0123456789abcdef' for c in media_id)):
        raise ClientError('bad_start', 'media_id must identify an uploaded file.')
    return dict(source_lang=source, target_lang=target, chunk_seconds=chunk,
                adaptive=data.get('adaptive', True), source_kind=data['source_kind'],
                source_name=str(data.get('source_name', ''))[:256], media_offset_s=offset,
                media_id=media_id,
                **({'include_microphone': include_microphone} if data['source_kind'] == 'system' else {}))


def origin_allowed(ws, settings):
    origin = ws.headers.get('origin')
    if not origin:  # Non-browser clients may authenticate using the same start message.
        return True
    if origin in settings.allowed_origins:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    # Browser default: same host AND matching transport security, not arbitrary CORS.
    scheme = 'https' if ws.url.scheme in ('wss', 'https') else 'http'
    return parsed.scheme == scheme and parsed.netloc == ws.headers.get('host') and not parsed.path


class PlaybackResponse(StreamingResponse):
    """Release a media reader even if ASGI fails before iterating the body."""
    def __init__(self, *args, release, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = release

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.release()


def create_app(settings=None):
    settings = settings or Settings()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='flex-gpu')
        app.state.media_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='flex-decode')
        if settings.temp_dir is not None:
            Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)
        app.state.media = MediaStore(max_slots=max(settings.max_uploads, settings.max_sessions), temp_dir=settings.temp_dir)
        app.state.results = ResultStore(temp_dir=settings.temp_dir, max_bytes=settings.max_result_bytes,
                                       ttl_seconds=settings.result_retention_hours * 3600,
                                       max_sessions=max(8, settings.max_sessions))
        expiry_task = None
        app.state.active_sessions = set()
        app.state.scheduler = None
        app.state.status = 'loading'
        app.state.error = None
        options = settings.model_options or ModelOptions(model=settings.model,
                                                         max_chunk_seconds=settings.max_chunk_seconds)
        options = replace(options, max_batch_size=settings.batch_size)
        app.state.model_options = options
        try:
            make = functools.partial(ModelEngine, options, backend=settings.backend)
            app.state.engine = await asyncio.get_running_loop().run_in_executor(app.state.executor, make)
            await asyncio.get_running_loop().run_in_executor(
                app.state.executor, app.state.engine.warmup, min(1.0, settings.max_chunk_seconds))
            app.state.scheduler = InferenceScheduler(
                app.state.engine, app.state.executor, max_sessions=settings.max_sessions,
                max_batch_size=settings.batch_size, batch_wait_ms=settings.batch_wait_ms,
                slow_inference_seconds=settings.slow_inference_seconds,
                stalled_inference_seconds=settings.stalled_inference_seconds)
            app.state.scheduler.start()
            app.state.status = 'ready'
            async def expire_uploads():
                while True:
                    await asyncio.sleep(30)
                    app.state.media.expire()
                    app.state.results.expire()
            expiry_task = asyncio.create_task(expire_uploads())
            yield
        finally:
            app.state.status = 'stopped'
            try:
                if app.state.scheduler is not None:
                    await app.state.scheduler.close()
                if getattr(app.state, 'engine', None) is not None:
                    await asyncio.get_running_loop().run_in_executor(app.state.executor, app.state.engine.close)
            finally:
                app.state.executor.shutdown(wait=True, cancel_futures=True)
                if expiry_task:
                    expiry_task.cancel()
                    await asyncio.gather(expiry_task, return_exceptions=True)
                app.state.media_executor.shutdown(wait=True, cancel_futures=True)
                app.state.media.close()
                app.state.results.close()

    app = FastAPI(title='FLEX Live', lifespan=lifespan, docs_url=None, redoc_url=None)

    def capacity():
        count = len(app.state.active_sessions)
        return dict(active_sessions=count, max_sessions=settings.max_sessions,
                    available_sessions=max(0, settings.max_sessions - count),
                    busy=count >= settings.max_sessions,
                    engine_health=app.state.scheduler.health())

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Permissions-Policy'] = 'microphone=(self), display-capture=(self), camera=()'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self' ws: wss:; media-src 'self' blob:; img-src 'self' data:; "
            "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/config')
    async def config():
        return dict(backend=settings.backend, model_label=Path(settings.model).name or 'Protocol demo',
                    directions=settings.directions, sample_rate=RATE,
                    chunk_seconds=settings.chunk_seconds, max_chunk_seconds=settings.max_chunk_seconds,
                    max_pending_seconds=settings.max_pending_seconds,
                    max_session_seconds=settings.max_session_seconds, auth_required=bool(settings.token),
                    file_mode='uploaded', input_sources=('file', 'system'),
                    max_upload_bytes=settings.max_upload_bytes,
                    result_retention_hours=settings.result_retention_hours,
                    context_policy=app.state.model_options.context_policy,
                    history_window_seconds=app.state.model_options.history_window_seconds,
                    history_retain_seconds=app.state.model_options.history_retain_seconds,
                    max_batch_size=settings.batch_size, batch_wait_ms=settings.batch_wait_ms,
                    slow_inference_seconds=settings.slow_inference_seconds,
                    stalled_inference_seconds=settings.stalled_inference_seconds,
                    status=app.state.status, **capacity())

    def require_media_access(request):
        if not origin_allowed(request, settings):
            raise HTTPException(403, detail=dict(code='origin', message='Cross-origin file upload is not allowed.'))
        supplied = request.headers.get('authorization', '')
        expected = f'Bearer {settings.token}'
        if settings.token and not hmac.compare_digest(supplied.encode('utf-8'), expected.encode('utf-8')):
            raise HTTPException(401, detail=dict(code='unauthorized', message='Invalid access token.'))

    @app.post('/api/media')
    async def upload_media(request: Request):
        require_media_access(request)
        item = None
        decode_future = None
        succeeded = False
        try:
            declared = request.headers.get('content-length')
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError as exc:
                    raise MediaError('upload_size', 'Invalid Content-Length.') from exc
                if length < 0 or length > settings.max_upload_bytes:
                    raise MediaError('upload_size', f'File exceeds upload limit ({settings.max_upload_bytes} bytes).', 413)
            item = app.state.media.reserve(request.query_params.get('name', 'media'))
            async def receive_file():
                total = 0
                with item.raw_path.open('xb') as output:
                    async for piece in request.stream():
                        total += len(piece)
                        if total > settings.max_upload_bytes:
                            raise MediaError('upload_size', f'File exceeds upload limit ({settings.max_upload_bytes} bytes).', 413)
                        output.write(piece)
                if not total:
                    raise MediaError('empty_upload', 'Choose a non-empty audio or video file.')
            await asyncio.wait_for(receive_file(), settings.upload_timeout_seconds)
            decode_future = asyncio.get_running_loop().run_in_executor(
                app.state.media_executor, functools.partial(
                    decode_upload, item, settings.max_session_seconds,
                    timeout=settings.decode_timeout_seconds, ffmpeg=settings.ffmpeg,
                    max_playback_bytes=settings.max_upload_bytes))
            try:
                await asyncio.shield(decode_future)
            except asyncio.CancelledError:
                # Do not delete an input while its decoding worker still owns it.
                # A server shutdown/disconnect can cancel the HTTP handler more
                # than once. Every cancellation must retain the same real worker,
                # rather than allowing a second cancel to jump to directory cleanup.
                while not decode_future.done():
                    try:
                        await asyncio.shield(decode_future)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break  # The worker has finished with a decoding error.
                with suppress(Exception, asyncio.CancelledError):
                    decode_future.result()  # Consume worker failure; cancellation wins.
                raise
            item.ready = True
            item.created = time.monotonic()
            item.raw_path.unlink()  # Keep PCM and, for WMV, a bounded browser-playable MP4.
            succeeded = True
            response = JSONResponse(item.public())
            if item.transcoded and settings.token:
                # HTML video cannot attach a Bearer header. Grant only this
                # file's playback path, never expose the service token in URLs.
                response.set_cookie('FLEX_MEDIA_PLAYBACK', item.playback_secret,
                                    path=f'/api/media/{item.media_id}/playback',
                                    max_age=math.ceil(settings.max_session_seconds + app.state.media.ttl_seconds),
                                    httponly=True, samesite='strict', secure=request.url.scheme == 'https')
            return response
        except MediaError as exc:
            raise HTTPException(exc.status, detail=dict(code=exc.code, message=str(exc))) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise HTTPException(408, detail=dict(code='upload_timeout', message='Upload timed out. Please retry with a smaller file.')) from exc
        except OSError as exc:
            raise HTTPException(507, detail=dict(code='media_storage', message='Temporary media storage failed. Check free disk space and --temp-dir on the server.')) from exc
        finally:
            if item is not None and not succeeded:
                app.state.media.discard(item.media_id, allow_claimed=True)

    @app.delete('/api/media/{media_id}')
    async def delete_media(media_id: str, request: Request):
        require_media_access(request)
        try:
            app.state.media.discard(media_id)
        except MediaError as exc:
            raise HTTPException(exc.status, detail=dict(code=exc.code, message=str(exc))) from exc
        return dict(deleted=True)

    @app.api_route('/api/media/{media_id}/playback', methods=['GET', 'HEAD'])
    async def playback_media(media_id: str, request: Request):
        # No cross-site embedding, even on tokenless trusted intranet deployments.
        if (not origin_allowed(request, settings)
                or request.headers.get('sec-fetch-site') in ('cross-site', 'same-site')):
            raise HTTPException(403, detail=dict(code='origin', message='Cross-origin media playback is not allowed.'))
        try:
            item = app.state.media.playback(media_id)
        except MediaError as exc:
            raise HTTPException(exc.status, detail=dict(code=exc.code, message=str(exc))) from exc
        bearer = request.headers.get('authorization', '')
        cookie = request.cookies.get('FLEX_MEDIA_PLAYBACK', '')
        if settings.token and not (
                hmac.compare_digest(bearer.encode('utf-8'), f'Bearer {settings.token}'.encode('utf-8'))
                or hmac.compare_digest(cookie.encode('utf-8'), item.playback_secret.encode('utf-8'))):
            raise HTTPException(401, detail=dict(code='unauthorized', message='Upload this file in the current browser before playing it.'))
        handle = None
        try:
            size = item.playback_path.stat().st_size
            etag = f'"flex-{item.media_id}-{size}"'
            response_headers = {'Accept-Ranges': 'bytes', 'ETag': etag}
            first, last, status_code = 0, size - 1, 200
            requested = request.headers.get('range', '') if request.method == 'GET' else ''
            if requested and request.headers.get('if-range', etag) == etag:
                try:
                    unit, _, interval = requested.partition('=')
                    if unit != 'bytes' or ',' in interval or '-' not in interval:
                        raise ValueError('Only a single bytes range is supported.')
                    left, right = interval.split('-', 1)
                    if left:
                        if not left.isdecimal() or (right and not right.isdecimal()):
                            raise ValueError('Invalid byte range.')
                        first, last = int(left), min(int(right) if right else size - 1, size - 1)
                    else:
                        if not right.isdecimal() or int(right) <= 0:
                            raise ValueError('Invalid suffix range.')
                        first, last = max(0, size - int(right)), size - 1
                    if first >= size or first < 0 or last < first:
                        raise ValueError('Byte range is outside the file.')
                except ValueError:
                    return Response(status_code=416, headers={**response_headers, 'Content-Range': f'bytes */{size}'})
                status_code = 206
                response_headers['Content-Range'] = f'bytes {first}-{last}/{size}'
            response_headers['Content-Length'] = str(last - first + 1)
            if request.method == 'HEAD':
                return Response(status_code=200, media_type='video/mp4', headers=response_headers)
            handle = app.state.media.open_playback(item)
            handle.seek(first)
        except OSError as exc:
            if handle is not None:
                app.state.media.close_playback(item, handle)
            raise HTTPException(404, detail=dict(code='media_missing', message='Temporary playback file is unavailable. Upload it again.')) from exc

        def close_reader():
            app.state.media.close_playback(item, handle)

        async def chunks():
            remaining = last - first + 1
            while remaining:
                block = handle.read(min(128 * 1024, remaining))
                if not block:
                    raise OSError('Playback file ended before the requested range.')
                remaining -= len(block)
                yield block

        return PlaybackResponse(chunks(), status_code=status_code, media_type='video/mp4',
                                headers=response_headers, release=close_reader)

    @app.get('/api/health')
    async def health():
        data = dict(status=app.state.status, backend=settings.backend, **capacity())
        return JSONResponse(data, status_code=200 if data['engine_health']['accepting'] else 503)

    @app.get('/api/sessions/{session_id}/export')
    async def export_result(session_id: str, request: Request, format: str = 'json'):
        require_media_access(request)
        if format not in ('txt', 'json'):
            raise HTTPException(400, detail=dict(code='export_format', message='Use format=txt or format=json.'))
        app.state.results.expire()
        item = app.state.results.items.get(session_id)
        if item is None:
            raise HTTPException(404, detail=dict(code='result_missing', message='Session results expired or the server restarted. Download results before leaving the session.'))
        snapshot = item.snapshot()
        iterator = snapshot.iter_text() if format == 'txt' else snapshot.iter_json()
        return StreamingResponse(iterator,
                                 media_type='text/plain; charset=utf-8' if format == 'txt' else 'application/json',
                                 headers={'Content-Disposition': f'attachment; filename="FLEX-{session_id}.{format}"'},
                                 background=BackgroundTask(snapshot.close))

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.websocket('/ws')
    async def websocket(ws: WebSocket):
        if not origin_allowed(ws, settings):
            await ws.close(code=1008)
            return
        await ws.accept()
        session_id = uuid.uuid4().hex
        owned = False
        journal = None
        result_store = None
        tasks = []
        send_lock = asyncio.Lock()
        begin = time.monotonic()
        buffer = None
        processed, in_flight = 0, False
        translation = []  # Small backward-compatible final preview only.
        translation_chars = turn_count = 0
        preview_complete = True
        last_tag = None
        failure = None
        early_error_sent = False
        model_session = None
        aborted = asyncio.Event()
        disconnected = False
        uploaded = None
        processing_monitor = ProcessingMonitor(settings.chunk_seconds)

        async def send(event_type, **fields):
            event = dict(type=event_type, session_id=session_id,
                         utc=datetime.now(timezone.utc).isoformat(),
                         server_elapsed_s=time.monotonic() - begin, **fields)
            if result_store:
                try:
                    result_store.write(event)
                except Exception:
                    if event_type not in ('error', 'done'):
                        raise
                    # Preserve termination diagnostics even when the disk budget
                    # cannot hold another event. Exported turns remain a valid prefix.
                    result_store.metadata[event_type] = event
                    LOG.exception('Could not persist final export event for %s', session_id)
            if journal:
                try:
                    journal.write(event)
                except Exception:
                    if event_type not in ('error', 'done'):
                        raise
                    LOG.exception('Could not persist final event for %s', session_id)
            async with send_lock:
                await asyncio.wait_for(ws.send_json(event), timeout=5)

        def stats():
            return dict(received_audio_s=buffer.received / RATE,
                        processed_audio_s=processed / RATE,
                        pending_audio_s=buffer.pending / RATE,
                        unprocessed_audio_s=(buffer.received - processed) / RATE,
                        in_flight=in_flight,
                        inference_state=app.state.scheduler.state(session_id),
                        processing_health=processing_monitor.update((buffer.received - processed) / RATE),
                        input_finished=buffer.closed,
                        context=getattr(model_session, 'context_metadata', {}), **capacity())

        def result_summary():
            summary = dict(translation_chars=translation_chars, turn_count=turn_count,
                           results_available=result_store is not None,
                           translation_truncated=not preview_complete)
            if preview_complete:
                summary['translation'] = ''.join(translation)
            return summary

        async def receive():
            nonlocal failure, disconnected, early_error_sent
            try:
                # A completed receive can race with cancellation inside
                # asyncio.wait_for on older Python versions. The explicit stop
                # flag prevents the reader from starting another long receive.
                while not aborted.is_set():
                    message = await asyncio.wait_for(ws.receive(), settings.idle_timeout_seconds)
                    if message['type'] == 'websocket.disconnect':
                        disconnected = True
                        raise ClientError('disconnected', 'Client disconnected; unprocessed audio was not translated.')
                    if message.get('bytes') is not None:
                        buffer.push(message['bytes'])
                    else:
                        raw = message.get('text', '')
                        if len(raw) > 4096:
                            raise ClientError('bad_control', 'Control message too large.')
                        try:
                            data = json.loads(raw)
                        except ValueError as exc:
                            raise ClientError('bad_control', 'Control message must be valid JSON.') from exc
                        kind = data.get('type') if isinstance(data, dict) else None
                        if kind == 'progress' and isinstance(buffer, FileQueue):
                            buffer.advance(data.get('position_s'))
                            continue
                        if kind == 'finish':
                            if isinstance(buffer, FileQueue):
                                if not isinstance(data.get('ended', False), bool):
                                    raise ClientError('bad_control', 'ended must be a boolean.')
                                buffer.advance(data.get('position_s'))
                            buffer.finish()
                            # Keep observing cancellation/disconnection while the
                            # worker drains. The worker closes the reader on EOF.
                            continue
                        if kind == 'cancel':
                            raise ClientError('cancelled', 'Session cancelled; pending audio may remain untranslated.')
                        if kind == 'ping':
                            await send('pong')
                        else:
                            raise ClientError('bad_control', 'Expected progress (uploaded files), finish, cancel or ping.')
            except Exception as exc:
                failure = exc
                aborted.set()
                app.state.scheduler.cancel_pending(session_id)
                buffer.finish()
                # Stop browser capture promptly on an input/session fault, even
                # if the shared GPU is still busy. This is NOT a done event:
                # admission/history remain owned until real thread completion.
                if not disconnected:
                    code = (exc.code if isinstance(exc, (ClientError, MediaError)) else
                            'timeout' if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else
                            'server_error')
                    with suppress(Exception):
                        await send('error', code=code, message=str(exc), **stats())
                        early_error_sent = True

        async def process(model_session, start):
            nonlocal processed, in_flight, failure, last_tag, translation_chars, turn_count, preview_complete
            base = round(start['chunk_seconds'] * RATE)
            maximum = round(settings.max_chunk_seconds * RATE)
            turn = 0
            previous_rollovers = 0
            try:
                while not aborted.is_set():
                    block = await buffer.take(base, maximum, start['adaptive'])
                    if block is None or aborted.is_set():
                        return
                    sample_start, audio, arrived = block
                    model_audio = audio
                    minimum = min(RATE, base)  # Match training/CLI's default one-second final tail.
                    if len(audio) < minimum:
                        model_audio = np.pad(audio, (0, minimum - len(audio)))
                    in_flight = True
                    await send('status', **stats())
                    submitted = time.monotonic()
                    # A cancelled queued job can be removed immediately. A job
                    # already in a GPU batch retains ownership until that batch
                    # returns; the scheduler never calls the shared engine concurrently.
                    scheduled = await app.state.scheduler.generate(session_id, model_session, model_audio)
                    if aborted.is_set():
                        return  # Never commit or publish after cancellation.
                    result = scheduled.result
                    try:
                        if result.finish_reason != 'stop':
                            raise ProtocolError(f'Generation ended with {result.finish_reason!r}; no partial output committed.')
                        token_tag, delta = parse_reply(result.reply)
                    except ProtocolError:
                        rejected = dict(type='rejected_turn', session_id=session_id, turn_id=turn,
                                        response=result.reply, token_ids=list(result.token_ids),
                                        finish_reason=result.finish_reason,
                                        source_start_s=sample_start / RATE,
                                        source_end_s=(sample_start + len(audio)) / RATE)
                        LOG.error('Rejected model output session=%s turn=%s finish=%r reply=%r token_ids=%r',
                                  session_id, turn, result.finish_reason, result.reply, result.token_ids)
                        if journal:
                            journal.write(rejected)
                        raise
                    tag = token_tag[2:-2]  # Wire/UI label; original token remains in model history.
                    model_session.commit(model_audio, result)
                    processed = sample_start + len(audio)
                    last_tag = tag
                    in_flight = False
                    translation_chars += len(delta)
                    turn_count += 1
                    if preview_complete and translation_chars <= 16384:
                        if delta:
                            translation.append(delta)
                    else:
                        preview_complete = False
                        translation.clear()
                    context = getattr(model_session, 'context_metadata', {})
                    if context.get('rollovers', 0) != previous_rollovers:
                        previous_rollovers = context['rollovers']
                        await send('context_window', turn_id=turn, context=context)
                    await send('turn', turn_id=turn, tag=tag, delta=delta, response=result.reply,
                               source_start_s=sample_start / RATE, source_end_s=processed / RATE,
                               media_offset_s=start['media_offset_s'], audio_seconds=len(audio) / RATE,
                               model_audio_seconds=len(model_audio) / RATE,
                               service_s=scheduled.service_s, queue_wait_s=max(0, submitted - arrived),
                               scheduler_wait_s=scheduled.queue_wait_s, batch_size=scheduled.batch_size,
                               prompt_tokens=result.prompt_tokens, **stats())
                    turn += 1
            except Exception as exc:
                # The receiver's cancellation/disconnect reason is authoritative
                # when it removed this job from the scheduler's pending queue.
                if not (aborted.is_set() and failure is not None):
                    failure = exc
                aborted.set()
                buffer.finish()
            finally:
                in_flight = False

        async def ticker():
            while not aborted.is_set():
                await asyncio.sleep(1)
                if aborted.is_set():
                    return
                await send('status', **stats())

        try:
            message = await asyncio.wait_for(ws.receive(), 10)
            raw = message.get('text', '')
            if not raw or len(raw) > 4096:
                raise ClientError('bad_start', 'Expected a small start JSON message before audio.')
            try:
                start = validate_start(json.loads(raw), settings)
            except (ValueError, TypeError) as exc:
                raise ClientError('bad_start', str(exc)) from exc
            model_health = app.state.scheduler.health()
            if not model_health['accepting']:
                code = ('engine_stalled' if model_health['state'] == 'stalled' else
                        'engine_failed' if model_health['state'] == 'failed' else 'engine_unavailable')
                raise ClientError(code, model_health['message'] or '模型服务暂不可用，请稍后重试。')
            if len(app.state.active_sessions) >= settings.max_sessions:
                raise ClientError('busy', f'All {settings.max_sessions} simultaneous session slots are in use. Retry after one session finishes.')
            app.state.active_sessions.add(session_id)  # Atomic check + acquire, no await.
            owned = True
            journal = SessionLog(settings.log_dir, session_id)
            result_store = app.state.results.create(session_id)
            if start['media_id']:
                uploaded = app.state.media.claim(start['media_id'])
                buffer = FileQueue(uploaded, start['media_offset_s'], round(start['chunk_seconds'] * RATE))
                start['adaptive'] = False
            else:
                buffer = PCMQueue(settings.max_pending_seconds, settings.max_session_seconds)
            processing_monitor = ProcessingMonitor(start['chunk_seconds'])
            model_session = app.state.engine.new_session(start['target_lang'])
            file_fields = dict(file_mode='uploaded', **uploaded.public()) if uploaded else {}
            await send('ready', sample_rate=RATE, backend=settings.backend,
                       chunk_seconds=start['chunk_seconds'], max_chunk_seconds=settings.max_chunk_seconds,
                       max_session_seconds=settings.max_session_seconds, config=start,
                       max_batch_size=settings.batch_size, **capacity(),
                       results_available=True, result_retention_hours=settings.result_retention_hours,
                       **{k: v for k, v in file_fields.items() if k != 'sample_rate'})
            reader = asyncio.create_task(receive())
            worker = asyncio.create_task(process(model_session, start))
            status = asyncio.create_task(ticker())
            tasks = [reader, worker, status]
            # Reader may finish normally (EOF), fail, or disconnect. The worker must
            # drain on EOF, but must not be killed halfway through a GPU request.
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if status in done:
                failure = status.exception() or RuntimeError('status task ended unexpectedly')
                aborted.set()
                app.state.scheduler.cancel_pending(session_id)
                buffer.finish()
            if worker in done:
                aborted.set()
                reader.cancel()
            await worker
            aborted.set()
            reader.cancel()
            status.cancel()
            await asyncio.gather(reader, status, return_exceptions=True)
            if failure:
                raise failure
            await send('done', reason='finished', last_tag=last_tag,
                       final_wait=last_tag == 'WAIT', **result_summary(), **stats())
        except Exception as exc:
            failure = exc
            if isinstance(exc, (ClientError, MediaError, SchedulerUnavailable)):
                code = exc.code
            elif isinstance(exc, ModelBackendError):
                code = 'engine_failed'
            elif isinstance(exc, ContextLimit):
                code = 'context_limit'
            elif isinstance(exc, ProtocolError):
                code = 'model_protocol'
            elif isinstance(exc, ResultLimit):
                code = 'result_limit'
            elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                code = 'timeout'
            else:
                code = 'server_error'
                LOG.exception('Session %s failed', session_id)
            if not early_error_sent:
                with suppress(Exception):
                    await send('error', code=code, message=str(exc), **(stats() if buffer else capacity()))
            with suppress(Exception):
                await send('done', reason=code, **result_summary(), **(stats() if buffer else capacity()))
        finally:
            aborted.set()
            app.state.scheduler.cancel_pending(session_id)
            if buffer:
                buffer.finish()
            # Do not release admission while an old model generation is still alive.
            for task in tasks:
                if task is not tasks[1]:
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if model_session is not None:
                try:
                    model_session.close()
                except Exception:
                    LOG.exception('Could not release history for %s', session_id)
            if isinstance(buffer, FileQueue):
                buffer.close()
            if uploaded is not None:
                try:
                    app.state.media.discard(uploaded.media_id, allow_claimed=True)
                except Exception:
                    LOG.exception('Could not remove temporary upload for %s', session_id)
            if owned:
                app.state.active_sessions.discard(session_id)
            if journal:
                try:
                    journal.close()
                except Exception:
                    LOG.exception('Could not close event log for %s', session_id)
            if result_store:
                try:
                    result_store.finish()
                except Exception:
                    LOG.exception('Could not close result spool for %s', session_id)
            with suppress(Exception):
                await ws.close(code=1000 if not failure and not disconnected else 1011)

    app.mount('/static', StaticFiles(directory=str(STATIC), check_dir=False), name='static')
    return app


def main():
    parser = argparse.ArgumentParser(description='FLEX Live: local bilingual SFT model + browser streaming UI')
    parser.add_argument('--backend', choices=['vllm', 'mock'], default='vllm')
    parser.add_argument('--model', default='', help='Local merged SFT checkpoint directory (not a LoRA-only adapter)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--https-host', action='append', default=[], metavar='IP_OR_NAME',
                        help='Generate/reuse a local TLS certificate for this browser-facing IP/name; repeat for aliases')
    parser.add_argument('--cert-dir', type=Path, default=Path('certs'),
                        help='Persistent local CA and server certificate directory (default: certs)')
    parser.add_argument('--ssl-certfile', type=Path, help='Use an existing PEM certificate/chain instead of --https-host')
    parser.add_argument('--ssl-keyfile', type=Path, help='Unencrypted PEM key matching --ssl-certfile')
    parser.add_argument('--token-env', default='FLEX_ACCESS_TOKEN',
                        help='Optional access token environment variable; unset or empty disables authentication')
    parser.add_argument('--allow-origin', action='append', default=[], help='Exact extra allowed Origin; useful behind TLS proxy')
    parser.add_argument('--directions', nargs='+', choices=SUPPORTED_DIRECTIONS, default=list(SUPPORTED_DIRECTIONS))
    parser.add_argument('--log-dir', type=Path, default=Path('logs'))
    parser.add_argument('--no-log', action='store_true', help='Disable server transcript/event logging')
    parser.add_argument('--chunk-seconds', type=float, default=1.0)
    parser.add_argument('--max-chunk-seconds', type=float, default=3.0)
    parser.add_argument('--max-pending-seconds', type=float, default=12.0)
    parser.add_argument('--max-session-seconds', type=float, default=28800.0, help='Total input duration cap; default 8 hours, separate from rolling model context')
    parser.add_argument('--max-sessions', type=int, default=4, help='Maximum independent simultaneous sessions sharing one model (default 4)')
    parser.add_argument('--max-batch-size', type=int, default=None, help='Ready turns per GPU call; default max-sessions, use 1 for serialized compatibility')
    parser.add_argument('--batch-wait-ms', type=float, default=10, help='Maximum ready-turn collection window, milliseconds (0..100)')
    parser.add_argument('--slow-inference-seconds', type=float, default=5,
                        help='Warn when a running model batch exceeds this duration; no cancellation (default 5)')
    parser.add_argument('--stalled-inference-seconds', type=float, default=60,
                        help='Suspected stall: reject new work but retain running GPU ownership (default 60)')
    parser.add_argument('--context-policy', choices=['rolling', 'stop'], default='rolling')
    parser.add_argument('--history-window-seconds', type=float, default=120.0)
    parser.add_argument('--history-retain-seconds', type=float, default=60.0)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-audio-turns', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=160)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.90)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'auto'], default='bfloat16')
    parser.add_argument('--protocol-form', choices=['auto', 'special', 'text'], default='auto')
    parser.add_argument('--mm-processor-cache-gb', type=float, default=2.0)
    parser.add_argument('--no-prefix-caching', action='store_true')
    parser.add_argument('--image-slots', type=int, choices=[0, 1], default=1,
                        help='Qwen3-Omni initialization compatibility: 1 retains vision initialization; input remains audio-only')
    parser.add_argument('--ffmpeg', default=None, help='Optional FFmpeg executable path for uploaded audio/video decoding')
    parser.add_argument('--max-upload-mb', type=int, default=4096, help='Per-upload disk limit in MiB')
    parser.add_argument('--upload-timeout-seconds', type=float, default=3600)
    parser.add_argument('--decode-timeout-seconds', type=float, default=1800)
    parser.add_argument('--temp-dir', type=Path, default=None, help='Temporary media/results directory; choose a disk with space for long files')
    parser.add_argument('--max-result-mb', type=int, default=256, help='Temporary result budget per session in MiB; does not silently truncate exports')
    parser.add_argument('--result-retention-hours', type=float, default=24)
    args = parser.parse_args()
    if args.https_host and (args.ssl_certfile or args.ssl_keyfile):
        parser.error('Use either --https-host or --ssl-certfile + --ssl-keyfile, not both')
    if bool(args.ssl_certfile) != bool(args.ssl_keyfile):
        parser.error('--ssl-certfile and --ssl-keyfile must be supplied together')
    tls_paths = ([args.cert_dir] if args.https_host else
                 [path for path in (args.ssl_certfile, args.ssl_keyfile) if path is not None])
    if any(path.resolve() == STATIC or STATIC in path.resolve().parents for path in tls_paths):
        parser.error('TLS certificates/keys must be outside the publicly served static directory; use --cert-dir certs')
    # Trusted intranet deployments may run without an application-level token.
    # An explicitly configured token still protects both HTTP media and WS APIs.
    token = os.environ.get(args.token_env, '')
    if not 0 < args.gpu_memory_utilization < 1 or min(args.max_model_len, args.max_audio_turns, args.max_tokens) <= 0:
        parser.error('Invalid GPU/context configuration')
    options = ModelOptions(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                           max_audio_turns=args.max_audio_turns, max_tokens=args.max_tokens,
                           gpu_memory_utilization=args.gpu_memory_utilization,
                           protocol_form=args.protocol_form, max_chunk_seconds=args.max_chunk_seconds,
                           mm_processor_cache_gb=args.mm_processor_cache_gb,
                           context_policy=args.context_policy, history_window_seconds=args.history_window_seconds,
                           history_retain_seconds=args.history_retain_seconds,
                           prefix_caching=not args.no_prefix_caching, image_slots=args.image_slots)
    settings = Settings(backend=args.backend, model=args.model, directions=tuple(args.directions),
                        log_dir=None if args.no_log else args.log_dir, token=token,
                        allowed_origins=tuple(args.allow_origin), chunk_seconds=args.chunk_seconds,
                        max_chunk_seconds=args.max_chunk_seconds, max_pending_seconds=args.max_pending_seconds,
                        max_session_seconds=args.max_session_seconds, model_options=options, ffmpeg=args.ffmpeg,
                        max_sessions=args.max_sessions, max_batch_size=args.max_batch_size, batch_wait_ms=args.batch_wait_ms,
                        slow_inference_seconds=args.slow_inference_seconds,
                        stalled_inference_seconds=args.stalled_inference_seconds,
                        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
                        upload_timeout_seconds=args.upload_timeout_seconds, decode_timeout_seconds=args.decode_timeout_seconds,
                        temp_dir=args.temp_dir, max_result_bytes=args.max_result_mb * 1024 * 1024,
                        result_retention_hours=args.result_retention_hours)
    # Validate deployment/TLS before loading the GPU model. Default HTTP is unchanged.
    settings.validate()
    tls_options = {}
    try:
        if args.https_host:
            certificate = prepare_local_tls(args.https_host, args.cert_dir)
            tls_options = dict(ssl_certfile=str(certificate.certfile), ssl_keyfile=str(certificate.keyfile))
            print(f'[HTTPS] Local CA certificate: {certificate.ca_certfile}', flush=True)
            print('[HTTPS] First use: trust this CA on client computers; never distribute .key files.', flush=True)
            for host in args.https_host:
                address = f'[{host}]' if ':' in host else host
                print(f'[HTTPS] After model startup, open https://{address}:{args.port}', flush=True)
        elif args.ssl_certfile:
            validate_tls_files(args.ssl_certfile, args.ssl_keyfile)
            tls_options = dict(ssl_certfile=str(args.ssl_certfile), ssl_keyfile=str(args.ssl_keyfile))
    except TLSConfigError as exc:
        parser.error(str(exc))
    import uvicorn
    # A single worker is essential: each extra worker would load another GPU model.
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1,
                ws_max_size=32768, ws_max_queue=8, ws_ping_interval=20, ws_ping_timeout=20,
                timeout_graceful_shutdown=120, **tls_options)


if __name__ == '__main__':
    main()
