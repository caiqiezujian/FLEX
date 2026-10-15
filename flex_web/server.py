"""Single-GPU WebSocket application. Browser capture and inference run independently.

Only complete, protocol-valid generations are committed. A disconnected client
never frees the GPU admission slot until its in-flight generation has finished.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import ContextLimit, ModelEngine, ModelOptions, ProtocolError, parse_reply

RATE = 16000
STATIC = Path(__file__).resolve().parent.parent / 'static'
LOG = logging.getLogger('flex_web')


@dataclass
class Settings:
    backend: str = 'vllm'
    model: str = ''
    log_dir: Path | None = field(default_factory=lambda: Path('logs'))
    directions: tuple[str, ...] = ('en-zh', 'zh-en')
    chunk_seconds: float = 1.0
    max_chunk_seconds: float = 3.0
    max_pending_seconds: float = 12.0
    max_session_seconds: float = 180.0
    token: str = ''
    allowed_origins: tuple[str, ...] = ()
    idle_timeout_seconds: float = 300.0
    model_options: ModelOptions | None = None

    def validate(self):
        if self.backend not in ('mock', 'vllm'):
            raise ValueError('backend must be mock or vllm')
        if self.backend == 'vllm' and not self.model:
            raise ValueError('--model is required for vllm (use a merged local checkpoint)')
        if not self.directions or any(x not in ('en-zh', 'zh-en') for x in self.directions):
            raise ValueError('directions must contain en-zh and/or zh-en')
        numeric = (self.chunk_seconds, self.max_chunk_seconds, self.max_pending_seconds,
                   self.max_session_seconds, self.idle_timeout_seconds)
        if not all(math.isfinite(v) and v > 0 for v in numeric):
            raise ValueError('time limits must be finite and positive')
        if not .5 <= self.chunk_seconds <= self.max_chunk_seconds <= 6:
            raise ValueError('require 0.5 <= chunk_seconds <= max_chunk_seconds <= 6')
        if self.max_pending_seconds < self.max_chunk_seconds:
            raise ValueError('max_pending_seconds must be >= max_chunk_seconds')


class ClientError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


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
            raise ClientError('session_limit', 'Session duration limit reached. Finish and start a new segment; history was not silently removed.')
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
    if data.get('source_kind') not in ('file', 'microphone'):
        raise ClientError('bad_start', 'source_kind must be file or microphone.')
    return dict(source_lang=source, target_lang=target, chunk_seconds=chunk,
                adaptive=data.get('adaptive', True), source_kind=data['source_kind'],
                source_name=str(data.get('source_name', ''))[:256], media_offset_s=offset)


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
    scheme = 'https' if ws.url.scheme == 'wss' else 'http'
    return parsed.scheme == scheme and parsed.netloc == ws.headers.get('host') and not parsed.path


def create_app(settings=None):
    settings = settings or Settings()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='flex-gpu')
        app.state.active = None
        app.state.status = 'loading'
        app.state.error = None
        options = settings.model_options or ModelOptions(model=settings.model,
                                                         max_chunk_seconds=settings.max_chunk_seconds)
        try:
            make = functools.partial(ModelEngine, options, backend=settings.backend)
            app.state.engine = await asyncio.get_running_loop().run_in_executor(app.state.executor, make)
            await asyncio.get_running_loop().run_in_executor(
                app.state.executor, app.state.engine.warmup, min(1.0, settings.max_chunk_seconds))
            app.state.status = 'ready'
            yield
        finally:
            app.state.status = 'stopped'
            try:
                if getattr(app.state, 'engine', None) is not None:
                    await asyncio.get_running_loop().run_in_executor(app.state.executor, app.state.engine.close)
            finally:
                app.state.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='FLEX Live', lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Permissions-Policy'] = 'microphone=(self), camera=()'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self' ws: wss:; media-src 'self' blob:; img-src 'self' data:; "
            "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/config')
    def config():
        return dict(backend=settings.backend, model_label=Path(settings.model).name or 'Protocol demo',
                    directions=settings.directions, sample_rate=RATE,
                    chunk_seconds=settings.chunk_seconds, max_chunk_seconds=settings.max_chunk_seconds,
                    max_pending_seconds=settings.max_pending_seconds,
                    max_session_seconds=settings.max_session_seconds, auth_required=bool(settings.token),
                    status=app.state.status, busy=app.state.active is not None)

    @app.get('/api/health')
    def health():
        return dict(status=app.state.status, backend=settings.backend, busy=app.state.active is not None)

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
        tasks = []
        send_lock = asyncio.Lock()
        begin = time.monotonic()
        buffer = None
        processed, in_flight = 0, False
        translation = []
        last_tag = None
        failure = None
        model_session = None
        aborted = asyncio.Event()
        disconnected = False

        async def send(event_type, **fields):
            event = dict(type=event_type, session_id=session_id,
                         utc=datetime.now(timezone.utc).isoformat(),
                         server_elapsed_s=time.monotonic() - begin, **fields)
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
                        in_flight=in_flight)

        async def receive():
            nonlocal failure, disconnected
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
                        if kind == 'finish':
                            buffer.finish()
                            # Keep observing cancellation/disconnection while the
                            # worker drains. The worker closes the reader on EOF.
                            continue
                        if kind == 'cancel':
                            raise ClientError('cancelled', 'Session cancelled; pending audio may remain untranslated.')
                        if kind == 'ping':
                            await send('pong')
                        else:
                            raise ClientError('bad_control', 'Expected finish, cancel or ping.')
            except Exception as exc:
                failure = exc
                aborted.set()
                buffer.finish()

        async def process(model_session, start):
            nonlocal processed, in_flight, failure, last_tag
            base = round(start['chunk_seconds'] * RATE)
            maximum = round(settings.max_chunk_seconds * RATE)
            turn = 0
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
                    # Do not cancel this future on disconnect: generation owns the GPU.
                    future = asyncio.get_running_loop().run_in_executor(
                        app.state.executor, model_session.generate, model_audio)
                    try:
                        result = await asyncio.shield(future)
                    except asyncio.CancelledError:
                        await asyncio.shield(future)
                        raise
                    finished = time.monotonic()
                    if aborted.is_set():
                        return  # Never commit or publish after cancellation.
                    if result.finish_reason != 'stop':
                        raise ProtocolError(f'Generation ended with {result.finish_reason!r}; no partial output committed.')
                    token_tag, delta = parse_reply(result.reply)
                    tag = token_tag[2:-2]  # Wire/UI label; original token remains in model history.
                    model_session.commit(model_audio, result)
                    processed = sample_start + len(audio)
                    last_tag = tag
                    in_flight = False
                    translation.append(delta)
                    await send('turn', turn_id=turn, tag=tag, delta=delta, response=result.reply,
                               source_start_s=sample_start / RATE, source_end_s=processed / RATE,
                               media_offset_s=start['media_offset_s'], audio_seconds=len(audio) / RATE,
                               model_audio_seconds=len(model_audio) / RATE,
                               service_s=finished - submitted, queue_wait_s=max(0, submitted - arrived),
                               prompt_tokens=result.prompt_tokens, **stats())
                    turn += 1
            except Exception as exc:
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
            if app.state.active is not None:
                raise ClientError('busy', 'GPU is in use by another session. Wait for it to finish.')
            app.state.active = session_id  # Atomic: no await between check and acquire.
            owned = True
            journal = SessionLog(settings.log_dir, session_id)
            buffer = PCMQueue(settings.max_pending_seconds, settings.max_session_seconds)
            model_session = app.state.engine.new_session(start['target_lang'])
            await send('ready', sample_rate=RATE, backend=settings.backend,
                       chunk_seconds=start['chunk_seconds'], max_chunk_seconds=settings.max_chunk_seconds,
                       max_session_seconds=settings.max_session_seconds, config=start)
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
            await send('done', reason='finished', translation=''.join(translation), last_tag=last_tag,
                       final_wait=last_tag == 'WAIT', **stats())
        except Exception as exc:
            failure = exc
            if isinstance(exc, ClientError):
                code = exc.code
            elif isinstance(exc, ContextLimit):
                code = 'context_limit'
            elif isinstance(exc, ProtocolError):
                code = 'model_protocol'
            elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                code = 'timeout'
            else:
                code = 'server_error'
                LOG.exception('Session %s failed', session_id)
            with suppress(Exception):
                await send('error', code=code, message=str(exc), **(stats() if buffer else {}))
            with suppress(Exception):
                await send('done', reason=code, translation=''.join(translation), **(stats() if buffer else {}))
        finally:
            aborted.set()
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
            if owned and app.state.active == session_id:
                app.state.active = None
            if journal:
                try:
                    journal.close()
                except Exception:
                    LOG.exception('Could not close event log for %s', session_id)
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
    parser.add_argument('--token-env', default='FLEX_ACCESS_TOKEN', help='Read access token from this environment variable')
    parser.add_argument('--allow-origin', action='append', default=[], help='Exact extra allowed Origin; useful behind TLS proxy')
    parser.add_argument('--directions', nargs='+', choices=['en-zh', 'zh-en'], default=['en-zh', 'zh-en'])
    parser.add_argument('--log-dir', type=Path, default=Path('logs'))
    parser.add_argument('--no-log', action='store_true', help='Disable server transcript/event logging')
    parser.add_argument('--chunk-seconds', type=float, default=1.0)
    parser.add_argument('--max-chunk-seconds', type=float, default=3.0)
    parser.add_argument('--max-pending-seconds', type=float, default=12.0)
    parser.add_argument('--max-session-seconds', type=float, default=180.0)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-audio-turns', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=160)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.90)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'auto'], default='bfloat16')
    parser.add_argument('--protocol-form', choices=['auto', 'special', 'text'], default='auto')
    parser.add_argument('--mm-processor-cache-gb', type=float, default=2.0)
    parser.add_argument('--no-prefix-caching', action='store_true')
    args = parser.parse_args()
    token = os.environ.get(args.token_env, '')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not token:
        parser.error('Non-loopback listening requires an access token. Set FLEX_ACCESS_TOKEN; use HTTPS or SSH forwarding.')
    if not 0 < args.gpu_memory_utilization < 1 or min(args.max_model_len, args.max_audio_turns, args.max_tokens) <= 0:
        parser.error('Invalid GPU/context configuration')
    options = ModelOptions(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                           max_audio_turns=args.max_audio_turns, max_tokens=args.max_tokens,
                           gpu_memory_utilization=args.gpu_memory_utilization,
                           protocol_form=args.protocol_form, max_chunk_seconds=args.max_chunk_seconds,
                           mm_processor_cache_gb=args.mm_processor_cache_gb,
                           prefix_caching=not args.no_prefix_caching)
    settings = Settings(backend=args.backend, model=args.model, directions=tuple(args.directions),
                        log_dir=None if args.no_log else args.log_dir, token=token,
                        allowed_origins=tuple(args.allow_origin), chunk_seconds=args.chunk_seconds,
                        max_chunk_seconds=args.max_chunk_seconds, max_pending_seconds=args.max_pending_seconds,
                        max_session_seconds=args.max_session_seconds, model_options=options)
    import uvicorn
    # A single worker is essential: each extra worker would load another GPU model.
    uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1,
                ws_max_size=32768, ws_max_queue=8, ws_ping_interval=20, ws_ping_timeout=20,
                timeout_graceful_shutdown=120)


if __name__ == '__main__':
    main()
