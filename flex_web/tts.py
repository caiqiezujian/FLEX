"""Optional Qwen3-TTS HTTP adapter. No TTS model or GPU is loaded here."""
from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
import io
import json
import logging
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid
import wave

import httpx

LOG = logging.getLogger('flex_web.tts')
DEFAULT_VOICES = {
    'en': {'audio': 'minimax_en_1_f.wav', 'text': "Hello. I'm delighted to assist you with our voice services. Choose a voice that resonates with you, and let's begin our creative audio journey together."},
    'zh': {'audio': 'minimax_zh_1_f.wav', 'text': '您好，很高兴能为您提供配音服务。选择您感兴趣的音色，让我们一起开启声音创作的奇幻之旅吧。'},
}
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 12 * 1024 * 1024


class TTSError(Exception):
    def __init__(self, code, message, status=502):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass
class TTSOptions:
    url: str = ''
    assets_dir: Path | None = None
    voices_config: Path | None = None
    timeout_seconds: float = 180.0
    max_text_chars: int = 200
    max_concurrent: int = 2
    x_vector_only: bool | None = None

    def validate(self):
        if self.url:
            parts = urlsplit(self.url)
            if (parts.scheme not in ('http', 'https') or not parts.hostname
                    or parts.username or parts.password or parts.query or parts.fragment):
                raise ValueError('--tts-url must be an http(s) service base URL without credentials/query')
            if not self.assets_dir and not self.voices_config:
                raise ValueError('--tts-url requires --tts-assets-dir or --tts-voices-config')
        elif self.assets_dir or self.voices_config:
            raise ValueError('TTS reference files require --tts-url')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('TTS timeout must be finite and positive')
        if type(self.max_text_chars) is not int or not 40 <= self.max_text_chars <= 500:
            raise ValueError('TTS max_text_chars must be 40..500')
        if type(self.max_concurrent) is not int or not 1 <= self.max_concurrent <= 16:
            raise ValueError('TTS max_concurrent must be 1..16')
        if self.x_vector_only is not None and type(self.x_vector_only) is not bool:
            raise ValueError('TTS x_vector_only must be bool or None')


class TTSService:
    def __init__(self, options: TTSOptions):
        options.validate()
        self.options = options
        self.voices = {}
        self.client = None
        self.active = 0
        self.error = ''
        if options.url:
            # A TTS setup error must not prevent text interpretation from starting.
            try:
                config = (json.loads(Path(options.voices_config).read_text(encoding='utf-8-sig'))
                          if options.voices_config else DEFAULT_VOICES)
                base = Path(options.assets_dir or Path(options.voices_config).parent)
                if not isinstance(config, dict):
                    raise ValueError('voice config must be an object with en/zh entries')
                for language in ('en', 'zh'):
                    entry = config.get(language)
                    if entry is None:
                        continue
                    if (not isinstance(entry, dict) or not isinstance(entry.get('text'), str)
                            or not entry['text'].strip() or not isinstance(entry.get('audio'), str)):
                        raise ValueError(f'{language}: audio path and matching nonempty text are required')
                    path = base / entry['audio']
                    if not 0 < path.stat().st_size <= MAX_REFERENCE_BYTES:
                        raise ValueError(f'{language}: reference audio exceeds 10 MiB or is empty')
                    audio = path.read_bytes()
                    # User-provided reference WAV bytes are sent verbatim; no ONNX/YAML is loaded.
                    self.voices[language] = dict(ref_audio_b64=base64.b64encode(audio).decode('ascii'),
                                                ref_text=entry['text'])
                if not self.voices:
                    raise ValueError('voice config has no en or zh voice')
            except (OSError, ValueError, TypeError) as exc:
                self.voices.clear()
                self.error = '参考音色配置不可用，请检查 Web 服务日志。'
                LOG.error('TTS reference setup failed: %s', exc)

    def public_config(self):
        return dict(enabled=bool(self.voices), languages=sorted(self.voices),
                    max_text_chars=self.options.max_text_chars,
                    max_concurrent=self.options.max_concurrent,
                    timeout_seconds=self.options.timeout_seconds,
                    message=self.error or ('' if self.voices else '服务端尚未配置译文播报。'))

    async def start(self):
        if self.voices:
            self.client = httpx.AsyncClient(
                trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(self.options.timeout_seconds, connect=10.0),
                limits=httpx.Limits(max_connections=self.options.max_concurrent,
                                   max_keepalive_connections=self.options.max_concurrent))

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def synthesize(self, text, language, *, diagnostics=None):
        # The optional mapping keeps the existing (WAV, rate, duration) API intact.
        diagnostics = diagnostics if diagnostics is not None else {}
        request_id = diagnostics.setdefault('request_id', uuid.uuid4().hex)
        started = time.perf_counter()
        admitted = False
        outcome = 'error'
        try:
            if not self.voices or not self.client:
                raise TTSError('tts_unavailable', self.error or '服务端未启用 TTS。', 503)
            if not isinstance(language, str) or language not in self.voices:
                raise TTSError('tts_language', '未配置该目标语言的参考音色。', 400)
            if (not isinstance(text, str) or not text.strip() or len(text) > self.options.max_text_chars
                    or not any(c.isalnum() for c in text) or '<|' in text or '|>' in text):
                raise TTSError('tts_text', '请提交限长的译文正文，不含策略标签或纯标点。', 400)
            # No unbounded server-side waiting room. The browser owns its ordered queue.
            if self.active >= self.options.max_concurrent:
                raise TTSError('tts_busy', '语音服务忙，请稍后重新开启播报；文字同传继续。', 503)
            self.active += 1
            admitted = True
            payload = dict(self.voices[language], text=text, language=language, request_id=request_id)
            if self.options.x_vector_only is not None:
                payload['x_vector_only'] = self.options.x_vector_only

            async def request_audio():
                url = self.options.url.rstrip('/') + '/api/tts/clone'
                submitted = time.perf_counter()
                async with self.client.stream('POST', url, json=payload) as response:
                    diagnostics['upstream_headers_seconds'] = time.perf_counter() - submitted
                    diagnostics['upstream_status'] = response.status_code
                    body = bytearray()
                    async for part in response.aiter_bytes():
                        body.extend(part)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise TTSError('tts_response_size', 'TTS 响应过大，已停止本次播报。')
                    diagnostics['upstream_body_seconds'] = time.perf_counter() - submitted
                    decoding = time.perf_counter()
                    try:
                        data = json.loads(body)
                    except (ValueError, UnicodeError) as exc:
                        raise TTSError('tts_response', 'TTS 服务没有返回有效 JSON。') from exc
                    if response.status_code != 200:
                        error = data.get('error', {}) if isinstance(data, dict) else {}
                        code = error.get('code', 'UNKNOWN') if isinstance(error, dict) else 'UNKNOWN'
                        LOG.warning('TTS request=%s HTTP=%s code=%s', request_id, response.status_code, code)
                        raise TTSError('tts_upstream', f'TTS 服务返回 HTTP {response.status_code}（{str(code)[:80]}）。')
                    audio = decode_audio(data)
                    diagnostics['decode_seconds'] = time.perf_counter() - decoding
                    return audio
            # Total timeout also covers trickling responses, unlike read timeout alone.
            audio = await asyncio.wait_for(request_audio(), timeout=self.options.timeout_seconds)
            outcome = 'ok'
            return audio
        except TTSError as exc:
            outcome = exc.code
            raise
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            outcome = 'tts_timeout'
            raise TTSError('tts_timeout', '语音合成超时；文字同传继续。', 504) from exc
        except httpx.HTTPError as exc:
            outcome = 'tts_connection'
            LOG.warning('TTS request=%s transport failed: %s', request_id, type(exc).__name__)
            raise TTSError('tts_connection', '无法连接 TTS 服务，请检查 Web 服务器到 TTS 的网络。') from exc
        finally:
            if admitted:
                self.active -= 1
            diagnostics['elapsed_seconds'] = time.perf_counter() - started
            LOG.info('TTS request=%s outcome=%s elapsed_s=%.6f upstream_headers_s=%s '
                     'upstream_body_s=%s decode_s=%s upstream_status=%s active=%s',
                     request_id, outcome, diagnostics['elapsed_seconds'],
                     diagnostics.get('upstream_headers_seconds'), diagnostics.get('upstream_body_seconds'),
                     diagnostics.get('decode_seconds'), diagnostics.get('upstream_status'), self.active)


def decode_audio(data):
    """Trust the actual WAV header, not upstream duration/sample-rate claims."""
    try:
        if not isinstance(data, dict) or not isinstance(data.get('audio_b64'), str):
            raise ValueError('missing audio_b64')
        encoded = data['audio_b64']
        if encoded.startswith('data:'):
            encoded = encoded.split(',', 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        with wave.open(io.BytesIO(raw), 'rb') as wav:
            rate, frames = wav.getframerate(), wav.getnframes()
            if (wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != 'NONE'
                    or not 8000 <= rate <= 96000 or not 0 < frames / rate <= 60):
                raise ValueError('expected nonempty PCM16 mono WAV <=60s')
            if len(wav.readframes(frames)) != frames * 2:
                raise ValueError('truncated WAV')
        return raw, rate, frames / rate
    except (ValueError, TypeError, IndexError, binascii.Error, wave.Error, EOFError) as exc:
        raise TTSError('tts_audio', 'TTS 返回的音频无效或不完整。') from exc
