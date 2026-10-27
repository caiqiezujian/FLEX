"""File-to-text FLEX inference without the Web server or browser.

Uses the same local model adapter and SFT history as FLEX Live. A file is fed
in fixed chronological chunks, never as a full-recording model request.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import wave

import numpy as np

from flex_web.model import ModelEngine, ModelOptions, ProtocolError, SAMPLE_RATE, SUPPORTED_DIRECTIONS, parse_reply


LOG = logging.getLogger('flex.s2tt')


class AudioReader:
    """Bounded PCM reader. Native PCM16 WAV needs no external decoder.

    Other sample rates, channel layouts and formats use an FFmpeg pipe, not a
    temporary full-recording WAV. FFmpeg stderr goes to a temporary file so it
    cannot fill a pipe and deadlock audio reading.
    """

    def __init__(self, path, ffmpeg='ffmpeg'):
        self.path = Path(path).resolve()
        self.ffmpeg = ffmpeg
        self.wav = self.process = self.errors = None
        self.samples = 0
        self.eof = False

    def __enter__(self):
        if not self.path.is_file():
            raise ValueError(f'Audio file does not exist: {self.path}')
        try:
            candidate = wave.open(str(self.path), 'rb')
        except (wave.Error, EOFError):
            candidate = None
        if candidate is not None:
            if (candidate.getframerate(), candidate.getnchannels(),
                    candidate.getsampwidth(), candidate.getcomptype()) == (SAMPLE_RATE, 1, 2, 'NONE'):
                self.wav = candidate
                return self
            candidate.close()
        executable = shutil.which(self.ffmpeg)
        if executable is None:
            raise ValueError('This file needs FFmpeg. Install ffmpeg or supply --ffmpeg /path/to/ffmpeg; '
                             '16-kHz mono PCM16 WAV works without it.')
        self.errors = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(
                [executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
                 '-protocol_whitelist', 'file,pipe', '-i', str(self.path), '-map', '0:a:0',
                 '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 's16le', 'pipe:1'],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.errors)
        except BaseException:
            self.close()
            raise
        return self

    def read(self, count):
        if count < 1:
            raise ValueError('sample count must be positive')
        if self.eof:
            return np.empty(0, dtype=np.float32)
        if self.wav is not None:
            data = self.wav.readframes(count)
            if len(data) < count * 2:
                self.eof = True
                if self.samples + len(data) // 2 != self.wav.getnframes():
                    raise ValueError('Truncated WAV: decoded length does not match its header')
        else:
            pieces, remaining = [], count * 2
            while remaining:
                part = self.process.stdout.read(remaining)
                if not part:
                    self.eof = True
                    status = self.process.wait(timeout=10)
                    if status:
                        self.errors.seek(0)
                        detail = self.errors.read(4096).decode('utf-8', errors='replace').strip()
                        raise ValueError(f'FFmpeg failed ({status}): {detail}')
                    break
                pieces.append(part)
                remaining -= len(part)
            data = b''.join(pieces)
        if len(data) % 2:
            raise ValueError('Decoder returned incomplete PCM16 sample')
        self.samples += len(data) // 2
        return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0

    def close(self):
        try:
            if self.wav is not None:
                self.wav.close()
                self.wav = None
            if self.process is not None:
                try:
                    if self.process.poll() is None:
                        self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
                finally:
                    if self.process.stdout:
                        self.process.stdout.close()
                    self.process = None
        finally:
            if self.errors is not None:
                self.errors.close()
                self.errors = None

    def __exit__(self, *args):
        self.close()


class OutputFiles:
    """Create a NEW run directory; never append to or overwrite an older run."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.events = self.text = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            self.events = (self.directory / 'events.jsonl').open('x', encoding='utf-8')
            self.text = (self.directory / 'translation.txt').open('x', encoding='utf-8', newline='')
        except BaseException:
            self.close()
            raise
        return self

    def event(self, kind, **data):
        self.events.write(json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n')
        self.events.flush()

    def append(self, delta):
        self.text.write(delta)
        self.text.flush()

    def close(self):
        try:
            if self.text:
                self.text.close()
        finally:
            if self.events:
                self.events.close()

    def __exit__(self, *args):
        self.close()


@dataclass
class RunState:
    turns: int = 0
    processed_samples: int = 0
    read_samples: int = 0
    last_tag: str | None = None
    translation_chars: int = 0

    def stats(self):
        return dict(turns=self.turns, processed_audio_s=self.processed_samples / SAMPLE_RATE,
                    read_audio_s=self.read_samples / SAMPLE_RATE, last_tag=self.last_tag,
                    translation_chars=self.translation_chars)


def run_stream(args, engine, reader, output, state, *, clock=time.monotonic,
               sleep=time.sleep, stdout=None):
    """Serial file replay. Pace deadlines are absolute, never sleep-after-generate.

    The file stays on disk when the model is slow. No producer queue is needed;
    dispatch_lag_s reports lateness relative to the simulated 1x arrival clock.
    It is NOT AL/LAAL and does not measure live microphone/network latency.
    """
    stdout = sys.stdout if stdout is None else stdout
    chunk_samples = round(args.chunk_seconds * SAMPLE_RATE)
    duration_limit = round(args.max_audio_seconds * SAMPLE_RATE)
    session = engine.new_session(args.direction.split('-')[1])
    start = clock()
    previous_rollovers = 0
    try:
        while True:
            audio = reader.read(chunk_samples)
            state.read_samples += len(audio)
            if not len(audio):
                if not state.turns:
                    raise ValueError('Audio contains no decoded samples')
                break
            if state.read_samples > duration_limit:
                raise ValueError('max-audio-seconds exceeded; the prefix is saved but the file is '
                                 'NOT fully translated. Use a shorter file or explicitly raise the limit.')
            source_start = state.processed_samples / SAMPLE_RATE
            source_end = state.read_samples / SAMPLE_RATE
            if args.pace == 'realtime':
                # Do not release a chunk before its LAST sample has arrived.
                while True:
                    remaining = start + source_end - clock()
                    if remaining <= 0:
                        break
                    sleep(min(remaining, 0.25))
            dispatch = clock()
            model_audio = audio
            # Pad only a short final tail; never invent extra input turns at EOF.
            minimum = min(chunk_samples, SAMPLE_RATE)
            if len(audio) < minimum:
                model_audio = np.pad(audio, (0, minimum - len(audio)))
            result = session.generate(model_audio)
            try:
                if result.finish_reason != 'stop':
                    raise ProtocolError(f'finish_reason={result.finish_reason!r}; output not committed')
                token_tag, delta = parse_reply(result.reply)
                session.commit(model_audio, result)
            except ProtocolError as exc:
                output.event('rejected_turn', source_start_s=source_start, source_end_s=source_end,
                             response=result.reply, token_ids=list(result.token_ids),
                             finish_reason=result.finish_reason, message=str(exc))
                LOG.error('Rejected model reply: %r; token_ids=%s; finish_reason=%r',
                          result.reply, result.token_ids, result.finish_reason)
                raise
            completed = clock()
            state.turns += 1
            state.processed_samples = state.read_samples
            state.last_tag = token_tag[2:-2]
            state.translation_chars += len(delta)
            context = getattr(session, 'context_metadata', {})
            if context.get('rollovers', 0) != previous_rollovers:
                previous_rollovers = context['rollovers']
                output.event('context_window', turn_id=state.turns, context=context)
            event = dict(turn_id=state.turns, tag=state.last_tag, delta=delta,
                         response=result.reply, token_ids=list(result.token_ids),
                         source_start_s=source_start, source_end_s=source_end,
                         model_audio_s=len(model_audio) / SAMPLE_RATE,
                         service_s=completed - dispatch, generation_finished_elapsed_s=completed - start,
                         dispatch_lag_s=max(0.0, dispatch - start - source_end)
                         if args.pace == 'realtime' else None,
                         prompt_tokens=result.prompt_tokens, context=context)
            output.event('turn', **event)
            output.append(delta)
            if delta:
                stdout.write(delta)
                stdout.flush()
            LOG.info('turn=%d source=%.3f..%.3fs %s service=%.3fs dispatch_lag=%s',
                     state.turns, source_start, source_end, state.last_tag, event['service_s'],
                     'n/a' if event['dispatch_lag_s'] is None else f"{event['dispatch_lag_s']:.3f}s")
        final_wait = state.last_tag == 'WAIT'
        output.event('done', reason='eof', complete_input=True, final_wait=final_wait,
                     elapsed_s=clock() - start, **state.stats())
        if final_wait:
            LOG.warning('Final action is WAIT. All input was processed, but the last meaning may '
                        'remain untranslated. No extra silence turn was injected.')
        return state
    finally:
        session.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', type=Path, required=True, help='Local audio file; never a remote URL')
    p.add_argument('--model', default='', help='Local merged checkpoint and matching tokenizer/processor')
    p.add_argument('--direction', choices=SUPPORTED_DIRECTIONS, default='en-zh')
    p.add_argument('--chunk-seconds', type=float, default=1.0)
    p.add_argument('--pace', choices=('realtime', 'fast'), default='realtime',
                   help='realtime: simulate 1x audio arrival; fast: chunked inference without waiting')
    p.add_argument('--output-dir', type=Path, help='NEW directory for translation.txt and events.jsonl')
    p.add_argument('--max-audio-seconds', type=float, default=28800.0,
                   help='Total input cap (default 8 hours), separate from rolling model context')
    p.add_argument('--context-policy', choices=('rolling', 'stop'), default='rolling')
    p.add_argument('--history-window-seconds', type=float, default=120.0)
    p.add_argument('--history-retain-seconds', type=float, default=60.0)
    p.add_argument('--ffmpeg', default='ffmpeg', help='FFmpeg executable for non-native audio formats')
    p.add_argument('--backend', choices=('vllm', 'mock'), default='vllm')
    p.add_argument('--protocol-form', choices=('auto', 'special', 'text'), default='auto')
    p.add_argument('--dtype', choices=('bfloat16', 'float16', 'auto'), default='bfloat16')
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--max-audio-turns', type=int, default=256)
    p.add_argument('--max-tokens', type=int, default=160)
    p.add_argument('--mm-processor-cache-gb', type=float, default=2.0)
    p.add_argument('--image-slots', type=int, choices=(0, 1), default=1,
                   help='1: keep vision initialization enabled to avoid affected Qwen3-Omni '
                        'meta/CUDA failures; input is still audio-only. 0: disable on fixed builds')
    p.add_argument('--no-prefix-caching', action='store_true')
    p.add_argument('--no-warmup', action='store_true', help='Skip the discarded pre-clock warmup')
    args = p.parse_args(argv)
    if not args.audio.is_file():
        p.error('--audio must be an existing local file')
    if args.backend == 'vllm' and (not args.model or not Path(args.model).is_dir()):
        p.error('--model must be an existing local merged checkpoint directory')
    if not math.isfinite(args.chunk_seconds) or not 0.5 <= args.chunk_seconds <= 6:
        p.error('--chunk-seconds must be between 0.5 and 6; start with 1 second')
    if not math.isfinite(args.max_audio_seconds) or args.max_audio_seconds <= 0:
        p.error('--max-audio-seconds must be finite and positive')
    try:
        model_options(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        args.output_dir = Path('outputs') / f's2tt_{stamp}_{uuid.uuid4().hex[:8]}'
    if args.output_dir.exists():
        p.error('--output-dir already exists; choose a new directory (no overwriting)')
    return args


def model_options(args):
    return ModelOptions(model=args.model, dtype=args.dtype,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_model_len=args.max_model_len, max_audio_turns=args.max_audio_turns,
                        max_tokens=args.max_tokens, protocol_form=args.protocol_form,
                        max_chunk_seconds=args.chunk_seconds,
                        mm_processor_cache_gb=args.mm_processor_cache_gb,
                        context_policy=args.context_policy, history_window_seconds=args.history_window_seconds,
                        history_retain_seconds=args.history_retain_seconds,
                        prefix_caching=not args.no_prefix_caching, image_slots=args.image_slots)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stderr)
    state = RunState()
    try:
        with OutputFiles(args.output_dir) as output:
            try:
                output.event('start', backend=args.backend, mock=args.backend == 'mock',
                             source_name=args.audio.name, direction=args.direction,
                             model_name=Path(args.model).name if args.model else None,
                             protocol_form=args.protocol_form, max_model_len=args.max_model_len,
                             image_slots=args.image_slots,
                             max_audio_turns=args.max_audio_turns, max_tokens=args.max_tokens,
                             chunk_seconds=args.chunk_seconds, pace=args.pace,
                             context_policy=args.context_policy, history_window_seconds=args.history_window_seconds,
                             history_retain_seconds=args.history_retain_seconds,
                             max_audio_seconds=args.max_audio_seconds)
                LOG.info('Outputs: %s', output.directory)
                if args.backend == 'mock':
                    LOG.warning('MOCK ONLY: this does not recognize or translate audio.')
                with ExitStack() as stack:
                    reader = stack.enter_context(AudioReader(args.audio, args.ffmpeg))
                    engine = ModelEngine(model_options(args), backend=args.backend)
                    stack.callback(engine.close)
                    if not args.no_warmup:
                        engine.warmup(min(1.0, args.chunk_seconds))
                    output.event('model_ready', **engine.metadata)
                    LOG.info('Model ready. Starting %s file replay; no audio playback/TTS.', args.pace)
                    run_stream(args, engine, reader, output, state)
                LOG.info('Finished: %d turns, %.3fs audio. Translation: %s',
                         state.turns, state.processed_samples / SAMPLE_RATE,
                         output.directory / 'translation.txt')
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                try:
                    output.event('error', code=type(exc).__name__, message=str(exc),
                                 complete_input=False, **state.stats())
                except Exception:
                    LOG.exception('Could not save the error event')
                raise
    except KeyboardInterrupt:
        LOG.warning('Interrupted; saved outputs are a partial prefix, not a completed translation.')
        return 130
    except Exception:
        LOG.exception('Inference failed; any saved outputs are partial. No automatic retry/history reset.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
