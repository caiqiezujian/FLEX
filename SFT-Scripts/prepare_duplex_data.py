#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Prepare duplex simultaneous-interpretation SFT data from full-session audio.

Input — every session is shaped like the repo's raw_data.json (trailing commas
tolerated), supplied as either:
  - a .jsonl/.ndjson file with ONE session object per line (read streaming, so files
    too large to load at once are fine; skip/error messages cite the line number), or
  - a .json file holding a single session object or an array of sessions.

    {
        "audio":   "/data/audio/123456.wav",            # full-session audio
        "srcs":    ["So, this culture ...", ...],        # ASR segments (source language)
        "chooses": ["所以，这种文化必须终结。", ...],      # translation block per src
        "tags":    ["<|DIRECT|>", "<|FLEX|>", ...],      # protocol tag per block
        "timestamps": [
            {"index": 1, "start_ms": 0, "end_ms": 1840, "text": "So, this culture ..."},
            ...
        ]
    }

Per session the script:
  1. probes the audio duration (ffprobe) and fixes the session timeline: by default the
     timeline ends at the last speech end (max end_ms) rounded UP to a whole second
     (trailing silence is dropped); --keep-tail-silence extends it to the full audio
     (also rounded up to a whole second);
  2. partitions the timeline into chunks of random whole seconds following a
     truncated geometric law over [--min-chunk, --max-chunk] (default 1..6; hard
     cap 8 for the current training recipe): each size is --chunk-decay (default
     0.5) times less likely than the second below it, so cuts are dominated by
     SHORT chunks with occasional large ones — mirroring deployment, where chunks
     stream in short and only pile up under concurrent load. A tail shorter than
     min-chunk becomes one final short chunk. RNG is seeded per session from
     (--seed, session id) → reproducible cuts;
  3. records original-audio intervals by default (--audio-mode virtual). Training
     resolves each interval lazily and pads only its final fractional second.
     --audio-mode files optionally materializes 16kHz mono pcm_s16le WAV chunks;
  4. assigns each translation block to the FIRST chunk whose end fully covers the block's
     source segment (chunk_end_ms >= end_ms): a translation is only sayable once its
     source has been heard. FLEX denotes strategic deferral or restructuring;
     DIRECT denotes direct incremental translation. Neither is a syntax-completeness
     or punctuation label. Chunks with no due block emit <|WAIT|>. Multiple blocks due in
     the same chunk merge into ONE turn (texts concatenated in order — pure-incremental
     semantics; turn tag = tag of the last non-empty block). Source indices and tags
     remain in turn_provenance, and mixed-tag merges are explicitly diagnosed;
  5. rejects whole sessions with invalid timestamps or protocol data, recording their
     input location, id and reason. Streams accepted rows to --output (jsonl).
     The default output is the
     string format (ms-swift's documented multimodal standard: '<audio>' tag per user
     turn + top-level 'audios' path list). The HF datasets loader requires every
     messages[].content in a file to be one uniform JSON type — the earlier mixed typed
     draft (string system/assistant + typed user) dies at load with
     "Column(/messages/[]/content) changed from string to array"; --typed emits the
     all-typed variant instead (system prompt moved to the top-level 'system' column,
     every content wrapped in typed parts). The system prompt names
     the target language via --target-lang (default 中文) — inference must reuse the
     same value (duplex_infer_vllm.py --target-lang).

Preparation probes duration with soundfile when installed, falling back to ffprobe.
Virtual references require the duplex SFT audio loader when training. File mode
requires ffmpeg; preparation never deletes original audio.

Examples:
    python3 prepare_duplex_data.py sessions.jsonl \
        --output /data/duplex/train.jsonl --seed 42

    python3 prepare_duplex_data.py raw_data.json \
        --output /data/duplex/train.jsonl --audio-mode files --audio-out /data/duplex/chunks \
        --min-chunk 1 --max-chunk 6 --chunk-decay 0.5 \
        --target-lang 中文 --keep-tail-silence
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import Counter
from contextlib import ExitStack, contextmanager

PROTOCOL_SYSTEM_PROMPT_TEMPLATE = (
    '将用户按顺序发送的音频同声翻译为{target_lang}。'
    '结合已听音频和历史译文，每轮只输出以下一种：'
    '暂无可靠新增译文时，仅输出<|WAIT|>；'
    '可直接自然续接时，输出<|DIRECT|>加新增译文；'
    '需暂缓部分内容或调整表达结构时，输出<|FLEX|>加可靠的新增译文。'
    '译文只追加，不重复、不回改、不猜测后文；条件允许时补齐此前暂缓且尚未译出的内容。'
    'DIRECT与FLEX按策略区分，不按句子是否完整区分。'
    '按语义使用标点，不随音频块机械加句号，不用省略号占位。'
    '除WAIT外，译文不能为空；不输出解释。'
)
DEFAULT_TARGET_LANG = '中文'
VALID_TAGS = ('<|WAIT|>', '<|DIRECT|>', '<|FLEX|>')
HARD_MAX_CHUNK = 8  # Current training recipe limit, not a universal model limit.
DEFAULT_CHUNK_DECAY = 0.5  # truncated-geometric ratio: P(size+1s) = decay * P(size)


def build_system_prompt(target_lang=DEFAULT_TARGET_LANG):
    """Protocol system prompt naming the target language (inference must use the same)."""
    return PROTOCOL_SYSTEM_PROMPT_TEMPLATE.format(target_lang=target_lang)


def is_finite_number(value):
    """JSON numbers only; huge integers must not raise during validation."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and (isinstance(value, int) or math.isfinite(value)))


def tolerant_loads(text):
    """Allow trailing commas without editing commas inside quoted translations."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned, in_string, escaped = [], False, False
        for i, char in enumerate(text):
            if in_string:
                cleaned.append(char)
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == ',':
                j = i + 1
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] in ']}':
                    continue
            cleaned.append(char)
        return json.loads(''.join(cleaned))


def iter_jsonl_lines(lines, path, on_error=None):
    """Yield (locator, obj) per non-blank line; every line must be one session object."""
    for ln, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = tolerant_loads(line)
        except json.JSONDecodeError as e:
            reason = f'not valid JSON: {e}'
            if on_error is None:
                raise ValueError(f'{path}: line {ln} {reason}') from e
            on_error(f'line {ln}', None, reason)
            continue
        if not isinstance(obj, dict):
            reason = f'expected one session object, got {type(obj).__name__}'
            if on_error is None:
                raise ValueError(f'{path}: line {ln}: {reason}')
            on_error(f'line {ln}', obj, reason)
            continue
        yield f'line {ln}', obj


def load_sessions(path, on_error=None):
    """Yield (locator, session) pairs; JSONL is never accumulated in memory.

    .jsonl/.ndjson are parsed streaming, one session object per line (files too large
    to read at once are fine). A .json holds a single session object or an array of
    sessions; a .json that fails whole-file parsing is retried as JSONL before giving up.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jsonl', '.ndjson'):
        with open(path, encoding='utf-8') as f:
            yield from iter_jsonl_lines(f, path, on_error)
        return
    with open(path, encoding='utf-8') as f:
        text = f.read()
    try:
        data = tolerant_loads(text)
    except json.JSONDecodeError as whole_err:
        # A pretty-printed (multi-line) .json fails per-line parsing too, so only retry
        # as JSONL when the first non-blank line is itself a complete object.
        first = next((ln for ln in text.splitlines() if ln.strip()), '')
        try:
            is_jsonl = isinstance(tolerant_loads(first), dict)
        except json.JSONDecodeError:
            is_jsonl = False
        if not is_jsonl:
            raise SystemExit(f'{path}: not valid JSON: {whole_err}') from None
        yield from iter_jsonl_lines(text.splitlines(), path, on_error)
        return
    if isinstance(data, dict):
        yield '#1', data
        return
    if isinstance(data, list):
        for i, x in enumerate(data, 1):
            if not isinstance(x, dict):
                reason = f'expected session object, got {type(x).__name__}'
                if on_error is None:
                    raise ValueError(f'{path}: element #{i}: {reason}')
                on_error(f'#{i}', x, reason)
                continue
            yield f'#{i}', x
        return
    raise SystemExit(f'{path}: expected a session object or an array of sessions, '
                     f'got {type(data).__name__}')


def probe_duration(ffprobe, audio):
    """Probe duration without decoding audio; prefer optional soundfile metadata."""
    try:
        import soundfile
        info = soundfile.info(audio)
        duration = info.frames / info.samplerate
        if math.isfinite(duration) and duration > 0:
            return duration
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    res = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', audio],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'ffprobe failed for {audio}: {res.stderr.strip()[:200]}')
    try:
        return float(res.stdout.strip())
    except ValueError:
        raise RuntimeError(f'ffprobe returned non-numeric duration for {audio}: {res.stdout!r}')


def cut_audio(ffmpeg, audio, start_s, length_s, out_path):
    """Cut [start_s, start_s+length_s) as 16kHz mono pcm_s16le wav.

    apad=whole_dur pads the source with silence when it ends early (final chunk whose
    whole-second boundary lies beyond the audio), so every chunk file is exactly
    length_s seconds. Patched out in offline tests.
    """
    res = subprocess.run(
        [ffmpeg, '-y', '-v', 'error', '-ss', str(start_s), '-i', audio, '-t', str(length_s),
         '-af', f'apad=whole_dur={length_s}', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
         out_path],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f'ffmpeg failed for {out_path}: {res.stderr.strip()[:200]}')


def plan_chunks(total_seconds, rng, lo, hi, decay=DEFAULT_CHUNK_DECAY):
    """Partition total_seconds into random whole-second chunks within [lo, hi].

    Sizes follow a truncated geometric law over [lo, hi]: each size is `decay` times
    less likely than the second below it (decay=1 → uniform). Deployment sends short
    chunks most of the time, with large ones only appearing when concurrent load
    piles up — the training cuts mirror that (default decay 0.5 over 1..6s ≈ 51% 1s,
    25% 2s, ..., 2% 6s, mean ≈ 1.9s). A final tail shorter than `lo` is emitted as
    one last short chunk (the timeline must tile exactly).
    """
    sizes, remaining = [], total_seconds
    while remaining > 0:
        if remaining <= lo:  # tail too short to draw: emit it as the final chunk
            sizes.append(remaining)
            break
        hi_eff = min(hi, remaining)
        weights = [decay ** (s - lo) for s in range(lo, hi_eff + 1)]
        size = rng.choices(range(lo, hi_eff + 1), weights=weights)[0]
        sizes.append(size)
        remaining -= size
    return sizes


def assign_due_chunks(end_ms_list, chunk_end_ms):
    """due chunk per block = first chunk whose end (ms) covers the block's source end."""
    if not chunk_end_ms or any(b <= a for a, b in zip([0] + list(chunk_end_ms), chunk_end_ms)):
        raise ValueError('chunk ends must be positive and strictly increasing')
    dues, prev_end = [], -1
    for end_ms in end_ms_list:
        if not is_finite_number(end_ms):
            raise ValueError('source end_ms must be finite numbers')
        if end_ms < 0 or end_ms < prev_end:
            raise ValueError('source end_ms must be nonnegative and ordered')
        k = bisect_left(chunk_end_ms, end_ms)
        if k >= len(chunk_end_ms):
            raise ValueError(f'source end_ms {end_ms} exceeds chunk timeline {chunk_end_ms[-1]}')
        dues.append(k)
        prev_end = end_ms
    return dues


def validate_translations(chooses, tags):
    """Reject malformed protocol examples instead of silently rewriting labels/text."""
    if not isinstance(chooses, list) or not isinstance(tags, list):
        raise ValueError('"chooses" and "tags" must be lists')
    if len(chooses) != len(tags):
        raise ValueError('chooses/tags length mismatch')
    for i, (text, tag) in enumerate(zip(chooses, tags)):
        if tag not in VALID_TAGS:
            raise ValueError(f'tags[{i}] unknown tag {tag!r}')
        if not isinstance(text, str):
            raise ValueError(f'chooses[{i}] must be a string')
        if tag == '<|WAIT|>' and text.strip():
            raise ValueError(f'chooses[{i}] WAIT must not carry translation text')
        if tag != '<|WAIT|>' and not text.strip():
            raise ValueError(f'chooses[{i}] {tag} requires nonempty translation text')
        if any(token in text for token in (*VALID_TAGS, '<|im_start|>', '<|im_end|>')):
            raise ValueError(f'chooses[{i}] contains a reserved protocol/control token')


def build_turn_provenance(n_chunks, dues, chooses, tags):
    """Trace every merged turn back to original zero-based source-block indices."""
    if len(dues) != len(chooses) or len(chooses) != len(tags):
        raise ValueError('dues/chooses/tags length mismatch')
    buckets = [[] for _ in range(n_chunks)]
    for i, k in enumerate(dues):
        if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < n_chunks:
            raise ValueError(f'dues[{i}] invalid chunk index {k!r}')
        buckets[k].append(i)
    result = []
    for k, due in enumerate(buckets):
        source_tags = [tags[i] for i in due]
        emission_tags = [tags[i] for i in due if chooses[i].strip()]
        result.append({
            'audio_chunk_index': k,
            'source_chunk_indices': due,
            'source_tags': source_tags,
            'mixed_source_tags': len(set(source_tags)) > 1,
            'mixed_emission_tags': len(set(emission_tags)) > 1,
            'merge_tag_policy': 'last_nonempty',
        })
    return result


def build_turns(n_chunks, dues, chooses, tags):
    """One assistant turn per chunk: WAIT / merged due blocks (concat, last tag)."""
    validate_translations(chooses, tags)
    provenance = build_turn_provenance(n_chunks, dues, chooses, tags)
    turns, warns = [], []
    for item in provenance:
        k, due = item['audio_chunk_index'], item['source_chunk_indices']
        nonempty = [i for i in due if chooses[i].strip()]
        if item['mixed_source_tags']:
            warns.append(f'chunk {k}: mixed source tags {item["source_tags"]}; '
                         'merge_tag_policy=last_nonempty (not relabeled)')
        if not nonempty:
            turns.append('<|WAIT|>')
            continue
        tag = tags[nonempty[-1]]
        text = ''.join(chooses[i] for i in nonempty)
        turns.append(f'{tag}{text}')
    return turns, warns


def parse_session(sess):
    """Validate and normalize one raw session. Returns (audio, srcs, chooses, tags, end_ms, warns)."""
    if not isinstance(sess, dict):
        raise ValueError('session must be an object')
    audio = sess.get('audio')
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError('missing or invalid "audio"')
    srcs = sess.get('srcs', sess.get('src', []))
    chooses = sess.get('chooses', sess.get('choose', []))
    tags = sess.get('tags', [])
    if not isinstance(srcs, list) or not srcs:
        raise ValueError('"srcs" must be a non-empty list')
    n = len(srcs)
    if any(not isinstance(src, str) for src in srcs):
        raise ValueError('every srcs element must be a string')
    validate_translations(chooses, tags)
    if len(chooses) != n or len(tags) != n:
        raise ValueError(f'length mismatch: srcs={n} chooses={len(chooses)} tags={len(tags)}')
    ts = sess.get('timestamps', [])
    if not isinstance(ts, list) or any(not isinstance(t, dict) for t in ts):
        raise ValueError('"timestamps" must be a list of objects')
    if len(ts) != n:
        raise ValueError(f'timestamps count {len(ts)} != srcs {n}')
    end_ms, warns = [], []
    if any('index' in t for t in ts):
        by_idx, indices = {}, []
        for i, t in enumerate(ts):
            index = t.get('index')
            if isinstance(index, str) and re.fullmatch(r'[1-9][0-9]*', index):
                index = int(index)
            if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= n:
                raise ValueError(f'timestamps[{i}] requires a valid 1-based integer index')
            if index in by_idx:
                raise ValueError(f'duplicate timestamps index {index}')
            by_idx[index] = t
            indices.append(index)
        if set(by_idx) != set(range(1, n + 1)):
            raise ValueError('timestamps indices must cover every source block exactly once')
        ts = [by_idx[i] for i in range(1, n + 1)]
        if indices != list(range(1, n + 1)):
            warns.append('timestamps reordered by complete unique 1-based indices')
    prev_end = 0
    for i, t in enumerate(ts):
        values = {}
        for key in ('start_ms', 'end_ms'):
            value = t.get(key)
            if not is_finite_number(value):
                raise ValueError(f'timestamps[{i}].{key} must be a finite number')
            if value < 0:
                raise ValueError(f'timestamps[{i}].{key} must be nonnegative')
            values[key] = value
        start, end = values['start_ms'], values['end_ms']
        if end <= start:
            raise ValueError(f'timestamps[{i}] end_ms must be greater than start_ms')
        if start < prev_end:
            raise ValueError(f'timestamps[{i}] out of order or overlapping the preceding segment')
        if 'text' in t and t['text'] != srcs[i]:
            raise ValueError(f'timestamps[{i}].text != srcs[{i}]')
        end_ms.append(end)
        prev_end = end
    return audio, srcs, chooses, tags, end_ms, warns


def build_row(sid, chunk_paths, sizes, turns, system_prompt, typed=False):
    """One training row per session.

    String format (default, the documented ms-swift multimodal standard): messages
    carry plain-string content — an '<audio>' tag per user turn — and chunk paths live
    in the top-level 'audios' list (tag order == list order). Typed format (--typed,
    closer to the original schema draft): the system prompt moves to the top-level
    'system' column and every content is a typed part list. Both keep content uniform
    per file — the pyarrow JSON loader behind --dataset rejects mixed string/array
    content, which is why the original draft (string system/assistant + typed user)
    is not loadable in any form.
    """
    if not len(chunk_paths) == len(sizes) == len(turns):
        raise ValueError('chunk paths/sizes/turns must have equal lengths')
    durations = [float(s) for s in sizes]
    if typed:
        messages = []
        for path, turn in zip(chunk_paths, turns):
            messages.append({'role': 'user', 'content': [{'type': 'audio', 'audio': path}]})
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': turn}]})
        return {'id': sid, 'system': system_prompt, 'messages': messages,
                'audio_chunk_durations': durations}
    messages = [{'role': 'system', 'content': system_prompt}]
    for path, turn in zip(chunk_paths, turns):
        messages.append({'role': 'user', 'content': '<audio>'})
        messages.append({'role': 'assistant', 'content': turn})
    return {'id': sid, 'messages': messages, 'audios': list(chunk_paths),
            'audio_chunk_durations': durations}


def process_session(sess, args, sid, warnings):
    """Probe, plan, cut, assign, and build the training row for one session."""
    audio, srcs, chooses, tags, end_ms, parse_warns = parse_session(sess)
    warnings.extend(f'[{sid}] {w}' for w in parse_warns)

    duration = probe_duration(args.ffprobe, audio)
    if not is_finite_number(duration) or duration <= 0:
        raise ValueError('audio duration must be finite and positive')
    # The tolerance is one billionth of a second, only for floating-point roundoff.
    if max(end_ms) > duration * 1000 + 1e-6:
        raise ValueError(f'timestamps end at {max(end_ms)}ms beyond actual audio duration '
                         f'{duration * 1000:.6f}ms')
    audio_ceil = math.ceil(duration)
    if args.keep_tail_silence:
        timeline = audio_ceil
    else:
        timeline = math.ceil(max(end_ms) / 1000)
    if timeline <= 0:
        raise ValueError('empty timeline (no speech / zero-length audio)')

    rng = random.Random(f'{args.seed}:{sid}')
    sizes = plan_chunks(timeline, rng, args.min_chunk, args.max_chunk, args.chunk_decay)
    chunk_end_ms, acc = [], 0
    for s in sizes:
        acc += s * 1000
        chunk_end_ms.append(acc)
    dues = assign_due_chunks(end_ms, chunk_end_ms)
    turns, turn_warns = build_turns(len(sizes), dues, chooses, tags)
    warnings.extend(f'[{sid}] {w}' for w in turn_warns)

    audio_mode = getattr(args, 'audio_mode', 'virtual')
    if audio_mode not in ('virtual', 'files'):
        raise ValueError(f'unknown audio_mode {audio_mode!r}')
    if audio_mode == 'files':
        if not args.audio_out:
            raise ValueError('--audio-out is required for --audio-mode files')
        chunk_dir = os.path.join(args.audio_out, sid)
        if not args.dry_run:
            os.makedirs(chunk_dir, exist_ok=True)
        audio_root = args.audio_out.rstrip('/\\')
    else:
        from duplex_audio import encode_chunk_ref
    absolute_audio = os.path.abspath(audio)
    chunk_paths, segments, start = [], [], 0
    for k, s in enumerate(sizes):
        end = min(start + s, duration)
        segments.append({'path': absolute_audio, 'start_seconds': start,
                         'end_seconds': end, 'input_seconds': s})
        if audio_mode == 'virtual':
            chunk_paths.append(encode_chunk_ref(absolute_audio, start, end, s))
        else:
            name = f'chunk_{k:03d}.wav'
            chunk_paths.append(f'{audio_root}/{sid}/{name}')
            if not args.dry_run:
                cut_audio(args.ffmpeg, audio, start, s, os.path.join(chunk_dir, name))
        start += s

    row = build_row(sid, chunk_paths, sizes, turns, build_system_prompt(args.target_lang),
                    typed=args.typed)
    row['audio_mode'] = audio_mode
    row['audio_segments'] = segments
    row['turn_provenance'] = build_turn_provenance(len(sizes), dues, chooses, tags)
    return row, sizes, turns, duration


def allocate_session_id(audio, seen, used):
    """Keep familiar basename ids while preventing natural/generated suffix collisions."""
    stem = re.sub(r'[\\/\s]+', '_', os.path.splitext(os.path.basename(audio.replace('\\', '/')))[0])
    stem = stem or 'session'
    count = seen.get(stem, 0) + 1
    sid = stem if count == 1 else f'{stem}_{count}'
    while sid in used:
        count += 1
        sid = f'{stem}_{count}'
    seen[stem] = count
    used.add(sid)
    return sid


@contextmanager
def atomic_text_output(path):
    """Stream to a sibling temporary file, publishing only after successful iteration."""
    path = os.path.abspath(path)
    handle = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         prefix='.' + os.path.basename(path) + '.',
                                         suffix='.tmp', dir=os.path.dirname(path), delete=False)
    temporary = handle.name
    try:
        with handle:
            yield handle
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Plan full-session audio as random whole-second chunks and build '
                    'duplex SFT training rows (string format; --typed for the all-typed '
                    'variant).')
    parser.add_argument('input', help='raw sessions: .jsonl (one session object per '
                                      'line) or .json (single object or array)')
    parser.add_argument('--output', required=True, help='training data output path (.jsonl)')
    parser.add_argument('--audio-mode', choices=('virtual', 'files'), default='virtual',
                        help='virtual: reference original-audio intervals (default); files: write WAV chunks')
    parser.add_argument('--audio-out', help='directory for WAV chunks; required only in files mode')
    parser.add_argument('--rejected-output', help='rejection report JSONL (default: OUTPUT.rejected.jsonl)')
    parser.add_argument('--fail-on-reject', action='store_true',
                        help='return a nonzero exit code if any session is rejected; valid rows are still saved')
    parser.add_argument('--min-chunk', type=int, default=1, help='min chunk seconds (default 1)')
    parser.add_argument('--max-chunk', type=int, default=6, help='max chunk seconds (default 6)')
    parser.add_argument('--chunk-decay', type=float, default=DEFAULT_CHUNK_DECAY,
                        help='truncated-geometric decay for chunk sizes: P(size+1s) = '
                             'decay * P(size), so cuts are dominated by short chunks with '
                             'occasional large ones (mirrors deployment backlog); 1.0 = '
                             f'uniform (default {DEFAULT_CHUNK_DECAY})')
    parser.add_argument('--target-lang', default=DEFAULT_TARGET_LANG,
                        help=f'target language of the interpretation, injected into the '
                             f'system prompt (default: {DEFAULT_TARGET_LANG}) — inference '
                             f'must use the same value')
    parser.add_argument('--typed', action='store_true',
                        help='emit the all-typed variant: system prompt in the top-level '
                             '"system" column, every messages[].content a typed part list '
                             '(default: string format — <audio> tags + top-level "audios" '
                             'list; mixed string/array content is rejected by the datasets '
                             'json loader)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; per-session streams are derived from (seed, session id)')
    parser.add_argument('--keep-tail-silence', action='store_true',
                        help='keep audio after the last speech end (default: trim at last '
                             'speech end rounded up to a whole second)')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='ffmpeg binary (default: PATH)')
    parser.add_argument('--ffprobe', default='ffprobe', help='ffprobe binary (default: PATH)')
    parser.add_argument('--dry-run', action='store_true',
                        help='plan and assign without cutting audio or writing output')
    args = parser.parse_args(argv)
    if not 1 <= args.min_chunk <= args.max_chunk <= HARD_MAX_CHUNK:
        parser.error(f'require 1 <= min-chunk <= max-chunk <= {HARD_MAX_CHUNK} '
                     '(current training recipe limit)')
    if not 0 < args.chunk_decay <= 1:
        parser.error('require 0 < chunk-decay <= 1 (1 = uniform, smaller = shorter chunks)')
    if args.audio_mode == 'files' and not args.audio_out:
        parser.error('--audio-out is required with --audio-mode files')
    if args.audio_mode == 'files' and not os.path.isabs(args.audio_out):
        print(f'NOTE: --audio-out is relative; generated audio paths will be relative '
              f'to the current directory ({os.getcwd()})')
    args.rejected_output = args.rejected_output or args.output + '.rejected.jsonl'
    paths = [os.path.normcase(os.path.realpath(p))
             for p in (args.input, args.output, args.rejected_output)]
    if len(set(paths)) != 3:
        parser.error('input, output and rejected-output must be different files')

    seen, used = {}, set()
    accepted = rejected = mixed_source = mixed_emission = 0
    size_hist, tag_hist = Counter(), Counter()
    with ExitStack() as stack:
        output = report = None
        if not args.dry_run:
            output = stack.enter_context(atomic_text_output(args.output))
            report = stack.enter_context(atomic_text_output(args.rejected_output))

        def reject(loc, sess, reason, sid=None):
            nonlocal rejected
            rejected += 1
            sess = sess if isinstance(sess, dict) else {}
            source_id = None if sess.get('id') is None else str(sess['id'])
            record = {
                'input': os.path.abspath(args.input), 'locator': loc,
                'line': int(loc[5:]) if loc.startswith('line ') else None,
                'id': sid or source_id, 'source_id': source_id,
                'audio': sess.get('audio'), 'reason': reason,
            }
            print(f'[{args.input}:{loc}] SKIPPED id={record["id"]!r}: {reason}')
            if report is not None:
                report.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f'streaming sessions from {args.input}; audio_mode={args.audio_mode}')
        for loc, sess in load_sessions(args.input, reject):
            raw_audio = sess.get('audio')
            sid = allocate_session_id(raw_audio, seen, used) if isinstance(raw_audio, str) and raw_audio.strip() else None
            warnings = []
            try:
                row, sizes, turns, duration = process_session(sess, args, sid, warnings)
            except (ValueError, RuntimeError, OSError) as e:
                reject(loc, sess, str(e), sid)
                continue
            row['source_locator'] = {'input': os.path.abspath(args.input), 'locator': loc,
                                     'source_id': None if sess.get('id') is None else str(sess['id'])}
            if output is not None:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
            accepted += 1
            size_hist.update(sizes)
            sess_tags = Counter(t.split('>')[0] + '>' for t in turns)
            tag_hist.update(sess_tags)
            mixed_source += sum(item['mixed_source_tags'] for item in row['turn_provenance'])
            mixed_emission += sum(item['mixed_emission_tags'] for item in row['turn_provenance'])
            print(f'[{sid}] {duration:.2f}s audio -> {len(sizes)} chunks '
                  f'({"+".join(map(str, sizes))}={sum(sizes)}s), '
                  f'turns: ' + ' '.join(f'{k}={v}' for k, v in sorted(sess_tags.items())))
            for warning in warnings:
                print(f'WARN {warning}')

    if args.dry_run:
        print(f'\ndry run: {accepted} session(s) planned, {rejected} rejected, nothing written')
    else:
        print(f'\nwrote {accepted} row(s) to {args.output}')
        print(f'rejected {rejected} session(s); report: {args.rejected_output}')
        if args.audio_mode == 'files':
            print(f'chunk wavs under {args.audio_out}/<session_id>/chunk_NNN.wav')
        else:
            print('original-audio interval references written; no chunk WAVs created')

    if size_hist:
        print('chunk size histogram: ' + ', '.join(f'{k}s x{v}' for k, v in sorted(size_hist.items())))
    print(f'mixed-tag chunks: source={mixed_source}, emission={mixed_emission}; '
          'merge_tag_policy=last_nonempty')
    if tag_hist:
        print('turn tag histogram: ' + ', '.join(f'{k} x{v}' for k, v in sorted(tag_hist.items())))
    return 1 if accepted == 0 or (args.fail_on_reject and rejected > 0) else 0


if __name__ == '__main__':
    sys.exit(main())
