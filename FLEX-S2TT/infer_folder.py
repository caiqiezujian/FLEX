"""Translate audio and update existing JSONL fields, or write a new result JSONL."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import sys
import time

import infer_audio
from flex_model import ModelEngine
from jsonl_update import prepare_update
from infer_audio import (
    AudioReader, RunState, add_inference_arguments, model_options, run_stream,
    validate_inference_arguments,
)


LOG = logging.getLogger('flex.batch')
AUDIO_SUFFIXES = frozenset({
    '.wav', '.wave', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.oga', '.opus',
    '.wma', '.aiff', '.aif', '.aifc', '.amr', '.mp2', '.mka', '.caf', '.ac3',
})


class StreamSink:
    """Keep only the last diagnostic, not a transcript/event copy for every turn."""

    def __init__(self):
        self.rejected = None

    def event(self, kind, **data):
        if kind == 'rejected_turn':
            self.rejected = data

    def append(self, delta):
        # run_stream already collects committed deltas in its per-file RunState.
        pass

    def write(self, text):
        # Batch progress goes to the logger, not interleaved translation stdout.
        return len(text)

    def flush(self):
        pass


def collect_audio_files(directory, *, recursive=False, vid_mode='filename'):
    """Stable traversal; reject ambiguous IDs before loading a model."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError('--audio-dir must be an existing directory')
    if vid_mode not in ('filename', 'relative'):
        raise ValueError('vid_mode must be filename or relative')
    candidates = directory.rglob('*') if recursive else directory.iterdir()
    paths = sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not paths:
        raise ValueError('No supported audio files found in --audio-dir'
                         + (' or its subdirectories' if recursive else
                            '; use --recursive to include subdirectories'))
    entries, seen = [], set()
    for path in paths:
        vid = path.name if vid_mode == 'filename' else path.relative_to(directory).as_posix()
        if vid in seen:
            raise ValueError(f'Duplicate vid {vid!r}; use --vid-mode relative for subdirectories')
        seen.add(vid)
        entries.append((path, vid))
    return entries


def errors_path(output):
    output = Path(output)
    return output.with_name(output.stem + '.errors.jsonl')


def write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()


def failure_record(path, vid, state, sink, exc):
    record = dict(vid=vid, source_path=str(path), error_type=type(exc).__name__,
                  message=str(exc) or type(exc).__name__, complete_input=False, **state.stats())
    if sink.rejected is not None:
        record['rejected_turn'] = sink.rejected
    return record


def run_batch(args, entries):
    """One engine, sequential independent sessions, commit one result only at EOF."""
    output_path = args.output_jsonl.resolve()
    error_path = errors_path(output_path)
    for path in (output_path, error_path):
        if path.exists():
            raise FileExistsError(f'Output already exists; nothing will be overwritten: {path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    succeeded = failed = 0
    began = time.monotonic()
    with ExitStack() as stack:
        results = stack.enter_context(output_path.open('x', encoding='utf-8', newline='\n'))
        errors = stack.enter_context(error_path.open('x', encoding='utf-8', newline='\n'))
        if args.backend == 'mock':
            LOG.warning('MOCK ONLY: output is demonstration text, not audio translation.')
        LOG.info('Found %d audio files. Results: %s', len(entries), output_path)
        engine = ModelEngine(model_options(args), backend=args.backend)
        stack.callback(engine.close)
        if not args.no_warmup:
            engine.warmup(min(1.0, args.chunk_seconds))
        LOG.info('Model loaded once. Starting %s replay, %.3fs chunks.',
                 args.pace, args.chunk_seconds)
        for index, (path, vid) in enumerate(entries, 1):
            state, sink = RunState(), StreamSink()
            started = time.monotonic()
            LOG.info('[%d/%d] %s', index, len(entries), vid)
            try:
                with AudioReader(path, args.ffmpeg) as reader:
                    run_stream(args, engine, reader, sink, state, stdout=sink)
            except KeyboardInterrupt as exc:
                write_record(errors, failure_record(path, vid, state, sink, exc))
                LOG.warning('Interrupted on %s; %d completed records remain saved.', vid, succeeded)
                raise
            except Exception as exc:
                write_record(errors, failure_record(path, vid, state, sink, exc))
                failed += 1
                LOG.error('FAILED %s: %s: %s', vid, type(exc).__name__, exc,
                          exc_info=args.verbose)
                if args.fail_fast:
                    break
                continue
            # An output-storage error must abort the run, not masquerade as an audio error.
            write_record(results, {'vid': vid, 'translation': ''.join(state.deltas)})
            succeeded += 1
            LOG.info('OK %s: %.3fs audio, %d turns, %.2fs elapsed%s',
                     vid, state.processed_samples / infer_audio.SAMPLE_RATE, state.turns,
                     time.monotonic() - started,
                     ' [final WAIT: check the ending]' if state.last_tag == 'WAIT' else '')
    LOG.info('Finished: total=%d, success=%d, failed=%d, unattempted=%d, elapsed=%.2fs',
             len(entries), succeeded, failed, len(entries) - succeeded - failed,
             time.monotonic() - began)
    LOG.info('Results: %s; errors: %s', output_path, error_path)
    return 1 if failed else 0


def run_update(args, entries):
    """Infer audio only, then commit real translation plus the existing transcript."""
    plan = prepare_update(args.jsonl_dir, entries, AUDIO_SUFFIXES)
    LOG.info('Match summary: audio=%d, JSONL files=%d, rows=%d, matched=%d, '
             'rows without audio=%d, audio without row=%d',
             len(entries), plan.jsonl_files, plan.jsonl_rows, len(plan.targets),
             plan.unmatched_rows, len(plan.unmatched_audios))
    for audio in plan.unmatched_audios:
        LOG.warning('SKIP audio without matching JSONL vid: %s', audio)
    LOG.info('In-place fields: %s_trans (model translation), %s_asr (copy of transcript).',
             args.field_prefix, args.field_prefix)
    if args.check_only:
        LOG.info('Check complete: no model loaded and no JSONL modified.')
        return 0 if plan.targets else 1
    if not plan.targets:
        LOG.error('No matching rows; no model loaded and no JSONL modified.')
        return 1
    if args.backend == 'mock':
        LOG.warning('MOCK ONLY: writes demonstration text, not real audio translation. '
                    'Use this only with disposable test JSONL files.')
    succeeded = failed = 0
    began = time.monotonic()
    LOG.info('Loading model once: backend=%s model=%s', args.backend, args.model)
    with ExitStack() as stack:
        engine = ModelEngine(model_options(args), backend=args.backend)
        stack.callback(engine.close)
        if not args.no_warmup:
            engine.warmup(min(1.0, args.chunk_seconds))
        LOG.info('Model ready. Starting %s replay, %.3fs chunks.', args.pace, args.chunk_seconds)
        for index, target in enumerate(plan.targets, 1):
            state, sink = RunState(), StreamSink()
            started = time.monotonic()
            LOG.info('[%d/%d] %s -> %s:%d', index, len(plan.targets), target.audio.name,
                     target.document.path.name, target.line_index + 1)
            try:
                with AudioReader(target.audio, args.ffmpeg) as reader:
                    run_stream(args, engine, reader, sink, state, stdout=sink)
            except KeyboardInterrupt:
                LOG.warning('Interrupted: current row unchanged; %d completed rows saved.', succeeded)
                raise
            except Exception as exc:
                failed += 1
                LOG.error('FAILED %s: original row unchanged; %s', target.vid,
                          json.dumps(failure_record(target.audio, target.vid, state, sink, exc),
                                     ensure_ascii=False), exc_info=args.verbose)
                if args.fail_fast:
                    break
                continue
            # Filesystem errors abort immediately, rather than continuing to lose results.
            translation = ''.join(state.deltas)
            target.save(args.field_prefix, translation)
            succeeded += 1
            LOG.info('SAVED [%d/%d] vid=%s: %.3fs audio, %d turns, %d characters, %.2fs elapsed%s',
                     index, len(plan.targets), target.vid,
                     state.processed_samples / infer_audio.SAMPLE_RATE, state.turns,
                     len(translation), time.monotonic() - started,
                     ' [final WAIT: check the ending]' if state.last_tag == 'WAIT' else '')
            if not translation:
                LOG.warning('EMPTY translation saved for %s: model emitted no text; '
                            'transcript was copied only to the _asr field.', target.vid)
    LOG.info('Finished in-place update: matched=%d, saved=%d, failed=%d, unattempted=%d, '
             'unmatched rows untouched=%d, unmatched audio skipped=%d, elapsed=%.2fs',
             len(plan.targets), succeeded, failed, len(plan.targets) - succeeded - failed,
             plan.unmatched_rows, len(plan.unmatched_audios), time.monotonic() - began)
    return 1 if failed else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', '--input-dir', dest='audio_dir', type=Path,
                        required=True, help='Directory containing local audio files')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--jsonl-dir', type=Path,
                             help='Existing JSONL directory to update IN PLACE, matching row vid')
    destination.add_argument('--output-jsonl', type=Path,
                             help='Legacy mode: NEW JSONL file containing only vid and translation')
    parser.add_argument('--field-prefix', help='In-place fields: PREFIX_trans and PREFIX_asr')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate in-place matching only; no model loading or file changes')
    parser.add_argument('--recursive', action='store_true', help='Also scan subdirectories')
    parser.add_argument('--vid-mode', choices=('filename', 'relative'), default='filename',
                        help='filename includes extension; relative uses POSIX paths under --audio-dir')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on the first failed audio')
    parser.add_argument('--verbose', action='store_true', help='Log every stream turn and error trace')
    add_inference_arguments(parser, default_pace='fast')
    args = parser.parse_args(argv)
    if args.jsonl_dir is not None:
        if not args.field_prefix or any(c.isspace() for c in args.field_prefix):
            parser.error('--jsonl-dir requires a nonempty --field-prefix without whitespace')
    else:
        if args.field_prefix or args.check_only:
            parser.error('--field-prefix and --check-only require --jsonl-dir')
        if args.output_jsonl.suffix.lower() != '.jsonl':
            parser.error('--output-jsonl must end in .jsonl')
    if not args.check_only:
        validate_inference_arguments(parser, args)
    try:
        entries = collect_audio_files(args.audio_dir, recursive=args.recursive,
                                      vid_mode=args.vid_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_jsonl is not None:
        for path in (args.output_jsonl, errors_path(args.output_jsonl)):
            if path.exists():
                parser.error(f'Output already exists; choose a new path: {path}')
    return args, entries


def main(argv=None):
    args, entries = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        stream=sys.stderr)
    previous_level = infer_audio.LOG.level
    infer_audio.LOG.setLevel(logging.INFO if args.verbose else logging.WARNING)
    try:
        return run_update(args, entries) if args.jsonl_dir is not None else run_batch(args, entries)
    except KeyboardInterrupt:
        return 130
    except Exception:
        LOG.exception('Batch stopped. Previously saved results remain; '
                      'the failing audio is not saved as a complete translation.')
        return 1
    finally:
        infer_audio.LOG.setLevel(previous_level)


if __name__ == '__main__':
    raise SystemExit(main())
