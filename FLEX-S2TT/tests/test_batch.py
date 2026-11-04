"""Standalone CPU checks; no checkpoint, Web dependencies or GPU required."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer_audio
import infer_folder as batch
from flex_model import ModelEngine


def wav(path, seconds=1.25, value=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.full(round(seconds * 16000), value, dtype='<i2').tobytes())
    return path


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='flex-batch-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / '音频 folder'
        self.inputs.mkdir()
        self.result = self.root / '输出' / 'result.jsonl'

    def options(self, *extra):
        return ['--backend', 'mock', '--audio-dir', str(self.inputs),
                '--output-jsonl', str(self.result), '--no-warmup',
                '--ffmpeg', 'intentionally-missing-ffmpeg', *extra]

    def test_single_engine_fresh_sessions_exact_chunks_tail_and_schema(self):
        wav(self.inputs / '02.wav', seconds=1.25, value=2000)
        wav(self.inputs / '01 中文.WAV', seconds=3.375, value=1000)
        (self.inputs / 'ignore.txt').write_text('skip', encoding='utf-8')
        wav(self.inputs / 'nested' / 'not_scanned.wav')
        engines, sessions, requests = [], [], []

        class AuditedEngine(ModelEngine):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                engines.append(inner)

            def new_session(inner, target):
                session = super().new_session(target)
                sessions.append(session)
                number, original = len(sessions) - 1, session.generate

                def generate(audio):
                    requests.append((number, len(session.audios), audio.copy()))
                    return original(audio)

                session.generate = generate
                return session

        with mock.patch.object(batch, 'ModelEngine', AuditedEngine):
            code = batch.main(self.options('--direction', 'zh-en'))
        self.assertEqual(code, 0)
        self.assertEqual(len(engines), 1)
        self.assertTrue(engines[0]._closed)
        self.assertEqual(len(sessions), 2)
        self.assertTrue(all(session._closed for session in sessions))
        self.assertTrue(all(session.target_lang == 'en' for session in sessions))
        self.assertEqual([(n, turn) for n, turn, _ in requests],
                         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)])
        for number, turn, audio in requests:
            count = 6000 if (number, turn) == (0, 3) else (
                4000 if (number, turn) == (1, 1) else 16000)
            expected = np.zeros(16000, dtype=np.float32)
            expected[:count] = (1000 if number == 0 else 2000) / 32768.0
            np.testing.assert_array_equal(audio, expected)
        rows = records(self.result)
        self.assertEqual([row['vid'] for row in rows], ['01 中文.WAV', '02.wav'])
        self.assertTrue(all(set(row) == {'vid', 'translation'} for row in rows))
        self.assertEqual(rows[0]['translation'], '[DEMO] Audio received; '
                         ' this is a transport test, not a translation. [DEMO] Audio received; ')
        self.assertEqual(rows[1]['translation'], '[DEMO] Audio received; ')
        self.assertEqual(records(batch.errors_path(self.result)), [])

    def test_failure_does_not_save_partial_translation_and_next_file_runs(self):
        wav(self.inputs / '01_too_long.wav', seconds=1.5)
        wav(self.inputs / '02_ok.wav', seconds=0.5)
        self.assertEqual(batch.main(self.options('--max-audio-seconds', '1')), 1)
        self.assertEqual([row['vid'] for row in records(self.result)], ['02_ok.wav'])
        errors = records(batch.errors_path(self.result))
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['vid'], '01_too_long.wav')
        self.assertEqual(errors[0]['processed_audio_s'], 1.0)
        self.assertFalse(errors[0]['complete_input'])
        self.assertNotIn('translation', errors[0])

    def test_bad_audio_is_logged_and_good_audio_still_finishes(self):
        (self.inputs / '01_bad.wav').write_bytes(b'not a wave file')
        wav(self.inputs / '02_ok.wav', seconds=0.5)
        self.assertEqual(batch.main(self.options()), 1)
        self.assertEqual([row['vid'] for row in records(self.result)], ['02_ok.wav'])
        self.assertEqual(records(batch.errors_path(self.result))[0]['vid'], '01_bad.wav')

    def test_fail_fast_stops_after_error(self):
        (self.inputs / '01_bad.wav').write_bytes(b'bad')
        wav(self.inputs / '02_ok.wav')
        self.assertEqual(batch.main(self.options('--fail-fast')), 1)
        self.assertEqual(records(self.result), [])
        self.assertEqual(len(records(batch.errors_path(self.result))), 1)

    def test_recursive_duplicate_names_require_relative_ids(self):
        wav(self.inputs / 'speaker1' / 'a.wav')
        wav(self.inputs / 'speaker2' / 'a.wav')
        with self.assertRaisesRegex(ValueError, 'Duplicate vid'):
            batch.collect_audio_files(self.inputs, recursive=True)
        entries = batch.collect_audio_files(self.inputs, recursive=True, vid_mode='relative')
        self.assertEqual([vid for _, vid in entries], ['speaker1/a.wav', 'speaker2/a.wav'])
        self.assertEqual(batch.main(self.options('--recursive', '--vid-mode', 'relative')), 0)
        self.assertEqual([row['vid'] for row in records(self.result)],
                         ['speaker1/a.wav', 'speaker2/a.wav'])

    def test_existing_output_or_errors_are_not_overwritten(self):
        wav(self.inputs / 'a.wav')
        for path in (self.result, batch.errors_path(self.result)):
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original', encoding='utf-8')
                with mock.patch.object(batch, 'ModelEngine') as engine:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                        batch.main(self.options())
                self.assertEqual(failure.exception.code, 2)
                self.assertFalse(engine.called)
                self.assertEqual(path.read_text(encoding='utf-8'), 'original')
                path.unlink()

    def test_empty_folder_fails_before_model_or_output_creation(self):
        with mock.patch.object(batch, 'ModelEngine') as engine:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                batch.main(self.options())
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse(engine.called)
        self.assertFalse(self.result.exists())

    def test_rejected_protocol_is_not_written_as_success(self):
        wav(self.inputs / 'a.wav', seconds=0.5)
        original = ModelEngine.new_session

        def invalid(engine, target):
            session = original(engine, target)
            generate = session.generate

            def bad_reply(audio):
                result = generate(audio)
                result.reply = 'missing protocol tag'
                return result

            session.generate = bad_reply
            return session

        with mock.patch.object(ModelEngine, 'new_session', invalid):
            self.assertEqual(batch.main(self.options()), 1)
        self.assertEqual(records(self.result), [])
        self.assertEqual(records(batch.errors_path(self.result))[0]['rejected_turn']['response'],
                         'missing protocol tag')

    def test_interrupt_keeps_completed_audio_and_closes_engine(self):
        wav(self.inputs / '01_ok.wav', seconds=0.5)
        wav(self.inputs / '02_interrupt.wav', seconds=0.5)
        original = batch.run_stream
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                self.assertEqual([row['vid'] for row in records(self.result)], ['01_ok.wav'])
                raise KeyboardInterrupt()
            return original(*args, **kwargs)

        with mock.patch.object(batch, 'run_stream', interrupted):
            self.assertEqual(batch.main(self.options()), 130)
        self.assertEqual([row['vid'] for row in records(self.result)], ['01_ok.wav'])
        self.assertEqual(records(batch.errors_path(self.result))[0]['error_type'], 'KeyboardInterrupt')
        self.assertTrue(calls[0]._closed)

    def test_old_single_file_entry_still_runs(self):
        path = wav(self.inputs / 'a.wav', seconds=0.5)
        destination = self.root / 'single'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(infer_audio.main([
                '--audio', str(path), '--output-dir', str(destination),
                '--backend', 'mock', '--pace', 'fast', '--no-warmup',
            ]), 0)
        self.assertIn('【演示】', (destination / 'translation.txt').read_text(encoding='utf-8'))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg unavailable')
    def test_compressed_audio_decodes_through_ffmpeg(self):
        source = wav(self.root / 'source.wav', seconds=0.375)
        compressed = self.inputs / '压缩.flac'
        subprocess.run([shutil.which('ffmpeg'), '-nostdin', '-hide_banner', '-loglevel',
                        'error', '-i', str(source), str(compressed)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        self.assertEqual(batch.main(self.options('--ffmpeg', shutil.which('ffmpeg'))), 0)
        self.assertEqual(records(self.result)[0]['vid'], '压缩.flac')

    def test_subprocess_cli_runs_outside_project_directory(self):
        wav(self.inputs / '输入.wav', seconds=0.5)
        environment = dict(os.environ, PYTHONIOENCODING='utf-8')
        environment.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable, str(ROOT / 'infer_folder.py'), *self.options()],
                                cwd=self.root, env=environment, capture_output=True,
                                text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('MOCK ONLY', result.stderr)
        self.assertEqual(records(self.result)[0]['vid'], '输入.wav')

    def test_warmup_runs_once_for_whole_folder(self):
        wav(self.inputs / '01.wav', seconds=0.5)
        wav(self.inputs / '02.wav', seconds=0.5)
        options = [part for part in self.options() if part != '--no-warmup']
        with mock.patch.object(ModelEngine, 'warmup', autospec=True) as warmup:
            self.assertEqual(batch.main(options), 0)
        self.assertEqual(warmup.call_count, 1)

    def test_batch_shell_launcher_supports_paths_with_spaces_and_preserves_gpu(self):
        shell = shutil.which('bash') if os.name != 'nt' else None
        if os.name == 'nt' and shutil.which('git'):
            candidate = Path(shutil.which('git')).resolve().parents[1] / 'bin' / 'bash.exe'
            if candidate.is_file():
                shell = str(candidate)
        if not shell:
            self.skipTest('Standalone Bash is unavailable')
        wav(self.inputs / 'a.wav', seconds=0.5)
        environment = dict(os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='6')
        result = subprocess.run([
            shell, (ROOT / 'run_batch.sh').as_posix(), 'unused mock model',
            self.inputs.as_posix(), self.result.as_posix(), '--backend', 'mock',
            '--no-warmup', '--direction', 'zh-en',
        ], cwd=self.root, env=environment, capture_output=True,
            text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(records(self.result)[0]['vid'], 'a.wav')
        self.assertIn('[DEMO]', records(self.result)[0]['translation'])


if __name__ == '__main__':
    unittest.main()
