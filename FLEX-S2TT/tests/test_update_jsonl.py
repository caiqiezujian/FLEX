"""In-place updates: preserve other systems' data, matching, failure and CLI checks."""
import codecs
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer_folder as batch
import jsonl_update
from test_batch import wav


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='flex-update-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / '音频 folder'
        self.jsonls = self.root / '原始 jsonl'
        self.audio.mkdir()
        self.jsonls.mkdir()

    def options(self, *extra):
        return ['--audio-dir', str(self.audio), '--jsonl-dir', str(self.jsonls),
                '--field-prefix', 'Flex_1022', '--backend', 'mock', '--no-warmup',
                '--ffmpeg', 'intentionally-missing-ffmpeg', *extra]

    def create(self, name, rows):
        path = self.jsonls / name
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                        encoding='utf-8')
        return path

    def test_updates_two_folders_separately_preserves_reference_and_all_other_fields(self):
        # Filenames deliberately do not identify the owning JSONL: vid is authoritative.
        wav(self.audio / 'en2zh-01-tech_004.wav', seconds=0.5)
        wav(self.audio / 'Dataflow_001.wav', seconds=0.5)
        row = dict(vid='en2zh-01-tech_004', transcript='  Original transcript.\n',
                   translation='参考原文', original_vids=['different_original_id'],
                   Flex_1016_trans='other model', Flex_1022_trans='old result',
                   Flex_1022_asr='old ASR', other=['1', '', {'text': '嵌套内容'}])
        path = self.create('任意名称.jsonl', [row])
        other = self.create('另一文件.jsonl', [dict(vid='Dataflow_001', transcript='English')])
        missing = self.create('en2zh-04-fin.jsonl', [dict(vid='absent', translation='保持')])
        original_missing = missing.read_bytes()
        with mock.patch.object(batch, 'ModelEngine', wraps=batch.ModelEngine) as engine:
            self.assertEqual(batch.main(self.options()), 0)
        self.assertEqual(engine.call_count, 1)
        updated = read_rows(path)[0]
        self.assertEqual(updated['Flex_1022_asr'], row['transcript'])
        self.assertIn('【演示】', updated['Flex_1022_trans'])
        for field in row.keys() - {'Flex_1022_trans', 'Flex_1022_asr'}:
            self.assertEqual(updated[field], row[field])
        self.assertIn('Flex_1022_trans', read_rows(other)[0])
        self.assertEqual(missing.read_bytes(), original_missing)
        self.assertEqual(sorted(p.name for p in self.jsonls.iterdir()),
                         sorted([path.name, other.name, missing.name]))

    def test_bom_crlf_blank_lines_unmatched_bytes_and_no_final_newline_survive(self):
        wav(self.audio / 'a.wav', seconds=0.5)
        wav(self.audio / 'b.wav', seconds=0.5)
        path = self.jsonls / 'preserve.jsonl'
        untouched = b'{ "vid" : "no-audio", "x": 1.000 }\r\n'
        path.write_bytes(codecs.BOM_UTF8 + b'{"vid":"a","transcript":"a"}\r\n' +
                         b'\r\n' + untouched + b'{"vid":"b","transcript":"b"}')
        self.assertEqual(batch.main(self.options()), 0)
        saved = path.read_bytes()
        self.assertTrue(saved.startswith(codecs.BOM_UTF8))
        self.assertIn(b'\r\n\r\n' + untouched, saved)
        self.assertFalse(saved.endswith(b'\n'))
        self.assertEqual([r['vid'] for r in read_rows(path)], ['a', 'no-audio', 'b'])
        self.assertTrue(all('Flex_1022_trans' in read_rows(path)[i] for i in (0, 2)))

    def test_missing_audio_rows_ignored_and_extra_audio_skipped_without_new_rows(self):
        wav(self.audio / 'a.wav', seconds=0.5)
        wav(self.audio / 'extra.wav', seconds=0.5)
        path = self.create('a.jsonl', [dict(vid='a', transcript='a'), dict(vid='not_found')])
        with mock.patch.object(batch, 'run_stream', wraps=batch.run_stream) as run:
            self.assertEqual(batch.main(self.options()), 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(read_rows(path)[1], {'vid': 'not_found'})
        self.assertEqual(len(read_rows(path)), 2)

    def test_extension_vid_and_relative_mode(self):
        wav(self.audio / 'folder' / 'a.WAV', seconds=0.5)
        path = self.create('unrelated.jsonl', [dict(vid='folder/a.WAV', transcript='test')])
        self.assertEqual(batch.main(self.options('--recursive', '--vid-mode', 'relative')), 0)
        self.assertEqual(read_rows(path)[0]['vid'], 'folder/a.WAV')
        self.assertEqual(read_rows(path)[0]['Flex_1022_asr'], 'test')

    def test_duplicate_audio_stems_rejected_before_model(self):
        wav(self.audio / 'a.wav')
        (self.audio / 'a.mp3').write_bytes(b'not decoded')
        path = self.create('a.jsonl', [dict(vid='a', transcript='a')])
        original = path.read_bytes()
        with mock.patch.object(batch, 'ModelEngine') as engine:
            self.assertEqual(batch.main(self.options()), 1)
        engine.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_duplicate_matching_rows_rejected_before_any_write(self):
        wav(self.audio / 'a.wav')
        first = self.create('first.jsonl', [dict(vid='a', transcript='a')])
        second = self.create('second.jsonl', [dict(vid='a.wav', transcript='other')])
        before = {p: p.read_bytes() for p in (first, second)}
        with mock.patch.object(batch, 'ModelEngine') as engine:
            self.assertEqual(batch.main(self.options()), 1)
        engine.assert_not_called()
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_missing_transcript_in_matched_row_rejected(self):
        wav(self.audio / 'a.wav')
        path = self.create('a.jsonl', [dict(vid='a', translation='Never use reference as ASR')])
        original = path.read_bytes()
        with mock.patch.object(batch, 'ModelEngine') as engine:
            self.assertEqual(batch.main(self.options()), 1)
        engine.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_malformed_json_preflight_does_not_partially_update(self):
        wav(self.audio / 'a.wav')
        path = self.jsonls / 'a.jsonl'
        path.write_bytes(b'{"vid":"a","transcript":"a"}\n{"broken"\n')
        original = path.read_bytes()
        with mock.patch.object(batch, 'ModelEngine') as engine:
            self.assertEqual(batch.main(self.options()), 1)
        engine.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_check_only_does_not_need_model_and_does_not_write(self):
        wav(self.audio / 'a.wav')
        path = self.create('a.jsonl', [dict(vid='a', transcript='a')])
        original = path.read_bytes()
        with mock.patch.object(batch, 'ModelEngine') as engine:
            self.assertEqual(batch.main(self.options('--backend', 'vllm', '--check-only')), 0)
        engine.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_no_matching_rows_reports_failure_without_loading_model(self):
        wav(self.audio / 'a.wav')
        path = self.create('a.jsonl', [dict(vid='other', transcript='other')])
        original = path.read_bytes()
        with mock.patch.object(batch, 'ModelEngine') as engine:
            self.assertEqual(batch.main(self.options()), 1)
        engine.assert_not_called()
        self.assertEqual(path.read_bytes(), original)

    def test_failed_audio_retains_old_results_and_next_audio_is_saved(self):
        wav(self.audio / '01.wav', seconds=1.5)
        wav(self.audio / '02.wav', seconds=0.5)
        old = dict(vid='01', transcript='a', Flex_1022_trans='previous', Flex_1022_asr='previous asr')
        path = self.create('a.jsonl', [old, dict(vid='02', transcript='b')])
        self.assertEqual(batch.main(self.options('--max-audio-seconds', '1')), 1)
        self.assertEqual(read_rows(path)[0], old)
        self.assertIn('Flex_1022_trans', read_rows(path)[1])
        self.assertEqual(len(list(self.jsonls.iterdir())), 1)

    def test_interrupt_keeps_completed_row_and_current_row_unchanged(self):
        for name in ('01', '02'):
            wav(self.audio / (name + '.wav'), seconds=0.5)
        old = dict(vid='02', transcript='second')
        path = self.create('a.jsonl', [dict(vid='01', transcript='first'), old])
        original_run = batch.run_stream
        count, engines = [0], []

        def interrupt(*args, **kwargs):
            count[0] += 1
            engines.append(args[1])
            if count[0] == 2:
                self.assertIn('Flex_1022_trans', read_rows(path)[0])
                raise KeyboardInterrupt()
            return original_run(*args, **kwargs)

        with mock.patch.object(batch, 'run_stream', interrupt):
            self.assertEqual(batch.main(self.options()), 130)
        self.assertEqual(read_rows(path)[1], old)
        self.assertTrue(engines[0]._closed)

    def test_empty_completed_translation_is_not_replaced_by_reference_or_transcript(self):
        wav(self.audio / 'a.wav', seconds=0.5)
        path = self.create('a.jsonl', [dict(vid='a', transcript='source', translation='reference')])

        def all_wait(args, engine, reader, output, state, **kwargs):
            state.last_tag = 'WAIT'
            state.turns = 1

        with mock.patch.object(batch, 'run_stream', all_wait):
            self.assertEqual(batch.main(self.options()), 0)
        row = read_rows(path)[0]
        self.assertEqual(row['Flex_1022_trans'], '')
        self.assertEqual(row['Flex_1022_asr'], 'source')
        self.assertEqual(row['translation'], 'reference')

    def test_storage_failure_is_atomic_and_removes_temp_file(self):
        wav(self.audio / 'a.wav', seconds=0.5)
        path = self.create('a.jsonl', [dict(vid='a', transcript='a')])
        original = path.read_bytes()
        with mock.patch.object(jsonl_update.os, 'replace', side_effect=OSError('disk failure')):
            self.assertEqual(batch.main(self.options()), 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.jsonls.iterdir()), [path])

    def test_external_edit_is_not_overwritten(self):
        wav(self.audio / 'a.wav', seconds=0.5)
        path = self.create('a.jsonl', [dict(vid='a', transcript='a')])
        external = b'{"vid":"a", "transcript":"edited during inference"}\n'
        original_run = batch.run_stream

        def edited(*args, **kwargs):
            original_run(*args, **kwargs)
            path.write_bytes(external)

        with mock.patch.object(batch, 'run_stream', edited):
            self.assertEqual(batch.main(self.options()), 1)
        self.assertEqual(path.read_bytes(), external)

    def test_prefix_required_and_legacy_flags_not_mixed(self):
        wav(self.audio / 'a.wav')
        self.create('a.jsonl', [dict(vid='a', transcript='a')])
        for options in (['--audio-dir', str(self.audio), '--jsonl-dir', str(self.jsonls)],
                        self.options('--field-prefix', 'bad prefix'),
                        self.options('--output-jsonl', str(self.root / 'result.jsonl'))):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                batch.main(options)
            self.assertEqual(error.exception.code, 2)

    def test_bash_four_positionals_and_rerun_updates_only_selected_prefix(self):
        shell = shutil.which('bash') if os.name != 'nt' else None
        if os.name == 'nt' and shutil.which('git'):
            candidate = Path(shutil.which('git')).resolve().parents[1] / 'bin' / 'bash.exe'
            if candidate.is_file():
                shell = str(candidate)
        if not shell:
            self.skipTest('Standalone Bash unavailable')
        wav(self.audio / '中文_001.wav', seconds=0.5)
        path = self.create('中文.jsonl', [dict(vid='中文_001', transcript='source',
                                            Flex_1016_trans='existing model result')])
        environment = dict(os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES='0')
        command = [shell, (ROOT / 'run_batch.sh').as_posix(), 'unused mock model',
                   self.audio.as_posix(), self.jsonls.as_posix(), 'Flex_1022',
                   '--backend', 'mock', '--no-warmup']
        for _ in range(2):
            result = subprocess.run(command, cwd=self.root, env=environment,
                                    capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('SAVED [1/1]', result.stderr)
            row = read_rows(path)[0]
            self.assertEqual(row['Flex_1022_asr'], 'source')
            self.assertEqual(row['Flex_1016_trans'], 'existing model result')
            self.assertIn('【演示】', row['Flex_1022_trans'])


if __name__ == '__main__':
    unittest.main()
