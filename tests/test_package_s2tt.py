"""Portable S2TT package checks, including isolated CPU-only CLI inference."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave
import zipfile

import numpy as np
import pytest

from tools import package_s2tt


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'infer_audio.py', 'flex_model.py', 'run_infer.sh',
    'README.md', 'requirements-infer.txt',
}
SAMPLE_RATE = 16000
SAMPLE_COUNT = 54000  # 3.375 seconds: three full chunks and a short tail.


@pytest.fixture
def extracted_bundle(tmp_path):
    directory = tmp_path / 'built source'
    archive_path = tmp_path / 'bundle.zip'
    assert package_s2tt.build(ROOT, directory, archive_path) == 5
    extracted = tmp_path / 'extracted copy'
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extracted)
    return extracted / 'FLEX-S2TT'


def test_bundle_has_exactly_five_files_and_only_one_import_rewrite():
    files = package_s2tt.bundle_files(ROOT)
    assert set(files) == FILES
    original = (ROOT / 'infer_audio.py').read_text(encoding='utf-8')
    expected = original.replace('from flex_web.model import ', 'from flex_model import ', 1)
    assert files['infer_audio.py'].decode('utf-8') == expected
    assert files['flex_model.py'].decode('utf-8') == (
        ROOT / 'flex_web/model.py').read_text(encoding='utf-8')
    assert files['run_infer.sh'].decode('utf-8') == (
        ROOT / 'run_infer.sh').read_text(encoding='utf-8')
    assert files['README.md'].decode('utf-8') == (
        ROOT / 'docs/S2TT-QUICKSTART.md').read_text(encoding='utf-8')
    for name, content in files.items():
        assert b'\r' not in content, name
        assert not content.startswith(b'\xef\xbb\xbf'), name
        if name.endswith('.py'):
            compile(content, name, 'exec')
            tree = ast.parse(content)
            for node in ast.walk(tree):
                modules = ([node.module] if isinstance(node, ast.ImportFrom)
                           else [entry.name for entry in node.names]
                           if isinstance(node, ast.Import) else [])
                assert not any(module and module.split('.')[0] in {
                    'flex_web', 'fastapi', 'uvicorn', 'starlette',
                } for module in modules), name


def test_folder_and_zip_contents_integrity_and_linux_permissions(tmp_path):
    directory, archive_path = tmp_path / 'source', tmp_path / 'portable.zip'
    package_s2tt.build(ROOT, directory, archive_path)
    expected = package_s2tt.bundle_files(ROOT)
    assert {path.name for path in directory.iterdir()} == FILES
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {'FLEX-S2TT/' + name for name in FILES}
        for name, content in expected.items():
            assert (directory / name).read_bytes() == content
            assert archive.read('FLEX-S2TT/' + name) == content
            info = archive.getinfo('FLEX-S2TT/' + name)
            assert info.create_system == 3
            assert (info.external_attr >> 16) == (0o100755 if name.endswith('.sh') else 0o100644)


def _bash_executable():
    if os.name != 'nt':
        return shutil.which('bash')
    # Windows' system32/bash.exe may be an unconfigured WSL launcher.
    git = shutil.which('git')
    if git:
        candidate = Path(git).resolve().parent.parent / 'bin/bash.exe'
        if candidate.is_file():
            return str(candidate)
    return None


def test_bundled_shell_has_valid_bash_syntax():
    bash = _bash_executable()
    if not bash:
        pytest.skip('Bash is unavailable; Python/ZIP checks remain active')
    result = subprocess.run(
        [bash, '-n'], input=package_s2tt.bundle_files(ROOT)['run_infer.sh'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')


@pytest.mark.parametrize('existing', ['directory', 'zip', 'both'])
def test_existing_destinations_are_preserved_without_partial_build(tmp_path, existing):
    directory, archive_path = tmp_path / 'source', tmp_path / 'portable.zip'
    if existing in ('directory', 'both'):
        directory.mkdir()
        (directory / 'keep.txt').write_bytes(b'original directory contents')
    if existing in ('zip', 'both'):
        archive_path.write_bytes(b'original archive contents')
    with pytest.raises(FileExistsError, match='nothing is overwritten'):
        package_s2tt.build(ROOT, directory, archive_path)
    if existing in ('directory', 'both'):
        assert list(directory.iterdir()) == [directory / 'keep.txt']
        assert (directory / 'keep.txt').read_bytes() == b'original directory contents'
    else:
        assert not directory.exists()
    if existing in ('zip', 'both'):
        assert archive_path.read_bytes() == b'original archive contents'
    else:
        assert not archive_path.exists()


@pytest.mark.parametrize('count', [0, 2])
def test_unexpected_model_import_refuses_packaging_before_writes(tmp_path, count):
    source = tmp_path / 'bad source'
    source.mkdir()
    (source / 'infer_audio.py').write_text(
        'from flex_web.model import ModelEngine\n' * count, encoding='utf-8')
    with pytest.raises(RuntimeError, match='exactly one shared model import'):
        package_s2tt.build(source, tmp_path / 'result', tmp_path / 'result.zip')
    assert not (tmp_path / 'result').exists()
    assert not (tmp_path / 'result.zip').exists()


def _write_test_wav(path):
    samples = ((np.arange(SAMPLE_COUNT, dtype=np.int64) * 37) % 30001 - 15000).astype('<i2')
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return path


# The subprocess imports only the extracted package. In addition to normal CLI
# output assertions, audit the actual PCM sent on every mock inference call so
# a skipped, reordered, replayed or future-audio chunk cannot pass unnoticed.
ISOLATED_MOCK = r'''
from pathlib import Path
import runpy
import sys
import numpy as np

bundle = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(bundle))
import flex_model
assert Path(flex_model.__file__).resolve().parent == bundle
assert 'flex_web' not in sys.modules
expected = ((np.arange(54000, dtype=np.int64) * 37) % 30001 - 15000).astype(np.int16)
expected = expected.astype(np.float32) / 32768.0
original_generate = flex_model.ModelSession._mock_generate

def audited_generate(self, audio, plan, extra):
    index = self.total_committed_turns
    assert index < 4, 'Unexpected extra inference round'
    chunk = expected[index * 16000:(index + 1) * 16000]
    chunk = np.pad(chunk, (0, 16000 - len(chunk)))
    np.testing.assert_array_equal(audio, chunk)
    for previous_index, previous_audio in enumerate(self.audios):
        np.testing.assert_array_equal(
            previous_audio, expected[previous_index * 16000:(previous_index + 1) * 16000])
    return original_generate(self, audio, plan, extra)

flex_model.ModelSession._mock_generate = audited_generate
sys.argv[0] = str(bundle / 'infer_audio.py')
runpy.run_path(sys.argv[0], run_name='__main__')
'''


@pytest.mark.parametrize('audio_format', ['native_wav', 'ffmpeg_flac'])
def test_extracted_zip_runs_isolated_file_stream_with_exact_prefix_and_tail(
        extracted_bundle, tmp_path, audio_format):
    ffmpeg = shutil.which('ffmpeg')
    if audio_format == 'ffmpeg_flac' and ffmpeg is None:
        pytest.skip('Optional FFmpeg conversion is unavailable')
    outside = tmp_path / 'working directory outside repo'
    outside.mkdir()
    wav_path = _write_test_wav(outside / 'short input.wav')
    audio_path = wav_path
    if audio_format == 'ffmpeg_flac':
        audio_path = outside / 'short input.flac'
        converted = subprocess.run(
            [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(wav_path),
             '-c:a', 'flac', str(audio_path)],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
        )
        assert converted.returncode == 0, converted.stderr
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run(
        [sys.executable, '-c', ISOLATED_MOCK, str(extracted_bundle),
         '--audio', audio_path.name, '--backend', 'mock', '--direction', 'zh-en',
         '--pace', 'fast', '--chunk-seconds', '1', '--no-warmup',
         '--output-dir', 'result',
         '--ffmpeg', ffmpeg if audio_format == 'ffmpeg_flac' else 'intentionally-unavailable-ffmpeg'],
        cwd=outside, env=environment, capture_output=True,
        text=True, encoding='utf-8', errors='replace', timeout=30,
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    events = [json.loads(line) for line in (outside / 'result/events.jsonl').read_text(
        encoding='utf-8').splitlines()]
    assert [event['type'] for event in events] == ['start', 'model_ready'] + ['turn'] * 4 + ['done']
    assert events[0]['mock'] is True
    turns = [event for event in events if event['type'] == 'turn']
    assert [event['turn_id'] for event in turns] == [1, 2, 3, 4]
    assert [event['tag'] for event in turns] == ['FLEX', 'WAIT', 'DIRECT', 'FLEX']
    assert [event['source_start_s'] for event in turns] == [0.0, 1.0, 2.0, 3.0]
    assert [event['source_end_s'] for event in turns] == [1.0, 2.0, 3.0, 3.375]
    assert [event['model_audio_s'] for event in turns] == [1.0] * 4
    assert all(event['dispatch_lag_s'] is None for event in turns)
    assert turns[1]['delta'] == ''
    expected_text = ('[DEMO] Audio received; '
                     ' this is a transport test, not a translation. '
                     '[DEMO] Audio received; ')
    assert ''.join(event['delta'] for event in turns) == expected_text
    assert (outside / 'result/translation.txt').read_text(encoding='utf-8') == expected_text
    assert result.stdout == expected_text
    assert events[-1]['reason'] == 'eof'
    assert events[-1]['complete_input'] is True
    assert events[-1]['final_wait'] is False
    assert events[-1]['turns'] == 4
    assert events[-1]['processed_audio_s'] == events[-1]['read_audio_s'] == 3.375
