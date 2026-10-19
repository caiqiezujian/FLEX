"""Exercise the Bash entrypoint without importing or loading GPU libraries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def bash():
    if os.name == 'nt':
        # Windows' system32/bash.exe is a WSL launcher, not a standalone shell.
        git = shutil.which('git')
        candidates = []
        if git:
            candidates.append(Path(git).resolve().parents[1] / 'bin' / 'bash.exe')
        candidates.extend(
            Path(os.environ.get(key, 'C:/Program Files')) / 'Git' / 'bin' / 'bash.exe'
            for key in ('ProgramFiles', 'ProgramW6432')
        )
        executable = next((str(path) for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which('bash')
    if executable is None:
        pytest.skip('Bash is required for launcher tests (Git Bash on Windows)')
    return executable


@pytest.fixture
def launcher(tmp_path):
    script_dir = tmp_path / 'project with spaces'
    script_dir.mkdir()
    script = script_dir / 'run_infer.sh'
    shutil.copy2(ROOT / 'run_infer.sh', script)
    # A fake inference entrypoint captures arguments/env; it never opens a model.
    (script_dir / 'infer_audio.py').write_text(
        'import json, os, sys\n'
        'print(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(),\n'
        '  "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),\n'
        '  "worker": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),\n'
        '  "unbuffered": sys.stdout.write_through}))\n',
        encoding='utf-8',
    )
    caller = tmp_path / 'caller with spaces'
    caller.mkdir()
    return script, caller


def invoke(bash, script, caller, args, **environment):
    env = os.environ.copy()
    env.update(PYTHON=Path(sys.executable).as_posix(), PYTHONIOENCODING='utf-8')
    env.update(environment)
    return subprocess.run(
        [bash, script.as_posix(), *args], cwd=caller, env=env,
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding='utf-8', timeout=20, check=False,
    )


@pytest.mark.parametrize('flag', ['-h', '--help'])
def test_help_does_not_start_python(bash, launcher, flag):
    script, caller = launcher
    result = invoke(bash, script, caller, [flag], PYTHON='nonexistent-flex-test-python')
    assert result.returncode == 0
    assert 'Usage: bash run_infer.sh MODEL_DIR AUDIO_FILE' in result.stdout
    assert result.stderr == ''


@pytest.mark.parametrize('args', [[], ['model directory']])
def test_missing_arguments_fail_before_python(bash, launcher, args):
    script, caller = launcher
    result = invoke(bash, script, caller, args, PYTHON='nonexistent-flex-test-python')
    assert result.returncode == 2
    assert 'Usage:' in result.stderr
    assert result.stdout == ''


def test_defaults_paths_spaces_and_caller_directory(bash, launcher):
    script, caller = launcher
    result = invoke(
        bash, script, caller,
        ['relative model', 'audio recordings/source clip.wav', '--output-dir', 'outputs/test run'],
        CUDA_VISIBLE_DEVICES='6', VLLM_WORKER_MULTIPROC_METHOD='fork',
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload['cwd']).resolve() == caller.resolve()
    assert payload['cuda'] == '6'
    assert payload['worker'] == 'spawn'
    assert payload['unbuffered'] is True
    assert payload['argv'] == [
        '--model', 'relative model', '--audio', 'audio recordings/source clip.wav',
        '--backend', 'vllm', '--direction', 'en-zh', '--chunk-seconds', '1',
        '--pace', 'fast', '--protocol-form', 'special', '--image-slots', '1',
        '--dtype', 'bfloat16', '--gpu-memory-utilization', '0.90',
        '--max-model-len', '8192', '--output-dir', 'outputs/test run',
    ]


def test_extra_options_are_last_for_argparse_override(bash, launcher):
    script, caller = launcher
    extra = [
        '--backend', 'mock', '--direction', 'zh-en', '--chunk-seconds', '2',
        '--pace', 'realtime', '--protocol-form', 'text', '--image-slots', '0',
        '--max-model-len', '4096', '--max-tokens', '256',
    ]
    result = invoke(bash, script, caller, ['model', 'audio.wav', *extra])
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)['argv']
    assert argv[-len(extra):] == extra
    assert argv.index('--backend') < len(argv) - len(extra)


@pytest.mark.parametrize('devices', ['5', '2,3', ''])
def test_cuda_selection_is_never_replaced(bash, launcher, devices):
    script, caller = launcher
    result = invoke(bash, script, caller, ['model', 'audio.wav'], CUDA_VISIBLE_DEVICES=devices)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cuda'] == devices


def test_unset_cuda_selection_is_left_unset(bash, launcher, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    script, caller = launcher
    result = invoke(bash, script, caller, ['model', 'audio.wav'])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cuda'] is None


def test_real_entrypoint_mock_inference_uses_caller_relative_paths(bash, tmp_path):
    caller = tmp_path / 'actual caller'
    caller.mkdir()
    audio_path = caller / 'my audio.wav'
    with wave.open(str(audio_path), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\x00\x00' * 20000)  # 1 second + a 0.25-second tail.
    result = invoke(
        bash, ROOT / 'run_infer.sh', caller,
        ['unused mock model', 'my audio.wav', '--backend', 'mock', '--direction', 'zh-en',
         '--no-warmup', '--output-dir', 'my output'],
    )
    assert result.returncode == 0, result.stderr
    output = caller / 'my output'
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text('utf-8').splitlines()]
    assert events[0]['backend'] == 'mock'
    assert events[0]['direction'] == 'zh-en'
    assert events[0]['pace'] == 'fast'
    assert events[0]['protocol_form'] == 'special'
    assert events[0]['image_slots'] == 1
    assert events[-1]['type'] == 'done'
    assert events[-1]['complete_input'] is True
    assert events[-1]['processed_audio_s'] == 1.25
    assert events[-1]['turns'] == 2
    transcript = (output / 'translation.txt').read_text('utf-8')
    assert transcript == ''.join(event['delta'] for event in events if event['type'] == 'turn')
    assert '<|' not in transcript
