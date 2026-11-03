"""CPU-only contract checks for the Ascend adapter; no NPU inference is run."""
import ast
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flex_web_d910b import model, server

ROOT = Path(__file__).resolve().parents[1]


def install_fake_platform(monkeypatch, device_type='npu'):
    torch = types.ModuleType('torch')
    torch.npu = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch_npu', types.ModuleType('torch_npu'))
    monkeypatch.setitem(sys.modules, 'vllm_ascend', types.ModuleType('vllm_ascend'))
    platform = types.ModuleType('vllm.platforms')
    platform.current_platform = types.SimpleNamespace(device_type=device_type)
    monkeypatch.setitem(sys.modules, 'vllm.platforms', platform)


def test_platform_guard_accepts_npu(monkeypatch):
    install_fake_platform(monkeypatch)
    model.check_ascend_environment()


def test_platform_guard_rejects_missing_torch_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch_npu', None)
    with pytest.raises(RuntimeError):
        model.check_ascend_environment()


def test_platform_guard_rejects_non_npu(monkeypatch):
    install_fake_platform(monkeypatch, 'cuda')
    with pytest.raises(RuntimeError):
        model.check_ascend_environment()


def test_mock_backend_does_not_require_npu(monkeypatch):
    def fail_if_called():
        raise AssertionError('mock backend must not initialize NPU')
    monkeypatch.setattr(model, 'check_ascend_environment', fail_if_called)
    engine = model.ModelEngine(backend='mock')
    try:
        assert engine.new_session('en').system_prompt == model.SYSTEM_PROMPTS['en']
        assert engine.new_session('zh').system_prompt == model.SYSTEM_PROMPTS['zh']
    finally:
        engine.close()


@pytest.mark.parametrize('bad', [0, -1, True, 1.5])
def test_tensor_parallel_size_must_be_positive_integer(bad):
    with pytest.raises(ValueError):
        model.ModelOptions(tensor_parallel_size=bad)


def test_tensor_parallel_size_reaches_engine(monkeypatch, tmp_path):
    install_fake_platform(monkeypatch)
    observed = {}

    class Tokenizer:
        def encode(self, value, add_special_tokens=False):
            return [1]
        def convert_tokens_to_ids(self, value):
            return 2

    class LLM:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.llm_engine = types.SimpleNamespace(
                model_config=object(),
                engine_core=types.SimpleNamespace(shutdown=lambda: None),
            )

    class InputPreprocessor:
        def __init__(self, model_config, tokenizer, mm_processor_cache=None):
            pass

    class SamplingParams:
        def __init__(self, **kwargs):
            pass

    vllm = types.ModuleType('vllm')
    vllm.__version__ = '0.11.1'
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    inputs = types.ModuleType('vllm.inputs')
    inputs.TokensPrompt = dict
    preprocess = types.ModuleType('vllm.inputs.preprocess')
    preprocess.InputPreprocessor = InputPreprocessor
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.inputs', inputs)
    monkeypatch.setitem(sys.modules, 'vllm.inputs.preprocess', preprocess)
    monkeypatch.setattr(model, 'load_tokenizer_checked', lambda *args: Tokenizer())
    counter = types.SimpleNamespace(mode='exact', reason='test', clear=lambda: None)
    monkeypatch.setattr(model.AudioTokenCounter, 'for_preprocessor', lambda *a, **k: counter)
    engine = model.ModelEngine(model.ModelOptions(model=str(tmp_path), tensor_parallel_size=2, cpu_offload_gb=12))
    try:
        assert observed['tensor_parallel_size'] == 2
        assert observed['cpu_offload_gb'] == 12
    finally:
        engine.close()


@pytest.mark.parametrize('direction,target', [('en', 'English'), ('zh', '中文')])
def test_prompt_matches_sft_template(direction, target):
    tree = ast.parse((ROOT / 'SFT-Scripts' / 'prepare_duplex_data.py').read_text(encoding='utf-8'))
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(
                  isinstance(name, ast.Name) and name.id == 'PROTOCOL_SYSTEM_PROMPT_TEMPLATE'
                  for name in node.targets)]
    assert len(values) == 1
    assert model.SYSTEM_PROMPTS[direction] == values[0].format(target_lang=target)


def test_static_and_mock_web(tmp_path):
    assert server.STATIC == ROOT / 'static'
    assert (server.STATIC / 'index.html').is_file()
    app = server.create_app(server.Settings(backend='mock', log_dir=tmp_path / 'logs'))
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        config = client.get('/api/config')
        assert config.status_code == 200
        assert config.json()['backend'] == 'mock'


def test_cli_help_exposes_tensor_parallel_option():
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'flex_web_d910b.server', '--help'],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '--tensor-parallel-size' in result.stdout
    assert '--cpu-offload-gb' in result.stdout


def test_legacy_audio_preprocessor_receives_uncached_configuration():
    seen = {}
    class Legacy:
        def __init__(self, model_config, tokenizer, mm_processor_cache):
            seen.update(config=model_config, tokenizer=tokenizer, cache=mm_processor_cache)
    config, tokenizer = object(), object()
    result = model.make_audio_preprocessor(Legacy, types.SimpleNamespace(model_config=config), tokenizer)
    assert isinstance(result, Legacy)
    assert seen == {'config': config, 'tokenizer': tokenizer, 'cache': None}


def test_new_audio_preprocessor_receives_vllm_configuration():
    class New:
        def __init__(self, vllm_config):
            self.config = vllm_config
    config = object()
    result = model.make_audio_preprocessor(New, types.SimpleNamespace(vllm_config=config), object())
    assert result.config is config


def test_new_audio_preprocessor_rejects_missing_configuration():
    class New:
        def __init__(self, vllm_config):
            pass
    with pytest.raises(RuntimeError, match='vllm_config'):
        model.make_audio_preprocessor(New, types.SimpleNamespace(), object())


def test_audio_preprocessor_rejects_unknown_api():
    class Unknown:
        def __init__(self, other):
            pass
    with pytest.raises(RuntimeError, match='Unsupported'):
        model.make_audio_preprocessor(Unknown, types.SimpleNamespace(), object())


def write_pcm_wav(path, rate, frames):
    import wave
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)


def test_audio_smoke_reads_wav_and_commits_each_turn(monkeypatch, tmp_path, capsys):
    from flex_web_d910b import smoke_audio
    import numpy as np
    wav_path = tmp_path / 'short.wav'
    samples = np.concatenate([np.full(model.SAMPLE_RATE, v, dtype='<i2')
                              for v in (-32768, 0, 16384)])
    write_pcm_wav(wav_path, model.SAMPLE_RATE, samples.tobytes())
    observed = {'generated': [], 'committed': [], 'closed': []}

    class Session:
        def generate(self, audio):
            index = len(observed['generated'])
            observed['generated'].append(audio.copy())
            return types.SimpleNamespace(reply=['<|WAIT|>', '<|FLEX|>A', '<|DIRECT|> B'][index],
                                         timings={'engine_generate_s': 0.1})
        def commit(self, audio, generation):
            observed['committed'].append((audio.copy(), generation.reply))
        def close(self):
            observed['closed'].append('session')

    class Engine:
        def __init__(self, options):
            observed['options'] = options
        def new_session(self, target):
            observed['target'] = target
            return Session()
        def close(self):
            observed['closed'].append('engine')

    monkeypatch.setattr(smoke_audio, 'ModelEngine', Engine)
    monkeypatch.setattr(sys, 'argv', ['smoke_audio', str(tmp_path), str(wav_path),
                                     '--target', 'en', '--tensor-parallel-size', '2',
                                     '--audio-token-cache-size', '64',
                                     '--mm-processor-cache-gb', '1', '--enable-graphs',
                                     '--enable-prefix-caching'])
    smoke_audio.main()
    assert observed['target'] == 'en'
    assert observed['options'].tensor_parallel_size == 2
    assert observed['options'].audio_token_cache_size == 64
    assert observed['options'].mm_processor_cache_gb == 1
    assert observed['options'].prefix_caching is True
    assert observed['options'].enforce_eager is False
    assert [entry[1] for entry in observed['committed']] == ['<|WAIT|>', '<|FLEX|>A', '<|DIRECT|> B']
    assert len(observed['generated']) == len(observed['committed']) == 3
    for audio, expected in zip(observed['generated'], [-1., 0., .5]):
        assert audio.dtype == np.float32 and len(audio) == model.SAMPLE_RATE
        assert np.all(audio == expected)
    assert observed['closed'] == ['session', 'engine']
    output = capsys.readouterr().out
    assert '<|WAIT|>' in output and '<|FLEX|>A' in output and '<|DIRECT|> B' in output


@pytest.mark.parametrize('rate,seconds,error', [
    (8000, 3, 'mono 16 kHz 16-bit PCM WAV'),
    (16000, 2, 'at least --turns seconds'),
])
def test_audio_smoke_rejects_invalid_wav_before_loading_model(monkeypatch, tmp_path, capsys,
                                                              rate, seconds, error):
    from flex_web_d910b import smoke_audio
    wav_path = tmp_path / 'invalid.wav'
    write_pcm_wav(wav_path, rate, b'\x00\x00' * rate * seconds)
    monkeypatch.setattr(smoke_audio, 'ModelEngine', lambda *a: pytest.fail('model must not load'))
    monkeypatch.setattr(sys, 'argv', ['smoke_audio', str(tmp_path), str(wav_path)])
    with pytest.raises(SystemExit) as exc:
        smoke_audio.main()
    assert exc.value.code == 2
    assert error in capsys.readouterr().err


def test_diagnose_checkpoint_reports_shards_and_real_tokenizer_without_weights(monkeypatch, tmp_path):
    import json
    from flex_web_d910b import diagnose
    (tmp_path / 'config.json').write_text(json.dumps({
        'architectures': ['Qwen3OmniMoeForConditionalGeneration'], 'model_type': 'qwen3_omni_moe',
        'torch_dtype': 'bfloat16', 'quantization_config': {'quant_method': 'example'},
    }), encoding='utf-8')
    (tmp_path / 'model-00001-of-00002.safetensors').write_bytes(b'123')
    (tmp_path / 'model-00002-of-00002.safetensors').write_bytes(b'4567')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        'first': 'model-00001-of-00002.safetensors', 'second': 'model-00002-of-00002.safetensors'
    }}), encoding='utf-8')
    calls = []
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [diagnose.EXPECTED_IDS[diagnose.TAGS.index(text)]]
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, local_files_only):
            calls.append((path, local_files_only))
            return Tokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    report = diagnose.checkpoint_report(tmp_path)
    assert report['config']['torch_dtype'] == 'bfloat16'
    assert report['config']['quant_method'] == 'example'
    assert report['weights']['total_bytes'] == 7
    assert report['weights']['indexed_shards'] == 2
    assert report['weights']['missing_indexed_shards'] == 0
    assert report['protocol_ids_match_training'] is True
    assert calls == [(str(tmp_path), True)]


def test_diagnose_reports_missing_dependencies_individually(monkeypatch):
    from flex_web_d910b import diagnose
    def missing(name):
        raise ModuleNotFoundError('secret path should not appear')
    monkeypatch.setattr(diagnose.importlib, 'import_module', missing)
    result = diagnose.environment_report()
    assert result['vllm_device_type'] == {'error': 'ModuleNotFoundError'}
    assert result['input_preprocessor_parameters'] == {'error': 'ModuleNotFoundError'}
    assert set(result['packages']) == set(diagnose.PACKAGES)
    assert 'secret path' not in repr(result)


def test_diagnose_does_not_double_count_checkpoint_formats(monkeypatch, tmp_path):
    from flex_web_d910b import diagnose
    (tmp_path / 'model.safetensors').write_bytes(b'12')
    (tmp_path / 'pytorch_model.bin').write_bytes(b'12345')
    monkeypatch.setitem(sys.modules, 'transformers', None)
    report = diagnose.checkpoint_report(tmp_path)
    assert report['weights']['format'] == 'safetensors'
    assert report['weights']['total_bytes'] == 2
    assert report['config'] == {'error': 'FileNotFoundError'}
    assert report['protocol_tokens'] == {'error': 'ModuleNotFoundError'}
