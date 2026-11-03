"""Read-only environment and checkpoint diagnostics; never load model weights."""
import argparse
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path


TAGS = ('<|WAIT|>', '<|DIRECT|>', '<|FLEX|>')
EXPECTED_IDS = (151676, 151677, 151678)
PACKAGES = ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'vllm-omni', 'transformers')


def safe_error(exc):
    # Paths, tokens and server addresses can be embedded in exception messages.
    return type(exc).__name__


def environment_report():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:
            versions[name] = {'error': safe_error(exc)}
    report = {'packages': versions}
    try:
        platform = importlib.import_module('vllm.platforms').current_platform
        report['vllm_device_type'] = getattr(platform, 'device_type', None)
    except Exception as exc:
        report['vllm_device_type'] = {'error': safe_error(exc)}
    try:
        cls = importlib.import_module('vllm.inputs.preprocess').InputPreprocessor
        report['input_preprocessor_parameters'] = list(inspect.signature(cls).parameters)
    except Exception as exc:
        report['input_preprocessor_parameters'] = {'error': safe_error(exc)}
    return report


def checkpoint_report(directory):
    path = Path(directory)
    if not path.is_dir():
        return {'error': 'checkpoint directory does not exist'}
    report = {}
    try:
        config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        quant = config.get('quantization_config')
        text_config = config.get('thinker_config') or {}
        report['config'] = {
            'architectures': config.get('architectures'),
            'model_type': config.get('model_type'),
            'torch_dtype': config.get('torch_dtype', config.get('dtype')),
            'thinker_dtype': text_config.get('torch_dtype', text_config.get('dtype')) if isinstance(text_config, dict) else None,
            'quant_method': quant.get('quant_method') if isinstance(quant, dict) else None,
        }
    except Exception as exc:
        report['config'] = {'error': safe_error(exc)}
    try:
        # Safetensors are authoritative when both formats are present; avoid
        # double-counting alternative checkpoint formats or non-weight files.
        weights = sorted(path.glob('*.safetensors'))
        weight_format = 'safetensors'
        if not weights:
            weights = sorted(path.glob('pytorch_model*.bin'))
            weight_format = 'pytorch_bin'
        report['weights'] = {'format': weight_format if weights else None,
                             'files': len(weights),
                             'total_bytes': sum(file.stat().st_size for file in weights)}
        index_path = path / ('model.safetensors.index.json' if weight_format == 'safetensors'
                             else 'pytorch_model.bin.index.json')
        report['weights']['index_present'] = index_path.is_file()
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding='utf-8'))
            entries = index.get('weight_map', {})
            if not isinstance(entries, dict):
                raise ValueError('invalid weight map')
            names = set(entries.values())
            if any(not isinstance(name, str) or Path(name).name != name for name in names):
                raise ValueError('invalid shard name')
            report['weights']['indexed_shards'] = len(names)
            report['weights']['missing_indexed_shards'] = sum(not (path / name).is_file() for name in names)
    except Exception as exc:
        report['weights'] = {'error': safe_error(exc)}
    try:
        tokenizer = importlib.import_module('transformers').AutoTokenizer.from_pretrained(
            str(path), local_files_only=True)
        ids = [tokenizer.encode(tag, add_special_tokens=False) for tag in TAGS]
        report['protocol_tokens'] = dict(zip(TAGS, ids))
        report['protocol_ids_match_training'] = ids == [[i] for i in EXPECTED_IDS]
    except Exception as exc:
        report['protocol_tokens'] = {'error': safe_error(exc)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Local checkpoint directory (optional)')
    args = parser.parse_args()
    result = {'environment': environment_report()}
    if args.checkpoint:
        result['checkpoint'] = checkpoint_report(args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
