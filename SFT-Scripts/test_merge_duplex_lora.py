"""CPU-only export validation tests; no real Qwen weights or GPU merge."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_duplex_lora as merge


class TinyTokenizer:
    unk_token_id = 0

    def __init__(self, added=False):
        self.added = added

    def __len__(self):
        return 7 if self.added else 4

    @property
    def all_special_tokens(self):
        return merge.PROTOCOL if self.added else []

    def add_special_tokens(self, value):
        assert value == {"additional_special_tokens": merge.PROTOCOL}
        self.added = True

    def convert_tokens_to_ids(self, value):
        return merge.PROTOCOL.index(value) + 4 if self.added else 0

    def encode(self, value, add_special_tokens=False):
        return [self.convert_tokens_to_ids(value)]


@pytest.fixture
def model_files(tmp_path, monkeypatch):
    base, adapter, output = [tmp_path / name for name in ("base", "adapter", "output")]
    for directory in (base, adapter, output):
        directory.mkdir()
    config = {"model_type": "qwen3_omni_moe", "thinker_config": {"text_config": {"vocab_size": 128}}}
    for directory in (base, output):
        (directory / "config.json").write_text(json.dumps(config))
    weights = {key: torch.zeros((128, 4), dtype=torch.bfloat16) for key in merge.VOCAB_KEYS}
    weights["thinker.audio_tower.test.weight"] = torch.zeros((2, 2), dtype=torch.bfloat16)
    weights["thinker.model.layers.0.test.weight"] = torch.zeros((2, 2), dtype=torch.bfloat16)
    save_file(weights, str(base / "model.safetensors"))
    saved = {"base_model.model." + key: torch.ones((128, 4), dtype=torch.bfloat16) for key in merge.VOCAB_KEYS}
    saved["base_model.model.thinker.model.layers.0.test.lora_A.weight"] = torch.ones((1, 2))
    saved["base_model.model.thinker.model.layers.0.test.lora_B.weight"] = torch.ones((2, 1))
    save_file(saved, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (adapter / "args.json").write_text(json.dumps({"new_special_tokens": merge.PROTOCOL, "model": "/old/base"}))
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("\n".join(merge.PROTOCOL))
    exported = {key: value.clone() for key, value in weights.items()}
    for key in merge.VOCAB_KEYS:
        exported[key] = torch.ones((128, 4), dtype=torch.bfloat16)
    save_file(exported, str(output / "model.safetensors"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda path, **kw: TinyTokenizer(added=Path(path) == output))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        lambda path, **kw: SimpleNamespace(tokenizer=TinyTokenizer(added=True)))
    return base, adapter, output, token_file, exported


def test_preflight_preserves_padded_vocabulary_and_output_protocol_rows(model_files):
    base, adapter, output, tokens, _ = model_files
    expected = merge.preflight(base, adapter, tokens)
    assert expected["tokenizer_length"] == 7
    assert expected["expected_vocab_size"] == 128  # Never shrink to tokenizer length.
    result = merge.verify_export(output, expected)
    assert result["processor_load_verified"]
    assert result["trained_protocol_rows_preserved_in"] == list(merge.VOCAB_KEYS)
    assert result["inference_verified"] is False


def test_protocol_rows_must_come_from_adapter_not_base(model_files):
    base, adapter, output, tokens, exported = model_files
    expected = merge.preflight(base, adapter, tokens)
    exported[merge.VOCAB_KEYS[0]][4] = 0
    save_file(exported, str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="row was not preserved"):
        merge.verify_export(output, expected)


def test_no_loss_of_audio_tower_during_export(model_files):
    base, adapter, output, tokens, exported = model_files
    expected = merge.preflight(base, adapter, tokens)
    del exported["thinker.audio_tower.test.weight"]
    save_file(exported, str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="Thinker tensor"):
        merge.verify_export(output, expected)


def test_mismatched_training_token_order_is_rejected(model_files):
    base, adapter, _, tokens, _ = model_files
    (adapter / "args.json").write_text(json.dumps({"new_special_tokens": list(reversed(merge.PROTOCOL))}))
    with pytest.raises(ValueError, match="new_special_tokens"):
        merge.preflight(base, adapter, tokens)


def test_missing_full_embedding_module_is_rejected(model_files):
    base, adapter, _, tokens, _ = model_files
    save_file({"lora_A.weight": torch.zeros(2), "lora_B.weight": torch.zeros(2)},
              str(adapter / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="saved full module"):
        merge.preflight(base, adapter, tokens)


def test_changed_vocab_config_is_rejected(model_files):
    base, adapter, output, tokens, _ = model_files
    expected = merge.preflight(base, adapter, tokens)
    config = merge.read_json(output / "config.json")
    config["thinker_config"]["text_config"]["vocab_size"] = 7
    (output / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="vocabulary size"):
        merge.verify_export(output, expected)


def test_missing_index_shard_is_rejected(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"key": "model-1.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="Missing"):
        merge.weight_inventory(tmp_path, "model")


def test_real_sharded_headers_and_index_checked(tmp_path):
    save_file({"a": torch.zeros(2)}, str(tmp_path / "model-1.safetensors"))
    save_file({"b": torch.zeros(3)}, str(tmp_path / "model-2.safetensors"))
    index = {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert merge.weight_inventory(tmp_path, "model")["b"]["shape"] == [3]
    index["weight_map"]["missing"] = "model-1.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="missing tensors"):
        merge.weight_inventory(tmp_path, "model")


def test_command_does_not_restore_old_plugin_or_use_mcore_export(tmp_path):
    args = SimpleNamespace(base=tmp_path / "base", adapter=tmp_path / "adapter",
                           special_tokens=tmp_path / "tokens.txt", device_map="auto")
    cmd = merge.export_command(args, tmp_path / "staging")
    assert cmd[cmd.index("--load_args") + 1] == "false"
    assert cmd[cmd.index("--template") + 1] == "qwen3_omni"
    assert "--to_hf" not in cmd and "--mcore_model" not in cmd
    assert cmd[cmd.index("--new_special_tokens") + 1] == str(args.special_tokens)
