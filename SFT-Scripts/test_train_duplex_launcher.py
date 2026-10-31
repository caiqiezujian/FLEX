"""Exercise shell argument/env construction with a fake Python; no GPU work."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).with_name("train_duplex_sft_virtual.sh")
BASH = "D:/Git/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


def run_launcher(tmp_path, extra):
    if not BASH or not Path(BASH).is_file():
        pytest.skip("Bash is unavailable")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python3"
    python.write_bytes(('#!/usr/bin/env bash\nprintf "CALL\\n"\nprintf "ARG:%s\\n" "$@"\n'
                      'printf "ENV:%s,%s,%s,%s,%s\\n" "${FLEX_SFT_ATTENTION_COMPAT-unset}" '
                      '"${NVTE_FLASH_ATTN-unset}" "${NVTE_FUSED_ATTN-unset}" '
                      '"${NVTE_UNFUSED_ATTN-unset}" "${NVTE_MASKED_SOFTMAX_FUSION-unset}"\n').encode("utf-8"))
    python.chmod(0o755)
    model = tmp_path / "model with spaces"
    model.mkdir()
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")
    env = dict(os.environ)
    for key in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN", "NVTE_MASKED_SOFTMAX_FUSION",
                "FLEX_SFT_ATTENTION_COMPAT", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    prefix = 'mock_bin="$1"; shift; '
    if os.name == "nt":
        prefix += 'mock_bin="$(cygpath -u "$mock_bin")"; '
    return subprocess.run([BASH, "-c", prefix + 'export PATH="$mock_bin:$PATH"; exec bash "$@"',
                           "test", str(fake_bin), str(SCRIPT), str(model), str(dataset), str(tmp_path / "output"),
                           "--gpu-ids", "0,1,2,3", *extra], env=env, capture_output=True, text=True, timeout=30)


def test_default_launcher_preserves_existing_fast_recipe(tmp_path):
    run = run_launcher(tmp_path, [])
    assert run.returncode == 0, run.stderr
    assert "ARG:--attention_backend\nARG:flash" in run.stdout
    assert "ARG:--packing\nARG:true" in run.stdout
    assert "ARG:--micro_batch_size\nARG:4" in run.stdout
    assert "ARG:--padding_free" not in run.stdout
    assert "ENV:unset,unset,unset,unset,unset" in run.stdout


def test_compat_launcher_routes_both_attention_branches_and_preserves_global_batch(tmp_path):
    run = run_launcher(tmp_path, ["--compat-old-driver"])
    assert run.returncode == 0, run.stderr
    for flag, value in (("attn_impl", "sdpa"), ("attention_backend", "unfused"), ("packing", "false"),
                        ("padding_free", "false"), ("micro_batch_size", "1"), ("global_batch_size", "16")):
        assert f"ARG:--{flag}\nARG:{value}" in run.stdout
    assert "ENV:1,0,0,1,0" in run.stdout
    assert "ARG:--merge_lora\nARG:false" in run.stdout


@pytest.mark.parametrize("override", [["--packing", "true"], ["--padding_free=true"],
                                      ["--attention_backend", "auto"], ["--attn_impl", "flash_attn"]])
def test_compat_rejects_conflicting_attention_and_packing_overrides(tmp_path, override):
    run = run_launcher(tmp_path, ["--compat-old-driver", *override])
    assert run.returncode == 2
    assert "controls" in run.stderr and "CALL" not in run.stdout
