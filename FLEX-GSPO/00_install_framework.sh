#!/usr/bin/env bash
# ==============================================================================
# 【00｜环境安装】安装本项目审查过的框架版本；不是数据处理或训练入口。
# 【损失/奖励】不训练模型，不计算 loss 或奖励，不设置质量/时延/长度权重。
# 【安装内容】requirements.txt、固定提交的 ms-swift、mcore-bridge==1.6.3。
# 【环境要求】在已有 CUDA/PyTorch/vLLM 的独立训练环境内显式执行。
#   此脚本不安装/升级 torch、vLLM、Megatron-Core、TransformerEngine 或 CUDA；
#   ms-swift/mcore-bridge 使用 --no-deps 安装，末尾检查依赖和训练能力。
# 【网络】本入口需要访问 pip 和 GitHub；本地安装包的离线步骤见 docx 完整指南。
# 【用法】bash 00_install_framework.sh --confirm-isolated-env
#   准备数据和启动训练不会自动执行此安装脚本。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" != "--confirm-isolated-env" ]]; then
  echo 'Use a separate environment/container with working CUDA/PyTorch/vLLM/TransformerEngine.' >&2
  echo 'Then explicitly run: bash 00_install_framework.sh --confirm-isolated-env' >&2
  echo 'This does NOT install/upgrade torch, vLLM, Megatron-Core, TransformerEngine or CUDA.' >&2
  exit 2
fi
python3 - <<'PY'
import os, sys
if sys.prefix == sys.base_prefix and not os.environ.get('CONDA_PREFIX'):
    raise SystemExit('Refusing base Python. Activate a separate virtualenv/conda environment.')
if os.environ.get('CONDA_DEFAULT_ENV') == 'base':
    raise SystemExit('Refusing conda base. Activate a separate environment.')
if sys.version_info < (3, 10):
    raise SystemExit('The audited framework requires Python >= 3.10.')
print('Explicit install interpreter:', sys.executable)
PY
python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"
python3 -m pip install --no-deps 'ms-swift @ git+https://github.com/modelscope/ms-swift.git@f15bb91439d756d4a366b51956d72d1f088b7af6'
python3 -m pip install --no-deps 'mcore-bridge==1.6.3'
echo 'Installed audited source with --no-deps, preserving the CUDA stack.'
echo 'Requires compatible Megatron-Core >= 0.16 / TransformerEngine / FlashAttention.'
echo 'Resolve dependency errors in this isolated environment; do not modify the accepted SFT environment.'
python3 -m pip check
python3 "${SCRIPT_DIR}/Scripts/check_environment.py" --require-training
