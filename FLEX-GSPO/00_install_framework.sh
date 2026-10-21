#!/usr/bin/env bash
# Explicit helper; never automatically run by preparation or training.
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
python3 "${SCRIPT_DIR}/check_environment.py" --require-training
