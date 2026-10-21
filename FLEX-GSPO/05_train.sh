#!/usr/bin/env bash
# Online GSPO after external preparation/sampling and reward calibration.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
CONFIG_FILE="${CONFIG_FILE:-configs/default.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CPU preflight is single-process and never loads actor or reward weights.
for arg in "$@"; do
    if [[ "$arg" == "--validate-only" ]]; then
        export FLEX_PREFLIGHT_WORLD_SIZE="$NPROC_PER_NODE"
        exec "$PYTHON_BIN" train.py --config "$CONFIG_FILE" "$@"
    fi
done
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" train.py --config "$CONFIG_FILE" "$@"
