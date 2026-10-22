#!/usr/bin/env bash
# Formal G8/B32 run, using the same trainer as 02_train.sh.
# Usage: bash 03_train_3000.sh /model /prepared/train.jsonl /new/output [options]
# Activate the accepted training environment before calling this script.
# Extra options at the end may override this recipe; --dry-run is supported.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Debug-only settings from the two-step investigation must not slow long runs.
# Do not change the tested allocator, offload or distributed transport settings.
unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES TORCH_DISTRIBUTED_DEBUG
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec bash "${SCRIPT_DIR}/02_train.sh" \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 1 \
  --steps-per-generation 1 \
  --train-iters 3000 \
  --save-steps 500 \
  --lr 1e-6 \
  --kl-beta 0.01 \
  --temperature 1.0 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --max-length 8192 \
  --max-new-tokens 128 \
  --max-turns 65 \
  --vllm-memory 0.4 \
  --vllm-max-num-seqs 16 \
  --vllm-mm-processor-cache-gb 0 \
  --performance-log on \
  --dataset-num-proc 4 \
  --rollout-log-every 5 \
  "$@"
