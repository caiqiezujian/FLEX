#!/usr/bin/env bash
# Opt-in throughput candidate. The accepted 03_train_4100.sh stays unchanged.
# Usage: bash 04_train_throughput.sh /model /prepared/train.jsonl /new/output [options]
# Start with --train-iters 5 --save-steps 5 in a NEW directory before a long run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Same optimizer batch, G8, objective and update frequency as the B32 baseline.
# Only raise actual concurrency and the vLLM memory budget. Other execution
# choices (TP/EP, eager, full recompute, sleep/offload, LoRA sync) stay unchanged.
exec bash "${SCRIPT_DIR}/03_train_4100.sh" \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --vllm-max-num-seqs 32 \
  --vllm-memory 0.50 \
  --performance-log on \
  "$@"
