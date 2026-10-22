#!/usr/bin/env bash
# G8/B32 micro2 throughput recipe, based on 03_train_3000.sh.
# Usage: bash 04_train_throughput.sh /model /prepared/train.jsonl /new/output [options]
# An isolated short smoke may override steps/save, but cannot be resumed as a
# 3000-step run. Exact resume requires an unchanged total LR horizon and code.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Same optimizer batch, G8, objective and update frequency as the B32 baseline.
# Only raise actual concurrency and the vLLM memory budget. Other execution
# choices (TP/EP, eager, full recompute, sleep/offload, LoRA sync) stay unchanged.
exec bash "${SCRIPT_DIR}/03_train_3000.sh" \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --vllm-max-num-seqs 32 \
  --vllm-memory 0.50 \
  --performance-log on \
  "$@"
