#!/usr/bin/env bash
# Opt-in QALCO v3 experiment; preserve the accepted G8/B32 four-A800 engine.
# Usage: bash 05_train_qalco_v3.sh /model /prepared/train.jsonl /new/output [options]
# 3000 optimizer updates, checkpoint every 500; user options override defaults.
# v2 default.json is deliberately unchanged for historical comparisons.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/04_train_throughput.sh" \
  --reward-config "${SCRIPT_DIR}/configs/qalco_v3.json" \
  "$@"
