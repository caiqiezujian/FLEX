#!/usr/bin/env bash
# Opt-in QALCO v4: separate quality/gated-delay scaling, no second std division.
# Uses the accepted G8/B32/micro2, four-GPU engine without raising memory budgets.
# Usage: bash 06_train_qalco_v4.sh /model /prepared/train.jsonl /new/output [options]
# Defaults: 3000 updates, save every 500. Trailing options override defaults.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/04_train_throughput.sh" \
  --reward-config "${SCRIPT_DIR}/configs/qalco_v4.json" \
  "$@"
