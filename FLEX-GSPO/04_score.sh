#!/usr/bin/env bash
# Continue after FLEX-GSPO-Data's 01/02/03; does not resample or train.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_FILE="${CONFIG_FILE:-configs/default.json}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON_BIN" score_candidates.py --config "$CONFIG_FILE" "$@"
