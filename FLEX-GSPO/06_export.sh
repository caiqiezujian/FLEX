#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON_BIN" export_adapter.py "$@"
