#!/usr/bin/env bash
# Legacy key/en.txt/zh.txt + key.wav input, using the current shared SFT prompt.
# Usage: bash 01_prepare_legacy.sh --input-jsonl /raw.jsonl --audio-dir /wavs --output-dir /prepared [options]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec python3 "${SCRIPT_DIR}/prepare_legacy_data.py" "$@"
