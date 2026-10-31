#!/usr/bin/env bash
# Run inside the existing SFT container with its python3 environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -u "$SCRIPT_DIR/merge_duplex_lora.py" "$@"
