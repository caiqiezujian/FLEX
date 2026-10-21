#!/usr/bin/env bash
# Build task metadata and virtual audio-chunk references, not offline trajectories.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec python3 "${SCRIPT_DIR}/prepare_data.py" "$@"
