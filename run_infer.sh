#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash run_infer.sh MODEL_DIR AUDIO_FILE [extra CLI args]' \
    '' \
    'Run local-file streaming translation without a server or browser.' \
    'MODEL_DIR, AUDIO_FILE and --output-dir are relative to your current directory.' \
    'Defaults: en-zh, 1-second chunks, fast replay, special protocol tokens.' \
    '' \
    'Example:' \
    '  CUDA_VISIBLE_DEVICES=6 bash run_infer.sh /path/to/checkpoint /path/to/audio.wav' \
    '  bash run_infer.sh ./model ./audio.wav --direction zh-en --output-dir ./result' \
    '' \
    'Extra arguments override defaults. Set PYTHON to choose a Python executable.' \
    'For all Python options: python infer_audio.py --help'
}

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

# Resolve only the script location; do not change the caller's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint="$1"
audio_file="$2"
shift 2
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "${PYTHON:-python}" -u "$script_dir/infer_audio.py" \
  --model "$checkpoint" --audio "$audio_file" \
  --backend vllm --direction en-zh --chunk-seconds 1 --pace fast \
  --protocol-form special --image-slots 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 "$@"
