#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash run_batch.sh MODEL_DIR AUDIO_DIR JSONL_DIR FIELD_PREFIX [extra CLI args]' \
    'Legacy: bash run_batch.sh MODEL_DIR AUDIO_DIR OUTPUT_JSONL [extra CLI args]' \
    '' \
    'Load one model; match audio stems to vid and update existing JSONL files in place.' \
    'Write FIELD_PREFIX_trans from the model and FIELD_PREFIX_asr from transcript.' \
    'Rows without audio stay untouched. No backup/result copy is created in in-place mode.' \
    'Defaults: en-zh, 1-second chunks, fast replay, special protocol tokens.' \
    'Paths are relative to your current working directory.' \
    '' \
    'Example:' \
    '  CUDA_VISIBLE_DEVICES=0 bash run_batch.sh /path/model /path/audios /path/jsonls Flex_1022' \
    '  bash run_batch.sh ./model ./audios ./jsonls Flex_1022 --check-only' \
    '  bash run_batch.sh ./model ./audios ./results.jsonl --direction zh-en' \
    '' \
    'Extra arguments override defaults. Interpreter: python3 (override with PYTHON).'
}

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
  usage
  exit 0
fi
if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint="$1"
audio_dir="$2"
destination="$3"
shift 3
if [[ $# -gt 0 && "$1" != -* ]]; then
  output_args=(--jsonl-dir "$destination" --field-prefix "$1")
  shift
else
  if [[ "${destination,,}" != *.jsonl ]]; then
    printf '%s\n' 'Missing FIELD_PREFIX. For in-place updates use MODEL AUDIO_DIR JSONL_DIR FIELD_PREFIX.' >&2
    exit 2
  fi
  output_args=(--output-jsonl "$destination")
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONIOENCODING=utf-8

exec "${PYTHON:-python3}" -u "$script_dir/infer_folder.py" \
  --model "$checkpoint" --audio-dir "$audio_dir" "${output_args[@]}" \
  --backend vllm --direction en-zh --chunk-seconds 1 --pace fast \
  --protocol-form special --image-slots 1 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 "$@"
