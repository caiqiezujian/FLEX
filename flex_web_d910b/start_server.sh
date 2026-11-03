#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'Usage: ASCEND_RT_VISIBLE_DEVICES=0 FLEX_HTTPS_HOST=SERVER_IP bash flex_web_d910b/start_server.sh /path/to/merged-checkpoint [extra options]' >&2
  exit 2
fi
checkpoint="$1"
shift
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
tls_args=()
if [[ -n "${FLEX_HTTPS_HOST:-}" ]]; then
  tls_args+=(--https-host "$FLEX_HTTPS_HOST")
else
  echo '[INFO] HTTP mode: set FLEX_HTTPS_HOST to the browser-facing IP for HTTPS and microphone capture.' >&2
fi
exec "${PYTHON:-python3}" -m flex_web_d910b.server \
  --backend vllm --model "$checkpoint" --host 0.0.0.0 --port 18889 \
  --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --chunk-seconds 1 --max-chunk-seconds 3 --max-pending-seconds 12 \
  --max-sessions 2 --max-batch-size 1 \
  --max-audio-turns 32 \
  --history-window-seconds 20 --history-retain-seconds 10 \
  "${tls_args[@]}" "$@"
