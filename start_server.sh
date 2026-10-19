#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Usage: CUDA_VISIBLE_DEVICES=0 bash start_server.sh /absolute/merged-checkpoint
# Extra arguments override the defaults, e.g. --max-model-len 4096.
# Intranet microphone: add --host 0.0.0.0 --port 18889 --https-host YOUR_SERVER_IP.
# TLS certificates are generated once on this server and reused on later starts.
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash start_server.sh /absolute/merged-checkpoint [extra CLI args]\n' >&2
  exit 2
fi
checkpoint="$1"
shift
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec python run.py --backend vllm --model "$checkpoint" \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --chunk-seconds 1 --max-chunk-seconds 3 --max-pending-seconds 12 "$@"
