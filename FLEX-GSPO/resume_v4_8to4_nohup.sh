#!/usr/bin/env bash
# Restore a COMPLETE native V4 checkpoint, retaining original SFT/COMET22/batch/LR.
# Run inside the same Docker with the original training environment available.
# Edit the marked paths and choose FOUR GPUs that are actually free.
set -euo pipefail

cd /data/yb/Code/FLEX/FLEX-GSPO

# Replace if your existing environment is under /data/yb/vens instead.
source /data/yb/venvs/flex-gspo-online/bin/activate

# The migration helper automatically reads original training settings from launch JSON.
# It finds matching historical Git code and creates a SEPARATE sibling worktree.
# No checkout/reset of the current repository; no edits to old checkpoints/logs.
unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES TORCH_DISTRIBUTED_DEBUG
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS FLEX_DEBUG_OFFLOAD
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
mkdir -p /data/yb/Code/FLEX/outputs/logs

# REQUIRED: replace launch-json and resume-from with files from the SAME old run.
# --launch-json: original output root/launch_*.json, NOT checkpoint args.json.
# --resume-from: .../v0-TIMESTAMP/checkpoint-500, NEVER checkpoint-500-merged.
# --output-dir: a NEW empty directory (also change the log/PID names for another retry).
# --gpu-ids: FOUR available physical GPU IDs; e.g. 0,2,4,6, not necessarily 0..3.
# Keep the original global batch (normally64), micro2, total3000 and save500:
# these are reconstructed from the saved launch; do not manually reduce batch to32.
# --master-port: choose another free port if29583 is used.
nohup python3 -u Scripts/resume_dp_migration.py \
  --launch-json "/替换为原八卡训练输出根目录/launch_实际时间.json" \
  --resume-from "/替换为原八卡训练输出根目录/v0-实际时间/checkpoint-500" \
  --output-dir /data/yb/Code/FLEX/outputs/gspo_v4_resume500_4gpu_g8_b64_001 \
  --gpu-ids 0,1,2,3 \
  --master-port 29583 \
  > /data/yb/Code/FLEX/outputs/logs/gspo_v4_resume500_4gpu_g8_b64_001.log 2>&1 < /dev/null &

training_pid=$!
printf '%s\n' "$training_pid" > /data/yb/Code/FLEX/outputs/logs/gspo_v4_resume500_4gpu_g8_b64_001.pid
printf 'Started launcher PID=%s. This confirms background launch, not checkpoint-load success.\n' "$training_pid"
printf 'Log: /data/yb/Code/FLEX/outputs/logs/gspo_v4_resume500_4gpu_g8_b64_001.log\n'
