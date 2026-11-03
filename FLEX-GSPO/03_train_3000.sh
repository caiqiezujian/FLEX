#!/usr/bin/env bash
# ==============================================================================
# 【03｜3000 步基线预设】四卡 G8/B32/micro1；作为 04/05/06 的共享训练基础。
# 【训练损失】Megatron-SWIFT 在线 GSPO：序列级重要性采样，loss_type=grpo；
#   epsilon=0.0003、epsilon_high=0.0004；KL beta=0.01，reward_weights=1。
#   KL 属于训练 loss；下面的质量/时延/长度系数属于 QALCO 轨迹奖励。
# 【单独调用的奖励】默认 configs/default.json，即 COMET22 / QALCO v2。
#   质量阈值 0.80、时延预算 3 秒；双达标分支的质量/时延/长度系数=1/0.2/0.1，
#   除以 1.3；不达标使用独立负奖励分段。空译/无效轨迹固定 -4。
#   长度惩罚从参考译文 1.2 倍开始；合成奖励后做完整 G8 组标准化。
# 【训练参数】3000 步，每 500 步保存；lr=1e-6，LoRA rank/alpha=8/32。
#   vLLM 显存比例=0.40，max_num_seqs=16；最多 65 轮（含 EOF），每轮最多 128 token。
#   关闭多模态处理器缓存，保留已有同步/offload；清除调试环境变量以免拖慢训练。
# 【注意】03 本身不锁定奖励版本；由 05/06 调用时采用上层传入的 v3/v4 配置。
# 【用法】bash 03_train_3000.sh /sft-merged /prepared/train.jsonl /new-run [参数]
#   先激活 Docker 内训练环境；尾部参数可覆盖默认值，支持 --dry-run。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 正式训练清除两步调试阶段的同步/详细日志开关；分布式和 offload 配置保持共用。
unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES TORCH_DISTRIBUTED_DEBUG
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec bash "${SCRIPT_DIR}/02_train.sh" \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 1 \
  --steps-per-generation 1 \
  --train-iters 3000 \
  --save-steps 500 \
  --lr 1e-6 \
  --kl-beta 0.01 \
  --temperature 1.0 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --max-length 8192 \
  --max-new-tokens 128 \
  --max-turns 65 \
  --vllm-memory 0.4 \
  --vllm-max-num-seqs 16 \
  --vllm-mm-processor-cache-gb 0 \
  --performance-log on \
  --dataset-num-proc 4 \
  --rollout-log-every 5 \
  "$@"
