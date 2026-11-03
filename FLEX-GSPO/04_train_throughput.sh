#!/usr/bin/env bash
# ==============================================================================
# 【04｜吞吐预设】在 03 基础上提高并发；它本身不是新的奖励函数版本。
# 【训练损失】仍为 Megatron-SWIFT 在线 GSPO，序列级重要性采样，loss_type=grpo；
#   epsilon=0.0003、epsilon_high=0.0004；KL beta=0.01，reward_weights=1。
# 【单独调用的奖励】默认 COMET22 / QALCO v2（configs/default.json）。
#   质量阈值 0.80，时延预算 3 秒；双达标分支质量/时延/长度系数=1/0.2/0.1，
#   除以 1.3；不达标另走负奖励分支。长度超过参考的 1.2 倍才惩罚；空译固定 -4。
#   默认总奖励按 G8 组标准化；被 05/06 调用时改用上层指定的 v3/v4 奖励。
# 【相对 03 的变化】micro batch 1 -> 2，vLLM max_num_seqs 16 -> 32，显存比例 .40 -> .50。
# 【保持一致】四卡、G8、global batch 32、3000 步、每 500 步保存、lr=1e-6、
#   LoRA rank/alpha=8/32、同步/offload/重计算/缓存策略；开启阶段耗时日志。
# 【用法】bash 04_train_throughput.sh /sft-merged /prepared/train.jsonl /new-run [参数]
#   尾部参数可覆盖预设；支持 --dry-run。短验收用新输出目录，与正式训练分开。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 只调整并发/显存预留和性能日志；奖励版本由传入的 --reward-config 决定。
exec bash "${SCRIPT_DIR}/03_train_3000.sh" \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --vllm-max-num-seqs 32 \
  --vllm-memory 0.50 \
  --performance-log on \
  "$@"
