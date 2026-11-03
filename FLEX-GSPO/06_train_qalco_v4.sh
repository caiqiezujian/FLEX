#!/usr/bin/env bash
# ==============================================================================
# 【06｜当前选用的 QALCO v4】COMET22 + 分项归一化 + 空译相对惩罚。
#   对应 2026-11-01 及之前 feat 分支中确认的奖励方案；配置为 configs/qalco_v4.json。
# 【训练损失】Megatron-SWIFT 在线 GSPO（序列级重要性采样/裁剪，loss_type=grpo）。
#   默认 epsilon=0.0003、epsilon_high=0.0004、num_iterations=1；reward_weights=1。
#   KL beta=0.01 加入训练 loss，不属于下面的 QALCO 轨迹奖励。
# 【轨迹奖励】R = 1.0*zQ - 0.25*zD - 0.08*g*C - 0.20*P
#   Q=COMET22 分数；d=音频时钟 mean_lag；C=组内长度超额；P=重复程度。
#   zQ/zD 分别是质量/门控时延在有效候选中的标准化值，均裁剪到 [-2, 2]；
#   标准差下限分别为 0.03 / 0.10，避免近乎同分时放大微小差异。
#   质量权重 1.0；时延权重 0.25；简洁性权重 0.08；重复权重 0.20。
# 【时延门控】g = clip((Q-0.80)/0.05, 0, 1)，门控区间 0.80~0.85，预算 3 秒。
#   D_raw = min(2, 0.05*(d/3) + 0.20*max(0, d/3-1)^2)
#   D_gated = (1-g)*2 + g*D_raw，再在组内计算 zD；0.25 不是每秒固定扣分。
# 【简洁性】质量 >= max(0.80, 组内最高质量-0.03) 且至少 3 条才建立长度基准；
#   基准为这些候选长度的 25% 分位数，宽限 max(2 单位, 基准的 5%)。
#   超出才计 C，且乘质量门控 g；无基准时 C=0，不直接奖励“越短越好”。
# 【优势】scale_rewards=none：最终仍减完整 G8 均值，不做第二次标准差除法。
# 【WAIT】中途 WAIT 不逐次固定扣分；全程空译/格式错误/严重循环属于失败候选。
#   失败奖励 = 本组最低有效候选奖励 - 0.25，不再固定给 -4；失败不参与 zQ/zD 统计。
#   全组都失败时每条为 -0.25，任务奖励优势为零（KL 等项仍可能产生更新）。
# 【训练参数】四卡 G8/B32/micro2，3000 步，每 500 步保存；lr=1e-6，LoRA=8/32。
#   vLLM 显存比例 .50、max_num_seqs=32、MM cache=0；最多 65 轮（含 EOF）。
# 【用法】bash 06_train_qalco_v4.sh /sft-merged /prepared/train.jsonl /new-run [参数]
#   旧/新格式准备的 train.jsonl 均可用；先激活 Docker 内训练 venv。
#   尾部参数（含 --reward-config）可覆盖预设；--dry-run 只打印实际生效配置。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 四卡训练参数显式列出；共用入口负责调试环境清理、Megatron/vLLM 与 TensorBoard。
# MM cache=0 保留此前缓存清理卡住问题的绕过设置。
exec bash "${SCRIPT_DIR}/04_train_throughput.sh" \
  --reward-config "${SCRIPT_DIR}/configs/qalco_v4.json" \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
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
  --vllm-memory 0.50 \
  --vllm-max-num-seqs 32 \
  --vllm-mm-processor-cache-gb 0 \
  --performance-log on \
  --dataset-num-proc 4 \
  --rollout-log-every 5 \
  "$@"
