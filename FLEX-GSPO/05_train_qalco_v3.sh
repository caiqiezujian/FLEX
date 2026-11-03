#!/usr/bin/env bash
# ==============================================================================
# 【05｜旧 QALCO v3】COMET22 + 原始质量门控奖励 + 空译固定 -4。
# 【配置】configs/qalco_v3.json；它不是“分项归一化”的 v4，v4 请用 06。
# 【训练损失】Megatron-SWIFT 在线 GSPO（序列级重要性采样/裁剪，loss_type=grpo）。
#   默认 epsilon=0.0003、epsilon_high=0.0004；KL beta=0.01 加入 loss；reward_weights=1。
# 【轨迹奖励】Q=COMET22 分数；d=音频时钟 mean_lag；C=组内长度超额；P=重复程度。
#   R = Q - 2*max(0, 0.80-Q) - D_gated - 0.08*g*C - 0.20*P
#   g = clip((Q-0.75)/0.10, 0, 1)，即质量门控区间 0.75~0.85。
#   D_raw = min(2, 0.05*(d/3) + 0.20*max(0, d/3-1)^2)
#   D_gated = (1-g)*2 + g*D_raw；时延预算 3 秒，未额外乘 v4 的 0.25。
#   长度基准来自组内质量相近且达标候选；简洁性系数 0.08，重复系数 0.20。
# 【优势】scale_rewards=group：合成奖励后，减完整 G8 均值并除组标准差。
# 【WAIT】正常中途 WAIT 不逐次罚分；其影响通过后续输出时延体现。
#   全程空译/格式错误/严重循环固定 -4；整组同为 -4 时，任务奖励优势仍为零。
# 【训练参数】四卡 G8/B32/micro2，3000 步，每 500 步保存；lr=1e-6，LoRA=8/32。
#   vLLM 显存比例 .50、max_num_seqs=32、MM cache=0；共用 04 的吞吐预设。
# 【用法】bash 05_train_qalco_v3.sh /sft-merged /prepared/train.jsonl /new-run [参数]
#   尾部参数（含 --reward-config）可覆盖预设；--dry-run 只打印实际生效配置。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/04_train_throughput.sh" \
  --reward-config "${SCRIPT_DIR}/configs/qalco_v3.json" \
  "$@"
