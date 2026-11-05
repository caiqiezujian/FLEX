#!/usr/bin/env bash
# ==============================================================================
# 【07｜MetricX QALCO v5】与 06 的 COMET22 v4 并列，显式选择此入口才启用 v5。
# 【训练损失】Megatron-SWIFT 在线 GSPO：序列级重要性采样/裁剪，loss_type=grpo。
#   epsilon=0.0003、epsilon_high=0.0004、num_iterations=1；KL beta=0.01 在 loss 中。
# 【质量服务】MetricX-24 reference：srcs + mts + refs，错误分 e 越低越好；Q=-e。
#   读取 configs/qalco_v5.json 的地址；不调用 COMET22，不在启动时自动评分预检。
# 【轨迹奖励】R = 1.0*zQ - 0.5*zD - 0.08*g*C - 0.20*P
#   质量权重 1.0；时延权重 0.5，使用原 MetricX v5 的系数配置。
#   g=1[e<=5]；D=min(d,10) 当质量过线，否则 D=10；d 为音频时钟 mean_lag。
#   zQ/zD 在完整 G8 的有效候选中分别归一化，标准差下限均为 0.5，裁剪 [-2,2]。
#   0.5 不是每秒固定扣分，也不代表实际贡献固定占50%；3秒预算只用于诊断。
#   全组低质量时 zD=0；加权取舍不保证任意微小质量优势都绝对优先。
# 【简洁性】e<=min(5,组内最小e+0.75) 且至少 3 条才建立长度基准，取 25% 分位数；
#   超出 max(2单位,基准5%) 才计 C，乘质量门控；P 沿用局部机械循环检查。
# 【WAIT/失败】中途 WAIT 无逐次固定罚分；全程空译/格式错误/严重循环进入失败分支。
#   失败奖励=最低有效奖励-0.25；全组失败为-0.25，任务奖励优势为0。
# 【优势】scale_rewards=none：分项归一化后仍减完整组均值，不再除总奖励标准差。
# 【默认训练】四卡 G8/B32/micro2、3000步、每500步保存、lr=1e-6、LoRA=8/32。
#   TP2/EP2，vLLM TP4，显存比例 .50、max_num_seqs=32、MM cache=0；最多65轮含EOF。
#   共用当前数据准备、最新 SFT 提示词、同步/offload/监控逻辑；未恢复临时诊断补丁。
# 【八卡】追加 --gpu-ids 0,1,2,3,4,5,6,7 --global-batch-size 64，仍为 G8/micro2。
# 【用法】bash 07_train_qalco_v5.sh /sft-merged /prepared/train.jsonl /new-run [参数]
#   尾部可覆盖默认参数；--dry-run 只打印生效配置，不加载模型或请求评分服务。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/04_train_throughput.sh" \
  --reward-config "${SCRIPT_DIR}/configs/qalco_v5.json" \
  "$@"
