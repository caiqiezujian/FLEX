#!/usr/bin/env bash
# ==============================================================================
# 【02｜通用训练入口】直接调用 train_megatron.py；默认奖励是旧 v2，不是 v4。
# 【训练损失】Megatron-SWIFT 的 GRPO 入口 + 序列级重要性采样/裁剪（GSPO 配置）。
#   rlhf_type=grpo、loss_type=grpo、importance_sampling_level=sequence；
#   默认 epsilon=0.0003、epsilon_high=0.0004、num_iterations=1。
#   KL 系数 beta=0.01 加入训练 loss；QALCO 是轨迹奖励，reward_weights=1。
# 【默认奖励】configs/default.json：COMET22 / QALCO v2，整条轨迹拼接后评分。
#   质量阈值 0.80，时延预算 3 秒；双达标分支的质量/时延/长度系数=1/0.2/0.1，
#   再除以系数和 1.3。质量不足和时延超预算各有独立负奖励分支，非统一加权和。
#   长度项只惩罚超过参考译文 1.2 倍的部分；空译/无效轨迹固定 -4。
#   默认 scale_rewards=group：总奖励减完整 G8 均值，再除组标准差。
# 【默认容量】四卡，G8/B16/micro1；3000 步，每 500 步保存；lr=1e-6。
#   LoRA rank/alpha=8/32；vLLM 显存比例=0.40，max_num_seqs=16。
# 【选择版本】上层 03/04/05/06 会覆盖部分默认参数；尤其 05 选 v3，06 选 v4。
#   --reward-config 优先于这里的默认说明，实际版本/权重见启动时的 reward_settings。
# 【用法】bash 02_train.sh /sft-merged /prepared/train.jsonl /new-run [参数]
#   --dry-run 只打印计划，不调用服务、不加载 GPU 模型、不写训练文件。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/train_megatron.py" "$@"
