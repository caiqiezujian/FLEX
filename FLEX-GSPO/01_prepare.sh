#!/usr/bin/env bash
# ==============================================================================
# 【01｜通用数据准备】生成在线 GSPO 所需 train.jsonl 和音频时间分块计划。
# 【损失/奖励】不加载模型、不采样候选、不计算 loss/奖励；此处不决定 v2/v3/v4。
# 【新格式】--input-format segments：audio + srcs + chooses；直接使用 audio 路径。
# 【旧格式】--input-format legacy：lang/key/en.txt/zh.txt，另传 --audio-dir 找 key.wav。
#   默认 auto 自动识别格式；固定旧格式的入口为 01_prepare_legacy.sh。
# 【数据特点】共用最新 SFT 短提示词；默认 1 秒虚拟分块，不另存切分 WAV。
#   srcs/chooses 拼接为评分原文/参考译文；参考文本不作为模型翻译输入。
#   在线多轨迹由后续训练生成；准备阶段不生成离线轨迹，也不修改模型权重。
# 【用法】bash 01_prepare.sh --input-jsonl /raw.jsonl --input-format segments \
#          --output-dir /prepared --workers 16 --skip-invalid
# 【下一步】06_train_qalco_v4.sh /sft-merged /prepared/train.jsonl /new-run
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec python3 "${SCRIPT_DIR}/prepare_data.py" "$@"
