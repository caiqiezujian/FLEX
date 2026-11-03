#!/usr/bin/env bash
# ==============================================================================
# 【01 旧格式｜数据准备】固定读取 lang/key/en.txt/zh.txt + 音频目录中的 key.wav。
# 【损失/奖励】不训练、不计算 loss/奖励；“旧格式”只指输入字段，不指旧奖励版本。
# 【提示词】使用当前最新 SFT 短提示词，不恢复历史旧提示词。
# 【输出】与 01_prepare.sh 共用处理核心，生成相同训练结构的 train.jsonl。
#   默认 1 秒虚拟分块，不复制/重切音频；进度、去重、错误过滤功能共用。
# 【用法】bash 01_prepare_legacy.sh --input-jsonl /raw.jsonl --audio-dir /wavs \
#          --output-dir /prepared --workers 16 --skip-invalid
# 【下一步】可直接使用 06_train_qalco_v4.sh，旧格式数据同样能训练 COMET22 v4。
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
exec python3 "${SCRIPT_DIR}/prepare_legacy_data.py" "$@"
