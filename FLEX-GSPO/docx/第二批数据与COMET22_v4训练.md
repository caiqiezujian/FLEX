# 第二批数据与 COMET22 v4 训练

当前服务器已使用 /data/yb/Code/FLEX 和 /data/yb/vens。完整的环境、准备、训练与 TensorBoard 1～9 步请使用 [新格式数据完整运行指南](GSPO_v4_新格式数据_完整运行指南.md)；本文后面的 data06 路径保留为历史示例，不作为当前复制执行命令。

更新日期：2026-11-02。本次适配第二批原始数据，并将 GSPO system prompt 与 `SFT-Scripts/prepare_duplex_data.py` 的最新短模板对齐。训练继续使用现有 **COMET22 QALCO v4 / MS-SWIFT Megatron 在线 GSPO**。

本次不改奖励公式、评分服务、四卡容量或采样参数：质量权重为 `1.0`，时延权重为 `0.25`，保留原连续质量门控、分项归一化、质量约束长度成本及重复/失败处理。无需新增运行依赖，也不需要部署新的质量模型。公式详见历史 [QALCO v4 训练说明](QALCO_v4训练说明.md)；该历史文档中的旧模型和旧 prepared 路径不能直接替代本文的新提示词流程。

## 1. 原始 JSONL 格式

每行一个 JSON 对象，例如：

```json
{"audio":"/data/xyh/omni_train_datas/download/en/8/audios/vpq-r5_I-hw#00-04-13_132.48_158.73.wav","srcs":["So, this culture must come to an end.","One culture that must end with this month of 2022","is police brutality in Kenya."],"chooses":["所以，这种文化必须终结。","有一种文化必须在2022年这个月结束，","那就是肯尼亚的警察暴力。"],"rejects":["其他译文"],"voice_end":false}
```

| 字段 | 处理方式 |
| --- | --- |
| `audio` | 直接引用原 WAV；绝对路径按原位置解析，相对路径以输入 JSONL 所在目录为基准 |
| `srcs` | 非空字符串列表，去除每项首尾空白后用空格按序拼接，作为英文 `source_text` |
| `chooses` | 非空字符串列表，去除每项首尾空白后按序直接拼接，作为中文 `reference_text` |
| `id` / `key` | 可选，优先取 `id`，其次 `key`，两者均未提供时取 WAV 文件名去掉扩展名 |
| `rejects` | 不使用，不构造静态偏好对 |
| `voice_end` | 不使用，不因 `false` 过滤样本，也不据此改变既有 EOF 轮次 |

`srcs` 与 `chooses` 的片段数应一致。程序只拼接，不改写译文、不补标点，也不自动保证转录与参考质量。当前方向是英译中，无需提供 `en/zh/lang` 字段；若提供 `lang`，应为 `en`。

绝对音频路径必须在训练容器中也能访问。新格式不需要 `--audio-dir`，即使传入该参数，也不会覆盖记录中的 `audio`。若不同目录中存在同名 WAV，请提供不同 `id`：相同 ID 指向不同音频会作为冲突报告，避免静默丢掉另一条音频。同 ID、同路径的重复项保留首条，并记录重复及文本冲突。

旧 `lang/key/en.txt/zh.txt` 格式仍然支持，使用 `--input-format legacy` 并显式传入 `--audio-dir`；默认 `auto` 也可按行识别新旧格式。

## 2. 为什么需要重新准备清单

prepared `train.jsonl` 的每一条 `messages` 都内嵌 system prompt。只更新 Python 文件不会同步修改旧清单。当前提示词已经改为最新 SFT 短模板，因此应从原始 JSONL 重新运行 `01_prepare.sh`，写入一个新的输出目录。

这一步仅读取 WAV 文件头与元数据，生成默认一秒一块的虚拟时间计划，**不复制音频、不预切成小 WAV，也不运行模型**。训练时再按计划读取音频。`srcs/chooses` 没有时间戳，不能直接当作音频分块边界或词级对齐。

参考与原文仅存入 `flex_task` 评分元数据，不作为文字发送给翻译模型；模型最初只收到当前 system prompt 和第一块音频，后续音频由既有多轮调度器逐步加入。

EOF 表示本条音频结束，并不证明一句话的语义已经完整。仍保留原 EOF 观察和 WAIT/FLEX/DIRECT 协议，模型不应补写未听到的后文。

**模型也必须匹配实际训练提示词。** 本文不假定历史 `checkpoint-4410-merged` 使用过最新短模板。应选择确实按 `SFT-Scripts/prepare_duplex_data.py` 最新短提示词训练得到的 SFT 合并 checkpoint。若要恢复旧实验，则使用对应原 Git 提交、原 prepared 数据和原配置；不在旧训练过程中静默替换提示词。

## 3. 准备第二批数据

先更新服务器的项目代码，在已配置好的训练容器与虚拟环境中运行。

**以下 `/data06/yb/Code/FLEX/datasets/batch2/raw.jsonl` 是示例路径，必须替换成第二批原始 JSONL 的实际绝对路径。** 音频已经由每行 `audio` 指定，不需要移动到该示例目录。

```bash
cd /data06/yb/Code/FLEX/FLEX-GSPO
source /data06/yb/venvs/flex-gspo-online/bin/activate

bash 01_prepare.sh \
  --input-jsonl /data06/yb/Code/FLEX/datasets/batch2/raw.jsonl \
  --input-format segments \
  --output-dir /data06/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102 \
  --chunk-seconds 1 \
  --workers 16 \
  --skip-invalid
```

输出目录不能已有同名准备结果。终端会打印读取、接受、拒绝、重复条数和进度；完成后得到：

```text
/data06/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/
├── train.jsonl
├── prepare_report.json
├── rejected.jsonl
└── duplicates.jsonl
```

`--skip-invalid` 保留合法记录，把缺失字段、列表错误、缺失/损坏 WAV、ID 冲突等写到 `rejected.jsonl`。默认时长上限仍为 40 秒；超长音频会被拒绝，不会静默截断。原始 JSONL 和音频不会被修改或删除。若要纳入更长音频，需要同时考虑训练上下文与最大轮数预算。

查看报告：

```bash
python3 -m json.tool /data06/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/prepare_report.json
```

确认 `manifest_published` 为 `true`、`accepted_rows` 大于零，并查看 `rejected_rows` 与重复统计。后续训练使用输出的 `train.jsonl`，不直接使用原始 JSONL，也无需提前运行离线轨迹采样脚本。

## 4. 启动现有 COMET22 v4 训练

**`/data06/yb/Code/FLEX/models/new-sft-merged` 只是模型路径示例，必须替换为实际、且已按最新短提示词训练的 SFT 合并 checkpoint。** 使用新训练输出目录，避免与历史运行混写。

```bash
cd /data06/yb/Code/FLEX/FLEX-GSPO
source /data06/yb/venvs/flex-gspo-online/bin/activate

bash 06_train_qalco_v4.sh \
  /data06/yb/Code/FLEX/models/new-sft-merged \
  /data06/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/train.jsonl \
  /data06/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --train-iters 3000 \
  --save-steps 500
```

入口仍为 `06_train_qalco_v4.sh`，配置仍为 `configs/qalco_v4.json`。质量来自原 COMET22 服务 `http://10.185.1.72:6668/22`，请求使用 `srcs/mts/refs`，响应读取 `score` 数组。训练卡不额外加载质量模型。

模型在训练时为同一音频在线采样 8 条完整轨迹，之后计算原 v4 奖励。保持四卡、G8、global batch 32、micro batch 2、3000 步、每 500 步保存，以及既有并行、offload、EOF 和缓存设置。此次数据及提示词适配不改变这些参数，也不增加自动语义评分前置门槛。

## 5. TensorBoard 两个监控终端

使用之前的宿主机监控环境，要求它能访问代码、训练输出和 prepared 清单。终端一导入奖励、性能与样例：

```bash
cd /data06/yb/Code/FLEX/FLEX-GSPO
/data06/yb/venvs/flex-tensorboard-host/bin/python3 Scripts/monitor_training.py \
  /data06/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --watch \
  --poll-seconds 10 \
  --dataset /data06/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/train.jsonl
```

终端二开启网页：

```bash
/data06/yb/venvs/flex-tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data06/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --host 0.0.0.0 \
  --port 6006
```

重点观察原始 COMET22 质量分、时延、空译/格式错误/重复比例，以及真实译文与逐轮输出。组内归一化 reward 的正负或短期跳动不能单独证明翻译质量改善；实际效果需要固定验证音频对比。本文说明数据与训练衔接，不代表已经执行或验收服务器 GPU 训练。
