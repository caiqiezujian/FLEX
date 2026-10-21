# FLEX：Megatron-SWIFT 在线 GSPO

当前主流程是：**原始 JSONL + 原音频 → 在线训练任务 → 当前策略生成多条完整同传轨迹 → QALCO 评分 → GSPO 更新。**

不需要先把全量数据离线采样成固定 G8 轨迹。已有离线轨迹可以保留用于分析和对照，但不是本训练入口的必需输入，也不会被当作当前策略的新采样直接回放。

适用范围：英→中、已合并的 Qwen3-Omni-30B-A3B SFT checkpoint、4 张 A800 80GB、独立部署的 MetricX 服务。这里输出文字，不训练语音播报。

**实现和 CPU 合约测试已完成；真实 checkpoint、4 卡前向/反向、权重同步和内网 MetricX 仍须在服务器验收。不能把单元测试通过理解为已经完成 GPU 训练。**

## 1. 目录与 SFT 的关系

```text
FLEX-GSPO/
├── 00_install_framework.sh     # 独立环境中的显式安装辅助，不替换 CUDA 栈
├── 01_prepare.sh               # 原始 JSONL + WAV → train.jsonl
├── 02_train.sh                 # Megatron-SWIFT 在线 GSPO 入口
├── prepare_data.py
├── train_megatron.py           # 参数、预检、四卡启动与恢复检查
├── swift_plugin.py             # Omni 音频模板、多轮环境、奖励函数注册
├── check_environment.py
├── check_metricx_service.py
├── configs/default.json        # 当前 QALCO 奖励参数和 MetricX 地址
├── flex_gspo/
│   ├── prepare.py              # 有界并发元数据校验、去重、原子发布任务
│   ├── audio.py                # 因果切片，只读取当前已释放的音频区间
│   ├── online.py               # 每块音频一次决策，构成完整多轮轨迹
│   ├── online_reward.py        # 完整轨迹评分、服务批量调用和诊断日志
│   ├── qalco.py                # 质量门槛、时延预算、长度成本
│   ├── metricx.py              # 用户提供的 MetricX HTTP 协议
│   └── prompts.py             # 原始 SFT system prompt
├── tests/
└── docx/                      # Markdown 文档，不是 Word 文件
```

原项目的 SFT 入口为 `Scripts/train_duplex_sft.sh`，本次没有修改；该脚本不包含在这个公开仓库中。训练消息保持原始 SFT 提示词和 `<|WAIT|>` / `<|FLEX|>` / `<|DIRECT|>` 协议，不通过另写一个新 prompt 改变任务定义。

必须使用正确合并的 SFT checkpoint，且三个协议 token 已在该 checkpoint 中训练。启动预检检查单 token 编码、不同 ID、解码和词表尺寸；不会重新随机初始化这些 token。该检查不能代替对 checkpoint 训练来源的确认。

## 2. 从你的服务器原始数据准备

以下为示例数据布局；实际路径由命令行参数指定：

```text
/data/flex/datasets/
├── 1019/
│   └── en2zh_train_gspo_1.jsonl
└── data_001/
    ├── example_001.wav
    └── example_002.wav
```

注意 Linux 路径大小写敏感。目录名和 JSONL 文件名均可通过参数指定，不要求其他使用者改成你的命名。

原始 JSONL 每行：

```json
{"lang":"en","key":"example_001","en":{"txt":"The meeting starts at nine."},"zh":{"txt":"会议九点开始。"}}
```

运行：

```bash
bash 01_prepare.sh \
  --input-jsonl /data/flex/datasets/1019/en2zh_train_gspo_1.jsonl \
  --audio-dir /data/flex/datasets/data_001 \
  --output-dir /data/flex/datasets/gspo_online_prepared \
  --workers 16
```

处理内容：按 `key` 去重、关联 `key.wav`、校验文本和 WAV 头、记录音频大小和修改时间、规划默认 1 秒一块的虚拟边界。使用流式读取、磁盘去重索引和有界并发，持续打印进度。**不做 ASR、不调用模型、不复制原音频、不生成成千上万个实体 chunk 文件。**

输出：

- `train.jsonl`：直接供在线 GSPO 使用。
- `prepare_report.json`：数量、耗时和处理配置。
- `rejected.jsonl`：损坏记录、缺失音频等问题。
- `duplicates.jsonl`：重复记录诊断。

默认只生成训练集，不自动划出测试集。默认最大音频长度 40 秒；超长记录被报告，不会静默截断。最后不足 1 秒的块在模型输入侧补零，时延计算仍使用真实结束时间。训练默认允许单块最长 2 秒；若主动修改 prepare 的 chunk 长度，也须同步训练的 `--max-chunk-seconds`。

默认严格模式：发现无效记录则保留报告但不发布最终训练清单。修复数据后使用新的输出目录重跑；若明确接受过滤，可以加 `--skip-invalid`。同 key 第一条合法候选优先，重复的文本冲突会记录。输出目录有占用锁和覆盖保护，不要并发写同一个目录。

准备后的每行有三个核心字段：

```json
{"messages":[{"role":"system","content":"原始 SFT 提示词"},{"role":"user","content":"<audio>"}],"audios":["flexaudio:v1:编码后的首块引用"],"flex_task":"编码成 JSON 字符串的任务元数据"}
```

`flex_task` 包含原音频路径、边界、英文转写、中文机器参考等。它是环境/奖励侧数据，**英文转写和参考译文不进入模型聊天内容**。初始 actor 输入只含第一块音频，后续由环境逐块释放。

## 3. 训练框架与环境

使用 **Megatron-SWIFT 训练 + vLLM colocate 多轮 rollout + 项目自定义插件**。GSPO 使用框架的 `rlhf_type=grpo` 入口配合 `importance_sampling_level=sequence`，不是把普通 SFT 换个名字，也不是自写一个 HF 单机 trainer。

本次接口审查基于官方源码提交：

```text
modelscope/ms-swift
f15bb91439d756d4a366b51956d72d1f088b7af6
4.7.0.dev0
```

原 SFT 的旧版 SWIFT 环境不能默认认为具备这些多轮、音频回传和 GSPO 接口。请保留已验收的 SFT 环境，在单独环境或容器中适配训练；Python 需要至少 3.10，并需要相互兼容的 PyTorch、CUDA、vLLM、Megatron-Core、TransformerEngine 和 FlashAttention。

`00_install_framework.sh` 只是安装审查过的 SWIFT 源码、mcore-bridge 和本项目轻量依赖的辅助脚本，**不是从空环境安装完整 CUDA 训练栈的一键脚本**。它不会主动升级 torch/vLLM/Megatron-Core/TransformerEngine，也不会像旧采样代码一样写死 vLLM 必须等于某个历史版本。

在准备好的独立环境中，可以显式执行：

```bash
bash 00_install_framework.sh --confirm-isolated-env
python3 check_environment.py --require-training
```

预检包含框架源码能力、相关依赖、CUDA 组件导入和 GPU 数量。导入检查仍不等于多轮 Omni 训练一定成功；真实模型测试见下面两步验收。

## 4. MetricX

公开配置中的 `service_url` 和 `service_id` 留空，部署前必须在 `configs/default.json` 填写你自己的服务地址（例如 `http://METRICX_HOST:16999`）和实例标识。默认 **带参考模式**：

```text
POST /metricx24
{"srcs":["原文"],"mts":["完整候选译文"],"refs":["机器参考译文"]}
→ {"scores":[0.4]}
```

同时支持 `/metricx24/qe`。客户端 `trust_env=False`，内网请求不走环境代理；分数越低越好。请求失败、响应维度错误或非有限分数会报错，不会用默认奖励继续训练。模型非法输出则是正常的负奖励样本，不等同于服务故障。

服务器上先检查：

```bash
python3 check_metricx_service.py --mode both
```

该服务在独立机器运行，不占这四张卡的训练显存。服务地址或评分参数需要更改时编辑配置文件，或使用训练入口的 `--reward-config` 指定另一份配置。

## 5. 先完成真实两步验收

下面的参数直接跟路径，不需要设置 `MODEL_PATH` 变量；请替换为你的真实模型和数据路径。请在 `FLEX-GSPO` 目录中执行：

```bash
bash 02_train.sh \
  /data/models/flex-sft-merged \
  /data/flex/datasets/gspo_online_prepared/train.jsonl \
  /data/flex/outputs/gspo_smoke_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 2 \
  --global-batch-size 4 \
  --train-iters 2 \
  --save-steps 1
```

至少需要两条合法任务。可以先在命令末尾加 `--dry-run` 查看最终参数：该模式不加载 GPU 模型、不调用服务、不写训练目录，也不代表训练通过。

两步验收重点：真实音频能够逐块输入、第一轮输出与 SFT 推理合理一致、完整轨迹进入评分、损失和梯度有限、确实发生优化器更新、下一批 rollout 使用更新后的权重、checkpoint 能保存。所有候选相同而导致组内 advantage 为零，不是报错，但也不能称为已经获得有效学习信号。

## 6. 验收后使用 G8

为正式训练使用新的输出目录，不在 smoke 目录上混跑：

```bash
bash 02_train.sh \
  /data/models/flex-sft-merged \
  /data/flex/datasets/gspo_online_prepared/train.jsonl \
  /data/flex/outputs/gspo_g8_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 16 \
  --train-iters 1000
```

主要默认配置：

| 配置 | 初始值 |
| --- | --- |
| Megatron TP / EP / PP | 2 / 2 / 1 |
| vLLM TP | 4 |
| 训练方式 | LoRA rank 8，alpha 32；冻结音频/视觉编码器与对齐模块 |
| 学习率 / KL 系数 | 1e-6 / 0.01 |
| GSPO clipping 下/上范围 | 0.0003 / 0.0004 |
| rollout prompt 预算 / 每轮新增 token | 8192 / 128；vLLM 总预算 8320 |
| 最大音频轮数 | 64 |
| sampling temperature / top_p | 1.0 / 1.0 |
| 每批 rollout 后更新轮次 | 1 |
| rollout 显存比例 | 0.4 |
| colocate sleep / offload | sleep level 2，模型与优化器 offload |

这些是起步配置，不是四卡显存和速度的保证。offload 依赖 CPU 内存和主机带宽，需结合服务器实测调整。默认采用完整合并权重同步，不假设未经验证的 Omni LoRA-only 在线刷新可用。8192 是 rollout prompt 编码预算，不是完整 teacher-forced 训练序列的绝对长度上限；框架在最终训练编码时会临时取消模板截断。

## 7. 日志和恢复

训练目录记录启动配置、任务清单哈希、奖励配置/代码标识，以及 reward 诊断 JSONL。每条诊断包含完整候选轨迹、标签、时间、完整译文、各奖励项和有效性原因。原文与参考留在任务清单，不在每条日志重复存储。

Megatron 负责优化器、随机状态和模型 checkpoint；本入口启用了这些状态保存，并请求框架同时导出合并 HF 权重。实际产物必须以服务器保存日志和文件检查为准。

恢复示例：

```bash
bash 02_train.sh \
  /data/models/flex-sft-merged \
  /data/flex/datasets/gspo_online_prepared/train.jsonl \
  /data/flex/outputs/gspo_g8_001 \
  --group-size 8 --global-batch-size 16 --train-iters 1000 \
  --resume-from /实际的Megatron原生adapter检查点目录
```

恢复目录须有框架原生 tracker 和完整分布式 checkpoint shards，不是 HF merged 导出目录。上游也会把 tracker 和部分元信息复制到 merged 目录，因此不能只看 tracker 判断可恢复；入口已对此做检查。总训练步数和关键配置保持一致，不能把“两步 smoke”直接改成 1000 步声称无缝恢复。新实验使用新输出目录。启动身份校验包含权重元信息，不宣称已对所有权重文件做内容级哈希。

## 8. 奖励与验证边界

详见 [QALCO 强化学习方案详解](docx/QALCO强化学习方案详解.md)。第一版是**完整译文的 MetricX 质量 + 整条轨迹的时延预算 + 超长成本**，不是每个短块单独做完整句子评分。WAIT token 参与策略梯度；FLEX 不因尚未成句就被单独判低质量。

默认时延使用所有输出单元的 mean-lag 代理，避免“把输出继续拖到 EOF，指标反而下降”的截断问题。它不是精确语义延迟，也不是 GPU 墙钟延迟。当前没有独立自然度模型、逐 WAIT 反事实分支、强制语义对齐或独立标点惩罚，不能宣称已经精确解决每一次等待是否值得。

本地运行测试：

```bash
python3 -m unittest discover -s tests -v
```

部分框架合约测试使用审查过的官方源码；其他机器可通过 `FLEX_SWIFT_SOURCE` 指定该 checkout，未提供则跳过相应源码测试。本地 CPU 测试覆盖真实 WAV + 本地 HTTP 评分替身的贯通流程；**没有执行真实 Omni 模型训练或调用用户内网 MetricX**。

旧自写训练器、旧固定轨迹评分入口及对应不用的测试已从本目录移除。原作者本地修改前已另行备份，备份不随仓库发布；`Scripts`、`FLEX-GSPO-Data`、原音频、模型和已有 G8 结果没有删除。

## 官方实现依据

- [Megatron 多轮 GRPO 启动例子（审查提交）](https://github.com/modelscope/ms-swift/blob/f15bb91439d756d4a366b51956d72d1f088b7af6/examples/megatron/grpo/multi_turn/frozen_lake.sh)
- [Megatron MoE colocate LoRA 例子（审查提交）](https://github.com/modelscope/ms-swift/blob/f15bb91439d756d4a366b51956d72d1f088b7af6/examples/megatron/grpo/moe_colocate_lora.sh)
- [GSPO 官方说明](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/GSPO.html)
- [Megatron-SWIFT 参数文档](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html)
