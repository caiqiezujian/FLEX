# FLEX：Megatron-SWIFT 在线 GSPO

当前主流程是：**原始 JSONL + 原音频 → 在线训练任务 → 当前策略生成多条完整同传轨迹 → QALCO 评分 → GSPO 更新。**

不需要先把全量数据离线采样成固定 G8 轨迹。已有离线轨迹可以保留用于分析和对照，但不是本训练入口的必需输入，也不会被当作当前策略的新采样直接回放。

适用范围：英→中、已合并的 Qwen3-Omni-30B-A3B SFT checkpoint、4 张 A800 80GB、独立部署的 MetricX 服务。这里输出文字，不训练语音播报。

**实现和 CPU 合约测试已完成；真实 checkpoint、4 卡前向/反向、权重同步和内网 MetricX 仍须在服务器验收。不能把单元测试通过理解为已经完成 GPU 训练。**

**针对四卡训练停在 `Train: 0/2` 的更新：** 已增加 FLEX 专用多轮退出同步修复。Megatron TP2 / vLLM TP4 时，退出判定必须覆盖真正参与推理的四卡，不能只覆盖两个训练进程。补丁随插件自动安装，不修改 `site-packages`，不改变训练数据分配或奖励公式。已有环境和 `train.jsonl` 可以复用；上传代码后请使用新输出目录完成两步复测。具体命令见 [多轮同步死锁修复与复测](docx/多轮同步死锁修复与复测.md)。

## 1. 目录与 SFT 的关系

```text
FLEX-GSPO/
├── 00_install_framework.sh     # 独立环境中的显式安装辅助，不替换 CUDA 栈
├── 01_prepare.sh               # 原始 JSONL + WAV → train.jsonl
├── 02_train.sh                 # Megatron-SWIFT 在线 GSPO 入口
├── 03_train_4100.sh            # 已通过两步验证后使用的 G8/B32、4100 步参数入口
├── 04_train_throughput.sh      # 可选提速试验：B32/micro2/推理并发32/显存预算0.50
├── monitor_training.py         # 独立进程：已有奖励/耗时日志 → TensorBoard，不参与训练
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
│   ├── rollout_sync.py         # FLEX-colocate 退出判定按真实 vLLM TP 组同步
│   ├── performance.py          # 可选本地阶段计时，不新增 CUDA 同步或分布式通信
│   ├── monitoring.py           # 训练外日志观察器，导出曲线/文本与可选 GPU 读数
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

### 6.1 已跑通 G8/B32 后，正式训练 4100 步

`03_train_4100.sh` 是 `02_train.sh` 的参数快捷入口，不是新的训练阶段。在已经完成两步验收的 Docker 和虚拟环境中执行：

```bash
cd /data/flex/FLEX-GSPO
source /data/venvs/flex-gspo-online/bin/activate

bash 03_train_4100.sh \
  /data/models/flex-sft-merged \
  /data/flex/datasets/gspo_online_prepared_clean/train.jsonl \
  /data/flex/outputs/gspo_g8_b32_4100_001
```

模型、JSONL 和输出目录都是直接传参，也可替换为其他路径。命令末尾可加 `--dry-run` 检查配置，不会启动训练。必须使用新的输出目录；本轮从原始合并 SFT checkpoint 开始，不恢复两步排障实验，以便学习率计划按 4100 步重新建立。

配置保留 G8、global batch 32、micro batch 1、每次生成更新 1 步、学习率 `1e-6`、KL `0.01`、LoRA rank 8/alpha 32、8192/128 token 预算、64 轮上限和 vLLM 显存比例 0.4。训练 4100 次优化更新，每 100 步保存，框架保存保留上限仍为 3；合并模型导出仍启用，应预留充足磁盘空间。warmup 比例仍为 0.03，对应 123 步。4100 不是 epoch 数。

按本配置，每次生成计划使用 4 条源音频任务，每条 8 个候选；4100 步计划约 16,400 次任务采样、131,200 条候选轨迹。它不等于全量约 25 万条数据训练一遍，也不保证这些任务全部互不重复；编码失败重采样等行为可能增加实际读取量。

快捷脚本清除 `CUDA_LAUNCH_BLOCKING`、C++ 堆栈、分布式详细调试等排障设置，将多轮进度日志恢复为每 5 轮打印，不改变 allocator 或已验收的 offload 设置。单次显存占用不到 40GB 不等于整个训练峰值只到 40GB；不要仅凭这个读数直接关闭 offload 或翻倍 batch。

当前启动器明确使用 `log_completions=false`：该固定框架的 prompt 日志会仅携 messages 调用音频编码，触发缺少 audios 的错误。关闭的是框架额外的 prompt/completion 展示，不是 QALCO 轨迹记录；`rewards/` 中完整轨迹与奖励日志、训练指标及 TensorBoard 仍然保留。此配置不放松真实模型输入的音频校验。固定上游的 reward/advantage 展示缓存为有限长度 deque，不会因关闭 completion 日志而无限堆积。

使用此入口前，同步本次修改的 `train_megatron.py`；仅复制新 shell 文件而保留旧的 `log_completions=true` 启动器会重新触发日志错误。服务器的实际 MetricX 配置保持不变。正式运行的最终质量、速度和稳定性仍以服务器观测为准，两步成功不是长程效果保证。

## 7. 日志和恢复

### 7.1 安全提速与 TensorBoard

保留原 `02_train.sh` / `03_train_4100.sh` 稳定配置。新 `04_train_throughput.sh` 只把训练 micro batch 从 1 改为 2、vLLM `max_num_seqs` 从 16 改为 32、vLLM 显存预算从 0.40 改为 0.50，并打开可关闭的阶段计时。G8、global batch 32、每批生成更新 1 步、学习率、KL、奖励公式、音频分块、多轮完成同步、TP/EP、完整重计算、eager 和 offload 全部保持原设置。它是待服务器验证的提速候选，不是已测出的加速倍数。

先用新输出目录验证 5 步及保存，再决定是否启动 4100 步；完整命令、回退方法、监控启动及曲线判读见 [提速与训练监控](docx/提速与训练监控.md)。不需要重新 prepare、采样离线轨迹或安装新版本训练框架。

`--performance-log on` 仅添加进程本地墙钟计时，不做 CUDA synchronize 或新增分布式通信。计时记录在 `performance/`，包含生成总耗时、权重同步、逐轮请求总耗时、CPU 卸载与 GPU 回载；它不是精确 CUDA kernel profiler，不要把四个 rank 的耗时相加当成一次更新耗时。默认关闭；`04` 默认打开，末尾可传 `--performance-log off` 关闭。

`monitor_training.py` 由另一个终端启动，只读现有日志并在 `monitor/` 写派生 TensorBoard 事件。不加载 Omni、不调用 MetricX、不占用模型训练显存、不重新启用 `log_completions`。框架已有的 loss、学习率、梯度范数、KL、reward_std、`frac_reward_zero_std` 等指标仍由原 TensorBoard 日志提供；新增质量、时延、长度成本、门槛通过率、动作比例和文本样本帮助解释奖励。应将整个实验输出目录作为 TensorBoard logdir。

不要把“loss 下降”或“奖励上升”单独当成翻译变好：还要看质量/时延门槛、无效率、实际译文和固定独立验证集。质量为 MetricX 代理、时延为源时间轴代理；当前并没有独立的自然度分数。

训练目录记录启动配置、任务清单哈希、奖励配置/代码标识，以及 reward 诊断 JSONL。每条诊断包含完整候选轨迹、标签、时间、完整译文、各奖励项和有效性原因。原文与参考留在任务清单，不在每条日志重复存储。

新增控制台阶段日志：`patch_installed`、`generation_begin`、`sync_ready`、`completion_check_begin/end`、`continuation_begin/end`、`generation_end`、`reward_begin/end`。包含 rank、当前轮次、本地请求数、组内仍活跃的进程数与耗时；不打印音频或参考文本。默认每 5 轮打印，首轮、本地活跃状态变化、空批次和最终退出也打印。排查时加 `--rollout-log-every 1` 可逐轮查看。`local_requests=0` 不代表故障：该进程已经完成自己的候选，但仍在配合同组其他进程通信。两步验收通过后可去掉该参数，恢复默认日志频率。

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
