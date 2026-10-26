# QALCO v3：质量门控时延与组内简洁性

本版是可试训的新奖励实验，不是已经证明更优的最终方案。保持现有 SFT 输入协议、在线音频轨迹、Megatron-SWIFT GSPO、COMET22 服务和四卡参数，仅更换奖励计算及必要的完整分组接口。没有新增大模型、BGE 或 NLI 服务。

## 1. 训练对象和信用分配

每条任务仍是原音频 + 英文转录 + 中文机器参考。音频每个块到来时，actor 输出 WAIT/FLEX/DIRECT；音频全部输入后还有一次已有的 EOF 文本收尾。英文转录和参考译文只给评分器，不进入 actor 对话。无需重新准备 train.jsonl、切分 WAV 或预采样离线轨迹。

每次更新：4 个不同的音频任务 × 8 个在线采样候选 = 32 条完整轨迹。一个奖励对应一条完整轨迹，而不是孤立评价某个 FLEX 碎片。框架将组内相对优势用于轨迹中模型生成的 token（包括动作标签）；用户音频和 EOF 观察不是模型动作。WAIT 没有译文但影响后续输出时间。不要把 WAIT 数量直接当成时延，不对 FLEX/DIRECT 数量发奖金。

这仍是整条轨迹的结果奖励，不是为每次 WAIT 单独计算“如果不等会怎样”的因果收益。若 G8 全部为空、都得到 −4，该组优势可能全零；增加惩罚常数不能凭空产生排序信号。保留 EOF 收尾机会并监控空译文率，不能以强制输出或伪造译文掩盖问题。

## 2. 新奖励公式

对协议有效、输入完整、输出非空、没有严重循环的候选 i：

$$
R_i=Q_i-2[0.80-Q_i]_+
-\mathcal D_{\mathrm{gated}}(Q_i,d_i)
-0.08w(Q_i)C_i-0.20P_i
$$

其中正部函数是：

$$
[x]_+=\max(0,x)
$$

Q 是 COMET22 对整段原始拼接译文的带参考分数，越大越好。它是质量代理值，不是准确率、完整性保证或前缀依据证明。机器参考本身也可能有错，不应将所有偏离参考的表达视为错误。

质量门控和时延成本：

$$
w(Q)=\operatorname{clip}\left(\frac{Q-0.75}{0.10},0,1\right)
$$

$$
z_i=\frac{d_i}{3},\qquad
D(d_i)=\min\left(2,0.05z_i+0.20[z_i-1]_+^2\right)
$$

$$
\mathcal D_{\mathrm{gated}}(Q_i,d_i)
=2(1-w(Q_i))+w(Q_i)D(d_i)
$$

- Q≤0.75：时延项固定扣 2，简洁性权重为零；模型不能靠更快漏译、更短乱译取得这两项收益。
- 0.75<Q<0.85：连续过渡，不使用 v2 的跨档跳跃奖励。
- Q≥0.85：按真实计算的调度时延代理扣分，质量相近时再比较简洁性。
- 3 秒是软预算，不是绝对截止。超预算成本增长，最终封顶 2；特别严重超时之后不再继续区分更晚程度，这是本次所选公式的边界。
- 协议错误、截断/不完整、最终空译文或严重循环：固定 −4，不进入长度基准池，不靠高 COMET 分翻盘。

默认合法候选的奖励高于 −4。奖励允许负值，GSPO 根据同组的相对优势更新；不用把所有数值平移到正区，也不应将 loss 单调下降当作验收标准。

### 时延 d 的精确定义

沿用已运行版本的 source-clock mean positive lag。N 为输出文本单位数，M 为机器参考文本单位数，T 为音频时长，t_j 为第 j 个译文单位产生时已听音频时长：

$$
d=\frac{1}{N}\sum_{j=1}^{N}
\max\left(0,t_j-\frac{(j-1)T}{\max(N,M)}\right)
$$

EOF 轮的 t_j=T+0.5，0.5 秒是明确设置的调度代理成本，不是实测 GPU 延迟。保留真实 EOF 生成耗时供诊断，但不把它混进该奖励。中文按字符、英文按词、数字按数值片段计数，标点不算文本单位。

这个 d 不是词级源语义对齐时延，也不是用户看到字幕的墙钟时延；它与候选和参考长度有关。固定译文时，推迟输出不会降低该 d；不同译文之间仍可能存在长度耦合，必须通过固定验证集检查，不应写成“已完整解决时延评估”。

## 3. C：用完整 G8 建立长度基准

不是选择 WAIT 最少的候选，也不是直接挑最短文本。对同一音频、同一次 prompt 抽样的完整 G8：

1. 排除无效、空译文、严重循环候选。
2. 找到剩余候选的最高质量 Q_best。
3. 基准池只收 Q≥max(0.80,Q_best−0.03) 的候选。
4. 池内不足 3 个时，全组 C=0，不猜一个不可靠的简洁性目标。
5. 取池内文本单位长度的 25% 分位数 L_star，采用线性插值。
6. 留出 max(2,0.05L_star) 的长度余量。

$$
C_i=\operatorname{clip}\left(
\frac{N_i-L_\star-\max(2,0.05L_\star)}{\max(L_\star,20)},0,1\right)
$$

这里 20 只是分母下限，不把短句的 L_star 强行拉到 20。短于基准没有额外奖金；略长也不罚。C 仍只是“相对长度过量”，不能声称等同于语义冗余。COMET 误把漏译评成高分时，池仍可能受到污染；本版不额外部署语义检查服务，因此不能保证识别所有这类情况。

### 数值示例

某组质量基准池长度为 105、77、74、68、69，则 L_star=69，余量=3.45。假设三条候选没有重复惩罚：

| 候选长度 N | COMET Q | 时延 d/秒 | 新奖励 |
|---:|---:|---:|---:|
| 105 | 0.869525 | 0.710291 | 0.819948 |
| 77 | 0.872080 | 1.698959 | 0.838488 |
| 74 | 0.895682 | 1.768176 | 0.864415 |

第一条出得早但较长，第二条多等了一点、质量相近且短很多，获得更高奖励；第三条质量更高，获得最高奖励。这是按公式计算的示例，不代表模型训练后一定能产生相同改善。若第二条等待到 4.5 秒，奖励约降到 0.741804，也不能无限等待换取短表达。

## 4. P：保守的局部机械循环检查

独立 v3 检查器识别至少 4 个单位的原始周期短语，连续重复至少 4 次；周期上限 64。对有句式证据或符合保守长短语条件的循环，从第 4 次开始计入多余单位；重叠区间取并集，除以全译文单位数得到 P。严重循环以有句式证据的完整循环覆盖量达到全文 60% 判定，而不是要求 P≥60%。

保留普通 FLEX 锚点、短实体、三次排比和参考中确实存在的可比重复。源文中有可比重复结构时保守豁免并记录原因；这种跨语言结构豁免不是语义对齐。所有检查只读原文，COMET 接收未清洗的原始译文。

它不检测所有近义重复、不保证识别无意义输出，不是幻觉/漏译检测模型。新版本修复了一类无主语中文机械循环漏检，但仍须抽样核查。

## 5. 分布式实现与安全边界

- ORM 先对本地完整轨迹调用 COMET22，暂存质量、时延、长度、P 和协议状态，不写成最终奖励。
- 在 Megatron trainer 的统一评分边界，按与框架优势计算相同的 world-rank 顺序汇总轻量元数据，划分连续 G8。
- 验证同组完整任务哈希、候选身份、组大小与排列。相同 task_id 在不同抽样块出现，仍是不同组；禁止按 task_id 混成大组。
- 只有通过验证才计算全组 C，并把最终奖励写回框架张量和日志。框架的组归一化、GSPO loss、优化器保持不变。
- 每次评分有元数据/错误交换及最终状态交换，两次小型 collective；不传输 WAV、整段参考或整个轨迹。服务错误不会变成假的 −4 策略奖励，而是全组明确报错退出。
- 单进程崩溃、硬件/网络故障仍由 torchrun/NCCL 处理，不能承诺任何情况下都不会挂。

此前多轮结束不同步修复、MM processor cache=0、EOF 协议全部保留。不升级 torch/vLLM，不编辑 site-packages。

## 6. 四卡配置保持不变

| 项目 | 参数 |
|---|---|
| 框架/算法 | Megatron-SWIFT RLHF；sequence importance sampling + GSPO clipping |
| 数据与轨迹 | 每步 4 音频 × 8 候选；在线新策略采样 |
| Batch | global32，micro2，steps_per_generation1，num_iterations1 |
| 总更新/保存 | 3000 / 每500步；save_total_limit=3 |
| 并行 | TP2、EP2、PP1、CP1；vLLM TP4 |
| LoRA | rank8、alpha32、all-linear；冻结音频相关模块配置沿用 |
| 学习率/KL | 1e−6，warmup3%，beta0.01 |
| 采样 | temperature1、top_p1；不强制标签分布 |
| 上下文 | max_length8192、每轮128新token、最多65轮（包含EOF） |
| vLLM | 内存比例0.5、max_num_seqs32、eager、MM cache0 |
| 显存策略 | full recompute、sleep2、model/optimizer offload；不增加batch |

用户之前的 Omni RL 脚本也是 Megatron-SWIFT。这里保留同样的框架和在线组采样思路，但不直接照搬 EP4、LoRA32、steps_per_generation4 或 packing/padding_free，因为当前是多轮音频轨迹，并且四卡峰值已接近 80GB。

## 7. 运行顺序

在已跑通的 Docker 和虚拟环境内，不在宿主机的 base Python 运行训练。先更新整个 FLEX-GSPO 文件夹；不改现有数据和老输出目录。

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO
source /data/yb/venvs/flex-gspo-online/bin/activate
python3 -c "import sys; print(sys.executable)"
python3 Scripts/check_comet22_service.py --config /data/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v3.json
~~~

先两步独立验收（含保存），不减小 G8/B32，确保覆盖真实分组：

~~~bash
bash 05_train_qalco_v3.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/megatron_output/Qwen3-Omni-Flex/v3-20261016-172842/checkpoint-4410-merged \
  /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl \
  /data/yb/FLEX/outputs/gspo_qalco_v3_smoke_001 \
  --train-iters 2 \
  --save-steps 2
~~~

正式训练使用新的目录和原 SFT 权重：

~~~bash
bash 05_train_qalco_v3.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/megatron_output/Qwen3-Omni-Flex/v3-20261016-172842/checkpoint-4410-merged \
  /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl \
  /data/yb/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --train-iters 3000 \
  --save-steps 500
~~~

05 自带其余参数，上述显式重复主要参数方便确认。可以加 `--dry-run` 查看启动计划，不加载模型、不评分、不写文件。`configs/default.json` 仍是 v2，05 明确指定的是 `configs/qalco_v3.json`。

这不是从旧 v2 的 optimizer 状态继续。若改用 checkpoint-500-merged 作为 model 参数，是一个从该权重热启动的新实验，应使用新输出目录，不要称作与原 SFT 起点相同的公平对照。两步 smoke 的学习率总周期不同，不能当作正式3000步的无缝恢复点。

## 8. 日志与 TensorBoard

原训练 loss、KL、组内 reward_std、零优势组比例、grad_norm 继续写入原生 TensorBoard。另一个终端运行现有 observer，将每项奖励扣分和样例导入 TensorBoard：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO
source /data/yb/venvs/flex-gspo-online/bin/activate
python3 Scripts/monitor_training.py \
  /data/yb/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --watch
~~~

宿主机仍使用已建立的 TensorBoard 专用环境：

~~~bash
/data/yb/venvs/flex-tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data/yb/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --host 0.0.0.0 \
  --port 6006
~~~

若旧6006服务还在，应停止它或使用另一个端口；不要覆盖另一个服务。

重点看 COMET22、quality_gate、latency_penalty、weighted_brevity_penalty、weighted_repetition_penalty、anchor_pool_size、empty_translation、severe_repetition、以及每条记录的 group_finalized。每进程日志的均值仍是局部统计；不要重新把局部8条假定为同音频组，组身份以新桥记录为准。

新 reward 和旧 v2 数值不在同一标尺，不能用均值变大证明训练变好。每步仅4条不同音频，原始 reward 波动仍会明显。用固定的音频集合、固定解码设置对比 SFT 与新 checkpoint，并同时检查准确性、时延、长度、空译文和循环。先看20～30步数值/输出是否正常，再看固定验证集是否改善。

取最近诊断（纯文本读取，不调用模型或评分服务）：

~~~bash
python3 Scripts/collect_training_status.py \
  /data/yb/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --reward-config /data/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v3.json \
  --window 10 \
  --latest 10 \
  --examples 4
~~~

launch JSON 保存完整已校验 reward_settings 和配置哈希，之后不要用新改的配置冒充本次参数。停止后的 `Scripts/analyze_training_run.py` 可以检查 v3 奖励算术，但使用的是日志中 C、P；它的每rank分析不独立重建全局长度池，也不认证语义正确性。

检查、审计和诊断入口已统一放到 `Scripts/`。另有 `Scripts/replay_reward_audit.py`，使用冻结 COMET22 分数和按原顺序提供的完整逐轮 TXT 离线回放 v3，不请求评分服务；完整局部 G8 的输入要求与命令见 [Scripts 使用索引](../Scripts/README.md)。当前 `Scripts/collect_brevity_samples.py --include-increments` 的普通导出会省略正常 WAIT，且不强制 G8，不能直接替代此前完整逐轮导出包装器的输出。

## 9. 当前没有承诺解决的事

- COMET22 可能高估漏译、添加内容或参考本身错误的候选；没有新语义服务意味着这个盲区仍在。
- 质量门控在0.75～0.85区间斜率较大，0.01质量差可能显著改变奖励；保留该公式是为了验证用户选定的方案，不声称它是最优校准。
- 全文高质量不能证明每一步都未使用未来信息；当前记录仍明确 `prefix_support_checked=false`。
- 简短输出不是天然正确，长输出不是天然错误；组内长度成本只是质量优先的弱约束。
- 如果完整 G8 都同样冗长，相对长度成本也可能全为零；这个目标不凭空提供绝对简洁性标准。
- 本地单元测试/模拟多rank不等于A800实机通信、长期稳定、服务评分质量或训练效果认证。

参考算法：[Qwen 官方 GSPO 说明](https://qwenlm.github.io/blog/gspo/)。本项目的 QALCO 奖励是独立实验设计，不能把 GSPO 的已有结果作为本项目有效性的证据。
