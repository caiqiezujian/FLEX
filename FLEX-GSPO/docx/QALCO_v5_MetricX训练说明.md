# QALCO v5：MetricX 质量与门控时延

更新日期：2026-11-05。本版已整合到唯一维护的完整仓库 `D:\ICLR\FLEX\FLEX-GSPO`；不再依赖旧的独立目录。质量权重为 1.0，时延权重为 0.5，使用原 MetricX v5 的系数配置。独立入口为 `07_train_qalco_v5.sh`，配置为 `configs/qalco_v5.json`。v4 仍使用 COMET22；v5 **只使用 MetricX-24 reference 模式评分，不与 COMET22 混合**。沿用原在线 GSPO：训练时从当前策略为同一音频生成 G8，整条轨迹结束后评分，再更新策略。

这是借鉴 HPO 源码后对 FLEX 的适配，不是 HPO 的完整复现。现有 WAV 与 WAIT/FLEX/DIRECT 协议继续使用，不需要重新切音频或预先生成离线训练轨迹。

**当前共享提示词已与 `SFT-Scripts/prepare_duplex_data.py` 最新短模板的中文展开结果逐字对齐。** `flex_gspo/prompts.py` 保存本地常量，不在服务器运行时导入外部 SFT 脚本。准备阶段写入的新 `messages` 与在线采样使用同一份提示词，EOF 观察及协议 token 不变。

旧 prepared JSONL 内嵌了旧 system prompt，不能直接认为兼容。请用更新后的 `01_prepare.sh` 从原始 JSONL 重新生成到新输出目录，音频仍引用原位置，无需复制或重切。模型也必须是确实使用该短提示词训练的 SFT checkpoint；修改数据或目录名称不会改变历史模型所接受的训练，不能据此认定 `checkpoint-4410-merged` 已匹配新提示词。历史 v4 公式与入口仍保留，但当前各入口共用新提示词，恢复历史运行请使用原提交。

## 1. HPO 实际实现与本版取舍

核对资料：

- [HPO 论文](https://aclanthology.org/2026.acl-long.80.pdf)。
- [HPO 评分源码](https://github.com/owaski/HPO/blob/d93140ea890fd96c8b17c4c32f9c3b1ba3ff05d2/nemo_rl/environments/games/infinisst.py)，核对提交 `d93140ea890fd96c8b17c4c32f9c3b1ba3ff05d2`，不宣称该提交始终为最新版本。
- [HPO 默认实验配置](https://github.com/owaski/HPO/blob/d93140ea890fd96c8b17c4c32f9c3b1ba3ff05d2/examples/configs/grpo_infinisst_4b_laal0.5_as_tgtq-5.0.yaml)。

| 项目 | 核对的 HPO 代码 | QALCO v5 |
|---|---|---|
| 质量输入 | MetricX，source + translation + reference，`qe: false` | 已有远端 MetricX reference 服务，相同三类文本 |
| 质量方向 | 原始错误分取负，越大越好 | `q = -metricx_error` |
| 质量门槛 | `q >= -5` | 原始错误分 `e <= 5`，含等号 |
| 时延门控 | 达标使用截顶时延，未达标采用固定最差时延项 | 达标使用 `min(d,10)` 秒；未达标固定 10 秒 |
| 分项缩放 | 质量、时延分别用各自的组标准差 | 分别缩放，加 0.5 的尺度下限与 ±2 截断 |
| 质量/时延权重 | 1 / 0.5 | 1 / 0.5 |
| 评分粒度 | 默认分段层级评分，依赖对齐；另有文档模式 | 保留整轨迹评分，完整同音频 G8 分组 |
| 时延口径 | 对齐片段上的有符号 lag | 全部输出单位的正 lag，加已有 EOF 代理成本 |
| 策略优化 | 仓库配置是 token-level GRPO | 现有 Megatron-SWIFT sequence-level GSPO |

源码中 HPO 的时延项可写成：质量达标时 `h = min(0, L - 10)`，未达标时 `h = 0`，之后减去归一化后的 h。v5 使用成本 `c = h + 10`；给整组加相同常数不会改变中心化后的结果。因此“未达标 h=0”不代表免罚，它对应最差成本 10 秒。

另外，论文公式与代码存在需要注意的区别：论文式(4)印出的时延归一化分母是质量标准差，源码实际分别使用时延自己的标准差；源码还明确把时延成本在 10 秒截顶。v5 按核对后的源码思路实现。HPO 的切分/对齐体系没有移入本项目，避免同时改变多轮训练链路。

## 2. 每条轨迹如何获得质量分

actor 每次只接收当前可用音频、已有对话和既定 EOF 观察。`source_text` 与 `reference_text` 只用于奖励，不进入 actor 输入。将该轨迹所有 FLEX/DIRECT 的新增译文按原顺序拼接，去除协议标签，然后发给服务：

```json
{
  "srcs": ["同一音频的完整英文转录"],
  "mts": ["该轨迹的完整中文增量拼接结果"],
  "refs": ["现有 DeepSeek 离线中文参考"]
}
```

地址：`http://10.185.1.72:16999/metricx24`，返回 `{"scores":[原始错误分]}`。使用已有 HTTP 客户端、禁用环境代理，训练卡不加载 MetricX 权重。源码不自行截断文本；远端的模型身份、输入上限与截断行为仍由实际服务决定。本次没有访问该内网服务或修改服务部署。

要求每个分数是有限数，范围 `[0,25]`。原始错误分 e 越小越好，优化使用：

$$
q_i=-e_i
$$

日志中的 `quality = 1 - e/25` **仅为兼容监控的线性变换**，不是准确率，不与 COMET22 分数直接比较。跨卡传递真实 `metricx_error`，不会从这个变换反推 e，以免 `e=5` 边界出现浮点翻转。

同一局部评分批次内完全相同的译文只请求一次质量分，时间轨迹仍分别计算。全程 WAIT、格式错误、严重循环不调用质量服务；它们的原始质量为 null，不能伪造成错误分 25。

## 3. 时延与“值不值得等”

保留已有调度时延，设音频总时长 S、输出 N 个文本单位、参考 M 个单位、第 j 个输出单位出现时刻 t：

$$
d_i=\frac{1}{N_i}\sum_{j=1}^{N_i}
\max\left(0,t_{ij}-\frac{(j-1)S}{\max(N_i,M)}\right)
$$

中文按字、拉丁文本按词、数字按数字单位计数。最后 EOF 输出时间额外增加已存在的 0.5 秒代理成本；它不是测量出的模型计算时间。没有文本的 WAIT 不制造输出单位，但它把后续译文推迟时，会反映在 t 中。

这是 **source-clock 调度代理**，不保证语义对齐，也不是端到端墙钟时延。参考长度及输出长度仍会影响它。对于相同译文，推迟任意输出单位不会降低该值；对于不同长度、不同内容的译文，必须结合质量看。

质量门控后的成本：

$$
c_i=\begin{cases}
\min(d_i,10),&e_i\le5\\
10,&e_i>5
\end{cases}
$$

- 质量不合格，立即输出也不能靠低时延拿到成本优势。
- 质量合格，等待只有换来足够的质量收益，才可能抵消增加的时延成本。
- 同样质量和长度，成本未饱和前更早输出的候选优先。
- 10 秒处饱和，因此本项不区分 10 秒和 20 秒；仍保留原始 d 及超时比例监控。这是本次采用 HPO 式截顶的明确限制。
- 配置 `latency_budget_seconds=3` 用于 `latency_pass/feasible` 诊断；**它不是上述成本公式的截顶值，也不是承诺所有输出都能在 3 秒以内**。

5 分和 10 秒来自 HPO 配置，尚不是在本数据、服务版本与时延口径上校准出的最优值。尺度下限是本版的实验起点。

## 4. 完整 G8 分项归一化与最终奖励

先收齐同一音频 G8。只有格式有效、非空且非严重循环的候选，进入质量和时延统计集合 V。用总体标准差，分别计算：

$$
z^q_i=\operatorname{clip}\left(\frac{q_i-\mu_q}{\max(\sigma_q,0.5)},-2,2\right)
$$

$$
z^c_i=\operatorname{clip}\left(\frac{c_i-\mu_c}{\max(\sigma_c,0.5)},-2,2\right)
$$

质量尺度下限单位是 MetricX 原始错误分，时延尺度下限单位是秒。这个下限避免把极小分差放大成完整强信号；截断限制单个离群候选的影响。它们是 QALCO 的适配，不是 HPO 原实现。

$$
R_i=z^q_i-0.5z^c_i-0.08\,\mathbf{1}[e_i\le5]C_i-0.20P_i
$$

时延系数 0.5 作用于组内归一化后的成本，不是每秒固定扣分，也不是实际奖励或梯度中恒占 50%。质量相同且成本未饱和时，鼓励更早输出；加权取舍不保证每个更高质量候选都排在前面。

P 沿用现有局部机械循环检查；不按普通句号、短句或 FLEX 语义锚点重复一概处罚。C 为质量约束的长度成本：

1. 找本组最小错误分 `e_best`，只收集 `e <= min(5, e_best + 0.75)` 的候选。
2. 至少有 3 个候选才建立长度基准，否则该组 C 为 0。
3. 基准 L 是这些候选文本单位数的 25% 分位数，允许冗余量 `s=max(2,0.05L)`。
4. `C=clip((N-L-s)/max(L,20),0,1)`，并由质量门控决定是否实际计入。

不会用 WAIT 次数或 DIRECT 数量推断简洁性。长度成本很轻，也不能证明短译没有漏译：如果 MetricX 同样高估某些漏译候选，该限制仍可能失效，需要真实候选核查。

失败候选使用：

$$
R_{\rm failure}=\min_{i\in V}R_i-0.25
$$

完整 G8 全部失败时统一 -0.25。框架随后只减组均值：

$$
A_i=R_i-\frac{1}{8}\sum_{k=1}^{8}R_k
$$

启动参数为 `scale_rewards=none`、`advantage_estimator=grpo`、`kl_in_reward=false`，保留 loss 内 KL beta=0.01。**不再次按总奖励标准差相除。**跨卡仍沿用现有两次同步汇集及异常广播。

失败候选低于本组所有可评分候选，是减少模型完全不输出的策略选择，不等于宣称“任何胡说都好于沉默”。八条都失败时没有组内差异，A 全为 0，不能凭空创造正确译文的学习信号。所有可评分候选都低质量时，时延项相同，主要按质量区别；其中较好的候选仍可能有正优势，日志 `low_quality_positive_advantage` 会标记这种情况。

## 5. 一组可复算示例

以下是**人工构造的数值算例，不是服务实测或真实翻译结果**。六个有效候选长度均为 100，重复成本为 0，另两个全程 WAIT；不足三个质量接近最优的候选，因此长度成本为 0。

| 候选 | 原始错误分 e | 原始时延 d | 门控成本 c | 最终 R | 中心化优势 A |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 2 | 1.7602 | 2.4075 |
| 2 | 2 | 3 | 3 | 1.2478 | 1.8951 |
| 3 | 3 | 4 | 4 | 0.7354 | 1.3826 |
| 4 | 4 | 5 | 5 | 0.2229 | 0.8702 |
| 5 | 7 | 0.5 | 10 | -1.6271 | -0.9798 |
| 6 | 9 | 2 | 10 | -2.3392 | -1.6919 |
| 7，全 WAIT | null | null | 不计算 | -2.5892 | -1.9419 |
| 8，全 WAIT | null | null | 不计算 | -2.5892 | -1.9419 |

候选 5 即使很快，也因质量未达标不能用 0.5 秒作为成本。候选 1 的正优势最大，但训练不是只复制第一名；所有有正/负优势的真实输出 token 都会通过序列级重要性权重参与更新，包括 WAIT/FLEX/DIRECT 的选择。模型不会被保证下一次逐字输出第一名。

## 6. 启动与原版本隔离

在服务器原已验收的 Docker 和环境中运行，无新增必装包。训练卡仍不加载质量模型。同步整个项目目录，避免只复制配置遗漏 v5 的评分与跨卡代码。

虚拟环境示例统一为 `/data/yb/vens`；如果服务器已在 `/data/yb/venvs` 等位置配置好可用环境，直接激活该环境即可，不必重建。

先按[新格式数据完整指南](GSPO_v4_新格式数据_完整运行指南.md)或[旧格式数据完整指南](GSPO_v4_旧格式数据_完整运行指南.md)用新版 `01_prepare.sh` 生成清单。以下 `/data/yb/Code/FLEX/models/new-sft-merged` 是**示例路径，必须替换成使用最新短提示词训练的实际 SFT 合并模型**；不能默认使用历史 4410 模型。输出目录使用新的实验目录。

```bash
cd /data/yb/Code/FLEX/FLEX-GSPO
source /data/yb/vens/flex-gspo-online/bin/activate

bash 07_train_qalco_v5.sh \
  /data/yb/Code/FLEX/models/new-sft-merged \
  /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v5_shortprompt_1105/train.jsonl \
  /data/yb/Code/FLEX/outputs/gspo_v5_metricx_batch2_shortprompt_g8_b32_3000_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --train-iters 3000 \
  --save-steps 500
```

八卡命令如下（同一份代码，G8/micro2 下使用 B64）：

```bash
cd /data/yb/Code/FLEX/FLEX-GSPO
source /data/yb/vens/flex-gspo-online/bin/activate

bash 07_train_qalco_v5.sh \
  /data/yb/Code/FLEX/models/new-sft-merged \
  /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v5_shortprompt_1105/train.jsonl \
  /data/yb/Code/FLEX/outputs/gspo_v5_metricx_8gpu_g8_b64_3000_001 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --group-size 8 \
  --global-batch-size 64 \
  --micro-batch-size 2 \
  --train-iters 3000 \
  --save-steps 500
```

八卡训练 TP2/EP2、DP4，vLLM 为两个 TP4 引擎组；不是一张卡对应一条轨迹。四卡 B32 每次通常生成 4 条音频各 8 条轨迹，八卡 B64 则为 8 条音频各 8 条轨迹，3000 步的样本量随之增加。监控八卡实验时，将下面两条监控命令的输出根目录替换成上述八卡输出目录。

在训练命令末尾加 `--dry-run` 可查看最终配置、评分提供方、卡数与 batch，不加载模型、不创建训练输出、不调用质量服务。

沿用 TP2 / EP2、vLLM TP4、LoRA rank8/alpha32、lr1e-6、beta0.01、epsilon0.0003/0.0004、temperature1、steps_per_generation1、max_length8192、每轮max_new_tokens128、max_turns65、vLLM memory0.5/max_num_seqs32、mm_processor_cache_gb0、sleep2、offload、recompute、同步补丁和阶段计时。没有增大 batch、切换框架或升级 CUDA/训练依赖。

若自行安排两步 GPU 验证，使用另一个新输出目录，尾部覆盖 `--train-iters 2 --save-steps 2` 即可；它不是正式训练的自动启动条件。

**启动不运行 MetricX/COMET 的语义排序测试，也不调用评分 API 做前置评分。**仍检查本地配置、任务路径、框架接口与恢复身份；训练期间真实评分服务超时或返回无效数组会报错，不用假分数继续更新。这与用户自行决定质量验证时机相容。

v4 的公式、配置及 06 入口保留；03/04 仍是旧 v2，05 为 v3。v5 用新目录从 SFT 起跑，不把旧 COMET 日志当 MetricX 重放。源码是 run identity 的一部分：若要精确恢复旧训练，请使用当时的 Git 提交和配置，不关闭身份核验。

## 7. TensorBoard 与验收依据

在可访问输出目录的主机侧监控环境，终端一：

```bash
cd /data/yb/Code/FLEX/FLEX-GSPO
/data/yb/vens/flex-tensorboard-host/bin/python3 Scripts/monitor_training.py \
  /data/yb/Code/FLEX/outputs/gspo_v5_metricx_batch2_shortprompt_g8_b32_3000_001 \
  --watch --poll-seconds 10 \
  --dataset /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v5_shortprompt_1105/train.jsonl
```

终端二：

```bash
/data/yb/vens/flex-tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data/yb/Code/FLEX/outputs/gspo_v5_metricx_batch2_shortprompt_g8_b32_3000_001 \
  --host 0.0.0.0 --port 6006
```

重点看原始 `metricx_error/mean`（越低越好）、`latency_seconds/mean`、质量通过率、空译/格式错误/循环比例、`group_all_invalid`、质量与时延的组内标准差、`low_quality_positive_advantage`，并抽查完整 G8 译文与逐轮输出。TensorBoard 会按本地回调/进程汇总，应区别完整组级诊断与全局训练均值。

归一化后的 reward 不是可跨样本比较的绝对质量，不要求始终为正或单调上升；换了音频难度也会波动。监控 loss、KL、梯度和耗时判断数值/运行状态，固定验证音频对比 SFT 才能判断实际提升。

MetricX 本身也可能误判论元方向、否定或漏译，机器参考可能有误，且完整文本评分不能证明每轮只使用当时听到的信息。v5 保留 `prefix_support_checked=false`，不声称修复全部语义缺陷、短句体验或实现零故障。本地测试覆盖公式、服务契约模拟、跨进程分组模拟和启动参数；没有服务器四卡训练实测。

2026-11-05 完整仓库整合后，本地 CPU 回归共 500 项，499 项通过，1 项音频重采样测试因本机 SciPy/NumPy 兼容问题跳过。设置 `FLEX_SWIFT_SOURCE` 指向已核对的框架源码后，源码契约检查也已运行。实际 Bash 入口的四卡/八卡 dry-run 均成功，确认 v5/reference、G8、B32/B64、3000 步/500 步保存及不自动请求质量服务。Windows 的 Bash 检查仅用临时 python3 包装调用本机 Python，未修改服务器环境；没有访问内网评分服务，也没有运行服务器 GPU 训练。

当前时延权重为 0.5。运行中的进程不会因本地配置文件修改而自动换用新奖励，应在新实验启动时使用并核对 launch 配置。
