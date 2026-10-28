# QALCO v4：分项缩放与有界失败间隔

更新日期：2026-10-28。独立入口：`06_train_qalco_v4.sh`；配置：`configs/qalco_v4.json`。v2/v3 文件继续用于历史复现。新一轮从已经验收的 SFT 合并模型开始，不能把旧优化器状态接到新奖励上当作精确续训。

## 1. 此版本针对什么问题

旧 v3 的完整 G8 计分和奖励公式复算没有发现算术错误。但审查实际轨迹发现：

1. 原始质量和门控时延的变化尺度不同。直接相加之后再统一除标准差，会掩盖各项的实际作用。
2. 混合空译组中，固定 −4 会显著拉低均值；部分低质量译文因此获得较强正优势。
3. 小评分差不应自动被放大到单位量级，尤其 COMET 分差并非人工判断的确定性差距。
4. COMET 对否定范围等局部错误存在漏检。归一化不能补出缺失的语义证据。

v4 修改前面三项的数值处理，增加第四项的启动前对照检查。仍只使用已有 COMET22 服务；没有新评分模型、没有额外 GPU 占用，也没有把参考译文送给 actor。

不能保证所有误译、漏译或幻觉都被识别。没有因为某条轨迹出现过 TITLE、某个实体或某种字母而加入针对该例的奖励规则。也不按短句数量、句号数量或 FLEX 数量直接扣分。

## 2. 计分单位和处理顺序

最小奖励对象是**同一条音频的一条完整输出轨迹**，包括音频轮次和一次 EOF 收尾轮。每个音频采样 8 条轨迹；global batch 32 对应每次生成 4 条音频、共 32 条轨迹。这不限制整个数据集只能有四条音频。

1. 在线采样当前策略的完整轨迹。
2. 检查协议、音频覆盖、EOF、空译、异常截断和严重循环。
3. 将每条可评分轨迹的新增译文拼接，只调用一次完整译文的 COMET 评分；相同译文按已有逻辑去重请求。
4. 跨 rank 只交换标量元数据，按框架真实顺序组成完整 G8。
5. 在完整 G8 内计算质量/时延矩、长度锚点和最终奖励。
6. 框架对 G8 奖励减均值，得到轨迹优势；沿用 sequence-level importance sampling 和 clipping 更新模型。

单轮 WAIT 没有文本，不单独调用 COMET。WAIT 会改变后续译文何时释放，从而影响整条轨迹的时延代理值。FLEX 和 DIRECT 都按其实际译文与释放时间参与评分；标签本身没有奖励配额。

## 3. 具体公式

令 $E$ 为完整 G8 中协议有效、输入和 EOF 完整、非空、无严重循环且具有真实 COMET22 分数的候选集合。质量 $Q_i$ 是 COMET22 的原始分值，不是准确率。未评分的空译和异常轨迹保持 `quality=null`，不补成 0。

门控保留绝对质量要求，默认在 0.80 到 0.85 之间平滑开启：

$$
w_i=\operatorname{clip}\left(\frac{Q_i-0.80}{0.85-0.80},0,1\right).
$$

设 $d_i$ 为 source-clock mean positive lag，预算 $B=3$ 秒：

$$
D_i=\min\left(2,\;0.05\frac{d_i}{B}+0.20\left[\frac{d_i}{B}-1\right]_+^2\right),
\qquad D_i^g=2(1-w_i)+w_iD_i.
$$

质量不超过 0.80 时，门控时延成本固定为 2；单纯提前输出或缩短译文不会获得这一项的改进。此值不是等待秒数，不能和 3 秒预算混为一谈。时延仍是调度代理，不能证明每个词都受到已听源语支持，也不包含服务推理耗时。

只在 $E$ 内分别求总体均值和总体标准差；分母设下限，缩放后截到 ±2：

$$
z_i^Q=\operatorname{clip}\left(\frac{Q_i-\mu_E(Q)}{\max(\sigma_E(Q),0.03)},-2,2\right),
$$

$$
z_i^D=\operatorname{clip}\left(\frac{D_i^g-\mu_E(D^g)}{\max(\sigma_E(D^g),0.10)},-2,2\right).
$$

单个可评分候选的标准差为 0，两个归一化分量也为 0。它不能凭空获得“组内质量优胜”的证据。

对可评分候选：

$$
S_i=z_i^Q-0.25z_i^D-0.08w_iC_i-0.20P_i,\qquad R_i=S_i.
$$

$C_i$ 仍使用同音频、质量相近候选的长度锚点：质量达到 `max(0.80, best_Q−0.03)` 的候选至少 3 条才建立锚点；取长度第 25 百分位，容忍 `max(2个单位, 5%锚点长度)`，超出部分除以 `max(锚点,20)`，截到 [0,1]。中文单位主要是汉字，非生成 tokenizer token。

$P_i$ 沿用保守的相邻循环检测，不是语义冗余模型；普通重复实体和必要语义锚点不直接处罚。严重循环由失败分支处理。漏译可能也很短，因此不能让长度无条件优先于质量。

**0.08 只表示保守的弱辅助。** Q 改成分项缩放后，原来的 0.08 不再等于 v3 中的相对简洁强度。新旧排名必须离线检查，不为恢复某个示例的原排序任意放大简洁权重。

失败轨迹使用组内相对间隔，不伪造语义分：

$$
R_i=\begin{cases}
\min_{j\in E}S_j-0.25,&E\ne\varnothing,\ i\notin E,\\
-0.25,&E=\varnothing.
\end{cases}
$$

这样失败候选仍低于所有可评分候选，但不会固定用 −4 把均值拉到很远。这里的“可评分”不是“语义正确”，严重误译若被 COMET 漏检仍可能受到错误偏好。

框架使用 `scale_rewards=none`，但仍减完整 G8 均值：

$$
A_i=R_i-\frac{1}{8}\sum_{j=1}^{8}R_j.
$$

这是保留 GSPO sequence-level 更新、修改奖励缩放和失败基线的实验版本；不是照搬原论文全部公式。v4 不再对最终奖励除一次标准差。

原始诊断效用 `utility_raw = Q−D_g−0.08wC−0.20P` 仅用于日志，不输入优化器。它也不是 v3 的原始奖励，因为门控阈值和质量不足项不同。

## 4. 必须正确理解的边界

- **仍可能有低质量正优势。** 一条质量很低的非空候选和七条空译组成 G8 时，非空候选可能得到 +0.21875，空译各 −0.03125。v4 减弱了这种偏好强度，没有把它消除。
- **全八条失败时，优势全为零。** 这一组没有相对排序信息；没有通过自动重试制造“模型已经学会不 WAIT”的假象。记录该比例和音频 ID，再检查数据及模型能力。
- **对 CASE 2 的关键排序没有数学保证。** 如果错误否定表达同时获更高 COMET、更低时延，任何沿用这些分数方向的单调奖励都无法自动知道它错了。
- **源语或参考污染不能靠公式修复。** 已发现标为英文的源转写实际含其他语言；需结合音频审查、清理。`lang=en` 只是元数据，当前 prepare 不会自动证明整段是英文。v4 不额外安装语言识别模型，也不硬编码排除某个样本 ID。
- EOF、音频时序、重复和格式检查属于确定性检查；没有加入未经验证的“短句必错”“句号必错”“拉丁字母必错”规则。

## 5. 服务与启动前检查

沿用 `http://10.185.1.72:6668/22`，请求字段 `srcs/mts/refs`，返回字段 `score`，高分更好。

v4 自动读取 `configs/quality_regressions_v4.jsonl` 的独立合成对照，检查否定范围、论元方向、条件与数量、无依据添加。文件不包含用户实际案例或用于推高实际分数的词表。未通过将停止在 GPU 启动之前，输出是哪一对排错；不能通过删掉对照或强行跳过来声称质量问题已解决。通过只表示少量基本检查成功，不是完整校准。

服务连接错误、缺少分数、NaN、顺序错误都是基础设施异常，不能给 actor 记一个低分后继续训练。失败由原有两个无条件 collective 传播，避免健康 rank 提前进入下一阶段。

## 6. 先离线重算已有完整 G8

无需重新采样、音频或服务。以下使用已提供的诊断导出位置；输出目录必须为新目录：

```bash
cd /data06/yb/Code/FLEX/FLEX-GSPO
python3 Scripts/replay_qalco_v4.py \
  --input-jsonl /data06/yb/Code/FLEX/outputs/gspo_v3_g8_b32_3000_1027_001/diagnostics/reward_revision_20261028T113257_679795Z/selected_groups.jsonl \
  --config /data06/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v4.json \
  --output-dir /data06/yb/Code/FLEX/outputs/qalco_v4_replay_1028_001
```

输入必须是完整原始 JSONL，不能使用聊天中截短、四舍五入的 TXT 代替。输出 `summary.txt`、`summary.json`、`groups.jsonl`；包含新旧奖励、优势、排序、分项和完整译文，原始轮次用输入文件/行号/hash/候选序号定位。冻结的是旧 COMET 分数，因此只回答“相同轨迹换新公式会如何排序”，不能回答“新模型已经更好”。

## 7. 四卡参数与运行命令

| 参数 | 默认值 |
|---|---|
| 框架 | 已审计 MS-SWIFT Megatron RLHF，GRPO入口 + sequence importance |
| G / global batch / micro batch | 8 / 32 / 2 |
| steps per generation / num iterations | 1 / 1 |
| 总步数 / 保存间隔 | 3000 / 500 |
| 学习率 / 最低学习率 / warmup | 1e−6 / 1e−7 / 3% |
| beta / epsilon / epsilon_high | 0.01 / 0.0003 / 0.0004 |
| LoRA rank / alpha | 8 / 32 |
| 训练 TP / EP / PP / CP | 2 / 2 / 1 / 1 |
| vLLM TP / memory / max_num_seqs | 4 / 0.50 / 32 |
| max_length / 每轮 max_new_tokens / max_turns | 8192 / 128 / 65（含EOF） |
| 重计算 / offload / sleep | full uniform 1层 / model+optimizer / level2 |
| mm processor cache / KV prefix cache | 0 / 开启 |
| TensorBoard / 自有完整轨迹日志 | 开启 / 开启 |

不因可见显存的某个瞬间较低就增加 batch；前一轮峰值已接近80G。原先多卡终止同步和 vLLM 清缓存绕过均保留，不能据此承诺永远无故障。

先在训练容器内激活**该机器创建的**虚拟环境，完成两步真实 GPU 验收：

```bash
cd /data06/yb/Code/FLEX/FLEX-GSPO
source /data06/yb/venvs/flex-gspo-online/bin/activate
bash 06_train_qalco_v4.sh \
  /data06/yb/Code/FLEX/models/checkpoint-4410-merged \
  /data06/yb/Code/FLEX/datasets/gspo_online_prepared_v3_1027/train.jsonl \
  /data06/yb/Code/FLEX/outputs/gspo_v4_smoke_1028_001 \
  --gpu-ids 0,1,2,3 \
  --train-iters 2 \
  --save-steps 2
```

确认两次 optimizer 更新、COMET 评分、G8 定稿、checkpoint 保存完成后，正式训练使用另一新目录：

```bash
bash 06_train_qalco_v4.sh \
  /data06/yb/Code/FLEX/models/checkpoint-4410-merged \
  /data06/yb/Code/FLEX/datasets/gspo_online_prepared_v3_1027/train.jsonl \
  /data06/yb/Code/FLEX/outputs/gspo_v4_g8_b32_3000_1028_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --train-iters 3000 \
  --save-steps 500
```

不要从两步验收目录精确恢复为3000步；LR horizon已改变。正式轮从原SFT重新开始。同一正式轮中断后只能用其原生 checkpoint 加 `--resume-from`，保持代码、配置、数据和总步数一致。

## 8. TensorBoard 与验收

宿主机监控环境仍可单独使用，不必加载训练框架。若已经存在无需重复创建。终端一：

```bash
cd /data06/yb/Code/FLEX/FLEX-GSPO
/data06/yb/venvs/flex-tensorboard-host/bin/python3 Scripts/monitor_training.py \
  /data06/yb/Code/FLEX/outputs/gspo_v4_g8_b32_3000_1028_001 \
  --watch --poll-seconds 10 \
  --dataset /data06/yb/Code/FLEX/datasets/gspo_online_prepared_v3_1027/train.jsonl
```

终端二：

```bash
/data06/yb/venvs/flex-tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data06/yb/Code/FLEX/outputs/gspo_v4_g8_b32_3000_1028_001 \
  --host 0.0.0.0 --port 6006
```

v4 的 normalized reward 均值是组相关的训练信号，不能要求它持续上升、必须为正或和 v3 的均值直接比较。应同时看原始 COMET 质量、可行率、时延、空译率、格式错误率、全失败组、低质量正优势比例、KL/grad_norm 和速度。监控中的 rank局部统计仍不能代替完整G8统计。

需要在相同音频、相同chunk策略、相同推理设置下比较 SFT 与新checkpoint，人工检查漏译、否定范围、无依据添加、等待与自然度。未做固定验证，不能声称改完公式已提升实际翻译。

## 9. 来源与实现差异

- [GSPO原论文](https://arxiv.org/html/2507.18071v2)：使用序列级重要性比率和裁剪。当前保留框架该路径，奖励/优势缩放是本项目的实验改动。
- [HPO原论文](https://arxiv.org/html/2604.21045v1)：借鉴质量约束时延及质量/时延分别归一化；本项目的总体方差下限、裁剪、相对失败间隔、简洁项和COMET22尺度是独立设计，不能直接宣称复现其效果。
- 框架检查针对既有审计提交；本地CPU测试不等于服务器四卡GPU、网络、磁盘、长时间训练均已验收。

## 10. 本次本地验收记录

2026-10-28，全库 `unittest discover -s tests -v` 收集431项，430项通过、1项因本机缺少SciPy跳过音频重采样测试。新增归一化、完整G8集成、冻结评分回放、启动器及监控测试通过；实际启动参数dry-run通过。测试HTTP仅使用本地模拟服务，未访问服务器COMET或加载Qwen/GPU。

本机Windows沙箱首次阻止了asyncio内部回环socket创建，堆栈定位为标准库socketpair；在允许本机回环的测试环境中完整回归通过。没有为此改动生产训练逻辑。尚未进行真实四卡GPU验收，也没有声称新奖励已经改善固定验证集翻译。
