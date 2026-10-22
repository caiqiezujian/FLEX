# FLEX-GSPO：QALCO v2 在线同传强化学习

当前正式配置：**3000 次优化更新，每 500 步保存 checkpoint**。使用原 SFT 合并 checkpoint 起跑，4 张 A800 80GB；训练框架仍是 **MS-SWIFT 的 Megatron RLHF**，GSPO 采用 sequence-level importance sampling，不是把离线候选当成新策略。

这次改动针对上一轮三类实际问题：质量指标放过复读、末尾 WAIT 没有收尾机会、多卡同步及 vLLM 缓存清理卡死。**本地测试不等于真实四卡验收；不能保证零故障或保证质量提升。**

## 1. 这版改变与保留什么

- 质量主分数改为 **COMET22 带参考评分**，不接 COMET20。MetricX 仅保留历史审计及可选诊断对照，不参与默认新训练奖励。
- 严重循环复读进入独立失败分支，不能被质量分抵消；不按句号数量或普通 FLEX 语义锚点重复处罚。
- 保留原 SFT system prompt，音频全部输入后增加一次文本 EOF 观察和增量收尾。最终没有待译内容时仍可 WAIT；不强制 DIRECT、不重译历史、不输入转录或参考。
- 完整音频 N 轮 + EOF 1 轮；默认 max-turns=65，可容纳64个音频块。已有 prepared train.jsonl 可以直接使用，无须重切 WAV。
- 新奖励版本2在低质量区也计入有限等待成本；超时区不再在两倍预算处直接饱和。仍有质量/时延门槛，不是无约束加权求和。
- 保留现有 TP2/EP2、vLLM TP4、多轮全局完成同步、offload、sleep_level=2、full recompute、eager 和 log_completions=false。
- 默认 vllm_mm_processor_cache_gb=0，规避此前缓存清理卡死路径；KV prefix cache 仍保留。没有升级训练环境或修改 site-packages。
- 新日志包含 reward_version、run_id、session_id、COMET22、严重复读标记和 EOF 指标；旧日志仍可读取。

**COMET22 也有误判。默认 quality_floor=0.80、3秒时延预算和0.5秒 EOF 调度代理成本均是待校准的实验起点，不是准确率或已验证的最优值。先审真实候选，再决定是否调整。**

## 2. 入口

| 文件 | 用途 |
|---|---|
| 01_prepare.sh / prepare_data.py | 原始JSONL + WAV目录 → 在线任务清单 |
| check_comet22_service.py | COMET22连通、返回数组、基本语义排序与反序批量一致性检查 |
| audit_reward_candidates.py | 利用旧候选检查新评分与复读保护；不生成新轨迹、不伪造EOF |
| 02_train.sh / train_megatron.py | 通用 Megatron-SWIFT 启动器 |
| 03_train_3000.sh | G8/B32、micro1、vLLM0.4/16并发，3000步/每500步保存 |
| 04_train_throughput.sh | 相同目标，micro2、vLLM0.5/32并发，开启阶段计时 |
| monitor_training.py | 将奖励、EOF、性能和样例日志转成 TensorBoard |
| analyze_training_run.py | 停止后的离线日志审查，无评分API |
| configs/default.json | 当前 v2 奖励及服务配置 |
| docx/QALCO强化学习方案详解.md | 公式、边界与验收说明 |

原 03_train_4100.sh 已重命名为 03_train_3000.sh，不再作为入口。手动上传时请同步整个文件夹，避免只更新一两个 Python 文件。

## 3. 环境与输入

在**此前已验收的 Docker 内**激活虚拟环境，不能只看提示符判断环境：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO
source /data/yb/venvs/flex-gspo-online/bin/activate
python3 -c "import sys; print(sys.executable)"
~~~

应看到 /data/yb/venvs/flex-gspo-online/bin/python3。本次没有新增必装依赖；HTTP使用已有 requests，训练端不安装COMET模型。不要为了本次更新重新安装 torch、CUDA、vLLM 或 cachetools。

此前准备好的输入可继续用：
/data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl

只有尚未 prepare 才运行：

~~~bash
bash 01_prepare.sh \
  --input-jsonl /data/yb/Flex/datasets/1019/en2zh_train_gspo_1.jsonl \
  --audio-dir /data/yb/Flex/datasets/data_001 \
  --output-dir /data/yb/FLEX/datasets/gspo_online_prepared_v2 \
  --workers 16 \
  --skip-invalid
~~~

原始一行仍是 key、lang、en.txt、zh.txt，key对应key.wav。准备阶段保留原音频位置，输出虚拟分块；不制造离线G8轨迹。源文和机器参考仅用于奖励，不进入actor消息。Linux路径区分 Flex 与 FLEX，请确认真实路径。

## 4. 先检查 COMET22

默认完整URL为 http://10.185.1.72:6668/22，请求 srcs/mts/refs；禁用环境代理。支持有序数值数组或 {"scores":[...]}，实际响应若是其他顶层键，在配置 response_key 中填写。拒绝整体均分、嵌套数组、缺失项、NaN、字符串或百分制分数。

~~~bash
python3 check_comet22_service.py \
  --config /data/yb/Code/FLEX/FLEX-GSPO/configs/default.json
~~~

检查包括正确译文相对否定翻转、明显无关译文的排序。循环复读排名另作诊断，因为新复读保护独立于模型分数。模型身份只是操作者声明；服务没有提供截断信息时，客户端不能证明远端没有截断。检查失败不要用 transport-only 绕过正式训练；该选项只用于定位网络/响应协议问题。

可用 --cases-jsonl 加载人工核查的真实成对例子，格式：

~~~json
{"id":"reviewed-1","source_text":"We leave on Friday.","reference_text":"我们星期五出发。","preferred":"我们周五出发。","dispreferred":["我们星期一回来了。"]}
~~~

不要把错误机器参考当成真值；人工核查需要以音频为准。

## 5. 用旧训练结果检查新奖励盲区

以下脚本只读旧日志及数据清单，调用COMET22，不加载Omni、不读取WAV。输出文件必须不存在：

~~~bash
python3 audit_reward_candidates.py \
  --run-dir /data/yb/FLEX/outputs/gspo_throughput_4100_001 \
  --dataset /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl \
  --output /data/yb/FLEX/outputs/gspo_reward_audit_3000_001.jsonl \
  --config /data/yb/Code/FLEX/FLEX-GSPO/configs/default.json \
  --pids 1795291,1795292,1795293,1795294 \
  --limit 100
~~~

如需同时对照原MetricX，加 --compare-metricx。生成诊断JSONL及其 .summary.json。选样刻意覆盖空译文、疑似循环、协议错误及普通候选，不是随机评估；不能据此估计总体质量。旧轨迹没有新EOF，脚本**不会伪造一次收尾，也不会声称算出了其真实v2训练奖励**。

旧分析工具的公式重算需要对应旧运行原始配置。不能拿现在的v2 default.json去证明旧v1公式错误；未匹配旧launch哈希时重算会跳过。

## 6. 服务器短验收：独立目录

先检查解释器、服务和环境，再验收新EOF实际模型输出及保存。以下2步测试用于检查贯通，**不能证明100步或3000步稳定**：

~~~bash
bash 04_train_throughput.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/megatron_output/Qwen3-Omni-Flex/v3-20261016-172842/checkpoint-4410-merged \
  /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl \
  /data/yb/FLEX/outputs/gspo_comet22_smoke_001 \
  --train-iters 2 \
  --save-steps 2
~~~

确认每个rank进入reward和optimizer更新；检查EOF没有重复历史、全WAIT失败不被丢掉、原生checkpoint和合并导出实际存在。保存/恢复还需在服务器验证，HF merged目录不等于含optimizer/RNG的原生checkpoint。

2步测试的LR总周期与3000不同，不能将其作为3000步的无缝恢复点。正式训练从原SFT重新开始，使用新目录。

## 7. 正式训练：3000步，每500步保存

与上一轮实际使用的吞吐配置一致：G8 / global batch32 / micro2 / 每次生成更新1步，vLLM内存0.5、并发32。额外保留缓存规避与新EOF：

~~~bash
bash 04_train_throughput.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/megatron_output/Qwen3-Omni-Flex/v3-20261016-172842/checkpoint-4410-merged \
  /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl \
  /data/yb/FLEX/outputs/gspo_comet22_g8_b32_3000_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --steps-per-generation 1 \
  --train-iters 3000 \
  --save-steps 500 \
  --max-length 8192 \
  --max-new-tokens 128 \
  --max-turns 65 \
  --vllm-memory 0.5 \
  --vllm-max-num-seqs 32 \
  --vllm-mm-processor-cache-gb 0 \
  --performance-log on
~~~

命令末尾加 --dry-run 可只打印计划，无文件写入、无服务请求、无GPU。
正式保存点：500、1000、1500、2000、2500、3000；框架save_total_limit=3，不承诺六个历史checkpoint都永久保留。首次500步前没有周期checkpoint，这是用户指定的间隔；不能误以为50或100步会自动保存。warmup比例0.03，即90步。

每次生成计划4条音频×8候选=32条轨迹，3000步计划约12000次任务采样、96000条轨迹，不等于全量数据一个epoch，不保证任务不重复。EOF可能让步骤时间增加，不能承诺比旧轮次更快。微批次2若长轨迹接近显存上限，可在新实验回退03入口；不要中途随意改变精确恢复配置。

开始正式训练后仍须观察超过此前68步失效点，例如到100步，检查日志持续推进、缓存sleep/wake、奖励服务以及峰值显存。两步、100步通过均不等于终身无故障。不要自动杀掉长轮次；先看phase日志及各rank栈。

### 恢复

使用同一完整代码、奖励、数据、模型和3000步LR总周期，在原命令追加 --resume-from 指向实际的原生 checkpoint-500 等目录。必须有optimizer/RNG及分片，不是 checkpoint-500-merged。启动器验证run identity，改变奖励或EOF协议后不得接着旧v1 optimizer跑；这次必须新起一轮。

## 8. TensorBoard 与日志

另一个终端（训练环境或能读取同目录且装了TensorBoard的环境）：

~~~bash
python3 monitor_training.py \
  /data/yb/FLEX/outputs/gspo_comet22_g8_b32_3000_001 \
  --watch \
  --dataset /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl

python3 -m tensorboard.main \
  --logdir /data/yb/FLEX/outputs/gspo_comet22_g8_b32_3000_001 \
  --host 0.0.0.0 \
  --port 6006
~~~

monitor是常驻进程；TensorBoard在第三个终端运行。暴露端口时注意网络访问控制。也可仅读取原始 rewards/ 和 performance/ JSONL。

优先看：

- comet22_score/mean、quality_pass_fraction、feasible_fraction；
- empty_translation_fraction、severe_repetition_fraction、tier/degenerate_fraction；
- latency_seconds/mean、diagnostics/eof_text_fraction、eof_generation_seconds；
- actions/WAIT/FLEX/DIRECT_fraction **只统计音频轮**，eof_actions/*单列，避免正常收尾WAIT污染主策略比例；
- KL、grad_norm、reward_std、frac_reward_zero_std、非有限值；
- generation_seconds、offload/reload、weight_sync 与原始 train_speed。

final_wait_fraction可能是“已译完后的正常EOF WAIT”，不等于整条空译文。各rank回调汇总不是完整全局G8组；不要用局部8条就断言全部候选相同。run_id/session_id用于避免重启日志混淆。

固定验证音频对比SFT与checkpoint，抽查事实/数字/否定、漏译、自然性、复读、输出时机。reward上升或loss下降都不能单独证明翻译变好；COMET22若用于训练，也不能再当完全独立的效果证据。

## 9. 测试与范围

~~~bash
python3 -m unittest discover -s tests -v
~~~

CPU测试覆盖真实WAV切片、本地HTTP、EOF原始token/mask/logprob、四rank不同结束长度模拟、退化保护、奖励数学和旧日志兼容。部分源码契约测试用本地审计过的SWIFT源码，可设置FLEX_SWIFT_SOURCE；没有该源码的机器会跳过。

本轮不改训练框架安装、不删用户数据、音频、权重或旧实验。当前工作区未做GPU实训或调用内网COMET；真正的服务响应格式、模型身份、长文本截断和新EOF策略效果必须服务器验收。VAD是后续部署触发边界的机制，不代替这里的训练收尾协议。

详见 [QALCO v2方案](docx/QALCO强化学习方案详解.md) 和 [训练监控](docx/提速与训练监控.md)。
