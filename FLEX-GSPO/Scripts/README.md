# 检查、审计与诊断脚本

以下命令从 `FLEX-GSPO` 项目根目录执行。脚本统一放在 `Scripts/`，数据准备与训练入口仍在根目录，自动化测试仍在 `tests/`。从其他目录运行时可使用脚本绝对路径，例如 `python3 /data/yb/Code/FLEX/FLEX-GSPO/Scripts/check_environment.py --cpu-only`；输入、输出路径仍应指向实际文件。

| 入口 | 用途与主要输入 | 服务与输出 |
|---|---|---|
| [Scripts/check_environment.py](check_environment.py) | 检查解释器、依赖与训练能力；可用 `--model` 检查模型目录，`--cpu-only` 跳过 GPU 检查 | 不安装依赖、不下载或加载模型 |
| [Scripts/check_comet22_service.py](check_comet22_service.py) | 独立手工诊断；`--config` 指定 COMET22 配置，可用 `--cases-jsonl` 提供人工核查成对样例 | 请求 COMET22，检查响应、语义排序及批量顺序；不作为训练启动条件 |
| [Scripts/check_metricx_service.py](check_metricx_service.py) | `--config` 与可选 `--mode qe/reference/both` | 请求 MetricX，仅供历史或可选诊断 |
| [Scripts/audit_reward_candidates.py](audit_reward_candidates.py) | `--run-dir`、`--dataset`、`--config`、`--output`，可按 `--pids` 选旧运行 | 请求 COMET22 重新评分；可加 `--compare-metricx`；输出新 JSONL 和摘要 |
| [Scripts/replay_reward_audit.py](replay_reward_audit.py) | `--input-txt` 按原顺序传入完整逐轮 TXT 片段，`--config` 指定 v3 配置 | 冻结已有 COMET22 分数，无 HTTP；输出到新目录 |
| [Scripts/monitor_training.py](monitor_training.py) | 运行目录、可选 `--watch` 和 `--dataset` | 读取已有日志，生成 TensorBoard 事件与监控摘要；不请求评分服务 |
| [Scripts/collect_training_status.py](collect_training_status.py) | 运行目录、可选 `--dataset`、`--reward-config` 和 `--output` | 生成文本快照；标准库读取，不请求评分服务；可选 `--gpu-ids` 查询占用 |
| [Scripts/summarize_training_history.py](summarize_training_history.py) | 运行目录、`--native-log`、可选 `--launch-file` 和 `--block-size 50`；`--native-only` 只汇总训练日志 | 扫描全部已完成步骤；核对运行身份和完整 G8，输出逐步 CSV、分段摘要与奖励分项；标准库读取，不请求评分服务 |
| [Scripts/collect_reward_revision.py](collect_reward_revision.py) | 运行目录、明确指定的 `--native-log`、`--launch-file`、`--dataset`；默认 `--groups 20 --block-size 50` | 新版奖励设计前采证：全程 G8/空译/EOF/分项方差统计，分层导出最多 160 个候选及全部原始轮次；不改训练、不请求服务 |
| [Scripts/collect_brevity_samples.py](collect_brevity_samples.py) | 运行目录、`--dataset`、`--output`，可选 `--include-increments` | 导出观测候选和非空增量；不请求评分服务，不保证完整 G8 或全部 WAIT 轮 |
| [Scripts/analyze_training_run.py](analyze_training_run.py) | 运行目录、必需的 `--native-log` 和 `--output-dir`，可选 `--reward-config` | 停止后的离线日志审查，不请求评分服务；公式复核受原运行身份约束 |
| [Scripts/relocate_prepared_data.py](relocate_prepared_data.py) | 旧 prepared `--input-jsonl`、已复制 WAV 的 `--audio-dir`、新 `--output-dir`，可用 `--workers 16` | 核验 WAV 元数据，重建两处音频引用；不复制音频、不请求服务、不覆盖旧清单 |

各入口可用 `python3 Scripts/<脚本名>.py --help` 查看参数。训练、手工服务诊断与监控的完整示例见[项目 README](../README.md)、[QALCO v4 训练说明](../docx/QALCO_v4训练说明.md)和 [QALCO v3 历史训练说明](../docx/QALCO_v3训练说明.md)。

`check_comet22_service.py` 和 `configs/quality_regressions_v4.jsonl` 保留为独立手工工具及样例。训练启动器不自动运行该脚本，不以它的结果作为启动条件，也不把该样例文件纳入 `run_identity` 恢复身份；验证由用户自行安排。检查通过不等于 COMET 误判已消除，未通过也不改变现有 COMET22 奖励公式。训练运行时仍校验服务响应；数据、框架和恢复一致性检查继续保留。

`collect_reward_revision.py` 只支持已验证的 G8、每次生成一次更新、标量 QALCO 组归一化配置。按 launch 的 session/run 筛选奖励，按已完成训练步核对全部 rank 与完整组；有缺失或重复的奖励步骤会整步排除并报告。它读取全部已完成步骤，选取整组全 WAIT、部分空译、异常输出、质量接近但长度有差异的组，并以固定哈希顺序补充不同音频。样例为诊断抽样，不是随机验证集；质量缺失保持缺失，省略号和短句计数不直接等同错误。原始音频、模型和训练环境均无需加载。

默认输出到运行目录下新的 `diagnostics/reward_revision_<时间>/`。先审查 `summary.txt`，再查看 `review_part_*.txt`（每份最多 5 个完整 G8）；`selected_groups.jsonl` 保存原始候选与对应任务，供后续离线奖励对照。另有逐步 `all_steps.csv`、全部组的 `group_statistics.jsonl`、空译任务统计 `empty_tasks.jsonl`，以及启动配置和日志快照来源 `provenance.json`。不会覆盖已有报告。

迁移后的清单生成、Docker 路径、环境检查和完整训练命令，见 [data06 迁移与 v3 训练](../docx/data06迁移与v3训练.md)。

## 离线奖励回放

`Scripts/audit_reward_candidates.py` 会把所选原文、参考和译文发送到配置的评分服务重新评分。`Scripts/replay_reward_audit.py` 只读取已有 TXT 与配置，用已导出的 COMET22 分数回放 v3 奖励，不读取音频、不加载模型、不请求 HTTP、不生成新轨迹或修改训练日志。

回放输入必须是此前完整逐轮导出包装器生成的 TXT：每个局部组包含同一回调、同一任务的 8 个候选，保留候选身份、有效评分和所有音频轮（包括 WAIT）及 EOF。当前 `Scripts/collect_brevity_samples.py --include-increments` 仍省略正常 WAIT，且不强制完整 G8，不能直接用其普通导出替代完整逐轮输入。缺失或不支持的组会整组跳过，不缩成 G7，也不补造评分；局部回调导出不能认证原训练的分布式 GSPO 全局分组。

下面的输入文件名仅示意：替换为已有完整逐轮导出的实际路径，并按原始连续顺序列出所有片段。片段可以在组内接续，但不能拆断一行或 JSON 对象，不能漏掉前文的组头或候选头。

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO
python3 Scripts/replay_reward_audit.py \
  --input-txt /path/to/full_turn_part_001.txt /path/to/full_turn_part_002.txt \
  --config /data/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v3.json \
  --output-dir /path/to/new_reward_replay_001
~~~

`--output-dir` 必须是尚不存在的新目录。结果包含 `replay.json` 和 `comparison.txt`，记录输入及代码哈希、奖励配置、成功回放结果和整组跳过原因；至少一组成功时退出码为 0，全部跳过或输入错误时为 2。这是固定旧候选与固定质量分数上的奖励对照，不能证明新策略或翻译质量提升。
