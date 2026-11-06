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
| [Scripts/resume_dp_migration.py](resume_dp_migration.py) | 原八卡 v4 的 `--launch-json`、原生 `--resume-from`、新 `--output-dir`、四个 `--gpu-ids` | 默认使用当前代码继续 COMET22 v4，不需要 Git；源码差异只记录，恢复原生优化器等状态，保持原全局 batch 和训练总步数 |

各入口可用 `python3 Scripts/<脚本名>.py --help` 查看参数。训练、手工服务诊断与监控的完整示例见[项目 README](../README.md)、[QALCO v4 训练说明](../docx/QALCO_v4训练说明.md)和 [QALCO v3 历史训练说明](../docx/QALCO_v3训练说明.md)。

`check_comet22_service.py` 和 `configs/quality_regressions_v4.jsonl` 保留为独立手工工具及样例。训练启动器不自动运行该脚本，不以它的结果作为启动条件，也不把该样例文件纳入 `run_identity` 恢复身份；验证由用户自行安排。检查通过不等于 COMET 误判已消除，未通过也不改变现有 COMET22 奖励公式。训练运行时仍校验服务响应；数据、框架和恢复一致性检查继续保留。

`collect_reward_revision.py` 只支持已验证的 G8、每次生成一次更新、标量 QALCO 组归一化配置。按 launch 的 session/run 筛选奖励，按已完成训练步核对全部 rank 与完整组；有缺失或重复的奖励步骤会整步排除并报告。它读取全部已完成步骤，选取整组全 WAIT、部分空译、异常输出、质量接近但长度有差异的组，并以固定哈希顺序补充不同音频。样例为诊断抽样，不是随机验证集；质量缺失保持缺失，省略号和短句计数不直接等同错误。原始音频、模型和训练环境均无需加载。

默认输出到运行目录下新的 `diagnostics/reward_revision_<时间>/`。先审查 `summary.txt`，再查看 `review_part_*.txt`（每份最多 5 个完整 G8）；`selected_groups.jsonl` 保存原始候选与对应任务，供后续离线奖励对照。另有逐步 `all_steps.csv`、全部组的 `group_statistics.jsonl`、空译任务统计 `empty_tasks.jsonl`，以及启动配置和日志快照来源 `provenance.json`。不会覆盖已有报告。

迁移后的清单生成、Docker 路径、环境检查和完整训练命令，见 [data06 迁移与 v3 训练](../docx/data06迁移与v3训练.md)。

## 跨服务器八卡转四卡恢复：不安装 Git

直接使用当前 `FLEX-GSPO` 代码继续原 COMET22 v4 训练，不需要 Git、旧代码快照或额外源码目录。
默认使用本脚本所在的 `FLEX-GSPO`；也可以用 `--source-dir` 明确指定当前目录。
该目录包含 `train_megatron.py`、`swift_plugin.py`、`flex_gspo/`、`Scripts/` 和配置；不需要 `.git`。
更新 `Scripts/resume_dp_migration.py` 即可使用此行为。保留当前代码中的 `metricx_reward.py`，不必删除或回退任何 V5 文件。

代码文件新增或修改只记入迁移日志，不再以“源码必须逐字一致”为条件拦住续训。
v4 仍通过原配置选择 COMET22 和 v4 的归一化奖励分支；项目支持 v5 不会自动切换奖励。
继续核对原奖励配置、模型、数据、batch、学习率和训练总步数；恢复原生优化器、调度器、RNG 状态，不修改旧 launch 的指纹。
实际执行代码的新指纹和原指纹会分别保存，不声称这是旧源码的逐字重现。
必需的运行文件仍需存在，框架能力检查继续执行；COMET22 服务须可访问。
默认及 `--source-dir` 模式均不调用 Git 查找源码、不创建 worktree。
显式 `--repo` 或 `--source-commit` 才启用可选的旧版 Git 历史查找；本次直接续训不使用这些参数。

下面是已知原运行的路径示例。输出目录必须为空，GPU 4、5、6、7 必须实际可用。
环境仍使用原训练 Docker 中的兼容 Python 环境。原 SFT、训练 JSONL 和引用音频在新机需可访问；此入口不做路径重写，原模型大小/时间戳及数据身份检查继续执行。

```bash
cd /data/yb/Code/FLEX/FLEX-GSPO
source /data/yb/venvs/flex-gspo-online/bin/activate
mkdir -p /data/yb/Code/FLEX/outputs/logs

nohup python3 -u Scripts/resume_dp_migration.py \
  --source-dir /data/yb/Code/FLEX/FLEX-GSPO \
  --launch-json /data/yb/Code/FLEX/outputs/gspo_v4_comet22_g8_b32_3000_1104_001/launch_20261104T175951_477020Z.json \
  --resume-from /data/yb/Code/FLEX/outputs/gspo_v4_comet22_g8_b32_3000_1104_001/v0-20261105-020043/checkpoint-500 \
  --output-dir /data/yb/Code/FLEX/outputs/gspo_v4_resume500_4gpu_nogit_001 \
  --gpu-ids 4,5,6,7 \
  --master-port 29583 \
  > /data/yb/Code/FLEX/outputs/logs/gspo_v4_resume500_4gpu_nogit_001.log 2>&1 < /dev/null &
```

`--dry-run` 也支持无 Git 路径：检查目录/配置并打印源码差异及迁移计划后退出，不加载模型或访问评分服务。
正常启动在新输出目录的 `dp_migration.json` 中记录 `source_mode=current`、`source_policy=use_current_v4`、实际源码路径、新旧源码指纹和差异；未查询 Git 时 `source_commit` 为 null。
这不代表已验证 GPU 上的优化器重分片、offload 或训练成功；以实际恢复步数和更新日志为准。

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
