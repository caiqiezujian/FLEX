# FLEX-GSPO：QALCO 第一版强化学习

本目录只保留实际使用的后续流程：

**已有 G8 轨迹评分与校准 → 在线 GSPO → 模型导出。**

数据准备、转写和 SFT 多轨迹采样由独立的 `FLEX-GSPO-Data` 工程负责，不要在这里重复运行。

## 先读文档

- [强化学习方案详解](docx/QALCO强化学习方案详解.md)：研究目标、奖励公式、五条英中轨迹的完整数值示例、GSPO、四卡训练与研究边界。
- [验证与清理记录](docx/验证与清理记录.md)：当前组件、验证方式、恢复位置和未验证事项。

`docx/` 是按要求指定的文档目录，里面保存的是 Markdown，不是 Word 文件。

## 当前状态

- 训练代码、轨迹接入、奖励计算、断点恢复与导出代码已提供。
- **MetricX HTTP 客户端已按提供的接口实现**，支持 QE 和带参考评分，没有默认分数、没有本地模型回退。
- 公开配置不包含部署地址：先填写 MetricX 服务地址和部署标签，默认保持 QE 模式。内网连通性和真实评分仍需在服务器上验收。
- 单元测试通过不等于已经在真实 SFT checkpoint、4×A800 或真实服务上验收。
- 第一版是完整轨迹、一次 fresh rollout 对应一次更新的保守 GSPO；不是逐 WAIT 因果奖励。

## 目录

```text
FLEX-GSPO/
├── 04_score.sh                 # 已有多轨迹评分
├── 05_train.sh                 # 自包含四卡在线训练入口
├── 06_export.sh                # LoRA 合并导出
├── score_candidates.py
├── train.py
├── export_adapter.py
├── check_environment.py
├── check_metricx_service.py    # 只用合成文本检查真实评分接口，无需模型或音频
├── requirements.txt
├── configs/default.json
├── flex_gspo/                  # 真实运行所需的模块
├── tests/                      # 当前流程的回归测试
└── docx/                       # 方法与验证文档
```

编号 04 / 05 / 06 承接独立数据工程；本目录只负责评分、在线训练和导出。

## 1. 环境

建议使用独立的 Python 3.11 环境，不升级或破坏现有可用的 SFT / vLLM 环境。

先按服务器驱动安装 CUDA 版 PyTorch（版本要求见依赖文件），再执行：

```bash
python3 -m pip install -r requirements.txt
python3 check_environment.py
python3 -m unittest discover -s tests -v
```

音频读取需要 FFmpeg 在 PATH 中。训练使用 Transformers，不要求把已有 vLLM 0.16.0 降级。

所有 Shell 默认 `python3`，可用 `PYTHON_BIN` 覆盖。`HF_HUB_OFFLINE=1` 仅约束 Hugging Face 资源下载，不阻止 MetricX HTTP 请求。

## 2. 配置模型、任务和评分服务

编辑 `configs/default.json`：

```text
model_path                   已合并的 SFT checkpoint
manifest_path                原 train.tasks.jsonl
output_dir                   新训练输出目录
reward.metricx.service_url   你的服务基础地址，例如 http://METRICX_HOST:16999
reward.metricx.service_id    固定部署标签，例如 metricx24-deployment-v1
reward.metricx.mode          qe（默认）或 reference
```

训练端不需要 MetricX 权重或 MT5 tokenizer。质量评分只通过外部 MetricX 服务取得；训练端负责计算 Q / L / C，并组合为奖励 R。

评分函数在 `flex_gspo/metricx_service.py`：

```python
request_metricx_scores(
    source_text,
    candidate_texts,
    reference_text=None,
    *,
    mode="qe",
    service_url="",
    service_id="",
) -> list[float]
```

返回与候选一一对应的 MetricX 原始错误分 `[0,25]`，越低越好。单条包装是 `ServiceMetricX.score(...) -> float`。

已实现的调用契约：

| 模式 | 接口 | JSON 请求 |
|---|---|---|
| `qe` | `POST /metricx24/qe` | `{"srcs": [...], "mts": [...]}` |
| `reference` | `POST /metricx24` | `{"srcs": [...], "mts": [...], "refs": [...]}` |

两种模式都读取响应 `{"scores": [...]}`。请求数组按位置配对；同一音频的多个候选共享源转录，reference 模式还共享参考译文。客户端检查数量、有限数值与 `[0,25]` 范围，按服务返回顺序对应候选；服务若在内部重排同长度数组，仅凭这个响应格式无法自动发现，需人工对照或服务侧保证。

原始数据中的 `en.txt` 转成任务的 `source_text`，`zh.txt` 转成 `reference_text`，`key` 对应音频标识。**`mts` 是每条模型轨迹实际输出拼成的完整中文译文，不是 `zh.txt`。** QE 不向服务发送 `refs`，但中文参考仍用于训练端的长度预算与 LAAL 参考计数；切换 reference 模式需要重新校准奖励，不能认为机器参考一定正确。

客户端使用 `requests.Session`，关闭环境代理（`trust_env=False`）及重定向，连接/读取超时分别为 10/180 秒。连接错误、超时、HTTP 429/502/503/504 最多尝试两次，两次之间等待 1 秒；其余错误不重试。最终失败抛错停止，不用假分数替代。

服务响应没有模型版本字段，因此 `service_id` 只是被写入运行身份的本地部署标签，**不等于已验证的远端 checkpoint 版本**。请在服务侧固定模型、精度和部署版本；更换服务模型时同步更新声明并重新校准。当前提供的接口不需要鉴权；不要在配置地址中嵌入密钥。

公开配置中的 `service_url` 和 `service_id` 初始为空；请先填入实际值。它们缺失时，脚本会明确报错，不会向示例地址发请求。完成配置后，在训练服务器执行：

```bash
python3 check_metricx_service.py --mode both
```

脚本默认读取 `configs/default.json`，用同一合成英文的两条中文候选实际请求 QE / reference 接口，不加载 Actor，不读取音频，不启动训练。`--mode qe` 或 `--mode reference` 可单独检查；不传 `--mode` 则使用配置模式。可以用 `--config /path/to/config.json` 指定配置。成功只能证明该次请求与响应检查通过，不能证明评分能力、远端版本或训练已验收。

`reward.provider="qalco"` 是唯一的奖励配置入口；`reward.metricx.backend="service"` 指定外部文本评分服务。

## 3. 04：对已有轨迹评分

下列路径均为服务器示例；将轨迹目录替换成你全量 G8 采样的真实输出位置。每个 `trajectories.jsonl` 的同级 `run.json` 必须保留。

只读检查：

```bash
bash 04_score.sh \
  --tasks /data/flex/datasets/flex_gspo_prepared/train.tasks.jsonl \
  --trajectories /data/flex/datasets/flex_sft_samples_g8_all \
  --validate-only
```

结构检查不会请求服务。`quality_service_ready` 只反映客户端配置是否就绪；即使为 `true`，也没有验证远端网络、权重或真实评分。网络检查使用上面的 `check_metricx_service.py`。

服务接通后评分：

```bash
bash 04_score.sh \
  --tasks /data/flex/datasets/flex_gspo_prepared/train.tasks.jsonl \
  --trajectories /data/flex/datasets/flex_sft_samples_g8_all \
  --output-dir /data/flex/datasets/qalco_scores_v1
```

支持多个明确文件或采样根目录。`--limit N` 只限制评分，不重新采样；`--check-audio` 额外检查音频路径。

输出包括全部候选的分数、`summary.json`、各组原子结果和运行身份记录。没有 best-only 筛选。不覆盖已有目录；失败后已完成的组保留供检查，新运行使用新目录。

## 4. 奖励校准

默认质量门槛 `.85`、LAAL 预算 `3 秒`、长度倍率 `1.2`、权重 `.2/.1` 都是初值。

先检查实际候选是否被合理区分，再设置：

```json
"calibration_acknowledged": true
```

自然度若不能被质量分区分，不要用更强的长度惩罚假装已经解决。详细的校准对照和限制见方法文档。

## 5. 05：四卡在线 GSPO

只读预检：

```bash
NPROC_PER_NODE=4 bash 05_train.sh --validate-only
```

工程冒烟，前提是服务已经接通：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash 05_train.sh \
  --max-updates 2 \
  --output-dir /data/flex/outputs/qalco_smoke_001 \
  --allow-uncalibrated-reward
```

校准后正式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash 05_train.sh \
  --max-updates 1000 --epochs 1 \
  --output-dir /data/flex/outputs/qalco_train_001
```

可用 `--model-path`、`--tasks`、`--group-size` 覆盖配置。不要将 `scored.trajectories.jsonl` 传给 `--tasks`；在线训练读任务，重新生成当前策略的轨迹。

默认每卡一个音频、每个音频 G=8，全局每步 32 条轨迹。每卡放完整 BF16 Thinker，并非模型分片；真实显存峰值需服务器验收。

与 SFT 保持一致的特殊 token ID 为 WAIT / DIRECT / FLEX：151676 / 151677 / 151678。缺失则停止，不擅自扩词表。

日志按 rank 保存完整轨迹、奖励、优势和训练指标。默认每 25 步及正常结束保存 checkpoint。

## 6. 恢复训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
bash 05_train.sh \
  --resume-from /data/flex/outputs/qalco_train_001/checkpoint-00000025 \
  --max-updates 1000 --epochs 1 \
  --output-dir /data/flex/outputs/qalco_train_001_resume
```

恢复必须使用新的输出目录。模型、音频/任务、奖励、服务声明版本、G、卡数与训练代码必须匹配；允许延长总步数/epochs 和调整日志、保存频率。

代码身份不同的 checkpoint 不能直接续训；恢复时必须使用生成该 checkpoint 的一致代码和配置。

仅加载自己信任的 checkpoint。声明的服务版本不等于客户端已校验远端权重，实际核对须由服务侧部署记录保证；当前 `scores` 响应不提供版本证明。

## 7. 06：导出

```bash
bash 06_export.sh \
  --base-model /data/models/flex-sft-merged \
  --checkpoint /data/flex/outputs/qalco_train_001/checkpoint-00001000 \
  --output-dir /data/models/flex-qalco-merged \
  --device cpu
```

基座必须与训练时相同。CPU 合并需要较多内存，GPU 合并需要确认空闲显存。

默认仅保留文字同传所需 Thinker，但保存完整 Omni 外层结构；`--include-talker` 才保留语音输出模块，且要求基座对应权重完整。

导出后再用原来的离线/流式推理系统验收，不以“文件保存成功”代替真实推理检查。
