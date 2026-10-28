# 迁移到 /data06 后，从 SFT 启动 QALCO v3

本流程用于**从原 SFT 合并模型开始一轮新的 v3 训练**，不恢复旧优化器或旧训练步数。使用现有四卡 A800 配置：G8、global batch 32、micro batch 2、3000 步、每 500 步保存。训练过程中在线生成候选，不需要搬运以前的离线 G8 轨迹才能训练。

## 1. 目录约定与实际需要修改的内容

建议统一成以下目录。这里的路径必须也是**训练 Docker 内能够访问的路径**。

```text
/data06/FLEX/
├── code/FLEX-GSPO/                        # 代码；仓库根目录是 code/
├── datasets/
│   ├── 1019/en2zh_train_gspo_1.jsonl       # 原始 key / lang / en.txt / zh.txt
│   ├── data_001/                         # 全部 key.wav
│   ├── gspo_online_prepared_clean/       # 复制过来的旧 prepared 清单，留作输入
│   └── gspo_online_prepared_v3/          # 迁移脚本生成的新清单
├── models/checkpoint-4410-merged/        # 原 SFT 合并模型完整目录
└── outputs/                             # 新实验结果
```

| 对象 | 迁移处理 |
|---|---|
| 原始 `en2zh_train_gspo_1.jsonl` | 现有格式通过 `key` 对应文件名，没有音频绝对路径，不需要替换文本 |
| WAV 文件 | 复制到 `data_001/`，文件名保持 `key.wav` |
| 已准备好的 `train.jsonl` | 更新 JSON 字符串 `flex_task` 内的 `audio_path`，并重建 `audios` 中编码后的虚拟分块引用；同时核验目标 WAV 元数据并刷新修改时间 |
| Python / Shell 源码 | 入口已使用 `SCRIPT_DIR` / `PROJECT_ROOT` 定位，无须全局替换代码中的 `/data`；调用命令显式使用新地址 |
| COMET22 配置 | 文件位置随代码迁移；服务 URL 只有服务地址变化时才改，并在新环境中检查网络可达性 |
| 旧训练日志、checkpoint、审计报告 | 可以归档复制，保留其原始记录；本次新训练不读取旧 optimizer/RNG，也不改写旧日志中的历史地址 |

不要对整个 JSONL 执行 `sed` 全局替换：`audios` 的 `flexaudio:v1:` 内容经过编码，普通字符串替换会遗漏；转录和参考中的普通文本也不应该被修改。

## 2. 先复制代码、音频、清单和 SFT 模型

下面命令适用于**同一台主机同时能看到旧目录和 `/data06`** 的情况。Linux 区分 `Flex` 与 `FLEX`；先确认以下源地址存在。若换到另一台服务器，先通过实际的 SSH/传输方式复制到相同目标目录，再执行后面的迁移脚本；不能把本机不存在的旧路径直接用于远端复制。

```bash
test -d /data/yb/Code/FLEX/FLEX-GSPO
test -d /data/yb/Flex/datasets/data_001
test -f /data/yb/FLEX/datasets/gspo_online_prepared_clean/train.jsonl
test -f /data/yb/Flex/datasets/1019/en2zh_train_gspo_1.jsonl
test -d /data/data_process/lzy/qwen3_omni_mega/train/megatron_output/Qwen3-Omni-Flex/v3-20261016-172842/checkpoint-4410-merged
```

只有这些检查都成功后，再复制。命令不带 `--delete`，不删除原文件；目标应为本次迁移准备的目录。

```bash
mkdir -p /data06/FLEX/code /data06/FLEX/datasets/1019 /data06/FLEX/datasets/data_001 /data06/FLEX/datasets/gspo_online_prepared_clean /data06/FLEX/models/checkpoint-4410-merged /data06/FLEX/outputs

rsync -a --info=progress2 --exclude '__pycache__/' \
  /data/yb/Code/FLEX/ \
  /data06/FLEX/code/

rsync -a --info=progress2 \
  /data/yb/Flex/datasets/data_001/ \
  /data06/FLEX/datasets/data_001/

rsync -a --info=progress2 \
  /data/yb/Flex/datasets/1019/en2zh_train_gspo_1.jsonl \
  /data06/FLEX/datasets/1019/

rsync -a --info=progress2 \
  /data/yb/FLEX/datasets/gspo_online_prepared_clean/ \
  /data06/FLEX/datasets/gspo_online_prepared_clean/

rsync -aL --info=progress2 \
  /data/data_process/lzy/qwen3_omni_mega/train/megatron_output/Qwen3-Omni-Flex/v3-20261016-172842/checkpoint-4410-merged/ \
  /data06/FLEX/models/checkpoint-4410-merged/
```

模型使用 `-L` 复制符号链接指向的实际文件，避免搬走目录后权重仍依赖旧缓存路径。音频若也使用了指向旧磁盘的符号链接，应在迁移前确认其目标，按需要同样复制实际 WAV 文件。

迁移脚本核验文件大小、WAV 头及时间范围，不读取全量音频采样值，也不证明两份音频逐字节一致。如需额外核验传输，可在复制完成后执行下面的只读 checksum 对照；它会读取两端全部音频，因此大数据量时耗时明显：

```bash
rsync -acn --itemize-changes \
  /data/yb/Flex/datasets/data_001/ \
  /data06/FLEX/datasets/data_001/
```

## 3. 确认 Docker 挂载和 Python 环境

迁移脚本与训练应使用同一个音频地址。若宿主机是 `/data06`，建议容器也挂载为 `/data06`，例如在创建容器的原命令中加入 `--mount type=bind,src=/data06,dst=/data06`。`docker exec` 进入旧容器不会自动给它新增挂载。保留原来已经验收的镜像、GPU、共享内存和网络等配置，不用本页的挂载片段替代完整 Docker 启动命令。

进入实际训练容器后检查：

```bash
test -d /data06/FLEX/datasets/data_001
test -f /data06/FLEX/code/FLEX-GSPO/Scripts/relocate_prepared_data.py
test -f /data06/FLEX/models/checkpoint-4410-merged/config.json
```

若仍是原来已验收的容器，而且旧虚拟环境依旧可访问，继续使用它即可；代码和数据迁移不要求虚拟环境一起改地址：

```bash
source /data/yb/venvs/flex-gspo-online/bin/activate
python3 -c "import sys; print(sys.executable); print(sys.version)"
cd /data06/FLEX/code/FLEX-GSPO
```

若是另一台服务器或旧环境不可访问，应在相同训练镜像内、目标位置重新创建并配置环境，不能仅复制 `venv` 目录后修改 `activate`。虚拟环境包含绝对解释器路径，通常不可直接迁移；这是 [Python 官方 venv 文档](https://docs.python.org/3/library/venv.html#how-venvs-work) 的明确限制。原环境的准确依赖版本和已验收镜像应作为复现依据，不在迁移时顺便升级 PyTorch、CUDA、vLLM 等核心依赖。新环境验收前不要启动正式训练。

## 4. 生成新路径的训练清单

已有 prepared `train.jsonl` 时，运行新脚本：

```bash
python3 Scripts/relocate_prepared_data.py \
  --input-jsonl /data06/FLEX/datasets/gspo_online_prepared_clean/train.jsonl \
  --audio-dir /data06/FLEX/datasets/data_001 \
  --output-dir /data06/FLEX/datasets/gspo_online_prepared_v3 \
  --workers 16
```

脚本只生成新清单及迁移报告，不复制或切分 WAV，不加载模型，不调用评分服务。它按照 `id.wav` 在新音频目录寻找文件，保留任务顺序、原文、参考、system prompt 和分块时间计划。目标文件与旧记录不一致或任何一行失败时，不发布可训练的 `train.jsonl`；不要忽略失败继续训练。输出目录必须是尚不存在的新目录，原 prepared 清单不覆盖。

如果只有原始 `key/en/zh` JSONL，没有旧 prepared 清单，则使用现有 01，**与上面迁移命令二选一**：

```bash
bash 01_prepare.sh \
  --input-jsonl /data06/FLEX/datasets/1019/en2zh_train_gspo_1.jsonl \
  --audio-dir /data06/FLEX/datasets/data_001 \
  --output-dir /data06/FLEX/datasets/gspo_online_prepared_v3 \
  --workers 16 \
  --skip-invalid
```

01 会按现有准备规则重新筛选原始数据；若希望严格保持上一版已筛选的样本集合，应选择旧 prepared 清单迁移。两种方式都不需要提前把 WAV 物理切成一秒文件。

## 5. 检查服务与训练环境

```bash
python3 Scripts/check_comet22_service.py \
  --config /data06/FLEX/code/FLEX-GSPO/configs/qalco_v3.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 Scripts/check_environment.py \
  --model /data06/FLEX/models/checkpoint-4410-merged \
  --require-training \
  --required-gpus 4
```

v3 使用 COMET22。服务 URL 仍是配置里的 `http://10.185.1.72:6668/22`；新服务器或容器必须能够访问它。检查失败时先解决对应环境或服务问题，不通过删除检查逻辑跳过。

## 6. 先运行两步，再启动正式训练

先用独立目录检验采样、奖励、更新以及 checkpoint 保存：

```bash
bash 05_train_qalco_v3.sh \
  /data06/FLEX/models/checkpoint-4410-merged \
  /data06/FLEX/datasets/gspo_online_prepared_v3/train.jsonl \
  /data06/FLEX/outputs/gspo_qalco_v3_smoke_001 \
  --gpu-ids 0,1,2,3 \
  --train-iters 2 \
  --save-steps 2
```

完成后，在另一个新目录从同一原 SFT 模型正式起跑：

```bash
bash 05_train_qalco_v3.sh \
  /data06/FLEX/models/checkpoint-4410-merged \
  /data06/FLEX/datasets/gspo_online_prepared_v3/train.jsonl \
  /data06/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --gpu-ids 0,1,2,3 \
  --train-iters 3000 \
  --save-steps 500
```

05 已继承 G8/B32/micro2、vLLM 内存比例 0.5、32 并发和阶段耗时日志，且显式选择 `configs/qalco_v3.json`。保留现有缓存规避、多轮同步、offload 等实现。两步检查使用不同的学习率总周期，不作为正式 3000 步的恢复点；短验收也不能保证长期稳定。

此流程没有 `--resume-from`。路径、清单内容和模型文件元数据变化会影响现有恢复身份校验；如果未来需要迁移后精确续训，应另行核对原生 checkpoint、代码、数据、奖励配置及 optimizer/RNG，不能通过篡改旧 launch 日志绕过校验。

## 7. 监控新目录

训练外另开一个终端，在已验收环境中运行奖励与性能日志监控：

```bash
cd /data06/FLEX/code/FLEX-GSPO
python3 Scripts/monitor_training.py \
  /data06/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --watch \
  --dataset /data06/FLEX/datasets/gspo_online_prepared_v3/train.jsonl
```

再开一个宿主机终端。若之前的 TensorBoard 独立环境仍存在，直接使用它并换成新日志路径：

```bash
/data/yb/venvs/flex-tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data06/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --host 0.0.0.0 \
  --port 6006
```

若新主机没有这个独立环境，则在宿主机创建一个新的监控环境；它只安装 TensorBoard，不影响训练环境：

```bash
python3 -m venv /data06/FLEX/venvs/tensorboard-host
/data06/FLEX/venvs/tensorboard-host/bin/python3 -m pip install 'tensorboard>=2.14,<3'
/data06/FLEX/venvs/tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data06/FLEX/outputs/gspo_qalco_v3_g8_b32_3000_001 \
  --host 0.0.0.0 \
  --port 6006
```

监控重点仍是质量、时延、空译文/复读、组内奖励差异、KL、梯度和实际生成/换入换出耗时。新训练曲线不要和旧 v2 的 reward 数值直接比较；翻译提升需要固定样本对照。
