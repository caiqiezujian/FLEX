# GSPO v4：新格式数据完整运行指南

更新日期：2026-11-02。适用于原始 JSONL 含 audio / srcs / chooses，音频路径已经写在每行 audio 中的新数据。

本指南独立包含环境、准备、训练和监控的 1～9 步。使用 **COMET22 QALCO v4 / Megatron 在线 GSPO**，
默认 **4 卡 / G8 / global batch 32 / micro batch 2 / 3000 步 / 每 500 步保存**。
两种数据入口共用最新 SFT 短提示词，奖励公式和训练容量相同。

**已经训练起来时，不需要为阅读本指南重装环境、重做数据或重启训练。**
监控部分使用你的实际训练输出目录；下面是新一轮训练的完整路径示例。

| 用途 | 路径 |
| --- | --- |
| 代码目录 | /data/yb/Code/FLEX/FLEX-GSPO |
| 训练虚拟环境（Docker 内） | /data/yb/vens/flex-gspo-online |
| 监控虚拟环境（宿主机） | /data/yb/vens/flex-tensorboard-host |
| 原始 JSONL（示例，需换为实际路径） | /data/yb/Code/FLEX/datasets/batch2/raw.jsonl |
| WAV 路径 | 原始 JSONL 每行 audio；不移动音频 |
| SFT 模型（示例，需换为实际路径） | /data/yb/Code/FLEX/models/new-sft-merged |
| prepared 数据输出 | /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102 |
| 本次训练输出 | /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 |

注意虚拟环境目录名为 **vens**，本指南不使用旧的 data06 路径。
每个命令把路径直接写在参数后面，不依赖 MODEL_PATH 等 shell 变量。
多行命令请完整复制，行末反斜杠后不要加空格；否则 bash 会把下一行路径当成独立命令。

## 1. 同步代码并确认入口

当前代码以本地 D:\ICLR\FLEX 为完整仓库。服务器可上传整套代码目录；
通过 Git 同步时，要先确保所需新增文件已发布到对应分支，git pull 不会获取未发布的本地文件。
不要在现有训练进程运行期间切换它正在使用的代码或配置。

在宿主机、对应 Git 分支中执行：

~~~bash
cd /data/yb/Code/FLEX

git status --short

git pull --ff-only origin feat/qalco-gspo-metricx

git log -1 --oneline

ls /data/yb/Code/FLEX/FLEX-GSPO/01_prepare.sh
ls /data/yb/Code/FLEX/FLEX-GSPO/06_train_qalco_v4.sh
~~~

如果 git pull 报本地修改冲突，先保留自己的配置改动再合并，不要直接覆盖服务地址。
目录需挂载到训练容器中，保证容器也能以相同路径访问代码、模型和原音频。

## 2. 在训练 Docker 内准备虚拟环境

**已经用该环境正常训练时，跳过创建和重装，只激活现有环境。** 下面创建命令用于首次部署，
不用于运行中的训练进程。训练虚拟环境依赖创建它的 Docker 内 Python 和底层包，不能搬到宿主机激活后当成独立环境。

先进入此前可用的训练 Docker。容器名称因机器而异，本指南不猜测 docker exec 的容器参数。
如果当前终端激活了 Conda base，先执行 conda deactivate；没有激活则跳过。

首次创建：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

mkdir -p /data/yb/vens

/usr/local/bin/python3 -m venv \
  --system-site-packages \
  /data/yb/vens/flex-gspo-online
~~~

每次新开训练终端后激活并核对：

~~~bash
source /data/yb/vens/flex-gspo-online/bin/activate

python3 -c "import sys; print('Python:', sys.executable); print('Version:', sys.version)"
~~~

解释器应为 /data/yb/vens/flex-gspo-online/bin/python3；本次镜像使用 Python 3.12。
如果打印的是宿主机 /data/miniconda3/bin/python3，先确认所在容器和解释器，不要在错误环境补装训练框架。

## 3. 安装依赖与核对环境

**环境已经可用时不重复安装。首次部署按网络情况选择 3A 或 3B，完成后执行 3C。**
两条安装路线使用同一个 SWIFT 提交和同一组目标版本，不需要同时执行。

| 组件 | 本指南使用的版本或方式 |
| --- | --- |
| MS-SWIFT | 源码提交 f15bb91439d756d4a366b51956d72d1f088b7af6；版本字符串为 4.7.0.dev0 |
| mcore-bridge | 1.6.3 |
| datasets | 4.8.4，提供当前 SWIFT 使用的 Json / List 接口 |
| PyTorch、CUDA、vLLM、Megatron-Core、TransformerEngine、FlashAttention | 复用已验证训练镜像，安装命令不主动替换这一套编译依赖 |

### 3A. Docker 可以联网

在训练 Docker 的虚拟环境内执行。先补齐 datasets，再运行原安装脚本：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

source /data/yb/vens/flex-gspo-online/bin/activate

python3 -m pip install \
  --index-url https://pypi.org/simple \
  "datasets==4.8.4" \
  "numpy==1.26.4"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLEX_REWARD_CONFIG=/data/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v4.json \
bash 00_install_framework.sh --confirm-isolated-env
~~~

00 脚本会通过 GitHub 获取 SWIFT。Docker 无法访问 GitHub 时，改用下方离线路线；
不要反复运行 00 期待它使用本地压缩包。

### 3B. Docker 不能联网：宿主机下载，容器内离线安装

宿主机与容器都需要能访问 /data/yb/packages/flex-gspo。若未共享该目录，将下载好的整个目录
复制到容器可见的同一路径后再安装。**下载可以在宿主机，安装必须在训练 Docker 的虚拟环境。**

在联网的 Linux x86_64 宿主机上下载：

~~~bash
mkdir -p /data/yb/packages/flex-gspo

curl -fL --connect-timeout 20 --max-time 300 --retry 2 \
  https://codeload.github.com/modelscope/ms-swift/zip/f15bb91439d756d4a366b51956d72d1f088b7af6 \
  -o /data/yb/packages/flex-gspo/ms-swift-f15bb91439d756d4a366b51956d72d1f088b7af6.zip

python3 -m pip download \
  --index-url https://pypi.org/simple \
  --dest /data/yb/packages/flex-gspo \
  --no-deps \
  "mcore-bridge==1.6.3"

python3 -m pip download \
  --index-url https://pypi.org/simple \
  --dest /data/yb/packages/flex-gspo \
  --only-binary=:all: \
  --python-version 3.12 \
  --implementation cp \
  --abi cp312 \
  --platform manylinux_2_28_x86_64 \
  --platform manylinux2014_x86_64 \
  -r /data/yb/Code/FLEX/FLEX-GSPO/requirements.txt \
  "datasets==4.8.4" \
  "numpy==1.26.4" \
  setuptools wheel
~~~

最后一条按**容器 Python 3.12 / Linux x86_64**下载数据处理依赖，避免宿主机 Python 3.13 导致下载错轮子。
numpy 沿用本次镜像的 1.26.4。此轮子集合不能替代完整训练镜像。

如果已有之前下载的 omni-rl-audit-swift-20261015.zip，它对应同一个固定提交，可直接使用，
不必再次 curl。将下面安装命令的 SWIFT zip 路径替换为其实际绝对路径即可。

进入训练 Docker 后安装：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

source /data/yb/vens/flex-gspo-online/bin/activate

python3 -m pip install \
  --no-index \
  --find-links /data/yb/packages/flex-gspo \
  -r /data/yb/Code/FLEX/FLEX-GSPO/requirements.txt \
  "datasets==4.8.4" \
  "numpy==1.26.4" \
  setuptools wheel

python3 -m pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  /data/yb/packages/flex-gspo/ms-swift-f15bb91439d756d4a366b51956d72d1f088b7af6.zip

python3 -m pip install \
  --no-index \
  --no-deps \
  /data/yb/packages/flex-gspo/mcore_bridge-1.6.3-py3-none-any.whl
~~~

SWIFT 和 mcore-bridge 采用 --no-deps，依赖训练镜像已有的框架；datasets 则连同需要的数据依赖安装。
如果本地包目录缺文件，补下载对应包，不要通过删掉错误检查掩盖依赖缺失。

### 3C. 核对实际解释器、datasets 接口和训练依赖

在训练 Docker 内执行：

~~~bash
source /data/yb/vens/flex-gspo-online/bin/activate

python3 - <<'PY'
import sys
from importlib.metadata import version
import datasets
import pyarrow
from datasets import Dataset, Features
from datasets.features import Json, List

print("Python:", sys.executable)
for name in ("ms-swift", "mcore-bridge", "datasets", "pyarrow"):
    print(name + ":", version(name))
print("datasets import path:", datasets.__file__)

sample = Dataset.from_dict(
    {"messages": [[{"role": "user", "content": "test"}]]},
    features=Features({"messages": List(Json())}),
)
sample = sample.map(lambda row: row, load_from_cache_file=False)
assert sample[0]["messages"][0]["content"] == "test"
print("PASS: Json/List and minimal Dataset.map")
PY

python3 -m pip check

cd /data/yb/Code/FLEX/FLEX-GSPO

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLEX_REWARD_CONFIG=/data/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v4.json \
python3 Scripts/check_environment.py \
  --require-training \
  --required-gpus 4
~~~

曾遇到的 “cannot import name 'Json' from 'datasets.features'” 就是在新版 SWIFT 调用了旧版 datasets 时触发。
原 check_environment.py 尚不包含该 API 检查，所以这里单独验证，不能只看最后的 ok=true。
这些检查不加载训练模型，也不能代替真实 GPU 运行。发现错误时保留具体输出再处理。

COMET22 配置在 configs/qalco_v4.json：quality_provider=comet22，
service_url=http://10.185.1.72:6668/22，response_key=score。训练容器需要能访问该地址；
如果服务地址变化，应在**新训练启动前**更新该配置。
当前训练启动不强制运行语义评分测试，实际采样后的奖励仍调用 COMET22。

## 4. 准备新格式数据训练清单

原始一行示例：

~~~json
{"audio":"/data/xyh/omni_train_datas/download/en/example.wav","srcs":["The meeting","starts at nine."],"chooses":["会议","九点开始。"],"rejects":["其他译文"],"voice_end":false}
~~~

audio 是可在训练 Docker 内访问的 WAV 路径；绝对路径按原位置读取，相对路径以原始 JSONL 所在目录为基准。
srcs / chooses 必须是非空且等长的字符串列表，每项均须非空。可提供 id 或 key；均未提供时取 WAV 文件名主体。
不同目录下的同名 WAV 应给不同 id，避免被判为 ID 冲突。
本方向为英译中；lang 可省略，若提供则为 en。
rejects 和 voice_end 不参与本版任务或参考构造，voice_end=false 不会取消 EOF 收尾轮。

在训练 Docker 内运行；将原始数据和音频的示例地址换成真实位置：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

source /data/yb/vens/flex-gspo-online/bin/activate

bash 01_prepare.sh \
  --input-jsonl /data/yb/Code/FLEX/datasets/batch2/raw.jsonl \
  --input-format segments \
  --output-dir /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102 \
  --chunk-seconds 1 \
  --workers 16 \
  --progress-seconds 5 \
  --skip-invalid
~~~

srcs 去除每项首尾空白后用空格拼接为英文原文，chooses 去除每项首尾空白后直接拼接为中文参考。音频直接引用 audio，不需要 --audio-dir；文本切片没有时间戳，不作为音频分块边界。

两种入口都遵循以下处理规则：

- 写入当前最新 SFT 短提示词，默认按 1 秒生成虚拟音频块计划，最后不足 1 秒保留真实结束时间。
- 不复制原 WAV，不生成大量小 WAV，不调用模型，不预先生成离线 G8。
- 输出初始 messages、第一块音频引用 audios，以及字符串形式的 flex_task 元数据。
- 原文和参考只用于奖励评分，不进入翻译模型的 messages；后续音频和 EOF 由训练调度器逐轮提供。
- 保留重复统计、无效行报告、进度和原子发布；原始 JSONL 和 WAV 不修改。
- --skip-invalid 允许过滤坏行后发布有效清单；缺少该参数时，有坏行不会发布 train.jsonl。
- 默认过滤超过 40 秒的音频。改大 --max-duration-seconds 时也要核对上下文与最大轮数。
- 输出目录中已有准备结果时拒绝覆盖；已经成功生成且路径、prompt 匹配的清单可复用。

查看准备报告：

~~~bash
python3 -m json.tool \
  /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/prepare_report.json
~~~

确认 manifest_published 为 true，accepted_rows 至少为 4（本文 G8/B32/steps1 每轮需要 4 条 prompt），
并检查 rejected_rows、duplicate_rows 和 accepted_by_format。
本命令的 input_format 应为 segments，accepted_by_format.segments 对应写入条数。

预期输出 train.jsonl、prepare_report.json、rejected.jsonl、duplicates.jsonl。
后续训练只把 **prepared/train.jsonl** 作为任务入口，不把原始 JSONL 直接传给训练器。

当前 prompt 明确 DIRECT/FLEX 按策略区分，并要求按语义使用标点、不随音频块机械加句号、不用省略号占位。
旧 prepared 文件不会因为更新代码而自动更换 prompt；若仍内嵌旧提示词，应从原始 JSONL 重新准备到新目录。
SFT 模型也应确实使用相匹配的提示词，不能仅凭旧 checkpoint 名字假定它匹配。

## 5. 启动正式 GSPO 训练

下面第一个模型路径是示例；替换为真实的 SFT 合并模型目录，包含匹配的权重、tokenizer 和 processor。
如果当前已有训练在跑，不要再执行本命令启动第二份任务。

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

source /data/yb/vens/flex-gspo-online/bin/activate

bash 06_train_qalco_v4.sh \
  /data/yb/Code/FLEX/models/new-sft-merged \
  /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/train.jsonl \
  /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --global-batch-size 32 \
  --micro-batch-size 2 \
  --steps-per-generation 1 \
  --train-iters 3000 \
  --save-steps 500 \
  --lr 1e-6 \
  --kl-beta 0.01 \
  --temperature 1.0 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --max-length 8192 \
  --max-new-tokens 128 \
  --max-turns 65 \
  --vllm-memory 0.5 \
  --vllm-max-num-seqs 32 \
  --vllm-mm-processor-cache-gb 0 \
  --dataset-num-proc 4 \
  --performance-log on \
  --rollout-log-every 5
~~~

入口自动选择 configs/qalco_v4.json，质量提供方为 COMET22。
每轮采样 4 条音频 × 8 个候选 = 32 条轨迹，在线采样后评分并更新；不需要旧的离线多轨迹构建脚本。
质量/时延的归一化信号权重为 1.0 / 0.25；长度、重复和失败处理沿用当前 v4。

最大新 token 数 128 是**每轮回复**限制；65 轮预算包含 EOF。
vllm-mm-processor-cache-gb 保持 0，保留此前缓存清理问题的规避设置。
当前本地启动器按四卡校验，本指南没有把讨论中的八卡建议写成已支持的启动命令。

从 SFT 新启动时，训练输出目录应未被旧运行占用。失败后重新启动新实验可换成末尾 _002，
不要把合并后的 HF 模型路径当作原生 Megatron 的 --resume-from。
真正断点续训需要原生 checkpoint、优化器/RNG 和对应运行身份，另行核对。

每 500 步保存一次，checkpoint 位于输出目录内自动生成的 vN-时间戳 子目录下。
训练终端保持运行；后续两个监控终端都放在宿主机。

## 6. 在宿主机准备 TensorBoard 环境

首次创建时执行；已有可用环境只需使用其解释器，无需重建：

~~~bash
mkdir -p /data/yb/vens

python3 -m venv /data/yb/vens/flex-tensorboard-host

/data/yb/vens/flex-tensorboard-host/bin/python3 -m pip install \
  "tensorboard>=2.14,<3"
~~~

这套监控环境独立于训练 Docker，不需要 PyTorch、CUDA、SWIFT 或 COMET 模型。
宿主机需要能读取代码、prepared 清单和训练日志，并能在运行目录下创建和写入 monitor/ 事件与游标文件。

## 7. 宿主机监控终端一：持续导入奖励、性能和样例

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

/data/yb/vens/flex-tensorboard-host/bin/python3 Scripts/monitor_training.py \
  /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --watch \
  --poll-seconds 10 \
  --dataset /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/train.jsonl
~~~

保持该终端运行，每 10 秒导入新增日志。同一个训练输出目录只启动一个这样的观察进程。
已有训练的输出目录若与本示例不同，必须把本节、下一节和第 9 节统一替换为实际路径。
第一次索引大清单可能需要一些时间，脚本不读取音频进行推理。
Ctrl-C 停止的是这个监控进程，训练在自己的终端继续运行。

## 8. 宿主机监控终端二：启动 TensorBoard 页面

~~~bash
/data/yb/vens/flex-tensorboard-host/bin/python3 -m tensorboard.main \
  --logdir /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --host 0.0.0.0 \
  --port 6007
~~~

浏览器打开 http://服务器IP:6007 。0.0.0.0 是监听地址，不是浏览器目标地址。
两个指南分别使用 6006 / 6007，方便同时查看两个实验；只看一个实验时可以选任一空闲端口。
端口不可直达时，在自己的电脑终端建立 SSH 隧道，下面登录地址需换成实际服务器：

~~~bash
ssh -N -L 6007:127.0.0.1:6007 root@服务器IP
~~~

然后在自己电脑浏览器打开 http://127.0.0.1:6007 。
--logdir 指向**整个训练输出根目录**，才能同时看到原生训练日志和监控脚本生成的事件文件。

建议查看：

| 内容 | 判断用途 |
| --- | --- |
| loss、kl、grad_norm、learning_rate | 优化过程是否出现非有限数、梯度异常或学习率异常 |
| 原始 quality/mean、COMET22 分数 | 当前采样译文的质量代理，不能脱离样本难度比较 |
| latency_seconds/mean | 音频时间轴上的输出滞后代理，不是 GPU 实际耗时 |
| empty_translation_fraction、protocol_valid_fraction、severe_repetition_fraction | 空译、协议和循环问题是否恶化 |
| 长度超出、门控、组内优势与文本样例 | 排序是否符合预期，短译是否伴随漏译 |
| performance 与 train_speed | 采样、卸载/回载、更新等耗时 |

监控脚本的奖励摘要以各 rank 回调为单位，不自动代表一整个 G8；
比较候选优劣应使用日志内完整同音频分组。
v4 已做组内分项缩放，reward 不必持续为正或单调上升。短期波动不等于训练失败，
固定验证音频的质量/时延对照才用于判断真实提升。

## 9. 跑到 20～30 步后提取诊断结果

宿主机另开终端，使用监控环境：

~~~bash
cd /data/yb/Code/FLEX/FLEX-GSPO

/data/yb/vens/flex-tensorboard-host/bin/python3 Scripts/collect_training_status.py \
  /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  --dataset /data/yb/Code/FLEX/datasets/gspo_prepared_batch2_v4_shortprompt_1102/train.jsonl \
  --reward-config /data/yb/Code/FLEX/FLEX-GSPO/configs/qalco_v4.json \
  --window 10 \
  --latest 5 \
  --examples 3 \
  --text-chars 1600 \
  --max-turns 65
~~~

这条命令将报告打印到终端，可以直接复制文本。脚本不调用评分服务、不加载模型、不停止训练。
如需同时保存文件，在完整命令末尾追加 --output 和一个新的报告文件绝对路径；已有同名文件不会覆盖。
无需等待训练结束，可以在后续阶段重复收集，保存时使用新的文件名。

核对当前实际启动参数或寻找 checkpoint，可额外执行：

~~~bash
ls -lt /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001/launch_*.json

find /data/yb/Code/FLEX/outputs/gspo_v4_comet22_batch2_shortprompt_g8_b32_3000_001 \
  -maxdepth 3 -type d -name 'checkpoint-*' -print
~~~

新一轮训练不要混用另一份指南的数据/输出路径。两种原始格式最终都变成同一个训练任务协议，
主要区别是第 4 步如何读取文本和定位音频，prompt、评分方式与训练逻辑共用当前实现。
