# FLEX · 本地模型同声传译与强化学习

FLEX Live 是面向 SFT 后双向中英 Qwen3-Omni 模型的流式推理系统，提供单音频命令行和 Web 两种入口。Web 页面支持播放上传音视频，或采集电脑正在播放的声音；右侧显示增量译文，并单独记录 WAIT / FLEX / DIRECT 策略与处理时间。一个本地 checkpoint 支持中→英、英→中两个方向，不调用第三方翻译 API；可选接入现有 Qwen3-TTS 服务播报译文。

文件模式会先将用户选择的完整音视频上传到服务器临时目录，服务器解码后按播放进度逐块推理；**文件已上传不等于模型能看到未来音频**。普通文件由浏览器本地播放；WMV/ASF 上传后自动转为 H.264/AAC MP4，浏览器播放服务器提供的临时视频。文件模式只报告播放位置，不再实时抓取播放器声音，不依赖 ScriptProcessorNode 或 AudioWorklet。视频烧录字幕或另选的 WebVTT 字幕仍可用于人工对照翻译时机，外部 VTT 不上传。

仓库根目录保留独立的 Web 推理系统及单音频推理入口；[FLEX-GSPO](FLEX-GSPO/README.md) 提供原始 JSONL + WAV 的任务准备、Megatron-SWIFT 在线 GSPO、QALCO 奖励、MetricX HTTP 评分和训练文档。多轨迹在训练过程中由当前策略生成，不要求先完成全量离线 G8 采样。两部分使用各自的依赖和配置，不需要为运行 Web 系统安装训练依赖。仓库不包含模型权重、训练数据、实际 MetricX 服务地址或蒸馏 prompt。前端无需 Node 构建；Node 仅用于音频处理逻辑测试。

> 当前版本：一份模型默认接纳 4 路独立同传、滚动模型上下文、每路默认最长 8 小时。完整结果落盘，页面保留当前会话全部已收到的译文，可滚动回看；正常等待、暂时变慢和服务故障分别显示。目标部署是一张 A800 80 GB；CPU/mock 并发测试不等于真实 GPU 四路实时性或数小时稳定性验收，实际吞吐和翻译质量仍需在目标服务器测量。mock 不是语音翻译模型。

![FLEX Live 界面：左侧播放测试视频，右侧显示增量文本与策略时间线](docs/preview.jpg)

*截图为明确标注的 mock 模式，使用合成测试视频和演示文本，不代表真实翻译质量。*

## 功能概览

- 电脑声音实时输入，可明确选择混入自己的讲话；另支持本地音视频按播放进度输入，包括自动准备的 WMV。
- 同一个双向 checkpoint 切换中→英、英→中，每次启动建立独立会话。
- 译文只追加，不回滚；正文不显示协议标签，时间线保留策略与时间记录。
- 逐轮记录默认跟随最新；手动上翻暂停跟随，可点击“回到最新记录”。
- 可选中英译文播报，新增译文立即请求、仅合并尚未提交的待播片段、按序播放，提供音量/试听/停止控制；[TTS 配置与部署](docs/TTS.md)。
- 文件固定分块、播放暂停/恢复、精确尾块处理；电脑声音使用实时采集与有界队列。
- TXT 纯译文导出、JSON 完整事件导出，以及可关闭的服务端 JSONL 日志。
- FastAPI + WebSocket 后端，无前端构建步骤；文件模式支持内网 HTTP，电脑声音需 HTTPS/localhost、浏览器共享授权和 AudioWorklet。

| 模型响应 | 右侧译文正文 | 策略记录与模型历史 |
|---|---|---|
| `<\|WAIT\|>` | 不追加内容，保留此前译文 | 保留 WAIT |
| `<\|FLEX\|>新增译文` | 立即追加新增译文，不显示标签 | 保留 FLEX 与原始响应 |
| `<\|DIRECT\|>新增译文` | 立即追加新增译文，不显示标签 | 保留 DIRECT 与原始响应 |

FLEX 译文不是可以事后改写的草稿；与 DIRECT 一样，输出后就进入已提交历史。策略由训练后的模型决定，服务端只做协议校验，不根据句子长短或完整程度擅自重标。

### 双向模型与训练提示词

同一个已完成双向训练的 checkpoint 默认启用英→中和中→英。页面选择“英语 → 中文”或“中文 → 英语”即可；文件播放和电脑声音输入使用相同的方向选择，不需要重新加载模型。

| 页面方向 | 请求语言代码 | 模型提示词的第一句 |
|---|---|---|
| 英语 → 中文 | `en-zh` | 你是一个同声传译专家，负责将用户音频中的语音同声翻译为中文。 |
| 中文 → 英语 | `zh-en` | 你是一个同声传译专家，负责将用户音频中的语音同声翻译为English。 |

两种完整提示词集中定义在 `flex_web/model.py` 的 `SYSTEM_PROMPTS`，后续 WAIT/FLEX/DIRECT 和增量输出规则完全一致。中→英的 `English` 保留训练原文，不从页面语言标签生成提示词。单音频 CLI 同样复用这套定义。已有会话的滚动上下文始终保留本会话的提示词。

会话进行中不能切换方向；结束后切换并重新开始，会建立新的音频与译文历史。切换下一次的方向时，上一会话的译文、语言标记和导出仍属于上一会话，直到开始新会话或手动清空。

## 获取代码与文档导航

```bash
git clone --branch feat/qalco-gspo-metricx https://github.com/caiqiezujian/FLEX.git
cd FLEX
```

仓库根目录即 Web 应用根目录，后面的 Web 命令在此目录执行。强化学习请进入 `FLEX-GSPO/`，按该目录 README 配置模型、任务文件和 MetricX 服务；公开配置不包含真实内网地址。

- [真实模型启动](#1-先用真实模型启动) / [无 GPU 演示](#2-无-gpu-的功能演示)
- [Megatron 在线 GSPO：任务准备、训练与恢复](FLEX-GSPO/README.md)
- [强化学习方法、奖励公式与完整示例](FLEX-GSPO/docx/QALCO强化学习方案详解.md)
- [先做单音频流式 S2TT，不启动网页](docs/S2TT.md)
- [客户端/服务端协议](docs/PROTOCOL.md)
- [已完成的验证与未验证边界](docs/VALIDATION.md)
- [开发与测试](CONTRIBUTING.md)
- [部署安全与数据隐私](SECURITY.md)

## 先验证模型：单音频 → 流式译文

最简单的启动方式（已修复 tokenizer、按独立标签训练的模型）：

```bash
CUDA_VISIBLE_DEVICES=6 bash run_infer.sh /absolute/path/to/checkpoint-merged /absolute/path/to/audio.wav --direction en-zh
```

默认 1 秒分块、fast 模式，逐块推理但不等待播放时钟；追加 `--pace realtime` 可模拟实时到达。先停止占用同一 GPU 的 Web 服务。也可以使用不含任何 Web 文件的五文件独立包，见 [快速说明](docs/S2TT-QUICKSTART.md)。

如果暂时不需要平台，可以直接运行 `infer_audio.py`。它复用同一模型适配层，但**不启动 Web 服务，不需要浏览器，也不需要 FastAPI**。先把音频放到 GPU 服务器，再指定本地文件路径：

```bash
python -m pip install -r requirements-infer.txt
CUDA_VISIBLE_DEVICES=0 python infer_audio.py \
  --model /absolute/path/to/checkpoint-merged \
  --audio /absolute/path/to/english.wav \
  --direction en-zh \
  --chunk-seconds 1 \
  --pace realtime \
  --output-dir outputs/en-zh-first
```

默认按真实时间逐块释放音频；`--pace fast` 取消时间等待，但仍逐块推理，不会把完整音频交给模型。中→英使用 `--direction zh-en`。终端增量显示译文，输出目录保存 `translation.txt` 和 `events.jsonl`；目录必须是新路径，避免覆盖前一次结果。16 kHz 单声道 PCM16 WAV 无需额外解码器，MP3/FLAC/其他采样率文件通过 FFmpeg 流式转换。

模型准备、协议形式、积压含义、尾块与错误处理见 [单音频 S2TT 使用说明](docs/S2TT.md)。GPU/CUDA 依赖仍需复用已经跑通模型的环境；`requirements-infer.txt` 不替代完整模型环境。

## 1. 先用真实模型启动

运行位置：Linux GPU 服务器，配置目标为一张 A800 80 GB。使用已经跑通 Qwen3-Omni 推理的 Python / CUDA / vLLM 环境。适配代码面向 **vLLM 0.11.1** 接口，尤其依赖其多模态输入预处理接口；没有声称兼容所有最新版 vLLM。

`requirements.txt` 只安装 Web/CPU 依赖，**不安装或替换 torch、transformers、vLLM 与 CUDA**。真实模型需要自行准备完整 merged checkpoint 及其训练时匹配的 tokenizer/processor；原始基座模型不会因为套上这个界面就自动具备三标签同传能力。

```bash
cd /path/to/FLEX
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 bash start_server.sh /absolute/path/to/checkpoint-merged
```

或直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --backend vllm \
  --model /absolute/path/to/checkpoint-merged \
  --host 127.0.0.1 --port 8000 \
  --protocol-form special \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-audio-turns 256 \
  --max-tokens 160 \
  --chunk-seconds 1 \
  --max-chunk-seconds 3 \
  --max-pending-seconds 12 \
  --max-session-seconds 28800
```

注意：上面 `special` 只适用于真正按三个 special token 训练并正确保存 tokenizer 的 checkpoint。历史 text-form 模型改用 `--protocol-form text`；默认 `auto` 只检查 tokenizer 的表现，**不能证明模型权重的训练形式**。服务不会替你新增协议 token 或修改模型文件。模型必须是完整 merged 权重及匹配的 tokenizer/processor，不是仅 LoRA adapter。

只加载一个双向 checkpoint；选择中→英或英→中会换目标语言 system prompt、重新建立会话，不会加载第二份模型。推理沿用 SFT 多轮音频协议：每块音频一个 user 轮，每个 WAIT/FLEX/DIRECT 响应原样回填历史。

模型初始化和预热完成前不要开始播放。服务只运行一个进程、一份模型，默认接纳 4 个活动会话，由同一调度器安排小批量推理。**不要用 `uvicorn --workers 2` 或 `--reload` 包装真实模型**，否则可能重复占用显存。多人使用同一个网址，无需各自启动服务或分配端口。

原来的 `start_server.sh` 不需增加参数，就会使用 4 路上限。可在末尾加 `--max-sessions 4` 显式指定；`--max-batch-size` 默认等于会话上限。每个会话有独立音频队列、语言方向、模型历史、滚动窗口、日志和导出，不会共享他人的对话。第五个用户在满员时收到提示，不自动排队，也不会开始录音后偷偷丢弃输入。页面显示当前使用人数；空闲、暂停但未结束、收尾和取消后仍在执行的推理都占用名额，使用完请结束会话。

调度器按已就绪请求的 FIFO 顺序取任务，每个会话最多一个待执行/正在执行的推理轮；默认收集窗口最多 10 ms，凑满 4 路立即提交，无需等待没有音频的用户。多个请求作为独立 prompt 交给同一次 vLLM `generate`，不会把几个人的音频拼到同一对话。模型只由一个线程调用，进程/权重不复制。排队中的取消可立即移除，已开始的 GPU 批次不能强杀；批次完成后丢弃已取消会话的未提交响应，其他会话正常继续。

这版采用兼容现有同步 vLLM 接口的小批量调度，不是迁移到原生 AsyncLLM 连续批处理：同一批较快的响应也要等待本批整体返回。显存占用和四路实际延迟需实测，不能仅凭 A800 80G 推定能持续实时。若当前 vLLM/模型组合不支持多请求音频批处理，可显式加 `--max-batch-size 1` 保持四路接入但轮流推理，吞吐可能更低；不会静默降级或降低四路人数上限。

### 在你自己的电脑打开服务器页面

**可信内网 HTTP：直接播放音视频文件并同传，不需要访问口令、SSH 或浏览器 flags。** 如果以前设置过 `FLEX_ACCESS_TOKEN`，先取消一次，再按原脚本启动：

```bash
unset FLEX_ACCESS_TOKEN
CUDA_VISIBLE_DEVICES=6 bash start_server.sh /absolute/path/to/checkpoint-merged \
  --host 0.0.0.0 --port 18889
```

同事在 Chrome / Edge 打开 `http://服务器真实内网IP:18889`，选择音视频文件并开始。未配置令牌时，页面不显示令牌输入框。`0.0.0.0` 是服务器监听地址，不是浏览器访问地址；Docker 部署还需将该端口映射到宿主机。`CUDA_VISIBLE_DEVICES=6` 选择物理 GPU 6，不需要再传 `cuda:6`。

认证仍可按需启用：只有 `FLEX_ACCESS_TOKEN`（或 `--token-env` 指定变量）为非空值时，上传、删除和推理会话才要求同一口令。不设令牌时，能够连接此端口的人都可以使用服务，请用内网路由和防火墙限制访问；`0.0.0.0` 本身不会限制为内网。

点击“开始同传”后先上传和准备文件，成功后才播放。服务器从临时解码文件读取已播放前缀，网页短暂卡顿后只需更新播放位置，不会因约 43 ms 的采集回调偏移终止。文件输入不创建浏览器音频采集节点；HTTP 与 HTTPS 使用同一文件路径。播放器暂停时不释放未来音频，恢复后连续读取；推理落后时显示待处理时长，保留磁盘上的音频，不触发麦克风的内存队列溢出规则。

默认上传上限 4 GiB、文件解码时长上限与 `--max-session-seconds` 一致（默认 28800 秒，即 8 小时）。上传超时默认 3600 秒，解码超时默认 1800 秒，均可配置。16 kHz 单声道 PCM16 WAV 可直接处理；其他格式需要服务器可执行 FFmpeg，可用 `--ffmpeg /path/to/ffmpeg` 指定。只复用现有模型环境，不需更换 torch/vLLM。临时文件正常在会话结束/取消后清理，未使用的上传有过期清理；完整文件会传到服务器，敏感媒体请先确认留存和访问策略。

长文件请给临时磁盘留足空间：8 小时的解码 PCM 约 0.92 GB，解码期间还存在原始上传；默认最多 4 个上传槽位，极端并发准备可占约 20 GB，另有结果与日志。可在原启动命令末尾追加 `--temp-dir /data/flex-tmp` 指定较大的本地磁盘。大视频仍需先完整上传/解码，再开始播放同传，不是边上传边播放。

WMV/ASF 还会生成一份临时 MP4，需额外磁盘空间和 CPU 转码时间；单份转码 MP4 预算同 `--max-upload-mb`，默认四份同时准备可能额外占约 16 GiB。这个参数分别限制原始上传和转换后的 MP4，**不是总磁盘预算**；准备期间可能同时保留原文件、MP4 和 PCM。选择 WMV 后点击开始，等待上传及准备完成即可，不必先手动改格式。原文件不覆盖；模型音轨从同一份转码 MP4 提取。需 FFmpeg 包含 `libx264` 与 AAC 编码器及配套 `ffprobe`，Ubuntu/Debian 可在运行服务的同一容器内执行 `apt-get update` 和 `apt-get install -y ffmpeg`，再用 `ffmpeg -version`、`ffprobe -version` 检查。只安装 Python 的 `ffmpeg` 包不能替代可执行程序。会话结束后临时 MP4 清理，再次开始会重新上传准备；不是为整个历史文件库做永久转码。

更新包含 Python 后端，**需要重启服务并刷新页面**。仅使用 HTTP 文件模式时，原来的启动命令不变，不需要证书。

### 内网电脑声音：自动生成证书并启动 HTTPS

在原启动命令后增加 `--https-host`，填写**浏览器访问的真实 IP 或域名**，不要填 `0.0.0.0`、协议或端口。下面的 `10.0.0.8` 是示例，请换成自己的服务器地址：

```bash
unset FLEX_ACCESS_TOKEN
CUDA_VISIBLE_DEVICES=6 bash start_server.sh /absolute/path/to/checkpoint-merged \
  --host 0.0.0.0 --port 18889 --https-host 10.0.0.8
```

首次启动会在服务器的 `certs/` 下生成本项目专用的自签根 CA 和带 IP/DNS SAN 的服务器证书，随后直接启动 HTTPS/WSS。需要系统有 OpenSSL 1.1.1 或更新版本（Ubuntu/Debian 缺少时安装 `apt-get install openssl`），不新增 Python 包，不更换模型环境，不需要 Nginx、域名或公网访问。

同事访问 **`https://10.0.0.8:18889`**。仍然是一个 Python 服务、一个 GPU worker、原端口，无访问令牌。**这个端口现在提供 HTTPS，不再提供 HTTP；浏览器需明确改为 `https://`，没有额外的 HTTP 重定向端口。** Docker 继续映射相同端口即可；`--https-host` 填客户端实际访问的宿主机 IP，而不是容器 IP。

**首次在使用者电脑上信任证书（以 Windows + Chrome/Edge 为例）：**

1. 管理员将服务器 `certs/root-ca.crt` 复制给同事，只分发这个公开证书，不分发 `.key` 文件。
2. 双击 `root-ca.crt` → “安装证书” → “当前用户” → “将所有的证书都放入下列存储” → **“受信任的根证书颁发机构”** → 完成。核对这是部署者提供的 FLEX 证书。
3. 完全退出并重新打开浏览器，访问上述 HTTPS 地址。选择“电脑声音”，开始时在浏览器共享弹窗中选择来源并勾选共享音频。

证书只需在每台电脑首次使用时导入；重启服务仍复用同一个 CA。**共享声音的浏览器选择弹窗仍须每次确认**，不能用证书或脚本跳过。单位统一管理的电脑也可以由管理员集中分发证书。证书由我们自己签发，不会自动成为其他电脑信任的证书；不能把“忽略证书警告继续访问”当作所有浏览器均可稳定采集的保证。如果仍有地址不匹配警告，检查访问 IP/域名是否与 `--https-host` 一致；HTTPS 正常但采集失败时，检查浏览器是否提供共享音频、是否已勾选，以及可选麦克风权限。

保留 `certs/`，不要随 `git pull` 删除；容器重建时需持久化该目录。证书/私钥已排除 Git 和源码打包。可通过 `--cert-dir /persistent/path/flex-certs` 指定持久化目录。多个访问地址可重复指定 `--https-host`；换 IP 时仍复用同一 CA，已信任它的电脑不需再次导入。服务器证书到期前会在下次启动时自动续签，不会在运行中热重载；根 CA 不会被静默替换。

已有单位证书时不需要自签，改用：

```bash
CUDA_VISIBLE_DEVICES=6 bash start_server.sh /absolute/path/to/checkpoint-merged \
  --host 0.0.0.0 --port 18889 \
  --ssl-certfile /path/to/fullchain.pem --ssl-keyfile /path/to/private.key
```

也支持已有 HTTPS 反向代理：后端保持原 HTTP 启动，代理转发 WebSocket Upgrade、保留 Host，并正确处理可信代理头。应用令牌始终可选，与 HTTPS 或麦克风授权无关。仅个人调试时，也可用 SSH 将服务器端口转发到本机 localhost；多人内网使用不需要采用这种方式。

## 2. 无 GPU 的功能演示

```bash
python -m pip install -r requirements.txt
python run.py --backend mock --host 127.0.0.1 --port 8000
```

此模式可验证文件上传、真实音视频播放、按播放进度分块、暂停/结束、界面、导出和协议，但**不识别也不翻译语音**。页面和输出内容都会明确标注“演示”。切换 `--backend vllm --model ...` 才是真实同传，不能拿演示结果评估翻译质量。

## 3. 使用方法

1. 选择“音视频”或“电脑声音”，选好中英方向和基础 chunk（建议先 1 秒）。普通文件先本地预览；WMV/ASF 等到点击开始、服务器转码后再播放。仅选择文件不会上传。
2. 文件模式可选浏览器支持的 MP4/WebM/WAV/MP3 等文件，WMV/ASF 自动准备成 MP4。视频在左侧实际播放；烧录字幕直接可见，也支持另选本地 WebVTT 字幕。系统不生成源语 ASR 字幕，也不能保证浏览器读取任意 MKV/嵌入式软字幕轨。
3. 点击开始：文件模式依次上传、准备音轨/视频、连接推理，然后播放；电脑声音先请求共享授权，会话建立后才发送采集到的新音频，准备阶段不偷偷缓存已有播放内容。FLEX/DIRECT 均直接显示，WAIT 只出现在策略记录中，不污染译文。
4. 暂停视频时不再释放新的音频，服务器继续处理已播放的部分；恢复后从原位置连续读取。结束会确认最后的播放位置，再处理此前尚未完成的音频。不要通过刷新网页代替正常结束。
5. 当前会话禁止随意跳播/倍速。要看其他位置：先结束，定位视频，再开始新会话。这样不会把新位置接到旧历史上。
6. 导出 TXT 获取原样拼接译文，导出 JSON 获取服务端事件及时间信息；运行中导出是当时已提交的快照。完整结果从服务器读取，不受页面显示数量限制。服务器默认另存 `logs/<session_id>.jsonl`，可用 `--no-log` 禁用持久日志。

页面正文保留当前会话全部已收到的译文，不再设 500 个片段 / 60000 字符的显示上限。手动上翻时不强制拉回底部，返回底部后继续跟随新译文。只有辅助策略记录保留近期 200 行、最多 120000 个 UTF-16 字符；模型上下文仍独立滚动，不影响页面全文。全文显示会随会话长度增加浏览器内存与 DOM 开销，不代表任意时长都不会变慢。

临时导出最多保留 24 小时、最多 8 个会话，新会话可能提前淘汰最旧已结束结果；服务重启会清除本进程的临时结果。重要内容请及时导出。`--no-log` 仍会临时落盘以支持完整导出，不表示不写任何译文文件。服务器结果不可用时，页面备份包含全部已收到译文和近期策略记录，仍明确标为 PARTIAL，不能代替服务端完整事件记录。

### 演示网页或播放器的声音

1. 用桌面版 Chrome/Edge 打开已受信任的 HTTPS 页面，选择“电脑声音”。
2. 只翻译网页视频时，在共享弹窗选择播放视频的标签页，并勾选共享标签页音频；要翻译外部播放器或整个电脑输出时，选择整个屏幕并勾选系统音频（浏览器/操作系统必须提供该选项）。单独选择一个窗口不保证有音轨。
3. 等页面进入同传中，再播放要演示的内容。网页只发送音频 PCM，不读取、编码或上传屏幕画面；浏览器仍会显示共享提示。
4. 想同时翻译自己现场讲话时，开始前勾选“同时收录我的讲话”，并允许额外的麦克风权限。默认不启用；仅共享电脑声音不会自动听到你的讲话。建议戴耳机，避免扬声器外放又被麦克风重复收进来；浏览器回声消除不能保证完全去重。
5. 点击结束同传，或浏览器的停止共享，结束输入并处理尾块。新会话重新选择共享来源。文件模式与电脑声音模式互斥，不把同一文件额外采集一次。

是否提供系统音频由浏览器和操作系统决定，`systemAudio: include` 不是强制能力。未勾选音频或返回流没有可用音轨时，页面明确提示，不会悄悄改用麦克风。听不到声音时先确认正确的共享来源和勾选项，不必重新安装服务器 FFmpeg；这条实时路径不使用文件解码器。[浏览器 Screen Capture 说明](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia)、[Chrome 共享控制说明](https://developer.chrome.com/docs/web-platform/screen-sharing-controls)。

仅视频画面及用户提供的字幕用于人工对照；页面“本轮耗时”不是模型语义延迟，不能直接当论文里的 AL/LAAL。完整字段见 [协议说明](docs/PROTOCOL.md)。

服务端日志、临时导出与下载文件包含译文、源文件名和时间信息。处理敏感内容时可用 `--no-log` 关闭持久日志，并保护临时目录及下载文件；不应将这些文件提交到仓库。详见 [安全与隐私说明](SECURITY.md)。

## 4. Chunk、积压与停止行为

页面“推理设置”中的 1 / 2 / 3 秒会实际改变每轮音频长度，不是仅用于显示。会话开始后参数锁定，切换块长需要结束当前会话再开始。

| 输入 | 基础分块 | 积压合并 |
|---|---|---|
| 上传音视频 | 按选择固定取 1 / 2 / 3 秒 | 不启用；已播放但未处理的音频保留在文件中，按原块长依次处理 |
| 电脑声音（可混入讲话） | 按选择取 1 / 2 / 3 秒 | 可启用；下一次取块时根据待处理量增大，默认最多 3 秒 |

电脑声音开启自适应且待处理量足够时，本轮长度为 `min(最大块长, floor(待处理量 / 基础块长) × 基础块长)`。例如基础 1 秒、积压 2.6 秒时取 2 秒；基础 2 秒、积压 4 秒且最大 3 秒时取 3 秒。基础已经为 3 秒、上限也是 3 秒时，没有继续增大的空间。该规则不会中途改变正在运行的推理；WAIT 则是进入模型历史的输出策略，不会让旧音频重新回到待处理队列。旧麦克风客户端仍兼容同一 PCM 协议。

```text
文件：上传 → 服务器解码到临时 PCM → 浏览器报告播放位置
    → 仅释放已播放前缀 → 固定大小 chunk（默认 1 s）
电脑声音（可混入麦克风）：AudioWorklet → 16 kHz PCM16 网络帧
    → 有界 FIFO（默认 1 s，积压可合并至 3 s）
两种输入均进入：
    → 同一模型串行推理：完整历史 + 当前新音频块
    → WAIT / FLEX / DIRECT 校验 → 追加字幕与事件日志
```

- 文件模式不传浏览器 PCM；短暂延迟的进度消息会释放连续的已播放前缀，不跳过中间音频，分块不因浏览器调度而改变。电脑声音网络帧与模型 chunk 分开，采集不会因模型正在生成而停下。
- WAIT 是模型策略，不等于计算积压，也不触发重复发送同一段音频。
- 不用 VAD 自动删静音，避免改变现有训练协议；可以后续通过实验验证再加。
- 正常结束的不足一块尾音只处理一次；默认不足 1 秒时仅对模型输入补静音至 1 秒，与现有脚本保持一致，原始音频时间不延长。最后一轮若仍是 WAIT，会提示人工检查尾句，不伪造额外音频轮强迫模型补译。“已处理全部音频”不代表语义上保证没有漏译。
- 电脑声音及旧麦克风客户端的积压上限默认 12 秒，超限报错停止，统计未翻译量；不跳帧伪装实时。文件使用磁盘上的完整音频，不适用该内存积压上限，但推理落后仍会增加字幕延迟。
- 断开或取消后，已在途 GPU 请求必须先结束，才释放单卡会话名额，不把两个历史混起来。
- 不修改、不删除已输出译文。格式错误/截断生成不会作为合法字幕发布，也不擅自将错误改成 WAIT。

### 正常等待、暂时变慢与故障

- **正常等待**：模型返回 WAIT 表示本轮暂不输出；正常分块等待、调度排队和收尾也不等于故障。不能根据没有新译文的时间判定模型卡死。
- **暂时变慢**：一次实际模型批次运行达到 5 秒，或未处理音频（含排队及在途块）持续 5 秒达到 `max(3 秒, 2 × 基础块长)`，显示变慢/积压提示，继续处理，不自动停会话或丢音频。单次短暂积压不会立即触发持续变慢提示。
- **疑似停滞**：实际模型批次达到 60 秒仍未返回，暂不接纳新任务，明确通知尚未执行的排队任务；已运行的 GPU 工作仍等待真实返回，不强杀、不提前释放其资源。该批次成功返回后可恢复接纳。60 秒不是“已确认模型崩溃”的证据。
- **确认故障**：共享模型调用抛出异常或返回数量异常，标记引擎失败、拒绝后续任务，需排查服务端日志并重启服务。单个会话的协议错误不将整个引擎标为故障。输入积压超限等确定错误会及时通知页面停止采集，但仍等在途推理真正结束后再释放会话。

这些是可调整的工程告警阈值，不是模型训练参数或已经测得的最佳延迟。`/api/health` 在疑似停滞或引擎失败时返回 503，单纯四路名额已满仍返回 200；`/api/config` 继续可读。状态显示复用原有位置，没有新增页面布局。

主要参数（完整列表见 `python run.py --help`）：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--chunk-seconds` | 1 | 基础模型音频块长度，秒 |
| `--max-chunk-seconds` | 3 | 可选基础块/实时声音自适应合并的上限，秒；文件不自适应合并 |
| `--max-pending-seconds` | 12 | 实时声音待处理音频上限，秒；不含正在推理的块 |
| `--max-session-seconds` | 28800 | 实时声音会话时长及上传文件解码时长上限，秒（8 小时） |
| `--max-sessions` | 4 | 同时接纳的独立会话数；模型仍只加载一份 |
| `--max-batch-size` | 跟随 max-sessions | 一次模型调用最多处理多少路就绪请求；不大于会话上限 |
| `--batch-wait-ms` | 10 | 收集就绪请求的小窗口，0–100 ms；不是等待未来音频 |
| `--slow-inference-seconds` | 5 | 实际模型批次运行变慢的提示阈值，不终止推理 |
| `--stalled-inference-seconds` | 60 | 疑似停滞阈值，须大于变慢阈值；不强杀在途 GPU 工作 |
| `--max-model-len` | 8192 | 包含音频展开 token 的模型上下文预算 |
| `--max-audio-turns` | 256 | 当前模型窗口中的音频轮数上限，不是全会话总轮数 |
| `--context-policy` | `rolling` | 自动滚动；`stop` 保留原先上下文触顶即停的对照行为 |
| `--history-window-seconds` | 120 | 模型当前窗口的音频时长上限；token/轮数可更早触发滚动 |
| `--history-retain-seconds` | 60 | 滚动后的近期音频保留目标，实际仍受 token/轮数约束 |
| `--max-tokens` | 160 | 每轮生成长度上限 |
| `--protocol-form` | `auto` | `auto` / `special` / `text`；必须与权重训练形式一致 |
| `--image-slots` | 1 | Qwen3-Omni 初始化兼容选项；不向翻译模型传入视频画面 |
| `--ffmpeg` | 系统 PATH | 上传音视频解码使用的 FFmpeg 可执行文件路径 |
| `--max-upload-mb` | 4096 | 单文件上传上限及单份 WMV 转码 MP4 预算，各自以 MiB 计 |
| `--upload-timeout-seconds` / `--decode-timeout-seconds` | 3600 / 1800 | 上传及文件解码超时 |
| `--temp-dir` | 系统临时目录 | 上传、解码 PCM 与临时导出的存储位置 |
| `--max-result-mb` | 256 | 单会话临时事件+纯文本预算，达到时显式停止并保留有效前缀 |
| `--result-retention-hours` | 24 | 已结束结果的最长临时保存时间；最多 8 个会话 |
| `--no-log` | 未启用 | 禁用持久 JSONL 日志；完整导出仍使用临时磁盘 |

## 5. 目前的真实边界

**本版支持长会话的资源管理，不宣称无限时间或无损语义记忆。** 默认总时长 8 小时，与模型窗口长度分开。保持原来的 `start_server.sh` 启动方式即可启用；无需更换 checkpoint、重训或重签证书。可以配置更长总时长，但 PCM 传输的 uint32 样本偏移限制约 74.56 小时，超出会在启动时拒绝。

默认滚动步骤：保留 system prompt；每次推理计入当前音频、展开 token、生成预算及结束符；即将超过 120 秒音频、8192 token 或 256 轮中的任一预算时，从最早开始移除完整的“音频 + 原始响应 token”轮次，批量降到近期约 60 秒及相应 token/轮数低水位。必要时 token 预算决定更短窗口。只有新响应完成协议校验、成功提交时才应用裁剪；失败不提前损坏原历史。窗口滚动发出 `context_window` 事件，可在页面记录和 JSON 中追踪。

这属于应用层重建模型上下文，不是 Qwen3-Omni 原生流式编码或底层 KV cache 驱逐；窗口变化也可能降低前缀缓存命中率。WAIT/FLEX/DIRECT 的语义和 SFT 协议保持不变，不在 DIRECT 后清空、不重放源音频、不生成摘要、不改写已输出字幕。旧轮次离开模型窗口后，完整译文仍保存在导出文件中，但模型不再看到那些原始音频/回复。标签不能证明那一段已经语义覆盖，长距离指代、术语一致性和延迟未译内容存在损失风险；120/60 秒是可调工程默认值，不是已证明最优的翻译参数。需要严格保留全历史做对照时，用 `--context-policy stop`，它仍会触顶停止。

先用两方向各 20–40 秒样例检查协议，再用 10–30 分钟真实音频检查滚动边界前后，最后在 A800 上连续运行数小时，记录 `service_s`、积压、显存和译文质量。电脑声音若平均处理速度持续慢于实时，滚动窗口不能创造算力；12 秒积压保护仍会显式终止，不丢音频伪装实时。断网后不自动无损续接，GPU 请求挂死也不强杀线程来接纳下一会话。

文件必须同时可由浏览器播放和服务器解码；WMV/ASF 是例外输入，由服务器先转码为可播放 MP4。若最终播放视频的容器时长与服务器音轨时长相差超过 0.5 秒，页面会在推理前提示转换为 WAV 或音视频等长的 MP4，不静默丢掉尾部；小幅容器/编码差异只在自然播放结束时对齐到服务器音频末尾，手动结束不补读未播放部分。

单张 A800 80 GB 是本项目的配置目标，**并非已经完成真机四路显存/实时性验证**。30B-A3B 的“激活 3B”不等于只需存储 3B 参数。四路共享权重，但活跃请求的 KV cache/计算和各自音频历史仍消耗资源。显存不足可试 `--max-model-len 4096 --max-audio-turns 128 --mm-processor-cache-gb 1`；保持 eager，必要时明确配置批大小，不自动改变人数。若模型权重自身仍放不下，需要另行验证量化或多卡方案，不能靠队列逻辑解决。

浏览器设备采集及音频输出存在固定/可变缓冲，网络传输也有延时。用页面事件检查时机是直观实验，不是严格时钟同步后的延迟基准。实际系统声音共享、可选麦克风权限、目标服务器 CUDA/vLLM 兼容性和 A800 吞吐需要在部署机器验收。

## 6. 测试与定位

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
node static/test_pcm.mjs
node static/test_capture.mjs
node static/test_file_playback.mjs
node static/test_long_session.mjs
node static/test_system_audio.mjs
python run.py --help
```

前端为静态模块，不需要 `npm install`。Python 测试位于 `tests/`，Node 测试覆盖实时声音 PCM、共享音频授权与清理、历史采集适配器、长会话 UI 和当前文件播放进度，均不需要模型或 GPU。真实浏览器上传回归见 `tools/browser_upload_smoke.mjs`，WMV 与电脑声音回归见 `tools/browser_media_sources_smoke.mjs`，使用方式见 [贡献说明](CONTRIBUTING.md)。GitHub Actions 配置会在推送和拉取请求时执行 CPU 测试；这不等于 GPU 部署验收。

常见问题：

- `busy`：活动会话已达到人数上限（默认 4）；暂停或取消后尚在执行的会话也可能暂时占用名额。
- `engine_stalled`：实际模型批次长时间未返回，暂不接纳新任务；不是 WAIT，也不是确认崩溃。在途批次成功返回后恢复接纳。
- `engine_failed`：共享模型后端已确认失败，已提交译文仍可导出；检查服务端 traceback，排除原因后重启服务，不要反复点击开始。
- `model_protocol`：模型没输出合法标签或达到生成长度上限；检查权重/tokenizer 匹配和 `--max-tokens`，不要在前端修造标签。
- `context_limit`：`stop` 模式触顶，或滚动后单个当前音频块仍无法容纳；检查窗口/生成预算，不能靠无限清空历史解决。
- `result_limit` / `media_storage`：临时结果预算或磁盘异常；先导出已提交前缀并检查磁盘，必要时调整 `--max-result-mb` / `--temp-dir`。
- `sequence_gap`：电脑声音/旧 PCM 客户端传输序号或样本偏移不连续；重新开始，不假装已无损续接。
- `backlog_limit`：实时声音持续吞吐慢于输入。先看耗时与队列曲线，再尝试 2 秒块；仅扩大队列会增加延时。上传文件模式不触发该队列限制。
- 浏览器无声音/不解码：换兼容编码（例如 H.264 + AAC 的 MP4），确认系统播放权限和媒体文件音轨；字幕文件请用 UTF-8 `.vtt`。
- HTTP 文件模式仍显示旧的“音频采集需要安全上下文”或“HTTP 兼容采集检测到…”错误：说明页面仍运行旧文件采集代码。部署完整更新、重启 Python 服务，再按 `Ctrl+Shift+R` 刷新；`/api/config` 应包含 `"file_mode":"uploaded"`。新路径不创建浏览器文件采集器。普通远程 HTTP 的电脑声音/麦克风采集被拒绝仍是预期限制。
- `ffmpeg_missing`：服务器缺少 FFmpeg。安装后重试，或先用 16 kHz 单声道 PCM16 WAV；无需为此升级 torch/vLLM。
- `ffprobe_missing` / `transcode_encoder`：WMV 自动播放准备缺少 ffprobe 或 H.264/AAC 编码器，安装完整 FFmpeg 软件包；不是模型故障。
- `media_timeline`：WMV 转换后音视频时长相差超过 0.5 秒，需检查或修复原文件；系统不会静默截短视频后假装完整处理。
- `playback_size` / `transcode_failed`：WMV 转码超过临时视频预算，或原视频缺少可解码音视频轨；检查文件和磁盘预算，不把部分转码结果当作完整视频。
- 模型加载失败：在本地保留完整服务端 traceback。优先复用已经跑通对应模型的环境，不盲目升级 transformers/vLLM；对外提问前先去除私人路径、文件名和访问凭据。

项目结构：

```text
FLEX/
  infer_audio.py               单音频文件流式 S2TT，不需要网页
  requirements-infer.txt       文件推理的轻量依赖入口
  run.py / start_server.sh       启动入口与单卡示例
  requirements*.txt             Web 与测试依赖
  flex_web/server.py            WebSocket、队列、会话隔离、事件日志
  flex_web/scheduler.py         单模型多会话公平排队与小批量调度
  flex_web/results.py           有界内存的磁盘结果与流式导出
  flex_web/tls.py               内网证书自动生成、校验与复用
  flex_web/media.py             临时上传、受限解码、按播放进度读取文件
  flex_web/model.py             本地 vLLM 模型与 SFT 协议；显式 mock 后端
  static/                      页面、交互、音频采集与重采样
  tests/                       无 GPU 协议与边界测试
  docs/PROTOCOL.md              可复用的客户端/服务端协议
  docs/VALIDATION.md            已完成的检查与未验证边界
  docs/S2TT.md                  单音频推理命令与行为说明
  .github/workflows/tests.yml   无 GPU 持续集成
  CONTRIBUTING.md / SECURITY.md 开发约定、安全与隐私
```

## 7. 参考与取舍

- [SimulStream 官方仓库](https://github.com/hlt-mt/simulstream)：参考其 WebSocket 音频输入、增量文本输出及事件日志的服务分层思路。本项目直接适配三标签协议，不引入另一套翻译模型。
- [Qwen3-Omni 官方 Web demo](https://github.com/QwenLM/Qwen3-Omni/blob/main/web_demo.py)：其 `format_history` 按完整多模态 user+assistant 轮次限制模型历史。本项目参考整轮裁剪与显示历史分离的方式，另按 S2TT 时间/token/轮数预算实现滚动，不照搬聊天 demo 的 5 个音频轮参数，也不将其视为同传质量验证。
- [vLLM 0.11.1 LLM 接口](https://github.com/vllm-project/vllm/blob/v0.11.1/vllm/entrypoints/llm.py)：`generate` 接收独立请求列表并按输入顺序返回结果。这里沿用已部署的同步接口，用单线程小批量调度封装；不以此宣称实现了原生异步连续批处理或验证了 Qwen3-Omni 四路 GPU 性能。
- [InfiniSST 官方仓库](https://github.com/LeiLiLab/InfiniSST)：参考流式语音翻译研究的 Web 演示/后端分离方式，不以其模型行为替换本项目的 FLEX 策略。
- [MDN：AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)、[getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)：麦克风使用浏览器音频处理线程，需要安全上下文和用户授权。文件模式独立走上传和播放进度协议。
- [MDN：ScriptProcessorNode](https://developer.mozilla.org/en-US/docs/Web/API/ScriptProcessorNode)：旧文件兼容实现所用的已弃用主线程接口；当前页面不再使用此路径，旧适配器仅保留作历史测试。
- [vLLM 0.11.1 多模态输入预处理源码](https://github.com/vllm-project/vllm/blob/v0.11.1/vllm/inputs/preprocess.py)：独立计数器检查音频展开后的上下文预算，不污染正式引擎的多模态缓存。

界面/服务自动化测试不等于真实模型质量验收。部署后请先测试双向各一条短样例，核对译文、策略、文件播放位置和日志，再开展正式评估。

本地已经完成的检查与未验证边界见 [验证记录](docs/VALIDATION.md)。可运行 `python tools/package_project.py --output /path/to/FLEX-Web.zip` 打包源码；压缩包排除模型、日志、缓存和录音，且不会覆盖已有同名文件。
