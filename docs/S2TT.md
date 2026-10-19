# 单音频流式 S2TT

先验证训练后的模型，再接网页或实时麦克风。`infer_audio.py` 接收一个服务器本地音频文件，按时间顺序分块生成目标语言文本。它不做 TTS，不打开浏览器，也不播放音频。

这里的“上传”只需要把音频传到 GPU 服务器，随后通过 `--audio` 指定路径；不需要额外的 HTTP 上传接口。若命令运行在服务器，音频路径也必须是服务器路径。

## 1. 环境与模型

- 使用已跑通 Qwen3-Omni 的 Linux GPU 环境。目标为单张 A800 80 GB，当前尚未在真实 checkpoint 上验证显存和实时性能。
- 真实推理复用 `flex_web/model.py`，面向 vLLM 0.11.1；无需启动 `run.py`。它不依赖项目外的 `Scripts/`。
- 模型是训练后的完整 merged checkpoint，包含匹配的 tokenizer/processor，不是单独 LoRA adapter。同一个双向 checkpoint 通过目标语言 prompt 切换方向。
- 不要仅把 `infer_audio.py` 单文件拷走；保留同目录下的 `flex_web/` 模块。无需安装 Web 依赖。

```bash
python -m pip install -r requirements-infer.txt
python infer_audio.py --help
```

`requirements-infer.txt` 只包含 NumPy，不安装或升级 torch、transformers、vLLM/CUDA。16 kHz、单声道、PCM16 WAV 可直接读取；其他格式、采样率或多声道通过 FFmpeg 解码为 16 kHz 单声道 PCM。FFmpeg 必须可执行，可用 `--ffmpeg /absolute/path/to/ffmpeg` 指定。程序不生成整段临时 WAV。

## 2. 英→中

最简单的入口（独立 special-token checkpoint）：

```bash
CUDA_VISIBLE_DEVICES=6 bash run_infer.sh /absolute/path/to/checkpoint-merged /absolute/path/to/english.wav --direction en-zh
```

启动脚本默认 1 秒分块、fast、special；尾随参数可以覆盖默认值。它不切换当前目录，所以相对路径保持原含义。无需运行 Web 服务。独立五文件包见 [快速使用说明](S2TT-QUICKSTART.md)，可用 `python tools/package_s2tt.py` 构建；包内将共享模型模块命名为 `flex_model.py`，不依赖原仓库。

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python infer_audio.py \
  --model /absolute/path/to/checkpoint-merged \
  --audio /absolute/path/to/english.wav \
  --direction en-zh \
  --chunk-seconds 1 \
  --pace realtime \
  --output-dir outputs/en-zh-first
```

## 3. 中→英

```bash
CUDA_VISIBLE_DEVICES=0 python infer_audio.py \
  --model /absolute/path/to/checkpoint-merged \
  --audio /absolute/path/to/chinese.wav \
  --direction zh-en \
  --chunk-seconds 1 \
  --pace realtime \
  --output-dir outputs/zh-en-first
```

目标文本通过每轮增量追加，不去重改写，也不裁剪英文前导空格。终端中模型/运行环境自身也可能打印日志；**纯译文以 `translation.txt` 为准**，不要把整个终端重定向内容当作纯译文。

默认 `--protocol-form auto` 只检查 tokenizer；它不能证明权重确实按该形式训练。确认使用三个独立 special token 训练时可加 `--protocol-form special`；历史 text-form checkpoint 用 `--protocol-form text`。不要为了通过检查而自行给 tokenizer 加 token。

## 4. 两种运行节奏

| 参数 | 音频输入方式 | 适合用途 |
|---|---|---|
| `--pace realtime`（默认） | 第一块截止 1 秒才送入，第二块截止 2 秒才可送入，以绝对回放时钟调度 | 检查文件模拟在线时序与模型处理是否落后 |
| `--pace fast` | 上一轮完成后立即处理下一块，不等待真实时间 | 尽快检查翻译文本和三标签行为 |

两者都是分块 S2TT：模型在第 N 轮只看到截至当前块的音频，以及此前已经提交的响应，不会看到后续音频或参考译文。固定 chunk 的内容相同；fast 不等于整段离线翻译。这里的“流式”指按 SFT 多轮协议持续追加输入，不是修改模型编码器实现内部原生流式计算，也不是生成 token 级实时推送。

文件本身留在磁盘，推理慢时下一块晚一些读取和提交，不需要另建一个持续增长的音频生产队列，不丢弃中间块。realtime 记录 `dispatch_lag_s`，表示当前块开始处理时，相对它本应在 1 倍速回放中到齐时刻的落后；推理慢时该值会增加。它不是纯 GPU 排队时间，也不是 AL/LAAL，更不能代替真实麦克风/网络延迟实验。

默认 1 秒，建议先使用训练同分布的 20–40 秒音频，分别验证两个方向，再比较 1 秒与 2 秒块。此入口只用固定分块，不做 VAD 或自适应合并，便于先观察模型本身的策略行为。

## 5. WAIT / FLEX / DIRECT 与尾块

1. 每块音频形成一个新的 user 轮，模型生成一个完整响应后才对外提交。
2. WAIT：保留本轮音频和 WAIT 响应到历史，TXT 不追加文字；不是清空音频，也不是重复发送同一块。
3. FLEX / DIRECT：验证协议后立即追加新增译文；两者都不允许事后回改。
4. 少于一块的最后尾音仍处理一次。短于 `min(1 秒, chunk 长度)` 时仅为模型输入补静音，真实源音频时长不增加。
5. 文件结束不额外构造一个静音轮强迫翻译。最后仍为 WAIT 时明确警告：输入处理完毕不等于最后一句语义一定翻译完整。
6. 超过总输入预算、格式错误、单块仍无法放入上下文或截断生成会报错退出，保留已提交前缀。错误不会伪装成 WAIT，失败响应不会写入纯译文。

预热使用独立会话，发生在文件回放时钟开始之前，不进入真实翻译历史。排查问题时可以加 `--no-warmup`。

## 6. 输出与限制

`model_ready` 记录模型适配层的版本/配置；每轮 `turn` 和协议失败的 `rejected_turn` 也记录原始 `token_ids`。协议错误同时在终端显示原始回复，避免只得到泛化的前缀错误。

文件入口默认 `--image-slots 1`，绕过受影响 vLLM 的 Qwen3-Omni 视觉禁用初始化问题；请求仍只有音频。它可能增加显存占用，已修复的版本可用 `--image-slots 0`。详见 [上游问题](https://github.com/vllm-project/vllm/issues/49384)。Web 入口默认值不随此文件入口改变。

```text
outputs/en-zh-first/
  translation.txt   精确保留增量空白的纯译文，没有协议标签
  events.jsonl      start、model_ready、turn、context_window、done 或 error；无效响应记为 rejected_turn
```

`turn` 包含 `tag`、`delta`、原始 `response`、源音频起止位置、实际模型输入长度、`service_s`、`dispatch_lag_s` 和 prompt token 数。`generation_finished_elapsed_s` 是从回放起点到生成/提交检查完成的墙钟时间，不是字幕实际渲染时刻；日志写入及终端输出发生在其后。源音频区间是本轮输入块位置，不是译文内容的精确语义对齐。统计中的 `read_audio_s` 仅代表已经读取的前缀，不代表完整文件时长；发生错误时两者可能不同。

未指定 `--output-dir` 时自动在 `outputs/` 创建带 UTC 时间与随机后缀的目录。显式指定的目录必须不存在，重复运行请换新目录。文件边处理边 flush，异常退出后已有内容只是部分结果。退出码：0 为输入正常处理结束，1 为运行错误，130 为用户中断；最后 WAIT 另通过警告和 `final_wait` 标识。

主要限制默认值：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--max-audio-seconds` | 28800 | 输入总预算（8 小时），超过显式失败，不把前缀冒充完整翻译 |
| `--max-model-len` | 8192 | 模型上下文预算，包含音频展开 token |
| `--max-audio-turns` | 256 | 当前窗口音频轮数上限，不是全会话总轮数 |
| `--context-policy` | `rolling` | 自动滚动；`stop` 保留旧触顶停止行为 |
| `--history-window-seconds` / `--history-retain-seconds` | 120 / 60 | 窗口音频上限 / 滚动后的保留目标（秒） |
| `--max-tokens` | 160 | 每轮生成上限；截断不能当作成功 |
| `--gpu-memory-utilization` | 0.90 | vLLM 显存配置 |

默认按完整音频+原始响应轮次滚动，system 保留；token/轮数可在音频达到 120 秒前先触发，必要时保留少于 60 秒。只在新响应验证并成功提交后应用裁剪，不按 WAIT/FLEX/DIRECT 标签决定清空。`turn.context` 与 `context_window` 记录累计统计，全局源时间轴和轮次不重置。纯译文始终增量写入磁盘，不保留无限增长的内存副本。

这只是应用层有界上下文，不是底层 KV 驱逐或原生流式编码。旧音频和回复离开模型窗口可能影响指代、术语和尚未表达的语义；120/60 是工程默认值，需要真实 checkpoint 长音频评估，不保证无漏译。严格全历史对照使用 `--context-policy stop`，上下文/轮数触顶仍停止。详见 [Web 长会话边界](../README.md#5-目前的真实边界)。

日志包含文件名、译文、错误信息，可能含私人路径；只应在受控环境保存和分享。默认 `outputs/` 已被 `.gitignore` 排除，自定义输出目录需自行保护。

## 7. 无 GPU 链路检查

```bash
python infer_audio.py \
  --backend mock --audio /absolute/path/to/test.wav \
  --direction en-zh --pace fast --output-dir outputs/mock-first
```

mock 会明确输出演示文字，**不会识别或翻译音频**。它只验证文件读取、分块、协议、输出和错误处理。真实翻译必须使用 `--backend vllm`（默认值）和训练后的 `--model`。
