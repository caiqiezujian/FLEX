# FLEX-S2TT：上传一个音频文件，逐块生成译文

这是独立的文件推理包。无需启动网页、HTTP 服务、端口、访问口令或浏览器音频采集。把音频文件上传到 GPU 服务器后，指定文件路径即可。

## 1. 环境

使用已经能加载你的 Qwen3-Omni checkpoint 的 Linux / Python / CUDA / vLLM 环境。不自动安装或升级 GPU 依赖；目标是单张 A800 80 GB，真实显存和速度以服务器实测为准。模型必须是完整 merged checkpoint，且包含与训练一致、已经补齐的 tokenizer/processor。

包内只有五个文件，必须放在同一目录：

```text
FLEX-S2TT/
  infer_audio.py             读音频、逐块推理、保存结果
  flex_model.py              本地模型加载与 SFT 多轮协议
  run_infer.sh               一条命令启动
  requirements-infer.txt     仅 NumPy；不安装 Web 或 GPU 框架
  README.md
```

16 kHz、单声道、PCM16 WAV 直接读取。MP3、FLAC、M4A、其他采样率或多声道文件需要已有 FFmpeg，可用 `--ffmpeg /path/to/ffmpeg` 指定。不生成整段临时 WAV，也不把完整音频一次送入模型。

先停止之前占用同一 GPU 的 Web 服务，避免重复加载两份权重。若环境已有 NumPy，无需安装任何新 Python 包；缺少时执行 `python -m pip install -r requirements-infer.txt`。

## 2. 一条命令运行

在解压后的目录执行，替换两个路径：

```bash
CUDA_VISIBLE_DEVICES=6 bash run_infer.sh \
  /absolute/path/to/checkpoint-merged \
  /absolute/path/to/english.wav \
  --direction en-zh
```

中→英只改成 `--direction zh-en`。物理 GPU 6 在当前进程中显示为逻辑 `cuda:0`，正常，不需要再传设备编号。不要同时设置第二个模型服务。脚本不切换工作目录，相对模型/音频/输出路径按你执行命令时的目录解释。

启动脚本默认：1 秒固定分块、`--pace fast`、`--protocol-form special`、单卡、bfloat16、显存利用率 0.90、上下文 8192。这里的 special 对应训练约定的 WAIT / DIRECT / FLEX 独立 ID 151676 / 151677 / 151678；只检查，不自动修改 tokenizer。历史普通文本标签模型可明确覆盖为 `--protocol-form text`。

如需模拟音频按真实时间到达，在命令后追加 `--pace realtime`。如需 2 秒分块，在命令后追加 `--chunk-seconds 2`。先用训练同分布的 20～40 秒音频验证，再比较不同块长。

直接运行 Python 也可以（Python 入口的默认 pace 是 realtime，下面显式使用 fast）：

```bash
CUDA_VISIBLE_DEVICES=6 python -u infer_audio.py \
  --model /absolute/path/to/checkpoint-merged \
  --audio /absolute/path/to/english.wav \
  --direction en-zh --chunk-seconds 1 --pace fast \
  --protocol-form special
```

全部参数：`python infer_audio.py --help`。`PYTHON=/path/to/python bash run_infer.sh ...` 可指定解释器。

## 3. 输出

默认在当前目录的 `outputs/` 下自动新建一个带时间戳和随机后缀的目录，每次运行独立保存：

```text
outputs/s2tt_.../
  translation.txt     按顺序追加的纯译文，不包含 WAIT / FLEX / DIRECT 标签
  events.jsonl        每轮原始回复、token ID、标签、音频区间、耗时与错误
```

可以追加 `--output-dir /path/to/new-result` 指定输出位置；该目录必须尚不存在，避免覆盖已有结果。文件逐轮写入并 flush。终端也输出增量译文和运行日志，纯文本以 `translation.txt` 为准。

`events.jsonl` 包含 start、model_ready、turn、done；协议失败另有 rejected_turn 和 error。遇到不带标签的响应，终端及日志会保留原始回复与 token ID，不再只看到笼统错误；无效回复不会混入译文。日志可能包含文件名、译文或错误中的私人路径，分享前请检查。

## 4. 流式逻辑

- 每次只给模型一个新的音频块，并携带此前音频和模型响应历史；不提前提供后面的音频或参考译文。
- WAIT：本轮音频和 WAIT 一起进入历史，不输出文字，不重复提交同一块。
- FLEX / DIRECT：逐轮校验后立即追加新增译文，不修改已输出的内容；策略由模型决定。
- fast：前一轮完成后就处理下一块；仍然是流式多轮推理，不是整段离线翻译。
- realtime：以绝对时钟模拟 1 倍速输入，块的最后一个样本到达后才能开始处理。模型慢时顺序继续、不丢块，文件留在磁盘；`dispatch_lag_s` 记录相对模拟到达时间的落后。它不是 AL/LAAL，也不是麦克风/浏览器延迟。
- EOF：最后不足一块的音频也处理；若短于 `min(1 秒, chunk 长度)`，仅为模型输入补静音，源音频时间不变。不额外添加静音轮或强迫 DIRECT。最后仍 WAIT 会明确警告，不保证语义已全部译完。

这是按 SFT 会话格式实现的逐块输入、逐轮输出，不是修改底层音频编码器实现原生流式，也不是逐 token 显示。

## 5. 兼容设置与限制

文件入口默认 `--image-slots 1`，保留一个图像初始化名额，以绕过受影响 vLLM 中 Qwen3-Omni 关闭视觉时的 `meta / cuda` 启动错误。实际请求始终只传音频，不上传图片。它可能增加显存占用；已修复的 vLLM 可显式使用 `--image-slots 0`。上游问题与修复：[Issue #49384](https://github.com/vllm-project/vllm/issues/49384)、[PR #49397](https://github.com/vllm-project/vllm/pull/49397)。此适配层原本面向 vLLM 0.11.1 接口，不承诺所有版本都兼容；请先使用你现有可工作的环境。

默认最长音频 180 秒、最多 256 个音频轮、上下文 8192、每轮输出最多 160 token。上下文或轮数可能先到上限，不保证每段 180 秒都能完成。可分别用 `--max-audio-seconds`、`--max-audio-turns`、`--max-model-len`、`--max-tokens` 调整。程序不悄悄清空历史，也不丢弃 WAIT 历史来假装支持无限长音频。

出错时退出码为 1，Ctrl+C 为 130，保存的只是已提交前缀；正常读到 EOF 为 0，最终 WAIT 另行标记。不要把错误前的部分译文当成完整结果。

本包进行了 CPU 协议、分块、文件解码、时间调度、打包独立性和命令行测试；没有在你的 A800 或真实 checkpoint 上代跑，翻译质量及实时性能仍需实测。
