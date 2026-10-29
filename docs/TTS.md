# 译文播报接入

Web 继续使用原来的双向同传模型，另通过 HTTP 调用现有 Qwen3-TTS 服务。Web 进程不加载 TTS 权重、ONNX 或第二个 GPU 模型。未配置 TTS 时仍正常显示字幕。

## 1. 准备参考音色

服务地址按本次提供的信息填写为 `http://7.242.105.6:9100`。Web 后端调用 `POST /api/tts/clone`，发送 `ref_audio_b64 / ref_text / text / language / request_id`；收到 `audio_b64` 后校验 PCM16 单声道 WAV，再交给浏览器播放。

参考 WAV **必须能被 Web 进程读取**。`/data/l00481362/qwen3tts_server/diy_asset` 是 72 机器上的目录，只有 Web 也在该机器、且容器挂载了该目录时，才能直接使用此路径。API 接收的是音频字节，不会替 Web 读取另一台机器的路径。

若 Web 在其他机器，把下面两个 WAV 原样复制到 Web 服务器，例如：

```text
/data06/yb/Code/FLEX/tts_assets/
  minimax_en_1_f.wav
  minimax_zh_1_f.wav
```

默认参考文本已按用户提供的内容内置：

| 目标语言 | WAV | 对应参考转写 |
|---|---|---|
| English | `minimax_en_1_f.wav` | Hello. I'm delighted to assist you with our voice services. Choose a voice that resonates with you, and let's begin our creative audio journey together. |
| 中文 | `minimax_zh_1_f.wav` | 您好，很高兴能为您提供配音服务。选择您感兴趣的音色，让我们一起开启声音创作的奇幻之旅吧。 |

无需复制服务端的 ONNX、YAML 或权重。若改用别的音色，需要同时改为与 WAV 内容匹配的参考文本。可创建自定义 `voices.json`：

```json
{
  "en": {"audio": "minimax_en_1_f.wav", "text": "实际英文参考音频的完整转写"},
  "zh": {"audio": "minimax_zh_1_f.wav", "text": "实际中文参考音频的完整转写"}
}
```

启动时附加 `--tts-voices-config /绝对路径/voices.json`。相对音频路径以 `--tts-assets-dir` 为根；未指定 assets-dir 时，以 JSON 所在目录为根。不要直接使用上表以外的文件却仍保留默认参考文本。程序不会自动猜测原服务 JSON/YAML 的结构。

## 2. 安装与单独检查

在运行 Web 的 Python 环境安装轻量 HTTP 客户端；无需修改 torch/vLLM：

```bash
cd /data06/yb/Code/FLEX
python3 -m pip install 'httpx>=0.27,<1'

python3 tools/check_tts.py \
  --url http://7.242.105.6:9100 \
  --assets-dir /data06/yb/Code/FLEX/tts_assets \
  --language zh \
  --output /data06/yb/Code/FLEX/outputs/tts_preview_zh.wav
```

用 `--language en` 和另一个 output 文件名检查英文。工具不会覆盖已有 WAV。检查只调用 TTS，不加载同传模型；仍需播放输出 WAV 检查音色和内容。客户端关闭环境代理，内网服务直接连接。

## 3. 启动 Web

保留已验证的模型、GPU、端口、HTTPS 参数，在原启动命令后追加：

```bash
--tts-url http://7.242.105.6:9100 \
--tts-assets-dir /data06/yb/Code/FLEX/tts_assets
```

完整命令示例（模型目录需确实包含 config.json 和合并权重）：

```bash
cd /data06/yb/Code/FLEX
CUDA_VISIBLE_DEVICES=0 bash start_server.sh \
  /data06/yb/Code/FLEX/models/checkpoint-4410-merged \
  --tts-url http://7.242.105.6:9100 \
  --tts-assets-dir /data06/yb/Code/FLEX/tts_assets
```

`start_server.sh` 保留原有 HTTPS 默认地址。新机器需沿用自己的证书和访问域名设置；新增 TTS 参数不改变原部署地址。也可直接 `python3 run.py --backend vllm --model ...` 并明确配置原有的 HTTPS/端口参数。

页面新增“朗读译文”“音量”“试听音色”“停止播报”。默认关闭，首次由用户点击启用。英→中自动用中文参考音色，中→英自动用英文参考音色。试听会发出真实合成请求。

试听的合成和播放期间，试听按钮会禁用，避免重复点击取消旧请求、又创建新请求。试听文本立即提交，浏览器声音授权与请求并行进行，声音播放仍等授权完成；授权超过 10 秒未完成会明确提示。试听按配置的 TTS 请求超时等待，可随时停止；试听失败不会取消“朗读译文”的勾选。

## 4. 增量播报行为

- 字幕和策略标签仍按模型输出立即显示。同一个 `turn` 事件把新增正文同时交给字幕与 TTS 队列；不扫描、重送右侧历史全文。WAIT 无文字，不请求 TTS；FLEX / DIRECT 仅送新增正文。每个播报片段关联其原始输出轮次，合并未提交文字时保留首末轮次。页面显示正在朗读的轮次和“已返回 / 已开播”段数。
- 新增可朗读译文到达且请求空闲时立即请求 TTS，不等句末，也没有固定 2 秒文本缓冲。请求在途或预取已满时，只合并队列中尚未提交的相邻片段，不修改已经请求或播放的内容。中文每段最多 80 个 Unicode 字符，英文最多 200 个，同时受服务端 `--tts-max-text-chars` 更小值限制；超长增量在现有文本内优先沿句末或短语边界切分，不等待未来文本。这减少本地等待，实际出声时间仍取决于远端合成耗时。
- 使用独立播放上下文，最多同时请求两段（不超过服务器公开的并发上限），播放当前段时提前准备后两段。已返回和请求中的片段合计最多两段，不包含当前正在播放的一段。每个请求按提交顺序编号，即使 C 先于 B 返回，也只按 A→B→C 播放。播报音量不影响源音频送入模型；文件原声可用播放器音量单独调整。
- 文件自然播放结束后，把最后的待播文字读完。手动结束同传、停止播报、清空结果、重开会话、页面关闭或会话出错，会取消所有在途请求、清空待播队列并忽略迟到响应。Web 在客户端断开后取消自己的上游 HTTP 调用、释放本地请求名额；这不能保证远端引擎立即停止已经开始的 GPU 合成。
- 已准备音频达到 30 秒时，暂缓新请求，继续播放已有音频。已返回音频不会因超过该阈值而被清空；最多两段预取限制仍生效。实时待播片段超过 30 秒会显示积压提示，继续按顺序播放，不把排队时间误报为合成请求耗时。这是缓冲管理，不是降低实际延迟的保证。
- 只有 Web 明确返回 `503 + tts_busy`（请求尚未提交上游）时，才按 1、2、4 秒间隔最多重试三次，始终保留同一段文字及顺序。其他错误、网络中断和超时不自动重试，避免重复合成；页面显示 HTTP 状态及返回的错误码。待播文字/请求/就绪片段总数超过 12 段时明确停止，保留字幕；不会悄悄跳过译文。重新勾选后从新增译文继续。
- API 一次返回一段完整 WAV，因此这是**分段合成与播放**；不是 TTS 首包音频流。实际听感还取决于上游合成耗时、分段和网络。
- 电脑声音模式仅在浏览器确认共享的是标签页时允许播报。共享整屏/窗口、无法识别共享范围时会关闭播报，避免 TTS 再次进入源音频形成循环。应选择其他音视频标签页；包含麦克风时建议戴耳机，防止扬声器声被麦克风收到。单纯戴耳机不能解决整机音频回采。

## 5. 可配置与排障

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--tts-url` | 空 | 不配置则关闭 TTS |
| `--tts-assets-dir` | 无 | Web 服务器上可读的参考目录 |
| `--tts-voices-config` | 无 | 自定义音色及准确参考转写 |
| `--tts-timeout-seconds` | 180 | 后端单请求总超时；浏览器给传输及有限重试多留 5 秒；实时积压超过 30 秒提示，不再提前取消正常音频 |
| `--tts-max-text-chars` | 200 | 单次最大文本字符数，需不超过上游服务限制 |
| `--tts-max-concurrent` | 2 | 整个 Web 进程的 TTS 在途请求上限；超限显式报忙 |
| `--tts-x-vector-only` | 不传 | 保留上游默认值；可明确传 true/false |

参考缺失时，Web 日志报告具体文件，页面禁用播报；文本服务继续。服务连通性不是启动时的 GPU 检查，由独立脚本或页面试听验证。403/401 检查原有访问域/令牌；本地 503 的 `tts_unavailable` 表示 Web 未就绪，`tts_busy` 表示 Web 在途请求已满；上游非成功状态映射为 502 / `tts_upstream`，超时为 504 / `tts_timeout`。应保留失败请求的完整错误码，不以另一次成功请求代替对失败原因的判断。

页面状态中的“请求最长 N 秒”只从 HTTP 提交时开始计算，不包含此前排队。浏览器控制台执行 `window.flexTtsDiagnostics()` 可查看最近 200 条阶段记录：`request_start`、`audio_ready`、`play_start`、`play_end`，包含请求等待、音频解码、已就绪到开播的等待、译文到开播的总等待、片段与 S2TT 首末轮次；不保存正文或音频。可复制 `JSON.stringify(window.flexTtsDiagnostics(), null, 2)` 结果排障。

成功或映射错误响应包含 `X-TTS-Request-ID`、`X-TTS-Service-Seconds`；完成的阶段另有 `X-TTS-Upstream-Headers-Seconds`、`X-TTS-Upstream-Body-Seconds`、`X-TTS-Decode-Seconds`。服务时间不包含浏览器排队、下载完成后的浏览器解码或播放等待。`tools/check_tts.py` 输出 `elapsed_seconds` 与 `diagnostics`，并支持 `--x-vector-only true/false` 对齐实际启动参数。

浏览器只访问当前 Web 站点，不直接请求跨域 HTTP TTS，避免 HTTPS 混合内容和 CORS 配置问题。服务地址与参考路径由服务端管理员配置，浏览器请求只接收 `text` 和 `language`。音色和合成内容不写入 Git。

实现参考：[HTTPX 异步客户端](https://www.python-httpx.org/async/)、[MDN Web Audio 用户手势与播放控制](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices)。

调度参考：[RealtimeTTS 的文本队列、独立合成与播放实现](https://github.com/KoljaB/RealtimeTTS/blob/master/RealtimeTTS/text_to_stream.py)。本项目复用现有 HTTP TTS 服务，两个请求并行及返回排序是针对当前接口的实现选择，不依赖安装 RealtimeTTS。
