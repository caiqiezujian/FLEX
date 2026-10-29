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

## 4. 增量播报行为

- 字幕和策略标签仍按模型输出立即显示。WAIT 无文字，不请求 TTS；FLEX / DIRECT 仅送新增正文。
- 新增可朗读译文到达且请求空闲时立即请求 TTS，不等句末，也没有固定 2 秒文本缓冲。请求在途或预取已满时，只合并队列中尚未提交的相邻片段，不修改已经请求或播放的内容。中文每段最多 80 个 Unicode 字符，英文最多 200 个，同时受服务端 `--tts-max-text-chars` 更小值限制；超长增量在现有文本内优先沿句末或短语边界切分，不等待未来文本。这减少本地等待，实际出声时间仍取决于远端合成耗时。
- 使用独立播放上下文，按顺序朗读，最多提前准备两段音频。播报音量不影响源音频送入模型；文件原声可用播放器音量单独调整。
- 文件自然播放结束后，把最后的待播文字读完。手动结束同传、停止播报、清空结果、重开会话、页面关闭或会话出错，会清空待播队列并忽略迟到响应。取消客户端请求不保证远端引擎立即停止已经开始的合成。
- TTS 错误/超时不会终止文字推理，也不自动重复重试。等候超过 30 秒、排队超过 12 段或待播音频超过 30 秒时，明确停止播报并提示；不会悄悄删字幕或修改译文。重新勾选后从新增译文继续。
- API 一次返回一段完整 WAV，因此这是**分段合成与播放**；不是 TTS 首包音频流。实际听感还取决于上游合成耗时、分段和网络。
- 电脑声音模式仅在浏览器确认共享的是标签页时允许播报。共享整屏/窗口、无法识别共享范围时会关闭播报，避免 TTS 再次进入源音频形成循环。应选择其他音视频标签页；包含麦克风时建议戴耳机，防止扬声器声被麦克风收到。单纯戴耳机不能解决整机音频回采。

## 5. 可配置与排障

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--tts-url` | 空 | 不配置则关闭 TTS |
| `--tts-assets-dir` | 无 | Web 服务器上可读的参考目录 |
| `--tts-voices-config` | 无 | 自定义音色及准确参考转写 |
| `--tts-timeout-seconds` | 180 | 后端单请求总超时；浏览器另有 30 秒积压限制 |
| `--tts-max-text-chars` | 200 | 单次最大文本字符数，需不超过上游服务限制 |
| `--tts-max-concurrent` | 2 | 整个 Web 进程的 TTS 在途请求上限；超限显式报忙 |
| `--tts-x-vector-only` | 不传 | 保留上游默认值；可明确传 true/false |

参考缺失时，Web 日志报告具体文件，页面禁用播报；文本服务继续。服务连通性不是启动时的 GPU 检查，由独立脚本或页面试听验证。403/401 检查原有访问域/令牌；503 检查上游是否就绪及并发；400/422 检查参考音频、参考文本和上游长度约束。

浏览器只访问当前 Web 站点，不直接请求跨域 HTTP TTS，避免 HTTPS 混合内容和 CORS 配置问题。服务地址与参考路径由服务端管理员配置，浏览器请求只接收 `text` 和 `language`。音色和合成内容不写入 Git。

实现参考：[HTTPX 异步客户端](https://www.python-httpx.org/async/)、[MDN Web Audio 用户手势与播放控制](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices)。
