# 开发与测试

本仓库包含 FLEX Live Web 推理系统及 `infer_audio.py` 单音频入口。模型权重、训练数据、训练脚本与蒸馏 prompt 不随此仓库发布。

## 本地开发

在独立 Python 环境中安装 CPU/Web 依赖：

```bash
python -m pip install -r requirements-dev.txt
python run.py --backend mock --host 127.0.0.1 --port 8000
```

浏览器打开 `http://localhost:8000`。静态 HTML/CSS/JavaScript 直接提供服务，无需 npm 或前端打包。改动静态文件后刷新页面；改动 Python 后重启服务。mock 只验证链路，不做语音识别或翻译。

真实模型开发应复用已验证的 Linux CUDA/vLLM 环境，不将 CPU 开发环境当作 GPU 环境模板。

## 提交前检查

```bash
python -m pytest -q tests
node static/test_pcm.mjs
node static/test_capture.mjs
node static/test_file_playback.mjs
python run.py --help
python infer_audio.py --help
git diff --check
```

Python 测试不加载 checkpoint，不要求 GPU；`tests/test_infer_audio.py` 覆盖独立文件推理，上传媒体测试覆盖鉴权、解码、逐块释放和清理。Node 测试覆盖重采样、PCM 编码、AudioWorklet、文件播放进度和历史兼容适配器；新文件模式不再使用历史兼容适配器。持续集成使用 Node 22；具体通过状态以对应运行记录为准。

涉及界面或音频链路的变更还需手动检查：

1. 选择浏览器可播放的短音视频，确认先上传、后播放，模型只收到已播放的音频前缀；文件 WS 不应发送二进制 PCM。
2. 检查暂停/恢复、自然结束及手动结束后的尾块处理。
3. 确认正文只有新增译文，WAIT 不追加内容，FLEX/DIRECT 标签保留在时间线。
4. 检查 TXT / JSON 导出、双向切换与小屏布局。
5. 改动模型接口后，在真实 checkpoint 上分别验证中→英、英→中。
6. HTTP 文件模式必须在真实非安全 origin 下验证，例如测试域名通过浏览器 host-resolver 映射本机；不能仅用 localhost（它是可信例外），也不能通过关闭浏览器安全检查伪装通过。同时确认 HTTP 麦克风被单独阻止。

请在变更说明中区分“自动化测试通过”“浏览器 mock 验证”和“真实 GPU 验证”，不要将它们混写成模型效果证明。

### 可选：真实 HTTP 浏览器回归

在 Node 20+ 测试环境安装 Playwright 及其 Chromium（仅开发测试依赖，生产部署不需要），启动本地 mock 服务后执行：

```bash
python run.py --backend mock --host 127.0.0.1 --port 8000 --no-log
# 另一个终端执行；Playwright 需已安装在 Node 模块搜索路径中。
node tools/browser_upload_smoke.mjs --base-url http://127.0.0.1:8000
```

脚本只接受本地 mock 服务，自动生成合成 WAV 并清理，不接触模型权重或用户麦克风。使用非安全测试域名 `flex-http.test`（仅在测试浏览器内映射 127.0.0.1），不设置安全豁免 flags；文件模式在 HTTP 与 localhost 都不应创建 AudioContext/采集器。人为阻塞主线程后应继续完成文件，而不是把“成功停止”当作本需求的通过标准。`--fixture /path/to/test.mp4` 可测试自备的合成视频；不要提供私密媒体。可用 `FLEX_PLAYWRIGHT_MODULE` 指定 Playwright 包的绝对路径，用 `FLEX_BROWSER_EXECUTABLE` 指定已安装的 Chrome/Edge。启用服务认证时，另在测试终端设置相同的 `FLEX_ACCESS_TOKEN`。旧 `browser_http_smoke.mjs` 仅保留为历史采集实现的测试记录，不用于当前文件上传模式验收。

紧凑布局及网页 1 / 2 / 3 秒分块回归：在未设置令牌、默认最大块长 3 秒的本地 mock 服务运行 `node tools/browser_layout_smoke.mjs --base-url http://127.0.0.1:8000`。检查 1366、1920、900、390 像素宽度、字号、设置面板、隐藏认证输入，以及三种块长实际经过上传/WS 到后端的结果。可加 `--screenshots` 保存预览到新建的系统临时目录；输出路径会打印出来，图片不会进入仓库。使用相同的 Playwright/浏览器环境变量。

### HTTPS 麦克风与上传回归

```bash
python run.py --backend mock --host 127.0.0.1 --port 31220 \
  --https-host localhost --cert-dir certs --no-log
# 另一个终端运行，环境变量与上面的 Playwright 测试相同：
node tools/browser_https_smoke.mjs --base-url https://localhost:31220 \
  --ca-cert certs/root-ca.crt --certfile certs/server.crt
```

只连接本地 mock，不访问物理麦克风、不加载 GPU 模型。Node 使用指定根 CA 验证 TLS 链、主机名和实际叶证书；隔离的测试浏览器仅为该叶证书公钥设置 SPKI 例外，使用浏览器合成麦克风执行真实 AudioWorklet → WSS → 协议响应 → 停止释放设备，再验证 HTTPS 文件上传与固定切块。没有全局关闭证书校验，没有修改电脑根证书存储；这不等于已经验证了用户电脑的系统证书导入。生产使用者应按 README 正常信任专用根 CA。

Python TLS 测试另行覆盖真实 OpenSSL 生成、SAN、可信/不可信握手、重启复用、换 IP 和到期续签、不完整 CA 拒绝；无 OpenSSL 的环境会跳过涉及生成的部分，发布时应确认这些测试实际执行。

## 必须保留的行为

- 译文 append-only：不修改、去重改写或替换已提交的内容。
- WAIT 是模型策略，不是错误兜底，不自动重放同一段音频。
- 原始协议响应进入历史；前端显示纯译文不能改变模型历史。
- 同一模型串行推理；取消/断连后，在途请求结束前不能释放会话名额。
- 队列、上下文与会话长度都有明确上限；不能静默丢帧或清空历史。
- 不添加或重排 checkpoint 的 tokenizer token，不自动修改用户模型文件。
- 认证信息不能出现在 URL、事件日志、导出或错误报告中。

## 发布范围

提交前检查 `git status` 和 `git diff --cached --stat`。不要使用强制添加绕过 `.gitignore` 发布模型、录音、字幕、日志或密钥。诊断信息先脱敏，最小复现优先使用合成音频和虚构文本。

源码打包：

```bash
python tools/package_project.py --output /path/to/FLEX-Web-source.zip
```

打包采用允许列表，不包含 `.git`、模型、运行日志或录音；输出文件若已存在会报错，不覆盖已有归档。
