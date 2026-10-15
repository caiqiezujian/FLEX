# 开发与测试

本仓库仅包含 FLEX Live Web 推理系统。模型权重、训练数据、训练脚本与蒸馏 prompt 不随此仓库发布。

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
python run.py --help
git diff --check
```

Python 测试不加载 checkpoint，不要求 GPU。Node 测试覆盖重采样、PCM 编码、AudioWorklet 暂停/结束/取消；本地曾在 Node 18.17 验证，持续集成使用 Node 22。GitHub Actions 同时检查 Python 3.10、3.11；具体通过状态以对应运行记录为准。

涉及界面或音频链路的变更还需手动检查：

1. 选择浏览器可播放的短音视频，确认只在实际播放时发送音频。
2. 检查暂停/恢复、自然结束及手动结束后的尾块处理。
3. 确认正文只有新增译文，WAIT 不追加内容，FLEX/DIRECT 标签保留在时间线。
4. 检查 TXT / JSON 导出、双向切换与小屏布局。
5. 改动模型接口后，在真实 checkpoint 上分别验证中→英、英→中。

请在变更说明中区分“自动化测试通过”“浏览器 mock 验证”和“真实 GPU 验证”，不要将它们混写成模型效果证明。

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
