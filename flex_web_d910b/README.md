# FLEX Web 昇腾 910B 实验适配

本目录从同仓库当前 `flex_web` 派生，继续使用项目根目录的 `static/` 和现有 WAIT/FLEX/DIRECT 双语 SFT 协议。原有 Web、训练和推理文件不受影响。此版本是**实验适配**：在真实 910B、当前 SFT checkpoint 及多轮音频上尚未验证，不能保证启动、翻译质量、速度或多人并发。

## 显存与环境

Ascend 910B 设备使用 `ASCEND_RT_VISIBLE_DEVICES` 选择，而非 `CUDA_VISIBLE_DEVICES`。若 checkpoint 是未量化的 Qwen3-Omni 30B-A3B，30B 是总参数、A3B 是每 token 激活规模；BF16 粗略按 2 字节/参数就约 60 GB（十进制）。vLLM 的文本输出路径可能不加载整个 checkpoint 的全部模块，具体应以加载结果为准；此外还需要运行时和 KV 缓存。因此单张 32 GiB 910B4 能否容纳必须实测，1～2 人的使用人数不改变所需权重大小。本目录不自动转换或量化权重。

目标环境是已有 `ascend/vllm-omni:v0.28.0` 容器，沿用其 CANN、PyTorch、torch-npu、vLLM 和 vLLM-Ascend 环境，不自动安装或升级。还需确认容器具备仓库根目录 `requirements.txt` 的 Web 依赖。代码调用普通 vLLM 的 Thinker 文本输出路径，容器中以前运行过的 vLLM-Omni 服务可能使用多阶段链路。已有 Omni 服务可作为环境证据，但不能直接证明该 SFT checkpoint 与多轮音频已跑通。上游 [v0.28 InputPreprocessor](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/inputs/preprocess.py) 和 [Qwen3-Omni Thinker](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen3_omni_moe_thinker.py) 源码的接口经静态核对，仍需真机验证。官方 main 文档说明该模型自 v0.30.0 不再维护，勿盲目升级。

先在服务器检查：

```bash
npu-smi info
python3 -c "import torch,torch_npu,vllm,vllm_ascend; from vllm.platforms import current_platform; print('torch:',torch.__version__,'torch_npu:',torch_npu.__version__,'vllm:',vllm.__version__,'platform:',current_platform.device_type,'NPU可用:',torch.npu.is_available(),'卡数:',torch.npu.device_count())"
python3 -m flex_web_d910b.diagnose --checkpoint /绝对路径/merged-checkpoint
```

诊断读取环境版本、预处理器接口、checkpoint 配置、权重文件总大小和标签编码，不加载模型权重。设备 0、1 在提供的快照中已经占用较多 HBM，实际启动前请用 `npu-smi info` 选择一张有足够空闲显存的卡，勿影响正在运行的服务。

## 启动

在仓库根目录，使用可访问服务的实际 IP 生成或复用自签 HTTPS 证书（客户端还需信任项目根 CA）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 FLEX_HTTPS_HOST=服务器内网IP \
  bash flex_web_d910b/start_server.sh /绝对路径/merged-checkpoint
```

把 `ASCEND_RT_VISIBLE_DEVICES=0` 的数字换成选择的空闲卡编号。浏览器打开 `https://服务器内网IP:18889`。脚本默认单卡 TP=1、1 秒分块、20/10 秒历史滚动、最多 32 个音频轮、eager 执行；音频轮上限可能使短块的历史提前裁剪。默认最多 2 个会话，每批处理 1 个请求，两人是否实时取决于实际速度。其余 Web 参数（如 `--tts-url`、`--tts-assets-dir`）可继续追加。`FLEX_HTTPS_HOST` 为空时启动 HTTP；远程浏览器通常无法在 HTTP 下使用麦克风。模型权重、tokenizer、原训练模板保持不变。默认不启用前缀缓存和两种音频处理缓存。

若单卡显存不足，可以在启动或诊断音频命令末尾显式尝试 `--cpu-offload-gb 20`。这是权重的 CPU 卸载，不是 KV cache 卸载，20 GiB 仅为试验参数；必须确认服务器有足够内存、启动日志确实显示卸载生效，并检查实时速度。该容器、910B4 和此 checkpoint 的组合尚未经真机验证，参数不保证可用或能装下模型。

启动 Web 前可先用一段真实、至少 3 秒的单声道 16 kHz、16 位 PCM WAV 检查多轮协议：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m flex_web_d910b.smoke_audio \
  /绝对路径/merged-checkpoint /绝对路径/测试.wav --target zh
```

此命令每秒推理一次并打印标签、译文和生成耗时；任一轮协议校验失败会停止，不将不合法的结果提交到历史。

代码根据已安装 vLLM 的 `InputPreprocessor` 构造签名区分旧版 `model_config/tokenizer` 和新版 `vllm_config`；音频 token 数通过独立预处理器计算并在真实生成后校验，缺少实际 token 或校验不符均拒绝提交。此接口和多模态数据结构依赖 vLLM 内部实现，API 静态适配不代表运行已验证。需要在实际服务器依次验证：模型加载、真实音频及三标签、多轮历史裁剪、两方向以及 1～2 人并发的时延和译文质量。

## 实验：音频 token 计数复用

默认关闭。在原 Web 启动命令末尾添加 `--audio-token-cache-size 64` 可申请启用；`0` 是关闭。仅在 **vLLM 0.28.0 + Transformers 5.6.2 或 5.5.3** 的原生 Qwen3-Omni/Whisper 处理链和固定配置检查通过时才启用，其他版本、处理器或自定义参数回退逐块精确计数，既不升级依赖也不猜算 token 数。首次遇到一种实际采样长度仍精确计算，再复用该整数；真实音频始终照常送入模型。缓存容量有界，不保存波形或译文。

查看启动日志 `Audio token counter mode=length_lru` 是否真正启用；`mode=exact` 代表没有启用并说明原因。导出的 JSON 每轮 `timings.audio_count_mode`、`audio_count_cache_hit`、`audio_count_s`、`engine_generate_s` 可用于 A/B。`request_prepare_s` 已包含 `audio_count_s`，不要相加。即便启用后也继续校验 vLLM 返回的输入 token 数；如果缺失或总数不一致，会关闭计数缓存并拒绝本轮提交，不会自动重试。

可在 `smoke_audio.py` 中加 `--audio-token-cache-size 64` 检查逐轮模式和命中；为对照当前 Web 的组合配置，可同时加 `--mm-processor-cache-gb 1 --enable-graphs` 和实际使用的 `--enable-prefix-caching`。单独计数预处理器在新版 vLLM 中有独立 renderer，其多模态缓存也会按配置占用主机内存；它不与正式引擎的 sender cache 共享。需在真实 910B、实际 checkpoint 和 Transformers 版本上对比缓存开关，覆盖不同音频、尾块与裁窗，确认翻译、标签、计数和实际延迟。静态源码检查及模拟测试不代表真机无冲突或保证提速。
