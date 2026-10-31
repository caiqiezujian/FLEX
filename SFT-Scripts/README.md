# SFT：原音频＋区间，不生成切分 WAV

适配用户已确认的 Docker `omni_ms`、Python 3.11、ms-swift 3.12.1。继续使用 Megatron SFT，不需要部署新模型服务，也不需要升级 SWIFT。

## 改了什么

- `prepare_duplex_data.py` 默认生成虚拟音频区间，保留原始录音，不创建 `chunk_000.wav` 等小文件。
- `duplex_audio.py` 在读取训练样本时，按区间读取原始采样，再下混、重采样到 16 kHz、末尾补零。内存 WAV bytes 交给原生模板；不创建临时音频文件。
- `duplex_sft_plugin.py` 只扩展音频加载入口，继承当前安装的 Qwen3 Omni 模板和元信息。不会把所有轮次拼成一整条波形，也不会先提取完整录音的特征再切分。
- `train_duplex_sft_virtual.sh` 使用现有 Python 启动四进程 Megatron；沿用用户提供的 LoRA r8/alpha32、micro4/global16、EP2、lr=1e-4、3 epochs、packing、max_length4096、每3000步保存的配置。显式使用 `--save_safetensors true --merge_lora false` 保存未合并的适配器，减少重复保存完整模型的占用。新增插件和模板参数；不默认启用 CUDA 同步调试。
- 原 `D:\ICLR\Scripts\train_duplex_sft.sh` 保留，可用于原来的实体 WAV 数据。本文件夹集中保存新的虚拟音频 SFT 流程。新虚拟 JSONL 必须配合新插件使用，不能直接交给旧默认模板。

原音频必须在训练容器内可访问。`audios` 保存 `flexaudio:v1:` 开头的区间引用；可读的路径与秒数同时保存在 `audio_segments`，不需要手工编辑引用。

## 输入仍然需要时间戳

一条输入仍是原有结构。脚本不会执行 ASR、翻译、强制对齐，也不会生成缺失的时间戳。

```json
{"audio":"/data/audio/session.wav","srcs":["The device is portable,"," with a sealed case."],"chooses":["这台设备可以随身携带。","它采用密封外壳。"],"tags":["<|DIRECT|>","<|FLEX|>"],"timestamps":[{"index":1,"start_ms":0,"end_ms":700,"text":"The device is portable,"},{"index":2,"start_ms":700,"end_ms":1300,"text":" with a sealed case."}]}
```

`start_ms/end_ms` 是相对于这一条原始音频起点的毫秒数。`index` 如存在，必须完整、唯一且从1开始；`text` 如存在，必须与对应 `srcs` 字符串一致。重复索引、非有限数值、负数、倒序/重叠/零长区间、超过实际音频时长等问题会整条过滤。不会删除原音频，也不会把越界译文强塞进最后一轮。

## 运行顺序

将整个 `D:\ICLR\SFT-Scripts` 文件夹一起上传，然后进入原 SFT 容器，并 `cd` 到上传后的 `SFT-Scripts` 目录运行下面命令。不要只上传训练 shell，音频加载模块和插件必须在同目录。本文件夹运行不依赖原 `Scripts` 目录。

以下数据路径沿用先前提供的命令；若本轮数据另有路径，只替换对应参数。

```bash
python3 -u prepare_duplex_data.py \
  /data/xyh/omni_train_datas/20260610/20260807_cpo_asr/en.asr.cpo.raw.0807.h1k.regular.jsonl \
  --output /data/xyh/omni_train_datas/flex_data/en.virtual.train.jsonl \
  --audio-mode virtual \
  --min-chunk 1 \
  --max-chunk 3 \
  --target-lang 中文 \
  --keep-tail-silence
```

这仍然是随机1、2、3秒音频轮次；若需要固定1秒，改为 `--min-chunk 1 --max-chunk 1`。不传 `--audio-out`，不加 `--dry-run`。输出为训练 JSONL 和 `en.virtual.train.jsonl.rejected.jsonl` 拒绝原因报告，报告记录输入位置、ID、原音频地址和错误原因，不复制原音频。合格行逐条写入同目录临时文件，全部处理结束后发布最终 JSONL，避免失败时覆盖已有输出。默认只要有合格样本就正常退出；全部被过滤时返回非零退出码。需要严格验收时可加 `--fail-on-reject`，让任何拒绝都返回非零退出码。

检查前两条样本的实际区间读取，并确认插件在当前环境能导入：

```bash
python3 check_duplex_virtual.py \
  /data/xyh/omni_train_datas/flex_data/en.virtual.train.jsonl \
  --rows 2 \
  --framework
```

随后训练。第一个参数是原脚本 `--load` 所用的 **MCore模型目录**，不是任意HF merged checkpoint。下例 `qwen3_omni_mcore_3.12.1` 与原脚本相同，是相对于当前工作目录的地址；也可以直接换成服务器上的绝对地址。

```bash
bash train_duplex_sft_virtual.sh \
  qwen3_omni_mcore_3.12.1 \
  /data/xyh/omni_train_datas/flex_data/en.virtual.train.jsonl \
  /data/yb/FLEX/outputs/sft_virtual_001 \
  --gpu-ids 0,1,2,6
```

训练命令自动先执行上述样本检查。GPU ID 可以换成你实际空闲的四张卡。参数末尾可以追加原生 Megatron 选项；不要把 GSPO 的参数名直接混进 SFT 3.12.1 命令。

确认流程时先用少量数据完成一次真实训练和保存，再运行全量。检查脚本不加载模型权重，也不能证明 GPU 训练、实际模板编码和 checkpoint 导出已验收。

## 保存适配器，按需合并模型

当前训练脚本显式设置 `--save_safetensors true --merge_lora false`。在ms-swift 3.12.1中，这会在保存点导出PEFT格式的适配器权重，而不是每次生成 `checkpoint-N-merged` 完整模型。保持 `--modules_to_save word_embeddings output_layer`：两层也在训练，必须一起保存，因此适配器可能仍达到GB级，并不只含LoRA小矩阵。保存间隔仍为3000步，优化参数、数据和模型加载方式不变。

保留匹配的原始基座、完整适配器目录、tokenizer及新增特殊token信息；选定一个checkpoint后再加载或合并。每个适配器checkpoint已经是该时刻相对于基座的完整状态，不能把checkpoint-3000与checkpoint-6000依次累加。同一次训练的多个保存点共用原始基座即可，不需要每次复制一份基座。新增token涉及词表扩展，正式推理前须核对三个协议token的ID以及嵌入/输出层尺寸。

修改shell只影响之后启动的训练，不会改变正在运行的进程，也不会自动压缩以前保存的完整模型。脚本末尾仍允许原生参数覆盖；要保持只保存适配器，不要另外传 `--merge_lora true`。既有 `--no_save_optim true --no_save_rng true` 保持不变，因此保存的是权重，不包含精确恢复训练所需的全部状态。该保存分支已核对官方3.12.1源码，仍需在服务器实际保存一个checkpoint后验收，确认可加载再清理旧模型。

### 将选中的适配器导出为完整模型

新增 `merge_duplex_lora.py` 和启动入口 `merge_duplex_lora.sh`。在原SFT容器及Python环境运行，使用原始未量化HF基座和一个已保存完成的PEFT适配器checkpoint，不需要训练数据或MCore目录，也不需要重新训练。下面前两个路径替换成当前服务器实际目录，第三个是你希望生成的新目录：

```bash
cd /data06/yb/Flex_web_server/SFT-Scripts

bash merge_duplex_lora.sh \
  /你的原始HF模型目录/Qwen3-Omni-30B-A3B-Instruct \
  /你的SFT输出目录/checkpoint-3000 \
  /data06/yb/Flex_web_server/outputs/sft_checkpoint_3000_merged \
  --gpu-ids 0
```

`--gpu-ids`指定用于合并的空闲显卡，允许如 `0,1`；这是单进程HF导出，`device_map=auto`可以使用可见GPU。也可以传 `--device-map cpu`，使用CPU内存加载和合并。合并需要加载完整基座，不应在已占满显存的训练卡上运行。依赖沿用原SFT环境中的ms-swift、torch、transformers、safetensors和Qwen音频组件，无需安装新框架。

第二个目录应有 `adapter_config.json`、`adapter_model.safetensors`（或其分片索引与全部分片）和 `args.json`。不是 `checkpoint-N-merged`，也不是只包含MCore `iter_*` 文件的目录。必须使用训练时对应的原始基座；结构检查不能证明两个同结构基座的权重完全相同。

脚本调用ms-swift 3.12.1的 `python3 -m swift.cli.export --merge_lora true`，显式指定 `--load_args false`、原生 `qwen3_omni` 模板和当前 `duplex_protocol_tokens.txt`，避免继承旧服务器的插件路径。只读取权重头和少量协议行进行验证：检查适配器保存了词嵌入/输出完整模块、训练token顺序与当前文件一致；导出后验证tokenizer和processor、完整Thinker张量名称/尺寸，以及WAIT/DIRECT/FLEX三行在嵌入层和输出层中与适配器一致（按导出dtype比较）。保持原有padded vocab，不主动缩小到tokenizer长度。发现不匹配时停止，不会用随机初始化或丢弃词表行的方法绕过。

输出目录必须不存在。实际导出先写同级 `.incomplete-随机串` 目录，验证通过后才改名为指定输出目录；失败时保留中间目录供排查，不能直接用于部署。输出是用于当前音频到文本同传的完整HF Thinker权重及processor/tokenizer，`ENABLE_AUDIO_OUTPUT=0`，不承诺语音播报分支。`flex_merge_report.json`记录来源、token ID和检查结果。模型文件检查不等于真实推理验收，完成后仍需用已知音频验证一次输出。

只想先检查输入而不加载完整模型，可在上述命令末尾追加 `--check-only`。合并成功后，原先离线推理/Web服务的模型路径可以指向这个新导出目录，推理时仍需提供与SFT一致的system prompt。

## 全量检查已有训练JSONL

`audit_duplex_dataset.py` 用于统计已经生成的虚拟音频训练JSONL有多少行通过、有多少行失败。它不会重新生成数据、修改区间、过滤原JSONL或删除原音频；无需模型、SWIFT或GPU。请在音频路径实际可访问的训练容器中运行，检查的是准备后的训练JSONL，而不是含srcs/chooses/timestamps的原始JSONL。

```bash
python3 -u audit_duplex_dataset.py \
  /你的实际训练文件绝对路径/train.jsonl \
  --output-dir /data06/yb/Flex_web_server/datasets/sft_audio_audit_001 \
  --workers 16
```

默认每5秒或每5000条输出进度，并发队列有上限，不会一次性加载全部JSONL。默认读取soundfile音频头，使用当前训练reader相同的严格边界条件；合法尾块补零可以通过。若需实际读取、重采样每个合法区间，在命令末尾增加 `--decode`，耗时会更长。默认模式不验证PCM损坏或非有限采样值；两种模式都不检查翻译准确性、语义时间戳对齐、模型token长度或GPU训练。

报告目录必须是尚不存在的新目录，避免覆盖上次结果：

- `summary.json`：全量是否完成、总行数、通过/失败行数、各类错误涉及行数和耗时。
- `invalid.jsonl`：失败行的原始行号、ID、出错轮次、音频路径、请求边界、实际时长和越界差值。并发完成顺序写入；默认每行最多保留10条详细错误，但所有错误均计入统计，可用 `--max-details` 调整。

一条训练数据中任意一块有问题，该行只计一次失败；不同原因的行数可以重叠。`bounds_seconds_only` 表示秒数检查失败但采样帧范围合法，是时长舍入差异的疑似证据，仍按当前reader计为失败；`bounds_frames` 表示帧越界或区间不足一个采样帧。报告会保留差值，不自动裁剪或放行。

退出码0表示扫描完成且所有行通过；1表示扫描完成但发现失败行；2表示脚本运行失败；130表示用户中断。先看 `status` 和 `full_dataset_checked`，避免把中断后的部分统计当成全量结果。默认字符串格式和typed格式均可按行检查，但同一JSONL混用两种格式可能触发Arrow加载问题，脚本会额外提示。

## FLEX和同轮合并

FLEX已改成策略定义；不按句子完整性或句末标点划分。保留全部合法译文的空格和标点。本文件夹只包含SFT数据准备、训练和检查代码。推理时应从新训练JSONL的system消息取得准确提示，传给推理程序；推理旧checkpoint则继续使用其原训练提示。

训练system prompt使用精简版，唯一模板是 `prepare_duplex_data.py` 中的 `PROTOCOL_SYSTEM_PROMPT_TEMPLATE`，目标语言由 `--target-lang` 填入。它只保留任务、三种动作、历史增量约束和标点要求；详细翻译规则仍由蒸馏和监督轨迹承担。准备脚本将提示写入默认格式的 `messages[0].content`（`--typed` 格式写入顶层 `system`），训练入口直接使用数据中的提示，不另设 `--system` 覆盖。

修改模板仅影响之后生成的JSONL，已有训练JSONL中的system消息不会自动更新。使用新提示时，需要重新生成JSONL或更新其system字段；无需因此重新切分音频。提示约束本身不会自动纠正已有的错误译文或标签，仍需保证监督答案与提示一致。

同一音频轮次内，可能有多个短 source 单元同时到期。译文必须都保留，仍按顺序拼接成这一轮的回答。暂不改变既有的“最后一个非空块标签”策略，不自动改成“含FLEX就全部FLEX”。新增 `turn_provenance` 记录原始块下标、原始标签、是否混合与合并策略；日志有混合计数。合并后的语义策略是否应该重新标注，是另一个需要根据真实输出讨论的问题。

## 开销和验证范围

这种方式减少准备阶段的ffmpeg启动、音频文件复制及小文件数量。训练时仍然需要分块读取和音频编码，不会自动减少音频token，也不能保证GPU每步更快。非16 kHz或多声道输入会产生CPU处理开销。虚拟读取用FLOAT内存WAV，旧文件模式用PCM16；不同重采样器及量化方式不保证逐字节等价。

本地测试覆盖原始采样区间、末尾补零、16/44.1 kHz下改变未来内容不影响早期片段、非法数据过滤、标点保留、混合标签来源和新旧SWIFT导入分支。模板分支使用stub，尚未在你的A800环境运行真实训练。

核对过的官方接口：

- [SWIFT v3.12.1音频加载](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/vision_utils.py)
- [SWIFT v3.12.1模板预处理入口](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/base.py)
- [SWIFT v3.12.1 Qwen模板](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/template/qwen.py)

本地测试可运行：

```bash
python3 -m pytest test_prepare_duplex_data.py test_duplex_virtual_audio.py -q
```
