# FLEX-S2TT：批量流式推理，原地更新已有 JSONL

加载一次服务器本地 FLEX checkpoint，依次推理音频文件夹中的音频，再按 JSONL 每行的 `vid` 匹配记录，原地加入指定模型的结果字段。TSC 和 Byte 分别运行同一个命令，不需要固定文件夹名或修改代码。

## 运行方式

在已经能够运行 FLEX 模型的 Linux / CUDA / vLLM 环境中执行。批量脚本默认使用 `python3`。

```bash
cd /data06/yb/Flex_web_server/FLEX-S2TT

CUDA_VISIBLE_DEVICES=0 bash run_batch.sh \
  /data06/yb/Code/FLEX/models/checkpoint-4410-merged \
  /实际路径/TSC音频文件夹 \
  /实际路径/TSC原始jsonl文件夹 \
  Flex_1022 \
  --direction en-zh \
  --verbose
```

四个位置参数依次为：**模型目录、音频目录、已有 JSONL 目录、字段前缀**。将两个数据路径换成实际绝对路径，模型目录换成这次要评测的完整 merged checkpoint。前缀只是结果字段名称，不会决定加载哪个模型。

Byte 数据集另跑一次，只需换成 Byte 的音频和 JSONL 目录。同一个模型可以使用相同前缀。中译英使用 `--direction zh-en`；一次命令处理一个方向。

也可以直接调用 Python：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u infer_folder.py \
  --model /data06/yb/Code/FLEX/models/checkpoint-4410-merged \
  --audio-dir /实际路径/音频文件夹 \
  --jsonl-dir /实际路径/原始jsonl文件夹 \
  --field-prefix Flex_1022 \
  --direction en-zh \
  --chunk-seconds 1 \
  --pace fast \
  --protocol-form special \
  --verbose
```

## 匹配规则与结果字段

默认读取两个目录的第一层文件。音频支持 WAV、MP3、FLAC、M4A 等常见后缀；JSONL 使用 UTF-8（可带 BOM），一行一个 JSON 对象。

按**音频文件名去掉扩展名，与行内的 vid 对应**：

| 音频文件 | JSONL 行内 vid |
| --- | --- |
| `en2zh-01-tech_004.wav` | `en2zh-01-tech_004` |
| `Dataflow_001.wav` | `Dataflow_001` |
| `中东教授访谈_001.wav` | `中东教授访谈_001` |

同时兼容 vid 本身带音频扩展名的情况。扩展名大小写不限，其他部分精确匹配，不模糊匹配、不按文件排序对齐。JSONL 文件可以任意命名；不使用 `original_vids` 寻找音频。

给定前缀 `Flex_1022`，成功完成音频推理后写入：

```json
{
  "vid": "en2zh-01-tech_004",
  "transcript": "Original English transcript.",
  "translation": "原来已有的参考译文。",
  "Flex_1016_trans": "其他模型已有的译文。",
  "Flex_1022_trans": "本次模型真实产生的流式译文拼接结果。",
  "Flex_1022_asr": "Original English transcript."
}
```

- `Flex_1022_trans`：各轮已提交的 FLEX / DIRECT 新增译文按顺序直接拼接，不含协议标签；WAIT 不增加文字。不使用参考代替模型结果，不额外润色或删改标点。
- `Flex_1022_asr`：原样复制 `transcript`，包括空格与标点；这是约定填充的兼容字段，**不是模型的 ASR 结果**。源转录与参考译文都不输入翻译模型。
- 原来的 `translation`、`transcript`、其他模型字段和嵌套数据保留。
- 已有同名前缀时，成功推理覆盖这两个字段；换前缀即可增加另一组结果。
- 没有对应音频的行保持原样。比如无音频的 `en2zh-04-fin.jsonl`、`6G.jsonl` 不会被改动。
- 音频没有对应 JSONL 行时打印 `SKIP` 并跳过，不创建新行。
- 音频去掉后缀后重名、或同一音频匹配多个行，会在加载模型前报错。匹配行的 `transcript` 必须是字符串；未匹配行无需具备它。

## 只检查匹配

```bash
python3 infer_folder.py \
  --audio-dir /实际路径/音频文件夹 \
  --jsonl-dir /实际路径/原始jsonl文件夹 \
  --field-prefix Flex_1022 \
  --check-only
```

不需要模型路径或 GPU。打印音频数、JSONL 文件数、总记录数、匹配数、没有音频的记录数、没有记录的音频。不会加载模型、解码音频或修改文件，不能替代真实推理验证。正式运行也会自动检查匹配。

## 进度、原地保存与失败处理

每完成一条音频就立即写回。终端显示 `[当前/总数]`、文件名、JSONL 行号、`SAVED`、音频时长、推理轮数、译文字符数和耗时；`--verbose` 同时显示逐轮标签与耗时。结束时汇总成功、失败、未尝试和跳过数量。

原地模式不创建结果副本、`.bak` 或错误 JSONL。写入使用同目录临时文件再原子替换，成功后临时文件消失。未修改的行保留原始字节，已修改行重新序列化；文件 BOM、行尾风格和行顺序保留。强制终止进程可能留下 `.tmp`，原文件不会因此被截成半行。

音频解码失败、超出限制或模型输出协议错误时，保留该行旧值并打印错误，默认继续下一条。添加 `--fail-fast` 可在首个失败处停止。磁盘写入失败立即停止；Ctrl+C 保留已完成的更新，当前未完成音频不回写。

模型全程 WAIT 时，真实译文为空字符串，写入 `_trans` 并打印 `EMPTY`。不使用参考或转录填充 `_trans`。最后一轮 WAIT 会提示检查尾句，不强迫模型补译。

WAIT 是完整动作：特殊标签模型生成 `<|WAIT|>`（151676）后立即停止本轮，普通文本标签模型在完整标签处停止，并保留标签及对应音频到历史。这样可避免模型在 WAIT 后反复生成换行，直到 `max_tokens` 导致 `finish_reason='length'`。DIRECT/FLEX 仍须正常结束；其他真正达到长度上限的输出仍按失败处理。最后不足一块的音频照常处理，短于一秒时仅补齐模型输入，源音频时长按真实采样数记录。输入全部处理完不等于尾句一定已完整翻译。

重复运行会重新推理所有匹配音频，没有自动跳过已有字段。不要同时对同一 JSONL 目录运行多个写入任务：程序会检测索引后内容变化并拒绝覆盖，但这不是跨进程锁。

退出码：成功 0；音频失败、无法写入或没有匹配行 1；命令参数错误 2；Ctrl+C 130。部分音频没有对应行只警告跳过，不单独导致失败。

## 流式推理和参数

- 模型加载一次，各音频会话独立，保留同一音频此前的输入和模型回复。
- 默认 1 秒分块，按顺序释放音频，不提前输入后续音频。`--chunk-seconds 2` 可改为 2 秒。
- `--pace fast` 当前块完成后立即处理下一块；`--pace realtime` 模拟按真实时间到达。
- 最后不足一块仍处理，必要时补静音，不添加额外 EOF 轮。
- `--no-warmup` 关闭开始前的一次预热。
- `--recursive` 递归扫描音频目录；有同名文件时使用 `--vid-mode relative`，此时 vid 应形如 `speaker1/001` 或 `speaker1/001.wav`。JSONL 目录仍只扫描第一层。
- 每条音频默认最多 180 秒、256 轮；上下文 8192、每轮最多 160 token。可分别用 `--max-audio-seconds`、`--max-audio-turns`、`--max-model-len`、`--max-tokens` 调整。不静默截断或清空历史，单独增大时长不保证超长音频可完成。

## 环境

沿用跑通模型的 CUDA / vLLM 环境，无需新增第三方包；新增 JSONL 更新模块只用 Python 标准库。仅缺 NumPy 时安装：

```bash
python3 -m pip install -r requirements-infer.txt
```

16 kHz 单声道 PCM16 WAV 可直接读取；其他格式、采样率或声道需要 FFmpeg，可用 `--ffmpeg /path/to/ffmpeg` 指定。采用管道解码，不生成 WAV 副本。

模型须包含完整 merged 权重以及训练一致的 tokenizer / processor。启动脚本默认特殊协议 token（WAIT / DIRECT / FLEX ID 为 151676 / 151677 / 151678）；普通文本标签模型使用 `--protocol-form text`。

默认单卡、bfloat16、显存利用率 0.90。`--image-slots 1` 保留兼容所需的初始化名额，实际只输入音频；适用环境可改为 0。本次文件回写改动沿用原模型适配、提示词和生成逻辑。

`PYTHON=/path/to/python3 bash run_batch.sh ...` 可以指定解释器。相对路径以当前工作目录为准。`--backend mock` 仅供临时测试数据使用，会写入演示文本；正式数据保持默认 vLLM 后端。

## 保留的旧入口

三个位置参数仍可输出一个新的结果 JSONL：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_batch.sh MODEL_DIR AUDIO_DIR /path/results.jsonl
```

旧模式生成 `vid`（含音频后缀）和 `translation` 两字段，以及 `results.errors.jsonl`；输出文件必须不存在。原地更新请使用前面的四参数命令。

单音频入口：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 bash run_infer.sh MODEL_DIR AUDIO_FILE --direction en-zh
```

## 文件与测试

```text
FLEX-S2TT/
  infer_folder.py            批量推理和原地更新入口
  jsonl_update.py            vid 索引、字段更新、原子写回
  run_batch.sh               批量启动，默认 python3
  infer_audio.py             音频读取和流式推理
  flex_model.py              模型与协议适配
  run_infer.sh               单音频启动
  requirements-infer.txt
  README.md
  tests/test_batch.py        原有批量与单文件流程检查
  tests/test_update_jsonl.py 原地回写、异常、中断和字段保留检查
```

```bash
python3 -m unittest discover -s tests -v
```

CPU 测试使用 mock 后端和临时音频/JSONL，不需要模型或 GPU。真实模型质量、显存与速度仍需在服务器验证。
