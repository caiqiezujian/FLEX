# FLEX Live 协议：文件上传与麦克风流

单个 WebSocket 对应一次新会话。一个服务实例默认同时接纳 4 个会话，共享一份模型；不同会话不共享音频或对话历史。协议验证失败必须新开会话，不做不透明的自动续传。

## 握手

`GET /api/config` 返回后端类型、已配置语言方向、分块/队列/会话上限和认证要求。`file_mode=uploaded` 表示支持新文件模式。不暴露服务器模型完整路径，也不返回 token。

`config/health/ready/status/done` 带 `active_sessions/max_sessions/available_sessions/busy` 容量统计；`busy=true` 表示所有名额都已占用，不是只要有一人就不可用。计数包含暂停、排空与取消后未完成的在途推理，不暴露其他会话 ID 或文件名。当前会话的终止事件可能仍包含自己的名额，资源清理后下一次健康查询更新。`max_batch_size/batch_wait_ms` 见 config，人数接纳与模型批大小是两个概念。

每会话一个待执行/在途任务，按照就绪请求 FIFO 排队；调度器最多收集 10 ms 的就绪任务，每批每会话最多一轮，最多 4 路。并不是等待凑齐所有用户，也不接收未来音频。批次只由一个模型线程执行，一次 `generate` 的多个请求具有各自的 prompt/audio。排队期间 cancel 可取消该任务；GPU 批次开始后等待其结束，不提交已取消会话的响应，不影响其他合法结果。满员时第五路收到 busy，不自动排队、不开始消耗其音频。

### 运行健康状态

容量统计同时带 `engine_health={state,accepting,running_seconds,queued_requests,message}`。`state` 为 `ready/slow/stalled/failed`，表示共享模型调度器的当前状态，不是某个会话的 WAIT/FLEX/DIRECT 策略。`config.status=ready` 表示启动已完成，运行中是否可接纳任务还须检查 `engine_health.accepting`。

计时从工作线程真正开始调用模型到其真实返回，不包含排队时长。默认运行 5 秒为 `slow`（仍接纳），60 秒为 `stalled`（疑似停滞，暂不接纳新任务，尚未执行的请求返回 `engine_stalled`）。不取消实际 GPU 线程，不提前释放在途名额或历史；该批次成功返回后恢复 `ready`。共享模型调用异常或输出数量异常置 `failed`，后续任务返回 `engine_failed`，需重启服务；单条输入准备或协议错误不污染共享引擎状态。阈值通过 config 的 `slow_inference_seconds/stalled_inference_seconds` 暴露。

会话状态另带 `processing_health={state,message,backlog_seconds,sustained_seconds}`：`normal/slow`。未处理音频包含排队与在途块，持续 5 秒达到 `max(3 秒, 2 × 当前基础块长)` 才触发积压告警；回落至半阈值以下解除，避免状态抖动。这是告警，不改变切块、输入或模型策略。WAIT 的数量、是否有新译文均不参与故障判定。

`GET /api/health` 在可接纳引擎任务时返回 200，在 `stalled/failed` 或调度器不可用时返回 503；单纯会话名额已满不返回 503。`GET /api/config` 仍返回配置和最新健康状态。已确定的输入错误（例如麦克风积压超限）可先发送 `error`，以便页面立即停止采集；`error` 不代表在途 GPU 工作已完成，最终 `done` 与资源释放仍等待真实工作结束，不重复发送同一错误。

## 文件上传与播放进度

1. `POST /api/media?name=example.mp4`，请求体是完整文件字节，Content-Type 为 `application/octet-stream`。开启认证时通过 `Authorization: Bearer ...` 传口令；文件名必须 URL 编码，仅作显示。
2. 服务限制上传大小/数量和解码长度，在临时目录解码为 16 kHz 单声道 PCM，返回 `media_id/duration_s/samples/sample_rate/source_name`。准备文件不加载另一份模型，也不进行翻译。
3. WebSocket start 中携带 `media_id`，服务器一次性认领上传。返回 ready 后才开始播放器；ready 包含 `file_mode=uploaded`、media_id 和解码时长。
4. 播放中约每 200 ms 发送 `{"type":"progress","position_s":1.25}`，position_s 是完整文件的绝对播放位置。服务器只释放截止该位置的音频，逐块读取；不会将完整文件一次送入模型。
5. 暂停/缓冲时提交最后位置，之后不推进。恢复或页面短暂卡顿后，用新的播放位置连续释放中间音频，不跳块；待处理音频留在磁盘，不触发实时 PCM 内存队列上限。
6. 文件模式固定基础 chunk，不根据网页回调是否晚到改变块边界；少于一块的尾部在 finish 时处理一次。
7. 手动结束发送 `{"type":"finish","position_s":2.35}`，只处理到实际暂停点。自然 EOF 另带 `ended:true`，用已校验的解码时长关闭最后边界。不得将接近 EOF 的手动结束自动当作完整文件。
8. 未用于会话的上传可用 `DELETE /api/media/{media_id}` 删除，认证方式与上传相同；已认领媒体由会话退出流程清理，未认领媒体过期清理。

上传文件模式不发送 WebSocket 二进制音频帧。浏览器在 HTTP/HTTPS 下均不创建音频采集器。完整文件已在服务器磁盘上，不代表推理请求可访问尚未释放的后文；模型请求构造只使用当前块和已提交历史。

连接同源 `/ws`，第一条消息：

```json
{"type":"start","source_lang":"en","target_lang":"zh","source_kind":"file","source_name":"example.mp4","media_id":"上传接口返回的ID","media_offset_s":0,"chunk_seconds":1,"adaptive":false}
```

启用认证时，在这条 JSON 中增加 `token`。不把 token 放进 URL、服务日志或浏览器持久化存储。只有收到 `type=ready` 后才启动音频发送/播放。`ready.backend=mock` 表示协议演示，不是真实模型翻译。

## 麦克风音频帧（以及旧客户端兼容路径）

二进制小端格式：

| 字段 | 长度 | 含义 |
|---|---:|---|
| sequence | uint32 / 4 bytes | 从 0 递增的帧编号 |
| start_sample | uint32 / 4 bytes | 此帧起点，首帧为 0，后续严格连续 |
| PCM | 2 × N bytes | 16 kHz、单声道、有符号 int16、小端 |

每帧 1–8000 个样本，浏览器麦克风通常每 100 ms 发送一次。网络传输帧不是推理 chunk：服务器按 1 秒等设定聚合，积压时增加本轮聚合长度，默认最大 3 秒。合并长度不保证是基础块的整数倍：例如基础块 2 秒、最大 3 秒、积压至少 4 秒时，本轮可以取 3 秒。无 media_id 的历史文件客户端仍保留旧二进制协议，新页面不再走该路径。

暂停文件播放时不释放新的音频，也不把暂停时长变成静音。音频流索引表示实际输入音频的累计长度，不包含暂停的墙钟时间。客户端每 20 秒 ping 保活，服务器长时间收不到任何消息时释放会话。跳播、倍速变化必须结束当前会话，不能沿用旧历史；上传文件的 `media_offset_s` 同时决定文件读取起点和播放位置对照，旧二进制客户端则由客户端自行保证输入起点。

## 控制消息

- `{"type":"ping"}` → `pong`。
- `{"type":"finish"}`：麦克风/旧二进制客户端发送前先等待 AudioWorklet 排空有效尾帧并确认；服务器处理所有已接受音频，最后不足一块的尾音只处理一次，返回 `done`。上传文件使用前述带 position_s 的 finish，不等待浏览器采集器。
- `{"type":"cancel"}`：不再提交新结果；已经在执行的 GPU 请求仍需安全结束，之后才能让下一会话使用 GPU。未翻译音频量保留在终止统计中。

## 事件

- `ready`：会话接纳、配置确认。
- `status`：已接收、已处理、等待处理音频量及是否有在途推理。
- `turn`：`turn_id/tag/delta/response/source_start_s/source_end_s/service_s/queue_wait_s/prompt_tokens`。
- `error`：`code/message`，停止当前会话输入；不要把错误解释成 WAIT。
- `context_window`：模型上下文滚动，包含全会话 `turn_id` 和 `context` 统计，不表示重新开始会话。
- `done`：终止原因、`translation_chars/turn_count/results_available`、输入和未处理量。短会话兼容返回 `translation`；超过 16384 字符时省略全文，`translation_truncated=true`，需从导出接口获取。随后关闭连接。

`status/turn/done` 的 `context` 包含该会话的窗口策略、累计滚动次数、移除的轮数与音频秒数、当前保留轮数/音频秒数/token 数、全局提交量和最近一次滚动原因。`input_finished` 区分尚在收音与停止输入后排空；`inference_state=idle|queued|running` 区分模型排队和执行，`in_flight` 兼容表示当前轮尚未结束（可包含 queued）。WAIT、音频积压与共享模型调度等待是不同维度。

`WAIT` 的 `delta` 必须为空，但音频轮和 WAIT 原始 token 进入模型历史。`FLEX` 与 `DIRECT` 的 `delta` 都非空，直接追加到译文，原有内容不替换；英文前导空格不裁剪。完整响应验证前不发布逐 token 文本。

`source_end_s` 是本次模型输入 chunk 相对会话起点的截止位置；加 media_offset_s 才是文件绝对位置，不代表该译文语义的精确源语对齐。`service_s` 是本轮服务耗时（包括前处理/生成）。麦克风的 `queue_wait_s` 从最后一个传输帧到达计时，上传文件从使该块可读的播放进度到达计时；它可能包含等待凑块和暂停时间，不能只解释为 GPU 排队，也不是 AL/LAAL。浏览器记录结果到达时的媒体位置与本地时间。上传/解码准备发生在播放之前，不计入流式阶段时钟，部署端到端耗时应另计这段准备时间。容器时间戳、播放器和解码器仍可能有差异，不宣称任意媒体样本级音画同步。

正常 `done` 的 `final_wait=true` 表示最后一轮仍是 WAIT。页面提示人工检查尾句；不额外虚构输入或改变模型策略强迫输出。`processed_audio_s == received_audio_s` 只证明接收音频处理完毕，不是语义无漏译的证明。

多用户版本将 `turn.service_s` 计为实际模型批次执行的墙钟时间（含预处理/生成），不再混入共享调度队列的等待；同批请求共享该批次耗时，不是每个用户独占 GPU 的耗时。新增 `scheduler_wait_s` 表示从提交任务到模型批次开始的等待，`batch_size` 表示本批调度任务数。`queue_wait_s` 仍是本用户音频就绪到任务提交前的等待。批次返回后才发布合法增量，因此本批快请求可能等慢请求，不能把这些时间直接等同于字幕渲染延迟或 AL/LAAL。

## 长会话与完整结果

默认总输入预算 28800 秒，模型窗口单独受 120 秒音频、8192 token、256 轮约束。`rolling` 在下一轮需要更多空间时移除最早完整音频+响应轮次，保留 system 和近期上下文，目标降至 60 秒及相应 token/轮数低水位。原始 assistant token 不重新编码；裁剪计划只在新响应成功 commit 时应用。`stop` 是可选的旧硬停行为，不滚动。上下文淘汰是显式、可记录的模型可见范围变化，不删除源文件或已输出译文，也不证明淘汰内容已经语义翻译完成。

`GET /api/sessions/{session_id}/export?format=txt|json` 获取完整已持久化前缀，开启认证时仍用同一 `Authorization: Bearer ...`。运行中下载捕获请求时已写入的字节边界；并发的新轮次不改变该快照。服务端分块读取磁盘，不把全部文本/事件载入内存。TXT 是原样拼接译文；JSON schema_version=2 包含语言/输入元数据、完整 `translation/turns/events`、`error`、`done` 及计数。`turns` 是服务端事件，不再带完整的浏览器接收时刻/播放位置记录；这些仅保留于页面的有限近期记录，不能把服务端时间解释为实际字幕渲染时间。

`complete=true` 仅表示 `done.reason=finished` 且结果计数相符；取消、异常、断连的有效前缀不是完整成功。它只针对本会话实际接收的输入（用户可能中途正常结束文件），不证明整份原文件已播放或语义完整。`session_finished` 单独表示结果写入资源是否已关闭。超过磁盘预算时仍可导出此前有效前缀，终止诊断也会保留，但不能把前缀当作成功全文。

每个会话临时结果预算默认 256 MiB，最多缓存 8 个会话，结束后最多保留 24 小时；容量满会提前淘汰最旧已结束结果，重启后失效。`--no-log` 只禁用额外持久 JSONL 日志，临时完整导出仍写磁盘。前端保留当前会话全部已收到的译文，只有策略详情限定为近期 200 行 / 120000 个 UTF-16 字符。成功导出读取服务器完整记录；服务器记录过期时，必须明确提示并征得确认，才能保存标记为 PARTIAL 的备份（全部已收到译文与近期策略详情，不保证包含尚未收到的结果或全量事件）。

## 安全和错误边界

同源 Origin 默认允许；额外 Origin 必须服务器显式配置。不接受来自其他站点的浏览器 WebSocket。可信内网允许不设 token；若配置非空 token，上传、删除和推理会话仍需认证。Origin 不是认证，非浏览器客户端可能没有 Origin；无令牌部署依赖网络隔离和防火墙，任何能连接服务端口的人均可使用。

麦克风超过音频内存队列、任何模式超过总时长/结果预算、或单块无法装入模型预算时显式终止，报告未处理量。文件不会仅因短暂页面卡顿或已播放音频积压而触发采集漂移错误。不能静默删帧、按标签清空历史、截断模型输出后当作成功。滚动窗口使长期推理的模型历史有界，不保证无限时间或无损长距离语义，也不宣称网络中断后可无损自动恢复。PCM start_sample 为 uint32，可配置总时长不得超过约 74.56 小时。
