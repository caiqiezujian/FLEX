# FLEX Live 协议：文件上传与麦克风流

单个 WebSocket 对应一次新会话。一个服务实例同时只接纳一个会话；不同会话不共享对话历史。协议验证失败必须新开会话，不做不透明的自动续传。

## 握手

`GET /api/config` 返回后端类型、已配置语言方向、分块/队列/会话上限和认证要求。`file_mode=uploaded` 表示支持新文件模式。不暴露服务器模型完整路径，也不返回 token。

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
- `done`：终止原因、已提交完整译文、输入和未处理量。随后关闭连接。

`WAIT` 的 `delta` 必须为空，但音频轮和 WAIT 原始 token 进入模型历史。`FLEX` 与 `DIRECT` 的 `delta` 都非空，直接追加到译文，原有内容不替换；英文前导空格不裁剪。完整响应验证前不发布逐 token 文本。

`source_end_s` 是本次模型输入 chunk 相对会话起点的截止位置；加 media_offset_s 才是文件绝对位置，不代表该译文语义的精确源语对齐。`service_s` 是本轮服务耗时（包括前处理/生成）。麦克风的 `queue_wait_s` 从最后一个传输帧到达计时，上传文件从使该块可读的播放进度到达计时；它可能包含等待凑块和暂停时间，不能只解释为 GPU 排队，也不是 AL/LAAL。浏览器记录结果到达时的媒体位置与本地时间。上传/解码准备发生在播放之前，不计入流式阶段时钟，部署端到端耗时应另计这段准备时间。容器时间戳、播放器和解码器仍可能有差异，不宣称任意媒体样本级音画同步。

正常 `done` 的 `final_wait=true` 表示最后一轮仍是 WAIT。页面提示人工检查尾句；不额外虚构输入或改变模型策略强迫输出。`processed_audio_s == received_audio_s` 只证明接收音频处理完毕，不是语义无漏译的证明。

## 安全和错误边界

同源 Origin 默认允许；额外 Origin 必须服务器显式配置。不接受来自其他站点的浏览器 WebSocket。非浏览器客户端可能没有 Origin，因此远程监听仍须 token 和网络访问控制。

麦克风超过音频内存队列、或任何模式超过会话/模型上下文限制时显式终止，报告未处理量。文件不会仅因短暂页面卡顿或已播放音频积压而触发采集漂移错误。不能静默删帧、清空 WAIT 历史、截断模型输出后当作成功。当前版本不具备无限时长上下文管理，也不宣称网络中断后可无损自动恢复。
