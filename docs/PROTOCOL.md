# FLEX Live 协议 v1

单个 WebSocket 对应一次新会话。一个服务实例同时只接纳一个会话；不同会话不共享对话历史。协议验证失败必须新开会话，不做不透明的自动续传。

## 握手

`GET /api/config` 返回后端类型、已配置语言方向、分块/队列/会话上限和认证要求。不暴露服务器模型完整路径，也不返回 token。

连接同源 `/ws`，第一条消息：

```json
{"type":"start","source_lang":"en","target_lang":"zh","source_kind":"file","source_name":"example.mp4","media_offset_s":0,"chunk_seconds":1,"adaptive":true}
```

启用认证时，在这条 JSON 中增加 `token`。不把 token 放进 URL、服务日志或浏览器持久化存储。只有收到 `type=ready` 后才启动音频发送/播放。`ready.backend=mock` 表示协议演示，不是真实模型翻译。

## 音频帧

二进制小端格式：

| 字段 | 长度 | 含义 |
|---|---:|---|
| sequence | uint32 / 4 bytes | 从 0 递增的帧编号 |
| start_sample | uint32 / 4 bytes | 此帧起点，首帧为 0，后续严格连续 |
| PCM | 2 × N bytes | 16 kHz、单声道、有符号 int16、小端 |

每帧 1–8000 个样本，浏览器通常每 100 ms 发送一次。网络传输帧不是推理 chunk：服务器按 1 秒等设定聚合，积压时增加本轮聚合长度，默认最大 3 秒。合并长度不保证是基础块的整数倍：例如基础块 2 秒、最大 3 秒、积压至少 4 秒时，本轮可以取 3 秒。只发送已经采集/播放的音频，不发送完整文件。

暂停文件播放时不发送音频，也不把暂停时长变成静音。音频流索引表示实际输入音频的累计长度，不包含暂停的墙钟时间。客户端每 20 秒 ping 保活，服务器长时间收不到任何消息时释放会话。跳播、倍速变化必须结束当前会话，不能沿用旧历史；新的 `media_offset_s` 仅用于对照播放位置。

## 控制消息

- `{"type":"ping"}` → `pong`。
- `{"type":"finish"}`：发送前必须先 flush AudioWorklet 尾帧；服务器处理所有已接受音频，最后不足一块的尾音只处理一次，返回 `done`。
- `{"type":"cancel"}`：不再提交新结果；已经在执行的 GPU 请求仍需安全结束，之后才能让下一会话使用 GPU。未翻译音频量保留在终止统计中。

## 事件

- `ready`：会话接纳、配置确认。
- `status`：已接收、已处理、等待处理音频量及是否有在途推理。
- `turn`：`turn_id/tag/delta/response/source_start_s/source_end_s/service_s/queue_wait_s/prompt_tokens`。
- `error`：`code/message`，停止采集；不要把错误解释成 WAIT。
- `done`：终止原因、已提交完整译文、输入和未处理量。随后关闭连接。

`WAIT` 的 `delta` 必须为空，但音频轮和 WAIT 原始 token 进入模型历史。`FLEX` 与 `DIRECT` 的 `delta` 都非空，直接追加到译文，原有内容不替换；英文前导空格不裁剪。完整响应验证前不发布逐 token 文本。

`source_end_s` 是本次模型输入 chunk 的截止位置，不代表该译文语义的精确源语对齐。`service_s` 是本轮服务耗时（包括前处理/生成），`queue_wait_s` 从该块最后一个传输帧被接收到开始计时，可能包含等待凑块和用户暂停的时间，不能只解释为 GPU 排队；都不是 AL/LAAL。浏览器另外记录事件到达时的播放位置与本地时间，避免直接相减两台机器未同步的时钟。媒体事件和音频渲染线程之间有块级调度误差，显示的源音频累计时长和媒体 currentTime 可能相差数毫秒至数十毫秒，不宣称样本级音画同步。

正常 `done` 的 `final_wait=true` 表示最后一轮仍是 WAIT。页面提示人工检查尾句；不额外虚构输入或改变模型策略强迫输出。`processed_audio_s == received_audio_s` 只证明接收音频处理完毕，不是语义无漏译的证明。

## 安全和错误边界

同源 Origin 默认允许；额外 Origin 必须服务器显式配置。不接受来自其他站点的浏览器 WebSocket。非浏览器客户端可能没有 Origin，因此远程监听仍须 token 和网络访问控制。

超过音频队列、会话时长或模型上下文限制时显式终止，报告未处理量。不能静默删帧、清空 WAIT 历史、截断模型输出后当作成功。当前版本不具备无限时长上下文管理，也不宣称网络中断后可无损自动恢复。
