import { StreamingResampler, floatToPCM16 } from './pcm-core.js';

/** File playback is allowed on remote HTTP; microphone access is not. */
export function selectCaptureMode({sourceKind, isSecureContext, hasWorklet, hasScriptProcessor}) {
  if (sourceKind === 'microphone' && !isSecureContext) {
    throw new Error('普通远程 HTTP 不能请求麦克风权限。请选择音视频文件进行同传；麦克风模式需要 HTTPS 或 localhost。');
  }
  if (isSecureContext && hasWorklet) return 'worklet';
  if (sourceKind === 'file' && hasScriptProcessor) return 'http-file';
  throw new Error(sourceKind === 'file'
    ? '当前浏览器没有可用的文件音频采集接口，请使用 Chrome / Edge，或通过 HTTPS 访问。'
    : '当前浏览器不支持 AudioWorklet，请使用近期版本 Chrome / Edge。');
}

/**
 * Sample-domain gate for the legacy file-only capture path. Audio callbacks may
 * arrive AFTER a pause/ended event. Keep bounded input history and drain the
 * recorded playback intervals, not a boolean sampled at callback delivery time.
 * No file decoding, lookahead, silence padding, or resampler reset on pause.
 */
export class FileCaptureTimeline {
  constructor({sampleRate, mediaOffset = 0, frameSamples = 1600, onMessage, onError}) {
    this.rate = sampleRate;
    this.mediaPosition = mediaOffset;
    this.resampler = new StreamingResampler(sampleRate, 16000);
    this.frame = new Int16Array(frameSamples);
    this.used = 0;
    this.onMessage = onMessage;
    this.onError = onError;
    this.blocks = [];
    this.windows = [];
    this.pending = [];
    this.open = null;
    this.latestEnd = null;
    this.finalized = false;
    this.stopping = false;
    this.failed = false;
  }
  fail(message) {
    if (this.failed || this.finalized) return;
    this.failed = true;
    this.used = 0;
    this.blocks = [];
    this.onError(new Error(message));
  }
  append(values) {
    for (const value of values) {
      this.frame[this.used++] = floatToPCM16(value);
      if (this.used === this.frame.length) this.flush();
    }
  }
  flush() {
    if (!this.used || this.failed) return;
    const buffer = this.frame.slice(0, this.used).buffer;
    this.used = 0;
    this.onMessage({type:'pcm', buffer});
  }
  ack(requestId) { this.onMessage({type:'ack', requestId}); }
  command(op, requestId, {contextTime, mediaTime}) {
    if (op === 'cancel') {
      this.finalized = true;
      this.open = null;
      this.blocks = [];
      this.used = 0;
      for (const request of this.pending.splice(0)) this.ack(request.requestId);
      this.ack(requestId);
      return;
    }
    if (this.finalized || this.failed) { this.ack(requestId); return; }
    if (op === 'active') {
      if (!this.open && !this.stopping) {
        const elapsed = mediaTime - this.mediaPosition;
        if (elapsed < -.02 || elapsed > .5) {
          this.fail('文件播放位置与采集边界不连续，请重新开始同传。');
          return;
        }
        // 'playing' can be delivered shortly after playback actually starts.
        // Include that already-played prefix from the bounded callback history.
        const start = Math.round((contextTime - Math.max(0, elapsed)) * this.rate);
        this.open = {start, next:start, end:null, mediaStart:this.mediaPosition};
        this.windows.push(this.open);
      }
      this.consume(mediaTime);
      this.ack(requestId);
      return;
    }
    if (op !== 'pause' && op !== 'finish') { this.ack(requestId); return; }
    if (this.open) {
      const duration = mediaTime - this.open.mediaStart;
      const end = this.open.start + Math.round(Math.max(0, duration) * this.rate);
      if (duration < -.02 || end > Math.round(contextTime * this.rate) + this.rate * .15 || end < this.open.next) {
        this.fail('文件播放时钟发生跳变，采集已停止，未跳过或补造音频。');
        return;
      }
      this.open.end = end;
      this.mediaPosition = mediaTime;
      this.open = null;
    }
    if (op === 'finish') this.stopping = true;
    const boundary = this.windows.length ? this.windows[this.windows.length - 1].end : this.latestEnd;
    this.pending.push({op, requestId, boundary});
    this.consume(mediaTime);
  }
  push(channels, startFrame, mediaTime) {
    if (this.finalized || this.failed) return;
    const length = channels[0]?.length || 0;
    if (!length) return;
    if (this.latestEnd !== null && startFrame !== this.latestEnd) {
      this.fail('HTTP 兼容采集检测到音频回调不连续，会话已停止。请保持页面在前台并减少浏览器负载。');
      return;
    }
    const mono = new Float32Array(length);
    for (const channel of channels) {
      if (channel.length !== length) { this.fail('浏览器返回了不一致的音频通道长度。'); return; }
      for (let i = 0; i < length; i++) mono[i] += channel[i] / channels.length;
    }
    this.blocks.push({start:startFrame, end:startFrame + length, mono});
    this.latestEnd = startFrame + length;
    this.consume(mediaTime);
  }
  consume(mediaTime) {
    if (this.finalized || this.failed) return;
    for (const window of this.windows) {
      const limit = window.end ?? window.start + Math.floor(Math.max(0, mediaTime - window.mediaStart) * this.rate);
      for (const block of this.blocks) {
        if (window.next >= limit) break;
        if (block.end <= window.next) continue;
        if (block.start > window.next) {
          this.fail('已播放音频超出了兼容采集缓冲区，会话已停止；不会跳过缺失片段。');
          return;
        }
        const end = Math.min(block.end, limit);
        this.append(this.resampler.push(block.mono.subarray(window.next - block.start, end - block.start)));
        window.next = end;
      }
    }
    while (this.pending.length) {
      const request = this.pending[0];
      const drained = request.boundary === null ||
        (this.latestEnd >= request.boundary && this.windows.every(window =>
          window.start >= request.boundary || window.next >= Math.min(window.end ?? Infinity, request.boundary)));
      if (!drained) break;
      this.pending.shift();
      if (request.op === 'finish' && !this.finalized) {
        this.append(this.resampler.flush());
        this.finalized = true;
      }
      this.flush();
      this.ack(request.requestId);
    }
    this.windows = this.windows.filter(window => window.end === null || window.next < window.end);
    // Preserve up to 0.5 s of pre-activation input for delayed 'playing'. Keep
    // unfinished windows longer only until the adapter's bounded drain timeout.
    const keepFrom = Math.min((this.latestEnd ?? 0) - Math.ceil(this.rate * .5),
      ...this.windows.map(window => window.next));
    while (this.blocks.length && this.blocks[0].end < keepFrom) this.blocks.shift();
    if (this.blocks.length && this.latestEnd - this.blocks[0].start > this.rate * 1.5) {
      this.fail('HTTP 兼容采集积压过多，会话已停止。请保持页面在前台，或改用 HTTPS 音频处理路径。');
    }
  }
}

/**
 * Deprecated ScriptProcessor is an explicit file-only HTTP compatibility path.
 * Audible playback bypasses its main-thread callback; the callback branch is
 * silent. Chromium/WebKit double-buffer input and output: for an on-time event,
 * input spans [playbackTime - 2*bufferDuration, playbackTime - bufferDuration).
 * playbackTime itself is assigned at dispatch, NOT a trustworthy capture stamp
 * after a main-thread stall. Anchor the input sample counter when the graph is
 * connected (before media.play), not at the first main-thread callback. Use the
 * event timestamp only to detect dispatch drift, never to skip input samples.
 */
export function createFileCaptureAdapter(context, {media, mediaOffset = 0, frameSamples = 1600, onError}) {
  const inputNode = context.createGain();
  const processor = context.createScriptProcessor(4096, 2, 1);
  const silent = context.createGain();
  silent.gain.value = 0;
  inputNode.connect(processor);
  processor.connect(silent);
  let closed = false, nextFrame = null, connected = false;
  const timers = new Map();
  const port = {
    onmessage:null,
    postMessage(data) {
      if (closed) return;
      if (data.op === 'pause' || data.op === 'finish') {
        const timer = setTimeout(() => {
          timers.delete(data.requestId);
          timeline.fail('等待文件尾部音频回调超时，当前会话已停止。请保持页面在前台后重试。');
        }, 1800);
        timers.set(data.requestId, timer);
      }
      timeline.command(data.op, data.requestId, {contextTime:context.currentTime, mediaTime:media.currentTime});
    },
    close() { close(); }
  };
  const timeline = new FileCaptureTimeline({sampleRate:context.sampleRate, mediaOffset, frameSamples,
    onMessage(data) {
      if (data.type === 'ack') { clearTimeout(timers.get(data.requestId)); timers.delete(data.requestId); }
      if (!closed) port.onmessage?.({data});
    },
    onError(error) { for (const timer of timers.values()) clearTimeout(timer); timers.clear(); onError(error); }
  });
  processor.onaudioprocess = event => {
    if (closed || timeline.finalized || timeline.failed) return;
    const size = event.inputBuffer.length;
    const estimatedStart = Math.round(event.playbackTime * context.sampleRate) - 2 * size;
    // Normal callback dispatch can shift by a few render quanta. More than
    // half a buffer is unsafe: JS callbacks can refer to overwritten buffers.
    if (!Number.isFinite(estimatedStart) || (nextFrame !== null && Math.abs(estimatedStart - nextFrame) > size / 2)) {
      timeline.fail('HTTP 兼容采集检测到浏览器主线程延迟或音频中断，会话已停止。请保持页面在前台并减少负载。');
      return;
    }
    // The graph connection establishes the render-time input origin. A delayed
    // first callback must not move that origin forward and lose the file head.
    if (nextFrame === null) { timeline.fail('音频采集节点尚未连接。'); return; }
    const channels = Array.from({length:event.inputBuffer.numberOfChannels}, (_,i) => event.inputBuffer.getChannelData(i));
    timeline.push(channels, nextFrame, media.currentTime);
    nextFrame += size;
    for (let i = 0; i < event.outputBuffer.numberOfChannels; i++) event.outputBuffer.getChannelData(i).fill(0);
  };
  function close() {
    if (closed) return;
    timeline.command('cancel', -1, {contextTime:context.currentTime, mediaTime:media.currentTime});
    closed = true;
    for (const timer of timers.values()) clearTimeout(timer);
    timers.clear();
    processor.onaudioprocess = null;
    inputNode.disconnect(); processor.disconnect(); silent.disconnect();
    port.onmessage = null;
  }
  return {
    inputNode, port,
    connect(destination) {
      if (!closed && !connected) {
        nextFrame = Math.round(context.currentTime * context.sampleRate);
        inputNode.connect(destination); silent.connect(destination); connected = true;
      }
    },
    disconnect:close
  };
}
