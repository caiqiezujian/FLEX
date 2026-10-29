/** Ordered, bounded playback of committed text. Never changes the transcript. */
const speakable = text => /[\p{L}\p{N}]/u.test(text);

// Prefer sentence/phrase boundaries, but don't wait indefinitely for punctuation.
// Return exact slices: no repeated prefix, invented punctuation or rewriting.
export function splitSpeech(text, maximum = 200, flush = false) {
  const pieces = [];
  while (text) {
    const chars = [...text];
    let boundary = 0;
    for (let i = 0; i < Math.min(chars.length, maximum); i++) {
      const c = chars[i];
      let end = /[。！？!?；;\n]/.test(c);
      if (c === '.' && (i + 1 === chars.length || /\s/.test(chars[i + 1]))) {
        const prefix = chars.slice(0, i).join('');
        end = !/(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)|\b[A-Z]|\b(?:e\.g|i\.e))$/i.test(prefix)
          && chars[i - 1] !== '.';
      }
      if (end) {
        let j = i + 1;
        while (j < Math.min(chars.length, maximum) && /[。！？!?.…；;\s"”’）)]/.test(chars[j])) j++;
        // A tiny phrase may continue in the next model turn; merge if possible.
        if (j >= 12) {boundary = j; break;}
      }
    }
    if (!boundary && chars.length >= maximum) {
      boundary = maximum;
      for (let i = maximum - 1; i >= maximum / 2; i--) {
        if (/[\s，,、：:]/.test(chars[i])) {boundary = i + 1; break;}
      }
    }
    if (!boundary && flush) boundary = Math.min(chars.length, maximum);
    if (!boundary) break;
    pieces.push(chars.slice(0, boundary).join(''));
    text = chars.slice(boundary).join('');
  }
  return {pieces, rest:text};
}

export class TranslationSpeech {
  constructor({fetcher = (...args) => fetch(...args),
    contextFactory = () => new (window.AudioContext || window.webkitAudioContext)(),
    onState = () => {}, clock = () => performance.now(),
    setTimer = (callback, delay) => setTimeout(callback, delay), clearTimer = id => clearTimeout(id),
    maxQueue = 12, maxLagMs = 30000, maxAudioSeconds = 30} = {}) {
    Object.assign(this, {fetcher, contextFactory, onState, clock, setTimer, clearTimer,
      maxQueue, maxLagMs, maxAudioSeconds});
    this.context = null; this.run = null; this.volume = .9;
  }

  async start({language, token = '', maximum = 200, timeoutSeconds = 180}) {
    this.stop();
    maximum = Math.min(maximum, language === 'zh' ? 80 : 200);
    const run = {language, token, maximum, timeoutSeconds, text:'', textAt:null,
      pending:[], ready:[], playing:null, fetching:null, ended:false, enabled:true, unlocked:false, timer:null};
    this.run = run;
    try {
      // start() must be called from a click/change gesture, before any awaits.
      this.context ||= this.contextFactory();
      if (!this.gain) {this.gain = this.context.createGain();this.gain.connect(this.context.destination);}
      this.setVolume(this.volume);
      await this.context.resume();
      if (this.run !== run || !run.enabled) return;
      if (this.context.state === 'suspended') throw new Error('浏览器未允许播放，请重新点击开启播报。');
      run.unlocked = true; void this.synthesize(run);
      this.report(run); this.tick(run);
    } catch (error) {if (this.run === run) this.fail(run, `无法开始播报：${error.message}`);}
  }

  setVolume(value) {
    this.volume = Math.min(1, Math.max(0, Number(value) || 0));
    if (this.gain) this.gain.gain.value = this.volume;
  }

  append(delta) {
    const run = this.run;
    if (!run?.enabled || run.ended || !delta) return;
    if (run.textAt === null) run.textAt = this.clock();
    run.text += delta;
    // A committed translation delta is ready to speak: do not wait for a
    // sentence terminator or a text-buffer timer before the first request.
    this.flush(run, true);
  }

  finish() {
    const run = this.run;
    if (!run?.enabled) return;
    run.ended = true; this.flush(run, true); this.report(run);
  }

  flush(run, force) {
    const result = splitSpeech(run.text, run.maximum, force);
    const created = run.textAt ?? this.clock();
    run.text = result.rest;
    run.textAt = run.text ? created : null;
    for (const text of result.pieces) {
      // Only join text that is already waiting for a request slot. Never hold
      // an idle request to collect more text, or modify an in-flight segment.
      const tail = run.pending[run.pending.length - 1];
      if (tail && [...tail.text].length + [...text].length <= run.maximum) tail.text += text;
      else if (speakable(text)) run.pending.push({text, created});
    }
    if (run.pending.length + run.ready.length + (run.fetching ? 1 : 0) > this.maxQueue) {
      this.fail(run, '播报积压过多，已停止播报；字幕保留。可重新开启，从新译文继续。');return;
    }
    void this.synthesize(run); this.report(run);
  }

  tick(run) {
    if (this.run !== run || !run.enabled) return;
    const waiting = [...run.pending, ...run.ready, ...(run.fetching ? [run.fetching] : [])];
    if (waiting.some(item => this.clock() - item.created > this.maxLagMs)) {
      this.fail(run, '播报落后超过 30 秒，已停止播报；字幕继续。可重新开启，从新译文继续。');return;
    }
    if (this.run !== run || !run.enabled) return;
    this.report(run);
    if (!run.ended || run.text || waiting.length || run.playing) {
      run.timer = this.setTimer(() => this.tick(run), 250);
    }
  }

  async synthesize(run) {
    if (this.run !== run || !run.enabled || !run.unlocked || run.fetching || run.ready.length >= 2 || !run.pending.length) return;
    const item = run.pending.shift(); run.fetching = item;
    const controller = new AbortController(); run.abort = controller;
    const timer = this.setTimer(() => controller.abort(), (run.timeoutSeconds + 5) * 1000);
    this.report(run);
    try {
      const response = await this.fetcher('/api/tts/synthesize', {
        method:'POST', headers:{'Content-Type':'application/json',
          ...(run.token ? {Authorization:`Bearer ${run.token}`} : {})},
        body:JSON.stringify({text:item.text, language:run.language}), signal:controller.signal,
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail?.message || `TTS HTTP ${response.status}`);
      }
      const bytes = await response.arrayBuffer();
      if (this.run !== run || !run.enabled) return;
      const buffer = await this.context.decodeAudioData(bytes);
      if (this.run !== run || !run.enabled) return;
      if (!Number.isFinite(buffer.duration) || buffer.duration <= 0) throw new Error('TTS 返回空音频。');
      const queued = run.ready.reduce((sum, value) => sum + value.buffer.duration, 0);
      if (queued + buffer.duration > this.maxAudioSeconds) throw new Error('待播音频超过 30 秒，已停止积压播报。');
      run.ready.push({...item, buffer}); run.fetching = null;
      this.play(run);
    } catch (error) {
      if (this.run === run && run.enabled) this.fail(run, `播报已停止：${error.name === 'AbortError' ? '合成超时' : error.message} 文字同传继续。`);
    } finally {
      this.clearTimer(timer);
      if (this.run === run && run.enabled) {
        run.fetching = null; run.abort = null;
        void this.synthesize(run); this.report(run);
      }
    }
  }

  play(run) {
    if (this.run !== run || !run.enabled || run.playing || !run.ready.length) return;
    if (this.context.state === 'suspended') {
      this.fail(run, '浏览器暂停了音频，请重新点击开启播报。文字同传继续。');return;
    }
    const item = run.ready.shift(), node = this.context.createBufferSource();
    node.buffer = item.buffer; node.connect(this.gain); run.playing = {node, item};
    node.onended = () => {
      node.disconnect();
      if (this.run !== run || !run.enabled) return;
      run.playing = null; this.play(run); void this.synthesize(run); this.report(run);
    };
    node.start(); this.report(run);
  }

  report(run) {
    if (this.run !== run || !run.enabled) return;
    const queued = run.pending.length + run.ready.length + (run.fetching ? 1 : 0);
    const state = run.playing ? 'playing' : run.fetching ? 'synthesizing' : run.ended && !queued && !run.text ? 'done' : 'waiting';
    const labels = {playing:'正在播报', synthesizing:'正在合成', done:'播报完成', waiting:'等待新增译文'};
    this.onState({state, queued, enabled:true, message:`${labels[state]}${queued ? ` · 待播 ${queued} 段` : ''}`});
  }

  fail(run, message) {
    if (this.run !== run) return;
    this.stop(); this.onState({state:'error', enabled:false, queued:0, message});
  }

  stop(message = '播报已关闭') {
    const run = this.run; this.run = null;
    if (run) {
      run.enabled = false; run.abort?.abort(); this.clearTimer(run.timer);
      if (run.playing) {
        run.playing.node.onended = null;
        try {run.playing.node.stop();run.playing.node.disconnect();} catch (_) { /* Already ended. */ }
      }
      run.pending = []; run.ready = []; run.text = ''; run.fetching = null;
    }
    this.onState({state:'off', enabled:false, queued:0, message});
  }
}
