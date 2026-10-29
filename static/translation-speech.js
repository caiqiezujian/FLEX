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
    onState = () => {}, onEvent = () => {}, clock = () => performance.now(),
    setTimer = (callback, delay) => setTimeout(callback, delay), clearTimer = id => clearTimeout(id),
    maxQueue = 12, maxLagMs = 30000, maxAudioSeconds = 30, maxConcurrent = 2, maxPrefetch = 2} = {}) {
    Object.assign(this, {fetcher, contextFactory, onState, onEvent, clock, setTimer, clearTimer,
      maxQueue, maxLagMs, maxAudioSeconds, maxConcurrent, maxPrefetch});
    this.context = null; this.run = null; this.volume = .9;
    this.events = []; this.runNumber = 0;
  }

  async start({language, token = '', maximum = 200, timeoutSeconds = 180, preview = false,
    maxConcurrent = this.maxConcurrent}) {
    this.stop();
    maximum = Math.min(maximum, language === 'zh' ? 80 : 200);
    const run = {language, token, maximum, timeoutSeconds, preview, text:'', textAt:null,
      pending:[], ready:[], inflight:new Map(), playing:null, ended:false, enabled:true,
      unlocked:false, canRequest:false, timer:null, nextRequest:0, nextPlay:0, received:0, played:0,
      id:++this.runNumber, maxConcurrent:preview ? 1 : Math.max(1, Math.min(2, Number(maxConcurrent) || 1))};
    this.run = run;
    this.report(run);
    this.event(run, 'start');
    try {
      // Invoke resume under the original click, but synthesis may proceed in
      // parallel. Ready audio waits for unlock; an unresolved resume is bounded.
      this.context ||= this.contextFactory();
      if (!this.gain) {this.gain = this.context.createGain();this.gain.connect(this.context.destination);}
      this.setVolume(this.volume);
      const resuming = this.context.resume();
      const deadline = new Promise((resolve, reject) => {
        run.cancelUnlock = () => reject(new DOMException('Aborted', 'AbortError'));
        run.unlockTimer = this.setTimer(() => reject(new Error('浏览器播放授权等待超时，请再次点击试听或开启播报。')), 10000);
      });
      run.canRequest = true; this.pump(run); this.tick(run);
      await Promise.race([resuming, deadline]);
      if (this.run !== run || !run.enabled) return;
      if (this.context.state !== 'running') throw new Error('浏览器未允许播放，请重新点击开启播报。');
      run.unlocked = true; this.event(run, 'audio_unlocked');
      this.play(run); this.pump(run); this.report(run);
    } catch (error) {if (this.run === run) this.fail(run, `无法开始播报：${error.message}`);}
    finally {this.clearTimer(run.unlockTimer);run.cancelUnlock = null;}
  }

  setVolume(value) {
    this.volume = Math.min(1, Math.max(0, Number(value) || 0));
    if (this.gain) this.gain.gain.value = this.volume;
  }

  append(delta, {turn, sourceEndSeconds} = {}) {
    const run = this.run;
    if (!run?.enabled || run.ended || !delta) return;
    if (run.textAt === null) run.textAt = this.clock();
    run.text += delta;
    this.flush(run, true, Number.isSafeInteger(turn) ? {firstTurn:turn,lastTurn:turn,sourceEndSeconds} : {});
  }

  finish() {
    const run = this.run;
    if (!run?.enabled) return;
    run.ended = true; this.flush(run, true); this.report(run);
  }

  flush(run, force, origin = {}) {
    const result = splitSpeech(run.text, run.maximum, force);
    const created = run.textAt ?? this.clock();
    run.text = result.rest;
    run.textAt = run.text ? created : null;
    for (const text of result.pieces) {
      // Coalesce only text not yet submitted. Never resend the transcript or
      // mutate an in-flight segment; preserve its first arrival/turn identity.
      const tail = run.pending[run.pending.length - 1];
      if (tail && [...tail.text].length + [...text].length <= run.maximum) {
        tail.text += text;
        if (origin.lastTurn !== undefined) {tail.lastTurn = origin.lastTurn;tail.sourceEndSeconds = origin.sourceEndSeconds;}
      } else if (speakable(text)) run.pending.push({text, created, ...origin});
    }
    if (run.pending.length + run.ready.length + run.inflight.size > this.maxQueue) {
      this.fail(run, '待播文字积压过多，已停止播报；字幕保留。可重新开启，从新译文继续。');return;
    }
    this.pump(run); this.report(run);
  }

  tick(run) {
    if (this.run !== run || !run.enabled) return;
    // Queue age is a visible lag warning, not a reason to discard successfully
    // synthesized speech. Request deadlines and bounded queue sizes still apply.
    this.report(run);
    if (!run.ended || run.text || run.pending.length || run.ready.length || run.inflight.size || run.playing || !run.unlocked) {
      run.timer = this.setTimer(() => this.tick(run), 250);
    }
  }

  pump(run) {
    if (this.run !== run || !run.enabled || !run.canRequest) return;
    while (run.pending.length && run.inflight.size < run.maxConcurrent
        && run.ready.length + run.inflight.size < this.maxPrefetch
        && run.ready.reduce((sum, item) => sum + item.buffer.duration, 0) < this.maxAudioSeconds) {
      const item = run.pending.shift();
      item.sequence = run.nextRequest++; item.requestAt = this.clock(); item.controller = new AbortController();
      run.inflight.set(item.sequence, item);
      this.event(run, 'request_start', item, {queue_seconds:(item.requestAt-item.created)/1000, characters:[...item.text].length});
      void this.synthesize(run, item);
    }
  }

  async synthesize(run, item) {
    const controller = item.controller;
    const timer = this.setTimer(() => controller.abort(), (run.timeoutSeconds + 5) * 1000);
    this.report(run);
    try {
      let response;
      for (let attempt = 0; ; attempt++) {
        if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
        response = await this.fetcher('/api/tts/synthesize', {
          method:'POST', headers:{'Content-Type':'application/json',
            ...(run.token ? {Authorization:`Bearer ${run.token}`} : {})},
          body:JSON.stringify({text:item.text, language:run.language}), signal:controller.signal,
        });
        if (this.run !== run || !run.enabled) return;
        if (response.ok) break;
        const data = await response.json().catch(() => ({}));
        const code = data.detail?.code;
        // tts_busy rejects work before synthesis; retries of transport errors
        // or upstream failures could duplicate remote work and are not attempted.
        if (response.status === 503 && code === 'tts_busy' && attempt < 3) {
          item.retrying = attempt + 1;
          this.event(run, 'busy_retry', item, {attempt:attempt+1});this.report(run);
          await this.retryDelay(controller.signal, 1000 * 2 ** attempt);
          item.retrying = 0;
          continue;
        }
        throw new Error(`TTS HTTP ${response.status}${code ? ` [${code}]` : ''}${data.detail?.message ? `：${data.detail.message}` : ''}`);
      }
      const bytes = await response.arrayBuffer();
      if (this.run !== run || !run.enabled) return;
      if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      const receivedAt = this.clock();
      const buffer = await this.context.decodeAudioData(bytes);
      if (this.run !== run || !run.enabled) return;
      if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
      if (!Number.isFinite(buffer.duration) || buffer.duration <= 0 || buffer.duration > 60) throw new Error('TTS 返回空音频或超过 60 秒的单段音频。');
      item.buffer = buffer;item.readyAt = this.clock();run.received++;
      const serviceSeconds = response.headers?.get?.('X-TTS-Service-Seconds');
      this.event(run, 'audio_ready', item, {request_seconds:(receivedAt-item.requestAt)/1000,
        decode_seconds:(item.readyAt-receivedAt)/1000, audio_seconds:buffer.duration,
        ...(serviceSeconds && Number.isFinite(Number(serviceSeconds)) ? {web_service_seconds:Number(serviceSeconds)} : {}),
        request_id:response.headers?.get?.('X-TTS-Request-ID') || null});
      run.inflight.delete(item.sequence);
      run.ready.push(item);run.ready.sort((a,b) => a.sequence-b.sequence);
      // The duration threshold pauses new work; it never kills the current
      // sound or discards an already returned buffer. At most two are ahead.
      this.play(run);
    } catch (error) {
      if (this.run === run && run.enabled) this.fail(run, `${run.preview ? '试听失败' : '播报已停止'}：${error.name === 'AbortError' ? '合成超时' : error.message} 文字同传继续。`);
    } finally {
      this.clearTimer(timer);run.inflight.delete(item.sequence);
      if (this.run === run && run.enabled) {this.pump(run);this.report(run);}
    }
  }

  retryDelay(signal, milliseconds) {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {reject(new DOMException('Aborted', 'AbortError'));return;}
      const cancel = () => {
        this.clearTimer(timer); signal.removeEventListener('abort', cancel);
        reject(new DOMException('Aborted', 'AbortError'));
      };
      const timer = this.setTimer(() => {signal.removeEventListener('abort', cancel);resolve();}, milliseconds);
      signal.addEventListener('abort', cancel, {once:true});
    });
  }

  play(run) {
    if (this.run !== run || !run.enabled || !run.unlocked || run.playing
        || !run.ready.length || run.ready[0].sequence !== run.nextPlay) return;
    if (this.context.state !== 'running') {
      this.fail(run, '浏览器暂停了音频，请重新点击开启播报。文字同传继续。');return;
    }
    const item = run.ready.shift(), node = this.context.createBufferSource();
    node.buffer = item.buffer; node.connect(this.gain); run.playing = {node, item};
    node.onended = () => {
      node.disconnect();
      if (this.run !== run || !run.enabled) return;
      this.event(run, 'play_end', item);
      run.playing = null; this.play(run);this.pump(run);this.report(run);
    };
    try {node.start();}
    catch (error) {this.fail(run, `无法播放音频：${error.message}`);return;}
    run.nextPlay++;run.played++;
    this.event(run, 'play_start', item, {ready_wait_seconds:(this.clock()-item.readyAt)/1000,
      text_to_play_seconds:(this.clock()-item.created)/1000, audio_seconds:item.buffer.duration});
    this.report(run);
  }

  event(run, phase, item, details = {}) {
    const entry = {run:run.id, preview:run.preview, phase, at_seconds:this.clock()/1000,
      ...(item ? {segment:item.sequence+1,first_turn:item.firstTurn,last_turn:item.lastTurn,source_end_seconds:item.sourceEndSeconds} : {}), ...details};
    this.events.push(entry);if (this.events.length > 200) this.events.shift();
    try {this.onEvent(entry);} catch (_) { /* Diagnostics must never interrupt speech. */ }
  }

  getDiagnostics() {return this.events.map(entry => ({...entry}));}

  report(run) {
    if (this.run !== run || !run.enabled) return;
    const waiting = [...run.pending, ...run.ready, ...run.inflight.values()];
    const lagging = !run.preview && waiting.some(item => this.clock()-item.created > this.maxLagMs);
    const fetching = [...run.inflight.values()];
    const retrying = fetching.find(item => item.retrying);
    const state = !run.unlocked ? 'unlocking' : run.playing ? 'playing' : retrying ? 'retrying'
      : fetching.length ? 'synthesizing' : run.ended && !waiting.length && !run.text ? 'done' : 'waiting';
    const labels = {unlocking:'正在开启浏览器声音', playing:'正在播报', synthesizing:'正在请求语音',
      retrying:`语音服务忙，正在重试 ${retrying?.retrying}/3`, done:'播报完成', waiting:'等待新增译文'};
    const currentItem = run.playing?.item;
    const turnLabel = currentItem?.firstTurn !== undefined
      ? ` · 译文第 ${currentItem.firstTurn}${currentItem.lastTurn !== currentItem.firstTurn ? `–${currentItem.lastTurn}` : ''} 轮` : '';
    const requestSeconds = fetching.length ? Math.floor(Math.max(...fetching.map(item => this.clock()-item.requestAt))/1000) : 0;
    const elapsed = fetching.length ? ` · ${fetching.length} 段请求中（最长 ${requestSeconds} 秒）` : '';
    const counts = ` · 已返回 ${run.received} 段 / 已开播 ${run.played} 段`;
    this.onState({state, queued:waiting.length, enabled:true, preview:run.preview, lagging,
      received:run.received, played:run.played, inflight:fetching.length,
      message:`${labels[state]}${turnLabel}${elapsed}${counts}${lagging ? ' · 播报有积压，正按顺序继续' : ''}`});
  }

  fail(run, message) {
    if (this.run !== run) return;
    this.event(run, 'error', null, {message});
    this.stop(); this.onState({state:'error', enabled:false, queued:0, preview:run.preview, message});
  }

  stop(message = '播报已关闭') {
    const run = this.run; this.run = null;
    if (run) {
      run.enabled = false;this.clearTimer(run.timer);run.cancelUnlock?.();
      for (const item of run.inflight.values()) item.controller.abort();
      if (run.playing) {
        run.playing.node.onended = null;
        try {run.playing.node.stop();run.playing.node.disconnect();} catch (_) { /* Already ended. */ }
      }
      this.event(run, 'stop');run.pending = [];run.ready = [];run.text = '';run.inflight.clear();
    }
    this.onState({state:'off', enabled:false, queued:0, message});
  }
}
