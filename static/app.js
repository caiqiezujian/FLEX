import { makeAudioPacket } from './pcm-core.js';
import { FilePlaybackClock } from './file-playback.js';

const $ = id => document.getElementById(id);
const ui = Object.fromEntries([
  'serverStatus','modelLabel','modelMeta','backendBadge','demoBanner','errorBox','errorTitle','errorMessage','dismissError',
  'sourceLang','targetLang','swapLang','settings','chunkSeconds','adaptive','adaptiveHint','authField','authToken','clearButton',
  'fileMode','micMode','fileArea','micArea','mediaStage','media','audioArtwork','dropZone','fileType','fileName','fileDetails',
  'chooseFile','chooseSubtitle','subtitleName','removeSubtitle','fileInput','subtitleInput','micTitle','micDescription','micMeter',
  'captureDot','sourceClock','clockDescription','targetBadge','exportTxt','exportJson','sessionStatus','latestTag','transcriptViewport',
  'transcriptEmpty','transcript','historyNotice','translationCount','followButton','turnCount','turnList','startButton','pauseButton','finishButton',
  'receivedMetric','pendingMetric','serviceMetric','processedMetric','footerHint'
].map(id => [id, $(id)]));

const languageNames = { en: '英语', zh: '中文' };
const BUSY_STATES = new Set(['connecting','live','paused','draining']);
const MAX_NETWORK_BUFFER_BYTES = 256 * 1024;
// The server keeps the full append-only result on disk. The browser is a
// bounded live view, including when a microphone session lasts for hours.
const MAX_VISIBLE_FRAGMENTS = 500;
const MAX_VISIBLE_CHARACTERS = 60000;
const MAX_RECENT_TURNS = 200;
const MAX_RECENT_RECORD_CHARACTERS = 120000;
let config = null, mode = 'file', state = 'idle', selectedFile = null;
let mediaUrl = null, subtitleUrl = null, subtitleTrack = null;
let audioContext = null, monitor = null, workletLoaded = false;
let session = null, sessionNumber = 0, followOutput = true, pollTimer = null;
let currentTranslation = '', turns = [], transcriptFragments = 0, lastMetrics = {};
let lastExport = null, microphoneAdaptive = ui.adaptive.checked;
let totalTurns = 0, translationCharacters = 0, recentRecordCharacters = 0, recordsTrimmed = false;
let visibleFragments = [], transcriptTrimmed = false, windowRollovers = 0, exportInProgress = false;

function busy() { return BUSY_STATES.has(state); }
function current(s) { return session === s; }
function time(value, tenths = true) {
  const seconds = Math.max(0, Number.isFinite(value) ? value : 0);
  const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds / 60) % 60;
  const rest = seconds % 60;
  return `${hours ? `${hours}:` : ''}${String(minutes).padStart(2,'0')}:${rest.toFixed(tenths ? 1 : 0).padStart(tenths ? 4 : 2,'0')}`;
}
function formatBytes(bytes) { return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`; }
function sessionLimitLabel() {
  const seconds = Number(config?.max_session_seconds);
  if (!Number.isFinite(seconds) || seconds <= 0) return '持续会话';
  return seconds >= 3600 ? `最长 ${Number((seconds/3600).toFixed(1))} 小时` : `最长 ${seconds} 秒`;
}
function showError(message, title = '会话提示') {
  ui.errorTitle.textContent = title;
  ui.errorMessage.textContent = String(message);
  ui.errorBox.hidden = false;
}
function statusLabel(text, kind = '') {
  ui.sessionStatus.className = `state-label ${kind}`;
  ui.sessionStatus.replaceChildren(Object.assign(document.createElement('i'), {}), document.createTextNode(text));
}
function updateControls() {
  const active = busy();
  ui.startButton.disabled = active || !config || config.status !== 'ready' || (mode === 'file' && !selectedFile);
  ui.startButton.lastChild.textContent = ['done','error'].includes(state) ? '重新开始' : '开始同传';
  ui.pauseButton.disabled = mode !== 'file' || !['live','paused'].includes(state);
  ui.pauseButton.textContent = state === 'paused' ? '继续播放' : '暂停播放';
  ui.finishButton.disabled = !['connecting','live','paused'].includes(state);
  ui.finishButton.lastChild.textContent = state === 'connecting' ? '取消连接' : '结束同传';
  for (const element of [ui.fileMode,ui.micMode,ui.sourceLang,ui.targetLang,ui.swapLang,ui.chooseFile,ui.chunkSeconds,ui.adaptive,ui.authToken]) element.disabled = active;
  ui.adaptive.disabled = active || mode === 'file';
  ui.adaptive.checked = mode === 'microphone' && microphoneAdaptive;
  ui.adaptiveHint.textContent = mode === 'file' ? '文件固定分块' : `最多合并 ${config?.max_chunk_seconds || 3} 秒`;
  ui.dropZone.disabled = active;
  ui.chooseSubtitle.disabled = active || !selectedFile;
  ui.removeSubtitle.disabled = active;
  ui.clearButton.disabled = active || !totalTurns;
  const hasServerResult = Boolean(session?.sessionId);
  ui.exportTxt.disabled = exportInProgress || (!hasServerResult && !currentTranslation);
  ui.exportJson.disabled = exportInProgress || (!hasServerResult && !turns.length && !lastExport);
  ui.captureDot.classList.toggle('active', state === 'live');
  ui.micArea.querySelector('.microphone-stage').classList.toggle('active', state === 'live' && mode === 'microphone');
  ui.micTitle.textContent = state === 'live' ? '正在录音' : '麦克风就绪';
  ui.micDescription.textContent = state === 'live' ? '' : '开始时需要麦克风授权';
  ui.audioArtwork.classList.toggle('playing', mode === 'file' && !ui.media.paused);
  const labels = {idle:'等待开始',connecting:'正在准备',live:'同传进行中',paused:'播放已暂停',draining:'处理剩余音频',done:'同传已结束',error:'会话已停止'};
  let label = state === 'connecting' ? session?.preparationLabel || labels.connecting : labels[state] || state;
  if (state === 'live') {
    if ((lastMetrics.pending_audio_s || 0) >= Math.max(3, (session?.chunkSeconds || 1) * 2)) label = '正在处理积压音频';
    else if (session?.lastTag === 'WAIT') label = '等待后续音频';
  }
  statusLabel(label, state === 'live' ? 'live' : state === 'error' ? 'error' : '');
}
function setState(value) { state = value; updateControls(); }
function setMode(value) {
  if (busy()) return;
  mode = value;
  ui.media.pause();
  ui.fileMode.classList.toggle('selected', value === 'file');
  ui.micMode.classList.toggle('selected', value === 'microphone');
  ui.fileMode.setAttribute('aria-pressed', String(value === 'file'));
  ui.micMode.setAttribute('aria-pressed', String(value === 'microphone'));
  ui.fileArea.hidden = value !== 'file';
  ui.micArea.hidden = value !== 'microphone';
  updateControls();
}
function updateDirection(source) {
  if (busy()) return;
  if (source === 'target') ui.sourceLang.value = ui.targetLang.value === 'zh' ? 'en' : 'zh';
  else ui.targetLang.value = ui.sourceLang.value === 'zh' ? 'en' : 'zh';
  ui.targetBadge.textContent = languageNames[ui.targetLang.value];
  ui.transcript.lang = ui.targetLang.value;
}
function clearTranscript() {
  if (busy()) return;
  currentTranslation = ''; turns = []; transcriptFragments = 0; lastExport = null; lastMetrics = {};
  visibleFragments = []; totalTurns = 0; translationCharacters = 0; recentRecordCharacters = 0;
  transcriptTrimmed = false; recordsTrimmed = false; windowRollovers = 0; session = null;
  ui.transcript.replaceChildren(); ui.turnList.replaceChildren();
  const hint = document.createElement('p'); hint.className = 'no-turns'; hint.textContent = '暂无记录'; ui.turnList.append(hint);
  ui.transcriptEmpty.hidden = false; ui.translationCount.textContent = '0 个输出片段'; ui.turnCount.textContent = '0 轮';
  ui.historyNotice.hidden = true;
  ui.latestTag.className = 'tag tag-neutral'; ui.latestTag.textContent = '—';
  followOutput = true; ui.followButton.hidden = true;
  renderMetrics({received_audio_s:0,processed_audio_s:0,pending_audio_s:0});
  ui.serviceMetric.replaceChildren(document.createTextNode('—'), Object.assign(document.createElement('small'), {textContent:' s'}));
  setState('idle');
}
function renderMetric(element, value) {
  element.replaceChildren(document.createTextNode(Number(value || 0).toFixed(1)), Object.assign(document.createElement('small'), {textContent:' s'}));
}
function renderMetrics(data) {
  // Do not accidentally retain a legacy done.translation or entire turn in
  // this long-lived metrics object.
  for (const key of ['received_audio_s','processed_audio_s','pending_audio_s','service_s','turn_count','translation_chars','context']) {
    if (data[key] !== undefined) lastMetrics[key] = data[key];
  }
  renderMetric(ui.receivedMetric,lastMetrics.received_audio_s);
  renderMetric(ui.processedMetric,lastMetrics.processed_audio_s);
  renderMetric(ui.pendingMetric,lastMetrics.pending_audio_s);
  ui.pendingMetric.classList.toggle('warning', (lastMetrics.pending_audio_s || 0) > 3);
  if (data.context?.rollovers !== undefined) windowRollovers = Number(data.context.rollovers) || 0;
  updateRecordCounts();
  if (state === 'live') updateControls();
}
function updateRecordCounts() {
  ui.translationCount.textContent = `${transcriptFragments} 个输出片段${transcriptTrimmed ? ` · 显示最近 ${visibleFragments.length} 段` : ''}`;
  ui.turnCount.textContent = `${totalTurns} 轮${recordsTrimmed ? ` · 最近 ${turns.length} 轮` : ''}${windowRollovers ? ` · 窗口滚动 ${windowRollovers} 次` : ''}`;
  ui.historyNotice.hidden = !transcriptTrimmed;
}
function appendResearchRow(row) {
  if (ui.turnList.querySelector('.no-turns')) ui.turnList.replaceChildren();
  ui.turnList.append(row);
  while (ui.turnList.children.length > MAX_RECENT_TURNS) ui.turnList.firstElementChild.remove();
}
function appendContextWindow(data) {
  renderMetrics(data);
  const context = data.context || {};
  const row = document.createElement('div'); row.className = 'turn-row context-window';
  const text = document.createElement('p');
  text.textContent = `上下文窗口已滚动${context.retained_turns !== undefined ? ` · 保留 ${context.retained_turns} 轮` : ''}${context.retained_audio_seconds !== undefined ? ` / ${Number(context.retained_audio_seconds).toFixed(1)} 秒音频` : ''}`;
  row.append(text); appendResearchRow(row);
}
function compactDone(event) {
  const {translation, ...summary} = event;
  // The full transcript belongs to the export endpoint, never browser state.
  return summary;
}
function appendTurn(data, s) {
  if (!['WAIT','FLEX','DIRECT'].includes(data.tag) || typeof data.delta !== 'string') throw new Error('服务器返回了无法识别的标签或译文格式。');
  if ((data.tag === 'WAIT' && data.delta) || (data.tag !== 'WAIT' && !data.delta.trim())) throw new Error('服务器输出违反 WAIT / FLEX / DIRECT 协议，会话已停止。');
  const record = {...data, client_received_at:new Date().toISOString(), client_elapsed_s:(performance.now()-s.startedAt)/1000, media_time_s:mode === 'file' ? ui.media.currentTime : null};
  const recordSize = JSON.stringify(record).length;
  if (recordSize <= MAX_RECENT_RECORD_CHARACTERS) {
    turns.push(record); recentRecordCharacters += recordSize;
  } else recordsTrimmed = true;
  while (turns.length > MAX_RECENT_TURNS || recentRecordCharacters > MAX_RECENT_RECORD_CHARACTERS) {
    recentRecordCharacters -= JSON.stringify(turns.shift()).length; recordsTrimmed = true;
  }
  totalTurns++; s.lastTag = data.tag;
  ui.latestTag.className = `tag tag-${data.tag.toLowerCase()}`; ui.latestTag.textContent = data.tag;
  if (data.tag !== 'WAIT') {
    translationCharacters += [...data.delta].length;
    const span = document.createElement('span'); span.className = `fragment ${data.tag.toLowerCase()}`;
    // A single unusually large output is bounded too. The visible suffix is
    // explicitly marked partial; server exports still contain the full delta.
    const visibleDelta = data.delta.length > MAX_VISIBLE_CHARACTERS ? data.delta.slice(-MAX_VISIBLE_CHARACTERS) : data.delta;
    if (visibleDelta !== data.delta) transcriptTrimmed = true;
    span.textContent = visibleDelta;
    span.title = `${data.tag} · 输入 ${time(data.source_start_s + s.mediaOffset)} – ${time(data.source_end_s + s.mediaOffset)} · 推理 ${Number(data.service_s).toFixed(2)} s`;
    ui.transcript.append(span); visibleFragments.push(visibleDelta); currentTranslation += visibleDelta;
    const beforeTrimHeight = ui.transcriptViewport.scrollHeight;
    while (visibleFragments.length > MAX_VISIBLE_FRAGMENTS || currentTranslation.length > MAX_VISIBLE_CHARACTERS) {
      currentTranslation = currentTranslation.slice(visibleFragments.shift().length);
      ui.transcript.firstElementChild.remove(); transcriptTrimmed = true;
    }
    if (!followOutput) ui.transcriptViewport.scrollTop = Math.max(0,ui.transcriptViewport.scrollTop + ui.transcriptViewport.scrollHeight-beforeTrimHeight);
    ui.transcriptEmpty.hidden = true; transcriptFragments++;
    if (followOutput) requestAnimationFrame(() => { ui.transcriptViewport.scrollTop = ui.transcriptViewport.scrollHeight; });
  }
  const row = document.createElement('div'); row.className = `turn-row ${data.tag.toLowerCase()}`;
  const header = document.createElement('div'); header.className = 'turn-row-header';
  const tag = document.createElement('span'); tag.className = `tag tag-${data.tag.toLowerCase()}`; tag.textContent = data.tag;
  const timing = document.createElement('time'); timing.textContent = `${time(data.source_start_s+s.mediaOffset)} – ${time(data.source_end_s+s.mediaOffset)}`;
  header.append(tag,timing);
  const latency = document.createElement('span'); latency.className = 'turn-latency'; latency.textContent = `推理 ${Number(data.service_s).toFixed(2)}s · 提交前等待 ${Number(data.queue_wait_s).toFixed(2)}s`;
  const text = document.createElement('p'); text.textContent = data.tag === 'WAIT' ? '等待后续音频' : data.delta.length > 4000 ? `${data.delta.slice(0,4000)}…（完整输出见导出）` : data.delta;
  row.append(header,latency,text); appendResearchRow(row);
  updateRecordCounts();
  renderMetric(ui.serviceMetric,data.service_s);
  renderMetrics({processed_audio_s:data.source_end_s,...(data.context ? {context:data.context} : {})});
  updateControls();
}

async function loadConfig() {
  try {
    const response = await fetch('/api/config', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    config = await response.json();
    ui.modelLabel.textContent = config.model_label || '本地双向模型';
    ui.backendBadge.textContent = config.backend === 'mock' ? 'MOCK' : 'LOCAL';
    ui.backendBadge.classList.toggle('mock',config.backend === 'mock');
    ui.demoBanner.hidden = config.backend !== 'mock';
    ui.modelMeta.textContent = config.backend === 'mock' ? '模拟输出' : '中英互译';
    ui.serverStatus.className = `server-status ${config.status}`;
    ui.serverStatus.replaceChildren(document.createElement('i'), document.createTextNode(config.status === 'ready' ? '服务已就绪' : config.status === 'error' ? '模型加载失败' : '模型加载中'));
    ui.authField.hidden = !config.auth_required;
    ui.adaptiveHint.textContent = `最多合并 ${config.max_chunk_seconds || 3} 秒`;
    if (!busy()) {
      if ([...ui.chunkSeconds.options].some(x => x.value === String(config.chunk_seconds))) ui.chunkSeconds.value = String(config.chunk_seconds);
      for (const option of ui.chunkSeconds.options) option.disabled = Number(option.value) > (config.max_chunk_seconds || 3);
    }
    if (config.status === 'error') showError(config.error || '请检查服务器日志和本地模型路径。','模型尚未就绪');
    if (config.status === 'loading') pollTimer = setTimeout(loadConfig,3000);
    else pollTimer = null;
    ui.footerHint.textContent = `${sessionLimitLabel()} · 单会话`;
    updateControls();
  } catch (error) {
    config = null; ui.serverStatus.className = 'server-status error'; ui.serverStatus.replaceChildren(document.createElement('i'),document.createTextNode('服务未连接'));
    showError(`无法连接本地同传服务：${error.message}。请确认服务器已启动，再刷新页面。`,'连接失败');
    updateControls();
  }
}

async function ensureAudioContext() {
  // This function is exclusively for microphone input. File playback never
  // constructs an AudioContext and works over ordinary intranet HTTP.
  if (!window.isSecureContext) {
    throw new Error('普通远程 HTTP 不能请求麦克风权限。请选择音视频文件进行同传；麦克风模式需要 HTTPS 或 localhost。');
  }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('当前浏览器不支持 Web Audio，请使用近期版本 Chrome 或 Edge。');
  if (!audioContext) {
    audioContext = new AudioContextClass({latencyHint:'interactive'});
    monitor = audioContext.createGain(); monitor.connect(audioContext.destination);
  }
  await audioContext.resume();
  if (!audioContext.audioWorklet || !window.AudioWorkletNode) throw new Error('麦克风模式需要支持 AudioWorklet 的近期版本 Chrome / Edge，以及 HTTPS 或 localhost。');
  if (!workletLoaded) {
    await audioContext.audioWorklet.addModule('/static/capture-worklet.js'); workletLoaded = true;
  }
  return 'worklet';
}
function releaseUploadedFile(s) {
  if (!s.mediaId || s.mediaClaimed || s.uploadReleased) return;
  s.uploadReleased = true;
  void fetch(`/api/media/${encodeURIComponent(s.mediaId)}`,{method:'DELETE',headers:s.authToken ? {Authorization:`Bearer ${s.authToken}`} : {},keepalive:true}).catch(() => {});
}
async function prepareUploadedFile(s) {
  if (config.file_mode !== 'uploaded') throw new Error('服务器尚未启用文件上传同传。请更新并重启服务，然后刷新页面；不会退回不稳定的浏览器采集模式。');
  ui.media.pause(); ui.media.playbackRate = 1;
  s.captureMode = 'server-file'; s.preparationLabel = '正在上传文件 · 完成后准备音轨'; updateControls();
  const result = await new Promise((resolve,reject) => {
    const xhr = new XMLHttpRequest(); s.uploadRequest = xhr;
    xhr.open('POST',`/api/media?name=${encodeURIComponent(s.sourceName)}`);
    xhr.setRequestHeader('Content-Type','application/octet-stream');
    if (s.authToken) xhr.setRequestHeader('Authorization',`Bearer ${s.authToken}`);
    xhr.upload.onprogress = event => {
      if (!current(s) || s.terminal) return;
      s.preparationLabel = event.lengthComputable && event.loaded < event.total
        ? `正在上传文件 · ${Math.floor(event.loaded/event.total*100)}%`
        : '上传完成 · 服务器正在准备音轨';
      updateControls();
    };
    xhr.onload = () => {
      s.uploadRequest = null;
      let body;
      try { body = JSON.parse(xhr.responseText); } catch (_) { reject(new Error(`文件上传失败（HTTP ${xhr.status}），服务器未返回有效结果。`)); return; }
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = typeof body.detail === 'string' ? body.detail : body.detail?.message;
        reject(new Error(detail || body.message || `文件上传失败（HTTP ${xhr.status}）。`)); return;
      }
      // Save the ID before checking cancellation so a late upload response is
      // still deleted rather than leaving an unclaimed file on the server.
      s.mediaId = body.media_id;
      if (!current(s) || s.terminal) releaseUploadedFile(s);
      resolve(body);
    };
    xhr.onerror = () => {s.uploadRequest = null;reject(new Error('文件上传连接中断，请检查网络后重试。'));};
    xhr.onabort = () => {s.uploadRequest = null;reject(new Error('文件上传已取消。'));};
    xhr.send(selectedFile);
  });
  if (!current(s) || s.terminal) return;
  if (typeof result.media_id !== 'string' || !result.media_id || result.sample_rate !== 16000 || !Number.isSafeInteger(result.samples) || result.samples <= 0) throw new Error('服务器返回的音频准备结果不完整，请检查服务器版本。');
  const duration = result.samples/16000;
  if (!Number.isFinite(result.duration_s) || Math.abs(result.duration_s-duration) > 1/16000) throw new Error('服务器返回的音频时长与采样数不一致。');
  s.fileClock = new FilePlaybackClock({offset:s.mediaOffset,duration,browserDuration:ui.media.duration});
  if (Math.abs(ui.media.currentTime-s.mediaOffset) > 1/16000) throw new Error('文件准备期间播放位置发生了变化，请在新位置重新开始同传。');
  s.preparationLabel = '音轨已准备 · 正在连接推理服务'; updateControls();
}
function filePosition(s, finishing = false) {
  const position = s.fileClock.read(ui.media.currentTime,finishing && ui.media.ended);
  s.sentSamples = Math.round((position-s.mediaOffset)*16000);
  return position;
}
function sendFileProgress(s, force = false) {
  if (!current(s) || s.terminal || !s.ready || s.sourceKind !== 'file' || !['live','paused'].includes(state)) return;
  try {
    const position = filePosition(s);
    if (!force && position === s.lastPosition) return;
    if (s.ws?.readyState !== WebSocket.OPEN) throw new Error('同传连接已断开，请重新开始。');
    // Progress messages carry positions, not audio. After a delayed callback
    // one update releases the missing prefix from the intact server-side file.
    s.ws.send(JSON.stringify({type:'progress',position_s:position})); s.lastPosition = position;
  } catch (error) {void failSession(s,error.message);}
}
function gate(s, operation) {
  if (!s.node) return Promise.resolve();
  const requestId = ++s.gateSequence;
  return new Promise((resolve,reject) => {
    const timer = setTimeout(() => {s.acks.delete(requestId);reject(new Error('音频采集线程没有响应，请结束会话后重新开始。'));},3000);
    s.acks.set(requestId,{resolve:() => {clearTimeout(timer);resolve();},reject:() => {clearTimeout(timer);reject(new Error('音频采集已关闭'));}});
    s.node.port.postMessage({op:operation,requestId});
  });
}
function sendPCM(s, samples) {
  if (!current(s) || s.terminal || !samples.length) return;
  if (!s.ws || s.ws.readyState !== WebSocket.OPEN) { void failSession(s,'音频连接已断开，未发送的音频不会自动补传。请重新开始会话。'); return; }
  const packet = makeAudioPacket(s.sequence,s.sentSamples,samples);
  if (s.ws.bufferedAmount + packet.byteLength > MAX_NETWORK_BUFFER_BYTES) {
    void failSession(s,'网络发送积压超过安全上限，会话已停止。没有静默丢帧或跳过音频；请检查网络后重新开始。'); return;
  }
  s.ws.send(packet); s.sequence++; s.sentSamples += samples.length;
}
async function prepareCapture(s) {
  if (s.sourceKind === 'file') {await prepareUploadedFile(s);return;}
  s.captureMode = await ensureAudioContext();
  if (!current(s) || s.terminal) return;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('浏览器无法访问麦克风，请确认 HTTPS / localhost 以及麦克风权限。');
  s.stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
  if (!current(s) || s.terminal) { s.stream.getTracks().forEach(track => track.stop()); return; }
  s.source = audioContext.createMediaStreamSource(s.stream);
  for (const track of s.stream.getAudioTracks()) track.addEventListener('ended',() => {if (!s.terminal && state === 'live') void failSession(s,'麦克风连接中断，请检查设备后重新开始。');});
  monitor.gain.value = 0;
  s.node = new AudioWorkletNode(audioContext,'flex-capture',{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[2],processorOptions:{sampleRate:16000,frameSamples:1600}});
  s.node.port.onmessage = ({data}) => {
    if (data.type === 'ack') { const ack = s.acks.get(data.requestId); if (ack) { s.acks.delete(data.requestId); ack.resolve(); } }
    if (data.type === 'pcm') sendPCM(s,new Int16Array(data.buffer));
    if (data.type === 'level' && current(s)) ui.micMeter.firstElementChild.style.width = `${Math.min(100,data.value * 400)}%`;
  };
  s.node.onprocessorerror = () => {void failSession(s,'浏览器音频处理线程发生错误，请重新开始会话。');};
  s.source.connect(s.node.inputNode || s.node); s.node.connect(monitor);
}
async function cleanupCapture(s) {
  clearInterval(s.pingTimer);
  clearInterval(s.progressTimer);
  s.uploadRequest?.abort();
  releaseUploadedFile(s);
  if (s.cleanupPromise) return s.cleanupPromise;
  s.cleanupPromise = (async () => {
    if (s.sourceKind === 'file' && current(s)) ui.media.pause();
    if (s.node) {
      try { await gate(s,'cancel'); } catch (_) { /* Teardown still disconnects nodes. */ }
      s.node.port.onmessage = null; s.node.disconnect(); s.node.port.close(); s.node = null;
    }
    if (s.source) {try{s.source.disconnect();}catch(_){}s.source = null;}
    if (s.stream) {s.stream.getTracks().forEach(track => track.stop());s.stream = null;}
    for (const ack of s.acks.values()) ack.reject();
    s.acks.clear(); ui.micMeter.firstElementChild.style.width = '0%';
  })();
  return s.cleanupPromise;
}
function openSocket(s) {
  return new Promise((resolve,reject) => {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`); s.ws = ws;
    const startupTimer = setTimeout(() => {reject(new Error('等待服务器准备超时，请检查模型加载日志。'));void failSession(s,'等待服务器准备超时，请检查模型加载日志。');},120000);
    s.startupTimer = startupTimer;
    ws.onopen = () => {
      if (s.terminal) {ws.close();return;}
      ws.send(JSON.stringify({type:'start',source_lang:s.sourceLang,target_lang:s.targetLang,chunk_seconds:s.chunkSeconds,adaptive:s.adaptive,source_kind:s.sourceKind,source_name:s.sourceName,media_offset_s:s.mediaOffset,...(s.mediaId ? {media_id:s.mediaId} : {}),...(s.authToken ? {token:s.authToken} : {})}));
    };
    ws.onmessage = ({data}) => {
      if (!current(s)) return;
      try {
        const event = JSON.parse(data);
        if (s.terminal) {
          // A server may send its accepted-prefix accounting after an error.
          // Preserve it without turning an errored session into a success.
          if (event.type === 'done' || event.type === 'status') {
            if (event.type === 'done') s.doneEvent = compactDone(event);
            renderMetrics(event); lastExport = exportData(s);
          }
          return;
        }
        if (event.type === 'ready') {
          clearTimeout(startupTimer);
          if (event.sample_rate && event.sample_rate !== 16000) throw new Error('服务器音频采样率与浏览器协议不一致。');
          if (s.sourceKind === 'file') {
            if (event.file_mode !== 'uploaded' || event.media_id !== s.mediaId) throw new Error('服务器未确认上传文件会话，请更新并重启服务。');
            s.mediaClaimed = true;
          }
          s.sessionId = event.session_id; s.readyEvent = event; s.ready = true;
          s.pingTimer = setInterval(() => {if(!s.terminal && ws.readyState === WebSocket.OPEN)ws.send(JSON.stringify({type:'ping'}));},20000);
          resolve();
        } else if (event.type === 'turn') appendTurn(event,s);
        else if (event.type === 'status') renderMetrics(event);
        else if (event.type === 'context_window') appendContextWindow(event);
        else if (event.type === 'error') {renderMetrics(event);reject(new Error(event.message || event.code)); void failSession(s,event.message || event.code,event.code || '服务器返回错误',false);}
        else if (event.type === 'done') {
          clearTimeout(startupTimer); s.terminal = true; s.doneEvent = compactDone(event); s.endedAt = new Date().toISOString();
          renderMetrics(event);
          if ((Number.isSafeInteger(event.turn_count) && event.turn_count !== totalTurns) || (Number.isSafeInteger(event.translation_chars) && event.translation_chars !== translationCharacters)) showError('服务器结束计数与本页面接收的输出不一致，请通过 JSON 导出检查完整服务端记录。','结束记录检查');
          else if (event.final_wait) showError('最后一轮为 WAIT。已处理所有接收的音频，但系统没有强制补译；请检查是否有未译完的尾句。','尾句检查');
          lastExport = exportData(s);
          void cleanupCapture(s).finally(() => {if(current(s)){setState('done');updateControls();}});
          if (ws.readyState === WebSocket.OPEN) ws.close(1000,'session finished');
        }
      } catch (error) {reject(error); void failSession(s,error.message);}
    };
    ws.onerror = () => {reject(new Error('WebSocket 连接失败。请检查服务器、反向代理的 WebSocket 配置或访问令牌。'));};
    ws.onclose = event => {
      clearTimeout(startupTimer);
      if (!s.ready) reject(new Error(`连接在会话准备前关闭（${event.code}）。请检查服务器日志或访问令牌。`));
      if (!s.terminal && current(s)) void failSession(s,`服务器连接意外关闭（${event.code}）。已显示译文保留，未处理音频不会静默续接；请重新开始。`,'连接中断',false);
    };
  });
}
async function failSession(s, message, title = '会话已停止', notifyServer = true) {
  if (!current(s) || s.terminal) return;
  s.terminal = true; s.failure = String(message); s.endedAt = new Date().toISOString(); clearTimeout(s.startupTimer);
  showError(message,title);
  if (notifyServer && s.ws?.readyState === WebSocket.OPEN) s.ws.send(JSON.stringify({type:'cancel'}));
  await cleanupCapture(s);
  if (s.ws && s.ws.readyState < WebSocket.CLOSING) s.ws.close(1000,'client stopped');
  lastExport = exportData(s); setState('error');
}
async function startSession() {
  if (busy() || !config || config.status !== 'ready') return;
  if (config.auth_required && !ui.authToken.value.trim()) {ui.settings.open = true;showError('请输入服务器访问令牌后再开始。');ui.authToken.focus();return;}
  const direction = `${ui.sourceLang.value}-${ui.targetLang.value}`;
  if (!(config.directions || ['en-zh','zh-en']).includes(direction)) {showError('当前服务器模型没有启用该翻译方向，请检查服务端配置。');return;}
  if (mode === 'file' && (!selectedFile || ui.media.error || ui.media.readyState < 1)) {showError('请先选择浏览器可以播放的音视频文件，并等待媒体加载完成。');return;}
  if (mode === 'file' && (ui.media.ended || (Number.isFinite(ui.media.duration) && ui.media.currentTime >= ui.media.duration - .05))) ui.media.currentTime = 0;
  if (session) await cleanupCapture(session);
  clearTranscript(); ui.errorBox.hidden = true; ui.settings.open = false;
  const s = {number:++sessionNumber,sourceLang:ui.sourceLang.value,targetLang:ui.targetLang.value,sourceKind:mode,sourceName:mode === 'file' ? selectedFile.name : 'microphone',mediaOffset:mode === 'file' ? ui.media.currentTime : 0,chunkSeconds:Number(ui.chunkSeconds.value),adaptive:mode === 'microphone' && ui.adaptive.checked,authToken:ui.authToken.value.trim(),startedAt:performance.now(),startedDate:new Date().toISOString(),sequence:0,sentSamples:0,gateSequence:0,acks:new Map(),terminal:false,ready:false};
  session = s; setState('connecting');
  try {
    await prepareCapture(s); if (!current(s) || s.terminal) return;
    await openSocket(s); if (!current(s) || s.terminal) return;
    s.startedAt = performance.now();
    if (mode === 'microphone') await gate(s,'active');
    if (!current(s) || s.terminal) return;
    setState('live');
    if (mode === 'file') {
      s.progressTimer = setInterval(() => sendFileProgress(s),200);
      ui.footerHint.textContent = `固定 ${s.chunkSeconds} 秒分块 · ${sessionLimitLabel()}`;
      sendFileProgress(s,true);
      await ui.media.play();
    } else ui.footerHint.textContent = `${s.chunkSeconds} 秒基础分块 · ${sessionLimitLabel()}`;
    updateControls();
  } catch (error) {
    const message = error.name === 'NotAllowedError' ? '浏览器拒绝了音频权限或自动播放。请允许麦克风 / 播放权限后重新开始。' : error.name === 'NotFoundError' ? '没有找到可用麦克风，请连接设备后重试。' : error.message;
    await failSession(s,message);
  }
}
async function finishSession() {
  const s = session;
  if (!s || s.terminal || !['connecting','live','paused'].includes(state)) return;
  if (state === 'connecting') {await failSession(s,'连接准备已取消。','已取消');return;}
  setState('draining');
  if (s.sourceKind === 'file') {clearInterval(s.progressTimer);ui.media.pause();}
  try {
    await gate(s,'finish');
    if (s.terminal) return;
    if (s.stream) s.stream.getTracks().forEach(track => track.stop());
    if (s.ws?.readyState !== WebSocket.OPEN) throw new Error('结束时服务器连接已断开，剩余音频无法确认。');
    s.ws.send(JSON.stringify({type:'finish',...(s.sourceKind === 'file' ? {position_s:filePosition(s,true),ended:ui.media.ended} : {})}));
  } catch (error) {await failSession(s,error.message);}
}
async function pauseCapture() {
  const s = session;
  if (!s || s.terminal || mode !== 'file' || state !== 'live') return;
  setState('paused');
  sendFileProgress(s,true);
}
async function resumeCapture() {
  const s = session;
  if (!s || s.terminal || mode !== 'file' || state !== 'paused') return;
  setState('live'); sendFileProgress(s,true);
}
function exportData(s = session) {
  return {schema_version:2,export_scope:'browser_tail_only',complete:false,partial:true,translation_truncated:transcriptTrimmed,turns_truncated:recordsTrimmed,total_received_turns:totalTurns,total_received_translation_chars:translationCharacters,backend:config?.backend,model_label:config?.model_label,session_id:s?.sessionId || null,source_language:s?.sourceLang || ui.sourceLang.value,target_language:s?.targetLang || ui.targetLang.value,source_kind:s?.sourceKind || mode,source_name:s?.sourceName || null,capture_mode:s?.captureMode || null,media_offset_s:s?.mediaOffset || 0,started_at:s?.startedDate || null,ended_at:s?.endedAt || null,sample_rate:16000,sent_audio_s:(s?.sentSamples || 0)/16000,chunk_seconds:s?.chunkSeconds,adaptive:s?.adaptive,translation:currentTranslation,turns:[...turns],final_metrics:{...lastMetrics},done:s?.doneEvent || null,error:s?.failure || null,timing_note:'PARTIAL browser backup, not the complete server result. source_start_s/source_end_s are input audio intervals, not word alignment or semantic latency. Add media_offset_s to compare against file playback time. client_elapsed_s excludes file upload/preparation and includes actual inference/network scheduling. server-file uploads the complete file but releases only the played prefix to the model; natural EOF reconciles small player/decoder duration differences. Browser progress events are not sample-accurate latency measurements.'};
}
function downloadLink(url, kind, data, partial = false) {
  const link = document.createElement('a');
  link.href = url; link.download = `FLEX-${partial ? 'PARTIAL-' : ''}${data.source_language}-${data.target_language}-${(data.started_at || new Date().toISOString()).replace(/[:.]/g,'-')}.${kind}`;
  document.body.append(link);link.click();link.remove();
}
function saveDownload(blob, kind, data, partial = false) {
  const url = URL.createObjectURL(blob);
  downloadLink(url,kind,data,partial);setTimeout(() => URL.revokeObjectURL(url),1000);
}
async function download(kind) {
  if (exportInProgress) return;
  const s = session, data = s ? exportData(s) : lastExport;
  if (!data) return;
  exportInProgress = true; updateControls();
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(),120000);
  try {
    if (!s?.sessionId) throw new Error('本次会话还没有服务端结果编号。');
    const url = `/api/sessions/${encodeURIComponent(s.sessionId)}/export?format=${kind}`;
    const response = await fetch(url,{headers:s.authToken ? {Authorization:`Bearer ${s.authToken}`} : {},cache:'no-store',signal:controller.signal});
    if (!response.ok) {
      let detail = '';
      try {const body = await response.json();detail = typeof body.detail === 'string' ? body.detail : body.message || '';} catch (_) { /* HTTP code remains useful. */ }
      throw new Error(detail || (response.status === 404 ? '服务端会话结果已过期或服务已重新启动。' : `HTTP ${response.status}`));
    }
    if (!s.authToken) {
      // The normal intranet mode needs no bearer header: check availability,
      // then let the browser's download manager stream straight to disk. Do
      // not materialize an hours-long JSON export in JavaScript memory.
      await response.body?.cancel();
      downloadLink(url,kind,data);
    } else {
      // Native downloads cannot attach Authorization. Only this optional
      // authenticated mode temporarily buffers the selected export as a Blob.
      saveDownload(await response.blob(),kind,data);
    }
  } catch (error) {
    const reason = error.name === 'AbortError' ? '下载超时。' : error.message;
    showError(`无法下载完整服务端结果：${reason} 页面只保留最近的译文和记录；可重试下载，或另存明确标注为 PARTIAL 的本地备份。`,'完整导出失败');
    if ((data.translation || data.turns.length) && window.confirm('完整服务端结果暂时无法下载。是否保存本页面保留的局部备份？文件将标注 PARTIAL，可能缺少早期译文或未接收的结果，不能作为完整会话。')) {
      const value = kind === 'txt' ? `【PARTIAL：仅浏览器保留的局部译文，不是完整会话】\n${data.translation}` : JSON.stringify(data,null,2);
      saveDownload(new Blob([value],{type:kind === 'txt' ? 'text/plain;charset=utf-8' : 'application/json;charset=utf-8'}),kind,data,true);
    }
  } finally {clearTimeout(timer);exportInProgress = false;updateControls();}
}
function removeSubtitle() {
  if (subtitleTrack) {subtitleTrack.remove();subtitleTrack = null;}
  if (subtitleUrl) {URL.revokeObjectURL(subtitleUrl);subtitleUrl = null;}
  ui.subtitleInput.value = '';ui.subtitleName.textContent = '';ui.removeSubtitle.hidden = true;
}
function chooseFile(file) {
  if (!file || busy()) return;
  ui.media.pause(); removeSubtitle();
  if (mediaUrl) URL.revokeObjectURL(mediaUrl);
  selectedFile = file; mediaUrl = URL.createObjectURL(file); ui.media.src = mediaUrl;ui.media.load();
  ui.media.hidden = false;ui.dropZone.hidden = true;ui.mediaStage.classList.add('has-file');
  const audioOnly = file.type.startsWith('audio/') || /\.(wav|mp3|m4a|ogg|flac|aac)$/i.test(file.name);
  ui.audioArtwork.hidden = !audioOnly;ui.fileType.textContent = audioOnly ? 'AUDIO' : 'VIDEO';
  ui.fileName.textContent = file.name;ui.fileDetails.textContent = `${formatBytes(file.size)} · 读取中`;
  ui.errorBox.hidden = true;ui.fileInput.value = '';updateControls();
}

ui.dismissError.onclick = () => {ui.errorBox.hidden = true;};
ui.fileMode.onclick = () => setMode('file');ui.micMode.onclick = () => setMode('microphone');
ui.sourceLang.onchange = () => updateDirection('source');ui.targetLang.onchange = () => updateDirection('target');
ui.adaptive.onchange = () => {if(mode === 'microphone' && !busy())microphoneAdaptive = ui.adaptive.checked;};
ui.swapLang.onclick = () => {ui.sourceLang.value = ui.targetLang.value;updateDirection('source');};
ui.chooseFile.onclick = ui.dropZone.onclick = () => {if(!busy())ui.fileInput.click();};
ui.fileInput.onchange = () => chooseFile(ui.fileInput.files[0]);
ui.mediaStage.ondragover = event => {event.preventDefault();if(!busy())ui.mediaStage.classList.add('dragover');};
ui.mediaStage.ondragleave = () => ui.mediaStage.classList.remove('dragover');
ui.mediaStage.ondrop = event => {event.preventDefault();ui.mediaStage.classList.remove('dragover');chooseFile(event.dataTransfer.files[0]);};
ui.chooseSubtitle.onclick = () => ui.subtitleInput.click();
ui.removeSubtitle.onclick = () => {if(!busy())removeSubtitle();};
ui.subtitleInput.onchange = async () => {
  const file = ui.subtitleInput.files[0];if(!file || busy())return;
  if (!/\.vtt$/i.test(file.name)) {showError('请提供 WebVTT (.vtt) 字幕文件。SRT 请先转换为 WebVTT。');return;}
  const text = await file.text();if(!text.replace(/^\uFEFF/,'').startsWith('WEBVTT')) {showError('字幕文件缺少 WEBVTT 标头，请检查格式。');return;}
  removeSubtitle();subtitleUrl = URL.createObjectURL(new Blob([text],{type:'text/vtt'}));
  subtitleTrack = document.createElement('track');subtitleTrack.kind = 'subtitles';subtitleTrack.label = file.name;subtitleTrack.srclang = ui.sourceLang.value;subtitleTrack.src = subtitleUrl;subtitleTrack.default = true;
  ui.media.append(subtitleTrack);subtitleTrack.track.mode = 'showing';ui.subtitleName.textContent = file.name;ui.removeSubtitle.hidden = false;
};
ui.media.addEventListener('loadedmetadata',() => {ui.fileDetails.textContent = `${formatBytes(selectedFile?.size || 0)} · ${Number.isFinite(ui.media.duration) ? time(ui.media.duration,false) : '未知时长'} · 本地播放`;updateControls();});
ui.media.addEventListener('error',() => {if(selectedFile){const message = '浏览器无法解码此音视频。建议转为 MP4 (H.264 + AAC)、WebM 或 WAV 后重试。';if(busy())void failSession(session,message);else showError(message,'媒体播放失败');}});
ui.media.addEventListener('pause',() => {ui.audioArtwork.classList.remove('playing');void pauseCapture();});
ui.media.addEventListener('play',() => {ui.audioArtwork.classList.add('playing');if(state === 'connecting' || state === 'draining')ui.media.pause();else void resumeCapture();});
ui.media.addEventListener('playing',() => {
  const s = session;
  if (s && s.sourceKind === 'file' && !s.terminal && state === 'live') {
    statusLabel('同传进行中','live');
    sendFileProgress(s,true);
  }
});
ui.media.addEventListener('waiting',() => {
  const s = session;
  if (s && s.sourceKind === 'file' && !s.terminal && state === 'live') {
    statusLabel('媒体缓冲中');
    sendFileProgress(s,true);
  }
});
ui.media.addEventListener('ended',() => {if(['live','paused'].includes(state))void finishSession();});
ui.media.addEventListener('seeking',() => {if(['live','paused'].includes(state) || (state === 'connecting' && session?.sourceKind === 'file' && Math.abs(ui.media.currentTime-session.mediaOffset) > 1/16000))void failSession(session,'播放位置发生改变，当前因果上下文已终止。请在新位置重新开始同传，历史译文不会被修改。','跳转后需要新会话');});
ui.media.addEventListener('ratechange',() => {if(['live','paused'].includes(state) && ui.media.playbackRate !== 1)void failSession(session,'同传会话只支持 1× 实时播放。变速会破坏时序比较，本次会话已停止。','播放速率已改变');});
ui.startButton.onclick = () => {void startSession();};
ui.pauseButton.onclick = async () => {
  if(state === 'live')ui.media.pause();
  else if(state === 'paused'){await resumeCapture();if(session && !session.terminal){try{await ui.media.play();}catch(error){void failSession(session,error.message);}}}
};
ui.finishButton.onclick = () => {void finishSession();};
ui.clearButton.onclick = clearTranscript;
ui.exportTxt.onclick = () => download('txt');ui.exportJson.onclick = () => download('json');
ui.transcriptViewport.addEventListener('scroll',() => {const box = ui.transcriptViewport;followOutput = box.scrollHeight-box.scrollTop-box.clientHeight < 55;ui.followButton.hidden = followOutput;});
ui.followButton.onclick = () => {followOutput = true;ui.transcriptViewport.scrollTop = ui.transcriptViewport.scrollHeight;ui.followButton.hidden = true;};
document.addEventListener('click',event => {if(!ui.settings.contains(event.target))ui.settings.open = false;});
window.addEventListener('beforeunload',event => {
  if (busy()) {event.preventDefault();event.returnValue = '';}
});
window.addEventListener('pagehide',() => {
  if(session && !session.terminal){session.terminal = true;clearInterval(session.progressTimer);session.uploadRequest?.abort();releaseUploadedFile(session);if(session.ws?.readyState === WebSocket.OPEN){session.ws.send(JSON.stringify({type:'cancel'}));session.ws.close();}session.stream?.getTracks().forEach(track => track.stop());}
});
setInterval(() => {
  const seconds = mode === 'file' ? ui.media.currentTime : (session?.sentSamples || 0)/16000;
  ui.sourceClock.textContent = time(seconds);
  ui.clockDescription.textContent = mode === 'file'
    ? '播放位置'
    : '已采集音频';
},200);
updateControls();void loadConfig();
