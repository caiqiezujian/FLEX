import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Execute the actual UI controller with a small DOM facade. The capture/file
// transport modules have separate tests; no alternate implementation of the
// transcript, counters, export or status logic lives in this test.
const source = readFileSync(new URL('./app.js', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '')
  .replace('updateControls();void loadConfig();', '');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.hidden = false; this._text = '';
    this.value = ''; this.checked = true; this.paused = true; this.options = [];
    this.scrollTop = 0; this.clientHeight = 400; this.style = {};
    this.listeners = new Map();
    this.classList = {toggle() {}, add() {}, remove() {}};
  }
  get firstElementChild() { return this.children[0] || null; }
  get lastChild() { return this.children.at(-1) || this; }
  get scrollHeight() { return this.children.length * 20; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  set textContent(text) { this._text = String(text); this.children = []; }
  append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
  replaceChildren(...children) { this.children = []; this._text = ''; this.append(...children); }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  querySelector(selector) {
    if (selector === '.no-turns') return this.children.find(child => child.className === 'no-turns') || null;
    return this.specialChild ||= new Element();
  }
  addEventListener(name,callback) {this.listeners.set(name,callback);}
  setAttribute() {}
  contains() { return false; }
  pause() { this.paused = true; }
  click() { this.onDownload?.(this); }
}

function harness() {
  const elements = new Map(), downloads = [], blobs = new Map(), prompts = [], listeners = new Map(), timers = new Map();
  let nextTimer = 0, deferFrames = false;
  const frames = [];
  let fetchHandler = () => { throw new Error('Unexpected fetch'); }, confirmed = false;
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement(tag) {
      const element = new Element(tag);
      if (tag === 'a') element.onDownload = link => downloads.push({name:link.download, href:link.href, blob:blobs.get(link.href)});
      return element;
    },
    createTextNode(text) { const node = new Element('#text'); node.textContent = text; return node; },
    addEventListener() {}, body:new Element('body'),
  };
  const sandbox = {
    document, performance:{now:() => 1000}, Date, Blob, AbortController,
    URL:{createObjectURL(blob) { const url = `blob:${blobs.size}`; blobs.set(url,blob); return url; },revokeObjectURL() {}},
    window:{addEventListener(name,callback) {listeners.set(name,callback);},confirm(message) {prompts.push(message);return confirmed;}},
    location:{protocol:'https:',host:'example.test'},
    setInterval() {return 1;},clearInterval() {},setTimeout(callback,ms) {const id=++nextTimer;timers.set(id,{callback,ms});return id;},clearTimeout(id) {timers.delete(id);},
    requestAnimationFrame(callback) {if(deferFrames)frames.push(callback);else callback();},fetch:(...args) => fetchHandler(...args),
  };
  vm.createContext(sandbox); vm.runInContext(source,sandbox);
  const evaluate = code => vm.runInContext(code,sandbox);
  evaluate(`config={status:'ready',backend:'mock',max_chunk_seconds:3,max_session_seconds:0};
    session={sessionId:'session-one',sourceKind:'microphone',sourceLang:'en',targetLang:'zh',mediaOffset:0,chunkSeconds:1,startedAt:0,startedDate:'2026-10-19T01:02:03.000Z',authToken:'token-test'};
    ui.sourceLang.value='en';ui.targetLang.value='zh';mode='microphone';state='live';`);
  return {evaluate,elements,downloads,prompts,listeners,timers,
    setFetch(handler) {fetchHandler=handler;},confirm(value) {confirmed=value;},
    deferAnimationFrames(value) {deferFrames=value;},flushAnimationFrames() {for(const callback of frames.splice(0))callback();}};
}

test('six hours preserve every transcript fragment while auxiliary research records stay bounded', () => {
  const h = harness();
  h.evaluate(`for(let i=0;i<21600;i++) appendTurn({type:'turn',tag:i%7===0?'WAIT':'DIRECT',delta:i%7===0?'':' 安全增量翻译。',source_start_s:i,source_end_s:i+1,service_s:.1,queue_wait_s:0},session);`);
  assert.equal(h.evaluate('totalTurns'),21600);
  assert.equal(h.evaluate('transcriptFragments'),18514);
  assert(h.evaluate('turns.length') <= 200);
  assert(h.evaluate('recentRecordCharacters') <= 120000);
  assert.equal(h.evaluate('currentTranslation'), ' 安全增量翻译。'.repeat(18514));
  assert(h.evaluate('currentTranslation.length') > 60000);
  assert.equal(h.elements.get('transcript').children.length,18514);
  assert.equal(h.elements.get('transcript').firstElementChild.textContent,' 安全增量翻译。');
  assert(h.elements.get('turnList').children.length <= 200);
  assert.equal(h.elements.get('transcript').textContent,h.evaluate('currentTranslation'));
  assert.equal(h.elements.get('transcriptEmpty').hidden,true);
  assert.equal(h.evaluate('exportData().partial'),true);
  assert.equal(h.evaluate('exportData().translation_truncated'),false);
  assert.equal(h.evaluate('exportData().turns_truncated'),true);
  assert(h.elements.get('turnCount').textContent.startsWith('21600 轮'));
});

test('oversized single output is displayed intact with exact Unicode and whitespace', () => {
  const h = harness();
  const expected='  BEGIN\n'+'😀'.repeat(70000)+'\nEND  ';
  h.evaluate(`appendTurn({tag:'FLEX',delta:${JSON.stringify(expected)},source_start_s:0,source_end_s:1,service_s:.1,queue_wait_s:0},session);`);
  assert.equal(h.evaluate('translationCharacters'),[...expected].length);
  assert.equal(h.evaluate('currentTranslation'),expected);
  assert.equal(h.elements.get('transcript').textContent,expected);
  assert.equal(h.evaluate('turns.length'),0);
  assert.equal(h.evaluate('recordsTrimmed'),true);
  assert(h.elements.get('turnList').textContent.length < 4200);
});

test('legacy full done transcript is never retained as metrics or done state', () => {
  const h = harness();
  h.evaluate(`const legacyDone={type:'done',translation:'x'.repeat(1000000),turn_count:123,translation_chars:1000000,processed_audio_s:999};
    renderMetrics(legacyDone);session.doneEvent=compactDone(legacyDone);`);
  assert.equal(h.evaluate('lastMetrics.translation'),undefined);
  assert.equal(h.evaluate('session.doneEvent.translation'),undefined);
  assert.equal(h.evaluate('session.doneEvent.translation_chars'),1000000);
});

test('WAIT, computational backlog and draining remain distinct', () => {
  const h = harness();
  h.evaluate(`appendTurn({tag:'WAIT',delta:'',source_start_s:0,source_end_s:1,service_s:.1,queue_wait_s:0},session);`);
  assert.equal(h.elements.get('sessionStatus').textContent,'模型本轮暂不输出');
  h.evaluate('renderMetrics({pending_audio_s:5});');
  assert.equal(h.elements.get('sessionStatus').textContent,'正在处理积压音频');
  h.evaluate("setState('draining');renderMetrics({pending_audio_s:5});");
  assert.equal(h.elements.get('sessionStatus').textContent,'处理剩余音频');
});

test('context rollover records remain bounded and do not count as audio turns', () => {
  const h = harness();
  h.evaluate('for(let i=1;i<=1500;i++)appendContextWindow({context:{rollovers:i,retained_turns:32,retained_audio_seconds:48}});');
  assert.equal(h.evaluate('totalTurns'),0);
  assert.equal(h.evaluate('windowRollovers'),1500);
  assert.equal(h.elements.get('turnList').children.length,200);
  assert.match(h.elements.get('turnCount').textContent,/窗口滚动 1500 次/);
});

test('exports download complete server snapshot using authentication, not browser backup', async () => {
  const h = harness(); let requested;
  h.setFetch(async (url,options) => {requested={url,options};return {ok:true,blob:async () => new Blob(['full server transcript'])};});
  h.evaluate("currentTranslation='tail';");
  await h.evaluate("download('txt')");
  assert.equal(requested.url,'/api/sessions/session-one/export?format=txt');
  assert.equal(requested.options.headers.Authorization,'Bearer token-test');
  assert.equal(await h.downloads[0].blob.text(),'full server transcript');
  assert(!h.downloads[0].name.includes('PARTIAL'));
  assert.equal(h.prompts.length,0);
  assert.equal(h.evaluate('exportInProgress'),false);
});

test('failed complete export labels browser backup accurately without claiming complete server records', async () => {
  const h = harness();
  h.evaluate("currentTranslation='tail';");
  h.setFetch(async () => ({ok:false,status:404,json:async () => ({detail:'expired'})}));
  await h.evaluate("download('txt')");
  assert.equal(h.downloads.length,0);
  assert.equal(h.prompts.length,1);
  assert.equal(h.elements.get('errorTitle').textContent,'完整导出失败');
  h.confirm(true);
  await h.evaluate("download('json')");
  assert(h.downloads[0].name.includes('PARTIAL'));
  const data = JSON.parse(await h.downloads[0].blob.text());
  assert.equal(data.partial,true);assert.equal(data.complete,false);
  assert.equal(data.export_scope,'browser_received_translation_and_recent_turns');
  assert.equal(data.translation_truncated,false);
  assert.equal(data.translation,'tail');
});

test('normal no-token export uses native download rather than a full in-memory Blob', async () => {
  const h = harness(); let cancelled = false;
  h.evaluate("session.authToken='';");
  h.setFetch(async () => ({ok:true,body:{cancel:async () => {cancelled=true;}},blob() {throw new Error('must not buffer full result');}}));
  await h.evaluate("download('json')");
  assert.equal(cancelled,true);
  assert.equal(h.downloads[0].href,'/api/sessions/session-one/export?format=json');
  assert.equal(h.downloads[0].blob,undefined);
  assert.equal(h.prompts.length,0);
});

test('clearing results removes old export session and unlimited config is not displayed as 180 seconds', () => {
  const h = harness();
  assert.equal(h.evaluate('sessionLimitLabel()'),'持续会话');
  h.evaluate("state='done';totalTurns=10;clearTranscript();");
  assert.equal(h.evaluate('session'),null);
  assert.equal(h.evaluate('totalTurns'),0);
  assert.equal(h.elements.get('exportJson').disabled,true);
});

test('capacity display allows clients one through four and blocks only a full service', () => {
  const h = harness();
  h.evaluate("state='idle';updateCapacity({max_sessions:4,active_sessions:0,available_sessions:4,busy:false});");
  assert.equal(h.elements.get('serverStatus').textContent,'使用中 0 / 4');
  assert.equal(h.elements.get('startButton').disabled,false);
  h.evaluate('updateCapacity({max_sessions:4,active_sessions:3,available_sessions:1,busy:false});');
  assert.equal(h.elements.get('serverStatus').textContent,'使用中 3 / 4');
  assert.equal(h.elements.get('startButton').disabled,false);
  h.evaluate('updateCapacity({max_sessions:4,active_sessions:4,available_sessions:0,busy:true});');
  assert.equal(h.elements.get('serverStatus').textContent,'使用中 4 / 4 · 已满');
  assert.equal(h.elements.get('startButton').disabled,true);
  assert.equal(h.elements.get('startButton').textContent,'当前已满');
  h.evaluate('updateCapacity({max_sessions:4,active_sessions:2,available_sessions:2,busy:false});');
  assert.equal(h.elements.get('startButton').disabled,false);
});

test('full server does not interrupt the current session and exposes its own scheduling state', () => {
  const h = harness();
  h.evaluate("updateCapacity({max_sessions:4,active_sessions:4,available_sessions:0,busy:true});renderMetrics({inference_state:'running'});");
  assert.equal(h.evaluate('state'),'live');
  assert.equal(h.elements.get('sessionStatus').textContent,'正在生成译文');
  assert.equal(h.elements.get('finishButton').disabled,false);
  assert.equal(h.elements.get('startButton').textContent,'开始同传');
  h.evaluate("renderMetrics({inference_state:'queued'});");
  assert.equal(h.elements.get('sessionStatus').textContent,'等待推理调度');
  h.evaluate("renderMetrics({inference_state:'idle',pending_audio_s:5});");
  assert.equal(h.elements.get('sessionStatus').textContent,'正在处理积压音频');
  h.evaluate("renderMetrics({inference_state:'idle',pending_audio_s:0});session.lastTag='WAIT';updateControls();");
  assert.equal(h.elements.get('sessionStatus').textContent,'模型本轮暂不输出');
});

test('turn latency keeps model time, scheduler wait, input wait and batch size separate', () => {
  const h = harness();
  h.evaluate("appendTurn({tag:'DIRECT',delta:'hello',source_start_s:0,source_end_s:1,service_s:.25,scheduler_wait_s:1.5,queue_wait_s:2.75,batch_size:4},session);");
  const row = h.elements.get('turnList').children.at(-1);
  assert.match(row.textContent,/推理 0\.25s · 调度等待 1\.50s · 提交前等待 2\.75s · 批大小 4/);
  assert.equal(h.evaluate('lastMetrics.service_s'),.25);
  assert.equal(h.evaluate('lastMetrics.scheduler_wait_s'),1.5);
  assert.equal(h.evaluate('lastMetrics.batch_size'),4);
});

test('ready capacity refresh repeats without resetting chosen chunk size or active footer', async () => {
  const h = harness();
  h.evaluate("ui.chunkSeconds.options=[{value:'1'},{value:'2'},{value:'3'}];ui.chunkSeconds.value='3';ui.footerHint.textContent='3 秒基础分块';");
  h.setFetch(async () => ({ok:true,json:async () => ({status:'ready',backend:'mock',chunk_seconds:1,max_chunk_seconds:3,max_sessions:4,active_sessions:2,available_sessions:2,busy:false})}));
  await h.evaluate('loadConfig()');
  assert.equal(h.elements.get('chunkSeconds').value,'3');
  assert.equal(h.elements.get('footerHint').textContent,'3 秒基础分块');
  assert.equal(h.elements.get('serverStatus').textContent,'使用中 2 / 4');
  assert.equal(h.timers.size,1);
  assert.equal([...h.timers.values()][0].ms,5000);
  h.evaluate("state='idle';");
  await h.evaluate('loadConfig()');
  assert.equal(h.elements.get('chunkSeconds').value,'3');
  assert.match(h.elements.get('footerHint').textContent,/最多 4 路同传/);
  assert.equal(h.timers.size,1);
});

test('transient config polling failure preserves a live websocket session and retries', async () => {
  const h = harness();
  h.evaluate("ui.errorBox.hidden=true;updateCapacity({max_sessions:4,active_sessions:2,busy:false});");
  h.setFetch(async () => {throw new Error('temporary HTTP outage');});
  await h.evaluate('loadConfig()');
  assert.equal(h.evaluate('state'),'live');
  assert.equal(h.evaluate('config.status'),'ready');
  assert.equal(h.elements.get('errorBox').hidden,true);
  assert.equal(h.elements.get('serverStatus').textContent,'使用中 2 / 4 · 状态暂未刷新');
  assert.equal(h.elements.get('finishButton').disabled,false);
  assert.equal(h.timers.size,1);
});

test('pagehide cancels capacity polling and a restored page resumes it', async () => {
  const h = harness();
  h.evaluate('session.terminal=true;');
  h.setFetch(async () => ({ok:true,json:async () => ({status:'ready',backend:'mock',max_sessions:4,active_sessions:0})}));
  await h.evaluate('loadConfig()');
  assert.equal(h.timers.size,1);
  h.listeners.get('pagehide')();
  assert.equal(h.timers.size,0);
  assert.equal(h.evaluate('pageHidden'),true);
  h.listeners.get('pageshow')({persisted:true});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.evaluate('pageHidden'),false);
  assert.equal(h.timers.size,1);
});

test('pagehide aborts an in-flight config request without adding a session error or another poll', async () => {
  const h = harness(); let signal;
  h.evaluate('session.terminal=true;ui.errorBox.hidden=true;');
  h.setFetch((_url,options)=>new Promise((_resolve,reject)=>{
    signal=options.signal;signal.addEventListener('abort',()=>reject(new Error('aborted')));
  }));
  const pending=h.evaluate('loadConfig()');
  h.listeners.get('pagehide')();
  await pending;
  assert.equal(signal.aborted,true);
  assert.equal(h.elements.get('errorBox').hidden,true);
  assert.equal(h.timers.size,0);
});

test('manual history scrolling is preserved, including when a follow callback was already queued', () => {
  const h=harness(), viewport=h.elements.get('transcriptViewport');
  Object.defineProperty(viewport,'scrollHeight',{get:()=>2000});
  h.deferAnimationFrames(true);
  h.evaluate("appendTurn({tag:'DIRECT',delta:'第一段 ',source_start_s:0,source_end_s:1,service_s:.1},session);");
  viewport.scrollTop=120;
  viewport.listeners.get('scroll')();
  assert.equal(h.evaluate('followOutput'),false);
  assert.equal(h.elements.get('followButton').hidden,false);
  h.flushAnimationFrames();
  assert.equal(viewport.scrollTop,120,'An already queued callback must not override manual scrolling.');
  h.evaluate("for(let i=0;i<600;i++)appendTurn({tag:'FLEX',delta:' 后续内容\\n',source_start_s:i+1,source_end_s:i+2,service_s:.1},session);");
  h.flushAnimationFrames();
  assert.equal(viewport.scrollTop,120);
  assert.equal(h.elements.get('transcript').children.length,601);
  assert.equal(h.elements.get('transcript').firstElementChild.textContent,'第一段 ');
  h.elements.get('followButton').onclick();
  assert.equal(h.evaluate('followOutput'),true);
  assert.equal(viewport.scrollTop,2000);
  assert.equal(h.elements.get('followButton').hidden,true);
});

test('a long stream of valid WAIT rounds remains a normal no-output decision, not an engine failure', () => {
  const h=harness();
  h.evaluate("ui.errorBox.hidden=true;updateCapacity({engine_health:{state:'ready',accepting:true,running_seconds:0,queued_requests:0,message:''}});for(let i=0;i<5000;i++)appendTurn({tag:'WAIT',delta:'',source_start_s:i,source_end_s:i+1,service_s:.2,processing_health:{state:'normal',message:''}},session);");
  assert.equal(h.evaluate('state'),'live');
  assert.equal(h.evaluate('totalTurns'),5000);
  assert.equal(h.evaluate('transcriptFragments'),0);
  assert.equal(h.elements.get('transcript').textContent,'');
  assert.equal(h.elements.get('sessionStatus').textContent,'模型本轮暂不输出');
  assert.equal(h.elements.get('errorBox').hidden,true);
  assert.equal(h.evaluate('config.engine_health.state'),'ready');
});

test('slow health is visible but does not terminate the live session or reject allowed new starts', () => {
  const h=harness();
  h.evaluate("ui.errorBox.hidden=true;renderMetrics({engine_health:{state:'slow',accepting:true,running_seconds:9,queued_requests:2,message:'模型仍在运行'},inference_state:'running',processing_health:{state:'normal',message:''}});");
  assert.equal(h.evaluate('state'),'live');
  assert.equal(h.evaluate('session.terminal'),undefined);
  assert.equal(h.elements.get('sessionStatus').textContent,'模型处理较慢，仍在等待结果');
  assert.match(h.elements.get('serverStatus').textContent,/处理较慢/);
  assert.equal(h.elements.get('finishButton').disabled,false);
  assert.equal(h.elements.get('errorBox').hidden,true);
  h.evaluate("renderMetrics({engine_health:{state:'ready',accepting:true},processing_health:{state:'slow',message:'持续积压'},pending_audio_s:10});");
  assert.equal(h.elements.get('sessionStatus').textContent,'处理较慢，音频正在积压');
  h.evaluate("state='idle';updateCapacity({engine_health:{state:'slow',accepting:true}});");
  assert.equal(h.elements.get('startButton').disabled,false);
});

test('stalled and failed engine health override ready configuration without cancelling existing sessions', () => {
  const h=harness();
  h.evaluate("ui.errorBox.hidden=true;updateCapacity({engine_health:{state:'stalled',accepting:false,running_seconds:61,queued_requests:3,message:'等待运行中的请求返回'}});");
  assert.match(h.elements.get('serverStatus').textContent,/模型长时间无响应/);
  assert.equal(h.elements.get('serverStatus').className,'server-status error');
  assert.equal(h.elements.get('sessionStatus').textContent,'模型长时间无响应，等待恢复');
  assert.equal(h.evaluate('state'),'live');
  assert.equal(h.evaluate('session.terminal'),undefined);
  assert.equal(h.elements.get('finishButton').disabled,false);
  assert.equal(h.elements.get('errorBox').hidden,true);
  h.evaluate("updateCapacity({engine_health:{state:'failed',accepting:false,message:'GPU运行失败'}});");
  assert.equal(h.elements.get('serverStatus').textContent,'模型故障 · 暂停新会话');
  assert.equal(h.elements.get('sessionStatus').textContent,'模型故障，等待服务器处理');
  h.evaluate("state='idle';updateControls();");
  assert.equal(h.elements.get('startButton').disabled,true);
  assert.equal(h.elements.get('startButton').textContent,'模型暂不可用');
  // State alone is sufficient to block a contradictory accepting=true field.
  h.evaluate("updateCapacity({engine_health:{state:'stalled',accepting:true}});");
  assert.equal(h.elements.get('startButton').disabled,true);
  h.evaluate("updateCapacity({engine_health:{state:'ready',accepting:true,running_seconds:0,queued_requests:0,message:''}});");
  assert.equal(h.elements.get('startButton').disabled,false);
  assert.equal(h.elements.get('serverStatus').textContent,'服务已就绪');
});

test('a successful config refresh surfaces model faults even when startup status stays ready', async () => {
  const h=harness();
  h.evaluate("state='idle';");
  h.setFetch(async()=>({ok:true,json:async()=>({status:'ready',backend:'mock',engine_health:{state:'failed',accepting:false,message:'故障'}})}));
  await h.evaluate('loadConfig()');
  assert.equal(h.elements.get('startButton').disabled,true);
  assert.equal(h.elements.get('serverStatus').textContent,'模型故障 · 暂停新会话');
  h.setFetch(async()=>({ok:true,json:async()=>({status:'ready',backend:'mock',engine_health:{state:'ready',accepting:true,message:''}})}));
  await h.evaluate('loadConfig()');
  assert.equal(h.elements.get('startButton').disabled,false);
});

test('draining health recovers without losing transcript or conflating WAIT and slowdown', () => {
  const h=harness();
  h.evaluate("currentTranslation='已有译文';state='draining';renderMetrics({engine_health:{state:'slow',accepting:true},processing_health:{state:'slow',message:'积压'}});");
  assert.equal(h.elements.get('sessionStatus').textContent,'处理剩余音频（较慢）');
  h.evaluate("updateCapacity({engine_health:{state:'stalled',accepting:false}});");
  assert.equal(h.elements.get('sessionStatus').textContent,'收尾等待模型恢复');
  h.evaluate("renderMetrics({engine_health:{state:'ready',accepting:true},processing_health:{state:'normal',message:''}});");
  assert.equal(h.elements.get('sessionStatus').textContent,'处理剩余音频');
  assert.equal(h.evaluate('currentTranslation'),'已有译文');
});

test('legacy backends without health fields retain normal start and status behavior', () => {
  const h=harness();
  h.evaluate("state='idle';updateCapacity({max_sessions:4,active_sessions:0,busy:false});");
  assert.equal(h.elements.get('startButton').disabled,false);
  assert.equal(h.elements.get('serverStatus').textContent,'使用中 0 / 4');
  h.evaluate("state='live';renderMetrics({inference_state:'running'});");
  assert.equal(h.elements.get('sessionStatus').textContent,'正在生成译文');
  h.evaluate("renderMetrics({inference_state:'queued'});");
  assert.equal(h.elements.get('sessionStatus').textContent,'等待推理调度');
});
