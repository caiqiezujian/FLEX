import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Execute the actual UI controller with a small DOM facade. The capture/file
// transport modules have separate tests; no alternate implementation of the
// tail, counters, export or status logic lives in this test.
const source = readFileSync(new URL('./app.js', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '')
  .replace('updateControls();void loadConfig();', '');

class Element {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.hidden = false; this._text = '';
    this.value = ''; this.checked = true; this.paused = true; this.options = [];
    this.scrollTop = 0; this.clientHeight = 400; this.style = {};
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
  addEventListener() {}
  setAttribute() {}
  contains() { return false; }
  pause() { this.paused = true; }
  click() { this.onDownload?.(this); }
}

function harness() {
  const elements = new Map(), downloads = [], blobs = new Map(), prompts = [];
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
    window:{addEventListener() {},confirm(message) {prompts.push(message);return confirmed;}},
    location:{protocol:'https:',host:'example.test'},
    setInterval() {return 1;},clearInterval() {},setTimeout() {return 1;},clearTimeout() {},
    requestAnimationFrame(callback) {callback();},fetch:(...args) => fetchHandler(...args),
  };
  vm.createContext(sandbox); vm.runInContext(source,sandbox);
  const evaluate = code => vm.runInContext(code,sandbox);
  evaluate(`config={status:'ready',backend:'mock',max_chunk_seconds:3,max_session_seconds:0};
    session={sessionId:'session-one',sourceKind:'microphone',sourceLang:'en',targetLang:'zh',mediaOffset:0,chunkSeconds:1,startedAt:0,startedDate:'2026-10-19T01:02:03.000Z',authToken:'token-test'};
    ui.sourceLang.value='en';ui.targetLang.value='zh';mode='microphone';state='live';`);
  return {evaluate,elements,downloads,prompts,
    setFetch(handler) {fetchHandler=handler;},confirm(value) {confirmed=value;}};
}

test('six hours of one-second outputs keep browser history and DOM bounded', () => {
  const h = harness();
  h.evaluate(`for(let i=0;i<21600;i++) appendTurn({type:'turn',tag:i%7===0?'WAIT':'DIRECT',delta:i%7===0?'':' 安全增量翻译。',source_start_s:i,source_end_s:i+1,service_s:.1,queue_wait_s:0},session);`);
  assert.equal(h.evaluate('totalTurns'),21600);
  assert.equal(h.evaluate('transcriptFragments'),18514);
  assert(h.evaluate('turns.length') <= 200);
  assert(h.evaluate('recentRecordCharacters') <= 120000);
  assert(h.evaluate('visibleFragments.length') <= 500);
  assert(h.evaluate('currentTranslation.length') <= 60000);
  assert(h.elements.get('transcript').children.length <= 500);
  assert(h.elements.get('turnList').children.length <= 200);
  assert.equal(h.elements.get('transcript').textContent,h.evaluate('currentTranslation'));
  assert.equal(h.elements.get('transcriptEmpty').hidden,true);
  assert.equal(h.elements.get('historyNotice').hidden,false);
  assert.equal(h.evaluate('exportData().partial'),true);
  assert.equal(h.evaluate('exportData().turns_truncated'),true);
  assert(h.elements.get('turnCount').textContent.startsWith('21600 轮'));
});

test('oversized single output and Unicode character counters are bounded', () => {
  const h = harness();
  h.evaluate(`appendTurn({tag:'FLEX',delta:'😀'.repeat(70000),source_start_s:0,source_end_s:1,service_s:.1,queue_wait_s:0},session);`);
  assert.equal(h.evaluate('translationCharacters'),70000);
  assert.equal(h.evaluate('currentTranslation.length'),60000);
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
  assert.equal(h.elements.get('sessionStatus').textContent,'等待后续音频');
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

test('exports download complete server snapshot using authentication, not local tail', async () => {
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

test('failed complete export cannot silently download a partial tail', async () => {
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
  assert.equal(data.export_scope,'browser_tail_only');
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
