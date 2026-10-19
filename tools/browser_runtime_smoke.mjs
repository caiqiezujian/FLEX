#!/usr/bin/env node
/** Real Chromium UI checks using synthetic WebSocket events, not model output.
 * Serve the real application with a local mock backend, then run:
 * node tools/browser_runtime_smoke.mjs --base-url http://127.0.0.1:31223
 * The WebSocket is intercepted: no GPU, physical microphone or real speech.
 * Covers full transcript retention/scrolling and WAIT versus runtime health.
 */
import assert from 'node:assert/strict';
import {stat} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';

const {values}=parseArgs({options:{'base-url':{type:'string',default:'http://127.0.0.1:31223'}}});
const base=new URL(values['base-url']);
assert.equal(base.protocol,'http:');
assert(['localhost','127.0.0.1'].includes(base.hostname));
assert(!base.username && !base.password && base.pathname==='/' && !base.search && !base.hash);
const config=await(await fetch(new URL('/api/config',base))).json();
assert.equal(config.backend,'mock','Run only against a local mock app.');
assert.equal(config.status,'ready');
assert.equal(config.auth_required,false);
const location=process.env.FLEX_PLAYWRIGHT_MODULE;
const resolved=location && ((await stat(location)).isDirectory()?createRequire(import.meta.url).resolve(location):location);
const playwright=await import(resolved?pathToFileURL(resolved).href:'playwright');
const chromium=playwright.chromium || playwright.default.chromium;
let browser,wire;
let health={state:'ready',accepting:true,running_seconds:0,queued_requests:0,message:''};
let processing={state:'normal',message:''}, turn=0, chars=0, received=0;
const controls=[], deltas=[], errors=[];
const send=event=>wire.send(JSON.stringify({session_id:'synthetic-ui-session',...event}));
const status=()=>send({type:'status',engine_health:health,processing_health:processing,inference_state:'idle'});
function output(delta,tag='DIRECT'){
  send({type:'turn',turn_id:turn,tag,delta,source_start_s:turn,source_end_s:turn+1,
    service_s:.1,scheduler_wait_s:0,queue_wait_s:0,batch_size:1,
    engine_health:health,processing_health:processing,inference_state:'idle'});
  turn++;chars += [...delta].length;if(delta)deltas.push(delta);
}
try {
  browser=await chromium.launch({headless:true,
    ...(process.env.FLEX_BROWSER_EXECUTABLE?{executablePath:process.env.FLEX_BROWSER_EXECUTABLE}:{}),
    args:['--use-fake-device-for-media-stream','--no-proxy-server']});
  const context=await browser.newContext({permissions:['microphone'],viewport:{width:1366,height:768}});
  const page=await context.newPage();
  page.on('pageerror',error=>errors.push(String(error)));
  await page.route('**/api/config',route=>route.fulfill({json:{...config,
    max_sessions:4,active_sessions:0,available_sessions:4,busy:false,engine_health:health}}));
  await page.routeWebSocket('**/ws',socket=>{
    wire=socket;
    socket.onMessage(message=>{
      if(typeof message!=='string'){received += Math.max(0,(message.byteLength-8)/2);return;}
      const event=JSON.parse(message);controls.push(event.type);
      if(event.type==='start')send({type:'ready',sample_rate:16000,engine_health:health,
        max_sessions:4,active_sessions:1,available_sessions:3,busy:false,results_available:false});
      if(event.type==='finish')send({type:'done',reason:'finished',engine_health:health,
        turn_count:turn,translation_chars:chars,processed_audio_s:received/16000});
    });
  });
  await page.goto(base.href,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.getElementById('serverStatus').classList.contains('ready'));
  await page.locator('#micMode').click();await page.locator('#startButton').click();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='同传进行中');

  for(let i=0;i<750;i++)output(` 片段${String(i).padStart(4,'0')} ${'原样保留😀 '.repeat(20)}\n`,i%2?'FLEX':'DIRECT');
  await page.waitForFunction(()=>document.getElementById('transcript').children.length===750);
  assert(deltas.join('').length>60000);
  assert.equal(await page.locator('#transcript').textContent(),deltas.join(''));
  assert.equal(await page.locator('#transcript > span').first().textContent(),deltas[0]);
  assert(await page.locator('#transcriptViewport').evaluate(el=>el.scrollHeight>el.clientHeight));
  assert((await page.locator('#turnList .turn-row').count())<=200);
  console.log('PASS more than 500 fragments and 60,000 characters remain intact and scrollable.');

  await page.locator('#transcriptViewport').evaluate(el=>{el.scrollTop=0;el.dispatchEvent(new Event('scroll'));});
  await page.waitForFunction(()=>document.getElementById('transcriptViewport').scrollTop<1 && !document.getElementById('followButton').hidden);
  for(let i=0;i<75;i++)output(` 后续片段${i}\n`);
  await page.waitForFunction(()=>document.getElementById('transcript').children.length===825);
  assert((await page.locator('#transcriptViewport').evaluate(el=>el.scrollTop))<1);
  assert.equal(await page.locator('#transcript').textContent(),deltas.join(''));
  await page.locator('#followButton').click();
  await page.waitForFunction(()=>{const el=document.getElementById('transcriptViewport');return el.scrollHeight-el.clientHeight-el.scrollTop<2;});
  console.log('PASS manually reading early text is not pulled down; following latest still works.');

  for(let i=0;i<700;i++)output('','WAIT');
  await page.waitForFunction(()=>document.getElementById('turnCount').textContent.startsWith('1525 轮'));
  assert.equal(await page.locator('#sessionStatus').textContent(),'模型本轮暂不输出');
  assert.equal(await page.locator('#errorBox').isVisible(),false);
  assert.equal(await page.locator('#transcript').textContent(),deltas.join(''));
  console.log('PASS long valid WAIT sequences neither erase text nor become a fault.');

  health={state:'slow',accepting:true,running_seconds:8,queued_requests:2,message:'合成慢请求'};status();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='模型处理较慢，仍在等待结果');
  assert.equal(await page.locator('#finishButton').isEnabled(),true);
  health={state:'ready',accepting:true,running_seconds:0,queued_requests:0,message:''};
  processing={state:'slow',message:'合成持续积压'};status();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='处理较慢，音频正在积压');
  health={state:'stalled',accepting:false,running_seconds:61,queued_requests:3,message:'合成长时间无响应'};status();
  await page.waitForFunction(()=>document.getElementById('serverStatus').textContent==='模型长时间无响应 · 暂停新会话');
  health={state:'failed',accepting:false,running_seconds:0,queued_requests:0,message:'合成故障'};status();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='模型故障，等待服务器处理');
  assert(!controls.includes('cancel'),'Health warnings must not silently cancel an active session.');
  assert.equal(await page.locator('#errorBox').isVisible(),false);
  console.log('PASS slow, sustained backlog, stalled and failed have distinct existing status labels without implicit cancellation.');

  await page.locator('#finishButton').click();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='同传已结束');
  assert.equal(await page.locator('#startButton').isDisabled(),true,'Failed engine must block a new session.');
  health={state:'ready',accepting:true,running_seconds:0,queued_requests:0,message:''};
  await page.waitForFunction(()=>!document.getElementById('startButton').disabled,undefined,{timeout:10000});
  assert.equal(await page.locator('#transcript').textContent(),deltas.join(''));
  assert.equal(errors.length,0,errors.join('\n'));
  console.log('PASS recovery re-enables new sessions and retains all previous text; no browser exceptions.');
  console.log('5 runtime UI checks passed (synthetic protocol events, no model accuracy/throughput claim).');
} finally {
  await browser?.close();
}
