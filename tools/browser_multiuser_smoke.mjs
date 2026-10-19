#!/usr/bin/env node
/**
 * Four independent Chromium clients against one local CPU/mock service.
 * python run.py --backend mock --host 127.0.0.1 --port 31223 --no-log
 * node tools/browser_multiuser_smoke.mjs --base-url http://127.0.0.1:31223
 *
 * Real file upload/playback/WebSockets, admission at four, rejection at five,
 * slot reuse, disconnect isolation and independent disk exports are exercised.
 * No GPU, physical microphone, external service or user files are touched.
 * This proves protocol isolation, not four-stream real-model throughput.
 */
import assert from 'node:assert/strict';
import { mkdtemp, stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const {values} = parseArgs({options:{
  'base-url':{type:'string',default:'http://127.0.0.1:31223'},
  'timeout-ms':{type:'string',default:'45000'},
  screenshots:{type:'boolean',default:false},
}});
const base = new URL(values['base-url']), timeoutMs = Number(values['timeout-ms']);
assert.equal(base.protocol,'http:');
assert(['127.0.0.1','localhost'].includes(base.hostname),'Only a local mock endpoint is allowed.');
assert(!base.username && !base.password && base.pathname === '/' && !base.search && !base.hash);
assert(Number.isFinite(timeoutMs) && timeoutMs >= 5000);
const getConfig = async () => {
  const response = await fetch(new URL('/api/config',base));
  assert.equal(response.status,200);return response.json();
};
const initial = await getConfig();
assert.equal(initial.backend,'mock','Never run this test against a real model.');
assert.equal(initial.status,'ready');
assert.equal(initial.auth_required,false,'Start the isolated mock without a token.');
assert.equal(initial.max_sessions,4,'Use the default four-session capacity.');
assert.equal(initial.active_sessions,0,'Do not interrupt other test sessions.');
const modulePath = process.env.FLEX_PLAYWRIGHT_MODULE;
const resolved = modulePath && ((await stat(modulePath)).isDirectory() ? createRequire(import.meta.url).resolve(modulePath) : modulePath);
const playwright = await import(resolved ? pathToFileURL(resolved).href : 'playwright');
const chromium = playwright.chromium || playwright.default.chromium;
const rate = 16000, samples = rate * 30;
const wav = Buffer.alloc(44+samples*2);
wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);
wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);
wav.writeUInt32LE(rate,24);wav.writeUInt32LE(rate*2,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);
wav.write('data',36);wav.writeUInt32LE(samples*2,40);
for(let i=0;i<samples;i++)wav.writeInt16LE(Math.round(6000*Math.sin(2*Math.PI*440*i/rate)),44+i*2);
const clients = [];
const screenshots=values.screenshots ? await mkdtemp(path.join(tmpdir(),'flex-multiuser-preview-')) : null;
let browser;
async function until(predicate,description) {
  const deadline=Date.now()+timeoutMs;
  while(Date.now()<deadline){if(await predicate())return;await delay(50);}
  throw new Error(`Timed out: ${description}`);
}
function events(client,type) {return client.connections.flatMap(c=>c.events).filter(event=>event.type===type);}
async function openClient(index) {
  const context=await browser.newContext(), page=await context.newPage();
  const client={index,context,page,connections:[],errors:[],name:`independent-${index}.wav`,source:index%2?'zh':'en'};
  clients.push(client);
  page.on('pageerror',error=>client.errors.push(String(error)));
  page.on('websocket',socket=>{
    const connection={events:[]};client.connections.push(connection);
    socket.on('framereceived',({payload})=>connection.events.push(JSON.parse(String(payload))));
  });
  await page.goto(base.href,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.getElementById('serverStatus').textContent.includes('使用中'));
  await page.locator('#sourceLang').selectOption(client.source);
  await page.locator('#fileInput').setInputFiles({name:client.name,mimeType:'audio/wav',buffer:wav});
  await page.waitForFunction(()=>document.getElementById('media').readyState>=1);
  return client;
}
async function start(client) {
  await client.page.locator('#startButton').click();
  await until(()=>events(client,'ready').length===1,`client ${client.index} ready`);
}
async function finish(client) {
  await client.page.locator('#finishButton').click();
  await until(()=>events(client,'done').some(event=>event.reason==='finished'),`client ${client.index} done`);
}
try {
  browser=await chromium.launch({headless:true,
    ...(process.env.FLEX_BROWSER_EXECUTABLE?{executablePath:process.env.FLEX_BROWSER_EXECUTABLE}:{}),
    args:['--autoplay-policy=no-user-gesture-required','--disable-background-timer-throttling','--disable-renderer-backgrounding','--disable-backgrounding-occluded-windows']});
  const all=await Promise.all([0,1,2,3,4].map(openClient));
  await Promise.all(all.slice(0,4).map(start));
  await until(async()=>((await getConfig()).active_sessions===4),'four admitted independent sessions');
  await until(()=>all.slice(0,4).every(client=>events(client,'turn').length>=2),'each client gets translation turns');
  assert.equal(new Set(all.slice(0,4).map(client=>events(client,'ready')[0].session_id)).size,4);
  console.log('PASS four concurrent file-playback clients receive independent session IDs and turns.');

  const extra=all[4];
  await until(async()=>(await extra.page.locator('#serverStatus').textContent()).includes('4 / 4'),'idle fifth browser sees full capacity');
  assert.equal(await extra.page.locator('#startButton').isDisabled(),true);
  const rejected=await extra.page.evaluate(()=>new Promise((resolve,reject)=>{
    const ws=new WebSocket(`ws://${location.host}/ws`),timer=setTimeout(()=>{ws.close();reject(new Error('admission probe timed out'));},10000);
    ws.onopen=()=>ws.send(JSON.stringify({type:'start',source_lang:'en',target_lang:'zh',source_kind:'microphone',chunk_seconds:1,adaptive:false}));
    ws.onmessage=({data})=>{const event=JSON.parse(data);if(event.type==='error'){clearTimeout(timer);ws.close();resolve(event);}};
    ws.onerror=()=>{clearTimeout(timer);reject(new Error('admission probe failed'));};
  }));
  assert.equal(rejected.code,'busy');
  if(screenshots){
    await all[0].page.screenshot({path:path.join(screenshots,'active-client-four-users.png'),fullPage:true});
    await extra.page.screenshot({path:path.join(screenshots,'fifth-client-full.png'),fullPage:true});
  }
  console.log('PASS fifth client sees a full service and backend rejects excess admission.');

  await finish(all[0]);
  await until(async()=>(await getConfig()).active_sessions===3,'completed slot released');
  await until(async()=>!(await extra.page.locator('#startButton').isDisabled()),'capacity polling makes freed slot available');
  await start(extra);
  await until(()=>events(extra,'turn').length>=2,'replacement session receives turns');
  assert.equal((await getConfig()).active_sessions,4);
  console.log('PASS a released slot can be reused without restarting the model or other clients.');

  const surviving=[all[2],all[3],extra], oldCounts=surviving.map(c=>events(c,'turn').length);
  await all[1].context.close();
  await until(async()=>(await getConfig()).active_sessions===3,'disconnected session slot released');
  await until(()=>surviving.every((client,i)=>events(client,'turn').length>oldCounts[i]),'other clients continue after one disconnects');
  await Promise.all(surviving.map(finish));
  await until(async()=>(await getConfig()).active_sessions===0,'all session slots released');
  console.log('PASS disconnect isolation: remaining clients keep translating and finish normally.');

  for(const client of [all[0],...surviving]) {
    const ready=events(client,'ready')[0], ownTurns=events(client,'turn');
    const response=await fetch(new URL(`/api/sessions/${ready.session_id}/export?format=json`,base));
    assert.equal(response.status,200);
    const result=await response.json();
    assert.equal(result.session_id,ready.session_id);
    assert.equal(result.source_language,client.source);
    assert.equal(result.source_name,client.name);
    assert.equal(result.translation,ownTurns.map(turn=>turn.delta).join(''));
    assert.equal(result.turns.length,ownTurns.length);
    assert.equal(result.turns[0].source_start_s,0);
    for(let i=0;i<result.turns.length;i++){
      assert.equal(result.turns[i].turn_id,i);
      assert(Number.isFinite(result.turns[i].scheduler_wait_s));
      assert(result.turns[i].batch_size>=1 && result.turns[i].batch_size<=4);
    }
    assert.equal(client.errors.length,0,client.errors.join('\n'));
    assert.equal(events(client,'error').length,client===extra?1:0);
  }
  console.log('PASS independent complete exports match each browser transcript, language and source; timing fields are present.');
  console.log('5 multiuser browser checks passed (CPU/mock only; no GPU throughput claim).');
  if(screenshots)console.log(`Screenshots: ${screenshots}`);
} finally {
  await browser?.close();
}
