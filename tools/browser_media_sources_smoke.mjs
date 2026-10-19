#!/usr/bin/env node
/** WMV conversion + computer-audio integration against a local mock backend.
 * Screen/audio permission APIs are explicitly mocked with canvas/oscillators.
 * No physical screen, microphone, system audio, GPU or user media is accessed.
 * Real Web Audio/AudioWorklet, WebSocket PCM, FFmpeg and native video playback
 * are exercised. No autoplay-policy or insecure-context exemptions are used.
 */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp,stat,unlink,rmdir} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';

const {values}=parseArgs({options:{'base-url':{type:'string',default:'http://127.0.0.1:31225'},screenshots:{type:'boolean',default:false}}});
const base=new URL(values['base-url']);
assert(['http:','https:'].includes(base.protocol) && ['localhost','127.0.0.1'].includes(base.hostname));
assert(!base.username && !base.password && base.pathname==='/' && !base.search && !base.hash);
const config=await(await fetch(new URL('/api/config',base))).json();
assert.equal(config.backend,'mock');assert.equal(config.status,'ready');
const token=process.env.FLEX_ACCESS_TOKEN || '';
assert(!config.auth_required || token,'Authenticated mock requires FLEX_ACCESS_TOKEN.');
const location=process.env.FLEX_PLAYWRIGHT_MODULE;
const resolved=location && ((await stat(location)).isDirectory()?createRequire(import.meta.url).resolve(location):location);
const playwright=await import(resolved?pathToFileURL(resolved).href:'playwright');
const chromium=playwright.chromium || playwright.default.chromium;
let browser,temporary,fixture;
const screenshots=values.screenshots?await mkdtemp(path.join(tmpdir(),'flex-media-source-ui-')):null;
const reports=[];

async function pageFor(scenarios=[{}]) {
  const context=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:1366,height:768}});
  const page=await context.newPage(),wires=[],errors=[],playbackRequests=[];
  page.on('pageerror',error=>errors.push(String(error)));
  page.on('request',request=>{if(new URL(request.url()).pathname.endsWith('/playback'))playbackRequests.push(request.url());});
  page.on('websocket',socket=>{
    const wire={sent:[],received:[],samples:0};wires.push(wire);
    socket.on('framesent',({payload})=>{
      if(typeof payload==='string')wire.sent.push(JSON.parse(payload));
      else wire.samples+=Math.max(0,(payload.length-8)/2);
    });
    socket.on('framereceived',({payload})=>wire.received.push(JSON.parse(String(payload))));
  });
  await page.addInitScript(({scenarios})=>{
    window.__captureCalls=[];window.__captures=[];window.__micCalls=0;window.__scenarios=scenarios;
    function synthetic(kind,scenario) {
      const tracks=[],entry={kind,tracks,context:null};
      if(kind==='display'){
        const canvas=document.createElement('canvas');canvas.width=16;canvas.height=16;
        canvas.getContext('2d').fillRect(0,0,16,16);entry.canvas=canvas;
        tracks.push(...canvas.captureStream(1).getVideoTracks());
      }
      if(!scenario.noAudio){
        const audio=new AudioContext(),oscillator=audio.createOscillator(),gain=audio.createGain(),destination=audio.createMediaStreamDestination();
        oscillator.frequency.value=kind==='display'?440:880;gain.gain.value=.35;
        oscillator.connect(gain);gain.connect(destination);oscillator.start();void audio.resume();
        tracks.push(...destination.stream.getAudioTracks());entry.context=audio;
      }
      for(const track of tracks){
        const stop=track.stop.bind(track);track.stop=()=>{stop();if(tracks.every(item=>item.readyState==='ended'))void entry.context?.close();};
      }
      window.__captures.push(entry);return new MediaStream(tracks);
    }
    navigator.mediaDevices.getDisplayMedia=options=>{
      const scenario=window.__scenarios.shift() || {};window.__currentScenario=scenario;
      window.__captureCalls.push({active:navigator.userActivation.isActive,options});
      if(scenario.deny)return Promise.reject(new DOMException('synthetic cancellation','NotAllowedError'));
      const stream=synthetic('display',scenario);
      if(scenario.manual)return new Promise(resolve=>{window.__resolvePicker=()=>resolve(stream);});
      return new Promise(resolve=>setTimeout(()=>resolve(stream),scenario.delay || 0));
    };
    navigator.mediaDevices.getUserMedia=()=>{
      window.__micCalls++;
      if(window.__currentScenario.denyMic)return Promise.reject(new DOMException('synthetic microphone denial','NotAllowedError'));
      return Promise.resolve(synthetic('microphone',{}));
    };
  },{scenarios});
  await page.goto(base.href,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>document.getElementById('serverStatus').classList.contains('ready'));
  if(token){await page.locator('#settings').evaluate(element=>{element.open=true;});await page.locator('#authToken').fill(token);await page.locator('#settings').evaluate(element=>{element.open=false;});}
  return {context,page,wires,errors,playbackRequests};
}
async function startSystem(test,{microphone=false,chunk=1}={}) {
  await test.page.locator('#micMode').click();
  assert.equal(await test.page.locator('#includeMicrophone').isChecked(),false);
  if(microphone)await test.page.locator('#includeMicrophone').check();
  await test.page.locator('#settings > summary').click();
  await test.page.locator('#chunkSeconds').selectOption(String(chunk));
  await test.page.locator('#startButton').click();
}
async function waitTurns(test,count=2) {
  await test.page.waitForFunction(count=>{
    const text=document.getElementById('turnCount').textContent;return parseInt(text,10)>=count;
  },count,{timeout:30000});
}
async function done(test) {
  await test.page.waitForFunction(()=>document.getElementById('sessionStatus').textContent.includes('同传已结束'),null,{timeout:30000});
  const wire=test.wires.at(-1),event=wire.received.find(event=>event.type==='done');
  assert(event);assert.equal(event.reason,'finished');
  assert.deepEqual(test.errors,[]);
  return {wire,event};
}
async function checkStopped(test){assert.equal(await test.page.evaluate(()=>window.__captures.every(entry=>entry.tracks.every(track=>track.readyState==='ended'))),true);}
async function report(test,name){assert.deepEqual(test.errors,[]);await test.context.close();reports.push(name);console.log(`PASS ${name}`);}

try{
  browser=await chromium.launch({headless:true,...(process.env.FLEX_BROWSER_EXECUTABLE?{executablePath:process.env.FLEX_BROWSER_EXECUTABLE}:{}),args:['--no-proxy-server']});
  {
    const test=await pageFor([{delay:6200},{}]);await startSystem(test,{chunk:2});
    assert.equal(test.wires.length,0,'No model session before chooser resolves.');
    await waitTurns(test);
    assert.equal(await test.page.evaluate(()=>window.__captureCalls[0].active),true);
    assert.equal(await test.page.evaluate(()=>window.__micCalls),0);
    assert.equal(test.wires[0].sent[0].source_kind,'system');assert.equal(test.wires[0].sent[0].chunk_seconds,2);
    assert.equal(test.wires[0].sent[0].include_microphone,false);
    await test.page.evaluate(()=>window.__captures[0].tracks.find(track=>track.kind==='video').dispatchEvent(new Event('ended')));
    const {wire,event}=await done(test);
    assert.equal(wire.sent.filter(event=>event.type==='finish').length,1);
    assert(Math.abs(event.received_audio_s-wire.samples/16000)<.001,'All flushed PCM must be accounted.');
    await checkStopped(test);
    await test.page.locator('#startButton').click();await waitTurns(test,1);
    assert.equal(await test.page.evaluate(()=>window.__captureCalls[1].active),true,'Restart keeps user activation.');
    await test.page.locator('#finishButton').click();await done(test);await checkStopped(test);
    if(screenshots){
      await test.page.screenshot({path:path.join(screenshots,'system-1366.png'),fullPage:true});
      await test.page.setViewportSize({width:390,height:844});
      assert(await test.page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
      await test.page.screenshot({path:path.join(screenshots,'system-390.png'),fullPage:true});
    }
    await report(test,'delayed display chooser, real PCM tail drain and restart without microphone');
  }
  {
    const test=await pageFor();await startSystem(test,{microphone:true,chunk:3});await waitTurns(test,1);
    assert.equal(await test.page.evaluate(()=>window.__micCalls),1);
    assert.equal(test.wires[0].sent[0].include_microphone,true);
    await test.page.locator('#finishButton').click();await done(test);await checkStopped(test);
    await report(test,'explicit microphone opt-in mixes computer sound and speech');
  }
  for(const [scenario,label] of [[{noAudio:true},'missing shared audio'],[{deny:true},'display permission cancelled'],[{denyMic:true},'microphone denied']]){
    const test=await pageFor([scenario]);await startSystem(test,{microphone:Boolean(scenario.denyMic)});
    await test.page.locator('#errorBox').waitFor({state:'visible'});
    assert.equal(test.wires.length,0);await checkStopped(test);
    if(scenario.noAudio)assert.match(await test.page.locator('#errorMessage').textContent(),/没有收到共享音频/);
    await report(test,`${label} creates no model session and cleans all capture tracks`);
  }
  {
    const test=await pageFor([{manual:true},{}]);await startSystem(test);
    await test.page.locator('#finishButton').click();await test.page.locator('#errorBox').waitFor({state:'visible'});
    await test.page.waitForFunction(()=>!document.getElementById('startButton').disabled);
    await test.page.locator('#startButton').click();await waitTurns(test,1);
    await test.page.evaluate(()=>window.__resolvePicker());
    await test.page.waitForFunction(()=>window.__captures[0].tracks.every(track=>track.readyState==='ended'));
    assert.equal(test.wires.length,1,'Cancelled picker must not start a late model session.');
    assert.equal(await test.page.evaluate(()=>window.__captures[1].tracks.every(track=>track.readyState==='live')),true);
    await test.page.locator('#finishButton').click();await done(test);await checkStopped(test);
    await report(test,'pending chooser cancel and late result cannot affect a new session');
  }
  temporary=await mkdtemp(path.join(tmpdir(),'flex-media-sources-'));fixture=path.join(temporary,'synthetic.wmv');
  execFileSync(process.env.FLEX_FFMPEG || 'ffmpeg',['-nostdin','-v','error','-f','lavfi','-i','testsrc=size=160x90:rate=10','-f','lavfi','-i','sine=frequency=523:sample_rate=16000','-t','3.2','-c:v','wmv2','-c:a','wmav2',fixture],{stdio:'pipe',timeout:30000});
  {
    const test=await pageFor();await test.page.locator('#fileInput').setInputFiles(fixture);
    assert.equal(await test.page.locator('#media').getAttribute('src'),null,'Never decode original WMV in browser.');
    assert.match(await test.page.locator('#fileDetails').textContent(),/开始后准备播放/);
    await test.page.locator('#startButton').click();
    if(screenshots){
      await test.page.waitForFunction(()=>document.getElementById('captureDot').classList.contains('active') && document.getElementById('media').currentTime>.7);
      await test.page.screenshot({path:path.join(screenshots,'wmv-playing-1366.png'),fullPage:true});
    }
    await done(test);
    assert(test.playbackRequests.length>0);assert(test.playbackRequests.every(url=>new URL(url).origin===base.origin && !new URL(url).search));
    assert.equal(test.wires[0].sent[0].source_kind,'file');assert.equal(test.wires[0].samples,0);
    assert.equal(await test.page.locator('#media').getAttribute('src'),null,'Expired prepared source detached at completion.');
    assert.equal(await test.page.evaluate(()=>window.__captureCalls.length),0);
    await test.page.locator('#startButton').click();await done(test);
    assert.equal(test.wires.length,2);assert.notEqual(test.wires[0].sent[0].media_id,test.wires[1].sent[0].media_id);
    await report(test,'WMV converts and streams natively, completes, cleans, and uploads fresh on restart');
  }
  console.log(JSON.stringify({passed:reports.length,tests:reports,screenshots},null,2));
}finally{
  await browser?.close();
  if(fixture)await unlink(fixture).catch(()=>{});
  if(temporary)await rmdir(temporary).catch(()=>{});
}
