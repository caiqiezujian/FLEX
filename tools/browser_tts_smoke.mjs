#!/usr/bin/env node
// Real browser, synthetic audio/protocol only. Never contacts the real TTS service.
import assert from 'node:assert/strict';
import {stat} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
const {values}=parseArgs({options:{'base-url':{type:'string',default:'http://127.0.0.1:31229'}, screenshot:{type:'string'}}});
const base=new URL(values['base-url']);
assert(base.protocol==='http:' && ['127.0.0.1','localhost'].includes(base.hostname));
const config=await(await fetch(new URL('/api/config',base))).json();
assert(config.backend==='mock' && config.status==='ready' && !config.auth_required);
const location=process.env.FLEX_PLAYWRIGHT_MODULE;
const resolved=location && ((await stat(location)).isDirectory()?createRequire(import.meta.url).resolve(location):location);
const module=await import(resolved?pathToFileURL(resolved).href:'playwright');
const chromium=module.chromium || module.default.chromium;
const wav=Buffer.alloc(44+144000);wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);
wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);wav.writeUInt32LE(24000,24);
wav.writeUInt32LE(48000,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);wav.write('data',36);wav.writeUInt32LE(144000,40);
let browser,wire,page,turns=0,chars=0,fail=false,busyResponses=0,holdPreview=false,releasePreview;
const requests=[],errors=[],heldTexts=new Map();
const send=event=>wire.send(JSON.stringify(event));
const output=(tag,delta)=>{send({type:'turn',tag,delta,source_start_s:turns,source_end_s:++turns,service_s:.1});chars += [...delta].length;};
const until = async predicate => {
  const deadline=Date.now()+5000;
  while(!predicate()){assert(Date.now()<deadline,'Timed out waiting for browser route handler');await new Promise(resolve=>setTimeout(resolve,10));}
};
try {
  browser=await chromium.launch({headless:true,args:['--no-proxy-server'],
    ...(process.env.FLEX_BROWSER_EXECUTABLE?{executablePath:process.env.FLEX_BROWSER_EXECUTABLE}:{})});
  const context=await browser.newContext({viewport:{width:1366,height:900}});page=await context.newPage();
  page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>{
    navigator.mediaDevices.getDisplayMedia=async()=>{
      const audio=new AudioContext(), oscillator=audio.createOscillator(), sink=audio.createMediaStreamDestination();
      oscillator.connect(sink);oscillator.start();void audio.resume();
      const canvas=document.createElement('canvas');canvas.width=16;canvas.height=16;
      const video=canvas.captureStream(1), vt=video.getVideoTracks()[0];
      vt.getSettings=()=>({displaySurface:'browser'});
      const tracks=[vt,...sink.stream.getAudioTracks()];
      for(const track of tracks){const stop=track.stop.bind(track);track.stop=()=>{stop();if(tracks.every(t=>t.readyState==='ended')){oscillator.stop();void audio.close();}};}
      return new MediaStream(tracks);
    };
  });
  await page.route('**/api/config',r=>r.fulfill({json:{...config,tts:{enabled:true,languages:['zh','en'],max_text_chars:200,timeout_seconds:180}}}));
  await page.route('**/api/tts/synthesize',async r=>{
    requests.push(r.request().postDataJSON());
    if (holdPreview) await new Promise(resolve=>{releasePreview=resolve;});
    const body=r.request().postDataJSON();
    if (heldTexts.has(body.text)) await new Promise(resolve=>{heldTexts.set(body.text,resolve);});
    if (busyResponses > 0) {busyResponses--;return r.fulfill({status:503,json:{detail:{code:'tts_busy',message:'Synthetic busy'}}});}
    return fail ? r.fulfill({status:503,json:{detail:{message:'Synthetic TTS unavailable'}}})
      : r.fulfill({contentType:'audio/wav',body:wav});
  });
  await page.routeWebSocket('**/ws',socket=>{
    wire=socket;socket.onMessage(message=>{
      if(typeof message!=='string')return;
      const e=JSON.parse(message);
      if(e.type==='start')send({type:'ready',sample_rate:16000,session_id:'synthetic-tts'});
      if(e.type==='finish')send({type:'done',turn_count:turns,translation_chars:chars});
    });
  });
  await page.goto(base.href,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>!document.getElementById('ttsEnabled').disabled);
  await page.locator('#ttsEnabled').check();holdPreview=true;
  const previewRequest=page.waitForRequest('**/api/tts/synthesize');
  await page.locator('#ttsPreview').click();await previewRequest;
  assert.equal(await page.locator('#ttsPreview').isDisabled(),true);
  await page.locator('#ttsPreview').evaluate(button=>{button.click();button.click();});
  assert.equal(requests.length,1);
  holdPreview=false;releasePreview();
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('正在播报'));
  assert.equal(await page.locator('#ttsPreview').isDisabled(),true);
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('播报完成'));
  assert.equal(await page.locator('#ttsPreview').isDisabled(),false);
  fail=true;await page.locator('#ttsPreview').click();
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('Synthetic TTS unavailable'));
  assert.equal(await page.locator('#ttsEnabled').isChecked(),true);
  assert.equal(await page.locator('#ttsPreview').isDisabled(),false);
  fail=false;busyResponses=1;await page.locator('#ttsPreview').click();
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('正在重试'));
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('正在播报'));
  assert.equal(requests.length,4);assert.deepEqual(requests.at(-1),requests.at(-2));
  await page.locator('#ttsStop').click();requests.length=0;
  console.log('PASS preview single-flight, visible busy recovery, failed preview preserves live speech preference.');
  await page.locator('#micMode').click();await page.locator('#startButton').click();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='同传进行中');
  for(let i=0;i<260;i++) output('WAIT','');
  await page.waitForFunction(()=>document.getElementById('turnCount').textContent.startsWith('260'));
  await page.locator('#turnDetails > summary').click();
  await page.waitForFunction(()=>{const b=document.getElementById('turnList');return b.scrollHeight-b.scrollTop-b.clientHeight<3;});
  await page.locator('#turnList').evaluate(el=>{el.dispatchEvent(new WheelEvent('wheel',{deltaY:-100}));el.scrollTop=0;el.dispatchEvent(new Event('scroll'));});
  output('WAIT','');
  await page.waitForFunction(()=>!document.getElementById('turnFollowButton').hidden);
  assert.equal(await page.locator('#turnList').evaluate(el=>el.scrollTop),0);
  await page.locator('#turnFollowButton').click();
  await page.waitForFunction(()=>{const b=document.getElementById('turnList');return b.scrollHeight-b.scrollTop-b.clientHeight<3;});
  for(let i=0;i<220;i++)output('WAIT','');
  await page.waitForFunction(()=>document.getElementById('turnCount').textContent.startsWith('481'));
  await page.waitForFunction(()=>{const b=document.getElementById('turnList');return b.scrollHeight-b.scrollTop-b.clientHeight<3;});
  console.log('PASS policy panel follows latest including initial opening, preserves manual reading, resumes on click.');
  await page.locator('#ttsEnabled').check();output('WAIT','');
  const firstSpeech=page.waitForResponse('**/api/tts/synthesize');
  output('FLEX','大学教授可以');await firstSpeech;
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('正在播报'));
  assert.equal(requests.length,1);assert.deepEqual(requests[0],{text:'大学教授可以',language:'zh'});
  heldTexts.set('为学生创作内容。',null);heldTexts.set('也能为听众制作内容。',null);
  const bRequest=page.waitForRequest(r=>r.url().endsWith('/api/tts/synthesize') && r.postDataJSON().text==='为学生创作内容。');
  const cRequest=page.waitForRequest(r=>r.url().endsWith('/api/tts/synthesize') && r.postDataJSON().text==='也能为听众制作内容。');
  output('DIRECT','为学生创作内容。');output('FLEX','也能为听众制作内容。');await Promise.all([bRequest,cRequest]);
  await until(()=>requests.length===3 && [...heldTexts.values()].every(value=>typeof value==='function'));
  assert.equal(requests.length,3);
  const pipeline=await page.evaluate(()=>{const e=window.flexTtsDiagnostics(),id=e.at(-1).run;return e.filter(x=>x.run===id);});
  assert.equal(pipeline.filter(e=>e.phase==='request_start').length,3);
  assert.equal(pipeline.filter(e=>e.phase==='play_end').length,0,'B and C must be requested while A is still playing');
  heldTexts.get('也能为听众制作内容。')();
  await page.waitForFunction(()=>{const e=window.flexTtsDiagnostics(),id=e.at(-1).run;return e.some(x=>x.run===id&&x.phase==='audio_ready'&&x.segment===3);});
  assert.equal(await page.evaluate(()=>{const e=window.flexTtsDiagnostics(),id=e.at(-1).run;return e.some(x=>x.run===id&&x.phase==='play_start'&&x.segment===3);}),false);
  heldTexts.get('为学生创作内容。')();heldTexts.clear();
  await page.waitForFunction(()=>{const e=window.flexTtsDiagnostics(),id=e.at(-1).run;return e.some(x=>x.run===id&&x.phase==='play_start'&&x.segment===3);});
  const played=await page.evaluate(()=>{const e=window.flexTtsDiagnostics(),id=e.at(-1).run;return e.filter(x=>x.run===id&&x.phase==='play_start').map(x=>x.segment);});
  assert.deepEqual(played,[1,2,3]);
  assert.equal(await page.locator('#transcript').textContent(),'大学教授可以为学生创作内容。也能为听众制作内容。');
  await page.locator('#ttsStop').click();assert.equal(await page.locator('#ttsEnabled').isChecked(),false);
  output('DIRECT','停止播报后字幕继续。');
  await page.waitForFunction(()=>document.getElementById('transcript').textContent.includes('停止播报后字幕继续'));
  assert.equal(requests.length,3);
  console.log('PASS S2TT WAIT silence, immediate A playback, concurrent B/C prefetch, reversed responses with ordered audio, and explicit stop.');
  fail=true;await page.locator('#ttsEnabled').check();output('DIRECT','语音服务报错时，文字同传仍然继续。');
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('Synthetic TTS unavailable'));
  assert.equal(await page.locator('#sessionStatus').textContent(),'同传进行中');
  await page.locator('#finishButton').click();
  await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent==='同传已结束');
  fail=false;await page.locator('#targetLang').selectOption('en');await page.locator('#ttsPreview').click();
  await page.waitForFunction(()=>document.getElementById('ttsStatus').textContent.includes('正在播报'));
  assert.equal(requests.at(-1).language,'en');
  await page.locator('#ttsStop').click();
  console.log('PASS TTS failure isolation and English target voice preview after direction switch.');
  await page.waitForFunction(()=>{const b=document.getElementById('turnList');return b.scrollHeight-b.scrollTop-b.clientHeight<3;});
  assert.equal(await page.locator('#turnFollowButton').isVisible(),false);
  if(values.screenshot)await page.screenshot({path:values.screenshot,fullPage:true});
  for(const width of [390,900,1366]){
    await page.setViewportSize({width,height:900});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`overflow at ${width}`);
  }
  assert.deepEqual(errors,[]);console.log('PASS responsive layout and no browser exceptions.');
} catch(error) {
  console.error('Browser diagnostics:',JSON.stringify({requests,errors,ui:await page?.evaluate(()=>({
    speech:document.getElementById('ttsStatus').textContent,
    checked:document.getElementById('ttsEnabled').checked,
    state:document.getElementById('sessionStatus').textContent,
    error:document.getElementById('errorMessage').textContent,
  }))}));
  throw error;
} finally {await browser?.close();}
