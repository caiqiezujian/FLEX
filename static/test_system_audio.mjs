/** CPU-only permission/lifecycle tests, not native OS audio capture verification. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('./system-audio.js', import.meta.url), 'utf8');
const {requestSystemAudio} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
let count = 0;
async function test(name, run) {
  await run();
  count++;
  console.log(`PASS ${name}`);
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve=yes; reject=no; });
  return {promise, resolve, reject};
}
async function tick() { await new Promise(resolve => setImmediate(resolve)); }
function track(kind='audio', readyState='live') {
  const handlers = new Set();
  return {
    kind, readyState, stopCount:0, handlers,
    addEventListener(type, handler) { assert.equal(type, 'ended'); handlers.add(handler); },
    removeEventListener(type, handler) { assert.equal(type, 'ended'); handlers.delete(handler); },
    stop() { this.stopCount++; this.readyState='ended'; },
    end() { this.readyState='ended'; for (const handler of [...handlers]) handler({type:'ended'}); },
  };
}
function stream(tracks=[track('video'),track('audio')]) {
  return {getTracks:()=>tracks, getAudioTracks:()=>tracks.filter(item=>item.kind==='audio')};
}
function devices(display=stream(), microphone=stream([track()])) {
  const calls = [];
  return {
    calls,
    getDisplayMedia(options) { calls.push(['display',options]); return Promise.resolve(display); },
    getUserMedia(options) { calls.push(['microphone',options]); return Promise.resolve(microphone); },
  };
}
function stopped(stream, value=1) {
  for (const item of stream.getTracks()) {
    assert.equal(item.stopCount,value);
    if (value) assert.equal(item.handlers.size,0);
  }
}
function namedError(name) { const error=new Error('browser detail'); error.name=name; return error; }
function errorCode(code) { return error => { assert.equal(error.code,code); return true; }; }

await test('display picker is called synchronously before the first await', async () => {
  const api=devices(), pending=requestSystemAudio({mediaDevices:api});
  assert.equal(api.calls.length,1);
  assert.equal(api.calls[0][0],'display');
  (await pending).stop();
});
await test('capture options request audio without suppressing original playback', async () => {
  const api=devices(), resource=await requestSystemAudio({mediaDevices:api});
  assert.deepEqual(api.calls[0][1], {
    video:true, audio:{suppressLocalAudioPlayback:false}, systemAudio:'include',
    selfBrowserSurface:'exclude', surfaceSwitching:'exclude', monitorTypeSurfaces:'include',
  });
  resource.stop();
});
await test('default computer audio never requests microphone and keeps video local/live', async () => {
  const display=stream(), api=devices(display), resource=await requestSystemAudio({mediaDevices:api});
  assert.deepEqual(resource.audioStreams,[display]);
  assert.equal(resource.microphoneStream,null);
  assert.equal(resource.displayStream,display);
  assert.equal(resource.stopped,false);
  assert.equal(api.calls.length,1);
  stopped(display,0);
  assert.equal(display.getTracks()[0].readyState,'live');
  resource.stop();
});
await test('default path does not need or access a microphone permission method', async () => {
  const api=devices();
  Object.defineProperty(api,'getUserMedia',{get(){throw Error('must not read microphone API');}});
  const resource=await requestSystemAudio({mediaDevices:api});
  resource.stop();
});
await test('only explicit boolean opt-in allows microphone capture', async () => {
  const api=devices();
  await assert.rejects(requestSystemAudio({mediaDevices:api,includeMicrophone:'yes'}),TypeError);
  assert.equal(api.calls.length,0);
});
await test('optional mic is requested after display permission and returns two audio sources', async () => {
  const display=stream(), microphone=stream([track()]), permission=deferred(), api=devices(permission.promise,microphone);
  const pending=requestSystemAudio({mediaDevices:api,includeMicrophone:true});
  assert.equal(api.calls.length,1);
  permission.resolve(display);
  const resource=await pending;
  assert.deepEqual(api.calls.map(item=>item[0]),['display','microphone']);
  assert.deepEqual(api.calls[1][1],{audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
  assert.deepEqual(resource.audioStreams,[display,microphone]);
  assert.equal(resource.microphoneStream,microphone);
  resource.stop(); stopped(display); stopped(microphone);
});
await test('explicit stop cleans all streams once and does not report external interruption', async () => {
  const display=stream(), mic=stream([track()]), api=devices(display,mic);
  const resource=await requestSystemAudio({mediaDevices:api,includeMicrophone:true});
  let ended=0; resource.onEnded(()=>ended++);
  resource.stop(); resource.stop();
  assert.equal(resource.stopped,true); assert.equal(ended,0);
  stopped(display); stopped(mic);
  resource.onEnded(()=>ended++); assert.equal(ended,0);
});
await test('missing shared audio stops video and explains checkbox without microphone fallback', async () => {
  const display=stream([track('video')]), api=devices(display);
  await assert.rejects(requestSystemAudio({mediaDevices:api}),error=>{
    assert.equal(error.code,'system_audio_missing_audio');
    assert.match(error.message,/勾选/); assert.match(error.message,/不会自动改用麦克风/); return true;
  });
  stopped(display); assert.equal(api.calls.length,1);
});
await test('ended audio is not accepted as a usable shared audio track', async () => {
  const display=stream([track('video'),track('audio','ended')]), api=devices(display);
  await assert.rejects(requestSystemAudio({mediaDevices:api}),errorCode('system_audio_missing_audio'));
  stopped(display);
});
await test('a muted but live audio track is accepted: temporary silence is not a failure', async () => {
  const display=stream(); display.getAudioTracks()[0].muted=true;
  const resource=await requestSystemAudio({mediaDevices:devices(display)});
  assert.equal(resource.stopped,false); resource.stop();
});
await test('display permission rejection is clear and never falls back to microphone', async () => {
  const browserError=namedError('NotAllowedError'), api=devices();
  api.getDisplayMedia=()=>Promise.reject(browserError);
  await assert.rejects(requestSystemAudio({mediaDevices:api,includeMicrophone:true}),error=>{
    assert.equal(error.code,'system_audio_capture_failed'); assert.equal(error.cause,browserError);
    assert.match(error.message,/取消共享/); return true;
  });
  assert.equal(api.calls.length,0);
});
await test('synchronous display API errors retain cause and use Chinese guidance', async () => {
  const browserError=namedError('InvalidStateError'), api=devices();
  api.getDisplayMedia=()=>{throw browserError;};
  await assert.rejects(requestSystemAudio({mediaDevices:api}),error=>{
    assert.equal(error.cause,browserError); assert.match(error.message,/直接点击开始/); return true;
  });
});
await test('unsupported display capture fails before any microphone request', async () => {
  let mic=0;
  await assert.rejects(requestSystemAudio({mediaDevices:{getUserMedia(){mic++;}},includeMicrophone:true}),errorCode('system_audio_unsupported'));
  assert.equal(mic,0);
});
await test('insecure contexts fail with HTTPS guidance before requesting any permission', async () => {
  const api=devices();
  await assert.rejects(requestSystemAudio({mediaDevices:api,isSecureContext:false}),error=>{
    assert.equal(error.code,'system_audio_insecure'); assert.match(error.message,/HTTPS/); return true;
  });
  assert.equal(api.calls.length,0);
});
await test('unavailable optional microphone fails before opening display picker', async () => {
  const api=devices(); delete api.getUserMedia;
  await assert.rejects(requestSystemAudio({mediaDevices:api,includeMicrophone:true}),errorCode('microphone_unsupported'));
  assert.equal(api.calls.length,0);
});
await test('microphone denial stops existing screen capture instead of silently continuing', async () => {
  const display=stream(), api=devices(display), reason=namedError('NotAllowedError');
  api.getUserMedia=()=>Promise.reject(reason);
  await assert.rejects(requestSystemAudio({mediaDevices:api,includeMicrophone:true}),error=>{
    assert.equal(error.code,'microphone_capture_failed'); assert.equal(error.cause,reason);
    assert.match(error.message,/取消勾选/); return true;
  });
  stopped(display);
});
await test('synchronous microphone API errors stop screen capture', async () => {
  const display=stream(), api=devices(display);
  api.getUserMedia=()=>{throw namedError('NotReadableError');};
  await assert.rejects(requestSystemAudio({mediaDevices:api,includeMicrophone:true}),errorCode('microphone_capture_failed'));
  stopped(display);
});
await test('missing microphone audio releases both streams', async () => {
  const display=stream(), mic=stream([track('video')]), api=devices(display,mic);
  await assert.rejects(requestSystemAudio({mediaDevices:api,includeMicrophone:true}),errorCode('microphone_missing_audio'));
  stopped(display); stopped(mic);
});
await test('pre-aborted requests open no picker', async () => {
  const api=devices(), controller=new AbortController(); controller.abort();
  await assert.rejects(requestSystemAudio({mediaDevices:api,signal:controller.signal}),errorCode('system_audio_cancelled'));
  assert.equal(api.calls.length,0);
});
await test('abort while display picker is pending rejects promptly and stops late result', async () => {
  const permission=deferred(), display=stream(), api=devices(permission.promise), controller=new AbortController();
  const pending=requestSystemAudio({mediaDevices:api,signal:controller.signal});
  controller.abort();
  await assert.rejects(pending,errorCode('system_audio_cancelled'));
  permission.resolve(display); await tick(); stopped(display);
  assert.equal(api.calls.length,1);
});
await test('late picker rejection after cancellation does not become unhandled', async () => {
  const permission=deferred(), controller=new AbortController();
  const pending=requestSystemAudio({mediaDevices:devices(permission.promise),signal:controller.signal});
  controller.abort(); await assert.rejects(pending,errorCode('system_audio_cancelled'));
  permission.reject(namedError('NotAllowedError')); await tick();
});
await test('abort during microphone permission stops screen now and late microphone on arrival', async () => {
  const display=stream(), mic=stream([track()]), permission=deferred(), api=devices(display,permission.promise), controller=new AbortController();
  const pending=requestSystemAudio({mediaDevices:api,includeMicrophone:true,signal:controller.signal});
  await tick(); assert.equal(api.calls.length,2);
  controller.abort(); await assert.rejects(pending,errorCode('system_audio_cancelled'));
  stopped(display); stopped(mic,0);
  permission.resolve(mic); await tick(); stopped(mic);
});
await test('stopping screen sharing during microphone permission rejects and cleans late mic', async () => {
  const display=stream(), mic=stream([track()]), permission=deferred(), api=devices(display,permission.promise);
  const pending=requestSystemAudio({mediaDevices:api,includeMicrophone:true});
  await tick(); display.getTracks()[0].end();
  await assert.rejects(pending,errorCode('system_audio_ended')); stopped(display);
  permission.resolve(mic); await tick(); stopped(mic);
});
await test('screen video end shuts down audio and calls the interruption callback once', async () => {
  const display=stream(), resource=await requestSystemAudio({mediaDevices:devices(display)}), messages=[];
  resource.onEnded(error=>messages.push(error.code));
  display.getTracks()[0].end(); display.getAudioTracks()[0].end();
  assert.deepEqual(messages,['system_audio_ended']); assert.equal(resource.stopped,true); stopped(display);
});
await test('shared audio ending also shuts down local video', async () => {
  const display=stream(), resource=await requestSystemAudio({mediaDevices:devices(display)});
  let code; resource.onEnded(error=>{code=error.code;}); display.getAudioTracks()[0].end();
  assert.equal(code,'system_audio_ended'); stopped(display);
});
await test('optional microphone ending shuts down both capture sources', async () => {
  const display=stream(), mic=stream([track()]), resource=await requestSystemAudio({mediaDevices:devices(display,mic),includeMicrophone:true});
  let code; resource.onEnded(error=>{code=error.code;}); mic.getAudioTracks()[0].end();
  assert.equal(code,'microphone_ended'); stopped(display); stopped(mic);
});
await test('abort after acquisition stops resources and emits one cancellation', async () => {
  const display=stream(), controller=new AbortController(), resource=await requestSystemAudio({mediaDevices:devices(display),signal:controller.signal});
  const messages=[]; resource.onEnded(error=>messages.push(error.code)); controller.abort(); resource.stop();
  assert.deepEqual(messages,['system_audio_cancelled']); stopped(display);
});
await test('onEnded unsubscribe and late registration have deterministic behavior', async () => {
  const display=stream(), resource=await requestSystemAudio({mediaDevices:devices(display)});
  let removed=0, late=0; const remove=resource.onEnded(()=>removed++); remove();
  display.getAudioTracks()[0].end(); resource.onEnded(()=>late++);
  assert.equal(removed,0); assert.equal(late,1);
});
await test('a failing consumer callback cannot prevent cleanup or other callbacks', async () => {
  const display=stream(), resource=await requestSystemAudio({mediaDevices:devices(display)});
  let second=0; resource.onEnded(()=>{throw Error('caller');}); resource.onEnded(()=>second++);
  display.getAudioTracks()[0].end(); stopped(display); assert.equal(second,1);
});
await test('already-ended video does not leak new track listeners during failed setup', async () => {
  const display=stream([track('video','ended'),track('audio')]);
  await assert.rejects(requestSystemAudio({mediaDevices:devices(display)}),errorCode('system_audio_ended'));
  stopped(display);
});
await test('a broken track stop cannot prevent releasing the other tracks', async () => {
  const display=stream(); display.getTracks()[0].stop=function(){this.stopCount++; throw Error('driver');};
  const resource=await requestSystemAudio({mediaDevices:devices(display)}); resource.stop(); stopped(display);
});
await test('duplicate owned track references are only stopped once', async () => {
  const audio=track(), display=stream([track('video'),audio]), mic=stream([audio]);
  const resource=await requestSystemAudio({mediaDevices:devices(display,mic),includeMicrophone:true});
  resource.stop(); stopped(display); stopped(mic);
});
await test('synchronous abort then synchronous browser error has no unhandled interruption', async () => {
  const controller=new AbortController(), api=devices();
  api.getDisplayMedia=()=>{controller.abort(); throw namedError('NotAllowedError');};
  await assert.rejects(requestSystemAudio({mediaDevices:api,signal:controller.signal}),errorCode('system_audio_cancelled'));
  await tick();
});
await test('capture module contains no frame reader, playback, or network sender', () => {
  assert.doesNotMatch(source,/\b(?:fetch|WebSocket|RTCPeerConnection|MediaRecorder|AudioContext|VideoFrame|ImageCapture)\s*\(/);
  assert.doesNotMatch(source,/\.(?:play|requestVideoFrameCallback|drawImage|send)\s*\(/);
});
console.log(`\n${count} system-audio tests passed.`);
