/** CPU-only adapter tests: node static/test_capture.mjs. Not a real-browser test. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const pcmSource = await readFile(new URL('./pcm-core.js', import.meta.url), 'utf8');
const pcmURL = `data:text/javascript;base64,${Buffer.from(pcmSource).toString('base64')}`;
const adapterSource = (await readFile(new URL('./capture-adapter.js', import.meta.url), 'utf8'))
  .replace(/(['"])\.\/pcm-core\.js\1/g, JSON.stringify(pcmURL));
const { selectCaptureMode, createFileCaptureAdapter, FileCaptureTimeline } = await import(
  `data:text/javascript;base64,${Buffer.from(adapterSource).toString('base64')}`
);
let count = 0;
async function test(name, run) {
  await run();
  count++;
  console.log(`PASS ${name}`);
}

await test('remote HTTP file mode selects the compatibility adapter', () => {
  assert.equal(selectCaptureMode({sourceKind:'file',isSecureContext:false,hasWorklet:false,hasScriptProcessor:true}), 'http-file');
});
await test('remote HTTP never tries AudioWorklet even when a partial API exists', () => {
  assert.equal(selectCaptureMode({sourceKind:'file',isSecureContext:false,hasWorklet:true,hasScriptProcessor:true}), 'http-file');
});
await test('HTTPS and localhost retain AudioWorklet for both file and microphone', () => {
  for (const sourceKind of ['file','microphone']) {
    assert.equal(selectCaptureMode({sourceKind,isSecureContext:true,hasWorklet:true,hasScriptProcessor:true}), 'worklet');
  }
});
await test('HTTP microphone is rejected independently of file compatibility', () => {
  for (const hasWorklet of [false,true]) {
    assert.throws(() => selectCaptureMode({sourceKind:'microphone',isSecureContext:false,hasWorklet,hasScriptProcessor:true}));
  }
});
await test('file playback can use compatibility mode when secure-context Worklet is unavailable', () => {
  assert.equal(selectCaptureMode({sourceKind:'file',isSecureContext:true,hasWorklet:false,hasScriptProcessor:true}), 'http-file');
});
await test('missing capture support fails explicitly instead of silently playing without PCM', () => {
  for (const sourceKind of ['file','microphone']) {
    assert.throws(() => selectCaptureMode({sourceKind,isSecureContext:false,hasWorklet:false,hasScriptProcessor:false}));
  }
  assert.throws(() => selectCaptureMode({sourceKind:'microphone',isSecureContext:true,hasWorklet:false,hasScriptProcessor:true}));
});

// The fake context intentionally does not emulate browser rendering. It only
// checks the adapter's deterministic scheduling and resource lifecycle contract.
function fakeContext(sampleRate = 48000) {
  const context = {sampleRate, currentTime:0, processors:[], gains:[]};
  const baseNode = () => ({
    connections:[], disconnectCount:0,
    connect(destination) { this.connections.push(destination); return destination; },
    disconnect() { this.connections.length = 0; this.disconnectCount++; },
  });
  context.createGain = () => {
    const node = {...baseNode(), gain:{value:1}};
    context.gains.push(node);
    return node;
  };
  context.createScriptProcessor = (bufferSize, inputChannels, outputChannels) => {
    const node = {
      ...baseNode(), bufferSize, inputChannels, outputChannels, onaudioprocess:null,
    };
    context.processors.push(node);
    return node;
  };
  return context;
}
function pcmMessages(messages) { return messages.filter(message => message.type === 'pcm'); }
function pcmSamples(messages) { return pcmMessages(messages).flatMap(message => [...new Int16Array(message.buffer)]); }

function fixture(sampleRate = 16000, mediaOffset = 0, frameSamples = 1600) {
  const messages = [], errors = [];
  const timeline = new FileCaptureTimeline({sampleRate, mediaOffset, frameSamples,
    onMessage:message => messages.push(message), onError:error => errors.push(error)});
  return {timeline, messages, errors};
}
function command(f, op, contextTime, mediaTime, requestId = 1) {
  f.timeline.command(op, requestId, {contextTime, mediaTime});
}
function samples(length, value = .25) { return new Float32Array(length).fill(value); }
function hasAck(f, requestId) { return f.messages.some(message => message.type === 'ack' && message.requestId === requestId); }

await test('stereo is averaged to mono and framed without modifying channel input', () => {
  const f = fixture(), left = samples(3200, .5), right = samples(3200, 0);
  command(f, 'active', 0, 0);
  f.timeline.push([left,right], 0, .2);
  command(f, 'finish', .2, .2, 99);
  assert.equal(pcmMessages(f.messages).length, 2);
  assert(pcmMessages(f.messages).every(message => message.buffer.byteLength === 3200));
  assert.deepEqual(pcmSamples(f.messages), new Array(3200).fill(8192));
  assert(left.every(value => value === .5));
  assert(right.every(value => value === 0));
  assert.deepEqual(f.errors, []);
});

await test('48 kHz audio resamples to exactly 16 kHz in 100 ms PCM frames', () => {
  const f = fixture(48000);
  command(f, 'active', 1, 0);
  for (let i=0;i<10;i++) f.timeline.push([samples(4800)], 48000+i*4800, (i+1)/10);
  command(f, 'finish', 2, 1, 99);
  assert.equal(pcmSamples(f.messages).length, 16000);
  assert.equal(pcmMessages(f.messages).length, 10);
  assert(pcmMessages(f.messages).every(message => message.buffer.byteLength === 3200));
  assert.deepEqual(f.errors, []);
  assert.deepEqual(f.messages.at(-1), {type:'ack', requestId:99});
});

await test('input blocks are clipped to the active media interval, not submitted wholesale', () => {
  const f = fixture();
  command(f, 'active', .01, 0);
  f.timeline.push([Float32Array.from({length:640}, (_,i) => i < 160 ? -.5 : .5)], 0, .02);
  command(f, 'finish', .03, .02, 99);
  assert.deepEqual(pcmSamples(f.messages), new Array(320).fill(16384));
  assert.deepEqual(f.errors, []);
});

await test('no PCM before activation; delayed playing preserves the already-played prefix', () => {
  const f = fixture();
  f.timeline.push([samples(800)], 0, .05);
  assert.equal(pcmSamples(f.messages).length, 0);
  command(f, 'active', .05, .05);
  command(f, 'finish', .05, .05, 99);
  assert.equal(pcmSamples(f.messages).length, 800);
  assert.deepEqual(f.errors, []);
});

await test('future buffered samples never exceed current media playback position', () => {
  const f = fixture();
  command(f, 'active', 0, 0);
  f.timeline.push([samples(3200, .25)], 0, .025);
  assert.equal(pcmSamples(f.messages).length, 0);
  command(f, 'pause', .025, .025, 99);
  assert.equal(pcmSamples(f.messages).length, 400);
  assert.deepEqual(f.errors, []);
});

await test('pause drains a delayed already-played tail before acknowledging', () => {
  const f = fixture();
  command(f, 'active', 1, 0);
  f.timeline.push([samples(320)], 16000, .02);
  command(f, 'pause', 1.03, .03, 99);
  assert.equal(hasAck(f, 99), false);
  assert.equal(pcmSamples(f.messages).length, 0);
  f.timeline.push([samples(320)], 16320, .03);
  assert.equal(pcmSamples(f.messages).length, 480);
  assert.deepEqual(f.messages.at(-1), {type:'ack', requestId:99});
  assert.deepEqual(f.errors, []);
});

await test('pause and resume exclude paused samples with no silence or duplicated prefix', () => {
  const f = fixture();
  command(f, 'active', 1, 0);
  f.timeline.push([samples(320, .25)], 16000, .02);
  command(f, 'pause', 1.03, .03, 99);
  f.timeline.push([samples(320, .25)], 16320, .03);
  f.timeline.push([samples(15360, -1)], 16640, .03);
  command(f, 'active', 2, .03, 100);
  f.timeline.push([samples(320, .5)], 32000, .05);
  command(f, 'finish', 2.02, .05, 101);
  assert.deepEqual(pcmSamples(f.messages), [...new Array(480).fill(8192),...new Array(320).fill(16384)]);
  assert.deepEqual(f.errors, []);
});

await test('finish drains the valid tail, emits PCM before ack, and never emits it twice', () => {
  const f = fixture(48000);
  command(f, 'active', 0, 0);
  command(f, 'finish', 128/48000, 128/48000, 99);
  assert.equal(hasAck(f, 99), false);
  f.timeline.push([samples(128)], 0, 128/48000);
  assert.equal(pcmSamples(f.messages).length, 43);
  assert.equal(f.messages.at(-2).type, 'pcm');
  assert.deepEqual(f.messages.at(-1), {type:'ack', requestId:99});
  command(f, 'finish', 128/48000, 128/48000, 100);
  command(f, 'active', .1, .1, 101);
  f.timeline.push([samples(128)], 128, .2);
  assert.equal(pcmSamples(f.messages).length, 43);
  assert.deepEqual(f.errors, []);
});

await test('cancel discards the unsubmitted tail and resolves pending control requests', () => {
  const f = fixture();
  command(f, 'active', 0, 0);
  f.timeline.push([samples(100)], 0, 100/16000);
  command(f, 'pause', .02, .02, 98);
  command(f, 'cancel', .02, .02, 99);
  command(f, 'finish', .02, .02, 100);
  f.timeline.push([samples(500)], 100, .02);
  assert.equal(pcmSamples(f.messages).length, 0);
  assert.equal(hasAck(f, 98), true);
  assert.equal(hasAck(f, 99), true);
  assert.equal(hasAck(f, 100), true);
});

await test('an empty session finishes without fabricated silent audio', () => {
  const f = fixture();
  command(f, 'finish', 0, 0, 99);
  assert.deepEqual(f.messages, [{type:'ack', requestId:99}]);
  assert.deepEqual(f.errors, []);
});

await test('a nonzero file starting position is local to the new session', () => {
  const f = fixture(16000, 18.5);
  command(f, 'active', 2, 18.5);
  f.timeline.push([samples(1600)], 32000, 18.6);
  command(f, 'finish', 2.1, 18.6, 99);
  assert.equal(pcmSamples(f.messages).length, 1600);
  assert.deepEqual(f.errors, []);
});

await test('gap or overlap in callback input is rejected without invented fill', () => {
  for (const start of [319,321,1000]) {
    const f = fixture();
    command(f, 'active', 0, 0);
    f.timeline.push([samples(320)], 0, .02);
    f.timeline.push([samples(320)], start, .04);
    assert.equal(f.errors.length, 1);
    assert.equal(pcmSamples(f.messages).length, 0);
    f.timeline.push([samples(320)], 640, .06);
    assert.equal(f.errors.length, 1);
  }
});

await test('unavailable already-played history fails instead of skipping to a later block', () => {
  const f = fixture();
  command(f, 'active', 0, 0);
  f.timeline.push([samples(320)], 160, .03);
  assert.equal(f.errors.length, 1);
  assert.equal(pcmSamples(f.messages).length, 0);
});

await test('late activation and backwards playback position fail explicitly', () => {
  for (const mediaTime of [.6,-.1]) {
    const f = fixture();
    command(f, 'active', 1, mediaTime);
    assert.equal(f.errors.length, 1);
    assert.equal(pcmSamples(f.messages).length, 0);
  }
});

await test('a playback jump on close and inconsistent channel lengths are rejected', () => {
  const jumped = fixture();
  command(jumped, 'active', 0, 0);
  command(jumped, 'finish', .1, 10, 99);
  assert.equal(jumped.errors.length, 1);
  const channels = fixture();
  command(channels, 'active', 0, 0);
  channels.timeline.push([samples(320),samples(319)], 0, .02);
  assert.equal(channels.errors.length, 1);
  assert.equal(pcmSamples(channels.messages).length, 0);
});

await test('rapid pause, resume, then finish still acknowledges in order after delayed input', () => {
  const f = fixture();
  command(f, 'active', 0, 0, 1);
  command(f, 'pause', .02, .02, 2);
  command(f, 'active', .03, .02, 3);
  command(f, 'finish', .05, .04, 4);
  assert.equal(hasAck(f,2), false);
  assert.equal(hasAck(f,4), false);
  f.timeline.push([Float32Array.from({length:800}, (_,i) => i < 320 ? .25 : i < 480 ? -1 : .5)], 0, .04);
  assert.deepEqual(pcmSamples(f.messages), [...new Array(320).fill(8192),...new Array(320).fill(16384)]);
  const acks = f.messages.filter(message => message.type === 'ack').map(message => message.requestId);
  assert.deepEqual(acks, [1,3,2,4]);
  assert.deepEqual(f.errors, []);
});

await test('duplicate active events do not duplicate source audio', () => {
  const f = fixture();
  command(f, 'active', 0, 0, 1);
  f.timeline.push([samples(160)], 0, .01);
  command(f, 'active', .01, .01, 2);
  f.timeline.push([samples(160)], 160, .02);
  command(f, 'finish', .02, .02, 3);
  assert.equal(pcmSamples(f.messages).length, 320);
  assert.deepEqual(f.errors, []);
});

await test('pre-activation history stays bounded without emitting idle silence', () => {
  const f = fixture();
  for (let i=0;i<1600;i++) f.timeline.push([samples(160,0)], i*160, 0);
  assert(f.timeline.blocks.length <= 52);
  assert.equal(pcmSamples(f.messages).length, 0);
  assert.deepEqual(f.errors, []);
});

await test('stalled media time cannot create an unbounded capture backlog', () => {
  const f = fixture();
  command(f, 'active', 0, 0);
  for (let i=0;i<160;i++) f.timeline.push([samples(160)], i*160, 0);
  assert.equal(f.errors.length, 1);
  assert.equal(pcmSamples(f.messages).length, 0);
  assert.equal(f.timeline.blocks.length, 0);
});

function adapterFixture(sampleRate = 48000, connect = true) {
  const context = fakeContext(sampleRate), media = {currentTime:0};
  const messages = [], errors = [];
  const adapter = createFileCaptureAdapter(context, {media, onError:error => errors.push(error)});
  adapter.port.onmessage = ({data}) => messages.push(data);
  if (connect) adapter.connect({});
  return {context,media,messages,errors,adapter,processor:context.processors[0]};
}
function render(f, startFrame, size = 4096, estimatedStart = startFrame) {
  const channels = [samples(size,.5),samples(size,0)];
  const output = samples(size,1);
  const event = {
    playbackTime:(estimatedStart+2*size)/f.context.sampleRate,
    inputBuffer:{length:size,numberOfChannels:channels.length,getChannelData:index => channels[index]},
    outputBuffer:{numberOfChannels:1,getChannelData:() => output},
  };
  f.processor.onaudioprocess(event);
  return output;
}

await test('adapter keeps the audible branch direct and the ScriptProcessor output silent', () => {
  const f = adapterFixture(48000,false), destination = {};
  try {
    f.adapter.connect(destination);
    f.adapter.connect(destination);
    assert.equal(f.context.processors.length, 1);
    assert.deepEqual(f.adapter.inputNode.connections, [f.processor,destination]);
    assert.equal(f.context.gains[1].gain.value, 0);
    assert.deepEqual(f.context.gains[1].connections, [destination]);
    assert(render(f,0).every(value => value === 0));
  } finally { f.adapter.disconnect(); }
});

await test('adapter passes commands and PCM through its Worklet-compatible message port', () => {
  const f = adapterFixture();
  try {
    f.adapter.port.postMessage({op:'active', requestId:1});
    f.context.currentTime = 4096/48000;
    f.media.currentTime = 4096/48000;
    render(f,0);
    f.adapter.port.postMessage({op:'finish', requestId:99});
    assert.equal(pcmSamples(f.messages).length, Math.ceil(4096/3));
    assert.deepEqual(f.messages.at(-1), {type:'ack', requestId:99});
    assert.deepEqual(f.errors, []);
  } finally { f.adapter.disconnect(); }
});

await test('small callback dispatch jitter keeps sample positions continuous', () => {
  const f = adapterFixture();
  try {
    f.adapter.port.postMessage({op:'active', requestId:1});
    for (let i=0;i<3;i++) {
      f.media.currentTime = (i+1)*4096/48000;
      f.context.currentTime = f.media.currentTime;
      render(f,i*4096,4096,i*4096+(i ? 128 : 0));
    }
    f.adapter.port.postMessage({op:'finish', requestId:99});
    assert.equal(pcmSamples(f.messages).length, 4096);
    assert.deepEqual(f.errors, []);
  } finally { f.adapter.disconnect(); }
});

await test('first delayed callback is anchored to graph connection, preserving the media prefix', () => {
  const f = adapterFixture();
  try {
    f.adapter.port.postMessage({op:'active', requestId:1});
    f.context.currentTime = 4096/48000;
    f.media.currentTime = 4096/48000;
    render(f,0,4096,256);
    f.adapter.port.postMessage({op:'finish', requestId:99});
    assert.equal(pcmSamples(f.messages).length, Math.ceil(4096/3));
    assert.deepEqual(f.messages.at(-1), {type:'ack', requestId:99});
    assert.deepEqual(f.errors, []);
  } finally { f.adapter.disconnect(); }
});

await test('large dispatch drift and invalid callback times fail once', () => {
  for (const estimatedStart of [4096+2049,NaN]) {
    const f = adapterFixture();
    try {
      render(f,0);
      render(f,4096,4096,estimatedStart);
      assert.equal(f.errors.length, 1);
      render(f,8192);
      assert.equal(f.errors.length, 1);
    } finally { f.adapter.disconnect(); }
  }
});

await test('adapter disconnect is idempotent and tears down nodes, callback and port', () => {
  const f = adapterFixture(), callback = f.processor.onaudioprocess;
  f.adapter.port.postMessage({op:'active', requestId:1});
  f.context.currentTime = .05;
  f.media.currentTime = .05;
  f.adapter.port.postMessage({op:'finish', requestId:99});
  assert.equal(hasAck(f,99), false);
  f.adapter.disconnect();
  const messageCount = f.messages.length;
  f.adapter.disconnect();
  f.adapter.port.close();
  f.adapter.port.postMessage({op:'active', requestId:100});
  callback({});
  assert.equal(f.messages.length, messageCount);
  assert.equal(f.processor.onaudioprocess, null);
  assert.equal(f.adapter.port.onmessage, null);
  for (const node of [...f.context.gains,f.processor]) assert.equal(node.disconnectCount, 1);
  assert.equal(pcmSamples(f.messages).length, 0);
  assert.deepEqual(f.errors, []);
});

await test('missing final callback times out explicitly and cleanup cancels remaining timers', () => {
  const originalSetTimeout = globalThis.setTimeout, originalClearTimeout = globalThis.clearTimeout;
  const scheduled = new Map();
  let timerNumber = 0, f;
  globalThis.setTimeout = (callback, delay) => {
    const id = ++timerNumber;
    scheduled.set(id, {callback,delay});
    return id;
  };
  globalThis.clearTimeout = id => scheduled.delete(id);
  try {
    f = adapterFixture();
    f.adapter.port.postMessage({op:'active', requestId:1});
    f.context.currentTime = f.media.currentTime = .05;
    f.adapter.port.postMessage({op:'finish', requestId:99});
    assert.equal(scheduled.size, 1);
    const [timerId,timer] = [...scheduled.entries()][0];
    assert.equal(timer.delay, 1800);
    scheduled.delete(timerId); // Real one-shot timers are removed before dispatch.
    timer.callback();
    assert.equal(f.errors.length, 1);
    assert.equal(pcmSamples(f.messages).length, 0);
    f.adapter.disconnect();
    assert.equal(scheduled.size, 0);
  } finally {
    f?.adapter.disconnect();
    globalThis.setTimeout = originalSetTimeout;
    globalThis.clearTimeout = originalClearTimeout;
  }
});

console.log(`${count} capture adapter tests passed (CPU mocks, not real-browser validation).`);
