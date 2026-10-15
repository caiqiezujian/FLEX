/** Run: node static/test_pcm.mjs (no npm install required). */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('./pcm-core.js',import.meta.url),'utf8');
const { StreamingResampler, floatToPCM16, makeAudioPacket } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
let count = 0;
function test(name, fn) {fn();count++;console.log(`PASS ${name}`);}
function run(rate, input, block = input.length || 1) {
  const r = new StreamingResampler(rate), result = [];
  for (let i=0;i<input.length;i+=block)result.push(...r.push(input.subarray(i,i+block)));
  result.push(...r.flush());return result;
}
for (const rate of [16000,32000,44100,48000,96000]) {
  test(`${rate} Hz preserves exact sample duration and block continuity`,() => {
    const input = Float32Array.from({length:rate+117},(_,i)=>Math.sin(i*.071));
    const large = run(rate,input), small = run(rate,input,128), uneven = run(rate,input,31);
    assert.equal(large.length,Math.ceil(input.length*16000/rate-1e-9));
    assert.deepEqual(large,small);assert.deepEqual(large,uneven);
    assert(large.every(Number.isFinite));
  });
}
test('one-second chunks have no duration growth',() => {
  const r = new StreamingResampler(44100);let total=0;
  for(let i=0;i<10;i++)total+=r.push(new Float32Array(44100)).length;
  total+=r.flush().length;assert.equal(total,160000);assert.deepEqual(r.flush(),[]);
  assert.throws(()=>r.push(new Float32Array(1)),/finished/);
});
test('empty and very short tails are exact',() => {
  for(const length of [0,1,2,3,4,127,129,1599,1601])assert.equal(run(48000,new Float32Array(length),7).length,Math.max(0,Math.ceil(length/3-1e-9)));
});
test('anti-alias filter attenuates 12 kHz before 48 kHz to 16 kHz conversion',() => {
  const rms = values => Math.sqrt(values.slice(100).reduce((sum,x)=>sum+x*x,0)/(values.length-100));
  const wave = hz => Float32Array.from({length:48000},(_,i)=>Math.sin(2*Math.PI*hz*i/48000));
  assert(rms(run(48000,wave(12000))) < .02);
  assert(rms(run(48000,wave(1000))) > .65);
});
test('PCM clamps finite values and uses signed full scale',() => {
  assert.equal(floatToPCM16(-1),-32768);assert.equal(floatToPCM16(1),32767);
  assert.equal(floatToPCM16(-10),-32768);assert.equal(floatToPCM16(10),32767);
  assert.equal(floatToPCM16(NaN),0);assert.equal(floatToPCM16(Infinity),0);
});
test('binary frame has little-endian sequence, offset, signed PCM',() => {
  const packet = makeAudioPacket(4,1600,Int16Array.from([-32768,0,32767]));const view = new DataView(packet);
  assert.equal(packet.byteLength,14);assert.equal(view.getUint32(0,true),4);assert.equal(view.getUint32(4,true),1600);
  assert.equal(view.getInt16(8,true),-32768);assert.equal(view.getInt16(12,true),32767);
  assert.throws(()=>makeAudioPacket(-1,0,[]),/bounds/);
  assert.throws(()=>makeAudioPacket(0,2**32,[]),/bounds/);
});
globalThis.__flexPCM = {StreamingResampler,floatToPCM16};
globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class {
  constructor() {this.port = {messages:[],postMessage(message){this.messages.push(message);}};}
};
let Processor;
globalThis.registerProcessor = (_,value) => {Processor = value;};
const workletSource = (await readFile(new URL('./capture-worklet.js',import.meta.url),'utf8')).replace(/^import[^\n]+\n/,'const { StreamingResampler, floatToPCM16 } = globalThis.__flexPCM;\n');
await import(`data:text/javascript;base64,${Buffer.from(workletSource).toString('base64')}`);
function command(processor,op,requestId=1) {processor.port.onmessage({data:{op,requestId}});}
function render(processor,frames=1) {
  const input = new Float32Array(128).fill(.2);
  for(let i=0;i<frames;i++)processor.process([[input]],[[new Float32Array(128),new Float32Array(128)]]);
}
function pcmCount(processor) {return processor.port.messages.filter(m=>m.type==='pcm').reduce((sum,m)=>sum+m.buffer.byteLength/2,0);}
test('worklet ignores idle/paused samples and emits precise active duration',() => {
  const p = new Processor({processorOptions:{sampleRate:16000,frameSamples:1600}});
  render(p,100);assert.equal(pcmCount(p),0);
  command(p,'active');render(p,375);assert.equal(pcmCount(p),16000);
  command(p,'pause');render(p,300);assert.equal(pcmCount(p),16000);
  command(p,'active');render(p,375);command(p,'finish');assert.equal(pcmCount(p),32000);
  render(p,100);command(p,'active');render(p,100);assert.equal(pcmCount(p),32000);
});
test('worklet EOF emits short tail before acknowledgement exactly once',() => {
  const p = new Processor({processorOptions:{sampleRate:16000,frameSamples:1600}});
  command(p,'active');render(p,1);assert.equal(pcmCount(p),0);
  command(p,'finish',99);assert.equal(pcmCount(p),43);
  assert.deepEqual(p.port.messages.at(-1),{type:'ack',requestId:99});
  command(p,'finish');assert.equal(pcmCount(p),43);
});
test('worklet cancellation never flushes unsubmitted partial audio',() => {
  const p = new Processor({processorOptions:{sampleRate:16000,frameSamples:1600}});
  command(p,'active');render(p,1);command(p,'cancel');command(p,'finish');
  assert.equal(pcmCount(p),0);
});
console.log(`${count} PCM / AudioWorklet tests passed.`);
