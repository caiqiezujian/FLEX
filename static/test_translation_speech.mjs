import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {TranslationSpeech, splitSpeech} = await import('data:text/javascript;base64,' +
  readFileSync(new URL('./translation-speech.js', import.meta.url)).toString('base64'));
const drain = async () => {for (let i=0;i<30;i++) await Promise.resolve();};

function fixture({fetcher, ...options} = {}) {
  const calls=[], states=[], nodes=[], timers=new Map(); let now=0, id=0;
  const context={state:'running', destination:{}, resume:async()=>{},
    createGain:()=>({gain:{value:1},connect(){}}),
    decodeAudioData:async()=>({duration:1}),
    createBufferSource(){const node={connect(){},disconnect(){},start(){node.started=true;},stop(){node.stopped=true;}};nodes.push(node);return node;}};
  const player = new TranslationSpeech({contextFactory:()=>context, onState:s=>states.push(s),
    fetcher:fetcher || (async(url,opts)=>{calls.push({url,...JSON.parse(opts.body),signal:opts.signal});return {ok:true,arrayBuffer:async()=>new ArrayBuffer(2)};}),
    clock:()=>now, setTimer:(fn,ms)=>{timers.set(++id,{fn,ms});return id;},clearTimer:id=>timers.delete(id), ...options});
  return {player, calls,states,nodes,context,timers,
    tick(ms){now+=ms;const callbacks=[...timers.entries()].filter(([,t])=>t.ms<=ms);for(const [key,t] of callbacks){timers.delete(key);t.fn();}}};
}

test('speech chunks preserve exact text without short fragment duplication', () => {
  for (const text of ['教授可以为学生创作内容，播客也可以为听众制作内容。你还可以使用视觉工具。',
    'Dr. Smith spoke at 3.14 p.m. today. This is the next complete sentence. ', '你😀'.repeat(250)]) {
    const {pieces,rest}=splitSpeech(text,40,true);
    assert.equal(pieces.join('')+rest,text);
    assert(pieces.every(p=>[...p].length<=40));
  }
  assert.equal(splitSpeech('大学教授可以').pieces.length,0);
  assert.equal(splitSpeech('University professor Dr. Smith can speak.').pieces[0], 'University professor Dr. Smith can speak.');
});

test('adjacent short deltas are buffered and EOF flushes once',async()=>{
  const f=fixture();await f.player.start({language:'zh'});
  f.player.append('大学教授可以');f.player.append('为学生创作内容');
  await drain();assert.equal(f.calls.length,0);
  f.tick(2100);await drain();assert.equal(f.calls[0].text,'大学教授可以为学生创作内容');
  f.player.append('，这是最后一段');f.player.finish();await drain();
  assert.deepEqual(f.calls.map(c=>c.text),['大学教授可以为学生创作内容','，这是最后一段']);
  f.player.finish();await drain();assert.equal(f.calls.length,2);
  assert.equal(f.nodes.length,1);f.nodes[0].onended();await drain();assert.equal(f.nodes.length,2);
  f.nodes[1].onended();assert.equal(f.states.at(-1).state,'done');
});

test('prefetch is bounded and playback is strictly ordered without overlapping sources',async()=>{
  const f=fixture();await f.player.start({language:'en'});
  for(let i=0;i<6;i++) f.player.append(`This is sentence number ${i}. `);
  await drain();assert.equal(f.nodes.length,1);assert.equal(f.calls.length,3);
  assert.equal(f.player.run.ready.length,2);assert.equal(f.player.run.pending.length,3);
  f.nodes[0].onended();await drain();assert.equal(f.nodes.length,2);assert.equal(f.calls.length,4);
  assert.match(f.calls[3].text,/number 3/);f.player.stop();
});

test('stop/new direction aborts outstanding work and ignores late old audio',async()=>{
  let resolve;
  const f=fixture({fetcher:(url,opts)=>new Promise(r=>{resolve=r;f.calls.push(JSON.parse(opts.body));})});
  await f.player.start({language:'zh'});f.player.append('这是一个足够长的句子，用来请求语音。');await drain();
  const old=f.player.run;f.player.stop();assert(old.abort.signal.aborted);
  await f.player.start({language:'en'});resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(2)});await drain();
  assert.equal(f.nodes.length,0);assert.equal(f.player.run.language,'en');f.player.stop();
});

test('transport error stops only speech and cannot keep retrying indefinitely',async()=>{
  const f=fixture({fetcher:async()=>({ok:false,status:503,json:async()=>({detail:{message:'ENGINE_NOT_READY'}})})});
  await f.player.start({language:'en'});f.player.append('This is a complete sentence.');await drain();
  assert.equal(f.player.run,null);assert.match(f.states.at(-1).message,/ENGINE_NOT_READY/);
  assert.equal(f.states.at(-1).state,'error');
});

test('bounded lag cancels hung synthesis and queue overflow reports an explicit stop',async()=>{
  const f=fixture({fetcher:()=>new Promise(()=>{})});await f.player.start({language:'en'});
  f.player.append('A whole sentence for a slow voice service.');await drain();
  f.tick(31000);assert.equal(f.player.run,null);assert.match(f.states.at(-1).message,/落后/);
  const g=fixture({maxQueue:2,fetcher:()=>new Promise(()=>{})});await g.player.start({language:'zh'});
  for(let i=0;i<5;i++)g.player.append('这是一条不断增加的较长译文。');
  assert.equal(g.player.run,null);assert.match(g.states.at(-1).message,/积压/);
});

test('punctuation-only increments are not sent to TTS',async()=>{
  const f=fixture();await f.player.start({language:'zh'});f.player.append('……...，。');f.player.finish();await drain();
  assert.equal(f.calls.length,0);assert.equal(f.states.at(-1).state,'done');
});

test('audio autoplay refusal is visible and volume is isolated from input capture',async()=>{
  const f=fixture();f.context.state='suspended';await f.player.start({language:'en'});
  assert.equal(f.player.run,null);assert.match(f.states.at(-1).message,/允许播放/);
  f.context.state='running';await f.player.start({language:'en'});f.player.setVolume(.4);
  assert.equal(f.player.gain.gain.value,.4);f.player.stop();
});

test('immediate translation waits for the user-gesture AudioContext resume to finish',async()=>{
  const f=fixture();let unlock;
  f.context.resume=()=>new Promise(resolve=>{unlock=resolve;});
  const starting=f.player.start({language:'en'});
  f.player.append('A full sentence arrives before the audio context is ready.');
  await drain();assert.equal(f.calls.length,0);
  unlock();await starting;await drain();assert.equal(f.calls.length,1);
  assert.equal(f.nodes.length,1);f.player.stop();
});
