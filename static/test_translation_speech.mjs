import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const {TranslationSpeech, splitSpeech} = await import('data:text/javascript;base64,' +
  readFileSync(new URL('./translation-speech.js', import.meta.url)).toString('base64'));
const drain = async () => {for (let i=0;i<30;i++) await Promise.resolve();};
const audioResponse = (marker=1) => ({ok:true,arrayBuffer:async()=>Uint8Array.of(marker).buffer});
const errorResponse = (status,code,message='Voice service is unavailable') =>
  ({ok:false,status,json:async()=>({detail:{code,message}})});

function fixture({fetcher, ...options} = {}) {
  const calls=[], states=[], nodes=[], timers=new Map(); let now=0, id=0;
  const context={state:'running', destination:{}, resume:async()=>{},
    createGain:()=>({gain:{value:1},connect(){}}),
    decodeAudioData:async()=>({duration:1}),
    createBufferSource(){const node={connect(){},disconnect(){},start(){node.started=true;},stop(){node.stopped=true;}};nodes.push(node);return node;}};
  const player = new TranslationSpeech({contextFactory:()=>context, onState:s=>states.push(s),maxConcurrent:1,
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

test('short unpunctuated deltas request immediately and EOF never repeats text',async()=>{
  const f=fixture();await f.player.start({language:'zh'});
  f.player.append('大学教授可以');
  assert.equal(f.calls.length,1);assert.equal(f.calls[0].text,'大学教授可以');
  f.player.append('为学生创作内容');
  await drain();assert.equal(f.calls.length,2);assert.equal(f.calls[1].text,'为学生创作内容');
  f.player.append('，这是最后一段');f.player.finish();await drain();
  assert.deepEqual(f.calls.map(c=>c.text),['大学教授可以','为学生创作内容','，这是最后一段']);
  f.player.finish();await drain();assert.equal(f.calls.length,3);
  assert.equal(f.nodes.length,1);f.nodes[0].onended();await drain();assert.equal(f.nodes.length,2);
  f.nodes[1].onended();await drain();assert.equal(f.nodes.length,3);
  f.nodes[2].onended();assert.equal(f.states.at(-1).state,'done');
});

test('text arriving during synthesis coalesces only unsent segments without delaying an idle request',async()=>{
  const resolves=[];
  const f=fixture({fetcher:(url,opts)=>{f.calls.push(JSON.parse(opts.body));return new Promise(resolve=>resolves.push(resolve));}});
  await f.player.start({language:'zh'});
  f.player.append('现在');assert.equal(f.calls.length,1);
  f.tick(1000);f.player.append('大学教授可以');
  f.tick(1000);f.player.append('为学生创作内容');f.player.append('，');
  assert.equal(f.calls[0].text,'现在');
  assert.deepEqual(f.player.run.pending,[{text:'大学教授可以为学生创作内容，',created:1000}]);
  resolves[0]({ok:true,arrayBuffer:async()=>new ArrayBuffer(2)});await drain();
  assert.equal(f.calls.length,2);assert.equal(f.calls[1].text,'大学教授可以为学生创作内容，');
  resolves[1]({ok:true,arrayBuffer:async()=>new ArrayBuffer(2)});await drain();
  f.player.finish();assert.equal(f.calls.length,2);
  f.nodes[0].onended();await drain();
  f.nodes[1].onended();assert.equal(f.states.at(-1).state,'done');
});

test('prefetch is bounded and playback is strictly ordered without overlapping sources',async()=>{
  const f=fixture();await f.player.start({language:'en',maximum:40});
  for(let i=0;i<6;i++) f.player.append(`This is sentence number ${i}. `);
  await drain();assert.equal(f.nodes.length,1);assert.equal(f.calls.length,3);
  assert.equal(f.player.run.ready.length,2);assert.equal(f.player.run.pending.length,3);
  f.nodes[0].onended();await drain();assert.equal(f.nodes.length,2);assert.equal(f.calls.length,4);
  assert.match(f.calls[3].text,/number 3/);f.player.stop();
});

test('default concurrency requests B and C during A and preserves FIFO when C returns first',async()=>{
  const resolves=[];let active=0,maximumActive=0;
  const f=fixture({maxConcurrent:undefined,fetcher:(url,opts)=>{
    f.calls.push({...JSON.parse(opts.body),signal:opts.signal});
    active++;maximumActive=Math.max(maximumActive,active);
    return new Promise(resolve=>resolves.push(resolve)).finally(()=>{active--;});
  }});
  f.context.decodeAudioData=async bytes=>({duration:1,marker:new Uint8Array(bytes)[0]});
  await f.player.start({language:'en',maximum:30});
  const segments=['First sentence A. ','Second sentence B. ','Third sentence C. ',
    'Fourth sentence D. ','Fifth sentence E. '];
  f.player.append(segments[0]);resolves[0](audioResponse(1));await drain();
  assert.equal(f.nodes.length,1);assert(f.nodes[0].started);
  for(const segment of segments.slice(1)) f.player.append(segment);
  f.player.finish();await drain();
  assert.equal(f.calls.length,3);assert.equal(active,2);assert.equal(f.nodes.length,1);
  assert.deepEqual(f.calls.map(call=>call.text),segments.slice(0,3));
  resolves[2](audioResponse(3));await drain();
  assert.equal(f.calls.length,3);assert.equal(active,1);assert.equal(f.nodes.length,1);
  f.nodes[0].onended();await drain();
  assert.equal(f.nodes.length,1,'C must wait for the missing B even after A ends');
  assert.equal(f.calls.length,3,'C and the unfinished B occupy both prefetch slots');
  resolves[1](audioResponse(2));await drain();
  assert.deepEqual(f.nodes.map(node=>node.buffer.marker),[1,2]);
  assert.equal(f.calls.length,4);
  resolves[3](audioResponse(4));await drain();
  f.nodes[1].onended();await drain();
  assert.deepEqual(f.nodes.map(node=>node.buffer.marker),[1,2,3]);
  assert.equal(f.calls.length,5);
  resolves[4](audioResponse(5));await drain();
  f.nodes[2].onended();await drain();f.nodes[3].onended();await drain();
  assert.deepEqual(f.nodes.map(node=>node.buffer.marker),[1,2,3,4,5]);
  f.nodes[4].onended();await drain();
  assert.equal(f.states.at(-1).state,'done');assert.equal(maximumActive,2);
  assert.deepEqual(f.calls.map(call=>call.text),segments);f.player.stop();
});

test('ready audio exceeding 30 seconds pauses admission and never stops current playback',async()=>{
  const f=fixture({maxConcurrent:undefined,fetcher:async(url,opts)=>{
    f.calls.push(JSON.parse(opts.body));return audioResponse(f.calls.length);
  }});
  f.context.decodeAudioData=async bytes=>({duration:16,marker:new Uint8Array(bytes)[0]});
  await f.player.start({language:'en',maximum:40});
  f.player.append('First sixteen second segment.');await drain();
  f.player.append('Second sixteen second segment.');
  f.player.append('Third sixteen second segment.');
  f.player.append('Fourth sixteen second segment.');f.player.finish();await drain();
  assert(f.player.run);assert.equal(f.calls.length,3);assert.equal(f.nodes.length,1);
  assert.equal(f.nodes[0].stopped,undefined);assert.equal(f.player.run.ready.length,2);
  assert(!f.states.some(state=>state.state==='error'));
  f.nodes[0].onended();await drain();
  assert.equal(f.calls.length,4);assert.deepEqual(f.nodes.map(node=>node.buffer.marker),[1,2]);
  f.nodes[1].onended();await drain();f.nodes[2].onended();await drain();
  assert.deepEqual(f.nodes.map(node=>node.buffer.marker),[1,2,3,4]);
  f.nodes[3].onended();await drain();
  assert.equal(f.states.at(-1).state,'done');assert(!f.nodes.some(node=>node.stopped));f.player.stop();
});

test('concurrency configuration is capped at two simultaneous requests',async()=>{
  const f=fixture({maxConcurrent:99,fetcher:(url,opts)=>{
    f.calls.push({...JSON.parse(opts.body),signal:opts.signal});return new Promise(()=>{});
  },maxPrefetch:5});
  await f.player.start({language:'en',maximum:40,maxConcurrent:99});
  for(let i=0;i<5;i++) f.player.append(`Sentence number ${i} waits for audio.`);
  await drain();assert.equal(f.calls.length,2);assert.equal(f.states.at(-1).inflight,2);
  f.player.stop();assert(f.calls.every(call=>call.signal.aborted));
});

test('stop aborts every concurrent request and old responses cannot enter a new run',async()=>{
  const resolves=[];
  const f=fixture({maxConcurrent:undefined,fetcher:(url,opts)=>{
    f.calls.push({...JSON.parse(opts.body),signal:opts.signal});
    return new Promise(resolve=>resolves.push(resolve));
  }});
  await f.player.start({language:'en'});
  f.player.append('First outstanding request.');f.player.append('Second outstanding request.');
  assert.equal(f.calls.length,2);
  f.player.stop();assert(f.calls.every(call=>call.signal.aborted));
  await f.player.start({language:'zh'});
  resolves[1](audioResponse(2));resolves[0](audioResponse(1));await drain();
  assert.equal(f.nodes.length,0);assert.equal(f.player.run.language,'zh');
  assert.equal(f.states.at(-1).received,0);f.player.stop();await drain();assert.equal(f.timers.size,0);
});

test('stop/new direction aborts outstanding work and ignores late old audio',async()=>{
  let resolve;
  const f=fixture({fetcher:(url,opts)=>new Promise(r=>{resolve=r;f.calls.push({...JSON.parse(opts.body),signal:opts.signal});})});
  await f.player.start({language:'zh'});f.player.append('这是一个足够长的句子，用来请求语音。');await drain();
  f.player.stop();assert(f.calls[0].signal.aborted);
  await f.player.start({language:'en'});resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(2)});await drain();
  assert.equal(f.nodes.length,0);assert.equal(f.player.run.language,'en');f.player.stop();
});

test('transport error stops only speech and cannot keep retrying indefinitely',async()=>{
  const f=fixture({fetcher:async()=>({ok:false,status:503,json:async()=>({detail:{message:'ENGINE_NOT_READY'}})})});
  await f.player.start({language:'en'});f.player.append('This is a complete sentence.');await drain();
  assert.equal(f.player.run,null);assert.match(f.states.at(-1).message,/ENGINE_NOT_READY/);
  assert.equal(f.states.at(-1).state,'error');
});

test('an explicit busy refusal retries the same segment before later text and plays each once',async()=>{
  const f=fixture({fetcher:async(url,opts)=>{
    f.calls.push({url,...JSON.parse(opts.body),signal:opts.signal});
    return f.calls.length===1 ? errorResponse(503,'tts_busy') : audioResponse(f.calls.length);
  }});
  f.context.decodeAudioData=async bytes=>({duration:1,marker:new Uint8Array(bytes)[0]});
  await f.player.start({language:'zh'});
  f.player.append('先播放这句。');await drain();
  assert.equal(f.states.at(-1).state,'retrying');
  assert.equal(f.states.at(-1).preview,false);
  assert([...f.timers.values()].some(timer=>timer.ms===1000));
  f.player.append('再播放这句。');f.player.finish();await drain();
  assert.equal(f.calls.length,1);assert.equal(f.nodes.length,0);
  f.tick(1000);await drain();
  assert.deepEqual(f.calls.map(({text,language})=>({text,language})),[
    {text:'先播放这句。',language:'zh'},
    {text:'先播放这句。',language:'zh'},
    {text:'再播放这句。',language:'zh'},
  ]);
  assert.equal(f.nodes.length,1);assert.equal(f.nodes[0].buffer.marker,2);
  f.nodes[0].onended();await drain();
  assert.equal(f.nodes.length,2);assert.equal(f.nodes[1].buffer.marker,3);
  f.nodes[1].onended();await drain();
  assert.equal(f.states.at(-1).state,'done');
  assert.equal(f.calls.length,3);assert.equal(f.nodes.length,2);f.player.stop();
});

test('busy refusals get only three retries with one, two, and four second waits',async()=>{
  const f=fixture({fetcher:async(url,opts)=>{
    f.calls.push(JSON.parse(opts.body));return errorResponse(503,'tts_busy','All workers are busy');
  }});
  await f.player.start({language:'en',preview:true});
  f.player.append('Read this preview exactly once.');f.player.finish();await drain();
  for(const [index,delay] of [1000,2000,4000].entries()) {
    assert.equal(f.calls.length,index+1);
    assert.equal(f.states.at(-1).state,'retrying');
    assert.equal(f.states.at(-1).preview,true);
    assert([...f.timers.values()].some(timer=>timer.ms===delay));
    f.tick(delay);await drain();
  }
  assert.equal(f.calls.length,4);
  assert(f.calls.every(call=>call.text==='Read this preview exactly once.' && call.language==='en'));
  assert.equal(f.player.run,null);assert.equal(f.nodes.length,0);
  assert.equal(f.states.at(-1).state,'error');assert.equal(f.states.at(-1).preview,true);
  assert.match(f.states.at(-1).message,/503/);assert.match(f.states.at(-1).message,/tts_busy/);
  assert.match(f.states.at(-1).message,/All workers are busy/);
  assert.equal(f.timers.size,0);f.tick(10000);await drain();assert.equal(f.calls.length,4);
});

test('stop during busy backoff aborts the request and cancels all later attempts',async()=>{
  const f=fixture({fetcher:async(url,opts)=>{
    f.calls.push({text:JSON.parse(opts.body).text,signal:opts.signal});return errorResponse(503,'tts_busy');
  }});
  await f.player.start({language:'en',preview:true});
  f.player.append('Stop this preview during backoff.');await drain();
  assert.equal(f.states.at(-1).state,'retrying');
  f.player.stop();await drain();
  assert(f.calls[0].signal.aborted);assert.equal(f.timers.size,0);
  f.tick(200000);await drain();
  assert.equal(f.calls.length,1);assert.equal(f.nodes.length,0);
  assert.equal(f.states.at(-1).state,'off');
  assert(!f.states.some(state=>state.state==='error'));
});

test('all failures other than HTTP 503 with the tts_busy code fail without retries',async t=>{
  for(const [status,code] of [[503,'engine_not_ready'],[503,undefined],[500,'tts_busy'],[429,'tts_busy']]) {
    await t.test(`HTTP ${status}, code ${code ?? 'absent'}`,async()=>{
      const f=fixture({fetcher:async(url,opts)=>{
        f.calls.push(JSON.parse(opts.body));return errorResponse(status,code,'Failure detail');
      }});
      await f.player.start({language:'en',preview:true});
      f.player.append('A preview that cannot be synthesized.');await drain();
      assert.equal(f.player.run,null);assert.equal(f.states.at(-1).state,'error');
      assert.equal(f.states.at(-1).preview,true);
      assert.match(f.states.at(-1).message,new RegExp(String(status)));
      if(code) assert(f.states.at(-1).message.includes(code));
      assert.match(f.states.at(-1).message,/Failure detail/);
      assert(!f.states.some(state=>state.state==='retrying'));
      f.tick(200000);await drain();assert.equal(f.calls.length,1);assert.equal(f.timers.size,0);
    });
  }
  await t.test('a network rejection',async()=>{
    let attempts=0;
    const f=fixture({fetcher:async()=>{attempts++;throw new Error('Connection closed');}});
    await f.player.start({language:'en'});f.player.append('This network request fails.');await drain();
    assert.equal(f.player.run,null);assert.match(f.states.at(-1).message,/Connection closed/);
    assert.equal(f.states.at(-1).preview,false);
    f.tick(200000);await drain();assert.equal(attempts,1);assert.equal(f.timers.size,0);
  });
});

test('live lag warns without aborting or dropping delayed audio',async()=>{
  let resolve;
  const f=fixture({fetcher:(url,opts)=>{
    f.calls.push({...JSON.parse(opts.body),signal:opts.signal});return new Promise(done=>{resolve=done;});
  }});
  await f.player.start({language:'en'});
  f.player.append('A whole sentence for a slow voice service.');f.player.finish();await drain();
  f.tick(31000);await drain();
  assert(f.player.run);assert.equal(f.calls[0].signal.aborted,false);
  assert.equal(f.states.at(-1).lagging,true);assert.match(f.states.at(-1).message,/积压/);
  assert(!f.states.some(state=>state.state==='error'));
  resolve(audioResponse());await drain();
  assert.equal(f.calls.length,1);assert.equal(f.nodes.length,1);assert(f.nodes[0].started);
  f.nodes[0].onended();assert.equal(f.states.at(-1).state,'done');f.player.stop();
});

test('text queue overflow still reports an explicit stop',async()=>{
  const g=fixture({maxQueue:2,fetcher:()=>new Promise(()=>{})});await g.player.start({language:'zh',maximum:40});
  for(let i=0;i<5;i++)g.player.append('这是一条不断增加的较长译文。');
  assert.equal(g.player.run,null);assert.match(g.states.at(-1).message,/积压/);
});

test('a preview response after 31 seconds still plays without a live lag warning',async()=>{
  let resolve;
  const f=fixture({fetcher:(url,opts)=>{
    f.calls.push({text:JSON.parse(opts.body).text,signal:opts.signal});
    return new Promise(done=>{resolve=done;});
  }});
  await f.player.start({language:'en',preview:true,timeoutSeconds:60});
  f.player.append('A voice preview may need a cold start.');f.player.finish();await drain();
  f.tick(31000);await drain();
  assert(f.player.run);assert.equal(f.calls[0].signal.aborted,false);
  assert.equal(f.states.at(-1).preview,true);assert.equal(f.states.at(-1).state,'synthesizing');
  assert.equal(f.states.at(-1).lagging,false);
  resolve(audioResponse());await drain();
  assert.equal(f.calls.length,1);assert.equal(f.nodes.length,1);assert(f.nodes[0].started);
  f.nodes[0].onended();await drain();
  assert.equal(f.states.at(-1).state,'done');assert.equal(f.states.at(-1).preview,true);
  assert(!f.states.some(state=>state.state==='error'));f.player.stop();
});

test('a preview still aborts hung synthesis at its configured HTTP deadline',async()=>{
  let aborted=false;
  const f=fixture({fetcher:(url,opts)=>{
    f.calls.push({text:JSON.parse(opts.body).text,signal:opts.signal});
    return new Promise((resolve,reject)=>opts.signal.addEventListener('abort',()=>{
      aborted=true;reject(new DOMException('Aborted','AbortError'));
    },{once:true}));
  }});
  await f.player.start({language:'en',preview:true,timeoutSeconds:40});
  f.player.append('This voice service never responds.');await drain();
  assert([...f.timers.values()].some(timer=>timer.ms===45000));
  f.tick(45000);await drain();
  assert(aborted);assert(f.calls[0].signal.aborted);
  assert.equal(f.player.run,null);assert.equal(f.nodes.length,0);assert.equal(f.calls.length,1);
  assert.equal(f.states.at(-1).state,'error');assert.equal(f.states.at(-1).preview,true);
  assert.match(f.states.at(-1).message,/合成超时/);assert.equal(f.timers.size,0);
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

test('synthesis runs during AudioContext unlock while playback waits for permission',async()=>{
  const f=fixture();let unlock;
  f.context.resume=()=>new Promise(resolve=>{unlock=resolve;});
  const starting=f.player.start({language:'en'});
  f.player.append('A full sentence arrives before the audio context is ready.');
  await drain();assert.equal(f.calls.length,1);assert.equal(f.nodes.length,0);
  assert.equal(f.states.at(-1).state,'unlocking');
  unlock();await starting;await drain();assert.equal(f.calls.length,1);
  assert.equal(f.nodes.length,1);f.player.stop();
});

test('preview state is reported synchronously before AudioContext resume resolves',async()=>{
  const f=fixture();let unlock,reportedBeforeResume=false;
  f.context.resume=()=>{
    reportedBeforeResume=f.states.at(-1)?.enabled===true && f.states.at(-1)?.preview===true;
    return new Promise(resolve=>{unlock=resolve;});
  };
  const starting=f.player.start({language:'en',preview:true});
  assert(reportedBeforeResume);
  assert.equal(f.states.at(-1).state,'unlocking');assert.equal(f.states.at(-1).preview,true);
  f.player.append('Preview starts after the browser allows audio.');await drain();
  assert.equal(f.calls.length,1);assert.equal(f.nodes.length,0);
  unlock();await starting;await drain();
  assert.equal(f.calls.length,1);assert.equal(f.nodes.length,1);
  assert(f.states.filter(state=>state.enabled).every(state=>state.preview===true));f.player.stop();
});

test('waiting fragments preserve spaces, unicode limits and original queue age',async()=>{
  const f=fixture({fetcher:()=>new Promise(()=>{})});await f.player.start({language:'en',maximum:40});
  f.player.append('First');f.tick(1000);
  for(const text of ['Good',' ','morning','，','界😀'.repeat(20)]) f.player.append(text);
  const pending=f.player.run.pending;
  assert.equal(pending.map(p=>p.text).join(''),'Good morning，'+'界😀'.repeat(20));
  assert(pending.every(p=>[...p.text].length<=40));
  assert.equal([...f.player.run.inflight.values()][0].text,'First');
  assert.equal(pending[0].created,1000);
  f.tick(1000);f.player.append('end');
  assert.equal(pending[0].created,1000);f.player.stop();
});

test('a full ready queue merges later fragments without waiting for new text',async()=>{
  const f=fixture();await f.player.start({language:'en'});
  for(const text of ['First.','Second.','Third.']) {f.player.append(text);await drain();}
  assert.equal(f.player.run.ready.length,2);assert.equal(f.calls.length,3);
  for(const text of ['Good',' ','morning']) f.player.append(text);
  assert.equal(f.calls.length,3);
  assert.equal(f.player.run.pending[0].text,'Good morning');
  f.nodes[0].onended();await drain();
  assert.equal(f.calls.length,4);assert.equal(f.calls[3].text,'Good morning');f.player.stop();
});

test('S2TT turn ranges and queue/request timings remain distinct through coalescing',async()=>{
  const responses=[];
  const f=fixture({fetcher:(url,opts)=>{
    f.calls.push({...JSON.parse(opts.body),signal:opts.signal});
    return new Promise(resolve=>responses.push(resolve));
  }});
  await f.player.start({language:'en'});
  f.player.append('Private first translation.',{turn:1,sourceEndSeconds:1});
  f.tick(1000);f.player.append('Private second ',{turn:2,sourceEndSeconds:2});
  f.player.append('and third translation.',{turn:4,sourceEndSeconds:4});
  f.tick(19000);responses[0](audioResponse());await drain();
  const requests=f.player.getDiagnostics().filter(event=>event.phase==='request_start');
  assert.deepEqual(requests.map(event=>[event.first_turn,event.last_turn]),[[1,1],[2,4]]);
  assert.equal(requests[1].queue_seconds,19);
  assert.match(f.states.at(-1).message,/最长 0 秒/);
  f.tick(2000);responses[1](audioResponse());await drain();
  f.nodes[0].onended();await drain();
  const ready=f.player.getDiagnostics().find(event=>event.phase==='audio_ready'&&event.segment===2);
  assert.equal(ready.request_seconds,2);
  const played=f.player.getDiagnostics().find(event=>event.phase==='play_start'&&event.segment===2);
  assert.equal(played.first_turn,2);assert.equal(played.last_turn,4);
  assert.equal(played.source_end_seconds,4);assert.equal(played.text_to_play_seconds,21);
  assert(!JSON.stringify(f.player.getDiagnostics()).includes('Private'));
  f.player.stop();
});

test('stalled browser unlock times out explicitly and never plays a ready buffer',async()=>{
  const f=fixture();f.context.resume=()=>new Promise(()=>{});
  const starting=f.player.start({language:'en',preview:true});
  f.player.append('Ready speech must still wait for browser permission.');
  await drain();assert.equal(f.calls.length,1);assert.equal(f.nodes.length,0);
  f.tick(10000);await starting;await drain();
  assert.equal(f.player.run,null);assert.equal(f.nodes.length,0);
  assert.match(f.states.at(-1).message,/播放授权等待超时/);assert.equal(f.timers.size,0);
});
