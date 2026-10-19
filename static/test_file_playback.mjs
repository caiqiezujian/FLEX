/** File prefix clock tests: node static/test_file_playback.mjs; no npm install. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('./file-playback.js',import.meta.url),'utf8');
const { FilePlaybackClock } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
let count = 0;
function test(name, run) {run();count++;console.log(`PASS ${name}`);}
const clock = (extra = {}) => new FilePlaybackClock({offset:0,duration:6.375,browserDuration:6.375,...extra});

test('new session releases no audio before playback',() => {
  assert.equal(clock().read(0),0);
  assert.equal(clock({offset:2}).read(2),2);
});
test('regular positions release only the observed prefix',() => {
  const c = clock();
  for (const position of [.03,.2,.63,1.19,2.8]) assert.equal(c.read(position),position);
});
test('pause/waiting never advances on wall-clock time',() => {
  const c = clock();
  for (let i=0;i<100;i++) assert.equal(c.read(1.2345),1.2345);
});
test('a long main-thread stall catches up in one progress update',() => {
  const c = clock();
  assert.equal(c.read(.2),.2);
  assert.equal(c.read(4.9),4.9);
});
test('resume and sequential progress do not duplicate or reverse prefix',() => {
  const c = clock();
  assert.equal(c.read(2),2);
  assert.equal(c.read(1.999999),2);
  assert.equal(c.read(3),3);
});
test('manual finish close to EOF does not expose unheard tail',() => {
  const c = clock();
  assert.equal(c.read(6.374,false),6.374);
});
test('manual finish never promotes decoder duration disagreement',() => {
  const c = clock({duration:6.4});
  assert.equal(c.read(6.375,false),6.375);
});
test('natural EOF can reconcile small decoder/container differences',() => {
  const c = clock({duration:6.4});
  assert.equal(c.read(6.375,true),6.4);
});
test('false early EOF does not unlock future audio',() => {
  assert.equal(clock().read(3,true),3);
});
test('video tail after audio end cannot read beyond decoded samples',() => {
  const c = clock({duration:6.3});
  assert.equal(c.read(6.35),6.3);
  assert.equal(c.read(6.375,true),6.3);
});
test('starting part-way through a file preserves absolute positions',() => {
  const c = clock({offset:3.125});
  assert.equal(c.read(3),3.125);
  assert.equal(c.read(4),4);
});
test('each session has independent progression state',() => {
  assert.equal(clock().read(6),6);
  assert.equal(clock().read(.1),.1);
});
test('invalid durations and offsets fail before session playback',() => {
  for (const extra of [{duration:NaN},{duration:0},{browserDuration:Infinity},{offset:-1},{offset:6.375}]) assert.throws(() => clock(extra));
  assert.throws(() => clock().read(NaN));
});
test('significant track/container mismatch is not silently trimmed',() => {
  assert.throws(() => clock({duration:8}),/时长/);
});
test('large files cannot increase duration tolerance without bound',() => {
  assert.throws(() => clock({duration:1000.6,browserDuration:1000}),/时长/);
});

console.log(`\n${count} file playback tests passed.`);
