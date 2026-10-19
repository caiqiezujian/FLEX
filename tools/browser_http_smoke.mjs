#!/usr/bin/env node
/**
 * Real-browser HTTP file-capture regression against a running MOCK server.
 *
 *   python run.py --backend mock --host 127.0.0.1 --port 8000 --no-log
 *   node tools/browser_http_smoke.mjs --base-url http://127.0.0.1:8000
 *
 * A short, synthetic PCM WAV is generated unless --fixture is supplied.
 * Optional environment variables: FLEX_SMOKE_BASE_URL, FLEX_SMOKE_FIXTURE,
 * FLEX_ACCESS_TOKEN, FLEX_PLAYWRIGHT_MODULE (absolute package entry/directory),
 * FLEX_BROWSER_EXECUTABLE (installed Chrome/Edge path).
 * No browser security exemptions are used. This only tests the mock system,
 * not model quality, a GPU, or microphone hardware.
 */
import assert from 'node:assert/strict';
import { mkdtemp, stat, unlink, rmdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: {
  'base-url': { type: 'string', default: process.env.FLEX_SMOKE_BASE_URL || 'http://127.0.0.1:8000' },
  fixture: { type: 'string', default: process.env.FLEX_SMOKE_FIXTURE },
  'timeout-ms': { type: 'string', default: '90000' },
  help: { type: 'boolean', default: false },
} });
if (values.help) {
  console.log('Usage: node tools/browser_http_smoke.mjs [--base-url http://127.0.0.1:8000] [--fixture /path/to/media] [--timeout-ms 90000]');
  process.exit(0);
}

const base = new URL(values['base-url']);
assert.equal(base.protocol, 'http:', 'The smoke server must use HTTP.');
assert(['127.0.0.1', 'localhost'].includes(base.hostname), 'Only a local mock server is allowed.');
assert(!base.username && !base.password && base.pathname === '/' && !base.search && !base.hash,
  'Provide an origin without credentials, path, query, or fragment.');
const timeoutMs = Number(values['timeout-ms']);
assert(Number.isFinite(timeoutMs) && timeoutMs >= 5000, '--timeout-ms must be at least 5000.');
const unsafeOrigin = `http://flex-http.test${base.port ? `:${base.port}` : ''}`;
const safeOrigin = `http://localhost${base.port ? `:${base.port}` : ''}`;
const token = process.env.FLEX_ACCESS_TOKEN || '';
const RATE = 16000;
// Audio rendering uses blocks, not sample-accurate HTML media events. Report
// this fixed bound explicitly; never enlarge it automatically after a failure.
const DURATION_TOLERANCE_S = 0.05;
const SYNTHETIC_DURATION_S = 6.375;
let temporaryDirectory, fixture, browser;
const reports = [];

async function loadPlaywright() {
  const location = process.env.FLEX_PLAYWRIGHT_MODULE;
  if (!location) return import('playwright');
  assert(path.isAbsolute(location), 'FLEX_PLAYWRIGHT_MODULE must be an absolute path.');
  const target = (await stat(location)).isDirectory()
    ? createRequire(import.meta.url).resolve(location) : location;
  return import(pathToFileURL(target).href);
}

async function syntheticFixture() {
  temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'flex-http-smoke-'));
  const filename = path.join(temporaryDirectory, 'synthetic-markers.wav');
  const rate = 48000, samples = Math.round(rate * SYNTHETIC_DURATION_S);
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28);
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) {
    const seconds = i / rate;
    const hz = seconds < 0.25 ? 997 : seconds >= SYNTHETIC_DURATION_S - 0.25 ? 1733 : 440;
    wav.writeInt16LE(Math.round(0.45 * 32767 * Math.sin(2 * Math.PI * hz * seconds)), 44 + i * 2);
  }
  await writeFile(filename, wav, { flag: 'wx' });
  return filename;
}

async function until(predicate, description, milliseconds = timeoutMs) {
  const deadline = Date.now() + milliseconds;
  while (Date.now() < deadline) {
    const result = await predicate();
    if (result) return result;
    await delay(50);
  }
  throw new Error(`Timed out: ${description}`);
}

function observeSocket(socket, wire) {
  const connection = { url: socket.url(), sent: [], received: [], closed: false };
  wire.push(connection);
  socket.on('framesent', ({ payload }) => {
    if (typeof payload === 'string') {
      const data = JSON.parse(payload);
      if ('token' in data) data.token = '[redacted]';
      connection.sent.push({ kind: 'control', data });
    } else {
      const bytes = Buffer.from(payload);
      assert(bytes.length >= 10 && (bytes.length - 8) % 2 === 0, 'Malformed PCM frame.');
      connection.sent.push({ kind: 'pcm', sequence: bytes.readUInt32LE(0),
        start: bytes.readUInt32LE(4), samples: bytes.subarray(8), length: (bytes.length - 8) / 2 });
    }
  });
  socket.on('framereceived', ({ payload }) => connection.received.push(JSON.parse(String(payload))));
  socket.on('close', () => { connection.closed = true; });
}

async function openPage(origin, secure) {
  const context = await browser.newContext();
  const page = await context.newPage(), wire = [], pageErrors = [];
  page.setDefaultTimeout(15000);
  page.on('pageerror', error => pageErrors.push(String(error)));
  page.on('websocket', socket => observeSocket(socket, wire));
  await page.addInitScript(() => {
    window.__flexSmokeCapture = { scriptNodes: 0, workletNodes: 0, scriptBuffers: [],
      firstCallbacks: [], mediaEvents: [], graphConnections: [] };
    let scriptContext;
    const originalConnect = window.AudioNode.prototype.connect;
    window.AudioNode.prototype.connect = function (...args) {
      const result = originalConnect.apply(this, args);
      if (window.__flexSmokeCapture.graphConnections.length < 16) {
        window.__flexSmokeCapture.graphConnections.push({ source: this.constructor.name,
          destination: args[0]?.constructor.name, contextTime: this.context.currentTime,
          mediaTime: document.getElementById('media')?.currentTime });
      }
      return result;
    };
    for (const type of ['playing', 'pause', 'waiting', 'ended']) document.addEventListener(type, event => {
      if (event.target.id === 'media') window.__flexSmokeCapture.mediaEvents.push({ type,
        mediaTime: event.target.currentTime, contextTime: scriptContext?.currentTime });
    }, true);
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass?.prototype.createScriptProcessor) {
      const create = AudioContextClass.prototype.createScriptProcessor;
      AudioContextClass.prototype.createScriptProcessor = function (...args) {
        const node = create.apply(this, args);
        window.__flexSmokeCapture.scriptNodes++;
        window.__flexSmokeCapture.scriptBuffers.push(node.bufferSize);
        scriptContext = this;
        node.addEventListener('audioprocess', event => {
          if (window.__flexSmokeCapture.firstCallbacks.length < 8) {
            window.__flexSmokeCapture.firstCallbacks.push({ playbackTime: event.playbackTime,
              contextTime: this.currentTime, rate: this.sampleRate, size: event.inputBuffer.length,
              mediaTime: document.getElementById('media')?.currentTime,
              firstNonzero: event.inputBuffer.getChannelData(0).findIndex(value => Math.abs(value) > 0.0001) });
          }
        });
        return node;
      };
    }
    if (window.AudioWorkletNode) {
      const Original = window.AudioWorkletNode;
      window.AudioWorkletNode = class extends Original {
        constructor(...args) { super(...args); window.__flexSmokeCapture.workletNodes++; }
      };
    }
  });
  await page.goto(origin, { waitUntil: 'networkidle' });
  const configuration = await page.evaluate(async () => (await fetch('/api/config')).json());
  assert.equal(configuration.backend, 'mock', 'Refusing to run this test against a real model.');
  assert.equal(configuration.status, 'ready', 'The mock server must be ready.');
  assert.equal(configuration.busy, false, 'The server already has an active session.');
  const capabilities = await page.evaluate(async () => {
    const audio = new (window.AudioContext || window.webkitAudioContext)();
    const result = { secure: window.isSecureContext, worklet: Boolean(audio.audioWorklet),
      getUserMedia: Boolean(navigator.mediaDevices?.getUserMedia) };
    await audio.close();
    return result;
  });
  assert.equal(capabilities.secure, secure, `Unexpected secure-context classification for ${origin}.`);
  assert.equal(capabilities.worklet, secure, 'AudioWorklet availability does not match the expected context.');
  if (!secure) assert.equal(capabilities.getUserMedia, false, 'HTTP test unexpectedly exposes microphone capture.');
  await page.locator('#settings summary').click();
  await page.locator('#chunkSeconds').selectOption('1');
  await page.locator('#adaptive').uncheck();
  if (configuration.auth_required) {
    assert(token, 'This server needs FLEX_ACCESS_TOKEN for the smoke test.');
    await page.locator('#authToken').fill(token);
  }
  await page.locator('#settings summary').click();
  return { context, page, wire, pageErrors, secure };
}

async function chooseAndStart(test) {
  await test.page.locator('#fileInput').setInputFiles(fixture);
  await test.page.waitForFunction(() => {
    const media = document.getElementById('media');
    return media.readyState >= 1 && Number.isFinite(media.duration);
  });
  const duration = await test.page.locator('#media').evaluate(media => media.duration);
  assert(duration >= 4, 'Use a fixture lasting at least four seconds.');
  await test.page.locator('#startButton').click();
  await until(async () => {
    const error = await visibleError(test.page);
    if (error) throw new Error(error);
    return test.wire[0]?.received.some(event => event.type === 'ready');
  }, 'server ready / capture initialization');
  return duration;
}

async function visibleError(page) {
  return page.locator('#errorBox').evaluate(box => box.hidden ? '' :
    `${document.getElementById('errorTitle').textContent}: ${document.getElementById('errorMessage').textContent}`);
}

async function assertSessionHealthy(test) {
  assert.deepEqual(test.pageErrors, [], 'Uncaught browser errors.');
  const fatal = test.wire[0]?.received.find(event => event.type === 'error');
  if (fatal) throw new Error(JSON.stringify(fatal));
  const stopped = await test.page.locator('#sessionStatus').evaluate(element => element.classList.contains('error'));
  if (stopped) throw new Error(await visibleError(test.page) || 'Frontend session stopped with an error.');
  // A final WAIT is a normal warning in #errorBox, not a failed session.
}

async function waitForMediaTime(test, seconds) {
  await until(async () => {
    await assertSessionHealthy(test);
    const result = await test.page.locator('#media').evaluate(media => ({ time: media.currentTime, ended: media.ended }));
    assert(!result.ended || result.time >= seconds, 'Media ended before the requested checkpoint.');
    return result.time >= seconds;
  }, `media time ${seconds}s`);
}

function pcmFrames(test) { return (test.wire[0]?.sent || []).filter(frame => frame.kind === 'pcm'); }
function sentSamples(test) { return pcmFrames(test).reduce((sum, frame) => sum + frame.length, 0); }

function measuredFrequency(samples) {
  let crossings = 0;
  for (let i = 1; i < samples.length; i++) if (samples[i - 1] <= 0 && samples[i] > 0) crossings++;
  return crossings * RATE / samples.length;
}

async function verifyFinished(test, name, expectedDuration, markerCheck = false) {
  await until(async () => {
    await assertSessionHealthy(test);
    return test.wire[0]?.received.some(event => event.type === 'done');
  }, `${name}: done`);
  await until(async () => { await assertSessionHealthy(test); return test.wire[0]?.closed; }, `${name}: WebSocket close`);
  assert.deepEqual(test.pageErrors, [], 'Uncaught browser errors.');
  const wire = test.wire[0], frames = pcmFrames(test);
  assert.equal(test.wire.length, 1, 'A session must use exactly one connection.');
  assert(frames.length > 0, 'No PCM was sent.');
  let total = 0;
  for (let i = 0; i < frames.length; i++) {
    assert.equal(frames[i].sequence, i, 'Non-contiguous PCM sequence.');
    assert.equal(frames[i].start, total, 'Repeated or skipped source sample offsets.');
    assert(frames[i].length > 0 && frames[i].length <= 1600, 'Expected nonempty frames of at most 100 ms.');
    total += frames[i].length;
  }
  const finish = wire.sent.findIndex(frame => frame.kind === 'control' && frame.data.type === 'finish');
  assert(finish > 0, 'Missing finish control.');
  assert(!wire.sent.slice(finish + 1).some(frame => frame.kind === 'pcm'), 'PCM sent after finish.');
  assert.equal(wire.sent.filter(frame => frame.kind === 'control' && frame.data.type === 'finish').length, 1,
    'Duplicate finish control.');
  const fatal = wire.received.filter(event => event.type === 'error');
  assert.deepEqual(fatal, [], 'Server rejected the stream.');
  const done = wire.received.find(event => event.type === 'done');
  assert.equal(done.received_audio_s, total / RATE, 'Server / client audio counts differ.');
  assert.equal(done.processed_audio_s, total / RATE, 'Valid tail audio was not processed.');
  const turns = wire.received.filter(event => event.type === 'turn');
  assert(turns.length > 0, 'No inference turns returned.');
  for (let i = 0; i < turns.length; i++) {
    assert.equal(turns[i].source_start_s, i ? turns[i - 1].source_end_s : 0, 'Non-contiguous inference intervals.');
  }
  assert.equal(turns.at(-1).source_end_s, total / RATE, 'Last turn did not include the valid tail.');
  const transcript = await test.page.locator('#transcript').textContent();
  assert(transcript, 'No translated text was displayed.');
  assert(!/<\|(?:WAIT|FLEX|DIRECT)\|>/.test(transcript), 'Protocol labels leaked into the transcript.');
  assert.equal(transcript, turns.map(turn => turn.delta).join(''), 'Transcript is not exact append-only model deltas.');
  const capture = await test.page.evaluate(() => window.__flexSmokeCapture);
  assert.equal(capture.scriptNodes > 0, !test.secure, 'Wrong capture backend for the origin.');
  assert.equal(capture.workletNodes > 0, test.secure, 'Wrong capture backend for the origin.');
  const errorSeconds = total / RATE - expectedDuration;
  assert(Math.abs(errorSeconds) <= DURATION_TOLERANCE_S,
    `${name}: sent ${total / RATE}s for ${expectedDuration}s of played audio; error ${errorSeconds}s exceeds the fixed ${DURATION_TOLERANCE_S}s bound.`);
  if (markerCheck) {
    const bytes = Buffer.concat(frames.map(frame => frame.samples));
    const pcm = Array.from({ length: bytes.length / 2 }, (_, i) => bytes.readInt16LE(i * 2));
    const firstHz = measuredFrequency(pcm.slice(320, 2720));
    const lastHz = measuredFrequency(pcm.slice(-2720, -320));
    assert(Math.abs(firstHz - 997) < 40, `Leading audio marker lost: ${firstHz} Hz.`);
    assert(Math.abs(lastHz - 1733) < 40, `Trailing audio marker lost: ${lastHz} Hz.`);
  }
  const report = { test: name, origin: new URL(test.page.url()).origin, secure: test.secure,
    capture: { script_nodes: capture.scriptNodes, worklet_nodes: capture.workletNodes, script_buffers: capture.scriptBuffers },
    frames: frames.length, samples: total, sent_seconds: total / RATE,
    played_seconds: expectedDuration, error_seconds: errorSeconds, fixed_tolerance_seconds: DURATION_TOLERANCE_S,
    last_frame_samples: frames.at(-1).length,
    last_turn_seconds: turns.at(-1).source_end_s - turns.at(-1).source_start_s,
    turns: turns.length };
  reports.push(report); console.log(`PASS ${JSON.stringify(report)}`);
}

async function runFileTest(origin, secure, manual) {
  const test = await openPage(origin, secure);
  try {
    const duration = await chooseAndStart(test);
    if (manual) {
      await waitForMediaTime(test, 1.15);
      await test.page.locator('#pauseButton').click();
      await test.page.waitForFunction(() => document.getElementById('media').paused);
      if (manual !== 'pause-finish') {
        // Allow an already queued render block to settle, then prove no advancing
        // source or fabricated silence while playback remains paused.
        await delay(150);
        const before = { samples: sentSamples(test), time: await test.page.locator('#media').evaluate(media => media.currentTime) };
        await delay(450);
        assert.equal(sentSamples(test), before.samples, 'PCM advanced during a stable pause.');
        assert.equal(await test.page.locator('#media').evaluate(media => media.currentTime), before.time,
          'Media clock advanced while paused.');
        await test.page.locator('#pauseButton').click();
        await waitForMediaTime(test, 3.27);
      }
      await test.page.locator('#finishButton').click();
      const end = await test.page.locator('#media').evaluate(media => media.currentTime);
      assert(end < duration, 'Manual finish accidentally played the whole fixture.');
      await verifyFinished(test, manual === 'pause-finish' ? 'HTTP file pause/immediate finish'
        : 'HTTP file pause/resume/manual finish', end);
    } else {
      await verifyFinished(test, secure ? 'localhost AudioWorklet natural EOF' : 'HTTP file natural EOF', duration, !values.fixture);
    }
  } catch (error) {
    const diagnostics = { location: test.page.url(), status: await test.page.locator('#sessionStatus').textContent(),
      visible_error: await visibleError(test.page), samples: sentSamples(test), page_errors: test.pageErrors,
      capture: await test.page.evaluate(() => window.__flexSmokeCapture) };
    console.error(`FAIL context ${JSON.stringify(diagnostics)}`);
    throw error;
  } finally { await test.context.close(); }
}

async function runHttpMicrophoneTest() {
  const test = await openPage(unsafeOrigin, false);
  try {
    if (await test.page.locator('#micMode').isEnabled()) {
      await test.page.locator('#micMode').click();
      if (await test.page.locator('#startButton').isEnabled()) await test.page.locator('#startButton').click();
      await until(async () => /HTTPS|安全上下文|localhost/.test(await visibleError(test.page)), 'HTTP microphone explanation');
    } else {
      const text = await test.page.locator('#micMode').getAttribute('title') || await test.page.locator('#micDescription').textContent();
      assert(/HTTPS|安全上下文|localhost/.test(text), 'Disabled microphone must explain the secure-context requirement.');
    }
    assert.equal(test.wire.length, 0, 'Insecure microphone mode must not start a model session.');
    assert.deepEqual(test.pageErrors, [], 'Microphone denial caused an uncaught browser error.');
    reports.push({ test: 'HTTP microphone rejected without model session', secure: false });
    console.log('PASS HTTP microphone rejected without model session');
  } finally { await test.context.close(); }
}

async function runRepeatedSessions() {
  const test = await openPage(unsafeOrigin, false);
  try {
    for (let round = 1; round <= 2; round++) {
      if (round > 1) {
        await until(() => test.page.locator('#startButton').isEnabled(), 'previous session cleanup');
        test.wire.splice(0);
      }
      await chooseAndStart(test);
      await waitForMediaTime(test, 1.27);
      await test.page.locator('#finishButton').click();
      const end = await test.page.locator('#media').evaluate(media => media.currentTime);
      await verifyFinished(test, `HTTP same-page session ${round}`, end);
      const capture = await test.page.evaluate(() => window.__flexSmokeCapture);
      assert.equal(capture.scriptNodes, round, 'Restart did not create exactly one new capture adapter.');
    }
  } catch (error) {
    console.error(`FAIL repeat context ${JSON.stringify({ visible_error: await visibleError(test.page),
      samples: sentSamples(test), capture: await test.page.evaluate(() => window.__flexSmokeCapture) })}`);
    throw error;
  } finally { await test.context.close(); }
}

async function runMainThreadStallTest() {
  const test = await openPage(unsafeOrigin, false);
  try {
    await chooseAndStart(test);
    await waitForMediaTime(test, 1.1);
    // Deliberately starve callbacks. Correct behavior is an explicit stop, not
    // silent skipping/duplication of buffers overwritten during the stall.
    await test.page.evaluate(() => {
      const deadline = performance.now() + 400;
      while (performance.now() < deadline) { /* deliberate test-only stall */ }
    });
    await until(() => test.page.locator('#sessionStatus').evaluate(element => element.classList.contains('error')),
      'explicit main-thread-stall failure', 5000);
    const error = await visibleError(test.page);
    assert(/主线程|音频中断|回调不连续/.test(error), `Unexpected stall error: ${error}`);
    await until(() => test.wire[0]?.closed, 'stalled session socket cleanup', 5000);
    const total = sentSamples(test);
    await delay(250);
    assert.equal(sentSamples(test), total, 'PCM continued after the detected stall.');
    assert.deepEqual(test.pageErrors, [], 'Stall handling caused an uncaught browser error.');
    assert(test.wire[0].sent.some(frame => frame.kind === 'control' && frame.data.type === 'cancel'),
      'Stalled capture did not cancel its model session.');
    const report = { test: 'HTTP main-thread stall fails explicitly', sent_samples: total, message: error };
    reports.push(report); console.log(`PASS ${JSON.stringify(report)}`);
  } finally { await test.context.close(); }
}

try {
  fixture = values.fixture ? path.resolve(values.fixture) : await syntheticFixture();
  assert((await stat(fixture)).isFile(), 'Fixture must be a local file.');
  const config = await (await fetch(new URL('/api/config', base))).json();
  assert.equal(config.backend, 'mock', 'Refusing to run this test against a real model.');
  const module = await loadPlaywright(), chromium = module.chromium || module.default?.chromium;
  assert(chromium, 'The Playwright module does not expose chromium.');
  browser = await chromium.launch({ headless: true,
    ...(process.env.FLEX_BROWSER_EXECUTABLE ? { executablePath: process.env.FLEX_BROWSER_EXECUTABLE } : {}),
    args: ['--host-resolver-rules=MAP flex-http.test 127.0.0.1', '--no-proxy-server'] });
  await runHttpMicrophoneTest();
  await runFileTest(unsafeOrigin, false, true);
  await runFileTest(unsafeOrigin, false, 'pause-finish');
  await runFileTest(unsafeOrigin, false, false);
  await runFileTest(safeOrigin, true, false);
  await runRepeatedSessions();
  await runMainThreadStallTest();
  console.log(`All ${reports.length} real-browser checks passed. This validates MOCK transport, not GPU translation.`);
} catch (error) {
  console.error(error.stack || String(error));
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  if (temporaryDirectory && fixture) {
    await unlink(fixture);
    await rmdir(temporaryDirectory);
  }
}
