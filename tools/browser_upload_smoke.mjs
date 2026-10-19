#!/usr/bin/env node
/**
 * Real Chromium regression for uploaded-file simultaneous interpretation.
 *
 *   python run.py --backend mock --host 127.0.0.1 --port 8000 --no-log
 *   node tools/browser_upload_smoke.mjs --base-url http://127.0.0.1:8000
 *
 * Always generates a synthetic 6.375-second, 48 kHz WAV. --fixture optionally
 * adds a natural-EOF check for an existing MP4 or other browser-playable file.
 * Set FLEX_PLAYWRIGHT_MODULE / FLEX_BROWSER_EXECUTABLE for installed runtimes.
 * FLEX_ACCESS_TOKEN is used only when the local mock server requires it.
 * No insecure-context exemptions, external uploads, GPU/model, or microphone
 * hardware are used. File playback must construct no browser audio processor.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, stat, readFile, unlink, rmdir, writeFile } from 'node:fs/promises';
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
  console.log('Usage: node tools/browser_upload_smoke.mjs [--base-url http://127.0.0.1:8000] [--fixture /path/to/optional.mp4] [--timeout-ms 90000]');
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
const RATE = 16000, SYNTHETIC_DURATION_S = 6.375;
let temporaryDirectory, fixture, browser;
const reports = [];
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

async function loadPlaywright() {
  const location = process.env.FLEX_PLAYWRIGHT_MODULE;
  if (!location) return import('playwright');
  assert(path.isAbsolute(location), 'FLEX_PLAYWRIGHT_MODULE must be an absolute path.');
  const target = (await stat(location)).isDirectory()
    ? createRequire(import.meta.url).resolve(location) : location;
  return import(pathToFileURL(target).href);
}

async function syntheticFixture() {
  temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'flex-upload-smoke-'));
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
  const connection = { url: socket.url(), sent: [], received: [], closed: false, released: 0 };
  wire.push(connection);
  socket.on('framesent', ({ payload }) => {
    if (typeof payload !== 'string') {
      connection.sent.push({ kind: 'binary', bytes: payload.length });
      return;
    }
    const data = JSON.parse(payload);
    if ('token' in data) data.token = '[redacted]';
    if (data.type === 'start') connection.offset = data.media_offset_s || 0;
    if (['progress', 'finish'].includes(data.type) && Number.isFinite(data.position_s)) {
      connection.released = Math.max(connection.released, data.position_s - (connection.offset || 0));
    }
    connection.sent.push({ kind: 'control', data });
  });
  socket.on('framereceived', ({ payload }) => {
    const data = JSON.parse(String(payload));
    connection.received.push({ ...data, test_released_at_receive_s: connection.released });
  });
  socket.on('close', () => { connection.closed = true; });
}

async function openPage(origin, secure) {
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage(), wire = [], pageErrors = [], uploads = [];
  page.setDefaultTimeout(15000);
  page.on('pageerror', error => pageErrors.push(String(error)));
  page.on('websocket', socket => observeSocket(socket, wire));
  page.on('request', request => {
    if (request.method() !== 'POST' || new URL(request.url()).pathname !== '/api/media') return;
    const bytes = request.postDataBuffer();
    uploads.push({ request, length: bytes?.length || 0, hash: bytes ? sha256(bytes) : null,
      contentType: request.headers()['content-type'] });
  });
  // Chromium omits Blob-backed upload bodies from Network.requestWillBeSent,
  // so Playwright postDataBuffer() can be null despite an intact HTTP upload.
  // Fingerprint the exact Blob handed to XHR, and independently assert the
  // HTTP Content-Length plus the server's decoded sample accounting.
  await page.exposeFunction('__flexSmokeHashBytes', base64 => sha256(Buffer.from(base64, 'base64')));
  await page.addInitScript(() => {
    window.__flexSmokeCapture = { audioContexts: 0, scriptNodes: 0, workletNodes: 0 };
    window.__flexSmokeUploadBodies = [];
    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function (body) {
      if (body instanceof Blob) {
        const entry = { length: body.size, hash: null };
        window.__flexSmokeUploadBodies.push(entry);
        this.addEventListener('load', () => {
          entry.status = this.status;
          try { entry.response = JSON.parse(this.responseText); } catch (_) { entry.response = null; }
        }, { once: true });
        body.arrayBuffer().then(buffer => {
          const bytes = new Uint8Array(buffer), pieces = [];
          for (let offset = 0; offset < bytes.length; offset += 32768) {
            pieces.push(String.fromCharCode(...bytes.subarray(offset, offset + 32768)));
          }
          return window.__flexSmokeHashBytes(btoa(pieces.join('')));
        })
          .then(hash => { entry.hash = hash; });
      }
      return originalSend.call(this, body);
    };
    for (const name of ['AudioContext', 'webkitAudioContext']) {
      const Original = window[name];
      if (!Original) continue;
      window[name] = new Proxy(Original, { construct(target, args, newTarget) {
        window.__flexSmokeCapture.audioContexts++;
        return Reflect.construct(target, args, newTarget);
      } });
      if (Original.prototype.createScriptProcessor) {
        const create = Original.prototype.createScriptProcessor;
        Original.prototype.createScriptProcessor = function (...args) {
          window.__flexSmokeCapture.scriptNodes++;
          return create.apply(this, args);
        };
      }
    }
    if (window.AudioWorkletNode) {
      const Original = window.AudioWorkletNode;
      window.AudioWorkletNode = new Proxy(Original, { construct(target, args, newTarget) {
        window.__flexSmokeCapture.workletNodes++;
        return Reflect.construct(target, args, newTarget);
      } });
    }
  });
  await page.goto(origin, { waitUntil: 'networkidle' });
  const configuration = await page.evaluate(async () => (await fetch('/api/config')).json());
  assert.equal(configuration.backend, 'mock', 'Refusing to run against a real model.');
  assert.equal(configuration.status, 'ready', 'The mock server must be ready.');
  assert.equal(configuration.active_sessions ?? Number(configuration.busy), 0,
    'The server already has an active session; use an isolated mock for this test.');
  assert.equal(configuration.file_mode, 'uploaded', 'Server does not support uploaded files.');
  const capabilities = await page.evaluate(() => ({ secure: window.isSecureContext,
    getUserMedia: Boolean(navigator.mediaDevices?.getUserMedia) }));
  assert.equal(capabilities.secure, secure, `Unexpected secure-context classification for ${origin}.`);
  if (!secure) assert.equal(capabilities.getUserMedia, false, 'HTTP test unexpectedly exposes microphone capture.');
  await page.locator('#settings summary').click();
  await page.locator('#chunkSeconds').selectOption('1');
  if (configuration.auth_required) {
    assert(token, 'This server needs FLEX_ACCESS_TOKEN for the smoke test.');
    await page.locator('#authToken').fill(token);
  }
  await page.locator('#settings summary').click();
  return { context, page, wire, pageErrors, uploads, secure };
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
  // A final WAIT is a normal warning, not a failed session.
}

async function chooseAndStart(test, source = fixture) {
  test.source = source;
  await test.page.locator('#fileInput').setInputFiles(source);
  await test.page.waitForFunction(() => {
    const media = document.getElementById('media');
    return media.readyState >= 1 && Number.isFinite(media.duration);
  });
  const duration = await test.page.locator('#media').evaluate(media => media.duration);
  assert(duration >= 4, 'Use a fixture lasting at least four seconds.');
  await test.page.locator('#startButton').click();
  await until(async () => {
    await assertSessionHealthy(test);
    return test.wire[0]?.received.some(event => event.type === 'ready');
  }, 'file upload / server ready');
  assert.equal(test.uploads.length, 1, 'Exactly one complete upload is required per session.');
  const upload = test.uploads[0], original = await readFile(source);
  await until(() => test.page.evaluate(() => Boolean(window.__flexSmokeUploadBodies.at(-1)?.hash)),
    'fingerprint of the Blob sent to the upload request');
  const body = await test.page.evaluate(() => window.__flexSmokeUploadBodies.at(-1));
  assert.equal(body.length, original.length, 'The request did not upload the entire source file.');
  assert.equal(body.hash, sha256(original), 'The uploaded body differs from the selected file.');
  const headers = await upload.request.allHeaders();
  assert.equal(Number(headers['content-length']), original.length, 'HTTP upload Content-Length is incomplete.');
  if (upload.hash !== null) assert.equal(upload.hash, body.hash, 'Network request bytes differ from the XHR Blob.');
  upload.length = body.length;
  upload.hash = body.hash;
  assert.equal(upload.contentType, 'application/octet-stream', 'File upload must use the raw request body.');
  const response = await upload.request.response();
  assert(response.ok(), `Upload failed with HTTP ${response.status()}.`);
  // Read the XHR-observed JSON: Chromium can evict upload response bodies
  // from its inspector cache before Playwright response.json() retrieves them.
  assert.equal(body.status, response.status());
  test.prepared = body.response;
  assert.equal(test.prepared.sample_rate, RATE);
  assert.equal(test.prepared.duration_s, test.prepared.samples / RATE);
  if (source === fixture) assert.equal(test.prepared.samples, SYNTHETIC_DURATION_S * RATE,
    'The synthetic source must decode to exactly 102000 samples.');
  const start = test.wire[0].sent.find(frame => frame.data?.type === 'start').data;
  assert.equal(start.media_id, test.prepared.media_id, 'Start did not bind the uploaded media.');
  assert.equal(start.source_kind, 'file');
  assert.equal(start.adaptive, false, 'File chunk boundaries must not depend on browser scheduling.');
  const ready = test.wire[0].received.find(event => event.type === 'ready');
  assert.equal(ready.file_mode, 'uploaded');
  assert.equal(ready.media_id, test.prepared.media_id);
  return duration;
}

async function waitForMediaTime(test, seconds) {
  await until(async () => {
    await assertSessionHealthy(test);
    const result = await test.page.locator('#media').evaluate(media => ({ time: media.currentTime, ended: media.ended }));
    assert(!result.ended || result.time >= seconds, 'Media ended before the requested checkpoint.');
    return result.time >= seconds;
  }, `media time ${seconds}s`);
}

async function getExport(test, kind) {
  const pending = test.page.waitForEvent('download');
  await test.page.locator(kind === 'txt' ? '#exportTxt' : '#exportJson').click();
  const download = await pending;
  assert.equal(await download.failure(), null, 'Export download failed.');
  const filename = await download.path();
  return readFile(filename, 'utf8');
}

async function verifyFinished(test, name, naturalEnd) {
  await until(async () => {
    await assertSessionHealthy(test);
    return test.wire[0]?.received.some(event => event.type === 'done');
  }, `${name}: done`);
  await until(async () => { await assertSessionHealthy(test); return test.wire[0]?.closed; }, `${name}: close`);
  assert.equal(test.wire.length, 1, 'A session must use exactly one connection.');
  const wire = test.wire[0];
  assert.deepEqual(wire.sent.filter(frame => frame.kind === 'binary'), [], 'Uploaded file mode sent binary PCM.');
  const controls = wire.sent.map(frame => frame.data);
  assert(!controls.some(control => control.type === 'cancel'), 'A healthy file session was cancelled.');
  assert.equal(controls.filter(control => control.type === 'finish').length, 1, 'Missing or duplicate finish control.');
  const finishIndex = controls.findIndex(control => control.type === 'finish');
  assert(!controls.slice(finishIndex + 1).some(control => control.type === 'progress'), 'Progress sent after finish.');
  let position = 0;
  for (const control of controls.filter(control => ['progress', 'finish'].includes(control.type))) {
    assert(Number.isFinite(control.position_s) && control.position_s >= position, 'Playback progress regressed.');
    assert(control.position_s <= test.prepared.duration_s, 'Playback progress exceeded decoded audio.');
    position = control.position_s;
  }
  const done = wire.received.find(event => event.type === 'done');
  assert.equal(done.reason, 'finished');
  const totalSamples = Math.round(done.received_audio_s * RATE);
  assert.equal(done.processed_audio_s, totalSamples / RATE, 'Released audio tail was not fully processed.');
  assert(Math.abs(totalSamples - position * RATE) <= 1, 'Final sample count differs from released playback position.');
  if (naturalEnd) assert.equal(totalSamples, test.prepared.samples, 'Natural EOF lost or invented source samples.');
  else assert(totalSamples < test.prepared.samples, 'Manual finish consumed future file audio.');
  const turns = wire.received.filter(event => event.type === 'turn');
  assert(turns.length > 0, 'No inference turns returned.');
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    assert.equal(turn.source_start_s, i ? turns[i - 1].source_end_s : 0, 'Inference source intervals are not contiguous.');
    assert(turn.source_end_s <= turn.test_released_at_receive_s + 1 / RATE,
      `Model consumed future audio: ${turn.source_end_s} > played ${turn.test_released_at_receive_s}.`);
    assert(turn.source_end_s > turn.source_start_s, 'Empty source interval.');
    if (i < turns.length - 1) assert.equal(turn.source_end_s - turn.source_start_s, 1,
      'Nonfinal fixed chunk changed with browser scheduling.');
  }
  assert.equal(turns.at(-1).source_end_s, totalSamples / RATE);
  const transcript = await test.page.locator('#transcript').textContent();
  assert(transcript, 'No translated text was displayed.');
  assert(!/<\|(?:WAIT|FLEX|DIRECT)\|>/.test(transcript), 'Protocol labels leaked into the transcript.');
  assert.equal(transcript, turns.map(turn => turn.delta).join(''), 'Transcript is not exact append-only model deltas.');
  assert.equal(done.translation, transcript, 'Final server transcript differs from the UI.');
  assert.equal(await getExport(test, 'txt'), transcript, 'TXT export changed model deltas.');
  const exported = JSON.parse(await getExport(test, 'json'));
  assert.equal(exported.capture_mode, 'server-file');
  assert.equal(exported.translation, transcript);
  assert.equal(exported.error, null);
  const capture = await test.page.evaluate(() => window.__flexSmokeCapture);
  assert.deepEqual(capture, { audioContexts: 0, scriptNodes: 0, workletNodes: 0 },
    'File mode constructed browser audio capture machinery.');
  const report = { test: name, secure: test.secure, upload_bytes: test.uploads[0].length,
    decoded_samples: test.prepared.samples, processed_samples: totalSamples,
    progress_events: controls.filter(control => control.type === 'progress').length,
    turns: turns.length, tags: [...new Set(turns.map(turn => turn.tag))],
    last_turn_seconds: turns.at(-1).source_end_s - turns.at(-1).source_start_s,
    browser_audio_contexts: capture.audioContexts };
  if (naturalEnd && test.source === fixture) {
    assert.deepEqual([...new Set(turns.map(turn => turn.tag))].sort(), ['DIRECT', 'FLEX', 'WAIT'],
      'Synthetic natural-EOF check must exercise all three mock protocol labels.');
    assert.equal(report.last_turn_seconds, 0.375);
  }
  reports.push(report); console.log(`PASS ${JSON.stringify(report)}`);
}

async function diagnostics(test) {
  return { url: test.page.url(), visible_error: await visibleError(test.page), page_errors: test.pageErrors,
    status: await test.page.locator('#sessionStatus').textContent(),
    media: await test.page.locator('#media').evaluate(media => ({ time: media.currentTime,
      duration: media.duration, paused: media.paused, ended: media.ended })),
    wire: test.wire.map(connection => ({
      controls: connection.sent.filter(frame => frame.data?.type !== 'progress'),
      progress_count: connection.sent.filter(frame => frame.data?.type === 'progress').length,
      last_progress: connection.sent.filter(frame => frame.data?.type === 'progress').slice(-3),
      events: connection.received.filter(event => ['ready', 'error', 'done'].includes(event.type)),
      turns: connection.received.filter(event => event.type === 'turn').map(event => ({
        start: event.source_start_s, end: event.source_end_s, released: event.test_released_at_receive_s,
        tag: event.tag })), closed: connection.closed })),
    capture: await test.page.evaluate(() => window.__flexSmokeCapture) };
}

async function runFileTest({ secure = false, pause = false, immediateFinish = false, stall = 0, source = fixture } = {}) {
  const test = await openPage(secure ? safeOrigin : unsafeOrigin, secure);
  const name = stall ? `HTTP file survives ${stall}ms main-thread stall`
    : pause ? immediateFinish ? 'HTTP pause/immediate finish' : 'HTTP pause/resume/manual finish'
    : `${secure ? 'localhost' : 'HTTP'} ${source === fixture ? 'WAV' : 'optional media'} natural EOF`;
  try {
    await chooseAndStart(test, source);
    if (pause) {
      await waitForMediaTime(test, 1.15);
      await test.page.locator('#pauseButton').click();
      await test.page.waitForFunction(() => document.getElementById('media').paused);
      if (!immediateFinish) {
        await delay(250);
        const before = { released: test.wire[0].released,
          count: test.wire[0].sent.filter(frame => frame.data?.type === 'progress').length,
          time: await test.page.locator('#media').evaluate(media => media.currentTime) };
        await delay(500);
        assert.equal(test.wire[0].released, before.released, 'Audio prefix advanced while paused.');
        assert.equal(test.wire[0].sent.filter(frame => frame.data?.type === 'progress').length, before.count,
          'Unchanged playback kept sending progress updates.');
        assert.equal(await test.page.locator('#media').evaluate(media => media.currentTime), before.time);
        await until(() => test.wire[0].received.some(event => event.type === 'turn' &&
          event.source_end_s >= Math.floor(before.released)), 'already-released full chunks drain during pause');
        await test.page.locator('#pauseButton').click();
        await waitForMediaTime(test, 3.27);
      }
      await test.page.locator('#finishButton').click();
    }
    if (stall) {
      await waitForMediaTime(test, 1.1);
      await test.page.evaluate(milliseconds => {
        const deadline = performance.now() + milliseconds;
        while (performance.now() < deadline) { /* deliberate test-only main-thread stall */ }
      }, stall);
      await assertSessionHealthy(test);
    }
    await verifyFinished(test, name, !pause);
  } catch (error) {
    console.error(`FAIL context ${JSON.stringify(await diagnostics(test))}`);
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
    assert.equal(test.wire.length, 0, 'Insecure microphone mode started a model session.');
    assert.equal(test.uploads.length, 0, 'Microphone mode unexpectedly uploaded a file.');
    assert.deepEqual(test.pageErrors, []);
    reports.push({ test: 'HTTP microphone rejected without model session', secure: false });
    console.log('PASS HTTP microphone rejected without model session');
  } finally { await test.context.close(); }
}

async function runUploadErrorTests() {
  for (const [status, code, message] of [
    [503, 'decoder_unavailable', '测试：服务器尚未安装 FFmpeg。'],
    [401, 'unauthorized', '测试：访问令牌无效。'],
    [413, 'upload_limit', '测试：文件超过上传大小限制。'],
  ]) {
    const test = await openPage(unsafeOrigin, false);
    try {
      await test.page.route('**/api/media?*', route => route.fulfill({ status,
        contentType: 'application/json', body: JSON.stringify({ detail: { code, message } }) }));
      await test.page.locator('#fileInput').setInputFiles(fixture);
      await test.page.waitForFunction(() => document.getElementById('media').readyState >= 1);
      await test.page.locator('#startButton').click();
      await until(() => test.page.locator('#sessionStatus').evaluate(element => element.classList.contains('error')),
        `HTTP ${status} upload error shown`);
      const error = await visibleError(test.page);
      assert(error.includes(message), `Upload error lost actionable server text: ${error}`);
      assert(!error.includes('[object Object]'), 'Upload error rendered an object instead of its message.');
      assert.equal(test.wire.length, 0, 'Failed upload still started a model session.');
      assert.deepEqual(test.pageErrors, [], 'Upload failure caused an uncaught browser error.');
      assert.equal(await test.page.locator('#startButton').isEnabled(), true, 'Upload failure did not allow retry.');
      const report = { test: `HTTP ${status} upload failure is readable and retryable`, code };
      reports.push(report); console.log(`PASS ${JSON.stringify(report)}`);
    } finally { await test.context.close(); }
  }
}

async function runUploadCancelTest() {
  const test = await openPage(unsafeOrigin, false);
  let release;
  const held = new Promise(resolve => { release = resolve; });
  try {
    await test.page.route('**/api/media?*', async route => {
      await held;
      try { await route.abort('aborted'); } catch (_) { /* XHR cancellation may already have closed the request. */ }
    });
    await test.page.locator('#fileInput').setInputFiles(fixture);
    await test.page.waitForFunction(() => document.getElementById('media').readyState >= 1);
    await test.page.locator('#startButton').click();
    await until(() => test.uploads.length === 1, 'pending upload request');
    await test.page.locator('#finishButton').click();
    await until(() => test.page.locator('#startButton').isEnabled(), 'cancelled upload allows retry');
    assert(/取消/.test(await visibleError(test.page)), 'Cancelling preparation did not explain cancellation.');
    release();
    await delay(100);
    assert.equal(test.wire.length, 0, 'Cancelled upload still opened a model session.');
    assert.deepEqual(test.pageErrors, [], 'Cancelling upload caused an uncaught error.');
    assert.equal(await test.page.locator('#media').evaluate(media => media.paused), true);
    reports.push({ test: 'HTTP pending upload cancellation opens no model session' });
    console.log('PASS HTTP pending upload cancellation opens no model session');
  } finally { release(); await test.context.close(); }
}

async function runRepeatedSessions() {
  const test = await openPage(unsafeOrigin, false);
  try {
    const mediaIds = [];
    for (let round = 1; round <= 2; round++) {
      if (round > 1) {
        await until(() => test.page.locator('#startButton').isEnabled(), 'previous session cleanup');
        test.wire.splice(0); test.uploads.splice(0);
      }
      await chooseAndStart(test);
      mediaIds.push(test.prepared.media_id);
      await waitForMediaTime(test, 1.27);
      await test.page.locator('#finishButton').click();
      await verifyFinished(test, `HTTP same-page session ${round}`, false);
    }
    assert.notEqual(mediaIds[0], mediaIds[1], 'Repeated sessions reused a consumed upload.');
  } catch (error) {
    console.error(`FAIL repeat context ${JSON.stringify(await diagnostics(test))}`);
    throw error;
  } finally { await test.context.close(); }
}

try {
  fixture = await syntheticFixture();
  const config = await (await fetch(new URL('/api/config', base))).json();
  assert.equal(config.backend, 'mock', 'Refusing to run against a real model.');
  assert.equal(config.file_mode, 'uploaded', 'Start the updated upload-capable mock server first.');
  const module = await loadPlaywright(), chromium = module.chromium || module.default?.chromium;
  assert(chromium, 'The Playwright module does not expose chromium.');
  browser = await chromium.launch({ headless: true,
    ...(process.env.FLEX_BROWSER_EXECUTABLE ? { executablePath: process.env.FLEX_BROWSER_EXECUTABLE } : {}),
    args: ['--host-resolver-rules=MAP flex-http.test 127.0.0.1', '--no-proxy-server'] });
  await runHttpMicrophoneTest();
  await runUploadErrorTests();
  await runUploadCancelTest();
  await runFileTest({ pause: true });
  await runFileTest({ pause: true, immediateFinish: true });
  await runFileTest();
  await runFileTest({ secure: true });
  await runRepeatedSessions();
  await runFileTest({ stall: 400 });
  await runFileTest({ stall: 1200 });
  if (values.fixture) {
    const extra = path.resolve(values.fixture);
    assert((await stat(extra)).isFile(), 'Optional fixture must be a local file.');
    await runFileTest({ source: extra });
  }
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
