#!/usr/bin/env node
/**
 * Local HTTPS + WSS integration, with synthetic computer audio and WAV.
 *
 * python run.py --backend mock --host 127.0.0.1 --port 31220 \
 *   --https-host localhost --cert-dir /tmp/flex-test-tls --no-log
 * node tools/browser_https_smoke.mjs --base-url https://localhost:31220 \
 *   --ca-cert /tmp/flex-test-tls/root-ca.crt --certfile /path/to/server.crt
 *
 * Node verifies TLS with the explicitly supplied CA. Chromium is given an
 * isolated-profile, exact-leaf-public-key exception, NOT a global certificate
 * bypass or an insecure-origin exemption. No OS/browser trust store is changed.
 * getDisplayMedia is explicitly mocked with oscillator audio and canvas video;
 * no physical microphone, screen capture, system audio or GPU is accessed.
 */
import assert from 'node:assert/strict';
import { createHash, X509Certificate } from 'node:crypto';
import { mkdtemp, readFile, rmdir, stat, unlink, writeFile } from 'node:fs/promises';
import https from 'node:https';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: {
  'base-url': { type: 'string', default: 'https://localhost:31220' },
  'ca-cert': { type: 'string' },
  certfile: { type: 'string' },
  'timeout-ms': { type: 'string', default: '45000' },
  help: { type: 'boolean', default: false },
} });
if (values.help) {
  console.log('Usage: node tools/browser_https_smoke.mjs --base-url https://localhost:31220 --ca-cert /path/to/root-ca.crt --certfile /path/to/server.crt [--timeout-ms 45000]');
  process.exit(0);
}
const base = new URL(values['base-url']);
assert.equal(base.protocol, 'https:', 'Use the HTTPS mock endpoint.');
assert(['localhost', '127.0.0.1'].includes(base.hostname), 'Only a local mock server is allowed.');
assert(!base.username && !base.password && base.pathname === '/' && !base.search && !base.hash,
  'Provide only the HTTPS origin, without credentials or a path.');
assert(values['ca-cert'] && values.certfile, '--ca-cert and --certfile are required.');
const timeoutMs = Number(values['timeout-ms']);
assert(Number.isFinite(timeoutMs) && timeoutMs >= 5000, '--timeout-ms must be at least 5000.');
const ca = await readFile(values['ca-cert']);
const certificate = new X509Certificate(await readFile(values.certfile));
const spki = createHash('sha256').update(certificate.publicKey.export({ type: 'spki', format: 'der' })).digest('base64');
const config = await new Promise((resolve, reject) => {
  const request = https.get(new URL('/api/config', base), { ca, family: 4, timeout: 10000 }, response => {
    try {
      assert.equal(response.statusCode, 200, 'Mock config request failed.');
      assert(response.socket.authorized, 'Server certificate was not verified against the supplied CA.');
      assert.deepEqual(response.socket.getPeerCertificate().raw, certificate.raw,
        '--certfile is not the leaf certificate served by this endpoint.');
    } catch (error) { response.resume(); reject(error); return; }
    const chunks = [];
    response.on('data', chunk => chunks.push(chunk));
    response.on('error', reject);
    response.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); } catch (error) { reject(error); }
    });
  });
  request.on('timeout', () => request.destroy(new Error('HTTPS config request timed out.')));
  request.on('error', reject);
});
assert.equal(config.backend, 'mock', 'Only the mock backend may be tested.');
assert.equal(config.auth_required, false, 'Start the local mock without FLEX_ACCESS_TOKEN.');
assert.equal(config.chunk_seconds, 1, 'Use the default one-second base chunk for this smoke test.');
console.log('PASS TLS: valid CA chain, hostname and exact served leaf certificate.');

const location = process.env.FLEX_PLAYWRIGHT_MODULE;
const resolved = location && ((await stat(location)).isDirectory()
  ? createRequire(import.meta.url).resolve(location) : location);
const playwright = await import(resolved ? pathToFileURL(resolved).href : 'playwright');
const chromium = playwright.chromium || playwright.default.chromium;
const temporary = await mkdtemp(path.join(tmpdir(), 'flex-https-smoke-'));
const fixture = path.join(temporary, 'synthetic.wav');
const rate = 16000, sampleCount = 54000;
const wav = Buffer.alloc(44 + sampleCount * 2);
wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 2, 28);
wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
wav.write('data', 36); wav.writeUInt32LE(sampleCount * 2, 40);
for (let i = 0; i < sampleCount; i++) wav.writeInt16LE(Math.round(6000 * Math.sin(i / rate * 440 * 2 * Math.PI)), 44 + i * 2);
await writeFile(fixture, wav, { flag: 'wx' });
let browser;

async function until(test, predicate, description) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    assert.deepEqual(test.errors, [], 'Uncaught browser exception.');
    const failure = test.wires.flatMap(wire => wire.received).find(event => event.type === 'error');
    assert(!failure, JSON.stringify(failure));
    const ui = await test.page.evaluate(() => ({
      failed: document.getElementById('sessionStatus').classList.contains('error'),
      message: document.getElementById('errorMessage').textContent,
    }));
    assert(!ui.failed, `Frontend session failed: ${ui.message}`);
    if (await predicate()) return;
    await delay(50);
  }
  throw new Error(`Timed out: ${description}; wire=${JSON.stringify(test.wires)}`);
}

async function openPage() {
  const context = await browser.newContext();
  const page = await context.newPage(), test = { context, page, wires: [], errors: [], uploads: [] };
  page.on('pageerror', error => test.errors.push(String(error)));
  page.on('response', response => {
    if (response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/media')
      test.uploads.push({ url: response.url(), status: response.status() });
  });
  page.on('websocket', socket => {
    const wire = { url: socket.url(), sent: [], received: [], binary: [], closed: false };
    test.wires.push(wire);
    socket.on('framesent', ({ payload }) => {
      if (typeof payload === 'string') wire.sent.push(JSON.parse(payload));
      else wire.binary.push({ sequence: payload.readUInt32LE(0), start: payload.readUInt32LE(4), samples: (payload.length - 8) / 2 });
    });
    socket.on('framereceived', ({ payload }) => wire.received.push(JSON.parse(String(payload))));
    socket.on('close', () => { wire.closed = true; });
  });
  await page.addInitScript(() => {
    window.__flexHttpsTracks = [];
    window.__flexHttpsWorklets = 0;
    window.__flexHttpsDisplayCalls = 0;
    window.__flexHttpsMicCalls = 0;
    window.__flexHttpsSources = [];
    navigator.mediaDevices.getUserMedia = () => {
      window.__flexHttpsMicCalls++;
      return Promise.reject(new Error('Test must not request a microphone.'));
    };
    navigator.mediaDevices.getDisplayMedia = () => {
      window.__flexHttpsDisplayCalls++;
      const audio = new AudioContext(), oscillator = audio.createOscillator();
      const destination = audio.createMediaStreamDestination();
      oscillator.connect(destination);oscillator.start();void audio.resume();
      const canvas = document.createElement('canvas');canvas.width=16;canvas.height=16;
      canvas.getContext('2d').fillRect(0,0,16,16);
      const video = canvas.captureStream(1);
      const tracks = [...video.getTracks(),...destination.stream.getAudioTracks()];
      for (const track of tracks) {
        const stop = track.stop.bind(track);
        track.stop = () => {
          stop();
          if (tracks.every(item => item.readyState === 'ended')) {
            oscillator.stop();void audio.close();
          }
        };
      }
      window.__flexHttpsTracks.push(...tracks);
      window.__flexHttpsSources.push({audio,oscillator,canvas,video});
      return Promise.resolve(new MediaStream(tracks));
    };
    window.AudioWorkletNode = new Proxy(window.AudioWorkletNode, {
      construct(target, args, newTarget) {
        window.__flexHttpsWorklets++;
        return Reflect.construct(target, args, newTarget);
      },
    });
  });
  await page.goto(base.origin, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.getElementById('serverStatus').classList.contains('ready'));
  assert.deepEqual(await page.evaluate(() => ({ secure: isSecureContext, display: typeof navigator.mediaDevices?.getDisplayMedia, worklet: typeof AudioWorkletNode })),
    { secure: true, display: 'function', worklet: 'function' });
  return test;
}

async function finished(test, name) {
  await until(test, () => test.wires[0]?.received.some(event => event.type === 'done') && test.wires[0]?.closed, `${name} done and closed`);
  await test.page.waitForFunction(() => document.getElementById('sessionStatus').textContent.includes('同传已结束'));
  assert.equal(test.wires.length, 1);
  const wire = test.wires[0], done = wire.received.find(event => event.type === 'done');
  assert.equal(wire.url, `${base.origin.replace('https:', 'wss:')}/ws`);
  assert.equal(done.reason, 'finished');
  assert.equal(done.processed_audio_s, done.received_audio_s, 'Buffered audio was not fully drained.');
  assert.equal(wire.sent.filter(event => event.type === 'finish').length, 1);
  assert(!wire.sent.some(event => event.type === 'cancel'));
  const turns = wire.received.filter(event => event.type === 'turn');
  assert.deepEqual([...new Set(turns.map(turn => turn.tag))].sort(), ['DIRECT', 'FLEX', 'WAIT']);
  const transcript = await test.page.locator('#transcript').textContent();
  assert(transcript && !/<\|(?:WAIT|FLEX|DIRECT)\|>/.test(transcript), 'Protocol labels leaked into visible translation.');
  assert.equal(transcript, turns.map(turn => turn.delta).join(''));
  assert.equal(done.translation, transcript);
  for (let i = 0; i < turns.length; i++) {
    assert.equal(turns[i].source_start_s, i ? turns[i - 1].source_end_s : 0);
    assert(turns[i].audio_seconds > 0);
  }
  assert.equal(turns.at(-1).source_end_s, done.processed_audio_s);
  console.log(`PASS ${name}: HTTPS + WSS, ${turns.length} turns, ${done.processed_audio_s}s, WAIT/FLEX/DIRECT, clean shutdown.`);
  return { wire, done, turns };
}

try {
  browser = await chromium.launch({ headless: true,
    ...(process.env.FLEX_BROWSER_EXECUTABLE ? { executablePath: process.env.FLEX_BROWSER_EXECUTABLE } : {}),
    args: [`--ignore-certificate-errors-spki-list=${spki}`, '--no-proxy-server'],
  });
  const mic = await openPage();
  try {
    await mic.page.locator('#micMode').click();
    await mic.page.locator('#startButton').click();
    await until(mic, () => mic.wires[0]?.received.filter(event => event.type === 'turn').length >= 3, 'three synthetic computer audio turns');
    await mic.page.locator('#finishButton').click();
    const { wire, done } = await finished(mic, 'synthetic computer audio');
    assert.equal(wire.sent.find(event => event.type === 'start').source_kind, 'system');
    assert.equal(wire.sent.find(event => event.type === 'start').include_microphone, false);
    assert(wire.binary.length > 0, 'Computer audio sent no PCM packets.');
    let samples = 0;
    wire.binary.forEach((packet, i) => {
      assert.equal(packet.sequence, i); assert.equal(packet.start, samples);
      assert(Number.isInteger(packet.samples) && packet.samples > 0);
      samples += packet.samples;
    });
    assert.equal(done.received_audio_s, samples / rate);
    const capture = await mic.page.evaluate(() => ({ worklets: __flexHttpsWorklets,
      displayCalls:__flexHttpsDisplayCalls,micCalls:__flexHttpsMicCalls,
      tracks: __flexHttpsTracks.map(track => track.readyState) }));
    assert.equal(capture.worklets, 1);
    assert.equal(capture.displayCalls,1);assert.equal(capture.micCalls,0);
    assert(capture.tracks.length === 2 && capture.tracks.every(state => state === 'ended'), 'Synthetic audio and video tracks were not released.');
    assert.equal(mic.uploads.length, 0);
  } finally { await mic.context.close(); }

  const file = await openPage();
  try {
    await file.page.locator('#fileInput').setInputFiles(fixture);
    await file.page.waitForFunction(() => document.getElementById('media').readyState >= 1);
    await file.page.locator('#startButton').click();
    const { wire, done, turns } = await finished(file, 'uploaded synthetic WAV');
    assert.equal(file.uploads.length, 1);
    assert.equal(new URL(file.uploads[0].url).protocol, 'https:');
    assert.equal(file.uploads[0].status, 200);
    assert.equal(wire.sent.find(event => event.type === 'start').source_kind, 'file');
    assert.equal(wire.sent.find(event => event.type === 'start').adaptive, false);
    assert.equal(wire.binary.length, 0);
    assert.equal(done.processed_audio_s, sampleCount / rate);
    assert.deepEqual(turns.map(turn => Math.round(turn.audio_seconds * rate)), [16000, 16000, 16000, 6000]);
    assert.deepEqual(await file.page.evaluate(() => ({ worklets: __flexHttpsWorklets, tracks: __flexHttpsTracks.length })), { worklets: 0, tracks: 0 });
  } finally { await file.context.close(); }
  console.log('All HTTPS browser checks passed (mock, synthetic audio, exact-key test exception only).');
} finally {
  await browser?.close();
  await unlink(fixture);
  await rmdir(temporary);
}
