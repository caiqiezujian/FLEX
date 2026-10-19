#!/usr/bin/env node
// Compact-layout and real 1/2/3-second file-chunk checks against a local mock.
import assert from 'node:assert/strict';
import { mkdtemp, stat, writeFile, unlink, rmdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({options:{
  'base-url':{type:'string',default:'http://127.0.0.1:31219'},
  screenshots:{type:'boolean',default:false},
  help:{type:'boolean',default:false},
}});
if (values.help) {
  console.log('Usage: node tools/browser_layout_smoke.mjs [--base-url http://127.0.0.1:31219] [--screenshots]');
  process.exit(0);
}
const base = new URL(values['base-url']);
assert(base.protocol === 'http:' && ['localhost','127.0.0.1'].includes(base.hostname));
assert(!base.username && !base.password && base.pathname === '/' && !base.search && !base.hash);
const origin = `http://flex-http.test${base.port ? `:${base.port}` : ''}`;
const config = await (await fetch(new URL('/api/config',base))).json();
assert.equal(config.backend,'mock','Only a local mock server may be tested.');
assert.equal(config.auth_required,false,'Use a mock server without FLEX_ACCESS_TOKEN.');
assert.equal(config.max_chunk_seconds,3,'Use default max chunk length.');
const packageLocation = process.env.FLEX_PLAYWRIGHT_MODULE;
const resolved = packageLocation && ((await stat(packageLocation)).isDirectory()
  ? createRequire(import.meta.url).resolve(packageLocation) : packageLocation);
const module = await import(resolved ? pathToFileURL(resolved).href : 'playwright');
const chromium = module.chromium || module.default.chromium;
const temporary = await mkdtemp(path.join(tmpdir(),'flex-layout-smoke-'));
const fixture = path.join(temporary,'synthetic.wav');
const rate=16000, sampleCount=54000, duration=sampleCount/rate;
const wav=Buffer.alloc(44+sampleCount*2);
wav.write('RIFF');wav.writeUInt32LE(wav.length-8,4);wav.write('WAVEfmt ',8);
wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);
wav.writeUInt32LE(rate,24);wav.writeUInt32LE(rate*2,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);
wav.write('data',36);wav.writeUInt32LE(sampleCount*2,40);
for(let i=0;i<sampleCount;i++)wav.writeInt16LE(Math.round(6000*Math.sin(i/rate*440*2*Math.PI)),44+i*2);
await writeFile(fixture,wav,{flag:'wx'});
let browser;
const screenshots=values.screenshots ? await mkdtemp(path.join(tmpdir(),'flex-ui-preview-')) : null;
try {
  browser=await chromium.launch({headless:true,
    ...(process.env.FLEX_BROWSER_EXECUTABLE ? {executablePath:process.env.FLEX_BROWSER_EXECUTABLE} : {}),
    args:['--host-resolver-rules=MAP flex-http.test 127.0.0.1','--no-proxy-server']});
  for(const [width,height] of [[1366,768],[1920,1080],[900,900],[390,844]]) {
    const context=await browser.newContext({viewport:{width,height}}), page=await context.newPage(), errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(origin,{waitUntil:'networkidle'});
    await page.waitForFunction(()=>document.getElementById('serverStatus').classList.contains('ready'));
    const layout=await page.evaluate(()=>{
      const rect=selector=>{const b=document.querySelector(selector).getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height,bottom:b.bottom};};
      return {scrollWidth:document.documentElement.scrollWidth,width:innerWidth,
        controls:rect('.session-bar'),stage:rect('.media-stage'),
        fonts:Object.fromEntries(['#sourceLang','#startButton','#fileMode','.panel-title h2','#transcript'].map(s=>[s,parseFloat(getComputedStyle(document.querySelector(s)).fontSize)]))};
    });
    assert(layout.scrollWidth<=width+1,`horizontal overflow at ${width}: ${JSON.stringify(layout)}`);
    for(const [selector,size] of Object.entries(layout.fonts))assert(size>=14,`${selector} is too small at ${width}: ${size}`);
    assert(layout.fonts['#transcript']>=22,'Translation text must remain readable.');
    if(width>=1366)assert(layout.controls.bottom<=height,`Main controls require scrolling: ${JSON.stringify(layout)}`);
    assert(layout.stage.width>=300,'Media stage became too narrow.');
    if(screenshots)await page.screenshot({path:path.join(screenshots,`${width}-idle.png`),fullPage:true});
    await page.locator('#settings > summary').click();
    assert.equal(await page.locator('#authField').isVisible(),false);
    assert.equal(await page.locator('#adaptive').isEnabled(),false);
    for(const seconds of [1,2,3]) {
      await page.locator('#chunkSeconds').selectOption(String(seconds));
      assert.equal(await page.locator('#chunkSeconds').inputValue(),String(seconds));
    }
    const panel=await page.locator('.settings-panel').boundingBox();
    assert(panel.x>=0 && panel.x+panel.width<=width+1,'Settings panel overflows viewport.');
    if(screenshots)await page.screenshot({path:path.join(screenshots,`${width}-settings.png`),fullPage:true});
    await page.locator('#settings > summary').click();
    await page.locator('#micMode').click();
    await page.locator('#settings > summary').click();
    assert.equal(await page.locator('#adaptive').isEnabled(),true);
    assert.equal(await page.locator('#adaptive').isChecked(),true);
    assert.deepEqual(errors,[]);
    console.log(`PASS layout ${width}x${height}: ${JSON.stringify(layout)}`);
    await context.close();
  }
  for(const seconds of [1,2,3]) {
    const context=await browser.newContext({viewport:{width:1366,height:768}}),page=await context.newPage();
    const sent=[],received=[],errors=[];
    page.on('pageerror',e=>errors.push(String(e)));
    page.on('websocket',socket=>{
      socket.on('framesent',({payload})=>sent.push(JSON.parse(String(payload))));
      socket.on('framereceived',({payload})=>received.push(JSON.parse(String(payload))));
    });
    await page.goto(origin,{waitUntil:'networkidle'});
    await page.locator('#fileInput').setInputFiles(fixture);
    await page.waitForFunction(()=>document.getElementById('media').readyState>=1);
    await page.locator('#settings > summary').click();
    await page.locator('#chunkSeconds').selectOption(String(seconds));
    await page.locator('#startButton').click();
    await page.waitForFunction(()=>document.getElementById('sessionStatus').textContent.includes('同传已结束'),{},{timeout:30000});
    const start=sent.find(e=>e.type==='start'),done=received.find(e=>e.type==='done');
    assert.equal(start.chunk_seconds,seconds);assert.equal(start.adaptive,false);assert(!start.token);
    assert.equal(done.reason,'finished');assert.equal(done.processed_audio_s,duration);
    const turns=received.filter(e=>e.type==='turn'),expected=[];
    for(let offset=0;offset<sampleCount;offset+=seconds*rate)expected.push(Math.min(seconds*rate,sampleCount-offset));
    assert.deepEqual(turns.map(e=>Math.round(e.audio_seconds*rate)),expected);
    assert.equal(await page.locator('#transcript').textContent(),turns.map(e=>e.delta).join(''));
    assert.equal(await page.locator('#errorBox').isVisible(),Boolean(done.final_wait));
    if(done.final_wait)assert.equal(await page.locator('#errorTitle').textContent(),'尾句检查');
    assert.deepEqual(errors,[]);
    if(screenshots && seconds===1)await page.screenshot({path:path.join(screenshots,'1366-completed.png'),fullPage:true});
    console.log(`PASS browser file ${seconds}s chunks: ${JSON.stringify(expected)}`);
    await context.close();
  }
  console.log('All 7 layout/chunk browser checks passed (mock only).');
  if(screenshots)console.log(`Screenshots: ${screenshots}`);
} finally {
  await browser?.close();
  await unlink(fixture);await rmdir(temporary);
}
