// Record genuine local inference on the v0.4.0 UI. Narration clips/timings are prepared
// in tmp/demo. Requires installed Chrome and Playwright FFmpeg; no prediction mocking or
// network model API.
import { chromium } from 'playwright-core';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const out = path.join(root, 'tmp/demo/recording');
await fs.mkdir(out, { recursive: true });
const segments = JSON.parse(await fs.readFile(path.join(root, 'tmp/demo/narration.json'), 'utf8'));
const base = process.env.SIGNALSCOPE_URL || 'http://127.0.0.1:8002';
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({ viewport: { width: 1600, height: 900 },
  recordVideo: { dir: out, size: { width: 1600, height: 900 } } });
const page = await context.newPage();
const started = Date.now();
const evidence = [];
const errors = [];
page.on('pageerror', e => errors.push(e.message));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const escape = text => text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
async function caption(text, number) {
  await page.evaluate(({text, number, total}) => {
    document.querySelector('#demo-caption')?.remove();
    const overlay = document.createElement('div'); overlay.id = 'demo-caption';
    overlay.style.cssText = 'position:fixed;z-index:10000;bottom:0;left:0;right:0;background:#102427;color:#fff;padding:17px 42px;box-shadow:0 -3px 20px #0002;font:24px/1.35 Arial;pointer-events:none';
    const label = document.createElement('div'); label.style.cssText = 'font-size:12px;letter-spacing:2px;color:#9cc6b8;margin-bottom:7px';
    label.textContent = `SIGNALSCOPE / LIVE CPU RECORDING / ${number+1} OF ${total}`;
    const line = document.createElement('div'); line.textContent = text;
    overlay.append(label, line); document.body.append(overlay); document.body.style.paddingBottom = '150px';
  }, {text, number, total: segments.length});
}
async function analyze(filename) {
  await page.locator('input[aria-label="Choose image"]').setInputFiles(path.join(root, 'report/explanation_samples', filename));
  const pending = page.waitForResponse(r => r.url().endsWith('/api/predict'));
  await page.locator('.analyze-button').click();
  const response = await pending;
  if (!response.ok()) throw new Error(await response.text());
  const result = await response.json();
  evidence.push({filename, prediction: result.prediction, explanation: result.explanation?.statements,
    masking: result.explanation?.localisation, robustness: result.robustness});
  await page.getByRole('heading', { name: /Likely AI-generated|Likely real|Review recommended/ }).waitFor();
  await page.locator('.verdict-card').evaluate(el => el.scrollIntoView({block:'center'}));
}
async function focusHeading(name) {
  await page.getByRole('heading', {name, exact:true}).evaluate(el => el.scrollIntoView({block:'start'}));
}
await page.goto(base, {waitUntil:'networkidle'});
await page.getByText('Local model connected', {exact:true}).waitFor();
const health = await (await page.request.get(base+'/api/health')).json();
if (health.device !== 'cpu' || !health.ready) throw new Error('Demo requires the real CPU detector');
const timings=[];
try {
 for (const segment of segments) {
  await page.locator('#demo-caption').evaluateAll(nodes => nodes.forEach(n=>n.remove()));
  switch(segment.action) {
   case 'intro': await page.evaluate(()=>window.scrollTo(0,0)); break;
   case 'workflow': await page.locator('.analysis-options').evaluate(el=>el.scrollIntoView({block:'center'})); break;
   case 'real': await analyze('real_photo.jpeg'); break;
   case 'generated': await analyze('generated_new_correct.png'); break;
   case 'explanation': await page.getByRole('tab',{name:'Evidence',exact:true}).click(); await page.locator('.tab-content').evaluate(el=>el.scrollIntoView({block:'center'})); break;
   case 'failure': await analyze('generated_new_missed.jpg'); break;
   case 'stability': await page.getByRole('tab',{name:'Stability',exact:true}).click(); await page.locator('.tab-content').evaluate(el=>el.scrollIntoView({block:'center'})); break;
   case 'metadata': await page.getByRole('tab',{name:'Metadata',exact:true}).click(); await page.locator('.tab-content').evaluate(el=>el.scrollIntoView({block:'center'})); break;
   case 'about': await page.getByRole('button',{name:'How it works',exact:true}).click(); await page.evaluate(()=>window.scrollTo(0,0)); break;
   case 'sources': await page.getByRole('button',{name:'Model report',exact:true}).click(); await page.evaluate(()=>window.scrollTo(0,0)); break;
   case 'final': await focusHeading('Published release evidence'); break;
   case 'method': await page.getByLabel('Evaluation set').selectOption({label:'AI Detect Arena · Fixed image holdout'}); await page.locator('.benchmark-toolbar').evaluate(el=>el.scrollIntoView({block:'center'})); break;
   case 'indomain': await page.getByLabel('Evaluation set').selectOption({label:'CommunityForensics · Fixed image holdout'}); await page.locator('.benchmark-toolbar').evaluate(el=>el.scrollIntoView({block:'center'})); break;
   case 'cli': {
    const stdout=execFileSync(path.join(root,'.venv/Scripts/python.exe'),['model/predict.py','--image','report/explanation_samples/generated_new_correct.png','--device','cpu'],{cwd:root,encoding:'utf8',env:{...process.env,SIGNALSCOPE_DEVICE:'cpu'}});
    const prediction=JSON.parse(stdout); evidence.push({actual_cli_output:prediction});
    await page.setContent(`<html><body style="margin:0;background:#f5f7f5;color:#102427;font:19px/1.4 Arial;padding:36px 55px"><h1>One checkpoint. App, API and CLI.</h1><div style="display:grid;grid-template-columns:1.3fr 1fr;gap:35px"><div><h2>Command executed during this recording</h2><pre style="font:15px/1.35 Consolas;white-space:pre-wrap;background:white;padding:20px">.venv/Scripts/python model/predict.py --image report/explanation_samples/generated_new_correct.png --device cpu\n\n${escape(stdout)}</pre></div><div><h2>Evaluator setup instructions</h2><pre style="font:16px/1.8 Consolas;white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px">git clone https://github.com/sibtainmunshi/Signal_Scope.git\ncd Signal_Scope\npython scripts/setup.py\npython scripts/run.py</pre><p>608 MB tower + 398 KB head, verified by SHA-256.<br>Python packages are additional first-setup downloads.<br>No training dataset or GPU is needed for inference.<br>Measured fresh-clone time: 243.75 seconds, well under 10 minutes.</p></div></div></body></html>`); break;
   }
   case 'ending': await page.setContent(`<html><body style="margin:0;background:#102427;color:white;font:24px/1.5 Arial;padding:65px 90px"><div style="font-size:15px;letter-spacing:4px;color:#a3c6b7">SIH 2026 / SIGNALSCOPE / v0.4.0</div><h1 style="font-size:50px;margin-bottom:10px">Evidence before certainty.</h1><p>Actual local predictions. Reproducible code. Reported failures.</p><div style="display:flex;gap:55px;margin:40px 0"><div><b style="font-size:44px">0.948</b><br>AI Detect Arena AUC (unseen)</div><div><b style="font-size:44px">0.936</b><br>CommunityForensics AUC (unseen)</div><div><b style="font-size:44px">85.2% / 71.8%</b><br>Holdout accuracy</div></div><p>Deployed by explicit, disclosed decision after two gated training attempts failed -- not a hidden gate pass.<br>Real-photo accuracy 92.7%/97.8%; AI recall 78.5%/54.4% on the same two untouched holdouts.</p><p style="color:#bbd4ca;font-size:20px">github.com/sibtainmunshi/Signal_Scope<br>README.md / report/releases/v0.4.0/model_report.pdf / docs/POST_RELEASE_EXPERIMENTS.md</p><p style="font-size:15px;color:#bbd4ca">Demo images: GenImage validation (Zhu et al., NeurIPS 2023); AI Detect Arena Benchmark; CommunityForensics-Eval (Park et al., CVPR 2025); one newly generated image supplied for this recording.<br>Computer narration: Windows text-to-speech. Human explanation-usefulness review remains pending.</p></body></html>`); break;
  }
  await caption(segment.caption,segment.id);
  const t0=(Date.now()-started)/1000;
  console.log(JSON.stringify({scene:segment.action,at_seconds:Math.round(t0),audio_seconds:segment.audio_seconds}));
  await page.screenshot({path:path.join(out,`${String(segment.id).padStart(2,'0')}-${segment.action}.png`)});
  await sleep((segment.audio_seconds+1.5)*1000);
  timings.push({...segment,start_seconds:t0,end_seconds:(Date.now()-started)/1000});
  await fs.writeFile(path.join(out,'timings.json'),JSON.stringify(timings,null,2));
 }
 if(errors.length) throw new Error(errors.join('; '));
 await fs.writeFile(path.join(out,'evidence.json'),JSON.stringify({health,evidence,javascript_errors:errors,simulated_predictions:false},null,2));
} finally {
 await context.close();
 await fs.writeFile(path.join(out,'video_path.txt'),await page.video().path());
 await browser.close();
}
console.log('Live demo recording complete.');
