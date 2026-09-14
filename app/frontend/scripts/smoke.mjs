// Production UI regression check. One real inference; explicitly mocked edge cases afterwards.
// Run against a local backend serving dist: npm run test:ui
import { chromium } from 'playwright-core';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.resolve(frontend, '../..');
const base = process.env.SIGNALSCOPE_URL || 'http://127.0.0.1:8000';
const out = path.resolve(frontend, process.env.SIGNALSCOPE_QA_DIR || '.qa');
const fixture = process.env.SIGNALSCOPE_FIXTURE || path.join(root, 'report/explanation_samples/generated_detected.png');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const errors = [];
const checks = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(15000);
  page.on('pageerror', error => errors.push(error.message));
  const shot = async name => {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({ path: path.join(out, name + '.png'), fullPage: true, animations: 'disabled' });
  };
  const analyze = () => page.getByRole('button', { name: 'Analyze image', exact: true }).last().click();
  const nav = name => page.getByRole('navigation').getByRole('button', { name, exact: true }).click();
  const overflow = async name => assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, name + ' horizontal overflow');
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByText('Local model connected', { exact: true }).waitFor();
  const model = await (await page.request.get(base + '/api/model')).json();
  await shot('desktop-empty');
  await page.getByLabel('Choose image', { exact: true }).setInputFiles(fixture);
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/predict'));
  await analyze();
  await shot('desktop-processing');
  const response = await responsePromise;
  assert.equal(response.status(), 200, await response.text());
  assert.equal(response.request().method(), 'POST');
  assert.match(await response.request().headerValue('content-type'), /multipart\/form-data; boundary=/);
  const result = await response.json();
  await page.getByRole('heading', { name: /Likely AI-generated|Likely real|Review recommended/ }).waitFor();
  assert.ok(result.explanation?.overlay_data_url);
  assert.ok(result.robustness?.results.length);
  await fs.writeFile(path.join(out, 'live-analysis.json'), JSON.stringify(result, null, 2));
  checks.push('Real production inference with unchanged multipart API contract');

  const slider = page.getByRole('slider', { name: 'Original and influence comparison' });
  await slider.waitFor();
  await slider.focus();
  await slider.press('ArrowRight');
  assert.equal(await slider.inputValue(), '51');
  await slider.press('End');
  assert.equal(await slider.inputValue(), '100');
  await slider.press('Home');
  assert.equal(await slider.inputValue(), '0');
  await slider.fill('50');
  const box = await slider.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * .72, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  assert.ok(Number(await slider.inputValue()) > 65, 'Pointer comparison must move divider');
  await slider.fill('50');
  await shot('desktop-result');
  await page.getByRole('button', { name: 'Original', exact: true }).click();
  assert.match(await page.locator('.inspector-image').getAttribute('src'), /^blob:/);
  await page.getByRole('button', { name: 'Influence', exact: true }).click();
  assert.equal(await page.locator('.inspector-image').getAttribute('src'), result.explanation.overlay_data_url);
  await shot('desktop-influence');
  await page.getByRole('button', { name: 'Compare', exact: true }).click();
  const imageGeometry = await page.locator('.inspector-image').evaluateAll(images => images.map(image => ({
    fit: getComputedStyle(image).objectFit, width: image.clientWidth, height: image.clientHeight,
    ratio: image.naturalWidth / image.naturalHeight,
  })));
  assert.equal(imageGeometry[0].fit, 'contain');
  assert.deepEqual(imageGeometry[0], imageGeometry[1], 'Comparison layers must share full-image geometry');
  checks.push('Original / unmodified influence / comparison; pointer and keyboard slider');

  await page.getByRole('tab', { name: 'Evidence', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  assert.equal(await page.getByRole('tab', { name: 'Stability', exact: true }).getAttribute('aria-selected'), 'true');
  assert.equal(await page.locator('.stability-row').count(), result.robustness.results.length);
  await shot('desktop-stability');
  await page.keyboard.press('End');
  assert.equal(await page.getByRole('tab', { name: 'Metadata', exact: true }).getAttribute('aria-selected'), 'true');
  await shot('desktop-metadata');
  await page.keyboard.press('Home');
  assert.equal(await page.getByRole('tab', { name: 'Evidence', exact: true }).getAttribute('aria-selected'), 'true');
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download analysis JSON' }).click();
  const download = await downloadPromise;
  await download.saveAs(path.join(out, 'export.json'));
  const exported = JSON.parse(await fs.readFile(path.join(out, 'export.json'), 'utf8'));
  assert.deepEqual(exported.prediction, result.prediction);
  assert.deepEqual(exported.explanation, result.explanation);
  checks.push('Evidence keyboard tabs, real robustness rows, exact JSON export');



  await nav('Model report');
  await page.getByRole('heading', { name: 'Model report.', exact: true }).waitFor();
  if (model.metrics === null) assert.equal(await page.getByText('Not supplied for this checkpoint', { exact: true }).count(), 4);
  if (model.external_reserved) await page.getByRole('heading', { name: 'Reserved generators: final results' }).waitFor();
  await shot('desktop-report');
  await nav('How it works');
  await shot('desktop-about');
  for (const width of [320, 390, 768]) {
    await page.setViewportSize({ width, height: 844 });
    for (const name of ['Analyze image', 'Model report', 'How it works']) {
      await nav(name);
      await overflow(width + ' ' + name);
      if (width === 390) await shot('mobile-' + name.split(' ')[0].toLowerCase());
    }
  }
  checks.push('All three pages at desktop, 320px, 390px and 768px without horizontal overflow');
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await nav('Analyze image');
  assert.equal(await page.locator('.verdict-card').evaluate(element => getComputedStyle(element).animationName), 'none');
  checks.push('Reduced motion honoured');

  await page.getByLabel('Choose image', { exact: true }).setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('invalid') });
  await page.getByRole('alert').filter({ hasText: 'Choose a JPEG' }).waitFor();
  assert.equal(await page.locator('.verdict-card').count(), 1, 'Invalid upload must preserve the existing result');
  await page.getByLabel('Choose image', { exact: true }).setInputFiles({ name: 'too-large.png', mimeType: 'image/png', buffer: Buffer.alloc(25 * 1024 * 1024 + 1) });
  await page.getByRole('alert').filter({ hasText: '25 MiB' }).waitFor();
  assert.equal(await page.locator('.verdict-card').count(), 1);
  checks.push('Invalid type and oversize upload preserve the current analysis');

  // Test-only response fixtures exercise states that a real uploaded image may not produce.
  const inject = async payload => {
    await page.unroute('**/api/predict');
    await page.route('**/api/predict', route => route.fulfill({ json: payload }));
    await analyze();
    await page.locator('.verdict-card').waitFor();
  };
  await page.setViewportSize({ width: 1440, height: 1050 });
  const uncertain = structuredClone(result);
  uncertain.prediction.review_recommended = true;
  uncertain.prediction.ai_score = .25;
  uncertain.prediction.label = 'real';
  uncertain.prediction.confidence = .75;
  uncertain.prediction.limitations = ['Low-information image: inspect the source before relying on this assessment.'];
  uncertain.explanation.localisation = { supported: false, returned_class_drop: -.02, comparison_drop: .001 };
  await inject(uncertain);
  await page.getByRole('heading', { name: 'Review recommended', exact: true }).waitFor();
  assert.equal(await page.locator('.model-limitations').getAttribute('open'), '');
  assert.ok(!(await page.locator('.verdict-card').innerText()).includes('near the operating threshold'));
  await page.getByText('Verdict is not localised', { exact: true }).waitFor();
  await shot('test-fixture-review');
  const supported = structuredClone(result);
  supported.explanation.localisation = { supported: true, returned_class_drop: .10, comparison_drop: .02 };
  await inject(supported);
  await page.getByText('Local influence supported', { exact: true }).waitFor();
  checks.push('Test fixtures: review reason and supported / unsupported localisation');

  for (const [status, label] of [
    ['no_marker_found', 'No marker found'], ['marker_found_unverified', 'Marker found · unverified'], ['not_checked', 'Not checked'],
  ]) {
    const metadataFixture = structuredClone(result);
    metadataFixture.metadata.c2pa_status = status;
    await inject(metadataFixture);
    await page.getByRole('tab', { name: 'Metadata', exact: true }).click();
    await page.getByText(label, { exact: true }).waitFor();
  }
  checks.push('Test fixtures: all three contract C2PA statuses');

  const optional = structuredClone(result);
  optional.explanation = null;
  optional.robustness = null;
  await page.getByRole('checkbox', { name: /Model influence map/ }).uncheck();
  await page.getByRole('checkbox', { name: /Robustness check/ }).uncheck();
  await inject(optional);
  await page.getByRole('heading', { name: 'Take a closer look', exact: true }).waitFor();
  assert.equal(await page.getByRole('slider').count(), 0);
  await page.getByRole('tab', { name: 'Stability', exact: true }).click();
  await page.getByRole('heading', { name: 'How well does the signal hold up?' }).waitFor();
  await page.getByRole('button', { name: 'Remove image' }).click();
  assert.equal(await page.locator('.verdict-card').count(), 0);
  await page.getByLabel('Choose image', { exact: true }).setInputFiles(fixture);
  // The actual browser picker dispatches change after the app clears the previous input value.
  const chooserPromise = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Change image' }).click();
  await (await chooserPromise).setFiles(fixture);
  assert.equal(await page.locator('.inspector-image').count(), 1);
  checks.push('Optional checks off, clearing, and choosing the same file again');

  const bytes = (await fs.readFile(fixture)).toString('base64');
  await page.getByRole('button', { name: 'Remove image' }).click();
  const transfer = await page.evaluateHandle(encoded => {
    const data = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
    const transfer = new DataTransfer();
    transfer.items.add(new File([data], 'dragged-image.png', { type: 'image/png' }));
    return transfer;
  }, bytes);
  await page.locator('.drop-zone').dispatchEvent('drop', { dataTransfer: transfer });
  assert.equal(await page.locator('.inspector-filename').innerText(), 'dragged-image.png');
  checks.push('Drag and drop');

  await page.unroute('**/api/predict');
  await page.route('**/api/predict', route => route.fulfill({ status: 422, json: { detail: [{ loc: ['body', 'image'], msg: 'Invalid image payload', type: 'value_error' }] } }));
  await analyze();
  await page.getByRole('alert').filter({ hasText: 'Invalid image payload' }).waitFor();
  await page.route('**/api/health', route => route.fulfill({ json: { ready: false, message: 'Model checkpoint unavailable.' } }));
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByText('Model checkpoint unavailable.', { exact: false }).waitFor();
  await page.getByLabel('Choose image', { exact: true }).setInputFiles(fixture);
  assert.equal(await page.getByRole('button', { name: 'Analyze image', exact: true }).last().isDisabled(), true);
  await shot('test-fixture-unavailable');
  checks.push('Structured API errors and unavailable model');
  assert.deepEqual(errors, []);
  const summary = { ok: true, checks, live_model: result.prediction.model_version, live_prediction: result.prediction, javascript_errors: errors };
  await fs.writeFile(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally { await browser.close(); }
