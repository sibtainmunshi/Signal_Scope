import { chromium } from 'playwright-core';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const out = path.join(root, 'tmp/ui-qa');
await fs.mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, deviceScaleFactor: 1 });
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('http://127.0.0.1:8000', { waitUntil: 'networkidle' });
  await page.getByText('Local model connected', { exact: true }).waitFor();
  await page.screenshot({ path: path.join(out, 'desktop-empty.png'), fullPage: true });
  const csv = await fs.readFile(path.join(root, 'data/manifests/cifake.csv'), 'utf8');
  const row = csv.split(/\r?\n/).slice(1).find(line => line.split(',')[4] === 'val');
  if (!row) throw new Error('No validation fixture');
  const image = path.join(root, 'data/raw/cifake', row.split(',')[1]);
  await page.getByLabel('Choose image', { exact: true }).setInputFiles(image);
  const predictionResponse = page.waitForResponse(r => r.url().endsWith('/api/predict'));
  await page.getByRole('button', { name: 'Analyze image', exact: true }).last().click();
  const response = await predictionResponse;
  if (response.status() !== 200) throw new Error(await response.text());
  const result = await response.json();
  await page.getByRole('heading', { name: /Likely AI-generated|Likely real|Review recommended/ }).waitFor();
  await page.screenshot({ path: path.join(out, 'desktop-result.png'), fullPage: true });
  await page.getByRole('tab', { name: 'Stability', exact: true }).click();
  await page.screenshot({ path: path.join(out, 'desktop-stability.png'), fullPage: true });
  await page.getByRole('button', { name: 'Model report', exact: true }).click();
  await page.getByRole('heading', { name: 'Model report.', exact: true }).waitFor();
  await page.screenshot({ path: path.join(out, 'desktop-report.png'), fullPage: true });
  await page.getByRole('button', { name: 'Analyze image', exact: true }).first().click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(out, 'mobile-result.png'), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  if (overflow) errors.push('Mobile horizontal overflow');
  if (errors.length) throw new Error(errors.join('; '));
  const summary = { ok: true, browser: 'installed Chrome, headless', server_device: 'cpu',
    fixture: row.split(',')[1], prediction: result.prediction,
    explanation_present: Boolean(result.explanation), transformations: result.robustness.results.length,
    javascript_errors: errors, mobile_horizontal_overflow: overflow };
  await fs.writeFile(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary, null, 2));
} finally { await browser.close(); }
