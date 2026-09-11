// Render the one-page HTML report to A4 PDF (and a preview PNG) with installed Chrome.
import { chromium } from 'playwright-core';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [, , input, output, preview] = process.argv;
const browser = await chromium.launch({ channel: 'chrome', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 794, height: 1123 } });
  await page.goto(pathToFileURL(path.resolve(input)).href, { waitUntil: 'load' });
  await page.pdf({ path: output, format: 'A4', printBackground: true, preferCSSPageSize: true });
  if (preview) {
    await page.emulateMedia({ media: 'print' });
    await page.screenshot({ path: preview, fullPage: true });
  }
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  console.log(JSON.stringify({ output, preview, content_height_px: height, a4_height_px: 1123 }));
} finally {
  await browser.close();
}
