#!/usr/bin/env node
/**
 * Headless introspection probe. Boots the game exactly like shoot.mjs but runs
 * an arbitrary expression against window.GAME and prints the JSON result.
 * Used to diagnose "the thing is not on screen" without guessing.
 *
 *   node qa/probe.mjs "GAME.ctx.viewScene.children.length"
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};
const EXPR = argv.filter((a) => !a.startsWith('--') && argv[argv.indexOf(a) - 1]?.startsWith('--') !== true)[0] ?? '1';
const PORT = Number(arg('port', 5198));
const SETTLE = Number(arg('settle', 4000));

function findChromium() {
  const roots = [process.env.PLAYWRIGHT_BROWSERS_PATH, '/opt/pw-browsers', `${process.env.HOME}/.cache/ms-playwright`];
  for (const root of roots) {
    if (!root || !existsSync(root)) continue;
    for (const dir of readdirSync(root)) {
      if (!/^chromium(_headless_shell)?-/.test(dir)) continue;
      const exe = resolve(root, dir, 'chrome-linux', 'chrome');
      if (existsSync(exe)) return exe;
    }
  }
  return undefined;
}

async function waitForServer(url, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch { /* not up */ }
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

const url = `http://127.0.0.1:${PORT}/`;
const server = spawn('npx', ['vite', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'], {
  cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
});
process.on('exit', () => { try { server.kill('SIGTERM'); } catch { /* gone */ } });

if (!(await waitForServer(url))) {
  console.error('[probe] dev server never came up');
  process.exit(1);
}

let executablePath;
try { if (!existsSync(chromium.executablePath())) executablePath = findChromium(); } catch { executablePath = findChromium(); }

const browser = await chromium.launch({
  executablePath,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--no-sandbox', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 640, height: 360 }, deviceScaleFactor: 1 });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
const ok = await page.waitForFunction(() => Boolean(window.GAME), null, { timeout: 120_000 }).then(() => true).catch(() => false);
if (!ok) {
  console.error('[probe] GAME never appeared');
  console.error(errors.join('\n'));
  await browser.close();
  process.exit(1);
}
await page.waitForTimeout(SETTLE);

const result = await page.evaluate((expr) => {
  try {
    // eslint-disable-next-line no-new-func
    const fn = new Function('GAME', 'THREE', `return (${expr});`);
    const out = fn(window.GAME, window.THREE);
    return JSON.stringify(out, (k, v) => (typeof v === 'function' ? '[fn]' : v), 2);
  } catch (e) {
    return `EVAL ERROR: ${e && e.stack ? e.stack : e}`;
  }
}, EXPR);

console.log(result);
if (errors.length) {
  console.log(`\n[probe] ${errors.length} console error(s):`);
  for (const e of errors.slice(0, 30)) console.log('  ' + e);
}
await browser.close();
process.exit(0);
