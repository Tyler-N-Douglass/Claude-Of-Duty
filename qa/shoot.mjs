#!/usr/bin/env node
/**
 * Visual QA harness.
 *
 * Boots the dev server, opens the game in headless Chromium with a software
 * WebGL backend, drives the camera to a set of fixed poses and writes PNGs to
 * qa/shots/. Any critic agent can run this and then Read the images.
 *
 *   node qa/shoot.mjs                    # all poses
 *   node qa/shoot.mjs --pose hero        # one pose
 *   node qa/shoot.mjs --out qa/shots/r3  # custom output dir
 *   node qa/shoot.mjs --settle 4000      # wait longer before capturing
 *
 * Poses live in qa/poses.json so they stay stable across runs — a critic
 * comparing round N to round N+1 must be looking at the same framing.
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, existsSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const argv = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};
const flag = (name) => argv.includes(`--${name}`);

const OUT = resolve(ROOT, arg('out', 'qa/shots'));
const WIDTH = Number(arg('width', 1920));
const HEIGHT = Number(arg('height', 1080));
const SETTLE = Number(arg('settle', 6000));
const ONLY = arg('pose', null);
const PORT = Number(arg('port', 5199));
const KEEP = flag('keep-server');

const POSES_PATH = resolve(ROOT, 'qa/poses.json');
const DEFAULT_POSES = [
  { name: 'hero', pos: [0, 1.65, 8], yaw: 0, pitch: -0.03, desc: 'Establishing shot down the main street' },
  { name: 'weapon', pos: [2, 1.65, 4], yaw: 0.6, pitch: -0.05, desc: 'Viewmodel readability, hip fire' },
  { name: 'interior', pos: [-8, 1.65, -6], yaw: 2.2, pitch: 0.0, desc: 'Indoor lighting, bounce, shadow contact' },
  { name: 'skyline', pos: [4, 1.65, -18], yaw: 3.1, pitch: 0.12, desc: 'Sky, atmosphere, distant fog' },
  { name: 'detail', pos: [-2, 1.4, 2], yaw: 1.1, pitch: -0.35, desc: 'Close surface detail: normals, roughness, AO' },
];

function loadPoses() {
  if (existsSync(POSES_PATH)) {
    try {
      return JSON.parse(readFileSync(POSES_PATH, 'utf8'));
    } catch (e) {
      console.warn(`[qa] poses.json unreadable (${e.message}); using defaults`);
    }
  }
  writeFileSync(POSES_PATH, JSON.stringify(DEFAULT_POSES, null, 2));
  return DEFAULT_POSES;
}

async function waitForServer(url, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  return false;
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  const poses = loadPoses().filter((p) => !ONLY || p.name === ONLY);
  if (poses.length === 0) {
    console.error(`[qa] no pose named "${ONLY}"`);
    process.exit(2);
  }

  const url = `http://127.0.0.1:${PORT}/`;
  console.log(`[qa] starting vite on ${PORT}`);
  const server = spawn('npx', ['vite', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let serverLog = '';
  server.stdout.on('data', (d) => (serverLog += d));
  server.stderr.on('data', (d) => (serverLog += d));

  const shutdown = () => {
    if (!KEEP) try { server.kill('SIGTERM'); } catch { /* already gone */ }
  };
  process.on('exit', shutdown);

  if (!(await waitForServer(url))) {
    console.error('[qa] dev server never came up:\n' + serverLog);
    shutdown();
    process.exit(1);
  }

  // Playwright's bundled-browser revision can drift from what this image has
  // installed. Fall back to any chromium build present under the browsers path
  // rather than failing the whole capture on a version number.
  const findChromium = () => {
    const roots = [process.env.PLAYWRIGHT_BROWSERS_PATH, '/opt/pw-browsers', `${process.env.HOME}/.cache/ms-playwright`];
    for (const root of roots) {
      if (!root || !existsSync(root)) continue;
      for (const dir of readdirSync(root)) {
        if (!/^chromium(_headless_shell)?-/.test(dir)) continue;
        const exe = resolve(root, dir, 'chrome-linux', 'chrome');
        if (existsSync(exe)) return exe;
        const shell = resolve(root, dir, 'chrome-linux', 'headless_shell');
        if (existsSync(shell)) return shell;
      }
    }
    return undefined;
  };

  let executablePath;
  try {
    chromium.executablePath();
    if (!existsSync(chromium.executablePath())) executablePath = findChromium();
  } catch {
    executablePath = findChromium();
  }
  if (executablePath) console.log(`[qa] using chromium at ${executablePath}`);

  const browser = await chromium.launch({
    executablePath,
    args: [
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
      '--ignore-gpu-blocklist',
      '--enable-webgl',
      '--disable-dev-shm-usage',
      '--no-sandbox',
    ],
  });
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });

  const consoleErrors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') consoleErrors.push(m.text());
  });
  page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));

  console.log('[qa] loading game');
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });

  const ready = await page
    .waitForFunction(() => Boolean(window.GAME), null, { timeout: 120_000 })
    .then(() => true)
    .catch(() => false);

  if (!ready) {
    const body = await page.evaluate(() => document.body.innerText.slice(0, 4000)).catch(() => '');
    console.error('[qa] window.GAME never appeared. Page said:\n' + body);
    console.error('[qa] console errors:\n' + consoleErrors.join('\n'));
    await page.screenshot({ path: `${OUT}/FAILED-boot.png` });
    await browser.close();
    shutdown();
    process.exit(1);
  }

  // Let TAA converge, textures generate, and the sun/IBL settle.
  console.log(`[qa] settling ${SETTLE}ms`);
  await page.waitForTimeout(SETTLE);

  const report = { poses: [], errors: consoleErrors, fps: null };

  for (const pose of poses) {
    await page.evaluate((p) => {
      const g = window.GAME;
      // Prefer an explicit debug hook if the player system exposes one.
      if (typeof g.setDebugPose === 'function') {
        g.setDebugPose(p);
        return;
      }
      const cam = g.ctx.camera;
      cam.position.set(p.pos[0], p.pos[1], p.pos[2]);
      cam.rotation.order = 'YXZ';
      cam.rotation.set(p.pitch, p.yaw, 0);
      cam.updateMatrixWorld(true);
      const vc = g.ctx.viewCamera;
      if (vc) {
        vc.position.copy(cam.position);
        vc.rotation.copy(cam.rotation);
        vc.updateMatrixWorld(true);
      }
    }, pose);

    // Several frames so TAA/motion blur history is clean at the new pose.
    await page.waitForTimeout(1200);

    const file = `${OUT}/${pose.name}.png`;
    await page.screenshot({ path: file });
    console.log(`[qa] wrote ${file}  — ${pose.desc}`);
    report.poses.push({ ...pose, file });
  }

  report.fps = await page.evaluate(() => window.GAME?.ctx?.time?.fps ?? null).catch(() => null);
  report.drawCalls = await page
    .evaluate(() => window.GAME?.ctx?.renderer?.info?.render?.calls ?? null)
    .catch(() => null);
  report.triangles = await page
    .evaluate(() => window.GAME?.ctx?.renderer?.info?.render?.triangles ?? null)
    .catch(() => null);

  writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 2));
  console.log(`[qa] fps≈${report.fps?.toFixed?.(1)} calls=${report.drawCalls} tris=${report.triangles}`);
  if (consoleErrors.length) {
    console.log(`[qa] ${consoleErrors.length} console error(s):`);
    for (const e of consoleErrors.slice(0, 20)) console.log('   ' + e);
  }

  await browser.close();
  shutdown();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
