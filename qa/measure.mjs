#!/usr/bin/env node
/**
 * Objective frame metrics.
 *
 * Several of the rubric's hard fails are measurable, and a number that moves
 * the wrong way between rounds is worth more than any amount of critic prose.
 * This computes them for a capture directory and writes metrics.json alongside
 * the PNGs, so rounds can be diffed instead of argued about.
 *
 *   node qa/measure.mjs qa/shots/r3a
 *   node qa/measure.mjs qa/shots/r3a --vs qa/shots/r2      # diff two rounds
 *   node qa/measure.mjs qa/shots/r3a --probe hero:400,200,120,80
 *
 * Zero dependencies — the PNG decoder is inline, because pulling an image
 * library in just to read a screenshot is not worth the install.
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { inflateSync } from 'node:zlib';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const arg = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : d;
};

// ---------------------------------------------------------------------------
// Minimal PNG decoder: 8-bit, non-interlaced, colour type 2 (RGB) or 6 (RGBA).
// That is what every headless screenshot in this project produces.
// ---------------------------------------------------------------------------

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

function decodePng(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 6;
  let bitDepth = 8;
  const idat = [];

  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      if (data[12] !== 0) throw new Error('interlaced PNG unsupported');
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    pos += 12 + len;
  }

  if (bitDepth !== 8) throw new Error(`bit depth ${bitDepth} unsupported`);
  const channels = colorType === 6 ? 4 : colorType === 2 ? 3 : 0;
  if (!channels) throw new Error(`colour type ${colorType} unsupported`);

  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(height * stride);

  let rp = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[rp++];
    const line = raw.subarray(rp, rp + stride);
    rp += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;

    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? cur[x - channels] : 0;
      const b = prev ? prev[x] : 0;
      const c = prev && x >= channels ? prev[x - channels] : 0;
      const v = line[x];
      cur[x] =
        filter === 0 ? v
        : filter === 1 ? (v + a) & 255
        : filter === 2 ? (v + b) & 255
        : filter === 3 ? (v + ((a + b) >> 1)) & 255
        : (v + paeth(a, b, c)) & 255;
    }
  }
  return { width, height, channels, data: out };
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

const lumaOf = (r, g, b) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;

/**
 * The HUD is deliberately high-contrast and would skew every global statistic,
 * so the corners it occupies are excluded. These are fractions of frame size,
 * matched to where HUD.ts actually places the minimap, ammo block and compass.
 */
function isHud(x, y, w, h) {
  const fx = x / w;
  const fy = y / h;
  if (fy < 0.045) return true;                     // compass strip
  if (fx < 0.22 && fy > 0.76) return true;         // minimap
  if (fx > 0.76 && fy > 0.84) return true;         // ammo block
  return false;
}

function measure(png) {
  const { width: w, height: h, channels: ch, data } = png;
  const lums = [];
  let sumSat = 0;
  let n = 0;
  let band = 0;
  let clipped = 0;
  let crushed = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;

  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      if (isHud(x, y, w, h)) continue;
      const i = (y * w + x) * ch;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const l = lumaOf(r, g, b);
      lums.push(l);
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      sumSat += mx === 0 ? 0 : (mx - mn) / mx;
      sumR += r;
      sumG += g;
      sumB += b;
      // The rubric's "uniform mid-grey mush" failure, as a number.
      if (l >= 0.13 && l <= 0.40) band++;
      // Veiling glare. Hard clipping (all channels at 255) turns out to be
      // rare; what actually destroys these frames is a large region sitting
      // near-white with its local contrast washed away, so an arch opening and
      // the wall around it become the same value. Measure that, not clipping.
      if (l >= 0.90) clipped++;
      // Crushed blacks. p1 alone is a trap: it rewards 0.000, which does not
      // mean "a true black exists", it means pixels are sitting AT zero with
      // their shadow detail already gone. Count them.
      if (mx <= 2) crushed++;
      n++;
    }
  }

  // Local contrast, measured on 32x32 tiles. Global percentiles cannot tell a
  // structured frame from a flat pale wash: pulling a blown region from 0.95
  // down to 0.85 passes a brightness threshold while remaining featureless.
  // What actually distinguishes them is whether detail survives inside a tile.
  const TILE = 32;
  const tileSd = [];
  let flatBright = 0;
  let tiles = 0;
  for (let ty = 0; ty + TILE <= h; ty += TILE) {
    for (let tx = 0; tx + TILE <= w; tx += TILE) {
      if (isHud(tx + TILE / 2, ty + TILE / 2, w, h)) continue;
      let s1 = 0;
      let s2 = 0;
      let m = 0;
      for (let y = ty; y < ty + TILE; y += 2) {
        for (let x = tx; x < tx + TILE; x += 2) {
          const i = (y * w + x) * ch;
          const l = lumaOf(data[i], data[i + 1], data[i + 2]);
          s1 += l;
          s2 += l * l;
          m++;
        }
      }
      const mean = s1 / m;
      const sd = Math.sqrt(Math.max(0, s2 / m - mean * mean));
      tileSd.push(sd);
      // Bright and featureless: the signature of veiling glare and of distant
      // geometry that has lifted into a pale cutout.
      if (mean > 0.62 && sd < 0.025) flatBright++;
      tiles++;
    }
  }
  const localContrast = tileSd.reduce((a, v) => a + v, 0) / Math.max(tiles, 1);

  lums.sort((a, b) => a - b);
  const pct = (p) => lums[Math.min(lums.length - 1, Math.floor(lums.length * p))];

  return {
    p1: +pct(0.01).toFixed(4),
    p50: +pct(0.5).toFixed(4),
    p99: +pct(0.99).toFixed(4),
    dynamicRange: +(pct(0.99) - pct(0.01)).toFixed(4),
    midBandFraction: +(band / n).toFixed(4),
    meanSaturation: +(sumSat / n).toFixed(4),
    glareFraction: +(clipped / n).toFixed(4),
    crushedFraction: +(crushed / n).toFixed(4),
    localContrast: +localContrast.toFixed(4),
    flatBrightFraction: +(flatBright / Math.max(tiles, 1)).toFixed(4),
    // Warm/cool balance of the whole frame. Golden hour should sit slightly
    // above 1; far above it means the grade has collapsed toward sepia and the
    // frame has stopped reading as light on varied materials.
    frameRedOverBlue: +(sumR / Math.max(sumB, 1)).toFixed(3),
  };
}

function probe(png, x, y, pw, ph) {
  const { width: w, channels: ch, data } = png;
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  const ls = [];
  for (let yy = y; yy < y + ph; yy++) {
    for (let xx = x; xx < x + pw; xx++) {
      const i = (yy * w + xx) * ch;
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      ls.push(lumaOf(data[i], data[i + 1], data[i + 2]));
      n++;
    }
  }
  const mean = ls.reduce((a, v) => a + v, 0) / n;
  const sd = Math.sqrt(ls.reduce((a, v) => a + (v - mean) ** 2, 0) / n);
  return {
    luma: +mean.toFixed(4),
    stdDev: +sd.toFixed(4),
    rgb: [Math.round(r / n), Math.round(g / n), Math.round(b / n)],
    rOverB: +(r / Math.max(b, 1)).toFixed(3),
  };
}

// ---------------------------------------------------------------------------

const dir = resolve(ROOT, argv.find((a) => !a.startsWith('--')) ?? 'qa/shots/r2');
const vsDir = arg('vs', null);
const poses = JSON.parse(readFileSync(resolve(ROOT, 'qa/poses.json'), 'utf8')).map((p) => p.name);

const results = {};
for (const name of poses) {
  const file = resolve(dir, `${name}.png`);
  if (!existsSync(file)) continue;
  results[name] = measure(decodePng(readFileSync(file)));
}

const probeArg = arg('probe', null);
if (probeArg) {
  const [pose, rect] = probeArg.split(':');
  const [x, y, pw, ph] = rect.split(',').map(Number);
  const png = decodePng(readFileSync(resolve(dir, `${pose}.png`)));
  console.log(`probe ${pose} [${x},${y} ${pw}x${ph}]:`, probe(png, x, y, pw, ph));
}

writeFileSync(resolve(dir, 'metrics.json'), JSON.stringify(results, null, 2));

const HEAD = ['pose', 'p1', 'p50', 'p99', 'range', 'midBand', 'sat', 'glare', 'crush', 'lcon', 'flat'];
const pad = (s, n) => String(s).padEnd(n);
console.log(`\n${basename(dir)}`);
console.log(HEAD.map((x, i) => pad(x, i === 0 ? 10 : 9)).join(''));
for (const [name, m] of Object.entries(results)) {
  console.log(
    pad(name, 10) + [m.p1, m.p50, m.p99, m.dynamicRange, m.midBandFraction,
                     m.meanSaturation, m.glareFraction, m.crushedFraction, m.localContrast, m.flatBrightFraction]
      .map((v) => pad(v.toFixed(3), 9)).join(''),
  );
}

if (vsDir) {
  const other = resolve(ROOT, vsDir);
  console.log(`\nvs ${basename(other)}  (positive = this round is higher)`);
  console.log(HEAD.map((x, i) => pad(x, i === 0 ? 10 : 9)).join(''));
  for (const name of Object.keys(results)) {
    const f = resolve(other, `${name}.png`);
    if (!existsSync(f)) continue;
    const o = measure(decodePng(readFileSync(f)));
    const d = (a, b) => {
      const v = a - b;
      return (v >= 0 ? '+' : '') + v.toFixed(3);
    };
    const m = results[name];
    console.log(
      pad(name, 10) +
        [d(m.p1, o.p1), d(m.p50, o.p50), d(m.p99, o.p99), d(m.dynamicRange, o.dynamicRange),
         d(m.midBandFraction, o.midBandFraction), d(m.meanSaturation, o.meanSaturation),
         d(m.glareFraction, o.glareFraction), d(m.crushedFraction, o.crushedFraction),
         d(m.localContrast, o.localContrast), d(m.flatBrightFraction, o.flatBrightFraction)]
          .map((v) => pad(v, 9)).join(''),
    );
  }
}

console.log(`\nTargets: p1 0.002-0.025 (low but NOT clipped) · range > 0.75 · midBand < 0.55`);
console.log(`         glare < 0.06 · crush < 0.010 · lcon > 0.075 · flat < 0.10`);
console.log(`midBand is the fraction of non-HUD pixels inside sRGB [0.13, 0.40] —`);
console.log(`the rubric's "uniform mid-grey mush" failure expressed as a number.\n`);
