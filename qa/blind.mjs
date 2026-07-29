#!/usr/bin/env node
/**
 * Blind A/B comparison harness.
 *
 * A critic who knows which frame is the newer round will find it better. That
 * is not a judgement, it is an expectation effect, and it makes every "it
 * improved" verdict worthless. This builds shuffled pairs so the critic cannot
 * tell which is which, and keeps the key in a file the critic is not given.
 *
 *   node qa/blind.mjs --a qa/shots/r2 --b qa/shots/r3
 *   node qa/blind.mjs --a qa/shots/r2 --b qa/shots/r3 --seed 7
 *
 * Writes qa/blind/<pose>/left.png and right.png, plus:
 *   qa/blind/BRIEF.md   — hand this to the critic
 *   qa/blind/KEY.json   — which side came from which round. DO NOT show the
 *                         critic this file until after it has ruled.
 *
 * Scoring afterwards:
 *   node qa/blind.mjs --reveal --picks hero=left,weapon=right,...
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync, copyFileSync, rmSync } from 'node:fs';
import { dirname, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const arg = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : d;
};
const flag = (n) => argv.includes(`--${n}`);

const OUT = resolve(ROOT, 'qa/blind');
const KEY_PATH = resolve(OUT, 'KEY.json');

/** Deterministic PRNG so a given seed always produces the same shuffle. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function reveal() {
  if (!existsSync(KEY_PATH)) {
    console.error('[blind] no KEY.json — run a comparison first');
    process.exit(2);
  }
  const key = JSON.parse(readFileSync(KEY_PATH, 'utf8'));
  const picksRaw = arg('picks', '');
  const picks = Object.fromEntries(
    picksRaw.split(',').filter(Boolean).map((p) => p.split('=').map((s) => s.trim())),
  );

  let newerWins = 0;
  let olderWins = 0;
  console.log(`\nA = ${key.a}   B = ${key.b}   (B is the newer round)\n`);
  for (const pose of key.poses) {
    const pick = picks[pose.name];
    if (!pick) {
      console.log(`  ${pose.name.padEnd(10)} — no pick recorded`);
      continue;
    }
    const chosen = pick === 'left' ? pose.left : pose.right;
    const isNewer = chosen === 'b';
    if (isNewer) newerWins++;
    else olderWins++;
    console.log(
      `  ${pose.name.padEnd(10)} picked ${pick.padEnd(5)} -> ${isNewer ? 'NEWER' : 'OLDER'} round`,
    );
  }
  console.log(`\n  newer round preferred: ${newerWins}/${newerWins + olderWins}`);
  if (olderWins > newerWins) {
    console.log('  REGRESSION — the previous round looked better. Do not ship this round.');
  } else if (newerWins === olderWins) {
    console.log('  NO MEASURABLE IMPROVEMENT — this round was wasted effort.');
  }
  console.log();
}

function build() {
  const aDir = resolve(ROOT, arg('a', 'qa/shots/r2'));
  const bDir = resolve(ROOT, arg('b', 'qa/shots/r3'));
  const seed = Number(arg('seed', 1337));

  for (const d of [aDir, bDir]) {
    if (!existsSync(d)) {
      console.error(`[blind] missing capture dir: ${d}`);
      process.exit(2);
    }
  }

  const poses = JSON.parse(readFileSync(resolve(ROOT, 'qa/poses.json'), 'utf8')).map((p) => p.name);
  const rnd = mulberry32(seed);

  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(OUT, { recursive: true });

  const key = { a: basename(aDir), b: basename(bDir), seed, poses: [] };
  let built = 0;

  for (const name of poses) {
    const aPng = resolve(aDir, `${name}.png`);
    const bPng = resolve(bDir, `${name}.png`);
    if (!existsSync(aPng) || !existsSync(bPng)) {
      console.warn(`[blind] skipping "${name}" — missing in one of the two rounds`);
      continue;
    }
    // Coin flip per pose, so the critic cannot learn "B is always on the right"
    // after the first couple of comparisons.
    const bOnLeft = rnd() < 0.5;
    const dir = resolve(OUT, name);
    mkdirSync(dir, { recursive: true });
    copyFileSync(bOnLeft ? bPng : aPng, resolve(dir, 'left.png'));
    copyFileSync(bOnLeft ? aPng : bPng, resolve(dir, 'right.png'));
    key.poses.push({ name, left: bOnLeft ? 'b' : 'a', right: bOnLeft ? 'a' : 'b' });
    built++;
  }

  writeFileSync(KEY_PATH, JSON.stringify(key, null, 2));

  writeFileSync(
    resolve(OUT, 'BRIEF.md'),
    `# Blind comparison

There are ${built} pose folders in this directory. Each holds \`left.png\` and
\`right.png\` — the same camera position, the same map, rendered by two different
builds of the same game.

You are NOT told which build is which, and the sides were shuffled independently
per pose, so "left" is not consistently one build. Do not go looking for the
answer; the point of this exercise is that you cannot.

For each pose:

1. Say which side you would rather ship, in the sense of: which one would a
   player be less able to distinguish from a real Call of Duty frame.
2. Say WHY, naming the specific region and property that decided it — value
   separation between lit and shadowed surfaces, whether there is a true black,
   material read, edge quality, atmosphere, whichever it was.
3. If the two are genuinely indistinguishable, say so. That is a real and
   useful answer, and it means the work between these two builds did not land.
4. If one is clearly worse, say so plainly. A regression matters more than an
   improvement and must be reported loudly.

Report as: \`<pose>: left|right — <one sentence of reasoning>\`

Do not hedge, do not split the difference, and do not pick a side to be
agreeable. Pick the one that is actually better and be able to defend it.
`,
  );

  console.log(`[blind] built ${built} pose pairs in qa/blind/`);
  console.log(`[blind] key written to ${KEY_PATH} — do not show it to the critic`);
  console.log(`[blind] hand the critic qa/blind/BRIEF.md and the pose folders`);
}

if (flag('reveal')) reveal();
else build();
