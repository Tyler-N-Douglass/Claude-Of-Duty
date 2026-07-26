/**
 * The single stylesheet for every piece of UI in the game, plus the small DOM
 * and easing helpers the HUD and menu share.
 *
 * Design language: monochrome warm-white on near-black, one amber accent, one
 * semantic red. Thin 1px strokes with a hard dark shadow underneath so nothing
 * disappears over a blown-out sky. Tight tracking, tabular numerals, no browser
 * defaults anywhere. Everything that moves moves on a cubic-bezier, never
 * linearly.
 */

// ---------------------------------------------------------------------------
// Palette — mirrored in CSS custom properties below. Canvas-drawn UI (minimap,
// blood overlay, grain) reads these so JS and CSS can never drift apart.
// ---------------------------------------------------------------------------

export const PALETTE = {
  white: '#f3f1ea',
  ink: '#05070a',
  accent: '#e2a53d',
  danger: '#e0413a',
  head: '#ffc94d',
  friend: '#8fb8dd',
  /** Minimap layers. */
  mapVoid: '#080b0e',
  mapFloor: '#171d23',
  mapUpper: '#242c34',
  mapWall: '#c6cfd7',
  mapGrid: 'rgba(198,207,215,0.055)',
} as const;

/** Named cubic-beziers. `punch` overshoots — used for anything that "hits". */
export const EASE = {
  out: 'cubic-bezier(.16,.84,.44,1)',
  outSoft: 'cubic-bezier(.22,.61,.36,1)',
  punch: 'cubic-bezier(.18,1.42,.4,1)',
  in: 'cubic-bezier(.55,.06,.68,.19)',
  inOut: 'cubic-bezier(.65,.05,.36,1)',
} as const;

// ---------------------------------------------------------------------------
// Maths helpers
// ---------------------------------------------------------------------------

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function smoothstep(t: number): number {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

export function easeOutCubic(t: number): number {
  const x = 1 - clamp01(t);
  return 1 - x * x * x;
}

/**
 * Frame-rate independent exponential approach. `rate` is roughly "how many
 * e-foldings per second", so 12 settles in about a quarter second.
 */
export function damp(current: number, target: number, rate: number, dt: number): number {
  return target + (current - target) * Math.exp(-rate * dt);
}

/** Shortest signed difference between two angles, radians. */
export function angleDelta(a: number, b: number): number {
  let d = (a - b) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  else if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  parent?: Element | null,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

export function svgNode<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs?: Record<string, string | number>,
  parent?: Element | null,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const k in attrs) node.setAttribute(k, String(attrs[k]));
  }
  if (parent) parent.appendChild(node);
  return node;
}

/** Writes text only when it actually differs — avoids needless style recalc. */
export function setText(node: HTMLElement, value: string): void {
  if (node.textContent !== value) node.textContent = value;
}

/** Toggles a class only on change. `classList.toggle` alone already does this,
 * but going through here keeps the intent explicit at call sites. */
export function setClass(node: HTMLElement, name: string, on: boolean): void {
  if (node.classList.contains(name) !== on) node.classList.toggle(name, on);
}

/** A rifle silhouette built from primitives — no external icon assets. */
export function weaponGlyph(kind: 'rifle' | 'explosive' | 'blade', parent?: Element | null): SVGSVGElement {
  const root = svgNode('svg', { viewBox: '0 0 30 12', class: 'cod-glyph' }, parent);
  const add = (t: 'rect' | 'path' | 'circle', a: Record<string, string | number>): void => {
    svgNode(t, a, root);
  };
  if (kind === 'rifle') {
    add('rect', { x: 15.6, y: 4.1, width: 13.4, height: 1.15, rx: 0.3 });
    add('rect', { x: 25.4, y: 2.5, width: 1.1, height: 1.8, rx: 0.25 });
    add('rect', { x: 6.4, y: 3.6, width: 9.6, height: 3.1, rx: 0.5 });
    add('rect', { x: 7.6, y: 1.9, width: 5.2, height: 1.2, rx: 0.35 });
    add('path', { d: 'M0.4 4.1 L6.4 3.7 L6.4 6.4 L1.6 6.9 Z' });
    add('path', { d: 'M9.8 6.6 L12.1 6.6 L11.2 10.6 L9.2 10.6 Z' });
    add('path', { d: 'M13.1 6.6 L16.0 6.6 L15.2 11.4 L12.4 11.4 Z' });
  } else if (kind === 'explosive') {
    add('circle', { cx: 15, cy: 7, r: 3.6 });
    add('rect', { x: 13.8, y: 1.2, width: 2.4, height: 2.2, rx: 0.4 });
    add('path', { d: 'M15 0.2 v2 M9 7 h-3 M21 7 h3 M10.8 2.8 l-2.1-2.1 M19.2 2.8 l2.1-2.1 M10.8 11.2 l-2.1 2.1 M19.2 11.2 l2.1 2.1', class: 'cod-glyph-stroke' });
  } else {
    add('path', { d: 'M2 10.6 L20.4 1.4 L23.6 2.6 L5.6 11.6 Z' });
    add('rect', { x: 21.8, y: 0.6, width: 6.6, height: 1.7, rx: 0.5 });
  }
  return root;
}

// ---------------------------------------------------------------------------
// Procedural textures used by the UI (grain, blood). Generated once, cached.
// ---------------------------------------------------------------------------

let grainCache: string | null = null;

/** Monochrome film grain tile, used behind the menu. */
export function grainDataURI(): string {
  if (grainCache) return grainCache;
  const size = 128;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const g = c.getContext('2d');
  if (!g) return '';
  const img = g.createImageData(size, size);
  const d = img.data;
  let seed = 0x9e3779b1;
  for (let i = 0; i < size * size; i++) {
    // xorshift — cheap and repeatable, so the tile is identical every boot.
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    const n = (seed >>> 0) / 0xffffffff;
    const v = n < 0.5 ? 0 : 255;
    const o = i * 4;
    d[o] = v;
    d[o + 1] = v;
    d[o + 2] = v;
    d[o + 3] = Math.round(n * n * 46);
  }
  g.putImageData(img, 0, 0);
  grainCache = c.toDataURL('image/png');
  return grainCache;
}

/**
 * Blood on the lens: heavy at the corners, thin toward the centre, with a few
 * gravity drips. `variant` reseeds so two overlays can cross-fade without the
 * repeat being obvious.
 */
export function bloodDataURI(variant: number, width = 768, height = 432): string {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  const g = c.getContext('2d');
  if (!g) return '';

  let s = 1234567 + variant * 7919;
  const rnd = (): number => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 100000) / 100000;
  };

  const splat = (x: number, y: number, r: number, a: number): void => {
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, `rgba(74,4,3,${a})`);
    grad.addColorStop(0.55, `rgba(58,3,3,${a * 0.8})`);
    grad.addColorStop(0.86, `rgba(44,2,2,${a * 0.3})`);
    grad.addColorStop(1, 'rgba(38,2,2,0)');
    g.fillStyle = grad;
    g.beginPath();
    // Lobed blob rather than a circle: a perfect disc reads as a bug.
    const lobes = 7;
    for (let i = 0; i <= 64; i++) {
      const t = (i / 64) * Math.PI * 2;
      const wob = 1 + 0.32 * Math.sin(t * lobes + variant) + 0.18 * Math.sin(t * 3.3 + x * 0.01);
      const px = x + Math.cos(t) * r * wob;
      const py = y + Math.sin(t) * r * wob * 0.86;
      if (i === 0) g.moveTo(px, py);
      else g.lineTo(px, py);
    }
    g.closePath();
    g.fill();
  };

  // Everything is pushed to the rim: `1 - r^2` biasing keeps the centre of
  // frame — where the player is actually looking — almost clear, which is what
  // separates "blood on the lens" from "someone spilled paint on the monitor".
  const rim = (): number => 1 - (1 - rnd()) * (1 - rnd()) * 0.62;

  g.filter = 'blur(4px)';
  for (let i = 0; i < 11; i++) {
    const ang = rnd() * Math.PI * 2;
    const rad = rim();
    const x = width * 0.5 + Math.cos(ang) * width * 0.54 * rad;
    const y = height * 0.5 + Math.sin(ang) * height * 0.58 * rad;
    splat(x, y, 14 + rnd() * 38, 0.42 + rnd() * 0.34);
  }
  g.filter = 'blur(1px)';
  for (let i = 0; i < 34; i++) {
    const ang = rnd() * Math.PI * 2;
    const rad = rim();
    const x = width * 0.5 + Math.cos(ang) * width * 0.56 * rad;
    const y = height * 0.5 + Math.sin(ang) * height * 0.6 * rad;
    splat(x, y, 1.6 + rnd() * 6, 0.45 + rnd() * 0.4);
  }
  // Drips: short tapered runs downward from the upper splats.
  g.filter = 'blur(1.5px)';
  for (let i = 0; i < 7; i++) {
    const x = rnd() * width;
    const y = (rnd() < 0.5 ? rnd() * 0.22 : 0.72 + rnd() * 0.24) * height;
    const len = 16 + rnd() * 64;
    const w = 1.2 + rnd() * 2.6;
    const grad = g.createLinearGradient(x, y, x, y + len);
    grad.addColorStop(0, 'rgba(68,4,3,0.5)');
    grad.addColorStop(1, 'rgba(48,2,2,0)');
    g.fillStyle = grad;
    g.beginPath();
    g.moveTo(x - w, y);
    g.quadraticCurveTo(x, y + len * 0.62, x, y + len);
    g.quadraticCurveTo(x, y + len * 0.62, x + w, y);
    g.closePath();
    g.fill();
  }
  g.filter = 'none';
  return c.toDataURL('image/png');
}

// ---------------------------------------------------------------------------
// Stylesheet
// ---------------------------------------------------------------------------

const STYLE_ID = 'cod-ui-style';
let refCount = 0;

const CSS = `
.cod-ui{
  --w:${PALETTE.white};
  --dim:rgba(243,241,234,.62);
  --faint:rgba(243,241,234,.30);
  --ghost:rgba(243,241,234,.12);
  --hair:rgba(243,241,234,.20);
  --ink:${PALETTE.ink};
  --accent:${PALETTE.accent};
  --accent-dim:rgba(226,165,61,.42);
  --danger:${PALETTE.danger};
  --head:${PALETTE.head};
  --friend:${PALETTE.friend};
  --sh:0 1px 2px rgba(0,0,0,.92),0 0 10px rgba(0,0,0,.5);
  --sh-hard:0 0 0 1px rgba(0,0,0,.72);
  --font:"Roboto Condensed","Barlow Condensed","Archivo Narrow","Liberation Sans Narrow","Arial Narrow",
         "Helvetica Neue",Inter,system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,"SF Mono","Roboto Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
  --hs:1;
  position:fixed; inset:0; margin:0; padding:0;
  font-family:var(--font);
  font-synthesis:none;
  color:var(--w);
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
  font-variant-numeric:tabular-nums;
  font-feature-settings:"tnum" 1,"ss01" 1;
  user-select:none; -webkit-user-select:none;
  overflow:hidden;
}
.cod-ui *{margin:0;padding:0;box-sizing:border-box;font:inherit;color:inherit;letter-spacing:inherit}
.cod-ui button{appearance:none;-webkit-appearance:none;background:none;border:0;cursor:pointer;
  font-family:var(--font);outline:none}
.cod-ui button:focus-visible{outline:1px solid var(--accent-dim);outline-offset:2px}
.cod-ui svg{display:block;fill:currentColor}
.cod-glyph{width:26px;height:10.4px;opacity:.82}
.cod-glyph-stroke{fill:none;stroke:currentColor;stroke-width:.9;stroke-linecap:round}

/* =======================================================================
   HUD root
   ======================================================================= */
.cod-hud{z-index:20;pointer-events:none;transition:opacity .28s ${EASE.out},filter .28s ${EASE.out}}
.cod-hud.muted{opacity:.16;filter:blur(1.5px) saturate(.4)}
.cod-hud.hidden{opacity:0}
.cod-corner{position:absolute;transform-origin:var(--o,50% 50%);transform:scale(var(--hs))}

/* ---- crosshair -------------------------------------------------------- */
.cod-xh{position:absolute;left:50%;top:50%;width:0;height:0;--gap:9px;
  transition:opacity .09s linear;will-change:opacity}
.cod-xh.ads{opacity:0}
.cod-xh.hide{opacity:0}
.cod-xh i{position:absolute;display:block;background:var(--w);box-shadow:var(--sh-hard),0 0 5px rgba(0,0,0,.55)}
.cod-xh i.v{width:1px;height:7px;margin-left:-.5px;margin-top:-3.5px}
.cod-xh i.h{width:7px;height:1px;margin-left:-3.5px;margin-top:-.5px}
.cod-xh i.up{transform:translateY(calc(-1 * (var(--gap) + 3.5px)))}
.cod-xh i.dn{transform:translateY(calc(var(--gap) + 3.5px))}
.cod-xh i.lf{transform:translateX(calc(-1 * (var(--gap) + 3.5px)))}
.cod-xh i.rt{transform:translateX(calc(var(--gap) + 3.5px))}
.cod-xh b{position:absolute;width:2px;height:2px;margin:-1px 0 0 -1px;border-radius:50%;
  background:var(--w);box-shadow:var(--sh-hard)}

/* ---- hitmarker -------------------------------------------------------- */
.cod-hm{position:absolute;left:50%;top:50%;width:0;height:0;opacity:0;--w-:1.5px;--len:9px;--r:7px;--c:var(--w)}
.cod-hm.head{--w-:2.4px;--len:11px;--r:8px;--c:var(--head)}
.cod-hm.lethal{--w-:2.6px;--len:12px;--r:7px;--c:var(--danger)}
.cod-hm i{position:absolute;display:block;width:var(--w-);height:var(--len);
  margin-left:calc(var(--w-) / -2);margin-top:calc(var(--len) / -2);
  background:var(--c);box-shadow:var(--sh-hard),0 0 6px rgba(0,0,0,.6)}
.cod-hm i.a{transform:rotate(45deg) translateY(calc(-1 * var(--r)))}
.cod-hm i.b{transform:rotate(135deg) translateY(calc(-1 * var(--r)))}
.cod-hm i.c{transform:rotate(225deg) translateY(calc(-1 * var(--r)))}
.cod-hm i.d{transform:rotate(315deg) translateY(calc(-1 * var(--r)))}
.cod-hm u{position:absolute;left:-15px;top:-15px;width:30px;height:30px;border-radius:50%;
  border:1.5px solid var(--danger);opacity:0}
.cod-hm.lethal u{opacity:1}

/* ---- damage ----------------------------------------------------------- */
/* Gradient stops are sized so the outermost stop lands ON the frame edge — an
   ellipse larger than the viewport never reaches full strength on screen. */
.cod-dmg{position:absolute;inset:0;opacity:0;will-change:opacity;
  background:
    radial-gradient(82% 76% at 50% 50%,transparent 20%,rgba(124,11,8,.34) 62%,rgba(112,8,6,.95) 100%),
    radial-gradient(62% 46% at 50% 104%,rgba(158,18,13,.42),transparent 74%)}
.cod-blood{position:absolute;inset:0;opacity:0;background-size:cover;background-position:center;
  mix-blend-mode:multiply;
  -webkit-mask-image:radial-gradient(96% 88% at 50% 50%,transparent 26%,#000 82%);
  mask-image:radial-gradient(96% 88% at 50% 50%,transparent 26%,#000 82%)}
.cod-blood.b{transform:scaleX(-1) scale(1.06)}
.cod-beat{position:absolute;inset:0;opacity:0;will-change:opacity;
  background:radial-gradient(86% 80% at 50% 50%,transparent 42%,rgba(172,18,14,.62) 100%);
  animation:cod-beat 1s ${EASE.out} infinite;animation-play-state:paused}
.cod-dirs{position:absolute;left:50%;top:50%;width:0;height:0}
.cod-dir{position:absolute;left:-140px;top:-140px;width:280px;height:280px;opacity:0;
  color:var(--danger);will-change:transform,opacity}
.cod-dir svg{width:280px;height:280px;overflow:visible;
  filter:drop-shadow(0 0 10px rgba(140,10,6,.55))}
.cod-dir path{fill:none;stroke:currentColor;stroke-linecap:round}
.cod-dir path.o{stroke:rgba(0,0,0,.62);stroke-width:15}
.cod-dir path.i{stroke-width:8.5}
.cod-dir path.g{stroke:rgba(255,226,220,.92);stroke-width:2}

/* ---- compass ---------------------------------------------------------- */
.cod-compass{position:absolute;top:13px;left:50%;width:min(48vw,620px);height:38px;
  transform:translateX(-50%) scale(var(--hs));transform-origin:50% 0;
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent);
  mask-image:linear-gradient(90deg,transparent,#000 12%,#000 88%,transparent);
  transition:opacity .3s ${EASE.out}}
.cod-compass .track{position:absolute;top:0;left:0;height:38px;will-change:transform}
.cod-compass .tk{position:absolute;top:30px;width:1px;height:4px;background:var(--faint);
  transform:translateX(-.5px)}
.cod-compass .tk.mj{top:27px;height:7px;background:rgba(243,241,234,.34)}
.cod-compass .lb{position:absolute;top:9px;font-size:12px;font-weight:600;letter-spacing:.14em;
  color:var(--dim);text-shadow:var(--sh);transform:translateX(-50%)}
.cod-compass .lb.card{font-size:14.5px;color:var(--w);font-weight:700;top:7px}
.cod-compass .lb.n{color:var(--accent)}
.cod-compass .pip{position:absolute;top:28px;left:0;width:7px;height:7px;margin-left:-3.5px;
  background:var(--danger);box-shadow:0 0 0 1px rgba(0,0,0,.65),0 0 7px rgba(224,65,58,.85);
  transform:rotate(45deg);opacity:0;will-change:transform,opacity}
.cod-compass .caret{position:absolute;left:50%;top:0;width:0;height:0;
  border-left:5px solid transparent;border-right:5px solid transparent;border-top:6px solid var(--w);
  transform:translateX(-50%);filter:drop-shadow(0 1px 1px rgba(0,0,0,.9))}
.cod-compass .rule{position:absolute;left:0;right:0;top:26px;height:1px;
  background:linear-gradient(90deg,transparent,var(--ghost) 14%,var(--ghost) 86%,transparent)}

/* ---- killfeed --------------------------------------------------------- */
.cod-killfeed{top:52px;right:26px;--o:100% 0;display:flex;flex-direction:column;align-items:flex-end;gap:3px}
.cod-kf{display:flex;align-items:center;gap:8px;padding:3px 8px 3px 9px;
  background:linear-gradient(90deg,rgba(6,8,11,0) 0%,rgba(6,8,11,.62) 30%,rgba(6,8,11,.72) 100%);
  border-right:2px solid var(--hair);
  font-size:13px;font-weight:600;letter-spacing:.055em;text-transform:uppercase;text-shadow:var(--sh);
  white-space:nowrap;will-change:transform,opacity}
.cod-kf .a{color:var(--dim)}
.cod-kf .b{color:var(--dim)}
.cod-kf.mine{border-right-color:var(--accent)}
.cod-kf.mine .a{color:var(--accent)}
.cod-kf.mine .b{color:var(--w)}
.cod-kf.victim{border-right-color:var(--danger)}
.cod-kf.victim .b{color:var(--danger)}
.cod-kf .hs{color:var(--head);font-size:11px;letter-spacing:.12em}

/* ---- ammo ------------------------------------------------------------- */
.cod-ammo{right:28px;bottom:24px;--o:100% 100%;text-align:right;text-shadow:var(--sh)}
.cod-ammo .name{display:flex;align-items:center;justify-content:flex-end;gap:7px;
  font-size:13px;font-weight:600;letter-spacing:.19em;color:var(--dim);text-transform:uppercase}
.cod-ammo .nums{display:flex;align-items:baseline;justify-content:flex-end;gap:5px;margin-top:1px;
  transform:scaleX(.955);transform-origin:100% 50%}
.cod-ammo .cur{font-size:46px;font-weight:700;line-height:.92;letter-spacing:-.012em;
  transition:color .16s linear}
.cod-ammo .sep{font-size:20px;color:var(--faint);font-weight:300;position:relative;top:-2px}
.cod-ammo .res{font-size:21px;font-weight:600;color:var(--dim);letter-spacing:.02em}
.cod-ammo.low .cur{color:var(--danger);animation:cod-lowammo .82s ${EASE.inOut} infinite}
.cod-ammo.empty .res{color:var(--danger)}
.cod-ammo .mag{display:flex;justify-content:flex-end;gap:2px;height:8px;margin-top:7px}
.cod-ammo .mag i{display:block;width:2px;height:8px;background:var(--w);opacity:.9;
  box-shadow:0 1px 1px rgba(0,0,0,.8);transition:opacity .1s linear,background-color .1s linear}
.cod-ammo .mag i.off{opacity:.16;background:var(--w);height:5px;margin-top:3px}
.cod-ammo.low .mag i:not(.off){background:var(--danger)}
.cod-ammo .mode{margin-top:6px;font-size:11px;letter-spacing:.22em;color:var(--faint);font-weight:600}
.cod-ammo .rl{position:relative;height:2px;margin-top:7px;background:var(--ghost);overflow:hidden;opacity:0;
  transition:opacity .16s linear}
.cod-ammo .rl i{position:absolute;inset:0;background:var(--accent);transform:scaleX(0);transform-origin:0 50%}
.cod-ammo.reloading .rl{opacity:1}

/* ---- reload prompt ---------------------------------------------------- */
.cod-prompt{position:absolute;left:50%;top:calc(50% + 54px);transform:translateX(-50%);
  font-size:13px;font-weight:700;letter-spacing:.26em;text-transform:uppercase;
  color:var(--w);text-shadow:var(--sh);opacity:0;transition:opacity .16s ${EASE.out}}
.cod-prompt.on{opacity:1;animation:cod-promptpulse 1.1s ${EASE.inOut} infinite}
.cod-prompt em{font-style:normal;color:var(--accent)}

/* ---- minimap ---------------------------------------------------------- */
.cod-minimap{left:26px;bottom:24px;--o:0 100%;width:178px;height:178px}
.cod-mm-frame{position:absolute;inset:0;overflow:hidden;
  background:${PALETTE.mapVoid};
  box-shadow:0 0 0 1px rgba(0,0,0,.85),0 0 0 2px rgba(243,241,234,.14),0 10px 26px rgba(0,0,0,.55);
  clip-path:polygon(0 0,100% 0,100% calc(100% - 13px),calc(100% - 13px) 100%,0 100%)}
.cod-mm-world{position:absolute;left:50%;top:50%;width:0;height:0;will-change:transform}
.cod-mm-world canvas{position:absolute;image-rendering:auto}
.cod-mm-pip{position:absolute;left:0;top:0;width:8px;height:8px;margin:-4px 0 0 -4px;background:var(--danger);
  transform:rotate(45deg);box-shadow:0 0 0 1px rgba(0,0,0,.7),0 0 8px rgba(224,65,58,.8);opacity:0;
  will-change:transform,opacity}
.cod-mm-you{position:absolute;left:50%;top:50%;width:0;height:0}
.cod-mm-you .cone{position:absolute;left:-26px;top:-46px;width:52px;height:52px;
  background:conic-gradient(from -32deg at 50% 100%,rgba(243,241,234,.20),rgba(243,241,234,0) 64deg);
  -webkit-mask-image:radial-gradient(circle at 50% 100%,#000 12%,transparent 76%);
  mask-image:radial-gradient(circle at 50% 100%,#000 12%,transparent 76%)}
.cod-mm-you .arw{position:absolute;left:-6px;top:-7px;width:12px;height:14px;
  background:var(--w);clip-path:polygon(50% 0,100% 100%,50% 78%,0 100%);
  filter:drop-shadow(0 0 2px rgba(0,0,0,.9))}
.cod-mm-cards{position:absolute;inset:0;will-change:transform}
.cod-mm-cards span{position:absolute;left:50%;top:50%;font-size:10px;font-weight:700;letter-spacing:.1em;
  color:var(--faint);text-shadow:0 1px 2px #000}
.cod-mm-cards span.n{color:var(--accent)}
.cod-mm-grade{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(120% 120% at 50% 50%,transparent 42%,rgba(0,0,0,.55) 100%)}
.cod-mm-tag{position:absolute;left:1px;top:-15px;font-size:10px;letter-spacing:.22em;color:var(--faint);
  font-weight:700;text-shadow:var(--sh)}

/* ---- notifications / objective ---------------------------------------- */
.cod-notify{position:absolute;left:50%;top:96px;transform:translateX(-50%);width:min(56vw,720px);
  display:flex;flex-direction:column;align-items:center;gap:2px;text-align:center;opacity:0}
.cod-notify .rule{width:0;height:1px;background:var(--accent);opacity:.7;transition:width .5s ${EASE.out}}
.cod-notify.on .rule{width:118px}
.cod-notify .kind{font-size:11px;font-weight:700;letter-spacing:.34em;color:var(--accent);
  text-shadow:var(--sh);text-transform:uppercase}
.cod-notify .txt{font-size:19px;font-weight:600;letter-spacing:.13em;color:var(--w);
  text-shadow:var(--sh);text-transform:uppercase}
.cod-notify.info .kind{color:var(--dim)}
.cod-notify.info .rule{background:var(--dim)}

/* ---- kill confirm / streak -------------------------------------------- */
.cod-confirm{position:absolute;left:50%;top:calc(50% - 74px);transform:translateX(-50%);
  display:flex;flex-direction:column;align-items:center;gap:1px;opacity:0;will-change:transform,opacity}
.cod-confirm .k{font-size:15px;font-weight:700;letter-spacing:.3em;color:var(--w);text-shadow:var(--sh)}
.cod-confirm .p{font-size:12px;font-weight:700;letter-spacing:.16em;color:var(--accent);text-shadow:var(--sh)}
.cod-confirm.head .k{color:var(--head)}
.cod-streak{right:28px;bottom:146px;--o:100% 100%;display:flex;align-items:center;gap:9px;
  opacity:0;transition:opacity .3s ${EASE.out};text-shadow:var(--sh)}
.cod-streak.on{opacity:1}
.cod-streak .lb{font-size:11px;font-weight:700;letter-spacing:.24em;color:var(--faint)}
.cod-streak .n{font-size:23px;font-weight:700;letter-spacing:.02em;color:var(--accent);
  min-width:26px;text-align:right;will-change:transform}
.cod-streak .bar{width:2px;height:20px;background:var(--accent-dim)}

/* ---- death overlay ---------------------------------------------------- */
.cod-death{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:8px;opacity:0;background:radial-gradient(78% 64% at 50% 50%,
  rgba(34,4,4,.58),rgba(3,3,4,.93));backdrop-filter:saturate(.18) brightness(.5) blur(3px);
  -webkit-backdrop-filter:saturate(.18) brightness(.5) blur(3px);
  transition:opacity .5s ${EASE.out};visibility:hidden}
.cod-death.on{opacity:1;visibility:visible}
.cod-death .t{font-size:44px;font-weight:800;letter-spacing:.3em;color:rgba(243,241,234,.94);
  text-shadow:0 3px 24px rgba(0,0,0,.9);transform:scaleX(.94);text-transform:uppercase}
.cod-death .s{font-size:12px;font-weight:700;letter-spacing:.34em;color:var(--danger);
  text-shadow:0 1px 8px rgba(0,0,0,.9)}
.cod-death .c{font-size:11.5px;font-weight:600;letter-spacing:.26em;color:var(--faint);margin-top:10px}
.cod-death hr{width:120px;height:1px;border:0;background:rgba(224,65,58,.5);margin-top:2px}

/* =======================================================================
   Menu
   ======================================================================= */
.cod-menu{z-index:40;display:none;pointer-events:auto}
.cod-menu.on{display:block}
/* Deliberately not opaque: the live scene stays visible through the blur, which
   is what stops the menu reading as a web page pasted over a canvas. */
.cod-menu .bg{position:absolute;inset:0;background:
  radial-gradient(130% 105% at 20% 10%,rgba(18,24,30,.34),rgba(4,5,7,.84) 72%),
  linear-gradient(180deg,rgba(5,6,9,.6),rgba(4,5,7,.9));
  backdrop-filter:blur(26px) saturate(.4) brightness(.62);
  -webkit-backdrop-filter:blur(26px) saturate(.4) brightness(.62)}
.cod-menu .scan{position:absolute;inset:0;opacity:.5;
  background:repeating-linear-gradient(180deg,rgba(255,255,255,.028) 0 1px,transparent 1px 3px)}
.cod-menu .grain{position:absolute;inset:-64px;opacity:.5;background-repeat:repeat;
  animation:cod-grain 900ms steps(5) infinite}
.cod-menu .sweep{position:absolute;left:0;right:0;height:38%;top:-38%;opacity:.5;
  background:linear-gradient(180deg,transparent,rgba(226,165,61,.05) 55%,transparent);
  animation:cod-sweep 9s linear infinite}
.cod-menu .vig{position:absolute;inset:0;
  background:radial-gradient(115% 95% at 50% 50%,transparent 38%,rgba(0,0,0,.72) 100%)}
.cod-menu .wrap{position:absolute;inset:0;display:flex;flex-direction:column;
  padding:clamp(28px,5.2vh,62px) clamp(34px,6vw,110px)}

.cod-menu .brand{display:flex;flex-direction:column;gap:6px}
.cod-menu .eyebrow{display:flex;align-items:center;gap:12px;font-size:11px;font-weight:700;
  letter-spacing:.42em;color:var(--accent);text-transform:uppercase}
.cod-menu .eyebrow s{display:block;width:44px;height:1px;background:var(--accent);opacity:.8;
  text-decoration:none}
.cod-menu h1{font-size:clamp(42px,7.4vw,104px);font-weight:800;line-height:.9;letter-spacing:.055em;
  text-transform:uppercase;transform:scaleX(.93);transform-origin:0 50%;
  /* Base is bright; only a narrow highlight band travels across it. */
  background:linear-gradient(100deg,#dedad1 0%,#dedad1 34%,#ffffff 44%,#f2efe7 50%,#dedad1 62%,#dedad1 100%);
  background-size:280% 100%;-webkit-background-clip:text;background-clip:text;color:var(--w);
  -webkit-text-fill-color:transparent;animation:cod-sheen 8s ${EASE.inOut} infinite;
  filter:drop-shadow(0 3px 18px rgba(0,0,0,.75))}
.cod-menu h1 em{font-style:normal;display:block;font-size:.44em;letter-spacing:.2em;
  -webkit-text-fill-color:rgba(243,241,234,.58);color:var(--dim);background:none;filter:none;
  margin-top:.12em}
.cod-menu .sub{font-size:12px;font-weight:600;letter-spacing:.3em;color:var(--faint);
  text-transform:uppercase;margin-top:10px}

.cod-menu .body{flex:1;display:flex;align-items:center;min-height:0}
.cod-menu .body.split{align-items:flex-end;justify-content:space-between;gap:60px;padding-bottom:22px}

.cod-menu .brief{display:flex;flex-direction:column;align-items:flex-end;gap:1px;
  width:min(380px,32vw);text-align:right}
.cod-menu .brief .bt{font-size:11px;font-weight:700;letter-spacing:.38em;color:var(--accent);
  text-transform:uppercase;padding-bottom:10px;margin-bottom:6px;border-bottom:1px solid var(--ghost);
  width:100%}
.cod-menu .brief .br{display:flex;justify-content:space-between;align-items:baseline;width:100%;
  padding:6px 0;border-bottom:1px solid rgba(243,241,234,.05)}
.cod-menu .brief .br span{font-size:11px;font-weight:600;letter-spacing:.2em;color:var(--faint);
  text-transform:uppercase}
.cod-menu .brief .br b{font-size:13px;font-weight:700;letter-spacing:.12em;color:var(--dim);
  text-transform:uppercase}
.cod-menu .brief .bn{margin-top:14px;font-size:11.5px;line-height:1.7;letter-spacing:.06em;
  color:var(--faint);text-transform:none;text-align:right}
.cod-menu .foot{display:flex;justify-content:space-between;align-items:flex-end;
  font-size:11px;letter-spacing:.2em;color:var(--faint);font-weight:600;text-transform:uppercase}
.cod-menu .foot .keys{display:flex;gap:18px}
.cod-menu .foot b{color:var(--dim);font-weight:700}

.cod-menu .list{display:flex;flex-direction:column;gap:2px;min-width:340px}
.cod-menu .item{position:relative;display:flex;align-items:center;gap:16px;
  padding:13px 20px 13px 16px;cursor:pointer;background:transparent;border:0;text-align:left;
  transition:background-color .16s ${EASE.out},padding-left .2s ${EASE.out}}
.cod-menu .item::before{content:"";position:absolute;left:0;top:50%;width:2px;height:0;
  background:var(--accent);transform:translateY(-50%);transition:height .22s ${EASE.out}}
.cod-menu .item .idx{font-size:11px;font-weight:700;letter-spacing:.14em;color:var(--faint);
  font-family:var(--mono);transition:color .16s linear}
.cod-menu .item .lb{font-size:22px;font-weight:700;letter-spacing:.2em;color:var(--dim);
  text-transform:uppercase;transition:color .16s linear}
.cod-menu .item .hint{margin-left:auto;font-size:11px;letter-spacing:.18em;color:transparent;
  transition:color .16s linear}
.cod-menu .item:hover,.cod-menu .item.sel{background:linear-gradient(90deg,rgba(243,241,234,.075),transparent 72%);
  padding-left:26px}
.cod-menu .item:hover::before,.cod-menu .item.sel::before{height:74%}
.cod-menu .item:hover .lb,.cod-menu .item.sel .lb{color:var(--w)}
.cod-menu .item:hover .idx,.cod-menu .item.sel .idx{color:var(--accent)}
.cod-menu .item:hover .hint,.cod-menu .item.sel .hint{color:var(--faint)}
.cod-menu .item.danger:hover .lb,.cod-menu .item.danger.sel .lb{color:var(--danger)}
.cod-menu .item.danger:hover::before,.cod-menu .item.danger.sel::before{background:var(--danger)}

.cod-menu .col{display:flex;flex-direction:column;gap:14px;min-width:360px}
.cod-menu .plabel{font-size:12px;font-weight:700;letter-spacing:.38em;color:var(--accent);
  text-transform:uppercase;padding-bottom:11px;border-bottom:1px solid var(--ghost)}

/* panel (settings / controls) */
.cod-menu .panel{display:flex;flex-direction:column;gap:2px;width:min(760px,62vw);
  max-height:76vh;padding-right:10px;overflow-y:auto;overscroll-behavior:contain}
.cod-menu .panel::-webkit-scrollbar{width:3px}
.cod-menu .panel::-webkit-scrollbar-track{background:rgba(243,241,234,.04)}
.cod-menu .panel::-webkit-scrollbar-thumb{background:var(--hair)}
.cod-menu .panel h2{font-size:12px;font-weight:700;letter-spacing:.36em;color:var(--accent);
  text-transform:uppercase;padding-bottom:9px;margin-bottom:6px;border-bottom:1px solid var(--ghost)}
.cod-menu .panel h2:not(:first-child){margin-top:26px}
.cod-menu .body.mid{justify-content:center}
.cod-menu .phead{margin-bottom:22px}
.cod-menu .peyebrow{display:flex;align-items:center;gap:11px;font-size:11px;font-weight:700;
  letter-spacing:.4em;color:var(--accent);text-transform:uppercase;margin-bottom:7px}
.cod-menu .peyebrow s{display:block;width:34px;height:1px;background:var(--accent);opacity:.8}
.cod-menu .ptitle{font-size:34px;font-weight:800;letter-spacing:.24em;text-transform:uppercase;
  transform:scaleX(.94);transform-origin:0 50%;color:var(--w);text-shadow:0 3px 20px rgba(0,0,0,.7)}
.cod-menu .row{display:grid;grid-template-columns:228px 1fr 66px;align-items:center;gap:18px;
  padding:7px 0;border-bottom:1px solid rgba(243,241,234,.05)}
.cod-menu .row .k{font-size:13px;font-weight:600;letter-spacing:.17em;color:var(--dim);
  text-transform:uppercase}
.cod-menu .row .v{font-size:13px;font-weight:700;letter-spacing:.1em;color:var(--w);
  text-align:right;font-family:var(--mono)}
.cod-menu .row .d{grid-column:1 / -1;font-size:11px;letter-spacing:.08em;color:var(--faint);
  margin-top:-4px;text-transform:none}

.cod-menu .seg{display:flex;gap:2px}
.cod-menu .seg button{flex:1;padding:8px 4px;font-size:11.5px;font-weight:700;letter-spacing:.18em;
  text-transform:uppercase;color:var(--faint);background:rgba(243,241,234,.045);border:1px solid transparent;
  cursor:pointer;transition:color .14s linear,background-color .14s linear,border-color .14s linear}
.cod-menu .seg button:hover{color:var(--w);background:rgba(243,241,234,.1)}
.cod-menu .seg button.on{color:var(--ink);background:var(--accent);border-color:var(--accent)}

.cod-menu .sw{position:relative;width:54px;height:20px;background:rgba(243,241,234,.08);
  border:1px solid var(--ghost);cursor:pointer;transition:background-color .18s ${EASE.out},
  border-color .18s ${EASE.out}}
.cod-menu .sw i{position:absolute;left:2px;top:2px;width:22px;height:14px;background:var(--faint);
  transition:transform .2s ${EASE.punch},background-color .18s linear}
.cod-menu .sw.on{background:rgba(226,165,61,.2);border-color:var(--accent-dim)}
.cod-menu .sw.on i{transform:translateX(26px);background:var(--accent)}

.cod-menu .sl{position:relative;height:20px;cursor:pointer;display:flex;align-items:center}
.cod-menu .sl .trk{position:absolute;left:0;right:0;height:2px;background:rgba(243,241,234,.12)}
.cod-menu .sl .fil{position:absolute;left:0;height:2px;background:var(--accent);transform-origin:0 50%}
.cod-menu .sl .nub{position:absolute;width:3px;height:14px;background:var(--w);margin-left:-1.5px;
  box-shadow:0 0 0 1px rgba(0,0,0,.6);transition:height .14s ${EASE.out}}
.cod-menu .sl:hover .nub{height:20px}
.cod-menu .sl .tick{position:absolute;width:1px;height:5px;background:rgba(243,241,234,.16);top:14px}

.cod-menu .binds{display:grid;grid-template-columns:repeat(2,1fr);gap:2px 42px}
.cod-menu .bind{display:flex;align-items:center;justify-content:space-between;gap:14px;
  padding:7px 0;border-bottom:1px solid rgba(243,241,234,.05)}
.cod-menu .bind .k{font-size:12.5px;font-weight:600;letter-spacing:.16em;color:var(--dim);
  text-transform:uppercase}
.cod-menu .bind .v{display:flex;gap:5px}
.cod-menu .bind kbd{font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.06em;
  padding:3px 7px;color:var(--w);background:rgba(243,241,234,.07);
  border:1px solid var(--ghost);border-bottom-color:rgba(0,0,0,.5)}

.cod-menu .back{margin-top:6px;align-self:flex-start;display:flex;align-items:center;gap:10px;
  padding:9px 18px 9px 12px;cursor:pointer;background:transparent;border:1px solid var(--ghost);
  font-size:12px;font-weight:700;letter-spacing:.24em;color:var(--dim);text-transform:uppercase;
  transition:color .15s linear,border-color .15s linear,background-color .15s linear}
.cod-menu .back:hover{color:var(--w);border-color:var(--hair);background:rgba(243,241,234,.06)}

.cod-menu .center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:18px}
.cod-menu .center .big{font-size:clamp(34px,5.4vw,68px);font-weight:800;letter-spacing:.3em;
  text-transform:uppercase;transform:scaleX(.94);text-shadow:0 4px 30px rgba(0,0,0,.8)}
.cod-menu .center .lead{font-size:12px;font-weight:600;letter-spacing:.32em;color:var(--accent);
  text-transform:uppercase}
.cod-menu .center .stat{display:flex;gap:44px;margin-top:6px}
.cod-menu .center .stat div{display:flex;flex-direction:column;align-items:center;gap:4px}
.cod-menu .center .stat b{font-size:26px;font-weight:700;color:var(--w)}
.cod-menu .center .stat span{font-size:10.5px;font-weight:600;letter-spacing:.22em;color:var(--faint);
  text-transform:uppercase}

/* dev stats */
/* Sized to its content: it shares the .cod-ui base, whose inset:0 would
   otherwise stretch the panel — and its background — over the whole viewport. */
.cod-stats{position:fixed;left:14px;top:12px;right:auto;bottom:auto;
  width:max-content;height:max-content;z-index:45;display:none;pointer-events:none;
  font-family:var(--mono);font-size:11px;line-height:1.55;letter-spacing:.02em;
  color:rgba(243,241,234,.72);text-shadow:0 1px 2px rgba(0,0,0,.95);
  background:rgba(4,6,9,.42);border-left:1px solid var(--accent-dim);padding:6px 12px 6px 9px;
  white-space:pre}
.cod-stats.on{display:block}

/* prompt to re-acquire pointer lock */
.cod-lock{position:absolute;inset:0;z-index:44;display:none;align-items:center;justify-content:center;
  background:rgba(4,5,7,.55);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
  cursor:pointer;pointer-events:auto}
.cod-lock.on{display:flex}
.cod-lock span{font-size:14px;font-weight:700;letter-spacing:.34em;text-transform:uppercase;
  color:var(--w);text-shadow:var(--sh);animation:cod-promptpulse 1.4s ${EASE.inOut} infinite}

/* =======================================================================
   Keyframes
   ======================================================================= */
@keyframes cod-beat{
  0%{opacity:0}
  7%{opacity:1}
  16%{opacity:.32}
  26%{opacity:.86}
  46%{opacity:.06}
  100%{opacity:0}
}
@keyframes cod-lowammo{0%,100%{opacity:1}50%{opacity:.42}}
@keyframes cod-promptpulse{0%,100%{opacity:1}50%{opacity:.45}}
@keyframes cod-grain{
  0%{transform:translate3d(0,0,0)}20%{transform:translate3d(-14px,7px,0)}
  40%{transform:translate3d(9px,-12px,0)}60%{transform:translate3d(-7px,-6px,0)}
  80%{transform:translate3d(12px,10px,0)}100%{transform:translate3d(0,0,0)}
}
@keyframes cod-sweep{0%{top:-38%}100%{top:100%}}
@keyframes cod-sheen{0%,58%{background-position:170% 0}92%,100%{background-position:-55% 0}}
`;

export function injectUIStyles(): void {
  refCount++;
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = CSS;
  document.head.appendChild(style);
}

export function releaseUIStyles(): void {
  refCount = Math.max(0, refCount - 1);
  if (refCount > 0) return;
  document.getElementById(STYLE_ID)?.remove();
}
