/**
 * GPU-simulated particle pools.
 *
 * The CPU only ever writes a spawn record (position, velocity, curves). All
 * integration — drag, gravity, turbulence, size/colour curves, billboarding and
 * velocity stretch — happens in the vertex shader from `uTime - spawnTime`, so
 * a burst of 400 sparks costs one buffer write and zero per-frame work.
 *
 * Two pools exist because blending cannot be per-instance: one premultiplied
 * alpha pool (smoke, dust, debris, blood) which is lit by the sun, and one
 * additive pool (sparks, fire, flashes, glows) which is pure emission.
 */
import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Atlas cells. Row/col layout is resolved in cellRect(); the shader only sees
// the index and derives the tile from it.
// ---------------------------------------------------------------------------

export const CELL = {
  smoke: 0,
  smokeDense: 1,
  dust: 2,
  spark: 3,
  flash: 4,
  ring: 5,
  chip: 6,
  splinter: 7,
  shard: 8,
  droplet: 9,
  blood: 10,
  ember: 11,
  fire: 12,
  wisp: 13,
  plume: 14,
  glow: 15,
} as const;

// ---------------------------------------------------------------------------
// Deterministic noise. Everything procedural in this subsystem is seeded from
// here so the same build always produces the same textures.
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 256-entry permutation + gradient table shared by the value-noise samplers. */
class NoiseField {
  private readonly perm = new Uint8Array(512);
  private readonly vals = new Float32Array(256);

  constructor(seed: number) {
    const rng = mulberry32(seed);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;
    for (let i = 255; i > 0; i--) {
      const j = (rng() * (i + 1)) | 0;
      const t = p[i]!;
      p[i] = p[j]!;
      p[j] = t;
    }
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255]!;
    for (let i = 0; i < 256; i++) this.vals[i] = rng();
  }

  at(x: number, y: number): number {
    const xi = x & 255;
    const yi = y & 255;
    return this.vals[this.perm[(this.perm[xi]! + yi) & 511]!]!;
  }

  noise(x: number, y: number): number {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    // Quintic fade: C2 continuous, no visible grid creases under fbm.
    const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
    const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
    const a = this.at(x0, y0);
    const b = this.at(x0 + 1, y0);
    const c = this.at(x0, y0 + 1);
    const d = this.at(x0 + 1, y0 + 1);
    const top = a + (b - a) * ux;
    const bot = c + (d - c) * ux;
    return top + (bot - top) * uy;
  }

  fbm(x: number, y: number, octaves = 4, lacunarity = 2.03, gain = 0.5): number {
    let amp = 0.5;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * this.noise(x * freq, y * freq);
      norm += amp;
      amp *= gain;
      freq *= lacunarity;
    }
    return sum / norm;
  }

  /** Ridged fbm — the creases read as turbulent smoke filaments. */
  turbulence(x: number, y: number, octaves = 4): number {
    let amp = 0.5;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let i = 0; i < octaves; i++) {
      sum += amp * Math.abs(this.noise(x * freq, y * freq) * 2 - 1);
      norm += amp;
      amp *= 0.52;
      freq *= 2.07;
    }
    return sum / norm;
  }
}

export const FX_NOISE = new NoiseField(0x5eed1);

export function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });
  if (!ctx) throw new Error('[fx] 2d context unavailable');
  return { canvas, ctx };
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0 || 1e-6)));
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// Atlas painting
// ---------------------------------------------------------------------------

type CellPainter = (img: ImageData, s: number, rng: () => number) => void;

function px(img: ImageData, s: number, x: number, y: number, r: number, g: number, b: number, a: number): void {
  const i = (y * s + x) * 4;
  img.data[i] = r < 0 ? 0 : r > 255 ? 255 : r;
  img.data[i + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
  img.data[i + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
  img.data[i + 3] = a < 0 ? 0 : a > 255 ? 255 : a;
}

/** Radial falloff blob broken up by fbm — the base for every gaseous cell. */
function paintCloud(
  img: ImageData, s: number, rng: () => number,
  opts: { freq: number; warp: number; edge: number; power: number; grain: number; squashY: number; ridged: boolean },
): void {
  const ox = rng() * 128;
  const oy = rng() * 128;
  const inv = 1 / s;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const nx = (x + 0.5) * inv - 0.5;
      const ny = ((y + 0.5) * inv - 0.5) * opts.squashY;
      const r = Math.sqrt(nx * nx + ny * ny) * 2;
      const fx = (x + 0.5) * inv * opts.freq + ox;
      const fy = (y + 0.5) * inv * opts.freq + oy;
      const n = opts.ridged ? FX_NOISE.turbulence(fx, fy, 4) : FX_NOISE.fbm(fx, fy, 4);
      const detail = FX_NOISE.fbm(fx * 3.1, fy * 3.1, 3);
      let d = smoothstep(1.0, opts.edge, r + (n - 0.5) * opts.warp);
      d = Math.pow(Math.max(0, d), opts.power);
      d *= 1 - opts.grain * (1 - detail);
      // Internal luminance structure so lit smoke has body instead of reading
      // as a flat decal of a cloud.
      const lum = 0.62 + 0.38 * (n * 0.7 + detail * 0.3);
      const v = Math.round(255 * lum);
      px(img, s, x, y, v, v, v, Math.round(255 * Math.min(1, d)));
    }
  }
}

function paintRadial(
  img: ImageData, s: number,
  falloff: (r: number, ang: number, x: number, y: number) => number,
  colour?: (r: number) => [number, number, number],
): void {
  const inv = 1 / s;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const nx = (x + 0.5) * inv - 0.5;
      const ny = (y + 0.5) * inv - 0.5;
      const r = Math.sqrt(nx * nx + ny * ny) * 2;
      const ang = Math.atan2(ny, nx);
      const a = Math.min(1, Math.max(0, falloff(r, ang, nx * 2, ny * 2)));
      const c = colour ? colour(r) : ([1, 1, 1] as [number, number, number]);
      px(img, s, x, y, c[0] * 255, c[1] * 255, c[2] * 255, a * 255);
    }
  }
}

const PAINTERS: Record<number, CellPainter> = {
  [CELL.smoke]: (img, s, rng) =>
    paintCloud(img, s, rng, { freq: 3.4, warp: 0.62, edge: 0.12, power: 1.35, grain: 0.22, squashY: 1, ridged: false }),
  [CELL.smokeDense]: (img, s, rng) =>
    paintCloud(img, s, rng, { freq: 2.6, warp: 0.44, edge: 0.3, power: 0.85, grain: 0.3, squashY: 1, ridged: true }),
  [CELL.dust]: (img, s, rng) => {
    paintCloud(img, s, rng, { freq: 5.2, warp: 0.78, edge: 0.1, power: 1.6, grain: 0.42, squashY: 1, ridged: false });
    // Sparse grit: a handful of hard specks that survive minification and stop
    // dust reading as pure gaussian mush.
    for (let i = 0; i < 90; i++) {
      const a = rng() * Math.PI * 2;
      const rad = Math.pow(rng(), 0.6) * 0.44 * s;
      const cx = (s * 0.5 + Math.cos(a) * rad) | 0;
      const cy = (s * 0.5 + Math.sin(a) * rad) | 0;
      const size = 1 + ((rng() * 2) | 0);
      for (let dy = -size; dy <= size; dy++) {
        for (let dx = -size; dx <= size; dx++) {
          const x = cx + dx;
          const y = cy + dy;
          if (x < 0 || y < 0 || x >= s || y >= s) continue;
          const f = 1 - Math.sqrt(dx * dx + dy * dy) / (size + 1);
          if (f <= 0) continue;
          const i4 = (y * s + x) * 4;
          img.data[i4 + 3] = Math.min(255, img.data[i4 + 3]! + f * 150);
        }
      }
    }
  },
  [CELL.spark]: (img, s) =>
    paintRadial(img, s, (r) => Math.exp(-r * r * 13) + Math.exp(-r * r * 90) * 0.9),
  [CELL.flash]: (img, s, rng) => {
    const phase = rng() * 6.283;
    const rays = 5 + ((rng() * 3) | 0);
    paintRadial(img, s, (r, ang) => {
      const core = Math.exp(-r * r * 26) * 1.4;
      const halo = Math.exp(-r * 3.4) * 0.55;
      const star = Math.pow(Math.max(0, Math.cos(ang * rays + phase)), 26) * Math.exp(-r * 2.6) * 0.85;
      return core + halo + star;
    });
  },
  [CELL.ring]: (img, s) =>
    paintRadial(img, s, (r, ang) => {
      // Wobble the radius so the shockwave is never a perfect circle.
      const wob = 0.055 * Math.sin(ang * 3.1) + 0.035 * Math.sin(ang * 7.7 + 1.3);
      const d = (r - (0.78 + wob)) / 0.11;
      return Math.exp(-d * d) * (0.75 + 0.25 * FX_NOISE.noise(Math.cos(ang) * 4 + 8, Math.sin(ang) * 4 + 8));
    }),
  [CELL.chip]: (img, s, rng) => {
    // Angular rock fragment: convex-ish polygon, lit top-left, gritty inside.
    const n = 7 + ((rng() * 3) | 0);
    const radii: number[] = [];
    for (let i = 0; i < n; i++) radii.push(0.26 + rng() * 0.2);
    const inv = 1 / s;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const nx = (x + 0.5) * inv - 0.5;
        const ny = (y + 0.5) * inv - 0.5;
        const r = Math.sqrt(nx * nx + ny * ny);
        let ang = Math.atan2(ny, nx);
        if (ang < 0) ang += Math.PI * 2;
        const seg = (ang / (Math.PI * 2)) * n;
        const i0 = Math.floor(seg) % n;
        const i1 = (i0 + 1) % n;
        const f = seg - Math.floor(seg);
        const edge = radii[i0]! + (radii[i1]! - radii[i0]!) * f;
        const a = r < edge ? 1 : 0;
        const grain = FX_NOISE.fbm(x * inv * 9 + 40, y * inv * 9 + 40, 3);
        // Fake a lit facet: brighter toward -x,-y.
        const shade = 0.34 + 0.5 * smoothstep(0.4, -0.4, nx + ny) + grain * 0.22;
        const v = Math.round(255 * Math.min(1, shade));
        px(img, s, x, y, v, v * 0.97, v * 0.93, a * 255);
      }
    }
  },
  [CELL.splinter]: (img, s, rng) => {
    const inv = 1 / s;
    const bend = (rng() - 0.5) * 0.16;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const u = (x + 0.5) * inv;
        const v = (y + 0.5) * inv;
        const axis = 0.5 + bend * Math.sin((v - 0.5) * 3.1);
        const halfWidth = 0.085 * (1 - Math.abs(v - 0.5) * 1.85);
        const inside = halfWidth > 0 && Math.abs(u - axis) < halfWidth ? 1 : 0;
        const grain = FX_NOISE.fbm(u * 26 + 12, v * 5 + 3, 3);
        const shade = 0.4 + grain * 0.55 - Math.abs(u - axis) / (halfWidth || 1) * 0.14;
        px(img, s, x, y, 255 * shade, 232 * shade, 196 * shade, inside * 255);
      }
    }
  },
  [CELL.shard]: (img, s, rng) => {
    const inv = 1 / s;
    const ax = 0.5 + (rng() - 0.5) * 0.2;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const u = (x + 0.5) * inv;
        const v = (y + 0.5) * inv;
        // Triangle: apex at top, base at bottom, slightly skewed.
        const t = v;
        const half = 0.06 + t * 0.3;
        const d = Math.abs(u - ax);
        const inside = t > 0.08 && t < 0.94 && d < half;
        if (!inside) {
          px(img, s, x, y, 0, 0, 0, 0);
          continue;
        }
        const edge = 1 - d / half;
        // Glass is mostly transparent with bright refractive edges.
        const body = 0.16;
        const rim = Math.pow(1 - edge, 6) * 0.9;
        const glint = Math.pow(Math.max(0, 1 - Math.abs(d - half * 0.45) * 22), 3) * 0.8;
        const a = Math.min(1, body + rim + glint);
        const c = 0.82 + glint * 0.5;
        px(img, s, x, y, 232 * c, 246 * c, 255 * c, a * 255);
      }
    }
  },
  [CELL.droplet]: (img, s) => {
    const inv = 1 / s;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const u = (x + 0.5) * inv - 0.5;
        const v = (y + 0.5) * inv - 0.5;
        // Teardrop: circle whose radius pinches toward +v.
        const pinch = 1 - smoothstep(0.0, 0.5, v) * 0.72;
        const r = Math.sqrt((u / (0.26 * pinch)) ** 2 + (v / 0.4) ** 2);
        const a = smoothstep(1.05, 0.75, r);
        const spec = Math.exp(-(((u + 0.07) * 9) ** 2 + ((v + 0.12) * 9) ** 2)) * 0.9;
        const c = 0.55 + spec;
        px(img, s, x, y, 255 * c, 255 * c, 255 * c, a * 255);
      }
    }
  },
  [CELL.blood]: (img, s, rng) => {
    for (let i = 0; i < s * s * 4; i++) img.data[i] = 0;
    // A mist cell is a scatter of tiny droplets, not one blob: the silhouette
    // is what sells "fine spray" at distance.
    const count = 46;
    for (let i = 0; i < count; i++) {
      const a = rng() * Math.PI * 2;
      const rad = Math.pow(rng(), 0.55) * 0.42 * s;
      const cx = s * 0.5 + Math.cos(a) * rad;
      const cy = s * 0.5 + Math.sin(a) * rad;
      const rr = 1.5 + rng() * (s * 0.035);
      const x0 = Math.max(0, (cx - rr - 1) | 0);
      const x1 = Math.min(s - 1, (cx + rr + 1) | 0);
      const y0 = Math.max(0, (cy - rr - 1) | 0);
      const y1 = Math.min(s - 1, (cy + rr + 1) | 0);
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / rr;
          const f = smoothstep(1, 0.25, d);
          if (f <= 0) continue;
          const i4 = (y * s + x) * 4;
          const v = 200 + f * 55;
          img.data[i4] = v;
          img.data[i4 + 1] = v * 0.82;
          img.data[i4 + 2] = v * 0.8;
          img.data[i4 + 3] = Math.min(255, img.data[i4 + 3]! + f * 235);
        }
      }
    }
  },
  [CELL.ember]: (img, s) => paintRadial(img, s, (r) => Math.exp(-r * r * 46)),
  [CELL.fire]: (img, s, rng) => {
    const ox = rng() * 64;
    const oy = rng() * 64;
    const inv = 1 / s;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const nx = (x + 0.5) * inv - 0.5;
        const ny = (y + 0.5) * inv - 0.5;
        const r = Math.sqrt(nx * nx + ny * ny) * 2;
        const t = FX_NOISE.turbulence((x + 0.5) * inv * 3.2 + ox, (y + 0.5) * inv * 3.2 + oy, 5);
        let d = smoothstep(1.0, 0.05, r + (t - 0.42) * 0.95);
        d = Math.pow(Math.max(0, d), 1.15);
        // Hot cores punch through the turbulence; that is what makes a fireball
        // roil instead of glow evenly.
        const hot = Math.pow(Math.max(0, 1 - r * 1.5), 2.2) * (0.45 + t * 0.9);
        const lum = Math.min(1, 0.35 + hot * 1.3 + t * 0.4);
        px(img, s, x, y, 255 * lum, 255 * lum, 255 * lum, Math.min(1, d) * 255);
      }
    }
  },
  [CELL.wisp]: (img, s, rng) =>
    paintCloud(img, s, rng, { freq: 4.1, warp: 0.9, edge: 0.02, power: 1.9, grain: 0.3, squashY: 2.1, ridged: false }),
  [CELL.plume]: (img, s, rng) => {
    paintCloud(img, s, rng, { freq: 2.9, warp: 0.7, edge: 0.06, power: 1.2, grain: 0.25, squashY: 0.62, ridged: false });
    // Ground-hugging plumes have a flat, dense underside.
    for (let y = 0; y < s; y++) {
      const v = (y + 0.5) / s;
      const cut = smoothstep(0.86, 0.99, v);
      if (cut <= 0) continue;
      for (let x = 0; x < s; x++) {
        const i4 = (y * s + x) * 4;
        img.data[i4 + 3] = img.data[i4 + 3]! * (1 - cut);
      }
    }
  },
  [CELL.glow]: (img, s) => paintRadial(img, s, (r) => Math.pow(Math.max(0, 1 - r), 3.1)),
};

/** Content is inset inside each tile so mip bleed pulls in transparency, not a neighbour. */
const CELL_INSET = 0.045;

function buildAtlas(cellSize: number): THREE.Texture {
  const s = cellSize;
  const { canvas, ctx } = makeCanvas(s * 4, s * 4);
  ctx.clearRect(0, 0, s * 4, s * 4);
  const rng = mulberry32(0xbeef77);
  for (let i = 0; i < 16; i++) {
    const img = ctx.createImageData(s, s);
    const painter = PAINTERS[i];
    if (painter) painter(img, s, rng);
    const col = i % 4;
    // Cell 0 must land in the bottom-left tile because three flips canvas
    // textures vertically on upload.
    const row = 3 - Math.floor(i / 4);
    ctx.putImageData(img, col * s, row * s);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = 2;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const PARTICLE_VERT = /* glsl */ `
precision highp float;

attribute vec3 aPos;
attribute vec3 aVel;
attribute vec4 aTiming;  // spawnTime, lifetime, drag, gravityScale
attribute vec4 aSize;    // size0, size1, sizeCurve, rotRate
attribute vec4 aColorA;  // rgb, alpha at birth
attribute vec4 aColorB;  // rgb, alpha at death
attribute vec4 aMisc;    // atlasCell, stretch, turbulence, seed

uniform float uTime;
uniform float uSizeScale;

varying vec4 vColor;
varying vec2 vAtlasUv;
varying vec2 vLocal;
varying vec3 vViewPos;

void main() {
  float t = uTime - aTiming.x;
  float life = max(aTiming.y, 1e-3);
  float u = t / life;
  if (t < 0.0 || u >= 1.0) {
    vColor = vec4(0.0);
    vAtlasUv = vec2(0.0);
    vLocal = vec2(0.0);
    vViewPos = vec3(0.0, 0.0, -1.0);
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // behind the far plane: clipped
    return;
  }

  // Analytic linear-drag ballistics. k is floored so the exponential stays
  // numerically sane in fp32 (g/k blows up as k -> 0).
  float k = max(aTiming.z, 0.05);
  vec3 acc = vec3(0.0, -9.81 * aTiming.w, 0.0);
  vec3 term = acc / k;
  float e = exp(-k * t);
  vec3 disp = (aVel - term) * ((1.0 - e) / k) + term * t;
  vec3 wp = aPos + disp;
  vec3 vel = (aVel - term) * e + term;

  float turb = aMisc.z;
  if (turb > 0.0) {
    float sd = aMisc.w * 17.0;
    vec3 q = wp * 0.9;
    vec3 n1 = vec3(
      sin(q.y * 1.71 + uTime * 0.83 + sd),
      sin(q.z * 1.43 + uTime * 0.61 + sd * 1.7),
      sin(q.x * 1.93 + uTime * 0.72 + sd * 2.3));
    vec3 n2 = vec3(
      sin(q.z * 3.91 - uTime * 1.31 + sd * 1.1),
      sin(q.x * 4.33 + uTime * 1.13 + sd * 0.7),
      sin(q.y * 3.77 - uTime * 1.67 + sd * 1.9));
    vec3 off = (n1 + n2 * 0.45) * turb;
    wp += off * t;
    vel += off * 0.8;
  }

  float uc = clamp(u, 0.0, 1.0);
  float sc = mix(aSize.x, aSize.y, pow(uc, max(aSize.z, 0.05))) * uSizeScale;

  vec3 rgb = mix(aColorA.rgb, aColorB.rgb, uc);
  float alpha = mix(aColorA.a, aColorB.a, uc);
  alpha *= smoothstep(0.0, 0.05, uc) * (1.0 - smoothstep(0.68, 1.0, uc));

  vec4 mv = viewMatrix * vec4(wp, 1.0);
  vec3 velView = (viewMatrix * vec4(vel, 0.0)).xyz;

  float rot = aSize.w * t + aMisc.w * 6.2831853;
  vec2 ax = vec2(cos(rot), sin(rot));
  float sx = sc;
  float sy = sc;
  float stretch = aMisc.y;
  if (stretch > 0.0) {
    float vl = length(velView.xy);
    if (vl > 1e-3) {
      ax = velView.xy / vl;
      sx = sc * (1.0 + stretch * vl);
    }
  }
  mv.xy += position.x * sx * ax + position.y * sy * vec2(-ax.y, ax.x);

  // Fade out as a particle enters the near plane so nothing slabs the view.
  alpha *= smoothstep(0.10, 0.55, -mv.z);

  float cell = aMisc.x;
  float cx = mod(cell, 4.0);
  float cy = floor(cell * 0.25);
  vAtlasUv = (vec2(cx, cy) + mix(vec2(CELL_INSET), vec2(1.0 - CELL_INSET), uv)) * 0.25;

  vColor = vec4(rgb, alpha);
  vLocal = position.xy * 2.0;
  vViewPos = mv.xyz;
  gl_Position = projectionMatrix * mv;
}
`;

const PARTICLE_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uAtlas;
uniform vec3 uSunViewDir;
uniform vec3 uSunTint;
uniform vec3 uShadowTint;

#ifdef SOFT_PARTICLES
uniform sampler2D uDepth;
uniform vec2 uInvDepthSize;
uniform vec2 uNearFar;
uniform float uSoftness;
#endif

varying vec4 vColor;
varying vec2 vAtlasUv;
varying vec2 vLocal;
varying vec3 vViewPos;

void main() {
  vec4 tex = texture2D(uAtlas, vAtlasUv);
  float a = vColor.a * tex.a;
  if (a < 0.004) discard;
  vec3 rgb = vColor.rgb * tex.rgb;

#ifdef LIT_PARTICLES
  // Sphere impostor normal: the quad is treated as the silhouette of a ball so
  // the sun can carve a light and a shadow side into the puff.
  float r2 = dot(vLocal, vLocal);
  vec3 nView = vec3(vLocal, sqrt(max(0.0, 1.0 - r2)));
  float wrapped = clamp(dot(nView, uSunViewDir) * 0.5 + 0.5, 0.0, 1.0);
  vec3 lit = mix(uShadowTint, uSunTint, pow(wrapped, 1.45));
  // Forward scattering: looking through the puff toward the sun lights it up.
  float fwd = pow(clamp(dot(normalize(vViewPos), -uSunViewDir), 0.0, 1.0), 7.0);
  lit += uSunTint * fwd * 0.55 * (1.0 - a * 0.55);
  rgb *= lit;
#endif

#ifdef SOFT_PARTICLES
  vec2 duv = gl_FragCoord.xy * uInvDepthSize;
  float dz = texture2D(uDepth, duv).x;
  float n = uNearFar.x;
  float f = uNearFar.y;
  float sceneZ = (2.0 * n * f) / (f + n - (dz * 2.0 - 1.0) * (f - n));
  a *= clamp((sceneZ + vViewPos.z) / uSoftness, 0.0, 1.0);
  if (a < 0.004) discard;
#endif

  gl_FragColor = vec4(rgb * a, a); // premultiplied
}
`;

// ---------------------------------------------------------------------------
// Spawn record
// ---------------------------------------------------------------------------

/** Mutable spawn description. One shared instance; never allocate per spawn. */
export class ParticleWriter {
  readonly pos = new THREE.Vector3();
  readonly vel = new THREE.Vector3();
  readonly colorA = new THREE.Color(1, 1, 1);
  readonly colorB = new THREE.Color(1, 1, 1);
  alphaA = 1;
  alphaB = 0;
  /** Seconds. */
  life = 1;
  /** Seconds into the future before the particle appears. Free staging. */
  delay = 0;
  /** Linear drag coefficient, 1/s. Floored at 0.05 by the shader. */
  drag = 0.6;
  /** Multiplier on 9.81 m/s². Negative floats upward (hot smoke). */
  gravity = 1;
  size0 = 0.2;
  size1 = 0.4;
  /** Exponent on normalised age for the size interpolation. */
  sizeCurve = 1;
  /** Radians per second. */
  rotRate = 0;
  cell: number = CELL.smoke;
  /** Screen-space stretch per unit of view-space velocity. 0 = round. */
  stretch = 0;
  /** Turbulence amplitude in m/s. */
  turbulence = 0;
  seed = 0;
  additive = false;

  reset(): this {
    this.pos.set(0, 0, 0);
    this.vel.set(0, 0, 0);
    this.colorA.setRGB(1, 1, 1);
    this.colorB.setRGB(1, 1, 1);
    this.alphaA = 1;
    this.alphaB = 0;
    this.life = 1;
    this.delay = 0;
    this.drag = 0.6;
    this.gravity = 1;
    this.size0 = 0.2;
    this.size1 = 0.4;
    this.sizeCurve = 1;
    this.rotRate = 0;
    this.cell = CELL.smoke;
    this.stretch = 0;
    this.turbulence = 0;
    this.seed = Math.random();
    this.additive = false;
    return this;
  }
}

interface Pool {
  mesh: THREE.Mesh;
  geometry: THREE.InstancedBufferGeometry;
  material: THREE.ShaderMaterial;
  attrs: THREE.InstancedBufferAttribute[];
  capacity: number;
  cursor: number;
  /** Latest death time of any particle written into this pool. */
  liveUntil: number;
  dirtyMin: number;
  dirtyMax: number;
  everWrapped: boolean;
}

const _sunView = new THREE.Vector3();
const _tmpColor = new THREE.Color();

export interface ParticlesOptions {
  budget: number;
  cellSize: number;
  /** Enables the depth-buffer soft fade. Requires a readable scene depth texture. */
  soft: boolean;
}

export class Particles {
  readonly atlas: THREE.Texture;
  private readonly pools: [Pool, Pool];
  private readonly writer = new ParticleWriter();
  private readonly root = new THREE.Group();
  private time = 0;
  private depthTexture: THREE.Texture | null = null;

  constructor(private readonly scene: THREE.Scene, opts: ParticlesOptions) {
    this.atlas = buildAtlas(opts.cellSize);
    const budget = Math.max(256, opts.budget);
    // Smoke and dust dominate the frame; sparks are short-lived and bursty.
    const alphaCap = Math.round(budget * 0.58);
    const addCap = Math.max(128, budget - alphaCap);
    this.root.name = 'fx-particles';
    this.root.frustumCulled = false;
    this.pools = [
      this.buildPool(alphaCap, false, opts.soft),
      this.buildPool(addCap, true, opts.soft),
    ];
    this.root.add(this.pools[0].mesh, this.pools[1].mesh);
    scene.add(this.root);
  }

  private buildPool(capacity: number, additive: boolean, soft: boolean): Pool {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(
      [-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3,
    ));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.instanceCount = 0;
    // The vertex shader owns the world position, so CPU-side bounds are
    // meaningless; culling is disabled on the mesh instead.
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

    const attrs: THREE.InstancedBufferAttribute[] = [];
    const add = (name: string, size: number): void => {
      const a = new THREE.InstancedBufferAttribute(new Float32Array(capacity * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, a);
      attrs.push(a);
    };
    add('aPos', 3);
    add('aVel', 3);
    add('aTiming', 4);
    add('aSize', 4);
    add('aColorA', 4);
    add('aColorB', 4);
    add('aMisc', 4);

    const defines: Record<string, string> = { CELL_INSET: CELL_INSET.toFixed(4) };
    if (!additive) defines.LIT_PARTICLES = '1';
    if (soft) defines.SOFT_PARTICLES = '1';

    const material = new THREE.ShaderMaterial({
      vertexShader: PARTICLE_VERT,
      fragmentShader: PARTICLE_FRAG,
      defines,
      uniforms: {
        uTime: { value: 0 },
        uSizeScale: { value: 1 },
        uAtlas: { value: this.atlas },
        uSunViewDir: { value: new THREE.Vector3(0, 1, 0) },
        uSunTint: { value: new THREE.Color(1.5, 1.36, 1.16) },
        uShadowTint: { value: new THREE.Color(0.28, 0.34, 0.46) },
        uDepth: { value: null },
        uInvDepthSize: { value: new THREE.Vector2(1 / 1920, 1 / 1080) },
        uNearFar: { value: new THREE.Vector2(0.05, 2000) },
        uSoftness: { value: 0.55 },
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      toneMapped: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: additive ? THREE.OneFactor : THREE.OneMinusSrcAlphaFactor,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.visible = false;
    mesh.renderOrder = additive ? 12 : 10;
    mesh.matrixAutoUpdate = false;
    mesh.name = additive ? 'fx-particles-additive' : 'fx-particles-alpha';

    return {
      mesh, geometry, material, attrs, capacity,
      cursor: 0, liveUntil: -1, dirtyMin: Infinity, dirtyMax: -Infinity, everWrapped: false,
    };
  }

  /** Grab the shared writer, pre-reset. Fill it, then call {@link commit}. */
  write(): ParticleWriter {
    return this.writer.reset();
  }

  /** Commits the writer's current contents as one particle. */
  commit(): void {
    const w = this.writer;
    const pool = this.pools[w.additive ? 1 : 0];
    const i = pool.cursor;
    pool.cursor = i + 1;
    if (pool.cursor >= pool.capacity) {
      pool.cursor = 0;
      pool.everWrapped = true;
    }

    const [aPos, aVel, aTiming, aSize, aColorA, aColorB, aMisc] = pool.attrs as [
      THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute,
      THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute, THREE.InstancedBufferAttribute,
      THREE.InstancedBufferAttribute,
    ];

    const p3 = i * 3;
    const p4 = i * 4;
    const pa = aPos.array as Float32Array;
    pa[p3] = w.pos.x; pa[p3 + 1] = w.pos.y; pa[p3 + 2] = w.pos.z;
    const va = aVel.array as Float32Array;
    va[p3] = w.vel.x; va[p3 + 1] = w.vel.y; va[p3 + 2] = w.vel.z;

    const spawn = this.time + w.delay;
    const life = Math.max(0.02, w.life);
    const ta = aTiming.array as Float32Array;
    ta[p4] = spawn; ta[p4 + 1] = life; ta[p4 + 2] = w.drag; ta[p4 + 3] = w.gravity;

    const sa = aSize.array as Float32Array;
    sa[p4] = w.size0; sa[p4 + 1] = w.size1; sa[p4 + 2] = w.sizeCurve; sa[p4 + 3] = w.rotRate;

    const ca = aColorA.array as Float32Array;
    ca[p4] = w.colorA.r; ca[p4 + 1] = w.colorA.g; ca[p4 + 2] = w.colorA.b; ca[p4 + 3] = w.alphaA;
    const cb = aColorB.array as Float32Array;
    cb[p4] = w.colorB.r; cb[p4 + 1] = w.colorB.g; cb[p4 + 2] = w.colorB.b; cb[p4 + 3] = w.alphaB;

    const ma = aMisc.array as Float32Array;
    ma[p4] = w.cell; ma[p4 + 1] = w.stretch; ma[p4 + 2] = w.turbulence; ma[p4 + 3] = w.seed;

    if (i < pool.dirtyMin) pool.dirtyMin = i;
    if (i > pool.dirtyMax) pool.dirtyMax = i;
    const death = spawn + life;
    if (death > pool.liveUntil) pool.liveUntil = death;
  }

  /** Convenience: fills pos/vel from a cone around `dir`. */
  cone(
    out: THREE.Vector3, dir: THREE.Vector3, spread: number, speed: number,
    rng: () => number,
  ): THREE.Vector3 {
    // Uniform cap sampling: acos of a uniform cosine, otherwise the rim gets
    // over-weighted and every burst looks like a hollow shell.
    const cosMax = Math.cos(spread);
    const cosT = 1 - rng() * (1 - cosMax);
    const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
    const phi = rng() * Math.PI * 2;
    _basisZ.copy(dir).normalize();
    if (Math.abs(_basisZ.y) > 0.94) _basisX.set(1, 0, 0);
    else _basisX.set(0, 1, 0);
    _basisY.crossVectors(_basisZ, _basisX).normalize();
    _basisX.crossVectors(_basisY, _basisZ).normalize();
    out.copy(_basisZ).multiplyScalar(cosT)
      .addScaledVector(_basisX, sinT * Math.cos(phi))
      .addScaledVector(_basisY, sinT * Math.sin(phi))
      .multiplyScalar(speed);
    return out;
  }

  setDepthTexture(tex: THREE.Texture | null, width: number, height: number, near: number, far: number): void {
    this.depthTexture = tex;
    for (const pool of this.pools) {
      const u = pool.material.uniforms;
      if (u.uDepth === undefined) continue;
      u.uDepth.value = tex;
      (u.uInvDepthSize.value as THREE.Vector2).set(1 / Math.max(1, width), 1 / Math.max(1, height));
      (u.uNearFar.value as THREE.Vector2).set(near, far);
    }
  }

  /** Sun tint for lit smoke. `dir` points toward the sun, world space. */
  setSun(viewMatrix: THREE.Matrix4, dir: THREE.Vector3, sunColor: THREE.Color, skyColor: THREE.Color, intensity: number): void {
    _sunView.copy(dir).transformDirection(viewMatrix).normalize();
    const lit = this.pools[0].material.uniforms;
    (lit.uSunViewDir.value as THREE.Vector3).copy(_sunView);
    _tmpColor.copy(sunColor).multiplyScalar(THREE.MathUtils.clamp(intensity * 0.3, 0.35, 2.4));
    (lit.uSunTint.value as THREE.Color).copy(_tmpColor);
    _tmpColor.copy(skyColor).multiplyScalar(0.42);
    (lit.uShadowTint.value as THREE.Color).copy(_tmpColor);
  }

  /** Global size multiplier — used to shrink effects on low presets. */
  setSizeScale(v: number): void {
    for (const pool of this.pools) pool.material.uniforms.uSizeScale.value = v;
  }

  get now(): number {
    return this.time;
  }

  update(dt: number): void {
    this.time += dt;
    for (const pool of this.pools) {
      pool.material.uniforms.uTime.value = this.time;

      if (pool.dirtyMax >= pool.dirtyMin) {
        const start = pool.dirtyMin;
        const count = pool.dirtyMax - pool.dirtyMin + 1;
        for (const attr of pool.attrs) {
          attr.addUpdateRange(start * attr.itemSize, count * attr.itemSize);
          attr.needsUpdate = true;
        }
        pool.dirtyMin = Infinity;
        pool.dirtyMax = -Infinity;
      }

      const alive = this.time < pool.liveUntil;
      pool.mesh.visible = alive;
      if (alive) {
        pool.geometry.instanceCount = pool.everWrapped ? pool.capacity : pool.cursor;
      } else if (pool.cursor !== 0 || pool.everWrapped) {
        // Nothing alive: rewind the ring so the next burst draws a tight range.
        pool.cursor = 0;
        pool.everWrapped = false;
        pool.geometry.instanceCount = 0;
      }
    }
  }

  dispose(): void {
    for (const pool of this.pools) {
      this.root.remove(pool.mesh);
      pool.geometry.dispose();
      pool.material.dispose();
    }
    this.scene.remove(this.root);
    this.atlas.dispose();
    this.depthTexture = null;
  }
}

const _basisX = new THREE.Vector3();
const _basisY = new THREE.Vector3();
const _basisZ = new THREE.Vector3();
