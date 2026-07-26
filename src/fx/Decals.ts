/**
 * Projected decals.
 *
 * A decal is a box projected against the receiving geometry: candidate
 * triangles are clipped to the projector volume, so a bullet hole shot into a
 * corner wraps both faces instead of hovering over the seam. All decals live in
 * one non-indexed buffer with a fixed vertex budget per slot, so the whole pool
 * is a single draw call and eviction is a pointer bump (round-robin = LRU).
 *
 * Surface response is procedural: an albedo/alpha atlas, a crater normal map
 * derived from a height field, and a packed AO/roughness/metalness atlas. The
 * material is a MeshStandardMaterial so decals get the scene's real lighting,
 * shadows and IBL; only the sampling is replaced, via onBeforeCompile.
 */
import * as THREE from 'three';
import type { PhysicsWorld, SurfaceKind } from '../core/Contracts';
import { RayMask } from '../core/Contracts';
import { FX_NOISE, makeCanvas, mulberry32 } from './Particles';

export const DECAL = {
  holeConcrete: 0,
  holeMetal: 1,
  holeWood: 2,
  holePlaster: 3,
  holeGlass: 4,
  holeDirt: 5,
  holeSoft: 6,
  scorchSmall: 7,
  scorchLarge: 8,
  bloodA: 9,
  bloodB: 10,
  bloodPool: 11,
  scrapeMetal: 12,
  soot: 13,
  waterRing: 14,
  smudge: 15,
} as const;

export function decalCellForSurface(surface: SurfaceKind): number {
  switch (surface) {
    case 'metal': return DECAL.holeMetal;
    case 'wood': return DECAL.holeWood;
    case 'plaster': return DECAL.holePlaster;
    case 'glass': return DECAL.holeGlass;
    case 'dirt':
    case 'sand': return DECAL.holeDirt;
    case 'rubber':
    case 'fabric':
    case 'foliage': return DECAL.holeSoft;
    case 'water': return DECAL.waterRing;
    default: return DECAL.holeConcrete;
  }
}

// ---------------------------------------------------------------------------
// Atlas
// ---------------------------------------------------------------------------

interface CellBuffers {
  albedo: Float32Array; // rgba, 0..1
  height: Float32Array; // 1 = surface, 0 = deepest
  rough: Float32Array;
  metal: Float32Array;
  ao: Float32Array;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * (3 - 2 * t);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

interface HoleConfig {
  core: number;          // radius of the punched hole, in 0..1 cell radius
  crater: number;        // radius of the depressed crater
  halo: number;          // radius of the surface staining
  rim: number;           // raised lip height
  jagged: number;        // radial radius modulation
  coreColor: [number, number, number];
  haloColor: [number, number, number];
  haloAlpha: number;
  cracks: number;
  crackLength: number;
  crackWidth: number;
  crackColor: [number, number, number];
  ejecta: number;
  roughCore: number;
  roughHalo: number;
  metalRim: number;
  /** Glass: mostly transparent body, bright fracture lines. */
  translucent: boolean;
}

function paintHole(s: number, rng: () => number, out: CellBuffers, cfg: HoleConfig): void {
  const inv = 1 / s;
  const nOff = rng() * 90;
  const crackAngles: number[] = [];
  const crackLens: number[] = [];
  for (let i = 0; i < cfg.cracks; i++) {
    crackAngles.push((i / cfg.cracks) * Math.PI * 2 + (rng() - 0.5) * 0.9);
    crackLens.push(cfg.crackLength * (0.55 + rng() * 0.75));
  }

  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const nx = (x + 0.5) * inv - 0.5;
      const ny = (y + 0.5) * inv - 0.5;
      const d = Math.sqrt(nx * nx + ny * ny) * 2;
      const ang = Math.atan2(ny, nx);

      // Break every radius with the same angular noise field so the hole, the
      // crater and the stain agree on where the material gave way.
      const wob = FX_NOISE.noise(Math.cos(ang) * 2.6 + nOff, Math.sin(ang) * 2.6 + nOff);
      const mod = 1 + cfg.jagged * (wob - 0.5) * 2;
      const grain = FX_NOISE.fbm(x * inv * 7 + nOff, y * inv * 7 + nOff, 4);
      const fine = FX_NOISE.fbm(x * inv * 23 + 11, y * inv * 23 + 11, 3);

      const coreR = cfg.core * mod;
      const craterR = cfg.crater * mod;
      const haloR = cfg.halo * (1 + cfg.jagged * (grain - 0.5));

      const core = smoothstep(coreR, coreR * 0.62, d);
      const crater = smoothstep(craterR, coreR * 0.9, d);
      const halo = smoothstep(haloR, haloR * 0.25, d) * (0.55 + grain * 0.75);

      let crack = 0;
      for (let c = 0; c < crackAngles.length; c++) {
        const da = Math.atan2(Math.sin(ang - crackAngles[c]!), Math.cos(ang - crackAngles[c]!));
        const perp = Math.abs(Math.sin(da)) * d;
        const along = Math.cos(da) * d;
        if (along <= coreR * 0.5) continue;
        const len = crackLens[c]!;
        const w = cfg.crackWidth * (0.35 + 0.9 * (1 - along / len)) * (0.6 + fine * 0.8);
        crack = Math.max(crack, smoothstep(w, w * 0.25, perp) * smoothstep(len, len * 0.55, along));
      }

      // Height: hole punched through, crater dished, lip raised just outside.
      const rimBand = Math.exp(-(((d - craterR * 1.05) / (craterR * 0.42 + 1e-4)) ** 2));
      let h = 1 - core - crater * 0.42 - crack * 0.28 + rimBand * cfg.rim * (0.6 + grain * 0.8);
      h = clamp01(h);

      const stain = clamp01(halo * cfg.haloAlpha);
      let r: number, g: number, b: number;
      const cc = cfg.coreColor;
      const hc = cfg.haloColor;
      const shade = 0.75 + grain * 0.5;
      r = hc[0] * shade;
      g = hc[1] * shade;
      b = hc[2] * shade;
      const inner = clamp01(core + crater * 0.85);
      r = r * (1 - inner) + cc[0] * inner;
      g = g * (1 - inner) + cc[1] * inner;
      b = b * (1 - inner) + cc[2] * inner;
      if (crack > 0.01) {
        const k = crack * 0.9;
        r = r * (1 - k) + cfg.crackColor[0] * k;
        g = g * (1 - k) + cfg.crackColor[1] * k;
        b = b * (1 - k) + cfg.crackColor[2] * k;
      }

      let a = Math.max(Math.max(core, crater * 0.96), Math.max(stain, crack * 0.92));
      if (cfg.translucent) {
        // Glass: the pane stays visible, only fractures and the punch read.
        a = Math.max(core, Math.max(crack, stain * 0.5)) * 0.95;
      }
      a = clamp01(a);
      if (a < 0.004) {
        out.albedo[i * 4 + 3] = 0;
        out.height[i] = 1;
        out.rough[i] = 0.6;
        out.metal[i] = 0;
        out.ao[i] = 1;
        continue;
      }

      out.albedo[i * 4] = r;
      out.albedo[i * 4 + 1] = g;
      out.albedo[i * 4 + 2] = b;
      out.albedo[i * 4 + 3] = a;
      out.height[i] = h;
      out.rough[i] = cfg.roughHalo + (cfg.roughCore - cfg.roughHalo) * inner + (fine - 0.5) * 0.12;
      out.metal[i] = cfg.metalRim * clamp01(rimBand * 1.4) * (1 - core);
      out.ao[i] = clamp01(1 - (core * 0.85 + crater * 0.5 + crack * 0.3));
    }
  }

  // Ejecta: chips thrown clear of the crater, mostly on one side.
  const bias = rng() * Math.PI * 2;
  for (let e = 0; e < cfg.ejecta; e++) {
    const a = bias + (rng() - 0.5) * 3.4;
    const rad = (cfg.crater * 0.7 + Math.pow(rng(), 0.7) * (cfg.halo * 0.75)) * 0.5 * s;
    const cx = s * 0.5 + Math.cos(a) * rad;
    const cy = s * 0.5 + Math.sin(a) * rad;
    const rr = 1 + rng() * (s * 0.016);
    for (let y = Math.max(0, (cy - rr - 1) | 0); y <= Math.min(s - 1, (cy + rr + 1) | 0); y++) {
      for (let x = Math.max(0, (cx - rr - 1) | 0); x <= Math.min(s - 1, (cx + rr + 1) | 0); x++) {
        const f = smoothstep(1, 0.2, Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / rr);
        if (f <= 0) continue;
        const i = y * s + x;
        const k = f * 0.85;
        out.albedo[i * 4] = out.albedo[i * 4]! * (1 - k) + cfg.coreColor[0] * 1.6 * k;
        out.albedo[i * 4 + 1] = out.albedo[i * 4 + 1]! * (1 - k) + cfg.coreColor[1] * 1.6 * k;
        out.albedo[i * 4 + 2] = out.albedo[i * 4 + 2]! * (1 - k) + cfg.coreColor[2] * 1.6 * k;
        out.albedo[i * 4 + 3] = Math.max(out.albedo[i * 4 + 3]!, f * 0.9);
        out.height[i] = Math.min(1, out.height[i]! + f * 0.18);
      }
    }
  }
}

interface ScorchConfig {
  radius: number;
  streaks: number;
  darkness: number;
  sootColor: [number, number, number];
  rough: number;
}

function paintScorch(s: number, rng: () => number, out: CellBuffers, cfg: ScorchConfig): void {
  const inv = 1 / s;
  const off = rng() * 70;
  const streakPhase = rng() * 6.283;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const nx = (x + 0.5) * inv - 0.5;
      const ny = (y + 0.5) * inv - 0.5;
      const d = Math.sqrt(nx * nx + ny * ny) * 2;
      const ang = Math.atan2(ny, nx);
      const n = FX_NOISE.fbm(x * inv * 4.2 + off, y * inv * 4.2 + off, 5);
      const fine = FX_NOISE.turbulence(x * inv * 13 + 5, y * inv * 13 + 5, 3);
      // Radial streaks: soot thrown outward, thinning with distance.
      const streak = Math.pow(Math.max(0, Math.sin(ang * cfg.streaks + streakPhase) * 0.5 + 0.5), 2.5);
      const edge = cfg.radius * (0.75 + n * 0.5 + streak * 0.35);
      let a = smoothstep(edge, edge * 0.15, d);
      a = Math.pow(a, 1.25) * cfg.darkness * (0.55 + n * 0.75);
      a = clamp01(a);
      const soot = cfg.sootColor;
      const v = 0.6 + fine * 0.8;
      out.albedo[i * 4] = soot[0] * v;
      out.albedo[i * 4 + 1] = soot[1] * v;
      out.albedo[i * 4 + 2] = soot[2] * v;
      out.albedo[i * 4 + 3] = a;
      out.height[i] = 1 - (1 - n) * 0.06 * a;
      out.rough[i] = cfg.rough + (fine - 0.5) * 0.16;
      out.metal[i] = 0;
      out.ao[i] = 1 - a * 0.35;
    }
  }
}

interface BloodConfig {
  splat: number;
  drips: number;
  dripLength: number;
  satellites: number;
  pool: boolean;
  color: [number, number, number];
}

function paintBlood(s: number, rng: () => number, out: CellBuffers, cfg: BloodConfig): void {
  const inv = 1 / s;
  const off = rng() * 55;
  const cy0 = cfg.pool ? 0.5 : 0.66; // splat sits high so drips have room below
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const u = (x + 0.5) * inv;
      const v = (y + 0.5) * inv;
      const nx = u - 0.5;
      const ny = (v - cy0) / (cfg.pool ? 1 : 0.78);
      const d = Math.sqrt(nx * nx + ny * ny) * 2;
      const ang = Math.atan2(ny, nx);
      const wob = FX_NOISE.noise(Math.cos(ang) * 3.2 + off, Math.sin(ang) * 3.2 + off);
      const n = FX_NOISE.fbm(u * 6 + off, v * 6 + off, 4);
      const edge = cfg.splat * (0.72 + wob * 0.62);
      let a = smoothstep(edge, edge * 0.35, d);
      a *= 0.65 + n * 0.6;
      a = clamp01(a);
      const dark = 0.55 + 0.45 * clamp01(1 - a) + n * 0.18;
      out.albedo[i * 4] = cfg.color[0] * (1.15 - dark * 0.4);
      out.albedo[i * 4 + 1] = cfg.color[1] * (1.15 - dark * 0.4);
      out.albedo[i * 4 + 2] = cfg.color[2] * (1.15 - dark * 0.4);
      out.albedo[i * 4 + 3] = a;
      out.height[i] = 1 - a * 0.05;
      // Wet blood is a mirror at grazing angles; that specular is the tell.
      out.rough[i] = 0.22 + (1 - a) * 0.35 + n * 0.08;
      out.metal[i] = 0;
      out.ao[i] = 1 - a * 0.25;
    }
  }

  // Satellite droplets flung outward from the splat.
  for (let k = 0; k < cfg.satellites; k++) {
    const a = rng() * Math.PI * 2;
    const rad = (cfg.splat * 0.45 + Math.pow(rng(), 0.6) * 0.42) * 0.5 * s;
    const cx = s * 0.5 + Math.cos(a) * rad;
    const cyy = cy0 * s + Math.sin(a) * rad * 0.85;
    const rr = 1 + rng() * (s * 0.022);
    for (let y = Math.max(0, (cyy - rr - 1) | 0); y <= Math.min(s - 1, (cyy + rr + 1) | 0); y++) {
      for (let x = Math.max(0, (cx - rr - 1) | 0); x <= Math.min(s - 1, (cx + rr + 1) | 0); x++) {
        const f = smoothstep(1, 0.15, Math.hypot(x + 0.5 - cx, y + 0.5 - cyy) / rr);
        if (f <= 0) continue;
        const i = y * s + x;
        out.albedo[i * 4] = cfg.color[0];
        out.albedo[i * 4 + 1] = cfg.color[1];
        out.albedo[i * 4 + 2] = cfg.color[2];
        out.albedo[i * 4 + 3] = Math.max(out.albedo[i * 4 + 3]!, f);
        out.rough[i] = 0.2;
      }
    }
  }

  // Drips: run downward from the bottom edge of the splat. The fragment shader
  // reveals them over time from v = 0.46 downward.
  for (let k = 0; k < cfg.drips; k++) {
    const startX = 0.5 + (rng() - 0.5) * cfg.splat * 0.85;
    const len = cfg.dripLength * (0.35 + rng() * 0.9);
    const w0 = (0.006 + rng() * 0.012) * s;
    const wobble = (rng() - 0.5) * 0.05;
    const steps = Math.max(8, (len * s) | 0);
    for (let t = 0; t < steps; t++) {
      const f = t / steps;
      const vY = 0.5 - f * len;
      if (vY < 0.02) break;
      const cx = (startX + wobble * Math.sin(f * 5.5)) * s;
      const cyy = vY * s;
      // Drips taper and end in a bead.
      const bead = f > 0.86 ? 1.9 : 1;
      const w = w0 * (1 - f * 0.55) * bead;
      for (let y = Math.max(0, (cyy - w - 1) | 0); y <= Math.min(s - 1, (cyy + w + 1) | 0); y++) {
        for (let x = Math.max(0, (cx - w - 1) | 0); x <= Math.min(s - 1, (cx + w + 1) | 0); x++) {
          const dd = Math.hypot(x + 0.5 - cx, y + 0.5 - cyy) / (w + 0.001);
          const fa = smoothstep(1, 0.25, dd);
          if (fa <= 0) continue;
          const i = y * s + x;
          out.albedo[i * 4] = cfg.color[0] * 0.92;
          out.albedo[i * 4 + 1] = cfg.color[1] * 0.92;
          out.albedo[i * 4 + 2] = cfg.color[2] * 0.92;
          out.albedo[i * 4 + 3] = Math.max(out.albedo[i * 4 + 3]!, fa * 0.96);
          out.height[i] = Math.min(1, out.height[i]! + fa * 0.05);
          out.rough[i] = 0.18;
        }
      }
    }
  }
}

function paintScrape(s: number, rng: () => number, out: CellBuffers): void {
  const inv = 1 / s;
  const off = rng() * 40;
  const bend = (rng() - 0.5) * 0.1;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const u = (x + 0.5) * inv;
      const v = (y + 0.5) * inv;
      const axis = 0.5 + bend * Math.sin(u * 4.1);
      const dv = Math.abs(v - axis);
      const grain = FX_NOISE.fbm(u * 34 + off, v * 6 + off, 3);
      const width = 0.055 * (0.4 + grain * 1.5) * smoothstep(0.02, 0.22, u) * smoothstep(0.98, 0.7, u);
      const a = clamp01(smoothstep(width, width * 0.2, dv) * (0.5 + grain * 0.8));
      const bright = 0.55 + grain * 0.75;
      out.albedo[i * 4] = 0.62 * bright;
      out.albedo[i * 4 + 1] = 0.63 * bright;
      out.albedo[i * 4 + 2] = 0.66 * bright;
      out.albedo[i * 4 + 3] = a;
      out.height[i] = 1 - a * 0.25;
      out.rough[i] = 0.28 + (1 - grain) * 0.25;
      out.metal[i] = a * 0.85;
      out.ao[i] = 1 - a * 0.3;
    }
  }
}

function paintRing(s: number, rng: () => number, out: CellBuffers, thickness: number, color: [number, number, number]): void {
  const inv = 1 / s;
  const off = rng() * 30;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const nx = (x + 0.5) * inv - 0.5;
      const ny = (y + 0.5) * inv - 0.5;
      const d = Math.sqrt(nx * nx + ny * ny) * 2;
      const ang = Math.atan2(ny, nx);
      const wob = 0.05 * Math.sin(ang * 3.3 + off) + 0.03 * Math.sin(ang * 8.1);
      const t = (d - (0.8 + wob)) / thickness;
      const a = clamp01(Math.exp(-t * t) * 0.9);
      out.albedo[i * 4] = color[0];
      out.albedo[i * 4 + 1] = color[1];
      out.albedo[i * 4 + 2] = color[2];
      out.albedo[i * 4 + 3] = a;
      out.height[i] = 1 - a * 0.15;
      out.rough[i] = 0.12;
      out.metal[i] = 0;
      out.ao[i] = 1;
    }
  }
}

function paintSmudge(s: number, rng: () => number, out: CellBuffers): void {
  const inv = 1 / s;
  const off = rng() * 25;
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const nx = (x + 0.5) * inv - 0.5;
      const ny = (y + 0.5) * inv - 0.5;
      const d = Math.sqrt(nx * nx + ny * ny) * 2;
      const n = FX_NOISE.fbm(x * inv * 5.5 + off, y * inv * 5.5 + off, 4);
      const a = clamp01(smoothstep(0.95, 0.1, d + (n - 0.5) * 0.6) * (0.35 + n * 0.6));
      const v = 0.55 + n * 0.55;
      out.albedo[i * 4] = 0.44 * v;
      out.albedo[i * 4 + 1] = 0.4 * v;
      out.albedo[i * 4 + 2] = 0.35 * v;
      out.albedo[i * 4 + 3] = a * 0.7;
      out.height[i] = 1 - a * 0.03;
      out.rough[i] = 0.85;
      out.metal[i] = 0;
      out.ao[i] = 1 - a * 0.2;
    }
  }
}

const HOLE_PRESETS: Record<number, HoleConfig> = {
  [DECAL.holeConcrete]: {
    core: 0.19, crater: 0.42, halo: 0.92, rim: 0.16, jagged: 0.34,
    coreColor: [0.018, 0.017, 0.016], haloColor: [0.52, 0.505, 0.48], haloAlpha: 0.62,
    cracks: 5, crackLength: 0.85, crackWidth: 0.03, crackColor: [0.1, 0.098, 0.094],
    ejecta: 26, roughCore: 0.95, roughHalo: 0.82, metalRim: 0, translucent: false,
  },
  [DECAL.holeMetal]: {
    core: 0.15, crater: 0.26, halo: 0.62, rim: 0.42, jagged: 0.18,
    coreColor: [0.01, 0.009, 0.008], haloColor: [0.34, 0.33, 0.32], haloAlpha: 0.72,
    cracks: 7, crackLength: 0.45, crackWidth: 0.022, crackColor: [0.72, 0.7, 0.66],
    ejecta: 8, roughCore: 0.55, roughHalo: 0.38, metalRim: 0.9, translucent: false,
  },
  [DECAL.holeWood]: {
    core: 0.17, crater: 0.36, halo: 0.8, rim: 0.3, jagged: 0.45,
    coreColor: [0.02, 0.014, 0.009], haloColor: [0.24, 0.15, 0.085], haloAlpha: 0.66,
    cracks: 9, crackLength: 0.95, crackWidth: 0.028, crackColor: [0.46, 0.33, 0.19],
    ejecta: 22, roughCore: 0.92, roughHalo: 0.78, metalRim: 0, translucent: false,
  },
  [DECAL.holePlaster]: {
    core: 0.2, crater: 0.5, halo: 1.0, rim: 0.2, jagged: 0.42,
    coreColor: [0.03, 0.028, 0.026], haloColor: [0.78, 0.765, 0.74], haloAlpha: 0.78,
    cracks: 6, crackLength: 1.0, crackWidth: 0.026, crackColor: [0.3, 0.29, 0.28],
    ejecta: 34, roughCore: 0.94, roughHalo: 0.88, metalRim: 0, translucent: false,
  },
  [DECAL.holeGlass]: {
    core: 0.11, crater: 0.2, halo: 0.95, rim: 0.1, jagged: 0.2,
    coreColor: [0.02, 0.024, 0.028], haloColor: [0.8, 0.86, 0.92], haloAlpha: 0.34,
    cracks: 14, crackLength: 1.05, crackWidth: 0.016, crackColor: [0.9, 0.95, 1.0],
    ejecta: 4, roughCore: 0.4, roughHalo: 0.1, metalRim: 0, translucent: true,
  },
  [DECAL.holeDirt]: {
    core: 0.14, crater: 0.58, halo: 1.0, rim: 0.26, jagged: 0.5,
    coreColor: [0.035, 0.026, 0.018], haloColor: [0.3, 0.235, 0.16], haloAlpha: 0.7,
    cracks: 2, crackLength: 0.4, crackWidth: 0.05, crackColor: [0.16, 0.12, 0.08],
    ejecta: 44, roughCore: 0.98, roughHalo: 0.95, metalRim: 0, translucent: false,
  },
  [DECAL.holeSoft]: {
    core: 0.13, crater: 0.22, halo: 0.44, rim: 0.14, jagged: 0.4,
    coreColor: [0.012, 0.012, 0.012], haloColor: [0.14, 0.14, 0.145], haloAlpha: 0.6,
    cracks: 4, crackLength: 0.4, crackWidth: 0.03, crackColor: [0.05, 0.05, 0.05],
    ejecta: 3, roughCore: 0.9, roughHalo: 0.85, metalRim: 0, translucent: false,
  },
};

interface DecalAtlas {
  albedo: THREE.Texture;
  normal: THREE.Texture;
  orm: THREE.Texture;
}

function buildDecalAtlas(cellSize: number): DecalAtlas {
  const s = cellSize;
  const dim = s * 4;
  const albedoCanvas = makeCanvas(dim, dim);
  const normalCanvas = makeCanvas(dim, dim);
  const ormCanvas = makeCanvas(dim, dim);
  const rng = mulberry32(0x10deca1);

  const buffers: CellBuffers = {
    albedo: new Float32Array(s * s * 4),
    height: new Float32Array(s * s),
    rough: new Float32Array(s * s),
    metal: new Float32Array(s * s),
    ao: new Float32Array(s * s),
  };

  const albedoImg = albedoCanvas.ctx.createImageData(s, s);
  const normalImg = normalCanvas.ctx.createImageData(s, s);
  const ormImg = ormCanvas.ctx.createImageData(s, s);

  for (let cell = 0; cell < 16; cell++) {
    buffers.albedo.fill(0);
    buffers.height.fill(1);
    buffers.rough.fill(0.7);
    buffers.metal.fill(0);
    buffers.ao.fill(1);

    const hole = HOLE_PRESETS[cell];
    if (hole) {
      paintHole(s, rng, buffers, hole);
    } else if (cell === DECAL.scorchSmall) {
      paintScorch(s, rng, buffers, { radius: 0.85, streaks: 9, darkness: 0.9, sootColor: [0.035, 0.031, 0.028], rough: 0.88 });
    } else if (cell === DECAL.scorchLarge) {
      paintScorch(s, rng, buffers, { radius: 0.98, streaks: 14, darkness: 0.96, sootColor: [0.022, 0.02, 0.019], rough: 0.92 });
    } else if (cell === DECAL.bloodA) {
      paintBlood(s, rng, buffers, { splat: 0.6, drips: 7, dripLength: 0.42, satellites: 26, pool: false, color: [0.29, 0.021, 0.015] });
    } else if (cell === DECAL.bloodB) {
      paintBlood(s, rng, buffers, { splat: 0.78, drips: 4, dripLength: 0.3, satellites: 38, pool: false, color: [0.24, 0.017, 0.013] });
    } else if (cell === DECAL.bloodPool) {
      paintBlood(s, rng, buffers, { splat: 0.94, drips: 0, dripLength: 0, satellites: 14, pool: true, color: [0.17, 0.012, 0.01] });
    } else if (cell === DECAL.scrapeMetal) {
      paintScrape(s, rng, buffers);
    } else if (cell === DECAL.soot) {
      paintRing(s, rng, buffers, 0.34, [0.05, 0.045, 0.042]);
    } else if (cell === DECAL.waterRing) {
      paintRing(s, rng, buffers, 0.1, [0.62, 0.7, 0.74]);
    } else {
      paintSmudge(s, rng, buffers);
    }

    // Height -> tangent-space normal, via central differences. Strength is in
    // texels so it stays consistent across atlas resolutions.
    const strength = s * 0.06;
    for (let y = 0; y < s; y++) {
      for (let x = 0; x < s; x++) {
        const i = y * s + x;
        const xl = buffers.height[y * s + Math.max(0, x - 1)]!;
        const xr = buffers.height[y * s + Math.min(s - 1, x + 1)]!;
        const yd = buffers.height[Math.max(0, y - 1) * s + x]!;
        const yu = buffers.height[Math.min(s - 1, y + 1) * s + x]!;
        let nx = (xl - xr) * strength;
        let ny = (yd - yu) * strength;
        const nz = 1;
        const len = Math.hypot(nx, ny, nz);
        nx /= len; ny /= len;
        const nzz = nz / len;
        normalImg.data[i * 4] = (nx * 0.5 + 0.5) * 255;
        normalImg.data[i * 4 + 1] = (ny * 0.5 + 0.5) * 255;
        normalImg.data[i * 4 + 2] = (nzz * 0.5 + 0.5) * 255;
        normalImg.data[i * 4 + 3] = buffers.height[i]! * 255;

        albedoImg.data[i * 4] = Math.sqrt(clamp01(buffers.albedo[i * 4]!)) * 255;
        albedoImg.data[i * 4 + 1] = Math.sqrt(clamp01(buffers.albedo[i * 4 + 1]!)) * 255;
        albedoImg.data[i * 4 + 2] = Math.sqrt(clamp01(buffers.albedo[i * 4 + 2]!)) * 255;
        albedoImg.data[i * 4 + 3] = clamp01(buffers.albedo[i * 4 + 3]!) * 255;

        ormImg.data[i * 4] = clamp01(buffers.ao[i]!) * 255;
        ormImg.data[i * 4 + 1] = clamp01(buffers.rough[i]!) * 255;
        ormImg.data[i * 4 + 2] = clamp01(buffers.metal[i]!) * 255;
        ormImg.data[i * 4 + 3] = 255;
      }
    }

    const col = cell % 4;
    const row = 3 - Math.floor(cell / 4);
    albedoCanvas.ctx.putImageData(albedoImg, col * s, row * s);
    normalCanvas.ctx.putImageData(normalImg, col * s, row * s);
    ormCanvas.ctx.putImageData(ormImg, col * s, row * s);
  }

  const mk = (canvas: HTMLCanvasElement, srgb: boolean): THREE.Texture => {
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.wrapS = THREE.ClampToEdgeWrapping;
    t.wrapT = THREE.ClampToEdgeWrapping;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = 4;
    t.needsUpdate = true;
    return t;
  };

  return {
    albedo: mk(albedoCanvas.canvas, true),
    normal: mk(normalCanvas.canvas, false),
    orm: mk(ormCanvas.canvas, false),
  };
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

const MAX_VERTS_PER_DECAL = 72;
const CELL_INSET = 0.006;
// Above this the accel structure costs more memory than the decal is worth and
// the conforming-quad path takes over.
const MAX_CACHED_TRIS = 80000;
const GRID_CELL = 3.0;

interface ReceiverCache {
  tris: Float32Array;
  count: number;
  /** Hash of grid cell -> triangle indices. */
  buckets: Map<number, number[]>;
  stamp: Int32Array;
  generation: number;
}

const _m4 = new THREE.Matrix4();
const _m4b = new THREE.Matrix4();
const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();
const _vC = new THREE.Vector3();
const _n = new THREE.Vector3();
const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
// The projector basis has to survive the clipper, which reuses _n/_right.
const _basisX = new THREE.Vector3();
const _basisY = new THREE.Vector3();
const _basisZ = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _rayOrigin = new THREE.Vector3();
const _rayDir = new THREE.Vector3();
const _polyA = new Float32Array(3 * 32);
const _polyB = new Float32Array(3 * 32);

function gridKey(x: number, y: number, z: number): number {
  // Cheap spatial hash; collisions only cost a few extra triangle tests.
  return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) | 0;
}

/** Sutherland-Hodgman against `sign * coord <= 0.5`. Returns the new vertex count. */
function clipAxis(src: Float32Array, count: number, dst: Float32Array, axis: number, sign: number): number {
  let out = 0;
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    const ai = i * 3;
    const bi = j * 3;
    const da = 0.5 - sign * src[ai + axis]!;
    const db = 0.5 - sign * src[bi + axis]!;
    const inA = da >= 0;
    const inB = db >= 0;
    if (inA) {
      if (out >= 32) break;
      dst[out * 3] = src[ai]!;
      dst[out * 3 + 1] = src[ai + 1]!;
      dst[out * 3 + 2] = src[ai + 2]!;
      out++;
    }
    if (inA !== inB) {
      if (out >= 32) break;
      const t = da / (da - db);
      dst[out * 3] = src[ai]! + (src[bi]! - src[ai]!) * t;
      dst[out * 3 + 1] = src[ai + 1]! + (src[bi + 1]! - src[ai + 1]!) * t;
      dst[out * 3 + 2] = src[ai + 2]! + (src[bi + 2]! - src[ai + 2]!) * t;
      out++;
    }
  }
  return out;
}

export interface DecalRequest {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  /** Cell index into the decal atlas. */
  cell: number;
  /** Metres. Width == height. */
  size: number;
  /** Depth of the projector box along the normal. Larger wraps more geometry. */
  depth?: number;
  /** Roll around the normal, radians. Randomised when omitted. */
  roll?: number;
  /** Seconds before the decal fades out. */
  life?: number;
  tint?: THREE.Color;
  opacity?: number;
  /** Parallax depth in UV units. 0 disables. */
  parallax?: number;
  /** Aligns the decal's +V with world up — required for blood drips. */
  alignUp?: boolean;
  /** Animated drip reveal length in UV units. 0 disables. */
  dripLength?: number;
  /** Explicit receiver; otherwise the receiver is found by a short raycast. */
  receiver?: THREE.Object3D | null;
}

export class Decals {
  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.MeshStandardMaterial;
  private readonly mesh: THREE.Mesh;
  private readonly atlas: DecalAtlas;
  private readonly capacity: number;

  private readonly aPosition: THREE.BufferAttribute;
  private readonly aNormal: THREE.BufferAttribute;
  private readonly aUv: THREE.BufferAttribute;
  private readonly aTangent: THREE.BufferAttribute;
  private readonly aCell: THREE.BufferAttribute;
  private readonly aTint: THREE.BufferAttribute;
  private readonly aTimes: THREE.BufferAttribute;
  private readonly aParams: THREE.BufferAttribute;
  private readonly attrs: THREE.BufferAttribute[];

  private readonly caches = new WeakMap<THREE.Object3D, ReceiverCache | null>();
  private readonly uTime = { value: 0 };
  private readonly uFade = { value: new THREE.Vector2(26, 46) };
  private cursor = 0;
  private wrapped = false;
  private time = 0;
  private dirtyMin = Infinity;
  private dirtyMax = -Infinity;
  private cacheBuildsThisFrame = 0;

  constructor(
    private readonly scene: THREE.Scene,
    private readonly physics: PhysicsWorld,
    budget: number,
    cellSize: number,
  ) {
    this.capacity = Math.max(16, budget);
    this.atlas = buildDecalAtlas(cellSize);

    const verts = this.capacity * MAX_VERTS_PER_DECAL;
    this.geometry = new THREE.BufferGeometry();
    const mk = (size: number): THREE.BufferAttribute => {
      const a = new THREE.BufferAttribute(new Float32Array(verts * size), size);
      a.setUsage(THREE.DynamicDrawUsage);
      return a;
    };
    this.aPosition = mk(3);
    this.aNormal = mk(3);
    this.aUv = mk(2);
    this.aTangent = mk(4);
    this.aCell = mk(1);
    this.aTint = mk(4);
    this.aTimes = mk(2);
    this.aParams = mk(4);
    this.attrs = [
      this.aPosition, this.aNormal, this.aUv, this.aTangent,
      this.aCell, this.aTint, this.aTimes, this.aParams,
    ];
    this.geometry.setAttribute('position', this.aPosition);
    this.geometry.setAttribute('normal', this.aNormal);
    this.geometry.setAttribute('uv', this.aUv);
    this.geometry.setAttribute('tangent', this.aTangent);
    this.geometry.setAttribute('aCell', this.aCell);
    this.geometry.setAttribute('aTint', this.aTint);
    this.geometry.setAttribute('aTimes', this.aTimes);
    this.geometry.setAttribute('aParams', this.aParams);
    this.geometry.setDrawRange(0, 0);
    this.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);

    this.material = new THREE.MeshStandardMaterial({
      map: this.atlas.albedo,
      normalMap: this.atlas.normal,
      roughnessMap: this.atlas.orm,
      metalnessMap: this.atlas.orm,
      roughness: 1,
      metalness: 1,
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      // Negative offset pulls decals toward the camera in depth. Factor covers
      // sloped surfaces, units covers the flat case.
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -8,
      side: THREE.FrontSide,
      normalScale: new THREE.Vector2(1, 1),
      envMapIntensity: 1,
    });
    this.material.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = this.uTime;
      shader.uniforms.uFade = this.uFade;

      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', /* glsl */`
#include <common>
attribute float aCell;
attribute vec4 aTint;
attribute vec2 aTimes;
attribute vec4 aParams;
varying float vCell;
varying vec4 vTint;
varying vec2 vTimes;
varying vec4 vParams;
varying vec2 vDecalUv;
`)
        .replace('#include <begin_vertex>', /* glsl */`
#include <begin_vertex>
vCell = aCell;
vTint = aTint;
vTimes = aTimes;
vParams = aParams;
vDecalUv = uv;
`);

      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', /* glsl */`
#include <common>
uniform float uTime;
uniform vec2 uFade;
varying float vCell;
varying vec4 vTint;
varying vec2 vTimes;
varying vec4 vParams;
varying vec2 vDecalUv;

vec2 fxAtlasUv(vec2 duv) {
  vec2 cellIdx = vec2(mod(vCell, 4.0), floor(vCell * 0.25));
  return (cellIdx + clamp(duv, ${CELL_INSET.toFixed(4)}, ${(1 - CELL_INSET).toFixed(4)})) * 0.25;
}
`)
        .replace('#include <map_fragment>', /* glsl */`
vec2 decalUv = vDecalUv;
#ifdef USE_TANGENT
{
  // Parallax offset in tangent space: at grazing angles the crater floor
  // shifts under the rim, which is what makes the hole read as depth.
  float pdepth = vParams.x;
  if (pdepth > 0.0) {
    vec3 tsView = normalize(vec3(
      dot(vViewPosition, vTangent),
      dot(vViewPosition, vBitangent),
      dot(vViewPosition, vNormal)));
    float h = texture2D(normalMap, fxAtlasUv(decalUv)).a;
    decalUv -= (tsView.xy / max(abs(tsView.z), 0.35)) * (1.0 - h) * pdepth;
    decalUv = clamp(decalUv, 0.0, 1.0);
  }
}
#endif
vec2 decalAtlasUv = fxAtlasUv(decalUv);
vec4 decalTex = texture2D(map, decalAtlasUv);
float decalAge = uTime - vTimes.x;
float decalAlpha = decalTex.a * vTint.a;

// Drips extend downward over the first couple of seconds after the hit.
float dripLen = vParams.z;
if (dripLen > 0.0) {
  float reveal = clamp(decalAge / 2.4, 0.0, 1.0);
  float need = clamp((0.46 - decalUv.y) / dripLen, 0.0, 1.0);
  float show = smoothstep(need - 0.08, need + 0.01, reveal);
  float below = smoothstep(0.48, 0.42, decalUv.y);
  decalAlpha *= mix(1.0, show, below);
}

// Age fade, then a distance fade so distant decals do not stipple.
float life = vTimes.y;
if (life > 0.0) decalAlpha *= 1.0 - smoothstep(life - 2.5, life, decalAge);
decalAlpha *= 1.0 - smoothstep(uFade.x, uFade.y, length(vViewPosition));
if (decalAlpha < 0.004) discard;

diffuseColor.rgb *= decalTex.rgb * vTint.rgb;
diffuseColor.a *= decalAlpha;
`)
        .replace('#include <roughnessmap_fragment>', /* glsl */`
float roughnessFactor = roughness * clamp(texture2D(roughnessMap, decalAtlasUv).g + vParams.y, 0.04, 1.0);
`)
        .replace('#include <metalnessmap_fragment>', /* glsl */`
float metalnessFactor = metalness * texture2D(metalnessMap, decalAtlasUv).b;
`)
        .replace('#include <normal_fragment_maps>', /* glsl */`
vec3 mapN = texture2D(normalMap, decalAtlasUv).xyz * 2.0 - 1.0;
mapN.xy *= normalScale;
normal = normalize(tbn * mapN);
`);
    };

    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.name = 'fx-decals';
    this.mesh.frustumCulled = false;
    this.mesh.matrixAutoUpdate = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = true;
    this.mesh.renderOrder = 3;
    this.mesh.visible = false;
    scene.add(this.mesh);
  }

  /** Distance at which decals start/finish fading out. */
  setFadeRange(start: number, end: number): void {
    (this.uFade.value as THREE.Vector2).set(start, end);
  }

  add(req: DecalRequest): boolean {
    const size = Math.max(0.02, req.size);
    const depth = req.depth ?? size * 0.9;

    // Decal basis: +Z is the surface normal, +Y is world-up-ish unless the
    // caller wants a random roll (bullet holes) instead of gravity alignment.
    _n.copy(req.normal);
    if (_n.lengthSq() < 1e-8) return false;
    _n.normalize();

    if (req.alignUp) {
      _up.copy(_worldUp).addScaledVector(_n, -_worldUp.dot(_n));
      if (_up.lengthSq() < 1e-6) _up.set(1, 0, 0).addScaledVector(_n, -_n.x);
      _up.normalize();
      _right.crossVectors(_up, _n).normalize();
    } else {
      const roll = req.roll ?? Math.random() * Math.PI * 2;
      _right.set(0, 1, 0);
      if (Math.abs(_n.y) > 0.92) _right.set(1, 0, 0);
      _up.crossVectors(_n, _right).normalize();
      _right.crossVectors(_up, _n).normalize();
      const c = Math.cos(roll);
      const sn = Math.sin(roll);
      _vA.copy(_right).multiplyScalar(c).addScaledVector(_up, sn);
      _up.copy(_up).multiplyScalar(c).addScaledVector(_right, -sn);
      _right.copy(_vA);
    }

    _basisX.copy(_right);
    _basisY.copy(_up);
    _basisZ.copy(_n);

    // World -> decal-unit-box.
    _m4.makeBasis(_basisX, _basisY, _basisZ);
    _m4.setPosition(req.point);
    _m4b.copy(_m4).invert();
    // Scale into a unit cube: x,y by size, z by depth.
    _m4b.premultiply(_m4Scale(1 / size, 1 / size, 1 / depth));

    const slot = this.cursor;
    const base = slot * MAX_VERTS_PER_DECAL;
    let written = 0;

    const receiver = req.receiver ?? this.findReceiver(req.point, _basisZ);
    if (receiver) {
      written = this.clipReceiver(receiver, req.point, size, depth, base);
    }
    if (written < 3) {
      written = this.conformQuad(req.point, _basisZ, _basisX, _basisY, size, base);
    }
    if (written < 3) return false;

    this.cursor = slot + 1;
    if (this.cursor >= this.capacity) {
      this.cursor = 0;
      this.wrapped = true;
    }

    // Per-vertex decal parameters.
    const tint = req.tint;
    const tr = tint ? tint.r : 1;
    const tg = tint ? tint.g : 1;
    const tb = tint ? tint.b : 1;
    const ta = req.opacity ?? 1;
    const life = req.life ?? 0;
    const parallax = req.parallax ?? 0;
    const drip = req.dripLength ?? 0;
    const cellArr = this.aCell.array as Float32Array;
    const tintArr = this.aTint.array as Float32Array;
    const timesArr = this.aTimes.array as Float32Array;
    const paramsArr = this.aParams.array as Float32Array;
    const posArr = this.aPosition.array as Float32Array;
    for (let i = 0; i < MAX_VERTS_PER_DECAL; i++) {
      const v = base + i;
      if (i >= written) {
        // Collapse the slack in the slot: a reused slot must not rasterise the
        // previous decal's triangles just because they are transparent.
        posArr[v * 3] = req.point.x;
        posArr[v * 3 + 1] = req.point.y;
        posArr[v * 3 + 2] = req.point.z;
      }
      cellArr[v] = req.cell;
      tintArr[v * 4] = tr;
      tintArr[v * 4 + 1] = tg;
      tintArr[v * 4 + 2] = tb;
      tintArr[v * 4 + 3] = i < written ? ta : 0;
      timesArr[v * 2] = this.time;
      timesArr[v * 2 + 1] = life;
      paramsArr[v * 4] = parallax;
      paramsArr[v * 4 + 1] = 0;
      paramsArr[v * 4 + 2] = drip;
      paramsArr[v * 4 + 3] = 0;
    }

    if (base < this.dirtyMin) this.dirtyMin = base;
    const end = base + MAX_VERTS_PER_DECAL;
    if (end > this.dirtyMax) this.dirtyMax = end;
    this.mesh.visible = true;
    return true;
  }

  private findReceiver(point: THREE.Vector3, normal: THREE.Vector3): THREE.Object3D | null {
    _rayOrigin.copy(point).addScaledVector(normal, 0.14);
    _rayDir.copy(normal).multiplyScalar(-1);
    const hit = this.physics.raycast(_rayOrigin, _rayDir, 0.32, RayMask.Solid);
    return hit ? hit.object : null;
  }

  /** Clips the receiver's triangles to the projector box. Returns vertices written. */
  private clipReceiver(
    receiver: THREE.Object3D, point: THREE.Vector3, size: number, depth: number, base: number,
  ): number {
    const cache = this.getCache(receiver);
    if (!cache) return 0;

    const half = Math.max(size * 0.5, depth * 0.5) * 1.7321; // box half-diagonal
    const minX = Math.floor((point.x - half) / GRID_CELL);
    const maxX = Math.floor((point.x + half) / GRID_CELL);
    const minY = Math.floor((point.y - half) / GRID_CELL);
    const maxY = Math.floor((point.y + half) / GRID_CELL);
    const minZ = Math.floor((point.z - half) / GRID_CELL);
    const maxZ = Math.floor((point.z + half) / GRID_CELL);

    cache.generation++;
    const gen = cache.generation;
    const posArr = this.aPosition.array as Float32Array;
    const nrmArr = this.aNormal.array as Float32Array;
    const uvArr = this.aUv.array as Float32Array;
    const tanArr = this.aTangent.array as Float32Array;
    let written = 0;

    for (let gz = minZ; gz <= maxZ; gz++) {
      for (let gy = minY; gy <= maxY; gy++) {
        for (let gx = minX; gx <= maxX; gx++) {
          const bucket = cache.buckets.get(gridKey(gx, gy, gz));
          if (!bucket) continue;
          for (let bi = 0; bi < bucket.length; bi++) {
            const tri = bucket[bi]!;
            if (cache.stamp[tri] === gen) continue;
            cache.stamp[tri] = gen;
            written = this.clipTriangle(cache.tris, tri, base, written, posArr, nrmArr, uvArr, tanArr, size, depth);
            if (written >= MAX_VERTS_PER_DECAL) return written;
          }
        }
      }
    }
    return written;
  }

  private clipTriangle(
    tris: Float32Array, tri: number, base: number, written: number,
    posArr: Float32Array, nrmArr: Float32Array, uvArr: Float32Array, tanArr: Float32Array,
    size: number, depth: number,
  ): number {
    const o = tri * 9;
    _vA.set(tris[o]!, tris[o + 1]!, tris[o + 2]!).applyMatrix4(_m4b);
    _vB.set(tris[o + 3]!, tris[o + 4]!, tris[o + 5]!).applyMatrix4(_m4b);
    _vC.set(tris[o + 6]!, tris[o + 7]!, tris[o + 8]!).applyMatrix4(_m4b);

    // Face normal in decal space; only front-facing triangles receive.
    const ax = _vB.x - _vA.x, ay = _vB.y - _vA.y, az = _vB.z - _vA.z;
    const bx = _vC.x - _vA.x, by = _vC.y - _vA.y, bz = _vC.z - _vA.z;
    let fx = ay * bz - az * by;
    let fy = az * bx - ax * bz;
    let fz = ax * by - ay * bx;
    const fl = Math.hypot(fx, fy, fz);
    if (fl < 1e-12) return written;
    fx /= fl; fy /= fl; fz /= fl;
    if (fz < 0.12) return written;

    _polyA[0] = _vA.x; _polyA[1] = _vA.y; _polyA[2] = _vA.z;
    _polyA[3] = _vB.x; _polyA[4] = _vB.y; _polyA[5] = _vB.z;
    _polyA[6] = _vC.x; _polyA[7] = _vC.y; _polyA[8] = _vC.z;

    let count = 3;
    let src = _polyA;
    let dst = _polyB;
    for (let axis = 0; axis < 3 && count >= 3; axis++) {
      count = clipAxis(src, count, dst, axis, 1);
      let t = src; src = dst; dst = t;
      if (count < 3) break;
      count = clipAxis(src, count, dst, axis, -1);
      t = src; src = dst; dst = t;
    }
    if (count < 3) return written;

    // World-space face normal for lighting.
    _n.set(fx, fy, fz).transformDirection(_m4).normalize();
    _right.set(1, 0, 0).transformDirection(_m4).normalize();

    for (let i = 1; i + 1 < count; i++) {
      if (written + 3 > MAX_VERTS_PER_DECAL) return written;
      const idx = [0, i, i + 1];
      for (let k = 0; k < 3; k++) {
        const p = idx[k]! * 3;
        _vA.set(src[p]!, src[p + 1]!, src[p + 2]!);
        const u = _vA.x + 0.5;
        const v = _vA.y + 0.5;
        _vB.copy(_vA).applyMatrix4(_m4Scale(size, size, depth)).applyMatrix4(_m4);
        // Lift off the receiver; polygonOffset alone cannot survive a 2000m
        // far plane on every GPU.
        _vB.addScaledVector(_n, 0.0025);
        const w = base + written;
        posArr[w * 3] = _vB.x; posArr[w * 3 + 1] = _vB.y; posArr[w * 3 + 2] = _vB.z;
        nrmArr[w * 3] = _n.x; nrmArr[w * 3 + 1] = _n.y; nrmArr[w * 3 + 2] = _n.z;
        uvArr[w * 2] = u; uvArr[w * 2 + 1] = v;
        tanArr[w * 4] = _right.x; tanArr[w * 4 + 1] = _right.y; tanArr[w * 4 + 2] = _right.z;
        tanArr[w * 4 + 3] = 1;
        written++;
      }
    }
    return written;
  }

  /**
   * Fallback for receivers we cannot clip against (characters, oversized
   * meshes): a subdivided quad whose vertices are raycast onto the surface, so
   * it still hugs bumps instead of floating.
   */
  private conformQuad(
    point: THREE.Vector3, normal: THREE.Vector3, right: THREE.Vector3, up: THREE.Vector3,
    size: number, base: number,
  ): number {
    const N = 3;
    const half = size * 0.5;
    const posArr = this.aPosition.array as Float32Array;
    const nrmArr = this.aNormal.array as Float32Array;
    const uvArr = this.aUv.array as Float32Array;
    const tanArr = this.aTangent.array as Float32Array;
    const grid = new Array<number>((N + 1) * (N + 1) * 3);
    const ok = new Array<boolean>((N + 1) * (N + 1));

    for (let gy = 0; gy <= N; gy++) {
      for (let gx = 0; gx <= N; gx++) {
        const fx = gx / N - 0.5;
        const fy = gy / N - 0.5;
        _vA.copy(point).addScaledVector(right, fx * size).addScaledVector(up, fy * size);
        _rayOrigin.copy(_vA).addScaledVector(normal, half + 0.05);
        _rayDir.copy(normal).multiplyScalar(-1);
        const hit = this.physics.raycast(_rayOrigin, _rayDir, size + 0.12, RayMask.Solid);
        const i = gy * (N + 1) + gx;
        if (hit && hit.normal.dot(normal) > 0.3) {
          grid[i * 3] = hit.point.x + normal.x * 0.0025;
          grid[i * 3 + 1] = hit.point.y + normal.y * 0.0025;
          grid[i * 3 + 2] = hit.point.z + normal.z * 0.0025;
          ok[i] = true;
        } else {
          grid[i * 3] = _vA.x + normal.x * 0.0025;
          grid[i * 3 + 1] = _vA.y + normal.y * 0.0025;
          grid[i * 3 + 2] = _vA.z + normal.z * 0.0025;
          ok[i] = gx > 0 && gx < N && gy > 0 && gy < N;
        }
      }
    }

    let written = 0;
    for (let gy = 0; gy < N; gy++) {
      for (let gx = 0; gx < N; gx++) {
        const i00 = gy * (N + 1) + gx;
        const i10 = i00 + 1;
        const i01 = i00 + (N + 1);
        const i11 = i01 + 1;
        if (!ok[i00] && !ok[i10] && !ok[i01] && !ok[i11]) continue;
        const quad = [i00, i10, i11, i00, i11, i01];
        if (written + 6 > MAX_VERTS_PER_DECAL) break;
        for (let k = 0; k < 6; k++) {
          const vi = quad[k]!;
          const vx = vi % (N + 1);
          const vy = (vi / (N + 1)) | 0;
          const w = base + written;
          posArr[w * 3] = grid[vi * 3]!;
          posArr[w * 3 + 1] = grid[vi * 3 + 1]!;
          posArr[w * 3 + 2] = grid[vi * 3 + 2]!;
          nrmArr[w * 3] = normal.x; nrmArr[w * 3 + 1] = normal.y; nrmArr[w * 3 + 2] = normal.z;
          uvArr[w * 2] = vx / N; uvArr[w * 2 + 1] = vy / N;
          tanArr[w * 4] = right.x; tanArr[w * 4 + 1] = right.y; tanArr[w * 4 + 2] = right.z;
          tanArr[w * 4 + 3] = 1;
          written++;
        }
      }
    }
    return written;
  }

  private getCache(receiver: THREE.Object3D): ReceiverCache | null {
    const existing = this.caches.get(receiver);
    if (existing !== undefined) return existing;
    // Only one accel structure is built per frame; the rest of that frame's
    // decals take the conforming-quad path and the next one gets the real clip.
    if (this.cacheBuildsThisFrame > 0) return null;
    this.cacheBuildsThisFrame++;

    const cache = buildReceiverCache(receiver);
    this.caches.set(receiver, cache);
    return cache;
  }

  update(dt: number, cameraPosition: THREE.Vector3): void {
    this.time += dt;
    this.uTime.value = this.time;
    this.cacheBuildsThisFrame = 0;

    if (this.dirtyMax > this.dirtyMin) {
      const start = this.dirtyMin;
      const count = this.dirtyMax - this.dirtyMin;
      for (const attr of this.attrs) {
        attr.addUpdateRange(start * attr.itemSize, count * attr.itemSize);
        attr.needsUpdate = true;
      }
      this.dirtyMin = Infinity;
      this.dirtyMax = -Infinity;
    }

    const used = this.wrapped ? this.capacity : this.cursor;
    this.geometry.setDrawRange(0, used * MAX_VERTS_PER_DECAL);
    this.mesh.visible = used > 0;
    // Bounds are meaningless for a pool that spans the level; culling is off,
    // but the sphere still has to exist for the shadow/render path.
    const bs = this.geometry.boundingSphere;
    if (bs) bs.center.copy(cameraPosition);
  }

  dispose(): void {
    this.scene.remove(this.mesh);
    this.geometry.dispose();
    this.material.dispose();
    this.atlas.albedo.dispose();
    this.atlas.normal.dispose();
    this.atlas.orm.dispose();
  }
}

// ---------------------------------------------------------------------------
// Receiver caching
// ---------------------------------------------------------------------------

const _scaleMatrix = new THREE.Matrix4();
function _m4Scale(x: number, y: number, z: number): THREE.Matrix4 {
  return _scaleMatrix.makeScale(x, y, z);
}

function buildReceiverCache(root: THREE.Object3D): ReceiverCache | null {
  const meshes: THREE.Mesh[] = [];
  root.updateWorldMatrix(true, false);
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh || !m.geometry) return;
    if ((m as unknown as THREE.SkinnedMesh).isSkinnedMesh) return; // animated: no static cache
    const pos = m.geometry.getAttribute('position');
    if (!pos) return;
    meshes.push(m);
  });
  if (meshes.length === 0) return null;

  let total = 0;
  for (const m of meshes) {
    const index = m.geometry.getIndex();
    const pos = m.geometry.getAttribute('position');
    total += index ? index.count / 3 : pos.count / 3;
  }
  total = Math.floor(total);
  if (total === 0 || total > MAX_CACHED_TRIS) return null;

  const tris = new Float32Array(total * 9);
  const buckets = new Map<number, number[]>();
  let t = 0;
  const v = new THREE.Vector3();

  for (const m of meshes) {
    m.updateWorldMatrix(true, false);
    const mat = m.matrixWorld;
    const pos = m.geometry.getAttribute('position');
    const index = m.geometry.getIndex();
    const triCount = Math.floor(index ? index.count / 3 : pos.count / 3);
    for (let i = 0; i < triCount; i++) {
      const o = t * 9;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let k = 0; k < 3; k++) {
        const vi = index ? index.getX(i * 3 + k) : i * 3 + k;
        v.fromBufferAttribute(pos, vi).applyMatrix4(mat);
        tris[o + k * 3] = v.x;
        tris[o + k * 3 + 1] = v.y;
        tris[o + k * 3 + 2] = v.z;
        if (v.x < minX) minX = v.x;
        if (v.y < minY) minY = v.y;
        if (v.z < minZ) minZ = v.z;
        if (v.x > maxX) maxX = v.x;
        if (v.y > maxY) maxY = v.y;
        if (v.z > maxZ) maxZ = v.z;
      }
      const gx0 = Math.floor(minX / GRID_CELL), gx1 = Math.floor(maxX / GRID_CELL);
      const gy0 = Math.floor(minY / GRID_CELL), gy1 = Math.floor(maxY / GRID_CELL);
      const gz0 = Math.floor(minZ / GRID_CELL), gz1 = Math.floor(maxZ / GRID_CELL);
      for (let gz = gz0; gz <= gz1; gz++) {
        for (let gy = gy0; gy <= gy1; gy++) {
          for (let gx = gx0; gx <= gx1; gx++) {
            const key = gridKey(gx, gy, gz);
            let bucket = buckets.get(key);
            if (!bucket) {
              bucket = [];
              buckets.set(key, bucket);
            }
            bucket.push(t);
          }
        }
      }
      t++;
    }
  }

  return { tris, count: total, buckets, stamp: new Int32Array(total), generation: 0 };
}
