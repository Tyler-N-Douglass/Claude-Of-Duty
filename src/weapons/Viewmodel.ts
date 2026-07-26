/**
 * Procedural weapon viewmodels.
 *
 * Nothing here is a primitive stuck to another primitive. Every solid is either
 * an extruded profile (so it has a real bevel on every edge and, where wanted, a
 * real hole through it) or a lathed profile (so barrels and optic tubes have
 * chamfered rims). Openings — the ejection port, M-LOK slots, muzzle-brake
 * ports, the trigger guard, the rear aperture — are genuine gaps in the mesh,
 * not dark decals.
 *
 * Surfacing is object-space triplanar so it does not depend on whatever UVs
 * ExtrudeGeometry happens to produce, plus a per-vertex `aWear` term computed
 * from the local normal at construction time: on a bevelled box the only faces
 * whose normal is off-axis are the bevels themselves, so "polish the edges"
 * falls out of the geometry instead of needing a hand-painted mask. Wear is
 * computed before any transform is applied, so a rotated part is not mistaken
 * for one made entirely of edges.
 *
 * Draw calls: every static part is merged per material into one mesh, and each
 * animated part (magazine, bolt, charging handle, trigger, dust cover, hands)
 * gets its own merged mesh. A full rifle lands at 10-14 draws.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { LAYERS, type QualitySettings, type WeaponSpec } from '../core/Contracts';
import type { WeaponVisual } from './WeaponSpecs';

// ---------------------------------------------------------------------------
// Deterministic noise for the procedural texture pack
// ---------------------------------------------------------------------------

function xorshift(seed: number): () => number {
  let s = (seed | 0) || 0x9e3779b9;
  return () => {
    s ^= s << 13; s |= 0;
    s ^= s >>> 17;
    s ^= s << 5; s |= 0;
    return (s >>> 0) / 4294967296;
  };
}

function lattice(cells: number, rng: () => number): Float32Array {
  const a = new Float32Array(cells * cells);
  for (let i = 0; i < a.length; i++) a[i] = rng();
  return a;
}

/** Bilinear sample with a quintic fade, wrapping so every octave tiles. */
function sampleLattice(a: Float32Array, cells: number, u: number, v: number): number {
  const x = u * cells;
  const y = v * cells;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const fx = x - x0;
  const fy = y - y0;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const i0 = ((x0 % cells) + cells) % cells;
  const j0 = ((y0 % cells) + cells) % cells;
  const i1 = (i0 + 1) % cells;
  const j1 = (j0 + 1) % cells;
  const a00 = a[j0 * cells + i0]!;
  const a10 = a[j0 * cells + i1]!;
  const a01 = a[j1 * cells + i0]!;
  const a11 = a[j1 * cells + i1]!;
  const top = a00 + (a10 - a00) * ux;
  const bot = a01 + (a11 - a01) * ux;
  return top + (bot - top) * uy;
}

interface Fbm {
  levels: { data: Float32Array; cells: number; amp: number }[];
  norm: number;
}

function makeFbm(baseCells: number, octaves: number, gain: number, rng: () => number): Fbm {
  const levels: Fbm['levels'] = [];
  let cells = baseCells;
  let amp = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    levels.push({ data: lattice(cells, rng), cells, amp });
    norm += amp;
    cells *= 2;
    amp *= gain;
  }
  return { levels, norm };
}

function fbmAt(f: Fbm, u: number, v: number): number {
  let sum = 0;
  for (const l of f.levels) sum += sampleLattice(l.data, l.cells, u, v) * l.amp;
  return sum / f.norm;
}

/** Anisotropic sample: stretches the lattice so the noise reads as tool marks. */
function fbmStretched(f: Fbm, u: number, v: number, sx: number, sy: number): number {
  let sum = 0;
  for (const l of f.levels) sum += sampleLattice(l.data, l.cells, u * sx, v * sy) * l.amp;
  return sum / f.norm;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0 || 1e-6));
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------
// Texture pack
// ---------------------------------------------------------------------------

/**
 * `detail` packs four independent masks into one fetch:
 *   R  fine albedo grain            G  roughness variation
 *   B  macro blotch (wear breakup)  A  scratch / scuff streaks
 * `normal` is derived from the same height field so the two never disagree.
 */
interface DetailPair {
  detail: THREE.DataTexture;
  normal: THREE.DataTexture;
}

function heightToNormal(height: Float32Array, size: number, strength: number): Uint8Array {
  const out = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    const ym = ((y - 1) + size) % size;
    const yp = (y + 1) % size;
    for (let x = 0; x < size; x++) {
      const xm = ((x - 1) + size) % size;
      const xp = (x + 1) % size;
      const dx = (height[y * size + xp]! - height[y * size + xm]!) * strength;
      const dy = (height[yp * size + x]! - height[ym * size + x]!) * strength;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const i = (y * size + x) * 4;
      out[i] = Math.round((-dx * inv * 0.5 + 0.5) * 255);
      out[i + 1] = Math.round((-dy * inv * 0.5 + 0.5) * 255);
      out[i + 2] = Math.round((inv * 0.5 + 0.5) * 255);
      out[i + 3] = 255;
    }
  }
  return out;
}

function finishTexture(tex: THREE.DataTexture, aniso: number): THREE.DataTexture {
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = aniso;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Machined / phosphated steel: a fine casting grain crossed by long anisotropic
 * tool marks and a sparse population of deep scratches. The macro channel is a
 * very low frequency blotch that stops the wear term banding uniformly along
 * every bevel.
 */
function buildMetalDetail(size: number, aniso: number): DetailPair {
  const rng = xorshift(0x51ed270b);
  const grain = makeFbm(16, 5, 0.55, rng);
  const brush = makeFbm(8, 4, 0.6, rng);
  const macro = makeFbm(3, 3, 0.5, rng);
  const scratch = makeFbm(24, 3, 0.5, rng);

  const px = size * size;
  const data = new Uint8Array(px * 4);
  const height = new Float32Array(px);

  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const g = fbmAt(grain, u, v);
      const b = fbmStretched(brush, u, v, 0.35, 14);
      const m = fbmAt(macro, u, v);
      // Fold the noise about its midpoint so only the crests survive: thin,
      // ridged lines rather than a soft blur.
      const sRaw = fbmStretched(scratch, u, v, 0.25, 9);
      const s = Math.pow(clamp01(1 - Math.abs(sRaw - 0.5) * 7.5), 3);

      height[y * size + x] = g * 0.45 + b * 0.35 - s * 0.9;

      const i = (y * size + x) * 4;
      data[i] = Math.round(clamp01(0.5 + (g - 0.5) * 0.55 + (b - 0.5) * 0.35) * 255);
      data[i + 1] = Math.round(clamp01(0.5 + (b - 0.5) * 0.85 + (g - 0.5) * 0.3 - s * 0.35) * 255);
      data[i + 2] = Math.round(clamp01(m * 1.15 - 0.05) * 255);
      data[i + 3] = Math.round(clamp01(s * 1.3) * 255);
    }
  }

  return {
    detail: finishTexture(new THREE.DataTexture(data, size, size), aniso),
    normal: finishTexture(new THREE.DataTexture(heightToNormal(height, size, size * 0.010), size, size), aniso),
  };
}

/**
 * Glass-filled polymer: a stippled pebble grain with mould-flow swirl. The
 * pebbles are the noise pushed through a hard smoothstep so they read as
 * discrete raised bumps instead of mush.
 */
function buildPolymerDetail(size: number, aniso: number): DetailPair {
  const rng = xorshift(0x2f6a3d17);
  const pebble = makeFbm(40, 3, 0.42, rng);
  const flow = makeFbm(5, 4, 0.55, rng);
  const macro = makeFbm(3, 3, 0.5, rng);
  const scuff = makeFbm(20, 3, 0.5, rng);

  const px = size * size;
  const data = new Uint8Array(px * 4);
  const height = new Float32Array(px);

  for (let y = 0; y < size; y++) {
    const v = y / size;
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const p = smoothstep(0.44, 0.60, fbmAt(pebble, u, v));
      const f = fbmAt(flow, u, v);
      const m = fbmAt(macro, u, v);
      const sc = Math.pow(clamp01(1 - Math.abs(fbmStretched(scuff, u, v, 0.4, 5) - 0.5) * 6.5), 3);

      height[y * size + x] = p * 0.8 + f * 0.18 - sc * 0.4;

      const i = (y * size + x) * 4;
      data[i] = Math.round(clamp01(0.5 + (p - 0.5) * 0.28 + (f - 0.5) * 0.30) * 255);
      data[i + 1] = Math.round(clamp01(0.62 - p * 0.30 + (f - 0.5) * 0.22 - sc * 0.3) * 255);
      data[i + 2] = Math.round(clamp01(m * 1.1) * 255);
      data[i + 3] = Math.round(clamp01(sc * 1.2) * 255);
    }
  }

  return {
    detail: finishTexture(new THREE.DataTexture(data, size, size), aniso),
    normal: finishTexture(new THREE.DataTexture(heightToNormal(height, size, size * 0.014), size, size), aniso),
  };
}

// --- reticles ---------------------------------------------------------------

export type ReticleKind = 'dot' | 'holo' | 'chevron' | 'mildot';

function buildReticleTexture(kind: ReticleKind, size = 256): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  const c = (size - 1) / 2;

  const put = (x: number, y: number, a: number): void => {
    const xi = Math.round(x);
    const yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= size || yi >= size) return;
    const i = (yi * size + xi) * 4;
    const val = Math.round(clamp01(a) * 255);
    if (val <= data[i + 3]!) return;
    data[i] = 255; data[i + 1] = 255; data[i + 2] = 255; data[i + 3] = val;
  };

  const dot = (cx: number, cy: number, r: number, soft: number, gain = 1): void => {
    const R = Math.ceil(r + soft + 2);
    for (let y = Math.floor(cy - R); y <= cy + R; y++) {
      for (let x = Math.floor(cx - R); x <= cx + R; x++) {
        put(x, y, (1 - smoothstep(r, r + soft, Math.hypot(x - cx, y - cy))) * gain);
      }
    }
  };

  const ring = (cx: number, cy: number, r: number, w: number, gain = 1): void => {
    const steps = Math.ceil(r * 14);
    for (let s = 0; s <= steps; s++) {
      const t = (s / steps) * Math.PI * 2;
      dot(cx + Math.cos(t) * r, cy + Math.sin(t) * r, w * 0.5, 0.9, gain);
    }
  };

  const line = (x0: number, y0: number, x1: number, y1: number, w: number, gain = 1): void => {
    const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2) + 1;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      dot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, w * 0.5, 0.85, gain);
    }
  };

  switch (kind) {
    case 'dot':
      // Tight core, wide skirt — the shape a real emitter's bloom actually has,
      // and the thing a uniform glow always gets wrong.
      dot(c, c, size * 0.034, size * 0.105, 0.09);
      dot(c, c, size * 0.017, size * 0.030, 0.32);
      dot(c, c, size * 0.011, size * 0.013, 1);
      break;
    case 'holo':
      dot(c, c, size * 0.34, size * 0.10, 0.06);
      ring(c, c, size * 0.30, size * 0.016, 0.95);
      for (let k = 0; k < 4; k++) {
        const a = k * Math.PI * 0.5;
        line(c + Math.cos(a) * size * 0.30, c + Math.sin(a) * size * 0.30,
          c + Math.cos(a) * size * 0.365, c + Math.sin(a) * size * 0.365, size * 0.014, 0.9);
      }
      dot(c, c, size * 0.011, size * 0.014, 1);
      break;
    case 'chevron':
      dot(c, c + size * 0.02, size * 0.055, size * 0.09, 0.05);
      line(c - size * 0.075, c - size * 0.075, c, c + size * 0.035, size * 0.016, 1);
      line(c + size * 0.075, c - size * 0.075, c, c + size * 0.035, size * 0.016, 1);
      for (let k = 1; k <= 3; k++) {
        const y = c + size * (0.075 + k * 0.075);
        const w = size * (0.055 - k * 0.008);
        line(c - w, y, c + w, y, size * 0.011, 0.72);
      }
      break;
    case 'mildot':
      line(c, 0, c, size * 0.40, size * 0.010, 0.95);
      line(c, size * 0.60, c, size - 1, size * 0.010, 0.95);
      line(0, c, size * 0.40, c, size * 0.010, 0.95);
      line(size * 0.60, c, size - 1, c, size * 0.010, 0.95);
      line(c - size * 0.10, c, c + size * 0.10, c, size * 0.007, 0.95);
      line(c, c - size * 0.10, c, c + size * 0.10, size * 0.007, 0.95);
      for (let k = 1; k <= 4; k++) {
        const d = size * 0.075 * k;
        dot(c, c + d, size * 0.010, size * 0.008, 0.9);
        dot(c - d, c, size * 0.010, size * 0.008, 0.9);
        dot(c + d, c, size * 0.010, size * 0.008, 0.9);
      }
      break;
  }

  const tex = new THREE.DataTexture(data, size, size);
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  // No mipmaps, deliberately. A reticle is a few bright texels in a field of
  // zeros, and this texture is not premultiplied: mipping averages the colour
  // toward black *and* the alpha toward zero, so the drawn contribution falls
  // off as the square of the coverage. The plane lands at ~65 px against a
  // 256 px texture — LOD 2 — and at LOD 2 the dot had been annihilated. This is
  // why the sight rendered as an empty tube with no aiming point at all.
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Radial star for the muzzle-flash lobes: asymmetric spokes, hot core. */
function buildFlashTexture(size = 192): THREE.DataTexture {
  const rng = xorshift(0x77c1a3);
  const spokes = 9;
  const phase: number[] = [];
  const weight: number[] = [];
  for (let i = 0; i < spokes; i++) { phase.push(rng() * Math.PI * 2); weight.push(0.35 + rng() * 0.65); }

  const data = new Uint8Array(size * size * 4);
  const c = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - c) / c;
      const dy = (y - c) / c;
      const r = Math.hypot(dx, dy);
      const a = Math.atan2(dy, dx);
      let star = 0;
      for (let i = 0; i < spokes; i++) {
        star += Math.pow(Math.max(0, Math.cos(a - phase[i]! + i * 0.7)), 26) * weight[i]!;
      }
      const core = Math.pow(Math.max(0, 1 - r), 2.4);
      const arms = Math.pow(Math.max(0, 1 - r), 1.15) * star * 0.55;
      const v = clamp01(core * 1.6 + arms);
      const i = (y * size + x) * 4;
      data[i] = Math.round(clamp01(v * 1.15) * 255);
      data[i + 1] = Math.round(clamp01(v * 0.92) * 255);
      data[i + 2] = Math.round(clamp01(v * 0.62) * 255);
      data[i + 3] = Math.round(v * 255);
    }
  }
  const tex = new THREE.DataTexture(data, size, size);
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Soft black annulus: the scope shadow that rings a magnified sight picture. */
function buildScopeShadow(size = 256): THREE.DataTexture {
  const data = new Uint8Array(size * size * 4);
  const c = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r = Math.hypot(x - c, y - c) / c;
      const a = smoothstep(0.60, 1.0, r);
      const i = (y * size + x) * 4;
      data[i + 3] = Math.round(clamp01(a * a) * 255);
    }
  }
  const tex = new THREE.DataTexture(data, size, size);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

const TRIPLANAR_VARYINGS = /* glsl */ `
varying float vWear;
varying vec3 vObjPos;
varying vec3 vObjNor;
varying vec3 vBX;
varying vec3 vBY;
varying vec3 vBZ;
`;

const TRIPLANAR_FRAG_COMMON = /* glsl */ `
uniform sampler2D uDetail;
uniform sampler2D uDetailNrm;
uniform float uDetailScale;
uniform vec3 uBaseColor;
uniform vec3 uWearColor;
uniform float uBaseRough;
uniform float uWearRough;
uniform float uBaseMetal;
uniform float uWearMetal;
uniform float uWearBias;
uniform float uWearGain;
uniform float uNormalStrength;
uniform float uDesat;
uniform float uAo;

vec3 codWeights( vec3 n ) {
  vec3 w = pow( abs( n ), vec3( 6.0 ) );
  return w / max( w.x + w.y + w.z, 1e-4 );
}
`;

/**
 * The viewmodel scene's only image-based light is the sky, so anything with a
 * high metalness renders as a mirror of a blue dome — which is precisely why
 * the gun read as blue plastic tubing. Real gunmetal under the same sky is
 * near-neutral: the sun does the colouring and the sky contributes a hint of
 * cool on up-facing surfaces only. `uDesat` pulls the environment response
 * toward luminance and re-tints it by how much of the sky a surface can
 * actually see; `uAo` is the self-occlusion a viewmodel otherwise has none of,
 * since it casts no shadows and carries no baked AO.
 */
const TRIPLANAR_ENV_FIX = /* glsl */ `
{
  float codUp = clamp( codN.y * 0.5 + 0.5, 0.0, 1.0 );
  vec3 codSkyTint = mix( vec3( 1.14, 1.00, 0.84 ), vec3( 0.88, 0.97, 1.14 ), codUp * codUp );

  float codRl = dot( radiance, vec3( 0.2126, 0.7152, 0.0722 ) );
  radiance = mix( radiance, vec3( codRl ) * codSkyTint, uDesat );

  float codIl = dot( irradiance, vec3( 0.2126, 0.7152, 0.0722 ) );
  irradiance = mix( irradiance, vec3( codIl ) * codSkyTint, uDesat );

  float codBl = dot( iblIrradiance, vec3( 0.2126, 0.7152, 0.0722 ) );
  iblIrradiance = mix( iblIrradiance, vec3( codBl ) * codSkyTint, uDesat );

  // Sky visibility: an underside sees the ground, a cavity sees its own walls.
  // The macro blotch breaks it up so the term is not a clean gradient.
  float codSky = mix( 1.0 - uAo, 1.0, codUp ) * ( 1.0 - uAo * 0.22 * ( 1.0 - codD.b ) );
  irradiance *= codSky;
  iblIrradiance *= codSky;
  radiance *= mix( 1.0 - uAo * 0.62, 1.0, codUp );
}
`;

export interface GunMaterialTuning {
  baseColor: number;
  wearColor: number;
  baseRough: number;
  wearRough: number;
  baseMetal: number;
  wearMetal: number;
  wearBias: number;
  wearGain: number;
  detailScale: number;
  normalStrength: number;
  envIntensity: number;
  /** 0..1 how hard the sky's colour cast is pulled out of the IBL response. */
  desat: number;
  /** 0..1 strength of the faked self-occlusion. */
  ao: number;
}

/**
 * All tints of a given family share one `customProgramCacheKey`, so six weapons
 * in six finishes still compile exactly three programs — the colours live in
 * per-material uniforms, which cost nothing.
 */
function makeTriplanarMaterial(
  pair: DetailPair, tuning: GunMaterialTuning, cacheKey: string,
): THREE.MeshStandardMaterial {
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 1,
    metalness: tuning.baseMetal,
    envMapIntensity: tuning.envIntensity,
    dithering: true,
  });

  const uniforms = {
    uDetail: { value: pair.detail },
    uDetailNrm: { value: pair.normal },
    uDetailScale: { value: tuning.detailScale },
    uBaseColor: { value: new THREE.Color(tuning.baseColor).convertSRGBToLinear() },
    uWearColor: { value: new THREE.Color(tuning.wearColor).convertSRGBToLinear() },
    uBaseRough: { value: tuning.baseRough },
    uWearRough: { value: tuning.wearRough },
    uBaseMetal: { value: tuning.baseMetal },
    uWearMetal: { value: tuning.wearMetal },
    uWearBias: { value: tuning.wearBias },
    uWearGain: { value: tuning.wearGain },
    uNormalStrength: { value: tuning.normalStrength },
    uDesat: { value: tuning.desat },
    uAo: { value: tuning.ao },
  };

  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\nattribute float aWear;\n${TRIPLANAR_VARYINGS}`)
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
  vObjPos = transformed;
  vObjNor = normalize( objectNormal );
  vWear = aWear;
  vBX = normalMatrix * vec3( 1.0, 0.0, 0.0 );
  vBY = normalMatrix * vec3( 0.0, 1.0, 0.0 );
  vBZ = normalMatrix * vec3( 0.0, 0.0, 1.0 );`,
      );

    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${TRIPLANAR_VARYINGS}\n${TRIPLANAR_FRAG_COMMON}`)
      .replace(
        '#include <map_fragment>',
        `vec3 codN = normalize( vObjNor );
  vec3 codW = codWeights( codN );
  vec2 codUvX = vObjPos.zy * uDetailScale;
  vec2 codUvY = vObjPos.xz * uDetailScale;
  vec2 codUvZ = vObjPos.xy * uDetailScale;
  vec4 codD = texture2D( uDetail, codUvX ) * codW.x
            + texture2D( uDetail, codUvY ) * codW.y
            + texture2D( uDetail, codUvZ ) * codW.z;
  // Break the geometric edge term with the macro blotch so wear is irregular,
  // and let deep scratches seed wear away from the edges too.
  // The detail alpha term is a *baseline* micro-wear, not the edge wear mask.
  // At 0.55 it pushed every surface more than half way to the polished-steel
  // colour and roughness, so a phosphated receiver rendered as bright chrome.
  // The threshold sits high on purpose: bright metal is supposed to appear
  // where a hand or a sling actually rubs, not as a uniform pass over every
  // edge on the model, which is how "worn" turns into "chrome-plated".
  float codWear = vWear * uWearGain * ( 0.42 + 1.06 * codD.b ) + uWearBias + codD.a * codD.a * 0.13;
  codWear = smoothstep( 0.30, 0.86, codWear );
  vec3 codAlbedo = mix( uBaseColor, uWearColor, codWear ) * ( 0.84 + 0.32 * codD.r );
  diffuseColor.rgb *= codAlbedo;`,
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `float roughnessFactor = clamp(
    mix( uBaseRough, uWearRough, codWear ) * ( 0.86 + 0.30 * codD.g ), 0.035, 1.0 );`,
      )
      .replace(
        '#include <metalnessmap_fragment>',
        `float metalnessFactor = mix( uBaseMetal, uWearMetal, codWear );`,
      )
      .replace(
        '#include <normal_fragment_maps>',
        `{
    vec3 nx = texture2D( uDetailNrm, codUvX ).xyz * 2.0 - 1.0;
    vec3 ny = texture2D( uDetailNrm, codUvY ).xyz * 2.0 - 1.0;
    vec3 nz = texture2D( uDetailNrm, codUvZ ).xyz * 2.0 - 1.0;
    // Polished metal has had its micro-relief rubbed off; fade detail with wear.
    float ns = uNormalStrength * ( 1.0 - 0.45 * codWear );
    nx.xy *= ns; ny.xy *= ns; nz.xy *= ns;
    // Whiteout blend keeps detail from all three planes instead of letting the
    // dominant axis flatten the other two.
    vec3 tx = vec3( nx.xy + codN.zy, abs( nx.z ) * codN.x );
    vec3 ty = vec3( ny.xy + codN.xz, abs( ny.z ) * codN.y );
    vec3 tz = vec3( nz.xy + codN.xy, abs( nz.z ) * codN.z );
    vec3 objN = normalize( tx.zyx * codW.x + ty.xzy * codW.y + tz.xyz * codW.z );
    normal = normalize( vBX * objN.x + vBY * objN.y + vBZ * objN.z );
    #ifdef DOUBLE_SIDED
      normal *= faceDirection;
    #endif
  }`,
      )
      .replace('#include <lights_fragment_maps>', `#include <lights_fragment_maps>\n${TRIPLANAR_ENV_FIX}`);
  };

  mat.customProgramCacheKey = () => cacheKey;
  return mat;
}

/**
 * Anti-reflective coated optic glass.
 *
 * The four things that make a lens read as glass rather than as a hole:
 * a near-black tinted core, a blue-green AR sheen that only appears off-axis,
 * one soft circular reflection of the sky sitting across the surface (a lens is
 * a mirror the moment it is not pointed at you), and a bright meniscus where
 * the glass curves into its housing. All four are cheap; the last two are the
 * ones normally missing, and their absence is exactly what makes procedural
 * optics look like painted discs.
 */
function makeLensMaterial(tint: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uTint: { value: new THREE.Color(tint).convertSRGBToLinear() },
      uCore: { value: new THREE.Color(0x05090c).convertSRGBToLinear() },
      uSky: { value: new THREE.Color(0x9fc0e2).convertSRGBToLinear() },
    },
    vertexShader: /* glsl */ `
      varying vec3 vN;
      varying vec3 vV;
      varying vec2 vUvL;
      void main() {
        vec4 mv = modelViewMatrix * vec4( position, 1.0 );
        vN = normalize( normalMatrix * normal );
        vV = normalize( -mv.xyz );
        vUvL = uv * 2.0 - 1.0;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uTint;
      uniform vec3 uCore;
      uniform vec3 uSky;
      varying vec3 vN;
      varying vec3 vV;
      varying vec2 vUvL;
      void main() {
        // A real AR coating passes ~99% on axis; what you see is the residual
        // blue-green reflection, which only shows up off-axis.
        float f = pow( 1.0 - clamp( dot( normalize( vN ), normalize( vV ) ), 0.0, 1.0 ), 2.2 );
        float r = length( vUvL );

        // Sky reflection: one soft ellipse high on the glass, plus its faint
        // secondary further down — the double bounce off a coated element.
        float sweep = smoothstep( 0.92, 0.06, length( ( vUvL - vec2( -0.30, 0.40 ) ) * vec2( 1.0, 1.35 ) ) );
        float sweep2 = smoothstep( 0.50, 0.02, length( vUvL - vec2( 0.34, -0.42 ) ) ) * 0.35;

        // Meniscus: the glass curves away into the housing and goes bright.
        float rim = pow( smoothstep( 0.58, 1.0, r ), 2.4 );

        vec3 col = mix( uCore, uTint, f );
        col += uSky * ( sweep * 0.20 + sweep2 * 0.12 ) * ( 0.35 + 0.85 * f );
        col += mix( uTint, uSky, 0.45 ) * rim * 0.55;

        float a = clamp( 0.14 + 0.66 * f + sweep * 0.16 + sweep2 * 0.10 + rim * 0.42, 0.0, 1.0 );
        gl_FragColor = vec4( col, a );
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

/**
 * Detail scale is in repeats per metre. At 190 one repeat was 5 mm across, and
 * the anisotropic tool-mark octave inside it repeated every 0.4 mm — on a
 * barrel 30 cm from the camera that is roughly one repeat per screen pixel, so
 * it moired into the chevron/basketweave pattern that made the receiver look
 * like woven plastic. 55-80 puts a repeat at 1.2-1.8 cm, which is where
 * machining marks actually live and where mipping can resolve them.
 */
const METAL_TUNING: GunMaterialTuning = {
  baseColor: 0x7c786f, wearColor: 0xc6c1b6,
  baseRough: 0.47, wearRough: 0.19,
  // Phosphating is a conversion coating, not bare steel: authoring it at a
  // full metalness left the diffuse term at zero, so the whole weapon was
  // nothing but a dim reflection and read as a silhouette against a sunlit
  // street. Part-metal gives the body something to be lit *by*.
  baseMetal: 0.62, wearMetal: 1.0,
  wearBias: -0.10, wearGain: 0.82,
  detailScale: 84, normalStrength: 0.70, envIntensity: 1.05,
  desat: 0.86, ao: 0.30,
};

const POLYMER_TUNING: GunMaterialTuning = {
  baseColor: 0x3c3933, wearColor: 0x7a756b,
  baseRough: 0.66, wearRough: 0.42,
  baseMetal: 0.02, wearMetal: 0.06,
  wearBias: -0.14, wearGain: 0.72,
  detailScale: 118, normalStrength: 0.78, envIntensity: 0.92,
  desat: 0.90, ao: 0.36,
};

const ACCENT_TUNING: GunMaterialTuning = {
  baseColor: 0x353334, wearColor: 0xb0aba2,
  baseRough: 0.36, wearRough: 0.16,
  baseMetal: 0.76, wearMetal: 1.0,
  wearBias: -0.16, wearGain: 0.84,
  detailScale: 96, normalStrength: 0.55, envIntensity: 1.00,
  desat: 0.88, ao: 0.32,
};

export interface WeaponMaterialSet {
  metal: THREE.MeshStandardMaterial;
  polymer: THREE.MeshStandardMaterial;
  accent: THREE.MeshStandardMaterial;
  glove: THREE.MeshStandardMaterial;
}

export class GunMaterials {
  readonly lens: THREE.ShaderMaterial;
  readonly flashTexture: THREE.DataTexture;
  readonly scopeShadow: THREE.DataTexture;

  private readonly metalPair: DetailPair;
  private readonly polymerPair: DetailPair;
  private readonly glove: THREE.MeshStandardMaterial;
  private readonly reticles = new Map<ReticleKind, THREE.DataTexture>();
  private readonly variants = new Map<string, WeaponMaterialSet>();
  private readonly owned: THREE.Material[] = [];

  constructor(quality: QualitySettings) {
    const size = quality.preset === 'low' ? 256 : quality.preset === 'medium' ? 384 : 512;
    const aniso = Math.max(1, Math.min(16, quality.anisotropy));

    this.metalPair = buildMetalDetail(size, aniso);
    this.polymerPair = buildPolymerDetail(size, aniso);

    this.glove = makeTriplanarMaterial(this.polymerPair, {
      baseColor: 0x484238, wearColor: 0x6d675b,
      baseRough: 0.93, wearRough: 0.76,
      baseMetal: 0.0, wearMetal: 0.0,
      wearBias: -0.06, wearGain: 0.66,
      detailScale: 78, normalStrength: 1.10, envIntensity: 0.86,
      desat: 0.92, ao: 0.38,
    }, 'cod-gun-polymer');
    this.owned.push(this.glove);

    this.lens = makeLensMaterial(0x2ee0c0);
    this.flashTexture = buildFlashTexture();
    this.scopeShadow = buildScopeShadow();
  }

  /** One material set per distinct finish, shared by every weapon using it. */
  forWeapon(v: WeaponVisual): WeaponMaterialSet {
    const key = `${v.metalColor}|${v.polymerColor}|${v.accentColor}`;
    let set = this.variants.get(key);
    if (set) return set;
    set = {
      metal: makeTriplanarMaterial(this.metalPair, { ...METAL_TUNING, baseColor: v.metalColor }, 'cod-gun-metal'),
      polymer: makeTriplanarMaterial(this.polymerPair, { ...POLYMER_TUNING, baseColor: v.polymerColor }, 'cod-gun-polymer'),
      accent: makeTriplanarMaterial(this.metalPair, { ...ACCENT_TUNING, baseColor: v.accentColor }, 'cod-gun-accent'),
      glove: this.glove,
    };
    this.owned.push(set.metal, set.polymer, set.accent);
    this.variants.set(key, set);
    return set;
  }

  reticle(kind: ReticleKind): THREE.DataTexture {
    let t = this.reticles.get(kind);
    if (!t) { t = buildReticleTexture(kind); this.reticles.set(kind, t); }
    return t;
  }

  dispose(): void {
    for (const m of this.owned) m.dispose();
    this.owned.length = 0;
    this.variants.clear();
    this.lens.dispose();
    this.metalPair.detail.dispose();
    this.metalPair.normal.dispose();
    this.polymerPair.detail.dispose();
    this.polymerPair.normal.dispose();
    for (const t of this.reticles.values()) t.dispose();
    this.reticles.clear();
    this.flashTexture.dispose();
    this.scopeShadow.dispose();
  }
}

// ---------------------------------------------------------------------------
// Geometry primitives
// ---------------------------------------------------------------------------

const BEVEL = 0.0011;
const HALF_PI = Math.PI * 0.5;

type WearMode = 'box' | 'round' | 'flat';

/**
 * Stores the *raw* edge term in `aWear`, always at construction time and always
 * before the part is rotated into place — a grip tilted 26 degrees must not read
 * as though every one of its faces were a bevel. `PartSink.add` rebases this
 * into the final 0..1 value with a per-part floor.
 */
function setWearRaw(geo: THREE.BufferGeometry, mode: WearMode): THREE.BufferGeometry {
  const count = geo.getAttribute('position').count;
  const arr = new Float32Array(count);
  const nrm = geo.getAttribute('normal');

  if (mode === 'box' && nrm) {
    for (let i = 0; i < count; i++) {
      const m = Math.max(Math.abs(nrm.getX(i)), Math.abs(nrm.getY(i)), Math.abs(nrm.getZ(i)));
      // 1 - 1/sqrt(3) = 0.4226 is the three-way corner; normalise against it.
      arr[i] = clamp01((1 - m) / 0.4226);
    }
  } else if (mode === 'round') {
    const pos = geo.getAttribute('position');
    geo.computeBoundingBox();
    const bb = geo.boundingBox!;
    const span = Math.max(1e-5, bb.max.z - bb.min.z);
    for (let i = 0; i < count; i++) {
      const t = (pos.getZ(i) - bb.min.z) / span;
      // Rims of a lathed part take the knocks; the shank stays dark. Kept well
      // under 1 so a scope tube or a barrel does not end up with a polished
      // chrome band at each end — that reads as jewellery, not as a weapon.
      arr[i] = Math.max(smoothstep(0.90, 1.0, t), smoothstep(0.10, 0.0, t)) * 0.50;
    }
  }

  geo.setAttribute('aWear', new THREE.BufferAttribute(arr, 1));
  return geo;
}

function roundedRectShape(w: number, h: number, r: number): THREE.Shape {
  const hw = w * 0.5;
  const hh = h * 0.5;
  const rad = Math.min(r, hw - 1e-4, hh - 1e-4);
  const s = new THREE.Shape();
  if (rad <= 1e-5) {
    s.moveTo(-hw, -hh); s.lineTo(hw, -hh); s.lineTo(hw, hh); s.lineTo(-hw, hh);
    s.closePath();
    return s;
  }
  s.moveTo(-hw + rad, -hh);
  s.lineTo(hw - rad, -hh);
  s.quadraticCurveTo(hw, -hh, hw, -hh + rad);
  s.lineTo(hw, hh - rad);
  s.quadraticCurveTo(hw, hh, hw - rad, hh);
  s.lineTo(-hw + rad, hh);
  s.quadraticCurveTo(-hw, hh, -hw, hh - rad);
  s.lineTo(-hw, -hh + rad);
  s.quadraticCurveTo(-hw, -hh, -hw + rad, -hh);
  s.closePath();
  return s;
}

function roundedRectPath(w: number, h: number, r: number): THREE.Path {
  const shape = roundedRectShape(w, h, r);
  const p = new THREE.Path();
  p.curves = shape.curves;
  return p;
}

interface ExtrudeOpts {
  bevel?: number;
  segments?: number;
  curveSegments?: number;
}

/** Extrudes a shape along +Z, centred, with a real bevel on every edge. */
function extrude(shape: THREE.Shape, depth: number, opts: ExtrudeOpts = {}): THREE.BufferGeometry {
  const b = Math.min(opts.bevel ?? BEVEL, depth * 0.45);
  const core = Math.max(1e-4, depth - 2 * b);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: core,
    bevelEnabled: b > 1e-5,
    bevelThickness: b,
    bevelSize: b,
    bevelOffset: 0,
    bevelSegments: opts.segments ?? 2,
    curveSegments: opts.curveSegments ?? 4,
    steps: 1,
  });
  geo.translate(0, 0, -core * 0.5);
  return setWearRaw(geo, 'box');
}

function bevelBox(w: number, h: number, d: number, corner = 0.0022, bevel = BEVEL): THREE.BufferGeometry {
  return extrude(roundedRectShape(w, h, corner), d, { bevel });
}

/** Box with a rectangular hole punched clean through: magwells, port frames. */
function framePlate(
  w: number, h: number, d: number, iw: number, ih: number,
  corner = 0.0022, innerCorner = 0.0018, bevel = BEVEL,
): THREE.BufferGeometry {
  const shape = roundedRectShape(w, h, corner);
  shape.holes.push(roundedRectPath(iw, ih, innerCorner));
  return extrude(shape, d, { bevel });
}

/** Disc with a concentric hole: rear apertures you genuinely look through. */
function apertureDisc(
  rOut: number, rIn: number, d: number, bevel = BEVEL * 0.7, seg = 16,
): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  shape.absarc(0, 0, rOut, 0, Math.PI * 2, false);
  const hole = new THREE.Path();
  hole.absarc(0, 0, rIn, 0, Math.PI * 2, true);
  shape.holes.push(hole);
  return extrude(shape, d, { bevel, curveSegments: seg });
}

/**
 * Annular sector extruded along Z. This is the primitive that gives muzzle
 * brakes real ports and handguards real M-LOK slots: a ring assembled from
 * sectors with gaps between them is open geometry you can see through.
 */
function arcSector(
  rOut: number, rIn: number, theta0: number, thetaLen: number, depth: number,
  bevel = BEVEL * 0.6, seg = 10,
): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  shape.absarc(0, 0, rOut, theta0, theta0 + thetaLen, false);
  shape.absarc(0, 0, Math.max(1e-4, rIn), theta0 + thetaLen, theta0, true);
  shape.closePath();
  return extrude(shape, depth, { bevel, curveSegments: seg });
}

/**
 * Lathed tube whose axis is Z, chamfered at both rims. `rIn <= 0` makes it
 * solid. The profile is authored in (radius, y) and rotated so +Y becomes -Z:
 * increasing y in a profile therefore means "further down the barrel".
 */
function latheTube(rOut: number, rIn: number, len: number, seg = 20, chamfer = 0.0008): THREE.BufferGeometry {
  const inner = Math.max(0, rIn);
  const c = Math.min(chamfer, len * 0.3, (rOut - inner) * 0.45);
  const half = len * 0.5;
  const pts: THREE.Vector2[] = [
    new THREE.Vector2(inner, -half),
    new THREE.Vector2(rOut - c, -half),
    new THREE.Vector2(rOut, -half + c),
    new THREE.Vector2(rOut, half - c),
    new THREE.Vector2(rOut - c, half),
    new THREE.Vector2(inner, half),
  ];
  // Walk the bore back down so the tube is closed and its interior faces inward.
  if (inner > 1e-5) pts.push(new THREE.Vector2(inner, -half + 1e-5));
  const geo = new THREE.LatheGeometry(pts, seg);
  geo.rotateX(-HALF_PI);
  return setWearRaw(geo, 'round');
}

/** Solid of revolution from an explicit (radius, y) profile; +y ends up forward. */
function latheProfile(profile: readonly (readonly [number, number])[], seg = 20): THREE.BufferGeometry {
  const pts = profile.map((p) => new THREE.Vector2(Math.max(0, p[0]), p[1]));
  const geo = new THREE.LatheGeometry(pts, seg);
  geo.rotateX(-HALF_PI);
  return setWearRaw(geo, 'round');
}

/** Knurling / cooling ribs: n thin proud rings spaced along Z. */
function ribbedRings(
  r: number, depth: number, count: number, spacing: number, thickness: number, seg = 16,
): THREE.BufferGeometry[] {
  const out: THREE.BufferGeometry[] = [];
  const z0 = -((count - 1) * spacing) * 0.5;
  for (let i = 0; i < count; i++) {
    const g = latheTube(r + depth, r, thickness, seg, thickness * 0.35);
    g.translate(0, 0, z0 + i * spacing);
    out.push(g);
  }
  return out;
}

/**
 * Hardware. A weapon 30 cm from the camera lives or dies on this scale of
 * detail: at that distance a 3 mm pin head is 12 screen pixels, and twelve
 * pixels of correctly-lit chamfer is the difference between "machined steel"
 * and "rendered box". Every one of these has its axis on +Z so the caller
 * rotates it onto whichever face it belongs to.
 */
function pinHead(rOut: number, depth: number, seg = 10): THREE.BufferGeometry {
  return latheProfile([
    [0, -depth * 0.5], [rOut * 0.70, -depth * 0.5], [rOut, -depth * 0.22],
    [rOut, depth * 0.26], [rOut * 0.76, depth * 0.5], [0, depth * 0.5],
  ], seg);
}

/** Six-sided lathe: a hex bolt head or a castle nut, for free. */
function hexHead(rOut: number, depth: number): THREE.BufferGeometry {
  return latheTube(rOut, 0, depth, 6, Math.min(depth * 0.28, rOut * 0.14));
}

/**
 * A slotted fastener: domed head plus the screwdriver slot as a genuine dark
 * recess. Returned as a pair so the caller can put the slot on the accent
 * material and have it read as a shadow rather than as a painted line.
 */
function slotScrew(rOut: number, depth: number, seg = 10): { head: THREE.BufferGeometry; slot: THREE.BufferGeometry } {
  const head = pinHead(rOut, depth, seg);
  const slot = bevelBox(rOut * 1.7, rOut * 0.38, depth * 0.34, rOut * 0.1, 0.0002);
  slot.translate(0, 0, depth * 0.42);
  return { head, slot };
}

/** Knurled cylinder: a lathe body wrapped in fine axial teeth. */
function knurledCap(rOut: number, depth: number, teeth: number): THREE.BufferGeometry[] {
  const out: THREE.BufferGeometry[] = [latheTube(rOut, 0, depth, Math.max(12, teeth), depth * 0.16)];
  for (let i = 0; i < teeth; i++) {
    const a = (i / teeth) * Math.PI * 2;
    const t = bevelBox(rOut * 0.20, rOut * 0.09, depth * 0.80, rOut * 0.03, 0.0002);
    t.rotateZ(a + HALF_PI);
    t.translate(Math.cos(a) * rOut * 1.02, Math.sin(a) * rOut * 1.02, 0);
    out.push(t);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Part accumulation and merging
// ---------------------------------------------------------------------------

export type MatKey = 'metal' | 'polymer' | 'accent' | 'glove';
export type PartGroup =
  | 'body' | 'mag' | 'bolt' | 'charge' | 'trigger' | 'dust' | 'lhand' | 'rhand' | 'selector';

interface SinkEntry {
  geo: THREE.BufferGeometry;
  mat: MatKey;
  group: PartGroup;
}

class PartSink {
  private readonly entries: SinkEntry[] = [];

  /**
   * `floor` is the part's baseline wear: a charging handle is rubbed all over,
   * a receiver flank is not. The stored edge term is rebased onto it.
   */
  add(geo: THREE.BufferGeometry, mat: MatKey, group: PartGroup, floor = 0.08): this {
    let attr = geo.getAttribute('aWear') as THREE.BufferAttribute | undefined;
    if (!attr) {
      setWearRaw(geo, 'box');
      attr = geo.getAttribute('aWear') as THREE.BufferAttribute;
    }
    const a = attr.array as Float32Array;
    for (let i = 0; i < a.length; i++) a[i] = floor + a[i]! * (1 - floor);
    attr.needsUpdate = true;

    // mergeGeometries refuses to mix indexed and non-indexed inputs.
    const flat = geo.index ? geo.toNonIndexed() : geo;
    if (flat !== geo) geo.dispose();
    flat.deleteAttribute('uv1');
    flat.deleteAttribute('uv2');
    this.entries.push({ geo: flat, mat, group });
    return this;
  }

  addMany(geos: THREE.BufferGeometry[], mat: MatKey, group: PartGroup, floor = 0.08): this {
    for (const g of geos) this.add(g, mat, group, floor);
    return this;
  }

  build(nodes: Record<PartGroup, THREE.Object3D>, resolve: (k: MatKey) => THREE.Material): THREE.Mesh[] {
    const buckets = new Map<string, SinkEntry[]>();
    for (const e of this.entries) {
      const key = `${e.group}|${e.mat}`;
      let list = buckets.get(key);
      if (!list) { list = []; buckets.set(key, list); }
      list.push(e);
    }

    const meshes: THREE.Mesh[] = [];
    for (const [key, list] of buckets) {
      const [group, mat] = key.split('|') as [PartGroup, MatKey];
      const merged = list.length === 1 ? list[0]!.geo : mergeGeometries(list.map((e) => e.geo), false);
      if (!merged) continue;
      if (list.length > 1) for (const e of list) e.geo.dispose();
      merged.computeBoundingSphere();
      const mesh = new THREE.Mesh(merged, resolve(mat));
      mesh.name = key;
      mesh.frustumCulled = false;
      nodes[group].add(mesh);
      meshes.push(mesh);
    }
    this.entries.length = 0;
    return meshes;
  }
}

// ---------------------------------------------------------------------------
// Component builders
// ---------------------------------------------------------------------------

/** Bottom of the lower receiver — where the magwell mouth sits. */
function magWellY(v: WeaponVisual): number {
  return v.receiverY - v.receiverHeight * 0.55;
}

/** On a pistol the magwell *is* the grip, so it tracks the grip, not the receiver. */
function magWellZ(v: WeaponVisual): number {
  return v.mag === 'pistol' ? v.receiverZ + v.receiverLength * 0.30 : v.receiverZ;
}

/**
 * Overall height of a picatinny rail above the surface it is cut into: 4.2 mm
 * of base plus 5.2 mm of cross-bar. Anything mounting to the rail has to clear
 * this, so it is named rather than repeated.
 */
const RAIL_HEIGHT = 0.0094;

/** Picatinny rail. The gaps between the cross-bars are gaps, not painted lines. */
function buildRail(sink: PartSink, len: number, y: number, z: number, slots: number, width = 0.0205): void {
  const base = bevelBox(width, 0.0042, len, 0.0008);
  base.translate(0, y + 0.0021, z);
  sink.add(base, 'metal', 'body', 0.10);

  const pitch = len / slots;
  const barW = pitch * 0.46;
  const hw = width * 0.5;
  const hw2 = width * 0.34;
  for (let i = 0; i < slots; i++) {
    const zz = z - len * 0.5 + pitch * (i + 0.5);
    // Trapezoid section: the recognisable 45-degree rail flanks.
    const s = new THREE.Shape();
    s.moveTo(-hw, 0);
    s.lineTo(hw, 0);
    s.lineTo(hw2, 0.0052);
    s.lineTo(-hw2, 0.0052);
    s.closePath();
    const g = extrude(s, barW, { bevel: 0.0006, segments: 1 });
    g.translate(0, y + 0.0042, zz);
    sink.add(g, 'metal', 'body', 0.16);
  }
}

/**
 * AR-pattern upper.
 *
 * This used to be a lathed cylinder, which cost the model its whole silhouette:
 * a receiver with a circular section has no flanks to catch the key light, and
 * — far worse at hip fire, where the receiver's back face is the closest
 * surface on the model to the camera — its rear cap projected as a large flat
 * black disc in the middle of the frame. It is now an extruded section: flat
 * machined sides, a domed crown carrying the flat-top rail, a slab bottom that
 * mates with the lower, and every edge chamfered by the extruder.
 */
function buildUpper(sink: PartSink, v: WeaponVisual): number {
  const len = v.receiverLength;
  const zc = v.receiverZ;
  const w = v.receiverWidth * 1.02;
  // Crown height sets where the flat top — and therefore the whole sight line —
  // sits above the bore. At 0.56 the rail stood 34 mm proud of a barrel whose
  // handguard tops out at 20 mm, so the top edge of the weapon jogged by more
  // than a centimetre halfway along its length. 0.34 puts the rail where an
  // AR's actually is, and the handguard's own rail lands on the same plane.
  const crownY = v.receiverY + v.receiverHeight * 0.34;
  const floorY = v.receiverY - v.receiverHeight * 0.18;
  const h = crownY - floorY;
  const yc = (crownY + floorY) * 0.5;

  const hw = w * 0.5;
  const hh = h * 0.5;
  const crown = hw * 0.94;
  const foot = 0.0018;

  const s = new THREE.Shape();
  s.moveTo(-hw, -hh + foot);
  s.quadraticCurveTo(-hw, -hh, -hw + foot, -hh);
  s.lineTo(hw - foot, -hh);
  s.quadraticCurveTo(hw, -hh, hw, -hh + foot);
  s.lineTo(hw, hh - crown);
  s.quadraticCurveTo(hw, hh, hw - crown * 0.58, hh);
  s.lineTo(-hw + crown * 0.58, hh);
  s.quadraticCurveTo(-hw, hh, -hw, hh - crown);
  s.closePath();

  const body = extrude(s, len, { bevel: 0.0013, segments: 2, curveSegments: 5 });
  body.translate(0, yc, zc);
  sink.add(body, 'metal', 'body', 0.10);

  // Machined relief running the length of each flank, just under the crown's
  // shoulder. One proud line per side is all it takes to stop a slab reading
  // as a slab, and it gives the key light something to break on.
  for (const side of [-1, 1]) {
    const rib = bevelBox(0.0022, 0.0042, len * 0.80, 0.0007);
    rib.translate(side * (hw - 0.0006), yc + hh * 0.30, zc - len * 0.03);
    sink.add(rib, 'metal', 'body', 0.14);
  }

  // Forward assist: the single most recognisable lump on an AR upper.
  const fa = v.ejectSide;
  const faZ = zc + len * 0.30;
  const faY = v.receiverY + v.receiverHeight * 0.20;
  const faBoss = latheTube(0.0082, 0, 0.0110, 12, 0.0012);
  faBoss.rotateY(HALF_PI);
  faBoss.translate(fa * (hw + 0.0026), faY, faZ);
  sink.add(faBoss, 'metal', 'body', 0.12);
  const faCap = knurledCap(0.0056, 0.0060, 8);
  for (const g of faCap) {
    g.rotateY(HALF_PI);
    g.translate(fa * (hw + 0.0100), faY, faZ);
  }
  sink.addMany(faCap, 'metal', 'body', 0.30);

  // Takedown / pivot pin heads, both flanks. Real fasteners, in the right
  // places, at the right scale.
  for (const side of [-1, 1]) {
    for (const [pz, pr] of [[zc + len * 0.44, 0.0056], [zc - len * 0.42, 0.0052]] as const) {
      const pin = pinHead(pr, 0.0034, 10);
      pin.rotateY(HALF_PI);
      pin.translate(side * (hw + 0.0006), v.receiverY - v.receiverHeight * 0.05, pz);
      sink.add(pin, 'accent', 'body', 0.24);
    }
  }

  const railY = crownY - 0.0012;
  buildRail(sink, len * 0.94, railY, zc, Math.max(6, Math.round(len * 40)));
  return railY + RAIL_HEIGHT;
}

/**
 * The back of the receiver: end plate, castle nut, sling socket, and the sharp
 * chamfer where the upper's rear face meets the buffer tube. Without it the
 * upper simply ended, and its rear cap — the nearest surface on the whole model
 * at hip fire — projected as a bare disc.
 */
function buildReceiverRear(sink: PartSink, v: WeaponVisual): void {
  if (v.stock === 'none' || v.stockLength <= 0) return;
  const zBack = v.receiverZ + v.receiverLength * 0.5;
  const y = v.receiverY - v.stockDrop;
  const w = v.receiverWidth;

  // End plate: a flat slab clamped between the receiver and the castle nut.
  const plate = bevelBox(w * 0.88, v.receiverHeight * 0.86, 0.0040, 0.0026);
  plate.translate(0, v.receiverY + v.receiverHeight * 0.06, zBack + 0.0021);
  sink.add(plate, 'metal', 'body', 0.20);

  // Castle nut. Six flats: the shape alone says "threaded fastener".
  const nut = hexHead(0.0182, 0.0108);
  nut.translate(0, y, zBack + 0.0098);
  sink.add(nut, 'metal', 'body', 0.26);

  // Staking notches around its rear face — the tell that it has been torqued.
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + 0.4;
    const notch = bevelBox(0.0026, 0.0038, 0.0030, 0.0006);
    notch.rotateZ(a);
    notch.translate(Math.cos(a) * 0.0132, Math.sin(a) * 0.0132, zBack + 0.0148);
    sink.add(notch, 'accent', 'body', 0.34);
  }

  // QD sling socket in the end plate: a hole a sling swivel actually enters.
  const qd = apertureDisc(0.0058, 0.0028, 0.0044, 0.0006, 10);
  qd.rotateY(HALF_PI);
  qd.translate(-v.ejectSide * (w * 0.44 + 0.0012), v.receiverY + v.receiverHeight * 0.10, zBack + 0.0022);
  sink.add(qd, 'metal', 'body', 0.34);
}

/** Lower receiver and magwell. The magazine really does slide into a hole. */
function buildLower(sink: PartSink, v: WeaponVisual): void {
  const w = v.receiverWidth;
  const h = v.receiverHeight * 0.62;
  const len = v.receiverLength * 0.78;
  const yc = v.receiverY - v.receiverHeight * 0.24;
  const zc = v.receiverZ + v.receiverLength * 0.06;

  const body = bevelBox(w, h, len, 0.0038);
  body.translate(0, yc, zc);
  sink.add(body, 'polymer', 'body', 0.06);

  for (const s of [-1, 1]) {
    const panel = bevelBox(0.0035, h * 0.5, len * 0.8, 0.0010);
    panel.translate(s * (w * 0.5 - 0.0012), yc - h * 0.08, zc);
    sink.add(panel, 'polymer', 'body', 0.10);
  }

  if (v.mag === 'none' || v.mag === 'tube') return;

  const wellH = 0.030;
  const wy = magWellY(v);
  const wz = magWellZ(v);
  const well = framePlate(
    v.magWidth + 0.0175, v.magDepth + 0.0175, wellH,
    v.magWidth + 0.0022, v.magDepth + 0.0022, 0.0030, 0.0016,
  );
  well.rotateX(HALF_PI); // hole axis becomes vertical
  well.translate(0, wy - wellH * 0.5 + 0.004, wz);
  sink.add(well, 'polymer', 'body', 0.34);

  // Flared magwell mouth: a chamfer a magazine is slammed into a few thousand
  // times, which is why its corners are one of the few places on a rifle that
  // genuinely polish through to bright steel.
  const flare = framePlate(
    v.magWidth + 0.0230, v.magDepth + 0.0230, 0.0060,
    v.magWidth + 0.0055, v.magDepth + 0.0055, 0.0042, 0.0022,
  );
  flare.rotateX(HALF_PI);
  flare.translate(0, wy - 0.0026, wz);
  sink.add(flare, 'polymer', 'body', 0.50);

  const rel = bevelBox(0.0090, 0.0090, 0.0035, 0.0018);
  rel.rotateY(HALF_PI);
  rel.translate(v.ejectSide * (w * 0.5 + 0.0016), v.receiverY - v.receiverHeight * 0.22, wz + 0.014);
  sink.add(rel, 'accent', 'body', 0.52);

  // Bolt catch on the opposite flank: the AR's most recognisable left-side
  // control, and a paddle a thumb hits on every reload.
  const catchSide = -v.ejectSide;
  const paddle = bevelBox(0.0230, 0.0072, 0.0032, 0.0014);
  paddle.rotateY(HALF_PI);
  paddle.translate(catchSide * (w * 0.5 + 0.0015), v.receiverY - v.receiverHeight * 0.20, wz + 0.020);
  sink.add(paddle, 'accent', 'body', 0.50);
  const lowerLug = bevelBox(0.0100, 0.0130, 0.0042, 0.0016);
  lowerLug.rotateY(HALF_PI);
  lowerLug.translate(catchSide * (w * 0.5 + 0.0008), v.receiverY - v.receiverHeight * 0.18, wz + 0.026);
  sink.add(lowerLug, 'metal', 'body', 0.16);
}

/**
 * Ejection port: a frame with a rectangular hole, a back wall set 6 mm inboard
 * so the port reads as a cavity, a brass deflector, and a sprung dust cover on
 * its own node for the animator to swing open.
 */
function buildEjectionPort(sink: PartSink, v: WeaponVisual, dustNode: THREE.Object3D): void {
  const side = v.ejectSide;
  const x = v.receiverWidth * 0.5 + 0.0004;
  const portY = v.receiverY + v.receiverHeight * 0.13;
  const portZ = v.receiverZ - v.receiverLength * 0.16;
  const pw = Math.min(0.052, v.receiverLength * 0.30);
  const ph = 0.0195;

  const frame = framePlate(pw + 0.016, ph + 0.013, 0.0042, pw, ph, 0.0026, 0.0012);
  frame.rotateY(side * HALF_PI);
  frame.translate(side * x, portY, portZ);
  sink.add(frame, 'metal', 'body', 0.46);

  const back = bevelBox(pw - 0.001, ph - 0.001, 0.0022, 0.0008);
  back.rotateY(side * HALF_PI);
  back.translate(side * (x - 0.0062), portY, portZ);
  sink.add(back, 'accent', 'body', 0.02);

  const defl = new THREE.Shape();
  defl.moveTo(0, 0);
  defl.lineTo(0.0135, 0);
  defl.lineTo(0.0135, 0.0075);
  defl.lineTo(0, 0.0155);
  defl.closePath();
  const deflGeo = extrude(defl, 0.0075, { bevel: 0.0007 });
  deflGeo.rotateY(side * HALF_PI);
  deflGeo.translate(side * (x + 0.0026), portY - 0.005, portZ + pw * 0.5 + 0.004);
  sink.add(deflGeo, 'metal', 'body', 0.44);

  // Hinged along the bottom edge of the port so a rotation opens it.
  const cover = bevelBox(pw + 0.010, ph + 0.006, 0.0026, 0.0018);
  cover.translate(0, (ph + 0.006) * 0.5, 0);
  cover.rotateY(side * HALF_PI);
  sink.add(cover, 'metal', 'dust', 0.42);
  dustNode.position.set(side * (x + 0.0018), portY - (ph + 0.006) * 0.5, portZ);
}

/** Bolt carrier visible through the port; strokes back on every shot. */
function buildBolt(sink: PartSink, v: WeaponVisual): void {
  const side = v.ejectSide;
  const portY = v.receiverY + v.receiverHeight * 0.13;
  const portZ = v.receiverZ - v.receiverLength * 0.16;
  const w = Math.min(0.050, v.receiverLength * 0.28);

  const face = bevelBox(0.0080, 0.0175, w, 0.0016);
  face.rotateY(side * HALF_PI);
  face.translate(side * (v.receiverWidth * 0.5 - 0.0038), portY, portZ);
  sink.add(face, 'metal', 'bolt', 0.42);

  const claw = bevelBox(0.0038, 0.0060, 0.0075, 0.0010);
  claw.rotateY(side * HALF_PI);
  claw.translate(side * (v.receiverWidth * 0.5 - 0.0028), portY + 0.0045, portZ - w * 0.28);
  sink.add(claw, 'accent', 'bolt', 0.46);
}

/** Charging handle; the style follows the action type. */
function buildCharging(sink: PartSink, v: WeaponVisual, node: THREE.Object3D): void {
  const side = v.ejectSide;
  switch (v.charging) {
    case 'ar': {
      const bar = bevelBox(0.030, 0.0075, 0.020, 0.0014);
      sink.add(bar, 'metal', 'charge', 0.40);
      const latch = bevelBox(0.016, 0.0050, 0.0095, 0.0012);
      latch.translate(side * 0.019, -0.0006, 0);
      sink.add(latch, 'metal', 'charge', 0.48);
      for (let i = 0; i < 4; i++) {
        const s = bevelBox(0.0016, 0.0052, 0.017, 0.0004);
        s.translate(side * (0.014 + i * 0.0028), -0.0006, 0);
        sink.add(s, 'metal', 'charge', 0.54);
      }
      node.position.set(
        0, v.receiverY + v.receiverHeight * 0.24, v.receiverZ + v.receiverLength * 0.46,
      );
      break;
    }
    case 'side': {
      const knob = latheProfile(
        [[0, -0.006], [0.0055, -0.006], [0.0068, -0.0035], [0.0068, 0.0035], [0.0050, 0.006], [0, 0.006]], 12,
      );
      knob.rotateY(side * HALF_PI);
      sink.add(knob, 'metal', 'charge', 0.28);
      const stem = bevelBox(0.0070, 0.0070, 0.0130, 0.0016);
      stem.rotateY(side * HALF_PI);
      stem.translate(side * -0.0075, 0, 0);
      sink.add(stem, 'metal', 'charge', 0.20);
      node.position.set(
        side * (v.receiverWidth * 0.5 + 0.008),
        v.receiverY + v.receiverHeight * 0.16,
        v.receiverZ + v.receiverLength * 0.24,
      );
      break;
    }
    case 'boltaction': {
      const root = latheTube(0.0090, 0, 0.016, 14, 0.0010);
      root.rotateY(side * HALF_PI);
      sink.add(root, 'metal', 'charge', 0.18);
      const arm = bevelBox(0.0072, 0.0072, 0.045, 0.0022);
      arm.rotateY(side * HALF_PI);
      arm.translate(side * 0.021, 0, 0);
      sink.add(arm, 'metal', 'charge', 0.24);
      const ball = latheProfile(
        [[0, -0.0085], [0.0060, -0.0070], [0.0088, 0], [0.0060, 0.0070], [0, 0.0085]], 14,
      );
      ball.rotateY(side * HALF_PI);
      ball.translate(side * 0.044, 0, 0);
      sink.add(ball, 'metal', 'charge', 0.32);
      node.position.set(
        side * (v.receiverWidth * 0.5),
        v.receiverY + v.receiverHeight * 0.10,
        v.receiverZ + v.receiverLength * 0.30,
      );
      break;
    }
    case 'slide':
    case 'pump':
      // Handled by the slide / pump assemblies themselves.
      break;
  }
}

/** Ambi fire selector with a real detent lever. */
function buildSelector(sink: PartSink, v: WeaponVisual, node: THREE.Object3D): void {
  const side = -v.ejectSide;
  const boss = latheTube(0.0062, 0, 0.0090, 14, 0.0008);
  boss.rotateY(HALF_PI);
  sink.add(boss, 'metal', 'selector', 0.22);
  const lever = bevelBox(0.0180, 0.0055, 0.0038, 0.0012);
  lever.rotateY(HALF_PI);
  lever.translate(side * 0.0055, -0.0080, 0);
  sink.add(lever, 'accent', 'selector', 0.30);
  const pad = bevelBox(0.0070, 0.0048, 0.0034, 0.0010);
  pad.rotateY(HALF_PI);
  pad.translate(side * 0.0060, -0.0150, 0);
  sink.add(pad, 'accent', 'selector', 0.48);
  node.position.set(
    side * (v.receiverWidth * 0.5 + 0.0012),
    v.receiverY - v.receiverHeight * 0.10,
    v.receiverZ + v.receiverLength * 0.30,
  );
}

/** Pistol grip with finger grooves, a palm swell and a beavertail. */
function buildGrip(sink: PartSink, v: WeaponVisual): void {
  const len = v.gripLength;
  const gz = v.receiverZ + v.receiverLength * 0.30;
  const gy = v.receiverY - v.receiverHeight * 0.52;

  const profile = new THREE.Shape();
  const bw = 0.0225;
  const tw = 0.0250;
  profile.moveTo(-tw * 0.5, 0);
  profile.lineTo(tw * 0.5, 0);
  profile.lineTo(bw * 0.52, -len * 0.55);
  profile.lineTo(bw * 0.46, -len);
  profile.lineTo(-bw * 0.50, -len);
  profile.lineTo(-bw * 0.58, -len * 0.45);
  profile.closePath();
  // Authored in the XY plane looking down the bore, extruded across the gun.
  const grip = extrude(profile, 0.0265, { bevel: 0.0020, segments: 2 });
  grip.rotateY(HALF_PI);
  grip.rotateX(-v.gripAngle);
  grip.translate(0, gy, gz);
  sink.add(grip, 'polymer', 'body', 0.07);

  const ca = Math.cos(-v.gripAngle);
  const sa = Math.sin(-v.gripAngle);
  for (let i = 0; i < 3; i++) {
    const t = (i + 0.5) / 3;
    const ridge = latheTube(0.0042, 0, 0.0272, 12, 0.0009);
    ridge.rotateY(HALF_PI);
    const yy = -len * (0.16 + t * 0.68);
    const zz = -bw * 0.5 - 0.0022;
    ridge.translate(0, gy + yy * ca - zz * sa, gz + yy * sa + zz * ca);
    sink.add(ridge, 'polymer', 'body', 0.30);
  }

  const beaver = bevelBox(0.0255, 0.0130, 0.0180, 0.0035);
  beaver.rotateX(-v.gripAngle * 0.5);
  beaver.translate(0, gy + 0.0035, gz + 0.0060);
  sink.add(beaver, 'polymer', 'body', 0.12);
}

/** Trigger guard: a shape with a hole, so it is a genuine closed loop. */
function buildTriggerGuard(sink: PartSink, v: WeaponVisual, triggerNode: THREE.Object3D): void {
  const zc = v.receiverZ + v.receiverLength * 0.13;
  const yc = v.receiverY - v.receiverHeight * 0.52;
  const ow = 0.052;
  const oh = 0.033;

  const outer = new THREE.Shape();
  outer.moveTo(-ow * 0.5, oh * 0.5);
  outer.lineTo(ow * 0.5, oh * 0.5);
  outer.lineTo(ow * 0.5, -oh * 0.16);
  outer.quadraticCurveTo(ow * 0.42, -oh * 0.5, ow * 0.05, -oh * 0.5);
  outer.quadraticCurveTo(-ow * 0.42, -oh * 0.5, -ow * 0.5, oh * 0.05);
  outer.closePath();

  const iw = ow - 0.0110;
  const ih = oh - 0.0090;
  const hole = new THREE.Path();
  hole.moveTo(-iw * 0.5, ih * 0.52);
  hole.lineTo(iw * 0.5, ih * 0.52);
  hole.lineTo(iw * 0.5, -ih * 0.18);
  hole.quadraticCurveTo(iw * 0.40, -ih * 0.5, iw * 0.02, -ih * 0.5);
  hole.quadraticCurveTo(-iw * 0.42, -ih * 0.5, -iw * 0.5, ih * 0.06);
  hole.closePath();
  outer.holes.push(hole);

  // -90 about Y so the shape's +X (the vertical rear face) ends up rearward.
  const guard = extrude(outer, 0.0130, { bevel: 0.0012, curveSegments: 6 });
  guard.rotateY(-HALF_PI);
  guard.translate(0, yc + 0.0030, zc);
  sink.add(guard, 'metal', 'body', 0.13);

  const blade = new THREE.Shape();
  blade.moveTo(-0.0026, 0);
  blade.lineTo(0.0026, 0);
  blade.lineTo(0.0040, -0.0130);
  blade.lineTo(-0.0016, -0.0155);
  blade.closePath();
  const bladeGeo = extrude(blade, 0.0068, { bevel: 0.0009 });
  bladeGeo.rotateY(-HALF_PI);
  sink.add(bladeGeo, 'accent', 'trigger', 0.28);
  const shoe = bevelBox(0.0072, 0.0026, 0.0038, 0.0008);
  shoe.translate(0, -0.0142, 0.0006);
  sink.add(shoe, 'accent', 'trigger', 0.52);

  triggerNode.position.set(0, yc + 0.0125, zc - 0.0035);
}

/** Detachable magazine: witness holes, floorplate, feed lips, visible follower. */
function buildMagazine(sink: PartSink, v: WeaponVisual, node: THREE.Object3D): void {
  if (v.mag === 'none' || v.mag === 'tube') return;
  const w = v.magWidth;
  const d = v.magDepth;
  const len = v.magLength;

  // A curved magazine is a swept extrusion, not a bent box: stack short
  // segments, each rotated a little further than the last.
  const segments = v.magCurve > 0.01 ? 7 : 2;
  const segLen = len / segments;
  const pieces: THREE.BufferGeometry[] = [];
  let y = 0;
  let z = 0;
  let ang = 0;
  for (let i = 0; i < segments; i++) {
    const t = i / Math.max(1, segments - 1);
    const taper = 1 - t * (v.mag === 'pistol' ? 0.06 : 0.10);
    const g = bevelBox(w * taper, segLen * 1.04, d * taper, 0.0016);
    g.rotateX(ang);
    g.translate(0, y - segLen * 0.5 * Math.cos(ang), z + segLen * 0.5 * Math.sin(ang));
    pieces.push(g);
    y -= segLen * Math.cos(ang);
    z += segLen * Math.sin(ang);
    ang += v.magCurve / Math.max(1, segments - 1);
  }
  sink.addMany(pieces, 'polymer', 'mag', 0.07);

  const holes = v.mag === 'pistol' ? 4 : 5;
  for (let i = 0; i < holes; i++) {
    const t = (i + 1) / (holes + 1);
    const a = v.magCurve * t;
    const ring = apertureDisc(0.0034, 0.0021, w + 0.0014, 0.0004, 8);
    ring.rotateY(HALF_PI);
    ring.rotateX(a);
    ring.translate(0, -len * t * Math.cos(a * 0.5), len * t * Math.sin(a * 0.5));
    sink.add(ring, 'polymer', 'mag', 0.14);
  }

  const endY = -len * Math.cos(v.magCurve * 0.5);
  const endZ = len * Math.sin(v.magCurve * 0.5);

  const plate = bevelBox(w + 0.0030, 0.0075, d + 0.0030, 0.0014);
  plate.rotateX(v.magCurve);
  plate.translate(0, endY - 0.0025, endZ);
  sink.add(plate, 'polymer', 'mag', 0.16);

  const ledge = bevelBox(w + 0.0030, 0.0034, 0.0075, 0.0010);
  ledge.rotateX(v.magCurve);
  ledge.translate(0, endY - 0.0060, endZ - d * 0.5);
  sink.add(ledge, 'polymer', 'mag', 0.40);

  // Stiffening ribs down the flanks. A magazine hanging under a receiver is a
  // large flat mass in shadow; without something to catch a grazing highlight
  // it disappears into the silhouette entirely.
  for (const s of [-1, 1]) {
    for (let i = 0; i < 2; i++) {
      const rib = bevelBox(0.0018, len * 0.82, 0.0060, 0.0005);
      rib.rotateX(v.magCurve * 0.5);
      rib.translate(s * (w * 0.5 - 0.0004), -len * 0.46, d * (0.16 + i * 0.30) + len * 0.10 * Math.sin(v.magCurve * 0.5));
      sink.add(rib, 'polymer', 'mag', 0.26);
    }
  }

  const follower = bevelBox(w - 0.0035, 0.0050, d - 0.0035, 0.0010);
  follower.translate(0, 0.0018, 0);
  sink.add(follower, 'accent', 'mag', 0.10);

  for (const s of [-1, 1]) {
    const lip = bevelBox(0.0022, 0.0090, d - 0.0020, 0.0006);
    lip.translate(s * (w * 0.5 - 0.0009), 0.0035, 0);
    sink.add(lip, 'metal', 'mag', 0.48);
  }

  node.position.set(0, magWellY(v) - 0.002, magWellZ(v));
}

/** Tube magazine (shotgun) with end cap and barrel band. */
function buildTubeMag(sink: PartSink, v: WeaponVisual, barrelZ0: number): void {
  if (v.mag !== 'tube') return;
  const r = v.magWidth * 0.5;
  const len = v.magLength;
  const y = -v.barrelRadius - r - 0.0018;
  const zc = barrelZ0 - len * 0.5;

  const tube = latheTube(r, 0, len, 16, 0.0010);
  tube.translate(0, y, zc);
  sink.add(tube, 'metal', 'body', 0.14);

  const cap = latheProfile(
    [[0, -0.0060], [r * 0.9, -0.0060], [r * 1.12, -0.0030], [r * 1.12, 0.0030], [r * 0.85, 0.0060], [0, 0.0060]], 16,
  );
  cap.translate(0, y, zc - len * 0.5 - 0.0055);
  sink.add(cap, 'metal', 'body', 0.28);

  const band = latheTube(r + 0.0055, r + 0.0020, 0.0090, 16, 0.0008);
  band.translate(0, y, zc - len * 0.32);
  sink.add(band, 'metal', 'body', 0.28);
}

/** Handguards. Every style has genuinely open geometry somewhere. */
function buildHandguard(sink: PartSink, v: WeaponVisual, z0: number): void {
  if (v.handguard === 'none' || v.handguardLength <= 0) return;
  const len = v.handguardLength;
  const zc = z0 - len * 0.5;
  const rOut = v.handguardRadius;
  const rIn = rOut - 0.0032;

  switch (v.handguard) {
    case 'mlok': {
      // Alternating full rings and slot stations. At a slot station only the
      // 45-degree webs survive, so you see through into the barrel channel.
      const stations = v.handguardSlots;
      const count = stations * 2 + 1;
      const seg = len / count;
      for (let s = 0; s < count; s++) {
        const zz = zc - len * 0.5 + seg * (s + 0.5);
        if (s % 2 === 1) {
          for (let q = 0; q < 4; q++) {
            const g = arcSector(rOut, rIn, q * HALF_PI + 0.30, 0.34, seg * 0.96, 0.0006, 5);
            g.translate(0, 0, zz);
            sink.add(g, 'metal', 'body', 0.18);
          }
        } else {
          const g = latheTube(rOut, rIn, seg * 1.02, 16, 0.0008);
          g.translate(0, 0, zz);
          sink.add(g, 'metal', 'body', 0.14);
        }
      }
      buildRail(sink, len * 0.98, rOut - 0.0012, zc, Math.max(4, Math.round(len * 38)));
      const collar = latheTube(rOut + 0.0028, rIn, 0.0130, 16, 0.0010);
      collar.translate(0, 0, zc + len * 0.5 + 0.004);
      sink.add(collar, 'metal', 'body', 0.28);
      break;
    }
    case 'vented': {
      const shell = latheTube(rOut, rIn, len, 18, 0.0010);
      shell.translate(0, 0, zc);
      sink.add(shell, 'metal', 'body', 0.14);
      const rings = Math.max(2, v.handguardSlots);
      for (let i = 0; i < rings; i++) {
        const zz = zc - len * 0.40 + (len * 0.80) * (i / (rings - 1));
        for (let k = 0; k < 5; k++) {
          const a = (k / 5) * Math.PI * 2 + (i % 2) * 0.62;
          // apertureDisc's axis is +Z: swing it to +X, then round to angle a.
          const hole = apertureDisc(0.0042, 0.0026, 0.0060, 0.0004, 8);
          hole.rotateY(HALF_PI);
          hole.rotateZ(a);
          hole.translate(Math.cos(a) * rOut, Math.sin(a) * rOut, zz);
          sink.add(hole, 'metal', 'body', 0.16);
        }
      }
      buildRail(sink, len * 0.92, rOut - 0.0010, zc, Math.max(4, Math.round(len * 38)));
      break;
    }
    case 'slab': {
      const half = rOut;
      for (let q = 0; q < 4; q++) {
        const a = q * HALF_PI;
        const face = bevelBox(half * 1.42, 0.0048, len, 0.0012);
        face.rotateZ(a);
        face.translate(Math.cos(a + HALF_PI) * (half - 0.0024), Math.sin(a + HALF_PI) * (half - 0.0024), zc);
        sink.add(face, 'metal', 'body', 0.09);
      }
      const slots = Math.max(2, v.handguardSlots);
      for (let i = 0; i < slots; i++) {
        const zz = zc - len * 0.40 + (len * 0.80) * (i / (slots - 1));
        const ring = latheTube(half * 0.98, half * 0.80, 0.0060, 18, 0.0008);
        ring.translate(0, 0, zz);
        sink.add(ring, 'metal', 'body', 0.22);
      }
      buildRail(sink, len * 0.96, half + 0.0008, zc, Math.max(4, Math.round(len * 38)));
      break;
    }
    case 'pump':
      // Built by buildPump so it can ride on its own animated node.
      break;
  }
}

/** Muzzle devices. Ports are the gaps between arc sectors: real holes. */
function buildMuzzle(sink: PartSink, v: WeaponVisual, zTip: number): number {
  const rb = v.barrelRadius;
  switch (v.muzzle) {
    case 'birdcage': {
      const len = 0.048;
      const rOut = rb + 0.0055;
      const rIn = rb + 0.0012;
      const zc = zTip - len * 0.5;
      const collar = latheTube(rOut, rIn, 0.010, 20, 0.0010);
      collar.translate(0, 0, zc + len * 0.5 - 0.005);
      sink.add(collar, 'metal', 'body', 0.32);
      // Six tines; the gaps between them are the flash-hider slots.
      for (let i = 0; i < 6; i++) {
        const a = (i / 6) * Math.PI * 2 + Math.PI / 6;
        const g = arcSector(rOut, rIn, a - 0.30, 0.60, len - 0.016, 0.0005, 6);
        g.translate(0, 0, zc - 0.004);
        sink.add(g, 'metal', 'body', 0.30);
      }
      const front = latheTube(rOut, rIn, 0.0055, 20, 0.0008);
      front.translate(0, 0, zc - len * 0.5 + 0.0027);
      sink.add(front, 'metal', 'body', 0.30);
      return zTip - len;
    }
    case 'brake': {
      const len = 0.062;
      const rOut = rb + 0.0080;
      const rIn = rb + 0.0015;
      const zc = zTip - len * 0.5;
      for (let i = 0; i <= 3; i++) {
        const zz = zc + len * 0.5 - 0.0055 - i * (len - 0.011) / 3;
        const baffle = latheTube(rOut, rIn, 0.0060, 22, 0.0009);
        baffle.translate(0, 0, zz);
        sink.add(baffle, 'metal', 'body', 0.34);
      }
      // Walls only at 4 and 8 o'clock: top and sides stay open ports.
      for (const a of [Math.PI * 1.15, Math.PI * 1.85]) {
        const wall = arcSector(rOut, rIn, a - 0.34, 0.68, len - 0.012, 0.0006, 6);
        wall.translate(0, 0, zc);
        sink.add(wall, 'metal', 'body', 0.26);
      }
      return zTip - len;
    }
    case 'comp': {
      const len = 0.036;
      const rOut = rb + 0.0048;
      const rIn = rb + 0.0010;
      const zc = zTip - len * 0.5;
      const body = latheTube(rOut, rIn, len, 20, 0.0010);
      body.translate(0, 0, zc);
      sink.add(body, 'metal', 'body', 0.26);
      for (let i = 0; i < 3; i++) {
        const zz = zc - len * 0.28 + i * len * 0.26;
        for (const s of [-1, 1]) {
          const port = bevelBox(0.0034, rOut * 2.4, 0.0055, 0.0006);
          port.rotateZ(s * 0.35);
          port.translate(0, rOut * 0.55, zz);
          sink.add(port, 'accent', 'body', 0.28);
        }
      }
      return zTip - len;
    }
    case 'choke': {
      const len = 0.030;
      const g = latheProfile([
        [rb * 0.62, -len * 0.5], [rb + 0.0042, -len * 0.5], [rb + 0.0042, len * 0.32],
        [rb + 0.0035, len * 0.5], [rb * 0.62, len * 0.5], [rb * 0.62, -len * 0.5 + 1e-5],
      ], 20);
      g.translate(0, 0, zTip - len * 0.5);
      sink.add(g, 'metal', 'body', 0.28);
      return zTip - len;
    }
    default: {
      const crown = latheTube(rb + 0.0012, rb * 0.55, 0.0060, 18, 0.0010);
      crown.translate(0, 0, zTip - 0.003);
      sink.add(crown, 'metal', 'body', 0.34);
      return zTip - 0.006;
    }
  }
}

/** Barrel, gas block and gas tube. Returns the Z of the bare muzzle crown. */
function buildBarrel(sink: PartSink, v: WeaponVisual, z0: number): number {
  const len = v.barrelLength;
  const rb = v.barrelRadius;

  // Stepped profile: heavy chamber, thin under the handguard, fatter at the
  // muzzle. A constant-diameter tube reads as plumbing, not as a barrel.
  const barrel = latheProfile([
    [0, -len * 0.5],
    [rb * 0.98, -len * 0.5],
    [rb * 0.98, -len * 0.34],
    [rb * 0.86, -len * 0.28],
    [rb * 0.86, len * 0.20],
    [rb * 1.10, len * 0.24],
    [rb * 1.10, len * 0.5 - 0.004],
    [rb * 0.95, len * 0.5],
    [0, len * 0.5],
  ], 18);
  barrel.translate(0, 0, z0 - len * 0.5);
  sink.add(barrel, 'metal', 'body', 0.16);

  if (v.handguard !== 'none' && v.handguard !== 'pump') {
    const gz = z0 - v.handguardLength - 0.016;
    const block = bevelBox(0.0175, 0.0175, 0.0230, 0.0018);
    block.translate(0, 0, gz);
    sink.add(block, 'metal', 'body', 0.14);
    const gasTube = latheTube(0.0022, 0, v.handguardLength * 0.9, 10, 0.0004);
    gasTube.translate(0, rb + 0.0055, gz + v.handguardLength * 0.45);
    sink.add(gasTube, 'metal', 'body', 0.12);
  }
  return z0 - len;
}

/** Stocks. Shapes are authored with +X rearward, then rotated onto the bore. */
function buildStock(sink: PartSink, v: WeaponVisual): void {
  if (v.stock === 'none' || v.stockLength <= 0) return;
  const zBack = v.receiverZ + v.receiverLength * 0.5;
  const len = v.stockLength;
  const y = v.receiverY - v.stockDrop;

  switch (v.stock) {
    case 'collapsible': {
      const tube = latheTube(0.0142, 0.0110, len * 0.92, 18, 0.0010);
      tube.translate(0, y, zBack + len * 0.46);
      sink.add(tube, 'metal', 'body', 0.20);
      // The body slides on the tube: a frame plate whose hole runs along it.
      const body = framePlate(0.0300, 0.0330, len * 0.58, 0.0230, 0.0240, 0.0035, 0.0022);
      body.translate(0, y - 0.0020, zBack + len * 0.62);
      sink.add(body, 'polymer', 'body', 0.07);
      const cheek = bevelBox(0.0230, 0.0110, len * 0.50, 0.0028);
      cheek.translate(0, y + 0.0160, zBack + len * 0.58);
      sink.add(cheek, 'polymer', 'body', 0.14);

      // Buttpad with a proper toe and heel, and a hard backing plate behind the
      // rubber. A single slab there was the largest unbroken mass on screen.
      const pad = bevelBox(0.0330, 0.0420, 0.0112, 0.0038);
      pad.translate(0, y - 0.0010, zBack + len * 0.93);
      sink.add(pad, 'polymer', 'body', 0.44);
      const backing = bevelBox(0.0300, 0.0400, 0.0044, 0.0030);
      backing.translate(0, y - 0.0010, zBack + len * 0.88);
      sink.add(backing, 'accent', 'body', 0.14);

      // Position rail on the underside with its detent holes drilled through:
      // the six clicks of length adjustment, and the clearest signal in the
      // whole silhouette that this is a collapsible stock and not a block.
      const detent = bevelBox(0.0074, 0.0062, len * 0.78, 0.0010);
      detent.translate(0, y - 0.0148, zBack + len * 0.52);
      sink.add(detent, 'metal', 'body', 0.22);
      for (let i = 0; i < 6; i++) {
        const hole = apertureDisc(0.0030, 0.0017, 0.0080, 0.0004, 8);
        hole.rotateX(HALF_PI);
        hole.translate(0, y - 0.0148, zBack + len * (0.22 + i * 0.118));
        sink.add(hole, 'metal', 'body', 0.34);
      }

      // Release lever hanging under the stock body, and a QD socket in its side.
      const lever = bevelBox(0.0120, 0.0130, 0.0180, 0.0022);
      lever.rotateX(0.22);
      lever.translate(0, y - 0.0230, zBack + len * 0.60);
      sink.add(lever, 'polymer', 'body', 0.46);
      for (const s of [-1, 1]) {
        const qd = apertureDisc(0.0056, 0.0027, 0.0042, 0.0006, 10);
        qd.rotateY(HALF_PI);
        qd.translate(s * 0.0150, y - 0.0055, zBack + len * 0.50);
        sink.add(qd, 'metal', 'body', 0.34);
      }
      break;
    }
    case 'fixed': {
      const shape = new THREE.Shape();
      shape.moveTo(0, 0.020);
      shape.lineTo(len, 0.026);
      shape.lineTo(len, -0.026);
      shape.lineTo(len * 0.35, -0.030);
      shape.lineTo(0, -0.016);
      shape.closePath();
      const body = extrude(shape, 0.0335, { bevel: 0.0026, segments: 2 });
      body.rotateY(-HALF_PI); // shape +X becomes +Z: rearward
      body.translate(0, y, zBack);
      sink.add(body, 'polymer', 'body', 0.07);
      const pad = bevelBox(0.0345, 0.0560, 0.0140, 0.0042);
      pad.translate(0, y - 0.0020, zBack + len + 0.006);
      sink.add(pad, 'polymer', 'body', 0.20);
      const loop = apertureDisc(0.0060, 0.0034, 0.0032, 0.0006, 10);
      loop.rotateY(HALF_PI);
      loop.translate(0, y - 0.0230, zBack + len * 0.55);
      sink.add(loop, 'metal', 'body', 0.42);
      break;
    }
    case 'skeleton': {
      const tube = latheTube(0.0110, 0.0082, len * 0.85, 16, 0.0008);
      tube.translate(0, y + 0.004, zBack + len * 0.42);
      sink.add(tube, 'metal', 'body', 0.28);
      const strutTop = bevelBox(0.0180, 0.0055, len * 0.55, 0.0012);
      strutTop.translate(0, y + 0.0175, zBack + len * 0.60);
      sink.add(strutTop, 'metal', 'body', 0.12);
      const strutBot = bevelBox(0.0180, 0.0055, len * 0.45, 0.0012);
      strutBot.rotateX(-0.16);
      strutBot.translate(0, y - 0.0150, zBack + len * 0.58);
      sink.add(strutBot, 'metal', 'body', 0.12);
      const pad = bevelBox(0.0230, 0.0470, 0.0110, 0.0032);
      pad.rotateX(-0.06);
      pad.translate(0, y + 0.0020, zBack + len * 0.90);
      sink.add(pad, 'polymer', 'body', 0.24);
      break;
    }
    case 'sniper': {
      const shape = new THREE.Shape();
      shape.moveTo(0, 0.026);
      shape.lineTo(len * 0.55, 0.030);
      shape.lineTo(len, 0.034);
      shape.lineTo(len, -0.030);
      shape.lineTo(len * 0.42, -0.036);
      shape.lineTo(0, -0.018);
      shape.closePath();
      const hole = new THREE.Path();
      hole.absarc(len * 0.30, -0.002, 0.0135, 0, Math.PI * 2, true);
      shape.holes.push(hole); // thumbhole: a real hole through the wrist
      const body = extrude(shape, 0.0320, { bevel: 0.0026, segments: 2, curveSegments: 12 });
      body.rotateY(-HALF_PI);
      body.translate(0, y, zBack);
      sink.add(body, 'polymer', 'body', 0.07);
      const cheek = bevelBox(0.0270, 0.0160, len * 0.52, 0.0034);
      cheek.translate(0, y + 0.0330, zBack + len * 0.52);
      sink.add(cheek, 'polymer', 'body', 0.14);
      for (let i = 0; i < 2; i++) {
        const post = latheTube(0.0030, 0, 0.0170, 10, 0.0004);
        post.rotateX(HALF_PI);
        post.translate((i ? 1 : -1) * 0.0080, y + 0.0210, zBack + len * (0.34 + i * 0.36));
        sink.add(post, 'metal', 'body', 0.30);
      }
      const pad = bevelBox(0.0330, 0.0620, 0.0150, 0.0044);
      pad.translate(0, y + 0.0010, zBack + len + 0.007);
      sink.add(pad, 'polymer', 'body', 0.20);
      break;
    }
  }
}

/** Pistol slide: upper, charging surface and barrel shroud in one assembly. */
function buildSlide(sink: PartSink, v: WeaponVisual): number {
  const len = v.receiverLength;
  const w = v.receiverWidth;
  const h = v.receiverHeight;
  const zc = v.receiverZ;
  const yc = v.receiverY + h * 0.55;

  const shape = new THREE.Shape();
  shape.moveTo(-w * 0.5, -h * 0.5);
  shape.lineTo(w * 0.5, -h * 0.5);
  shape.lineTo(w * 0.5, h * 0.34);
  shape.lineTo(w * 0.30, h * 0.5);
  shape.lineTo(-w * 0.30, h * 0.5);
  shape.lineTo(-w * 0.5, h * 0.34);
  shape.closePath();
  const body = extrude(shape, len, { bevel: 0.0016, segments: 2 });
  body.translate(0, yc, zc);
  sink.add(body, 'metal', 'bolt', 0.10);

  // Cocking serrations — the single most recognisable pistol detail.
  for (let i = 0; i < 7; i++) {
    const g = bevelBox(w + 0.0012, h * 0.62, 0.0022, 0.0005);
    g.translate(0, yc - h * 0.04, zc + len * 0.5 - 0.006 - i * 0.0044);
    sink.add(g, 'metal', 'bolt', 0.40);
  }
  for (let i = 0; i < 4; i++) {
    const g = bevelBox(w + 0.0010, h * 0.50, 0.0020, 0.0005);
    g.translate(0, yc - h * 0.04, zc - len * 0.34 - i * 0.0042);
    sink.add(g, 'metal', 'bolt', 0.40);
  }

  const side = v.ejectSide;
  const port = framePlate(0.030, 0.0135, 0.0034, 0.024, 0.0090, 0.0016, 0.0010);
  port.rotateY(side * HALF_PI);
  port.translate(side * (w * 0.5 - 0.0004), yc + h * 0.10, zc - len * 0.20);
  sink.add(port, 'metal', 'bolt', 0.30);

  const crown = latheTube(v.barrelRadius, v.barrelRadius * 0.52, 0.0110, 18, 0.0010);
  crown.translate(0, yc + h * 0.06, zc - len * 0.5 - 0.003);
  sink.add(crown, 'metal', 'bolt', 0.32);
  const plug = latheTube(0.0052, 0.0024, 0.0060, 14, 0.0008);
  plug.translate(0, yc - h * 0.26, zc - len * 0.5 - 0.001);
  sink.add(plug, 'metal', 'bolt', 0.28);

  return yc + h * 0.06;
}

/** Pump forend riding on the magazine tube; cycles on every shot. */
function buildPump(sink: PartSink, v: WeaponVisual, barrelZ0: number, node: THREE.Object3D): void {
  const len = v.handguardLength;
  const rOut = v.handguardRadius;

  const shell = latheProfile([
    [0, -len * 0.5], [rOut * 0.70, -len * 0.5], [rOut, -len * 0.34],
    [rOut, len * 0.34], [rOut * 0.70, len * 0.5], [0, len * 0.5],
  ], 22);
  sink.add(shell, 'polymer', 'charge', 0.12);
  sink.addMany(
    ribbedRings(rOut, 0.0018, v.handguardSlots, len / (v.handguardSlots + 1), 0.0044, 20),
    'polymer', 'charge', 0.36,
  );
  const bar = bevelBox(0.0070, 0.0060, len * 0.9, 0.0012);
  bar.translate(0, v.magWidth * 0.5 + 0.0060, 0);
  sink.add(bar, 'metal', 'charge', 0.24);

  node.position.set(0, -v.barrelRadius - v.magWidth * 0.5 - 0.0018, barrelZ0 - v.magLength * 0.42);
}

// ---------------------------------------------------------------------------
// Optics
// ---------------------------------------------------------------------------

interface OpticResult {
  /** Local position of the eye point; the sight line runs -Z from here. */
  sightPos: THREE.Vector3;
  reticleNode: THREE.Object3D | null;
  reticleMaterial: THREE.MeshBasicMaterial | null;
  /** Radius of the visible sight picture, for the scope occluder. */
  scopeRadius: number;
}

function makeReticleMesh(
  materials: GunMaterials, kind: ReticleKind, size: number, color: number, opacity: number,
): { mesh: THREE.Mesh; material: THREE.MeshBasicMaterial } {
  const material = new THREE.MeshBasicMaterial({
    map: materials.reticle(kind),
    // Deliberately over unity: an illuminated reticle is an emitter, so its
    // core has to clip white and leave the hue in the skirt. At exactly 1.0 the
    // dot renders as flat paint the colour of the LED.
    color: new THREE.Color(color).multiplyScalar(2.6),
    transparent: true,
    opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), material);
  mesh.renderOrder = 12;
  mesh.frustumCulled = false;
  return { mesh, material };
}

function buildOptic(
  sink: PartSink, v: WeaponVisual, spec: WeaponSpec, materials: GunMaterials,
  opticNode: THREE.Object3D, railY: number, muzzleZ: number,
): OpticResult {
  const sightY = v.receiverY + v.opticHeight;
  const baseZ = v.receiverZ - v.receiverLength * 0.08;
  const out: OpticResult = {
    sightPos: new THREE.Vector3(0, sightY, 0),
    reticleNode: null,
    reticleMaterial: null,
    scopeRadius: 0.02,
  };

  /**
   * Rail clamp plus riser. The clamp is a frame whose hole genuinely swallows
   * the rail — you can see daylight between the two — and it is bolted through
   * with hex screws that have real heads. `topY` is the underside of whatever
   * is being mounted, so the riser meets it instead of guessing.
   */
  const mountBase = (w: number, lenAlongBore: number, topY: number, z: number): void => {
    const clampH = 0.0150;
    const clampY = railY - 0.0044;
    const clamp = framePlate(w, clampH, lenAlongBore, 0.0216, 0.0102, 0.0018, 0.0010);
    clamp.translate(0, clampY, z);
    sink.add(clamp, 'metal', 'body', 0.11);

    // Recoil lug: the tab that drops into a rail slot and stops the optic
    // walking forward. Small, but it is why the mount reads as a mount.
    const lug = bevelBox(0.0190, 0.0044, 0.0046, 0.0008);
    lug.translate(0, railY - RAIL_HEIGHT + 0.0022, z - lenAlongBore * 0.5 + 0.0030);
    sink.add(lug, 'metal', 'body', 0.30);

    const riseTop = topY + 0.0010;
    const riseBot = clampY + clampH * 0.5 - 0.0012;
    if (riseTop > riseBot + 0.0012) {
      const riser = bevelBox(w * 0.50, riseTop - riseBot, lenAlongBore * 0.62, 0.0020);
      riser.translate(0, (riseTop + riseBot) * 0.5, z);
      sink.add(riser, 'metal', 'body', 0.09);
    }

    for (const s of [-1, 1]) {
      const bolt = hexHead(0.0038, 0.0034);
      bolt.rotateY(HALF_PI);
      bolt.translate(s * (w * 0.5 + 0.0014), clampY - 0.0034, z);
      sink.add(bolt, 'accent', 'body', 0.30);
    }
  };

  switch (spec.optic) {
    case 'irons': {
      const rearZ = v.receiverZ + v.receiverLength * 0.34;
      const ap = apertureDisc(0.0072, 0.0027, 0.0026, 0.0005);
      ap.translate(0, sightY, rearZ);
      sink.add(ap, 'metal', 'body', 0.34);
      const rearBase = bevelBox(0.0180, Math.max(0.004, sightY - railY), 0.0100, 0.0014);
      rearBase.translate(0, (sightY + railY) * 0.5 - 0.0030, rearZ);
      sink.add(rearBase, 'metal', 'body', 0.12);
      for (const s of [-1, 1]) {
        const ear = bevelBox(0.0032, 0.0130, 0.0070, 0.0008);
        ear.translate(s * 0.0088, sightY - 0.0010, rearZ);
        sink.add(ear, 'metal', 'body', 0.30);
      }

      const frontZ = muzzleZ + 0.022;
      const post = bevelBox(0.0022, 0.0125, 0.0026, 0.0004);
      post.translate(0, sightY - 0.0005, frontZ);
      sink.add(post, 'metal', 'body', 0.34);
      for (const s of [-1, 1]) {
        const wing = bevelBox(0.0028, 0.0155, 0.0060, 0.0007);
        wing.rotateZ(s * 0.10);
        wing.translate(s * 0.0072, sightY - 0.0018, frontZ);
        sink.add(wing, 'metal', 'body', 0.34);
      }
      const fBase = bevelBox(0.0180, 0.0080, 0.0130, 0.0014);
      fBase.translate(0, sightY - 0.0110, frontZ);
      sink.add(fBase, 'metal', 'body', 0.16);

      out.sightPos.set(0, sightY, rearZ);
      out.scopeRadius = 0.05;
      break;
    }

    /**
     * A tube red dot, built the way one is actually made: an ocular ring, a
     * slim waist, an objective bell that steps back out, and a chamfer onto the
     * front face. That silhouette — fat, thin, fat — is what says "optic" at a
     * glance, where a constant-diameter tube says "pipe". Turrets, a battery
     * cap, brightness buttons and hex hardware do the rest.
     */
    case 'reddot': {
      const len = 0.072;
      const L2 = len * 0.5;
      const rOcu = 0.0180;   // ocular housing
      const rMid = 0.0158;   // machined waist
      const rObj = 0.0212;   // objective bell
      const rBore = 0.0140;  // clear bore you actually look through
      const z = baseZ;
      mountBase(0.0300, len * 0.46, sightY - rMid, z + 0.0110);

      const body = latheProfile([
        [rBore, -L2],
        [rOcu, -L2],
        [rOcu, -L2 + 0.0072],
        [rMid, -L2 + 0.0098],
        [rMid, L2 - 0.0250],
        [rObj, L2 - 0.0176],
        [rObj, L2 - 0.0022],
        [rObj - 0.0020, L2],
        [rBore + 0.0022, L2],
        [rBore, -L2 + 1e-5],
      ], 26);
      body.translate(0, sightY, z);
      sink.add(body, 'metal', 'body', 0.11);

      // Protective rims stand proud of the glass at both ends.
      const objRim = latheTube(rObj + 0.0016, rObj - 0.0026, 0.0046, 18, 0.0008);
      objRim.translate(0, sightY, z - L2 + 0.0023);
      sink.add(objRim, 'metal', 'body', 0.22);
      const ocuRim = latheTube(rOcu + 0.0014, rOcu - 0.0030, 0.0042, 18, 0.0007);
      ocuRim.translate(0, sightY, z + L2 - 0.0021);
      sink.add(ocuRim, 'metal', 'body', 0.26);

      // Panel seam where the two housing halves meet.
      const seam = latheTube(rMid + 0.0011, rMid - 0.0004, 0.0018, 16, 0.0004);
      seam.translate(0, sightY, z + L2 - 0.0220);
      sink.add(seam, 'accent', 'body', 0.10);

      /** Elevation / windage turret: boss, knurled cap, slotted adjuster. */
      const turret = (rot: (g: THREE.BufferGeometry) => void, ox: number, oy: number, oz: number): void => {
        const boss = latheTube(0.0092, 0, 0.0040, 14, 0.0008);
        rot(boss); boss.translate(ox, oy, oz);
        sink.add(boss, 'metal', 'body', 0.12);
        for (const g of knurledCap(0.0074, 0.0092, 10)) {
          rot(g); g.translate(ox, oy, oz);
          sink.add(g, 'metal', 'body', 0.24);
        }
        const s = slotScrew(0.0044, 0.0026, 10);
        rot(s.head); s.head.translate(ox, oy, oz);
        sink.add(s.head, 'metal', 'body', 0.34);
        rot(s.slot); s.slot.translate(ox, oy, oz);
        sink.add(s.slot, 'accent', 'body', 0.06);
      };
      const turretZ = z + L2 - 0.0300;
      turret((g) => g.rotateX(-HALF_PI), 0, sightY + rMid + 0.0050, turretZ);
      turret((g) => g.rotateY(-HALF_PI), -(rMid + 0.0050), sightY, turretZ);

      // Battery compartment on the far side, with its own knurl and slot.
      const batBoss = latheTube(0.0098, 0, 0.0032, 16, 0.0008);
      batBoss.rotateY(HALF_PI);
      batBoss.translate(rMid + 0.0016, sightY, z + 0.0050);
      sink.add(batBoss, 'metal', 'body', 0.12);
      for (const g of knurledCap(0.0086, 0.0072, 12)) {
        g.rotateY(HALF_PI);
        g.translate(rMid + 0.0068, sightY, z + 0.0050);
        sink.add(g, 'metal', 'body', 0.26);
      }
      const batSlot = bevelBox(0.0016, 0.0110, 0.0030, 0.0004, 0.0002);
      batSlot.rotateY(HALF_PI);
      batSlot.translate(rMid + 0.0104, sightY, z + 0.0050);
      sink.add(batSlot, 'accent', 'body', 0.08);

      // Brightness rocker: two proud pads, rubbed bright by a thumb.
      for (let i = 0; i < 2; i++) {
        const b = bevelBox(0.0030, 0.0062, 0.0066, 0.0012);
        b.translate(rMid + 0.0026, sightY - 0.0074 + i * 0.0148, z - 0.0090);
        sink.add(b, 'accent', 'body', 0.40);
      }

      // Glass. Objective and ocular are merged into one mesh so a second lens
      // costs geometry, not a draw call.
      const objGlass = new THREE.CircleGeometry(rBore + 0.0018, 28);
      objGlass.translate(0, 0, -L2 + 0.0072);
      const ocuGlass = new THREE.CircleGeometry(rBore - 0.0004, 28);
      ocuGlass.translate(0, 0, L2 - 0.0064);
      const glass = mergeGeometries([objGlass, ocuGlass], false);
      objGlass.dispose();
      ocuGlass.dispose();
      const lens = new THREE.Mesh(glass ?? new THREE.CircleGeometry(rBore, 28), materials.lens);
      lens.position.set(0, sightY, z);
      lens.renderOrder = 10;
      lens.frustumCulled = false;
      opticNode.add(lens);

      // The dot belongs at the ocular, not the objective. Sitting on the front
      // element it was 64 mm down a 28 mm bore, so at the ~19 degrees off-axis
      // that hip fire always views the sight from, the housing occluded it
      // completely — the aiming point was never drawn in a single frame. Here it
      // floats 1 mm proud of the rear element, where the collimated dot appears
      // to the shooter anyway, and the parallax shift reads as the dot sliding
      // across the glass exactly as it does on a real sight.
      const r = makeReticleMesh(materials, 'dot', rBore * 2.1, v.reticleColor, 1.0);
      r.mesh.position.set(0, sightY, z + L2 - 0.0054);
      opticNode.add(r.mesh);
      out.reticleNode = r.mesh;
      out.reticleMaterial = r.material;
      out.sightPos.set(0, sightY, z + L2 + 0.004);
      out.scopeRadius = rBore;
      break;
    }

    case 'holo': {
      const len = 0.070;
      const w = 0.0330;
      const hoodH = 0.0330;
      const z = baseZ - 0.004;
      mountBase(0.0320, 0.026, sightY - hoodH * 0.5, z + len * 0.42);

      // Open hood: four walls, front and rear fully open.
      for (const s of [-1, 1]) {
        const side = bevelBox(0.0038, hoodH, len * 0.72, 0.0012);
        side.translate(s * (w * 0.5 - 0.0019), sightY, z - len * 0.10);
        sink.add(side, 'metal', 'body', 0.12);
      }
      for (const s of [-1, 1]) {
        const cap = bevelBox(w, 0.0042, len * 0.72, 0.0012);
        cap.translate(0, sightY + s * (hoodH * 0.5 - 0.0021), z - len * 0.10);
        sink.add(cap, 'metal', 'body', 0.12);
      }
      const rear = bevelBox(w, hoodH * 0.92, len * 0.34, 0.0026);
      rear.translate(0, sightY, z + len * 0.36);
      sink.add(rear, 'polymer', 'body', 0.08);
      for (let i = 0; i < 3; i++) {
        const btn = latheTube(0.0034, 0, 0.0026, 12, 0.0004);
        btn.rotateX(HALF_PI);
        btn.translate(-0.0060 + i * 0.0060, sightY + hoodH * 0.48, z + len * 0.36);
        sink.add(btn, 'accent', 'body', 0.34);
      }

      const glassW = w - 0.0090;
      const glassH = hoodH - 0.0110;
      const glass = new THREE.Mesh(new THREE.PlaneGeometry(glassW, glassH), materials.lens);
      glass.rotation.x = 0.14; // holographic windows sit slightly canted
      glass.position.set(0, sightY, z - len * 0.28);
      glass.renderOrder = 10;
      glass.frustumCulled = false;
      opticNode.add(glass);

      const r = makeReticleMesh(materials, 'holo', glassH * 1.55, v.reticleColor, 0.95);
      r.mesh.position.set(0, sightY, z - len * 0.26);
      opticNode.add(r.mesh);
      out.reticleNode = r.mesh;
      out.reticleMaterial = r.material;
      out.sightPos.set(0, sightY, z + len * 0.50);
      out.scopeRadius = glassH * 0.5;
      break;
    }

    case 'acog': {
      const len = 0.098;
      const z = baseZ - 0.004;
      mountBase(0.0300, 0.030, sightY - 0.0148, z + len * 0.30);

      // Objective bell forward, slim ocular aft. Profile +y ends up forward.
      const body = latheProfile([
        [0, -len * 0.5], [0.0140, -len * 0.5], [0.0158, -len * 0.46],
        [0.0158, -len * 0.26], [0.0125, -len * 0.16],
        [0.0135, len * 0.20], [0.0180, len * 0.30],
        [0.0180, len * 0.44], [0.0175, len * 0.5], [0, len * 0.5],
      ], 24);
      body.translate(0, sightY, z);
      sink.add(body, 'metal', 'body', 0.16);

      const pipe = latheTube(0.0022, 0, len * 0.44, 10, 0.0004);
      pipe.translate(0, sightY + 0.0158, z - len * 0.02);
      sink.add(pipe, 'accent', 'body', 0.22);

      const t1 = latheTube(0.0068, 0, 0.0120, 14, 0.0008);
      t1.rotateX(HALF_PI);
      t1.translate(0, sightY + 0.0155, z + len * 0.06);
      sink.add(t1, 'metal', 'body', 0.34);
      const t2 = latheTube(0.0068, 0, 0.0120, 14, 0.0008);
      t2.rotateY(HALF_PI);
      t2.translate(-0.0150, sightY, z + len * 0.06);
      sink.add(t2, 'metal', 'body', 0.34);

      const shade = latheTube(0.0182, 0.0158, 0.0180, 24, 0.0008);
      shade.translate(0, sightY, z - len * 0.5 - 0.008);
      sink.add(shade, 'metal', 'body', 0.28);
      const eyeRing = latheTube(0.0165, 0.0122, 0.0080, 22, 0.0010);
      eyeRing.translate(0, sightY, z + len * 0.5 + 0.003);
      sink.add(eyeRing, 'metal', 'body', 0.26);

      // Ocular and objective glass in one mesh: from the hip you are looking at
      // the front of this scope, and an unglazed objective is a black hole.
      const acogOc = new THREE.CircleGeometry(0.0120, 26);
      acogOc.translate(0, 0, len * 0.5 - 0.002);
      const acogObj = new THREE.CircleGeometry(0.0142, 26);
      acogObj.translate(0, 0, -len * 0.5 + 0.005);
      const acogGlass = mergeGeometries([acogOc, acogObj], false);
      acogOc.dispose();
      acogObj.dispose();
      const oc = new THREE.Mesh(acogGlass ?? new THREE.CircleGeometry(0.0120, 26), materials.lens);
      oc.position.set(0, sightY, z);
      oc.renderOrder = 10;
      oc.frustumCulled = false;
      opticNode.add(oc);

      const r = makeReticleMesh(materials, 'chevron', 0.0230, v.reticleColor, 0.9);
      r.mesh.position.set(0, sightY, z + len * 0.5 - 0.004);
      opticNode.add(r.mesh);
      out.reticleNode = r.mesh;
      out.reticleMaterial = r.material;
      out.sightPos.set(0, sightY, z + len * 0.5 + 0.012);
      out.scopeRadius = 0.0120;
      break;
    }

    case 'sniper': {
      const len = 0.185;
      const z = baseZ - 0.010;
      for (const s of [-1, 1]) {
        const ringZ = z + s * len * 0.26;
        const h = Math.max(0.004, sightY - 0.0180 - railY);
        const post = bevelBox(0.0260, h + 0.0060, 0.0155, 0.0020);
        post.translate(0, railY + (h + 0.006) * 0.5 - 0.0010, ringZ);
        sink.add(post, 'metal', 'body', 0.12);
        const ring = latheTube(0.0180, 0.0138, 0.0155, 22, 0.0010);
        ring.translate(0, sightY, ringZ);
        sink.add(ring, 'metal', 'body', 0.28);
        for (const t of [-1, 1]) {
          const bolt = latheTube(0.0026, 0, 0.0050, 10, 0.0004);
          bolt.rotateY(HALF_PI);
          bolt.translate(t * 0.0182, sightY - 0.0060, ringZ);
          sink.add(bolt, 'accent', 'body', 0.28);
        }
      }

      const body = latheProfile([
        [0, -len * 0.5], [0.0185, -len * 0.5], [0.0205, -len * 0.46],
        [0.0205, -len * 0.30], [0.0136, -len * 0.20],
        [0.0140, len * 0.24], [0.0245, len * 0.32],
        [0.0245, len * 0.44], [0.0240, len * 0.5], [0, len * 0.5],
      ], 26);
      body.translate(0, sightY, z);
      sink.add(body, 'metal', 'body', 0.14);

      const turret = latheTube(0.0092, 0, 0.0160, 16, 0.0010);
      turret.rotateX(HALF_PI);
      turret.translate(0, sightY + 0.0175, z - len * 0.02);
      sink.add(turret, 'metal', 'body', 0.32);
      for (const g of ribbedRings(0.0092, 0.0007, 3, 0.0035, 0.0018, 16)) {
        g.rotateX(HALF_PI);
        g.translate(0, sightY + 0.0210, z - len * 0.02);
        sink.add(g, 'metal', 'body', 0.28);
      }
      const turretW = latheTube(0.0088, 0, 0.0150, 16, 0.0010);
      turretW.rotateY(HALF_PI);
      turretW.translate(-0.0180, sightY, z - len * 0.02);
      sink.add(turretW, 'metal', 'body', 0.32);

      for (const g of ribbedRings(0.0206, 0.0012, 5, 0.0042, 0.0024, 22)) {
        g.translate(0, sightY, z + len * 0.40);
        sink.add(g, 'metal', 'body', 0.26);
      }
      const objRing = latheTube(0.0250, 0.0212, 0.0080, 26, 0.0010);
      objRing.translate(0, sightY, z - len * 0.5 + 0.003);
      sink.add(objRing, 'metal', 'body', 0.26);

      const snOc = new THREE.CircleGeometry(0.0130, 28);
      snOc.translate(0, 0, len * 0.5 - 0.003);
      const snObj = new THREE.CircleGeometry(0.0206, 28);
      snObj.translate(0, 0, -len * 0.5 + 0.007);
      const snGlass = mergeGeometries([snOc, snObj], false);
      snOc.dispose();
      snObj.dispose();
      const oc = new THREE.Mesh(snGlass ?? new THREE.CircleGeometry(0.0130, 28), materials.lens);
      oc.position.set(0, sightY, z);
      oc.renderOrder = 10;
      oc.frustumCulled = false;
      opticNode.add(oc);

      const r = makeReticleMesh(materials, 'mildot', 0.0250, v.reticleColor, 0.85);
      r.mesh.position.set(0, sightY, z + len * 0.5 - 0.005);
      opticNode.add(r.mesh);
      out.reticleNode = r.mesh;
      out.reticleMaterial = r.material;
      out.sightPos.set(0, sightY, z + len * 0.5 + 0.020);
      out.scopeRadius = 0.0130;
      break;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Hands
// ---------------------------------------------------------------------------

/**
 * A gloved hand wrapped around a cylinder of radius `R` whose axis is +Z, palm
 * centred on +X. Palm and fingers are arc sectors, which means the hand is
 * genuinely curved around what it is holding rather than being a box with
 * finger-shaped boxes glued on — and it costs the same. The caller rotates the
 * node to aim the palm wherever the grip actually is.
 */
function buildWrapHand(
  sink: PartSink, group: PartGroup, R: number, depth: number, wristDir: 1 | -1, forearm: boolean,
): void {
  const palm = arcSector(R + 0.0175, R + 0.0012, -1.15, 2.30, depth, 0.0030, 12);
  sink.add(palm, 'glove', group, 0.05);

  // Heel of the hand: a thicker pad on the wrist side.
  const heel = arcSector(R + 0.0205, R + 0.0060, -0.85, 1.70, depth * 0.34, 0.0026, 10);
  heel.translate(0, 0, wristDir * depth * 0.30);
  sink.add(heel, 'glove', group, 0.08);

  for (let f = 0; f < 4; f++) {
    const t = f / 3;
    // Index nearest the wrist, pinky furthest: the natural hand rake.
    const z = wristDir * (depth * 0.34 - t * depth * 0.72);
    const reach = 1 - Math.abs(t - 0.28) * 0.26;
    const width = depth * 0.20 * (1 - t * 0.18);
    const rF = R + 0.0112 - t * 0.0008;

    const proximal = arcSector(rF, R + 0.0010, 1.32, 1.55 * reach, width, 0.0022, 8);
    proximal.translate(0, 0, z);
    sink.add(proximal, 'glove', group, 0.10);

    const distal = arcSector(rF - 0.0012, R + 0.0009, 1.32 + 1.55 * reach, 1.35 * reach, width * 0.94, 0.0020, 8);
    distal.translate(0, 0, z);
    sink.add(distal, 'glove', group, 0.16);

    // Knuckle: a small proud bead at the first joint, where gloves scuff.
    const a = 1.32 + 1.55 * reach;
    const knuckle = latheTube(width * 0.44, 0, width * 0.72, 10, width * 0.16);
    knuckle.rotateY(HALF_PI);
    knuckle.translate(Math.cos(a) * (rF + 0.0012), Math.sin(a) * (rF + 0.0012), z);
    sink.add(knuckle, 'glove', group, 0.26);
  }

  // Thumb: crosses over the top of the grip beside the index finger.
  let ang = 0.55;
  let rad = R + 0.0090;
  for (let s = 0; s < 2; s++) {
    const segLen = s === 0 ? 0.0235 : 0.0175;
    const rr = 0.0080 - s * 0.0012;
    const seg = latheProfile([
      [0, -segLen * 0.5], [rr, -segLen * 0.42], [rr * 0.90, segLen * 0.42], [0, segLen * 0.5],
    ], 10);
    seg.rotateY(HALF_PI);
    seg.rotateZ(ang);
    seg.translate(
      Math.cos(ang) * rad, Math.sin(ang) * rad,
      wristDir * (depth * 0.40 - s * segLen * 0.55),
    );
    sink.add(seg, 'glove', group, 0.14);
    ang += 0.34;
    rad += 0.0022;
  }

  if (!forearm) return;

  // Forearm: tapers away from the wrist and leaves frame. The profile is
  // authored extending toward +Z, so a -Z wrist needs it flipped.
  const arm = latheProfile([
    [0, -0.0500], [0.0245, -0.0500], [0.0225, -0.0180], [0.0200, 0.0], [0, 0.0],
  ], 14);
  if (wristDir < 0) arm.rotateY(Math.PI);
  arm.translate(R * 0.35, 0, wristDir * (depth * 0.5 + 0.0010));
  sink.add(arm, 'glove', group, 0.06);

  const cuff = latheTube(0.0250, 0.0205, 0.0090, 16, 0.0012);
  cuff.translate(R * 0.35, 0, wristDir * (depth * 0.5 + 0.0130));
  sink.add(cuff, 'glove', group, 0.26);
}

/**
 * The firing arm, placed directly in weapon space. It cannot live in the hand's
 * canonical frame: that frame's wrist axis points up the grip, and a forearm
 * extending up the grip would run straight through the receiver.
 */
function buildRightForearm(sink: PartSink, v: WeaponVisual): void {
  const gz = v.receiverZ + v.receiverLength * 0.30;
  const gy = v.receiverY - v.receiverHeight * 0.52;
  const wristY = gy - Math.cos(v.gripAngle) * v.gripLength * 0.20;
  const wristZ = gz + Math.sin(v.gripAngle) * v.gripLength * 0.20;

  const arm = latheProfile([
    [0, -0.1050], [0.0260, -0.1050], [0.0240, -0.0520], [0.0198, 0], [0, 0],
  ], 14);
  arm.rotateY(Math.PI);      // taper runs back from the wrist
  arm.rotateX(-0.62);        // back and down, the natural shooting-arm angle
  arm.rotateY(-0.13);
  arm.translate(0.0075, wristY - 0.0060, wristZ + 0.0080);
  sink.add(arm, 'glove', 'body', 0.06);

  const cuff = latheTube(0.0252, 0.0206, 0.0095, 16, 0.0012);
  cuff.rotateX(-0.62);
  cuff.rotateY(-0.13);
  cuff.translate(0.0075 + 0.0016, wristY - 0.0175, wristZ + 0.0245);
  sink.add(cuff, 'glove', 'body', 0.26);
}

// ---------------------------------------------------------------------------
// The viewmodel
// ---------------------------------------------------------------------------

export interface MuzzleFlash {
  root: THREE.Object3D;
  lobes: THREE.Mesh[];
  material: THREE.MeshBasicMaterial;
  light: THREE.PointLight;
}

export class WeaponViewmodel {
  /** Parent for the whole weapon; the animator writes this transform. */
  readonly root = new THREE.Group();
  readonly nodes: Record<PartGroup, THREE.Object3D>;
  readonly opticNode = new THREE.Object3D();

  /** Local transform of the eye point; -Z from here is the sight line. */
  readonly sightPos = new THREE.Vector3();
  readonly muzzleNode = new THREE.Object3D();
  readonly ejectNode = new THREE.Object3D();

  readonly reticleNode: THREE.Object3D | null;
  readonly reticleMaterial: THREE.MeshBasicMaterial | null;
  /** Where the reticle sits with the eye dead on axis; parallax offsets from it. */
  readonly reticleRest = new THREE.Vector3();
  readonly flash: MuzzleFlash;
  readonly scopeRadius: number;
  readonly usesScopeOverlay: boolean;

  /** Rest transforms so the animator can offset from a known zero. */
  readonly magRest = new THREE.Vector3();
  readonly chargeRest = new THREE.Vector3();
  readonly boltRest = new THREE.Vector3();
  readonly lhandRest = new THREE.Vector3();

  readonly spec: WeaponSpec;
  readonly visual: WeaponVisual;

  private readonly ownedGeometries: THREE.BufferGeometry[] = [];
  private readonly ownedMaterials: THREE.Material[] = [];

  constructor(spec: WeaponSpec, visual: WeaponVisual, materials: GunMaterials) {
    this.spec = spec;
    this.visual = visual;
    this.root.name = `viewmodel:${spec.id}`;

    const nodes: Record<PartGroup, THREE.Object3D> = {
      body: new THREE.Object3D(),
      mag: new THREE.Object3D(),
      bolt: new THREE.Object3D(),
      charge: new THREE.Object3D(),
      trigger: new THREE.Object3D(),
      dust: new THREE.Object3D(),
      lhand: new THREE.Object3D(),
      rhand: new THREE.Object3D(),
      selector: new THREE.Object3D(),
    };
    this.nodes = nodes;
    for (const key of Object.keys(nodes) as PartGroup[]) {
      nodes[key].name = key;
      this.root.add(nodes[key]);
    }
    this.root.add(this.opticNode, this.muzzleNode, this.ejectNode);

    const sink = new PartSink();
    const isPistol = visual.charging === 'slide';
    const barrelZ0 = visual.receiverZ - visual.receiverLength * 0.5;

    let railY: number;
    let muzzleZ: number;
    let muzzleY = 0;

    if (isPistol) {
      muzzleY = buildSlide(sink, visual);
      railY = visual.receiverY + visual.receiverHeight * 1.05;
      muzzleZ = visual.receiverZ - visual.receiverLength * 0.5 - 0.009;
    } else if (visual.upperTube) {
      railY = buildUpper(sink, visual);
      muzzleZ = buildBarrel(sink, visual, barrelZ0);
      buildHandguard(sink, visual, barrelZ0);
      muzzleZ = buildMuzzle(sink, visual, muzzleZ) - 0.001;
    } else {
      const len = visual.receiverLength;
      const upper = bevelBox(visual.receiverWidth, visual.receiverHeight * 0.56, len, 0.0032);
      upper.translate(0, visual.receiverY + visual.receiverHeight * 0.22, visual.receiverZ);
      sink.add(upper, 'metal', 'body', 0.09);
      const ry = visual.receiverY + visual.receiverHeight * 0.5;
      buildRail(sink, len * 0.90, ry, visual.receiverZ, Math.max(5, Math.round(len * 40)));
      railY = ry + RAIL_HEIGHT;
      muzzleZ = buildBarrel(sink, visual, barrelZ0);
      buildHandguard(sink, visual, barrelZ0);
      buildTubeMag(sink, visual, barrelZ0);
      muzzleZ = buildMuzzle(sink, visual, muzzleZ) - 0.001;
    }

    buildLower(sink, visual);
    if (!isPistol) {
      buildReceiverRear(sink, visual);
      buildEjectionPort(sink, visual, nodes.dust);
      buildBolt(sink, visual);
    }
    buildCharging(sink, visual, nodes.charge);
    if (visual.charging === 'pump') buildPump(sink, visual, barrelZ0, nodes.charge);
    buildSelector(sink, visual, nodes.selector);
    buildGrip(sink, visual);
    buildTriggerGuard(sink, visual, nodes.trigger);
    buildStock(sink, visual);
    buildMagazine(sink, visual, nodes.mag);

    const optic = buildOptic(sink, visual, spec, materials, this.opticNode, railY, muzzleZ);
    this.sightPos.copy(optic.sightPos);
    this.reticleNode = optic.reticleNode;
    this.reticleMaterial = optic.reticleMaterial;
    if (optic.reticleNode) this.reticleRest.copy(optic.reticleNode.position);
    this.scopeRadius = optic.scopeRadius;
    this.usesScopeOverlay = spec.optic === 'sniper';

    if (visual.hands) {
      // Right hand on the pistol grip: canonical +Z maps to "down the grip",
      // so a rotation of (90 - gripAngle) about X puts it on the backstrap.
      const gy = visual.receiverY - visual.receiverHeight * 0.52;
      const gz = visual.receiverZ + visual.receiverLength * 0.30;
      buildWrapHand(sink, 'rhand', 0.0135, 0.062, -1, false);
      buildRightForearm(sink, visual);
      nodes.rhand.position.set(
        0,
        gy - Math.cos(visual.gripAngle) * visual.gripLength * 0.46,
        gz + Math.sin(visual.gripAngle) * visual.gripLength * 0.46,
      );
      nodes.rhand.rotation.set(HALF_PI - visual.gripAngle, 0, 0);

      if (isPistol) {
        // Support hand cups the firing hand rather than gripping anything.
        buildWrapHand(sink, 'lhand', 0.0210, 0.052, -1, true);
        nodes.lhand.position.set(
          -0.0125,
          gy - Math.cos(visual.gripAngle) * visual.gripLength * 0.52,
          gz + Math.sin(visual.gripAngle) * visual.gripLength * 0.52 - 0.0035,
        );
        nodes.lhand.rotation.set(HALF_PI - visual.gripAngle, 0, Math.PI * 0.86);
      } else {
        const hgR = Math.max(0.011, visual.handguardRadius);
        const hgZ = barrelZ0 - visual.handguardLength * 0.52;
        buildWrapHand(sink, 'lhand', hgR, 0.066, 1, true);
        nodes.lhand.position.set(0, 0, hgZ);
        // Palm rolled under and slightly outboard: a C-clamp support grip.
        nodes.lhand.rotation.set(0, 0, -Math.PI * 0.72);
      }
    }

    const matSet = materials.forWeapon(visual);
    const resolve = (k: MatKey): THREE.Material => {
      switch (k) {
        case 'metal': return matSet.metal;
        case 'polymer': return matSet.polymer;
        case 'accent': return matSet.accent;
        case 'glove': return matSet.glove;
      }
    };
    for (const m of sink.build(nodes, resolve)) this.ownedGeometries.push(m.geometry);

    this.muzzleNode.position.set(0, muzzleY, muzzleZ);
    this.ejectNode.position.set(
      visual.ejectSide * (visual.receiverWidth * 0.5 + 0.010),
      isPistol ? visual.receiverY + visual.receiverHeight * 0.65 : visual.receiverY + visual.receiverHeight * 0.13,
      visual.receiverZ - visual.receiverLength * 0.16,
    );

    this.flash = this.buildFlash(materials);
    this.muzzleNode.add(this.flash.root);

    this.magRest.copy(nodes.mag.position);
    this.chargeRest.copy(nodes.charge.position);
    this.boltRest.copy(nodes.bolt.position);
    this.lhandRest.copy(nodes.lhand.position);

    this.root.traverse((o) => {
      o.layers.enable(LAYERS.viewmodel);
      o.frustumCulled = false;
    });
    this.root.visible = false;
  }

  /**
   * Three overlapping additive lobes with an independent random roll per shot.
   * A real flash is asymmetric and gone inside two frames, so this never settles
   * into the tidy symmetric star that gives a WebGL demo away.
   */
  private buildFlash(materials: GunMaterials): MuzzleFlash {
    const root = new THREE.Object3D();
    root.visible = false;

    const material = new THREE.MeshBasicMaterial({
      map: materials.flashTexture,
      color: 0xffd9a0,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
      side: THREE.DoubleSide,
    });
    this.ownedMaterials.push(material);

    const lobes: THREE.Mesh[] = [];
    const scales = [1.0, 0.66, 0.42];
    for (let i = 0; i < 3; i++) {
      const geo = new THREE.PlaneGeometry(1, 1);
      this.ownedGeometries.push(geo);
      const m = new THREE.Mesh(geo, material);
      m.scale.setScalar(scales[i]! * 0.06);
      m.position.z = -i * 0.006;
      m.renderOrder = 14;
      m.frustumCulled = false;
      root.add(m);
      lobes.push(m);
    }

    const light = new THREE.PointLight(0xffc27a, 0, 4, 2);
    light.castShadow = false;
    root.add(light);

    return { root, lobes, material, light };
  }

  setVisible(v: boolean): void {
    this.root.visible = v;
  }

  dispose(): void {
    this.root.removeFromParent();
    for (const g of this.ownedGeometries) g.dispose();
    this.ownedGeometries.length = 0;
    for (const m of this.ownedMaterials) m.dispose();
    this.ownedMaterials.length = 0;
    this.reticleMaterial?.dispose();
    this.opticNode.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) mesh.geometry.dispose();
    });
  }
}

/**
 * The black surround that occludes the world through a magnified optic. Lives
 * in camera space rather than weapon space so it stays locked to the screen,
 * and draws with depth testing off so nothing pokes through it.
 */
export function buildScopeOverlay(materials: GunMaterials, innerRadius: number): {
  root: THREE.Object3D;
  materials: THREE.Material[];
  geometries: THREE.BufferGeometry[];
} {
  const root = new THREE.Object3D();
  root.visible = false;

  const ringGeo = new THREE.RingGeometry(innerRadius, innerRadius * 26, 72, 1);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x000000, toneMapped: false, depthTest: false, depthWrite: false,
  });
  const black = new THREE.Mesh(ringGeo, ringMat);
  black.renderOrder = 20;
  black.frustumCulled = false;
  root.add(black);

  const shadowGeo = new THREE.PlaneGeometry(innerRadius * 2.04, innerRadius * 2.04);
  const shadowMat = new THREE.MeshBasicMaterial({
    map: materials.scopeShadow,
    transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
  });
  const shadow = new THREE.Mesh(shadowGeo, shadowMat);
  shadow.renderOrder = 21;
  shadow.frustumCulled = false;
  root.add(shadow);

  root.traverse((o) => o.layers.enable(LAYERS.viewmodel));
  return { root, materials: [ringMat, shadowMat], geometries: [ringGeo, shadowGeo] };
}
