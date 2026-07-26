/**
 * Procedural soldier: geometry, materials and the instanced renderer that draws
 * every enemy in the level in ten draw calls regardless of how many there are.
 *
 * The rig is a plain Object3D hierarchy per agent — no skinning. Each named
 * joint owns exactly one body part whose geometry is authored in that joint's
 * local space, so posing is "rotate the joint" and rendering is "copy the
 * joint's world matrix into an InstancedMesh slot". That buys articulated,
 * shadow-casting, hitbox-accurate characters at instancing cost, which is the
 * only way a browser affords a dozen of them alongside a full post stack.
 *
 * Everything here is generated at runtime: camo, weave normals, kit nylon,
 * skin, gunmetal. No files.
 */
import * as THREE from 'three';
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { QualitySettings } from '../core/Contracts';

// ---------------------------------------------------------------------------
// Deterministic noise helpers
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

function hash2(x: number, y: number, seed: number): number {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(seed, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Tiling value noise: wraps on `period` so the texture is seamless. */
function valueNoise(x: number, y: number, period: number, seed: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = smooth(x - x0);
  const ty = smooth(y - y0);
  const wx0 = ((x0 % period) + period) % period;
  const wy0 = ((y0 % period) + period) % period;
  const wx1 = (wx0 + 1) % period;
  const wy1 = (wy0 + 1) % period;
  const a = hash2(wx0, wy0, seed);
  const b = hash2(wx1, wy0, seed);
  const c = hash2(wx0, wy1, seed);
  const d = hash2(wx1, wy1, seed);
  return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
}

function fbm(x: number, y: number, basePeriod: number, octaves: number, seed: number): number {
  let sum = 0;
  let amp = 0.5;
  let norm = 0;
  let freq = 1;
  for (let o = 0; o < octaves; o++) {
    sum += valueNoise(x * freq, y * freq, basePeriod * freq, seed + o * 7919) * amp;
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

// ---------------------------------------------------------------------------
// Texture synthesis
// ---------------------------------------------------------------------------

function makeCanvas(size: number): { c: HTMLCanvasElement; d: ImageData } {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  if (!ctx) throw new Error('2d context unavailable');
  return { c, d: ctx.createImageData(size, size) };
}

function commit(c: HTMLCanvasElement, d: ImageData): THREE.CanvasTexture {
  const ctx = c.getContext('2d');
  if (ctx) ctx.putImageData(d, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.RepeatWrapping;
  t.needsUpdate = true;
  return t;
}

/** Sobel a height field into a tangent-space normal map. */
function heightToNormal(height: Float32Array, size: number, strength: number): THREE.CanvasTexture {
  const { c, d } = makeCanvas(size);
  const px = d.data;
  const at = (x: number, y: number): number => height[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const len = Math.hypot(dx, dy, 1);
      const i = (y * size + x) * 4;
      px[i] = Math.round(((-dx / len) * 0.5 + 0.5) * 255);
      px[i + 1] = Math.round(((-dy / len) * 0.5 + 0.5) * 255);
      px[i + 2] = Math.round((1 / len) * 0.5 * 255 + 127.5);
      px[i + 3] = 255;
    }
  }
  return commit(c, d);
}

export interface SoldierTextures {
  camoMap: THREE.Texture;
  camoNormal: THREE.Texture;
  camoRough: THREE.Texture;
  gearMap: THREE.Texture;
  gearNormal: THREE.Texture;
  gearRough: THREE.Texture;
  skinMap: THREE.Texture;
  skinNormal: THREE.Texture;
  metalMap: THREE.Texture;
  metalNormal: THREE.Texture;
  metalRough: THREE.Texture;
  dispose(): void;
}

/**
 * Four-tone multicam-ish pattern. Real camo is built from overlapping blobs at
 * three scales with hard edges; a single noise threshold reads as marble, which
 * is the give-away in every WebGL soldier ever shipped.
 */
function buildCamo(size: number): { map: THREE.CanvasTexture; normal: THREE.CanvasTexture; rough: THREE.CanvasTexture } {
  const { c, d } = makeCanvas(size);
  const px = d.data;
  const height = new Float32Array(size * size);
  const { c: rc, d: rd } = makeCanvas(size);
  const rpx = rd.data;

  // Field-drab palette, linear-ish sRGB bytes.
  const tones = [
    [78, 76, 58],
    [104, 97, 68],
    [58, 58, 46],
    [126, 116, 88],
    [44, 45, 38],
  ];

  const P = 8; // noise lattice period in the 0..1 UV domain

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x / size) * P;
      const v = (y / size) * P;

      const big = fbm(u * 0.55, v * 0.55, P, 3, 101);
      const mid = fbm(u * 1.6 + 3.1, v * 1.6 - 1.7, P * 2, 3, 733);
      const fine = fbm(u * 4.2 - 5.5, v * 4.2 + 2.2, P * 4, 2, 1571);

      // Hard-edged blob selection. The `+ fine*0.16` warps the boundaries so the
      // three scales interlock instead of stacking concentrically.
      const s = big * 0.62 + mid * 0.3 + fine * 0.16;
      let idx: number;
      if (s < 0.36) idx = 2;
      else if (s < 0.47) idx = 0;
      else if (s < 0.585) idx = 1;
      else idx = 3;
      // Sparse dark speckle clusters, the way printed camo breaks up flats.
      if (fine > 0.74 && mid > 0.5) idx = 4;

      const tone = tones[idx];

      // Twill weave: a 2-over-2 diagonal rib. This is what makes cloth read as
      // cloth at 40cm from the camera under a hard sun.
      const weave = ((x + y * 2) % 6) < 3 ? 1 : 0;
      const warp = (x % 2) === 0 ? 0.5 : 0;
      const thread = weave * 0.55 + warp * 0.45;
      const lint = valueNoise(u * 34, v * 34, P * 34, 4111);

      const wear = Math.max(0, fbm(u * 0.8 + 11, v * 0.8 - 4, P, 3, 9001) - 0.5) * 2;
      const shade = 0.86 + thread * 0.15 + lint * 0.09 + wear * 0.16;

      const i = (y * size + x) * 4;
      px[i] = Math.min(255, tone[0] * shade);
      px[i + 1] = Math.min(255, tone[1] * shade);
      px[i + 2] = Math.min(255, tone[2] * shade);
      px[i + 3] = 255;

      height[y * size + x] = thread * 0.6 + lint * 0.4;

      // Worn threads polish: high spots get slightly less rough.
      const rough = 0.94 - thread * 0.08 - wear * 0.13;
      rpx[i] = 255;
      rpx[i + 1] = Math.round(THREE.MathUtils.clamp(rough, 0, 1) * 255); // green = roughness
      rpx[i + 2] = 0;
      rpx[i + 3] = 255;
    }
  }

  return { map: commit(c, d), normal: heightToNormal(height, size, 2.6), rough: commit(rc, rd) };
}

/** Nylon webbing / cordura for the plate carrier, helmet cover and pack. */
function buildGear(size: number): { map: THREE.CanvasTexture; normal: THREE.CanvasTexture; rough: THREE.CanvasTexture } {
  const { c, d } = makeCanvas(size);
  const px = d.data;
  const height = new Float32Array(size * size);
  const { c: rc, d: rd } = makeCanvas(size);
  const rpx = rd.data;
  const P = 8;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x / size) * P;
      const v = (y / size) * P;

      // Cordura is a coarse basket weave: chunky ribs both ways, not a twill.
      const rx = Math.abs(((x % 8) / 8) * 2 - 1);
      const ry = Math.abs(((y % 8) / 8) * 2 - 1);
      const rib = 1 - Math.min(rx, ry) * 0.85;
      const slub = valueNoise(u * 26, v * 26, P * 26, 5501);
      const grime = fbm(u * 1.1, v * 1.1, P, 4, 2027);
      const scuff = Math.max(0, fbm(u * 3.3 - 7, v * 3.3 + 9, P * 3, 3, 6421) - 0.58) * 2.4;

      const base = 42 + grime * 22;
      const shade = 0.8 + rib * 0.26 + slub * 0.1;
      const i = (y * size + x) * 4;
      px[i] = Math.min(255, (base + 6) * shade + scuff * 26);
      px[i + 1] = Math.min(255, (base + 3) * shade + scuff * 23);
      px[i + 2] = Math.min(255, base * 0.86 * shade + scuff * 18);
      px[i + 3] = 255;

      height[y * size + x] = rib * 0.75 + slub * 0.25;

      rpx[i] = 255;
      rpx[i + 1] = Math.round(THREE.MathUtils.clamp(0.9 - scuff * 0.3 - rib * 0.06, 0.35, 1) * 255);
      rpx[i + 2] = 0;
      rpx[i + 3] = 255;
    }
  }
  return { map: commit(c, d), normal: heightToNormal(height, size, 3.4), rough: commit(rc, rd) };
}

/** Skin: pore noise plus a little blotch variation. Hands and face only. */
function buildSkin(size: number): { map: THREE.CanvasTexture; normal: THREE.CanvasTexture } {
  const { c, d } = makeCanvas(size);
  const px = d.data;
  const height = new Float32Array(size * size);
  const P = 8;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x / size) * P;
      const v = (y / size) * P;
      const pore = valueNoise(u * 44, v * 44, P * 44, 8123);
      const blotch = fbm(u * 2.4, v * 2.4, P * 2, 3, 3331);
      const stubble = Math.max(0, valueNoise(u * 60, v * 60, P * 60, 991) - 0.62) * 2.6;
      const shade = 0.9 + pore * 0.12 + blotch * 0.12 - stubble * 0.22;
      const i = (y * size + x) * 4;
      px[i] = Math.min(255, 168 * shade);
      px[i + 1] = Math.min(255, 126 * shade);
      px[i + 2] = Math.min(255, 104 * shade);
      px[i + 3] = 255;
      height[y * size + x] = pore * 0.8 + stubble * 0.6;
    }
  }
  return { map: commit(c, d), normal: heightToNormal(height, size, 1.5) };
}

/** Phosphate-finish gunmetal with machining marks and edge wear. */
function buildMetal(size: number): { map: THREE.CanvasTexture; normal: THREE.CanvasTexture; rough: THREE.CanvasTexture } {
  const { c, d } = makeCanvas(size);
  const px = d.data;
  const height = new Float32Array(size * size);
  const { c: rc, d: rd } = makeCanvas(size);
  const rpx = rd.data;
  const P = 8;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = (x / size) * P;
      const v = (y / size) * P;
      // Anisotropic tool marks run along the barrel axis (U).
      const machining = valueNoise(u * 3, v * 90, P * 90, 3701);
      const grain = fbm(u * 8, v * 8, P * 8, 3, 4801);
      const rub = Math.max(0, fbm(u * 2.2 + 13, v * 2.2 - 6, P * 2, 3, 7717) - 0.62) * 2.6;

      const base = 34 + grain * 16 + machining * 6;
      const i = (y * size + x) * 4;
      // Rubbed-through phosphate exposes bright steel.
      px[i] = Math.min(255, base + rub * 96);
      px[i + 1] = Math.min(255, base + rub * 94);
      px[i + 2] = Math.min(255, base * 1.03 + rub * 92);
      px[i + 3] = 255;

      height[y * size + x] = grain * 0.55 + machining * 0.45;

      rpx[i] = 255;
      rpx[i + 1] = Math.round(THREE.MathUtils.clamp(0.62 - rub * 0.36 + grain * 0.12, 0.12, 0.95) * 255);
      rpx[i + 2] = Math.round(THREE.MathUtils.clamp(0.55 + rub * 0.45, 0, 1) * 255); // blue = metalness
      rpx[i + 3] = 255;
    }
  }
  return { map: commit(c, d), normal: heightToNormal(height, size, 1.9), rough: commit(rc, rd) };
}

export function buildSoldierTextures(quality: QualitySettings): SoldierTextures {
  const size = quality.preset === 'low' ? 128 : quality.preset === 'medium' ? 256 : 512;
  const camo = buildCamo(size);
  const gear = buildGear(size);
  const skin = buildSkin(Math.max(128, size >> 1));
  const metal = buildMetal(Math.max(128, size >> 1));

  const all: THREE.Texture[] = [
    camo.map, camo.normal, camo.rough,
    gear.map, gear.normal, gear.rough,
    skin.map, skin.normal,
    metal.map, metal.normal, metal.rough,
  ];
  for (const t of all) {
    t.anisotropy = quality.anisotropy;
    t.generateMipmaps = true;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.magFilter = THREE.LinearFilter;
  }
  camo.map.colorSpace = THREE.SRGBColorSpace;
  gear.map.colorSpace = THREE.SRGBColorSpace;
  skin.map.colorSpace = THREE.SRGBColorSpace;
  metal.map.colorSpace = THREE.SRGBColorSpace;

  return {
    camoMap: camo.map, camoNormal: camo.normal, camoRough: camo.rough,
    gearMap: gear.map, gearNormal: gear.normal, gearRough: gear.rough,
    skinMap: skin.map, skinNormal: skin.normal,
    metalMap: metal.map, metalNormal: metal.normal, metalRough: metal.rough,
    dispose(): void {
      for (const t of all) t.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

const _vtx = new THREE.Vector3();
const _inner = new THREE.Vector3();

/**
 * Superellipsoid-clamped box: soft silhouette edges without a subdivision
 * surface. `taperTop`/`taperBottom` scale X and Z at the extremes so a torso
 * can narrow at the waist and a thigh at the knee.
 */
function roundedBox(
  w: number, h: number, d: number, radius: number, seg = 3,
  taperTop = 1, taperBottom = 1, bulge = 0,
): THREE.BufferGeometry {
  let g: THREE.BufferGeometry = new THREE.BoxGeometry(w, h, d, seg, seg + 1, seg);
  g = mergeVertices(g, 1e-4);
  const pos = g.attributes.position as THREE.BufferAttribute;
  const hx = w * 0.5;
  const hy = h * 0.5;
  const hz = d * 0.5;
  const r = Math.min(radius, hx * 0.98, hy * 0.98, hz * 0.98);
  const ix = Math.max(1e-5, hx - r);
  const iy = Math.max(1e-5, hy - r);
  const iz = Math.max(1e-5, hz - r);

  for (let i = 0; i < pos.count; i++) {
    _vtx.fromBufferAttribute(pos, i);
    _inner.set(
      THREE.MathUtils.clamp(_vtx.x, -ix, ix),
      THREE.MathUtils.clamp(_vtx.y, -iy, iy),
      THREE.MathUtils.clamp(_vtx.z, -iz, iz),
    );
    _vtx.sub(_inner);
    if (_vtx.lengthSq() > 1e-12) _vtx.setLength(r);
    _vtx.add(_inner);

    // Taper along Y, then a barrel bulge in the middle.
    const t = (_vtx.y / hy) * 0.5 + 0.5;
    const s = taperBottom + (taperTop - taperBottom) * t;
    const b = 1 + bulge * Math.sin(t * Math.PI);
    pos.setXYZ(i, _vtx.x * s * b, _vtx.y, _vtx.z * s * b);
  }
  pos.needsUpdate = true;
  g.computeVertexNormals();
  return g;
}

function limb(length: number, rTop: number, rBottom: number, seg = 8): THREE.BufferGeometry {
  // Authored hanging down from the joint at the origin: y in [-length, 0].
  const g = new THREE.CylinderGeometry(rTop, rBottom, length, seg, 2, false);
  g.translate(0, -length * 0.5, 0);
  return g;
}

function transformed(g: THREE.BufferGeometry, m: THREE.Matrix4): THREE.BufferGeometry {
  const c = g.clone();
  c.applyMatrix4(m);
  g.dispose();
  return c;
}

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3(1, 1, 1);
const _t = new THREE.Vector3();
const _e = new THREE.Euler();

function place(
  g: THREE.BufferGeometry,
  x: number, y: number, z: number,
  rx = 0, ry = 0, rz = 0,
  sx = 1, sy = 1, sz = 1,
): THREE.BufferGeometry {
  _e.set(rx, ry, rz);
  _q.setFromEuler(_e);
  _t.set(x, y, z);
  _s.set(sx, sy, sz);
  _m.compose(_t, _q, _s);
  return transformed(g, _m);
}

function join(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  if (!merged) throw new Error('soldier geometry merge failed');
  merged.computeVertexNormals();
  return merged;
}

// ---------------------------------------------------------------------------
// Body part geometry
// ---------------------------------------------------------------------------

export const PART = {
  pelvis: 0,
  torso: 1,
  head: 2,
  helmet: 3,
  upperArm: 4,
  foreArm: 5,
  thigh: 6,
  shin: 7,
  rifle: 8,
  pack: 9,
} as const;

export type PartId = (typeof PART)[keyof typeof PART];

/** Parts that exist twice per soldier (left/right). */
const PAIRED = new Set<number>([PART.upperArm, PART.foreArm, PART.thigh, PART.shin]);

export const PART_COUNT = 10;

type MatKind = 'camo' | 'gear' | 'skin' | 'metal';

const PART_MATERIAL: Record<number, MatKind> = {
  [PART.pelvis]: 'camo',
  [PART.torso]: 'gear',
  [PART.head]: 'skin',
  [PART.helmet]: 'gear',
  [PART.upperArm]: 'camo',
  [PART.foreArm]: 'camo',
  [PART.thigh]: 'camo',
  [PART.shin]: 'camo',
  [PART.rifle]: 'metal',
  [PART.pack]: 'gear',
};

function buildPelvis(): THREE.BufferGeometry {
  const hips = roundedBox(0.30, 0.20, 0.21, 0.075, 3, 1.02, 0.94);
  const belt = place(roundedBox(0.315, 0.055, 0.225, 0.024, 2), 0, 0.085, 0);
  const pouchL = place(roundedBox(0.085, 0.10, 0.06, 0.02, 2), -0.135, 0.02, 0.06, 0, 0.25, 0);
  const pouchR = place(roundedBox(0.085, 0.10, 0.06, 0.02, 2), 0.135, 0.02, 0.06, 0, -0.25, 0);
  const dumpBag = place(roundedBox(0.11, 0.12, 0.07, 0.03, 2), 0.11, -0.02, -0.10);
  return join([hips, belt, pouchL, pouchR, dumpBag]);
}

/** Torso includes the plate carrier; the two never separate so they merge. */
function buildTorso(): THREE.BufferGeometry {
  const chest = roundedBox(0.36, 0.50, 0.235, 0.085, 4, 1.06, 0.86, 0.03);
  chest.translate(0, 0.25, 0);
  const plateFront = place(roundedBox(0.30, 0.34, 0.05, 0.028, 3, 0.92, 1.0), 0, 0.30, 0.115);
  const plateBack = place(roundedBox(0.30, 0.36, 0.045, 0.028, 3, 0.92, 1.0), 0, 0.30, -0.115);
  const cummerbundL = place(roundedBox(0.05, 0.20, 0.20, 0.022, 2), -0.165, 0.22, 0);
  const cummerbundR = place(roundedBox(0.05, 0.20, 0.20, 0.022, 2), 0.165, 0.22, 0);
  // Triple mag stack, slightly fanned — nothing on a real rig sits square.
  const mag1 = place(roundedBox(0.08, 0.16, 0.05, 0.014, 2), -0.085, 0.235, 0.155, 0.07, 0.04, 0.03);
  const mag2 = place(roundedBox(0.08, 0.16, 0.05, 0.014, 2), 0.0, 0.225, 0.16, 0.05, 0, -0.02);
  const mag3 = place(roundedBox(0.08, 0.16, 0.05, 0.014, 2), 0.085, 0.24, 0.153, 0.09, -0.05, 0.04);
  const radio = place(roundedBox(0.07, 0.12, 0.05, 0.016, 2), 0.13, 0.40, 0.10, 0, -0.2, 0.1);
  const antenna = place(new THREE.CylinderGeometry(0.005, 0.0035, 0.26, 5, 1), 0.145, 0.55, 0.09, 0.18, 0, 0.12);
  const collar = place(roundedBox(0.20, 0.075, 0.17, 0.03, 2), 0, 0.505, 0);
  const shoulderL = place(roundedBox(0.11, 0.10, 0.15, 0.04, 2), -0.175, 0.455, 0, 0, 0, 0.28);
  const shoulderR = place(roundedBox(0.11, 0.10, 0.15, 0.04, 2), 0.175, 0.455, 0, 0, 0, -0.28);
  return join([
    chest, plateFront, plateBack, cummerbundL, cummerbundR,
    mag1, mag2, mag3, radio, antenna, collar, shoulderL, shoulderR,
  ]);
}

function buildHead(): THREE.BufferGeometry {
  const skull = new THREE.SphereGeometry(0.098, 14, 12);
  skull.scale(0.94, 1.12, 1.02);
  const jaw = place(roundedBox(0.145, 0.09, 0.15, 0.045, 2, 0.86, 1.0), 0, -0.062, 0.012);
  const nose = place(roundedBox(0.032, 0.05, 0.038, 0.014, 1), 0, -0.018, 0.098);
  const neck = place(new THREE.CylinderGeometry(0.048, 0.058, 0.10, 10, 1), 0, -0.115, -0.005);
  return join([skull, jaw, nose, neck]);
}

function buildHelmet(): THREE.BufferGeometry {
  const shell = new THREE.SphereGeometry(0.122, 16, 12, 0, Math.PI * 2, 0, Math.PI * 0.62);
  shell.scale(1.0, 1.08, 1.06);
  const rim = place(new THREE.TorusGeometry(0.118, 0.011, 6, 20), 0, -0.017, 0.004, Math.PI * 0.5, 0, 0);
  const earL = place(roundedBox(0.035, 0.10, 0.10, 0.03, 2), -0.118, -0.045, 0.0);
  const earR = place(roundedBox(0.035, 0.10, 0.10, 0.03, 2), 0.118, -0.045, 0.0);
  // NVG shroud + mount stub: the silhouette detail that reads "modern soldier".
  const shroud = place(roundedBox(0.05, 0.03, 0.055, 0.01, 1), 0, 0.048, 0.098, -0.4, 0, 0);
  const rail = place(roundedBox(0.012, 0.02, 0.16, 0.005, 1), -0.108, 0.028, -0.01, 0, 0, 0.25);
  const rail2 = place(roundedBox(0.012, 0.02, 0.16, 0.005, 1), 0.108, 0.028, -0.01, 0, 0, -0.25);
  const strap = place(new THREE.TorusGeometry(0.098, 0.006, 5, 14), 0, -0.06, 0.02, Math.PI * 0.42, 0, 0);
  return join([shell, rim, earL, earR, shroud, rail, rail2, strap]);
}

function buildUpperArm(): THREE.BufferGeometry {
  const arm = limb(0.27, 0.056, 0.046, 9);
  const pad = place(roundedBox(0.10, 0.11, 0.10, 0.04, 2), 0, -0.02, 0);
  return join([arm, pad]);
}

function buildForeArm(): THREE.BufferGeometry {
  const arm = limb(0.245, 0.046, 0.036, 9);
  const glove = place(roundedBox(0.062, 0.10, 0.045, 0.022, 2, 0.85, 1.0), 0, -0.285, 0.006);
  const cuff = place(new THREE.CylinderGeometry(0.048, 0.044, 0.05, 9, 1), 0, -0.225, 0);
  return join([arm, cuff, glove]);
}

function buildThigh(): THREE.BufferGeometry {
  const leg = limb(0.42, 0.085, 0.062, 10);
  const kneePad = place(roundedBox(0.10, 0.13, 0.075, 0.032, 2), 0, -0.37, 0.035);
  const cargo = place(roundedBox(0.075, 0.13, 0.05, 0.02, 2), -0.078, -0.20, 0.02, 0, 0, 0.05);
  return join([leg, kneePad, cargo]);
}

function buildShin(): THREE.BufferGeometry {
  const leg = limb(0.42, 0.062, 0.048, 9);
  const boot = place(roundedBox(0.095, 0.115, 0.135, 0.035, 2), 0, -0.455, 0.012);
  const toe = place(roundedBox(0.088, 0.055, 0.10, 0.026, 2), 0, -0.492, 0.086);
  const sole = place(roundedBox(0.10, 0.026, 0.235, 0.012, 1), 0, -0.516, 0.038);
  return join([leg, boot, toe, sole]);
}

function buildPack(): THREE.BufferGeometry {
  const body = roundedBox(0.28, 0.34, 0.16, 0.055, 3, 0.94, 1.0);
  const lid = place(roundedBox(0.27, 0.10, 0.15, 0.04, 2), 0, 0.19, -0.005);
  const strapL = place(roundedBox(0.035, 0.30, 0.02, 0.008, 1), -0.10, 0.02, 0.09);
  const strapR = place(roundedBox(0.035, 0.30, 0.02, 0.008, 1), 0.10, 0.02, 0.09);
  const roll = place(new THREE.CylinderGeometry(0.045, 0.045, 0.26, 9, 1), 0, -0.20, -0.02, 0, 0, Math.PI * 0.5);
  return join([body, lid, strapL, strapR, roll]);
}

/**
 * Enemy carbine. Authored in the right-hand joint space: barrel down -Z,
 * grip at the origin. It only ever appears at 6–40m so the detail budget goes
 * on silhouette (mag curve, stock skeleton, optic) rather than screws.
 */
function buildRifle(): THREE.BufferGeometry {
  const receiver = place(roundedBox(0.052, 0.085, 0.34, 0.016, 2), 0, 0.015, -0.06);
  const upper = place(roundedBox(0.05, 0.045, 0.30, 0.014, 2), 0, 0.062, -0.09);
  const rail = place(roundedBox(0.026, 0.014, 0.30, 0.004, 1), 0, 0.086, -0.09);
  const handguard = place(new THREE.CylinderGeometry(0.028, 0.026, 0.26, 8, 1), 0, 0.05, -0.35, Math.PI * 0.5, 0, 0);
  const barrel = place(new THREE.CylinderGeometry(0.0105, 0.0115, 0.30, 8, 1), 0, 0.05, -0.55, Math.PI * 0.5, 0, 0);
  const brake = place(new THREE.CylinderGeometry(0.017, 0.014, 0.06, 8, 1), 0, 0.05, -0.71, Math.PI * 0.5, 0, 0);
  const gasBlock = place(roundedBox(0.03, 0.05, 0.045, 0.008, 1), 0, 0.058, -0.46);
  const grip = place(roundedBox(0.036, 0.115, 0.052, 0.016, 2, 0.8, 1.0), 0, -0.062, 0.012, 0.22, 0, 0);
  const trigger = place(new THREE.TorusGeometry(0.024, 0.005, 5, 10), 0, -0.012, -0.03, 0, Math.PI * 0.5, 0);
  // Curved magazine: five staggered segments, each rotated a little further.
  const magParts: THREE.BufferGeometry[] = [];
  for (let i = 0; i < 5; i++) {
    const t = i / 4;
    const a = t * 0.30;
    magParts.push(place(
      roundedBox(0.030, 0.056, 0.048, 0.008, 1),
      Math.sin(a) * 0.0,
      -0.055 - Math.cos(a) * (0.045 + t * 0.052),
      -0.075 + Math.sin(a) * (0.055 + t * 0.075),
      -a, 0, 0,
    ));
  }
  const stockTube = place(new THREE.CylinderGeometry(0.017, 0.017, 0.18, 7, 1), 0, 0.018, 0.20, Math.PI * 0.5, 0, 0);
  const stockPad = place(roundedBox(0.045, 0.10, 0.038, 0.012, 2), 0, 0.008, 0.285);
  const stockRail = place(roundedBox(0.036, 0.05, 0.14, 0.01, 1), 0, -0.02, 0.20);
  const optic = place(roundedBox(0.036, 0.045, 0.09, 0.012, 2), 0, 0.115, -0.10);
  const opticGlass = place(new THREE.CylinderGeometry(0.016, 0.016, 0.012, 10, 1), 0, 0.118, -0.148, Math.PI * 0.5, 0, 0);
  const sling = place(new THREE.TorusGeometry(0.012, 0.0035, 4, 10), 0, 0.04, -0.20, 0, Math.PI * 0.5, 0);

  return join([
    receiver, upper, rail, handguard, barrel, brake, gasBlock, grip, trigger,
    ...magParts, stockTube, stockPad, stockRail, optic, opticGlass, sling,
  ]);
}

const BUILDERS: Record<number, () => THREE.BufferGeometry> = {
  [PART.pelvis]: buildPelvis,
  [PART.torso]: buildTorso,
  [PART.head]: buildHead,
  [PART.helmet]: buildHelmet,
  [PART.upperArm]: buildUpperArm,
  [PART.foreArm]: buildForeArm,
  [PART.thigh]: buildThigh,
  [PART.shin]: buildShin,
  [PART.rifle]: buildRifle,
  [PART.pack]: buildPack,
};

// ---------------------------------------------------------------------------
// Instanced renderer
// ---------------------------------------------------------------------------

const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0);

/**
 * Draws every soldier. One InstancedMesh per body part; paired parts (arms,
 * legs) get two slots per soldier. Slot index is `agent * stride + side`, which
 * keeps writes contiguous and lets a dead/despawned agent be hidden by writing
 * a zero-scale matrix without disturbing anyone else.
 */
export class SoldierRenderer {
  readonly root = new THREE.Group();
  private readonly meshes: THREE.InstancedMesh[] = [];
  private readonly strides: number[] = [];
  private readonly materials: THREE.Material[] = [];
  private readonly textures: SoldierTextures;
  private readonly capacity: number;

  constructor(capacity: number, quality: QualitySettings) {
    this.capacity = capacity;
    this.root.name = 'soldiers';
    this.textures = buildSoldierTextures(quality);

    const camo = new THREE.MeshStandardMaterial({
      map: this.textures.camoMap,
      normalMap: this.textures.camoNormal,
      roughnessMap: this.textures.camoRough,
      metalnessMap: this.textures.camoRough,
      roughness: 1,
      metalness: 0,
      normalScale: new THREE.Vector2(0.85, 0.85),
      dithering: true,
    });
    const gear = new THREE.MeshStandardMaterial({
      map: this.textures.gearMap,
      normalMap: this.textures.gearNormal,
      roughnessMap: this.textures.gearRough,
      metalnessMap: this.textures.gearRough,
      roughness: 1,
      metalness: 0,
      normalScale: new THREE.Vector2(1.0, 1.0),
      dithering: true,
    });
    const skin = new THREE.MeshStandardMaterial({
      map: this.textures.skinMap,
      normalMap: this.textures.skinNormal,
      roughness: 0.72,
      metalness: 0,
      normalScale: new THREE.Vector2(0.45, 0.45),
      dithering: true,
    });
    const metal = new THREE.MeshStandardMaterial({
      map: this.textures.metalMap,
      normalMap: this.textures.metalNormal,
      roughnessMap: this.textures.metalRough,
      metalnessMap: this.textures.metalRough,
      roughness: 1,
      metalness: 1,
      normalScale: new THREE.Vector2(0.7, 0.7),
      dithering: true,
    });
    const byKind: Record<MatKind, THREE.Material> = { camo, gear, skin, metal };
    this.materials.push(camo, gear, skin, metal);

    for (let p = 0; p < PART_COUNT; p++) {
      const stride = PAIRED.has(p) ? 2 : 1;
      const geo = BUILDERS[p]();
      const mesh = new THREE.InstancedMesh(geo, byKind[PART_MATERIAL[p]], capacity * stride);
      // Named so the HUD's scene rasteriser can exclude characters from the
      // minimap's building silhouette.
      mesh.name = `soldier-part-${p}`;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      // Instances are scattered across the whole level; a per-part bounding
      // sphere from the source geometry would cull them all on the first frame.
      mesh.frustumCulled = false;
      mesh.count = capacity * stride;
      const colors = new Float32Array(capacity * stride * 3).fill(1);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      for (let i = 0; i < capacity * stride; i++) mesh.setMatrixAt(i, HIDDEN);
      this.meshes.push(mesh);
      this.strides.push(stride);
      this.root.add(mesh);
    }
  }

  /** Per-soldier fabric tint so a squad does not look stamped from one mould. */
  setTint(agent: number, r: number, g: number, b: number): void {
    for (let p = 0; p < PART_COUNT; p++) {
      const stride = this.strides[p];
      const col = this.meshes[p].instanceColor;
      if (!col) continue;
      for (let s = 0; s < stride; s++) {
        col.setXYZ(agent * stride + s, r, g, b);
      }
      col.needsUpdate = true;
    }
  }

  write(part: PartId, agent: number, side: number, m: THREE.Matrix4): void {
    if (agent < 0 || agent >= this.capacity) return;
    const stride = this.strides[part];
    this.meshes[part].setMatrixAt(agent * stride + Math.min(side, stride - 1), m);
  }

  hide(agent: number): void {
    for (let p = 0; p < PART_COUNT; p++) {
      const stride = this.strides[p];
      for (let s = 0; s < stride; s++) this.meshes[p].setMatrixAt(agent * stride + s, HIDDEN);
    }
  }

  flush(): void {
    for (const m of this.meshes) m.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    for (const m of this.meshes) {
      m.geometry.dispose();
      this.root.remove(m);
      m.dispose();
    }
    this.meshes.length = 0;
    for (const mat of this.materials) mat.dispose();
    this.materials.length = 0;
    this.textures.dispose();
    this.root.parent?.remove(this.root);
  }
}
