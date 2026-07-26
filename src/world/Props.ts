/**
 * Instanced props, scatter and foliage.
 *
 * Everything repeated in the level lives here. Three things keep it from
 * reading as copy-paste:
 *
 *  - Every instance gets its own scale (0.88-1.14), yaw, a few degrees of tilt
 *    and a per-instance colour multiplier, so no two barrels are the same
 *    barrel.
 *  - Scatter uses a Poisson-disc so debris has the irregular-but-never-clumped
 *    spacing of real detritus, not the visible clusters of pure rejection
 *    sampling.
 *  - Foliage is alpha-tested cards with a two-frequency wind vertex shader,
 *    phase-offset by world position, so a field of weeds does not breathe in
 *    unison.
 *
 * Materials come from the shared MaterialLibrary, but always through
 * `MaterialVault`, which hands back a private clone with `vertexColors`
 * enabled. Mutating a library material in place would poison it for every
 * other subsystem that asked for the same look.
 */
import * as THREE from 'three';
import type { QualitySettings, SurfaceKind } from '../core/Contracts';
import { MaterialLibrary, triplanarMaterial, type SurfaceLook } from './Materials';
import {
  Rng,
  bevelBox,
  plainBox,
  cylinderGeo,
  torusGeo,
  extrudeProfile,
  jerseyProfile,
  finalizeGeometry,
  mergeAll,
  scaleUv,
  tintGeometry,
  rubbleCone,
  rebarGeo,
  type BoxSpec,
  type ColliderSet,
} from './Geometry';

// ---------------------------------------------------------------------------
// Material vault
// ---------------------------------------------------------------------------

export interface VaultOpts {
  color?: THREE.ColorRepresentation;
  seed?: number;
  roughness?: number;
  metalness?: number;
  normalScale?: number;
  emissive?: THREE.ColorRepresentation;
  emissiveIntensity?: number;
  side?: THREE.Side;
  envMapIntensity?: number;
  /** World-space triplanar tiles per metre. Ignored on the low preset. */
  triplanar?: number;
  transparent?: boolean;
  opacity?: number;
}

/**
 * Private, vertex-colourable views onto the shared MaterialLibrary.
 *
 * The library caches by option key and hands the *same* object to every caller,
 * so `vertexColors = true` cannot be set on it directly — a mesh without a
 * colour attribute would read black. Every material here is therefore a clone.
 * Clones share texture objects, so this costs one program, not one atlas.
 *
 * The library is asked for `repeat: 1` in every case: that is the only path
 * that leaves the material on the stock program with no `onBeforeCompile`, so
 * the clone is guaranteed lossless. Tiling density is authored into the UVs by
 * the geometry builders instead.
 */
export class MaterialVault {
  private readonly cache = new Map<string, THREE.MeshStandardMaterial>();
  private readonly owned: THREE.Material[] = [];
  private readonly lowPreset: boolean;

  constructor(
    private readonly lib: MaterialLibrary,
    private readonly quality: QualitySettings,
  ) {
    this.lowPreset = quality.preset === 'low';
  }

  surfaceOf(look: SurfaceLook): SurfaceKind {
    return this.lib.surfaceOf(look);
  }

  get(look: SurfaceLook, opts: VaultOpts = {}): THREE.MeshStandardMaterial {
    const tri = this.lowPreset ? 0 : (opts.triplanar ?? 0);
    const key = [
      look,
      opts.color ?? '-',
      opts.seed ?? 0,
      opts.roughness ?? '-',
      opts.metalness ?? '-',
      opts.normalScale ?? '-',
      opts.emissive ?? '-',
      opts.emissiveIntensity ?? '-',
      opts.side ?? '-',
      opts.envMapIntensity ?? '-',
      opts.transparent ?? '-',
      opts.opacity ?? '-',
      tri,
    ].join('|');
    const hit = this.cache.get(key);
    if (hit) return hit;

    const base = this.lib.get(look, {
      repeat: 1,
      seed: opts.seed,
      color: opts.color,
      roughness: opts.roughness,
      metalness: opts.metalness,
      normalScale: opts.normalScale,
      emissive: opts.emissive,
      emissiveIntensity: opts.emissiveIntensity,
      side: opts.side,
      envMapIntensity: opts.envMapIntensity,
    });

    let mat = base.clone() as THREE.MeshStandardMaterial;
    mat.vertexColors = true;
    if (opts.transparent) {
      mat.transparent = true;
      mat.opacity = opts.opacity ?? 1;
      mat.depthWrite = false;
    }
    if (tri > 0) {
      const t = triplanarMaterial(mat, tri);
      mat.dispose();
      mat = t;
    }
    mat.name = `vault:${look}`;
    this.cache.set(key, mat);
    this.owned.push(mat);
    return mat;
  }

  /** Registers an externally-created material for disposal with the vault. */
  own<T extends THREE.Material>(mat: T): T {
    this.owned.push(mat);
    return mat;
  }

  get triplanarEnabled(): boolean {
    return !this.lowPreset;
  }

  dispose(): void {
    for (const m of this.owned) m.dispose();
    this.owned.length = 0;
    this.cache.clear();
  }
}

// ---------------------------------------------------------------------------
// Wind
// ---------------------------------------------------------------------------

export interface WindOpts {
  /** Distance from the anchor at which sway reaches full strength, metres. */
  height: number;
  /** Peak lateral displacement at full strength, metres. */
  strength: number;
  /**
   * Pins the object's local origin as the *top* and lets everything below it
   * swing — hanging cloth, not a plant. Geometry must therefore be authored
   * with the pin at local y = 0 and the free end at negative y, which also
   * means a taut awning (all local y >= 0) correctly does not move at all.
   */
  hanging?: boolean;
  key: string;
}

/**
 * Two-frequency sway with a per-instance phase taken from world position.
 * A slow 0.18Hz carrier gives the mass its weight; a 0.6Hz overtone gives the
 * edges their flutter. Displacement is weighted by height^2 so the root stays
 * planted — linear weighting is the classic tell of a fake wind shader.
 */
function injectWind(mat: THREE.Material, uTime: { value: number }, opts: WindOpts): void {
  const uStrength = { value: opts.strength };
  const invH = 1 / Math.max(0.05, opts.height);
  const weight = opts.hanging
    ? `float bend = clamp( -transformed.y * ${invH.toFixed(4)}, 0.0, 1.0 );
    bend = bend * bend;`
    : `float bend = clamp( transformed.y * ${invH.toFixed(4)}, 0.0, 1.0 );
    bend = bend * bend;`;

  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uWindTime = uTime;
    shader.uniforms.uWindStrength = uStrength;
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
uniform float uWindTime;
uniform float uWindStrength;`,
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
  {
    #ifdef USE_INSTANCING
      vec3 windAnchor = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
    #else
      vec3 windAnchor = ( modelMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
    #endif
    float phase = windAnchor.x * 0.83 + windAnchor.z * 0.61;
    ${weight}
    float s1 = sin( uWindTime * 1.15 + phase );
    float s2 = sin( uWindTime * 3.77 + phase * 2.31 + 1.7 );
    float sway = s1 * 0.74 + s2 * 0.26;
    float gust = 0.68 + 0.32 * sin( uWindTime * 0.31 + phase * 0.17 );
    transformed.x += sway * bend * uWindStrength * gust;
    transformed.z += sway * bend * uWindStrength * gust * 0.42;
    // Shorten as it leans: a bending stem does not stretch.
    transformed.y -= bend * uWindStrength * abs( sway ) * 0.22 * gust;
  }`,
      );
  };
  mat.customProgramCacheKey = () => `cod-wind-${opts.key}`;
  mat.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Foliage textures
// ---------------------------------------------------------------------------

function canvas2d(size: number): { c: HTMLCanvasElement; g: CanvasRenderingContext2D } {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const g = c.getContext('2d');
  if (!g) throw new Error('2D canvas unavailable');
  return { c, g };
}

/** One tapered blade, drawn as a filled quadratic sliver with a spine highlight. */
function drawBlade(
  g: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  len: number,
  width: number,
  lean: number,
  rng: Rng,
  hue: number,
): void {
  const tipX = x0 + lean;
  const tipY = y0 - len;
  const ctrlX = x0 + lean * 0.32;
  const ctrlY = y0 - len * 0.55;
  const sat = rng.range(22, 46);
  const light = rng.range(18, 42);
  const grad = g.createLinearGradient(x0, y0, tipX, tipY);
  grad.addColorStop(0, `hsl(${hue - 6}, ${sat}%, ${light * 0.55}%)`);
  grad.addColorStop(0.55, `hsl(${hue}, ${sat}%, ${light}%)`);
  grad.addColorStop(1, `hsl(${hue + 12}, ${sat * 0.85}%, ${light * 1.7}%)`);
  g.fillStyle = grad;
  g.beginPath();
  g.moveTo(x0 - width * 0.5, y0);
  g.quadraticCurveTo(ctrlX - width * 0.34, ctrlY, tipX, tipY);
  g.quadraticCurveTo(ctrlX + width * 0.34, ctrlY, x0 + width * 0.5, y0);
  g.closePath();
  g.fill();
}

function drawLeaf(
  g: CanvasRenderingContext2D,
  x: number,
  y: number,
  len: number,
  wide: number,
  angle: number,
  rng: Rng,
  hue: number,
): void {
  g.save();
  g.translate(x, y);
  g.rotate(angle);
  const grad = g.createLinearGradient(0, 0, 0, -len);
  const light = rng.range(20, 38);
  grad.addColorStop(0, `hsl(${hue - 8}, 30%, ${light * 0.6}%)`);
  grad.addColorStop(1, `hsl(${hue + 10}, 34%, ${light * 1.5}%)`);
  g.fillStyle = grad;
  g.beginPath();
  g.moveTo(0, 0);
  g.quadraticCurveTo(wide, -len * 0.45, 0, -len);
  g.quadraticCurveTo(-wide, -len * 0.45, 0, 0);
  g.fill();
  g.strokeStyle = `hsla(${hue + 16}, 30%, ${light * 1.9}%, 0.5)`;
  g.lineWidth = Math.max(0.7, len * 0.02);
  g.beginPath();
  g.moveTo(0, 0);
  g.lineTo(0, -len * 0.92);
  g.stroke();
  g.restore();
}

/**
 * 2x2 atlas of alpha-cut foliage: dry grass, broadleaf weed, dense shrub, dead
 * brush. Generated once; four looks out of one texture and one draw call each.
 */
function makeFoliageAtlas(size: number): { map: THREE.CanvasTexture; normalMap: THREE.DataTexture } {
  const { c, g } = canvas2d(size);
  const h = size / 2;
  g.clearRect(0, 0, size, size);
  const rng = new Rng(0x51af7);

  // Tile 0,0 — dry grass tuft.
  g.save();
  g.beginPath();
  g.rect(0, 0, h, h);
  g.clip();
  for (let i = 0; i < 46; i++) {
    const x = h * 0.5 + rng.jitter(h * 0.34);
    const len = h * rng.range(0.42, 0.94);
    drawBlade(g, x, h - 2, len, h * rng.range(0.018, 0.042), rng.jitter(h * 0.34), rng, rng.range(48, 74));
  }
  g.restore();

  // Tile 1,0 — broadleaf weed.
  g.save();
  g.beginPath();
  g.rect(h, 0, h, h);
  g.clip();
  for (let i = 0; i < 16; i++) {
    const x = h * 1.5 + rng.jitter(h * 0.2);
    const y = h - rng.range(0, h * 0.42);
    drawLeaf(g, x, y, h * rng.range(0.28, 0.56), h * rng.range(0.07, 0.15), rng.jitter(1.15), rng, rng.range(72, 104));
  }
  for (let i = 0; i < 10; i++) {
    drawBlade(g, h * 1.5 + rng.jitter(h * 0.16), h - 2, h * rng.range(0.5, 0.8), h * 0.02, rng.jitter(h * 0.2), rng, 66);
  }
  g.restore();

  // Tile 0,1 — dense low shrub.
  g.save();
  g.beginPath();
  g.rect(0, h, h, h);
  g.clip();
  for (let i = 0; i < 120; i++) {
    const a = rng.range(0, Math.PI * 2);
    const r = h * 0.36 * Math.pow(rng.next(), 0.6);
    const x = h * 0.5 + Math.cos(a) * r;
    const y = h * 1.62 + Math.sin(a) * r * 0.68;
    drawLeaf(g, x, y, h * rng.range(0.07, 0.16), h * rng.range(0.025, 0.05), rng.range(0, Math.PI * 2), rng, rng.range(60, 96));
  }
  g.restore();

  // Tile 1,1 — dead brush and twigs.
  g.save();
  g.beginPath();
  g.rect(h, h, h, h);
  g.clip();
  for (let i = 0; i < 34; i++) {
    const x = h * 1.5 + rng.jitter(h * 0.3);
    g.strokeStyle = `hsl(${rng.range(26, 44)}, ${rng.range(16, 34)}%, ${rng.range(20, 46)}%)`;
    g.lineWidth = rng.range(0.8, 2.6);
    g.beginPath();
    g.moveTo(x, size - 2);
    g.quadraticCurveTo(x + rng.jitter(h * 0.2), size - h * rng.range(0.3, 0.6), x + rng.jitter(h * 0.42), size - h * rng.range(0.5, 0.95));
    g.stroke();
  }
  for (let i = 0; i < 26; i++) {
    drawBlade(g, h * 1.5 + rng.jitter(h * 0.3), size - 2, h * rng.range(0.25, 0.6), h * 0.016, rng.jitter(h * 0.3), rng, rng.range(36, 52));
  }
  g.restore();

  const map = new THREE.CanvasTexture(c);
  map.colorSpace = THREE.SRGBColorSpace;
  map.wrapS = THREE.ClampToEdgeWrapping;
  map.wrapT = THREE.ClampToEdgeWrapping;
  map.generateMipmaps = true;
  map.minFilter = THREE.LinearMipmapLinearFilter;
  map.magFilter = THREE.LinearFilter;
  map.needsUpdate = true;

  // Normal map: sobel of a blurred alpha field. Gives the cards a soft rounded
  // shading instead of the flat cut-out look alpha testing usually produces.
  const img = g.getImageData(0, 0, size, size).data;
  const height = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) height[i] = img[i * 4 + 3] / 255;
  const blurred = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let s = 0;
      let n = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= size) continue;
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= size) continue;
          s += height[yy * size + xx];
          n++;
        }
      }
      blurred[y * size + x] = s / n;
    }
  }
  const nrm = new Uint8Array(size * size * 4);
  const at = (x: number, y: number): number =>
    blurred[Math.min(size - 1, Math.max(0, y)) * size + Math.min(size - 1, Math.max(0, x))];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * 3.2;
      const dy = (at(x, y + 1) - at(x, y - 1)) * 3.2;
      let nx = -dx;
      let ny = dy;
      let nz = 1;
      const len = Math.hypot(nx, ny, nz);
      nx /= len;
      ny /= len;
      nz /= len;
      const o = (y * size + x) * 4;
      nrm[o] = (nx * 0.5 + 0.5) * 255;
      nrm[o + 1] = (ny * 0.5 + 0.5) * 255;
      nrm[o + 2] = (nz * 0.5 + 0.5) * 255;
      nrm[o + 3] = 255;
    }
  }
  const normalMap = new THREE.DataTexture(nrm, size, size, THREE.RGBAFormat);
  normalMap.needsUpdate = true;
  normalMap.wrapS = THREE.ClampToEdgeWrapping;
  normalMap.wrapT = THREE.ClampToEdgeWrapping;
  normalMap.minFilter = THREE.LinearFilter;
  normalMap.magFilter = THREE.LinearFilter;

  return { map, normalMap };
}

/** UV rect for one atlas tile. */
const ATLAS: Record<string, [number, number]> = {
  grass: [0, 0],
  weed: [1, 0],
  shrub: [0, 1],
  brush: [1, 1],
};

/**
 * Crossed cards with a bowed spine. Flat quads read as cardboard the moment
 * the camera moves; three planes at 60 degrees with a 12cm bow do not.
 */
function foliageCard(width: number, height: number, tile: [number, number], planes: number, rng: Rng): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const [tx, ty] = tile;
  const rows = 3;
  for (let p = 0; p < planes; p++) {
    const pos: number[] = [];
    const nor: number[] = [];
    const uv: number[] = [];
    const bow = rng.range(-0.16, 0.16) * height;
    for (let r = 0; r < rows; r++) {
      const v0 = r / rows;
      const v1 = (r + 1) / rows;
      const y0 = v0 * height;
      const y1 = v1 * height;
      const z0 = bow * v0 * v0;
      const z1 = bow * v1 * v1;
      const w0 = width * (0.5 + 0.5 * (1 - v0 * 0.25));
      const w1 = width * (0.5 + 0.5 * (1 - v1 * 0.25));
      const quad = [
        [-w0 * 0.5, y0, z0, 0, 1 - v0],
        [w0 * 0.5, y0, z0, 1, 1 - v0],
        [w1 * 0.5, y1, z1, 1, 1 - v1],
        [-w0 * 0.5, y0, z0, 0, 1 - v0],
        [w1 * 0.5, y1, z1, 1, 1 - v1],
        [-w1 * 0.5, y1, z1, 0, 1 - v1],
      ];
      for (const [x, y, z, u, v] of quad) {
        pos.push(x, y, z);
        nor.push(0, 0.35, 0.94);
        uv.push((tx + u) * 0.5, (ty + v) * 0.5);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array((pos.length / 3) * 3).fill(1), 3));
    g.rotateY((p / planes) * Math.PI + rng.jitter(0.25));
    parts.push(g);
  }
  return finalizeGeometry(mergeAll(parts));
}

// ---------------------------------------------------------------------------
// Poisson-disc
// ---------------------------------------------------------------------------

/**
 * Bridson's algorithm. Debris in a gutter is neither gridded nor clumped; a
 * Poisson-disc is the cheapest distribution that reads as "settled".
 */
export function poissonDisc(width: number, height: number, radius: number, rng: Rng, k = 12): THREE.Vector2[] {
  const cell = radius / Math.SQRT2;
  const gw = Math.max(1, Math.ceil(width / cell));
  const gh = Math.max(1, Math.ceil(height / cell));
  const grid = new Int32Array(gw * gh).fill(-1);
  const points: THREE.Vector2[] = [];
  const active: number[] = [];
  const r2 = radius * radius;

  const insert = (p: THREE.Vector2): void => {
    const gx = Math.min(gw - 1, (p.x / cell) | 0);
    const gy = Math.min(gh - 1, (p.y / cell) | 0);
    grid[gy * gw + gx] = points.length;
    active.push(points.length);
    points.push(p);
  };

  const fits = (x: number, y: number): boolean => {
    if (x < 0 || y < 0 || x >= width || y >= height) return false;
    const gx = Math.min(gw - 1, (x / cell) | 0);
    const gy = Math.min(gh - 1, (y / cell) | 0);
    for (let yy = Math.max(0, gy - 2); yy <= Math.min(gh - 1, gy + 2); yy++) {
      for (let xx = Math.max(0, gx - 2); xx <= Math.min(gw - 1, gx + 2); xx++) {
        const id = grid[yy * gw + xx];
        if (id < 0) continue;
        const p = points[id];
        const dx = p.x - x;
        const dy = p.y - y;
        if (dx * dx + dy * dy < r2) return false;
      }
    }
    return true;
  };

  insert(new THREE.Vector2(rng.range(0, width), rng.range(0, height)));
  let guard = 0;
  while (active.length > 0 && guard++ < 200000) {
    const ai = Math.min(active.length - 1, (rng.next() * active.length) | 0);
    const p = points[active[ai]];
    let placed = false;
    for (let i = 0; i < k; i++) {
      const a = rng.range(0, Math.PI * 2);
      const d = radius * rng.range(1.001, 2);
      const x = p.x + Math.cos(a) * d;
      const y = p.y + Math.sin(a) * d;
      if (!fits(x, y)) continue;
      insert(new THREE.Vector2(x, y));
      placed = true;
      break;
    }
    if (!placed) active.splice(ai, 1);
  }
  return points;
}

// ---------------------------------------------------------------------------
// Prop definitions
// ---------------------------------------------------------------------------

export type PropKind =
  | 'barrel'
  | 'crate_large'
  | 'crate_small'
  | 'pallet'
  | 'jersey'
  | 'sandbag'
  | 'tyre'
  | 'ac_unit'
  | 'dish'
  | 'planter'
  | 'stall'
  | 'rubble_chunk'
  | 'brick_shard'
  | 'gravel'
  | 'plank'
  | 'paper'
  | 'can'
  | 'rebar_tuft'
  | 'grass'
  | 'weed'
  | 'shrub'
  | 'brush';

interface PropPart {
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
  castShadow: boolean;
  receiveShadow: boolean;
}

interface PropDef {
  parts: PropPart[];
  /** Local-space solid mass. Empty for clutter you walk over. */
  boxes: BoxSpec[];
  surface: SurfaceKind;
  /** Poisson spacing hint, metres. */
  radius: number;
  /** Per-instance scale spread. */
  scaleJitter: number;
  /** Per-instance tilt, radians. */
  tilt: number;
}

interface Instance {
  matrix: THREE.Matrix4;
  color: THREE.Color;
}

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _s = new THREE.Vector3();
const _p = new THREE.Vector3();

// ---------------------------------------------------------------------------
// Prop system
// ---------------------------------------------------------------------------

export interface GroundSample {
  y: number;
  /** Surface tilt, radians, around a random axis. */
  slope: number;
}

export class PropSystem {
  readonly root = new THREE.Group();

  private readonly defs = new Map<PropKind, PropDef>();
  private readonly instances = new Map<PropKind, Instance[]>();
  private readonly meshes: THREE.InstancedMesh[] = [];
  private readonly ownedGeometries: THREE.BufferGeometry[] = [];
  private readonly windTime = { value: 0 };
  private foliageAtlas: { map: THREE.CanvasTexture; normalMap: THREE.DataTexture } | null = null;
  private foliageMaterial: THREE.MeshStandardMaterial | null = null;
  private clothMaterial: THREE.MeshStandardMaterial | null = null;
  private readonly rng: Rng;
  private built = false;

  constructor(
    private readonly vault: MaterialVault,
    private readonly quality: QualitySettings,
    seed = 0x2f11a7,
  ) {
    this.root.name = 'level-props';
    this.rng = new Rng(seed);
  }

  /** Density multiplier applied to every scatter call. */
  get densityScale(): number {
    switch (this.quality.preset) {
      case 'low':
        return 0.42;
      case 'medium':
        return 0.7;
      case 'ultra':
        return 1.2;
      default:
        return 1;
    }
  }

  get windUniform(): { value: number } {
    return this.windTime;
  }

  /** Cloth material with the hanging-wind shader; shared by awnings and laundry. */
  cloth(): THREE.MeshStandardMaterial {
    if (this.clothMaterial) return this.clothMaterial;
    const mat = this.vault
      .get('fabric_canvas', { side: THREE.DoubleSide, seed: 7 })
      .clone() as THREE.MeshStandardMaterial;
    mat.vertexColors = true;
    injectWind(mat, this.windTime, { height: 1.7, strength: 0.115, hanging: true, key: 'cloth' });
    // Shadows must sway with the geometry or the cloth detaches from its shadow.
    const depth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
    injectWind(depth, this.windTime, { height: 1.7, strength: 0.115, hanging: true, key: 'cloth' });
    mat.userData.clothDepth = depth;
    this.vault.own(mat);
    this.vault.own(depth);
    this.clothMaterial = mat;
    return mat;
  }

  private foliage(): THREE.MeshStandardMaterial {
    if (this.foliageMaterial) return this.foliageMaterial;
    const size = this.quality.preset === 'low' ? 256 : 512;
    this.foliageAtlas = makeFoliageAtlas(size);
    this.foliageAtlas.map.anisotropy = Math.min(4, this.quality.anisotropy);
    const mat = new THREE.MeshStandardMaterial({
      map: this.foliageAtlas.map,
      normalMap: this.foliageAtlas.normalMap,
      normalScale: new THREE.Vector2(0.7, 0.7),
      alphaTest: 0.42,
      transparent: false,
      side: THREE.DoubleSide,
      roughness: 0.86,
      metalness: 0,
      vertexColors: true,
      dithering: true,
    });
    injectWind(mat, this.windTime, { height: 0.6, strength: 0.075, key: 'foliage' });
    this.vault.own(mat);
    this.foliageMaterial = mat;
    return mat;
  }

  // -------------------------------------------------------------------------
  // Placement
  // -------------------------------------------------------------------------

  place(kind: PropKind, x: number, y: number, z: number, yaw = 0, scale = 1, tilt = 0, tiltAxis = 0): void {
    const def = this.def(kind);
    let list = this.instances.get(kind);
    if (!list) {
      list = [];
      this.instances.set(kind, list);
    }
    const s = scale * (1 + this.rng.jitter(def.scaleJitter));
    const t = tilt + this.rng.jitter(def.tilt);
    _e.set(Math.cos(tiltAxis) * t, yaw, Math.sin(tiltAxis) * t, 'YXZ');
    _q.setFromEuler(_e);
    const m = new THREE.Matrix4().compose(_p.set(x, y, z), _q, _s.set(s, s * (1 + this.rng.jitter(def.scaleJitter * 0.5)), s));
    // Per-instance albedo multiplier: ±7% value, ±4% warmth. Enough to break
    // the "one object repeated" read without turning the set into confetti.
    const v = 1 + this.rng.jitter(0.075);
    const warm = this.rng.jitter(0.045);
    list.push({ matrix: m, color: new THREE.Color(v + warm, v, v - warm * 0.6) });
  }

  /** Solid props register their collision the moment they are placed. */
  placeSolid(
    collider: ColliderSet,
    kind: PropKind,
    x: number,
    y: number,
    z: number,
    yaw = 0,
    scale = 1,
  ): void {
    this.place(kind, x, y, z, yaw, scale);
    const def = this.def(kind);
    if (def.boxes.length === 0) return;
    _m.compose(_p.set(x, y, z), _q.setFromEuler(_e.set(0, yaw, 0, 'YXZ')), _s.set(scale, scale, scale));
    collider.addBoxes(def.surface, def.boxes, _m);
  }

  /**
   * Poisson scatter inside a rectangle. `accept` rejects points that fall on
   * road, inside geometry or outside the intended band.
   */
  scatter(
    kinds: readonly PropKind[],
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    spacing: number,
    ground: (x: number, z: number) => GroundSample | null,
    accept?: (x: number, z: number) => boolean,
    scale = 1,
  ): number {
    const w = x1 - x0;
    const d = z1 - z0;
    if (w <= 0 || d <= 0) return 0;
    const r = spacing / Math.max(0.35, this.densityScale);
    const pts = poissonDisc(w, d, r, this.rng);
    let n = 0;
    for (const p of pts) {
      const x = x0 + p.x;
      const z = z0 + p.y;
      if (accept && !accept(x, z)) continue;
      const g = ground(x, z);
      if (!g) continue;
      const kind = kinds[Math.min(kinds.length - 1, (this.rng.next() * kinds.length) | 0)];
      this.place(kind, x, g.y, z, this.rng.range(0, Math.PI * 2), scale, g.slope, this.rng.range(0, Math.PI * 2));
      n++;
    }
    return n;
  }

  // -------------------------------------------------------------------------
  // Build
  // -------------------------------------------------------------------------

  build(): THREE.Group {
    if (this.built) return this.root;
    this.built = true;
    for (const [kind, list] of this.instances) {
      if (list.length === 0) continue;
      const def = this.def(kind);
      for (const part of def.parts) {
        const mesh = new THREE.InstancedMesh(part.geometry, part.material, list.length);
        mesh.name = `prop:${kind}`;
        mesh.castShadow = part.castShadow;
        mesh.receiveShadow = part.receiveShadow;
        mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
        for (let i = 0; i < list.length; i++) {
          mesh.setMatrixAt(i, list[i].matrix);
          mesh.setColorAt(i, list[i].color);
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        mesh.computeBoundingSphere();
        // Collision is authored separately; never let the extractor walk these.
        mesh.userData.noCollide = true;
        this.root.add(mesh);
        this.meshes.push(mesh);
      }
    }
    return this.root;
  }

  update(elapsed: number): void {
    this.windTime.value = elapsed;
  }

  dispose(): void {
    for (const m of this.meshes) {
      m.dispose();
      m.parent?.remove(m);
    }
    this.meshes.length = 0;
    for (const g of this.ownedGeometries) g.dispose();
    this.ownedGeometries.length = 0;
    this.defs.clear();
    this.instances.clear();
    this.foliageAtlas?.map.dispose();
    this.foliageAtlas?.normalMap.dispose();
    this.foliageAtlas = null;
    this.root.clear();
  }

  // -------------------------------------------------------------------------
  // Geometry
  // -------------------------------------------------------------------------

  private def(kind: PropKind): PropDef {
    const hit = this.defs.get(kind);
    if (hit) return hit;
    const built = this.buildDef(kind);
    for (const p of built.parts) this.ownedGeometries.push(p.geometry);
    this.defs.set(kind, built);
    return built;
  }

  private part(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    castShadow = true,
    receiveShadow = true,
  ): PropPart {
    return { geometry, material, castShadow, receiveShadow };
  }

  private buildDef(kind: PropKind): PropDef {
    const rng = new Rng(0x9d1 + kind.length * 7919 + kind.charCodeAt(0) * 104729);
    switch (kind) {
      case 'barrel':
        return this.barrelDef(rng);
      case 'crate_large':
        return this.crateDef(0.92, rng);
      case 'crate_small':
        return this.crateDef(0.56, rng);
      case 'pallet':
        return this.palletDef();
      case 'jersey':
        return this.jerseyDef();
      case 'sandbag':
        return this.sandbagDef(rng);
      case 'tyre':
        return this.tyreDef();
      case 'ac_unit':
        return this.acUnitDef();
      case 'dish':
        return this.dishDef();
      case 'planter':
        return this.planterDef();
      case 'stall':
        return this.stallDef(rng);
      case 'rubble_chunk':
        return this.rubbleDef(rng);
      case 'brick_shard':
        return this.shardDef(rng);
      case 'gravel':
        return this.gravelDef(rng);
      case 'plank':
        return this.plankDef();
      case 'paper':
        return this.paperDef(rng);
      case 'can':
        return this.canDef();
      case 'rebar_tuft':
        return this.rebarDef(rng);
      case 'grass':
        return this.foliageDef('grass', 0.52, 0.46, 3, rng);
      case 'weed':
        return this.foliageDef('weed', 0.44, 0.6, 3, rng);
      case 'shrub':
        return this.foliageDef('shrub', 0.86, 0.62, 4, rng);
      default:
        return this.foliageDef('brush', 0.66, 0.72, 3, rng);
    }
  }

  private barrelDef(rng: Rng): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    const r = 0.29;
    const hgt = 0.88;
    const body = cylinderGeo(r, r * 0.985, hgt, 14);
    body.translate(0, hgt * 0.5, 0);
    parts.push(body);
    for (const y of [0.26, 0.66]) {
      const rim = torusGeo(r * 1.03, 0.021, 5, 14);
      rim.rotateX(Math.PI * 0.5);
      rim.translate(0, hgt * y + 0.02, 0);
      parts.push(rim);
    }
    const lip = torusGeo(r * 0.98, 0.026, 5, 14);
    lip.rotateX(Math.PI * 0.5);
    lip.translate(0, hgt - 0.02, 0);
    parts.push(lip);
    const bung = cylinderGeo(0.05, 0.05, 0.022, 8);
    bung.translate(r * 0.5, hgt + 0.004, 0);
    parts.push(bung);
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 1.35);
    tintGeometry(merged, 0xe8e2d8);

    return {
      parts: [
        this.part(
          merged,
          this.vault.get('rusted_metal', { seed: 3, roughness: 0.94, color: rng.chance(0.5) ? 0xb8623a : 0x7c8b7e }),
        ),
      ],
      boxes: [{ cx: 0, cy: hgt * 0.5, cz: 0, sx: r * 1.9, sy: hgt, sz: r * 1.9 }],
      surface: 'metal',
      radius: 0.7,
      scaleJitter: 0.05,
      tilt: 0.03,
    };
  }

  private crateDef(size: number, rng: Rng): PropDef {
    const body = bevelBox(size, size * 0.86, size * 0.92, 0.016);
    body.translate(0, size * 0.43, 0);
    scaleUv(body, 1.6 / size);

    const frameParts: THREE.BufferGeometry[] = [];
    const t = 0.038;
    for (const sx of [-1, 1]) {
      const g = bevelBox(t, size * 0.86, t, 0.005);
      g.translate(sx * (size * 0.5 - t * 0.4), size * 0.43, size * 0.46 - t * 0.4);
      frameParts.push(g);
      const g2 = g.clone();
      g2.translate(0, 0, -(size * 0.92 - t * 0.8));
      frameParts.push(g2);
    }
    for (const sy of [0.09, 0.77]) {
      const g = bevelBox(size + 0.008, t * 0.8, size * 0.92 + 0.008, 0.005);
      g.translate(0, size * sy, 0);
      frameParts.push(g);
    }
    // A sprung board on one face; nothing in a war zone is intact.
    if (rng.chance(0.6)) {
      const board = bevelBox(size * 0.94, size * 0.16, 0.022, 0.005);
      board.rotateX(rng.range(0.12, 0.3));
      board.translate(0, size * 0.62, size * 0.48);
      frameParts.push(board);
    }
    const frame = finalizeGeometry(mergeAll(frameParts));
    scaleUv(frame, 2.2);

    return {
      parts: [
        this.part(body, this.vault.get('wood_crate', { seed: 2 })),
        this.part(frame, this.vault.get('wood_plank', { seed: 5, color: 0xa08055 })),
      ],
      boxes: [{ cx: 0, cy: size * 0.43, cz: 0, sx: size, sy: size * 0.86, sz: size * 0.92 }],
      surface: 'wood',
      radius: size * 1.15,
      scaleJitter: 0.06,
      tilt: 0.025,
    };
  }

  private palletDef(): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    for (let i = 0; i < 3; i++) {
      const g = bevelBox(0.11, 0.09, 1.16, 0.006);
      g.translate(-0.44 + i * 0.44, 0.045, 0);
      parts.push(g);
    }
    for (let i = 0; i < 6; i++) {
      const g = bevelBox(1.0, 0.02, 0.11, 0.004);
      g.translate(0, 0.1, -0.52 + i * 0.208);
      parts.push(g);
    }
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 1.8);
    return {
      parts: [this.part(merged, this.vault.get('wood_plank', { seed: 11, color: 0x8f7550 }))],
      boxes: [{ cx: 0, cy: 0.06, cz: 0, sx: 1.0, sy: 0.12, sz: 1.16 }],
      surface: 'wood',
      radius: 1.1,
      scaleJitter: 0.04,
      tilt: 0.02,
    };
  }

  private jerseyDef(): PropDef {
    const geo = extrudeProfile(jerseyProfile(0.82), 2.0, { bevel: 0.014 });
    geo.rotateY(-Math.PI * 0.5);
    scaleUv(geo, 0.9);
    // Lift hole recesses in the top, which is what the real castings have.
    const holes: THREE.BufferGeometry[] = [geo];
    for (const z of [-0.42, 0.42]) {
      const h = bevelBox(0.16, 0.07, 0.1, 0.012);
      h.translate(0, 0.82, z);
      holes.push(h);
    }
    const merged = finalizeGeometry(mergeAll(holes));
    return {
      parts: [this.part(merged, this.vault.get('concrete_wall', { seed: 21, color: 0xb9b3a6, roughness: 1 }))],
      boxes: [
        { cx: 0, cy: 0.2, cz: 0, sx: 0.6, sy: 0.4, sz: 2.0 },
        { cx: 0, cy: 0.6, cz: 0, sx: 0.3, sy: 0.44, sz: 2.0 },
      ],
      surface: 'concrete',
      radius: 1.4,
      scaleJitter: 0.02,
      tilt: 0.012,
    };
  }

  private sandbagDef(rng: Rng): PropDef {
    // A heavy chamfer turns a box into a slumped bag in 44 triangles.
    const g = bevelBox(0.56, 0.2, 0.32, 0.078);
    g.scale(1, 1, 1 + rng.jitter(0.06));
    g.translate(0, 0.1, 0);
    scaleUv(g, 2.4);
    return {
      parts: [this.part(g, this.vault.get('fabric_canvas', { seed: 4, color: 0xb0a483, roughness: 1 }))],
      boxes: [{ cx: 0, cy: 0.1, cz: 0, sx: 0.56, sy: 0.2, sz: 0.32 }],
      surface: 'fabric',
      radius: 0.4,
      scaleJitter: 0.055,
      tilt: 0.05,
    };
  }

  private tyreDef(): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    const t = torusGeo(0.31, 0.115, 6, 16);
    t.rotateX(Math.PI * 0.5);
    t.translate(0, 0.115, 0);
    parts.push(t);
    // Tread blocks, otherwise a torus reads as a doughnut.
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2;
      const b = bevelBox(0.09, 0.03, 0.19, 0.008);
      b.rotateY(-a);
      b.translate(Math.cos(a) * 0.415, 0.115, Math.sin(a) * 0.415);
      parts.push(b);
    }
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 2.6);
    return {
      parts: [this.part(merged, this.vault.get('gun_polymer', { seed: 9, color: 0x2b2a29, roughness: 1.05 }))],
      boxes: [{ cx: 0, cy: 0.11, cz: 0, sx: 0.84, sy: 0.23, sz: 0.84 }],
      surface: 'rubber',
      radius: 0.6,
      scaleJitter: 0.045,
      tilt: 0.04,
    };
  }

  private acUnitDef(): PropDef {
    const shell = bevelBox(0.82, 0.62, 0.36, 0.018);
    shell.translate(0, 0.31, 0);
    scaleUv(shell, 1.5);

    const detail: THREE.BufferGeometry[] = [];
    for (let i = 0; i < 6; i++) {
      const s = plainBox(0.7, 0.03, 0.02);
      s.rotateX(-0.34);
      s.translate(0, 0.12 + i * 0.076, 0.185);
      detail.push(s);
    }
    const ring = torusGeo(0.2, 0.022, 5, 12);
    ring.translate(0, 0.33, 0.19);
    detail.push(ring);
    for (const sx of [-1, 1]) {
      const brk = bevelBox(0.05, 0.4, 0.3, 0.008);
      brk.translate(sx * 0.44, 0.2, -0.06);
      detail.push(brk);
    }
    const pipe = cylinderGeo(0.024, 0.024, 0.34, 8);
    pipe.rotateZ(Math.PI * 0.5);
    pipe.translate(-0.5, 0.14, -0.12);
    detail.push(pipe);
    const merged = finalizeGeometry(mergeAll(detail));
    scaleUv(merged, 2.4);

    return {
      parts: [
        this.part(shell, this.vault.get('painted_metal', { seed: 14, color: 0xc9c4b4 })),
        this.part(merged, this.vault.get('rusted_metal', { seed: 15, color: 0x8d8a83 })),
      ],
      boxes: [{ cx: 0, cy: 0.31, cz: 0, sx: 0.82, sy: 0.62, sz: 0.36 }],
      surface: 'metal',
      radius: 1.0,
      scaleJitter: 0.05,
      tilt: 0.01,
    };
  }

  private dishDef(): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    const dish = finalizeGeometry(
      new THREE.SphereGeometry(0.44, 20, 10, 0, Math.PI * 2, 0, Math.PI * 0.28),
    );
    dish.rotateX(Math.PI * 0.86);
    dish.translate(0, 0.72, 0.1);
    parts.push(dish);
    const arm = cylinderGeo(0.016, 0.016, 0.42, 6);
    arm.rotateX(-0.9);
    arm.translate(0, 0.66, -0.16);
    parts.push(arm);
    const lnb = cylinderGeo(0.035, 0.05, 0.13, 8);
    lnb.rotateX(-0.9);
    lnb.translate(0, 0.56, -0.32);
    parts.push(lnb);
    const mast = cylinderGeo(0.028, 0.028, 0.62, 8);
    mast.translate(0, 0.31, 0);
    parts.push(mast);
    const foot = bevelBox(0.24, 0.05, 0.24, 0.01);
    foot.translate(0, 0.025, 0);
    parts.push(foot);
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 1.6);
    return {
      parts: [this.part(merged, this.vault.get('painted_metal', { seed: 17, color: 0xd6d2c6, side: THREE.DoubleSide }))],
      boxes: [],
      surface: 'metal',
      radius: 1.2,
      scaleJitter: 0.09,
      tilt: 0.05,
    };
  }

  private planterDef(): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    const body = cylinderGeo(0.42, 0.34, 0.56, 14);
    body.translate(0, 0.28, 0);
    parts.push(body);
    const rim = torusGeo(0.43, 0.035, 6, 18);
    rim.rotateX(Math.PI * 0.5);
    rim.translate(0, 0.55, 0);
    parts.push(rim);
    const soil = cylinderGeo(0.38, 0.38, 0.06, 14);
    soil.translate(0, 0.53, 0);
    parts.push(soil);
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 1.4);
    return {
      parts: [this.part(merged, this.vault.get('concrete_wall', { seed: 23, color: 0xa89b84 }))],
      boxes: [{ cx: 0, cy: 0.28, cz: 0, sx: 0.84, sy: 0.56, sz: 0.84 }],
      surface: 'concrete',
      radius: 1.0,
      scaleJitter: 0.05,
      tilt: 0.015,
    };
  }

  private stallDef(rng: Rng): PropDef {
    const w = 2.4;
    const d = 1.5;
    const h = 2.15;
    const timber: THREE.BufferGeometry[] = [];
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const post = bevelBox(0.07, h, 0.07, 0.008);
        post.translate(sx * (w * 0.5 - 0.05), h * 0.5, sz * (d * 0.5 - 0.05));
        timber.push(post);
      }
    }
    for (const sz of [-1, 1]) {
      const beam = bevelBox(w, 0.07, 0.06, 0.008);
      beam.translate(0, h - 0.05, sz * (d * 0.5 - 0.05));
      timber.push(beam);
    }
    // Counter top and its skirt.
    const top = bevelBox(w - 0.05, 0.055, d - 0.08, 0.008);
    top.translate(0, 0.92, 0);
    timber.push(top);
    for (let i = 0; i < 5; i++) {
      const slat = bevelBox(w - 0.12, 0.16, 0.022, 0.004);
      slat.rotateX(rng.jitter(0.03));
      slat.translate(0, 0.14 + i * 0.17, d * 0.5 - 0.08);
      timber.push(slat);
    }
    const timberGeo = finalizeGeometry(mergeAll(timber));
    scaleUv(timberGeo, 1.5);

    // Canopy: a sagging four-panel cloth, pinned at the ridge.
    const cloth: THREE.BufferGeometry[] = [];
    const panels = 5;
    for (let i = 0; i < panels; i++) {
      const t0 = i / panels;
      const t1 = (i + 1) / panels;
      const sag = (t: number): number => -0.19 * Math.sin(t * Math.PI);
      const quad = new THREE.BufferGeometry();
      const x0 = -w * 0.5 + w * t0;
      const x1 = -w * 0.5 + w * t1;
      const y0 = h + sag(t0);
      const y1 = h + sag(t1);
      const pos: number[] = [];
      const uv: number[] = [];
      const nor: number[] = [];
      const zf = d * 0.5 + 0.34;
      const zb = -d * 0.5 - 0.06;
      const drop = 0.28;
      const push = (px: number, py: number, pz: number, u: number, v: number): void => {
        pos.push(px, py, pz);
        uv.push(u, v);
        nor.push(0, 1, 0);
      };
      push(x0, y0, zb, t0, 0);
      push(x1, y1, zb, t1, 0);
      push(x1, y1 - drop, zf, t1, 1);
      push(x0, y0, zb, t0, 0);
      push(x1, y1 - drop, zf, t1, 1);
      push(x0, y0 - drop, zf, t0, 1);
      quad.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      quad.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
      quad.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
      quad.computeVertexNormals();
      cloth.push(finalizeGeometry(quad));
    }
    // Scalloped valance along the front edge.
    for (let i = 0; i < 8; i++) {
      const cx = -w * 0.5 + (w * (i + 0.5)) / 8;
      const s = bevelBox(w / 8 - 0.01, 0.19, 0.012, 0.004);
      s.translate(cx, h - 0.28 - 0.09, d * 0.5 + 0.34);
      cloth.push(s);
    }
    const clothGeo = finalizeGeometry(mergeAll(cloth));
    scaleUv(clothGeo, 1.1);
    tintGeometry(clothGeo, rng.pick([0xd8b48a, 0xc45a4a, 0xd6c9a8, 0x7f9aa6]));

    return {
      parts: [
        this.part(timberGeo, this.vault.get('wood_plank', { seed: 31, color: 0x9c7d55 })),
        this.part(clothGeo, this.cloth(), true, true),
      ],
      boxes: [
        { cx: 0, cy: 0.55, cz: 0, sx: w - 0.05, sy: 1.1, sz: d - 0.08 },
        { cx: 0, cy: h - 0.1, cz: 0, sx: w, sy: 0.2, sz: d },
      ],
      surface: 'wood',
      radius: 2.6,
      scaleJitter: 0.035,
      tilt: 0.012,
    };
  }

  private rubbleDef(rng: Rng): PropDef {
    const g = rubbleCone(0.55, 0.3, 5, rng.int(1, 9999), 0.13);
    scaleUv(g, 2.2);
    return {
      parts: [this.part(g, this.vault.get('rubble', { seed: 6 }), false, true)],
      boxes: [],
      surface: 'concrete',
      radius: 0.9,
      scaleJitter: 0.22,
      tilt: 0.06,
    };
  }

  private shardDef(rng: Rng): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    for (let i = 0; i < 3; i++) {
      const b = bevelBox(rng.range(0.1, 0.22), rng.range(0.05, 0.09), rng.range(0.06, 0.11), 0.008);
      b.rotateY(rng.range(0, Math.PI));
      b.rotateZ(rng.jitter(0.5));
      b.translate(rng.jitter(0.17), rng.range(0.02, 0.06), rng.jitter(0.17));
      parts.push(b);
    }
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 4);
    return {
      parts: [this.part(merged, this.vault.get('brick', { seed: 8 }), false, true)],
      boxes: [],
      surface: 'concrete',
      radius: 0.42,
      scaleJitter: 0.25,
      tilt: 0.1,
    };
  }

  private gravelDef(rng: Rng): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    // Gravel is 3-7cm: below the size where a chamfer survives to the screen.
    for (let i = 0; i < 5; i++) {
      const s = rng.range(0.028, 0.075);
      const b = plainBox(s * rng.range(0.8, 1.5), s * 0.7, s * rng.range(0.8, 1.4));
      b.rotateY(rng.range(0, Math.PI));
      b.rotateX(rng.jitter(0.6));
      b.translate(rng.jitter(0.14), s * 0.3, rng.jitter(0.14));
      parts.push(b);
    }
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 8);
    return {
      parts: [this.part(merged, this.vault.get('dirt_gravel', { seed: 12 }), false, true)],
      boxes: [],
      surface: 'dirt',
      radius: 0.3,
      scaleJitter: 0.3,
      tilt: 0.12,
    };
  }

  private plankDef(): PropDef {
    const g = bevelBox(1.5, 0.035, 0.16, 0.006);
    g.translate(0, 0.02, 0);
    scaleUv(g, 1.4);
    return {
      parts: [this.part(g, this.vault.get('wood_plank', { seed: 19, color: 0x8a7150 }), false, true)],
      boxes: [],
      surface: 'wood',
      radius: 1.0,
      scaleJitter: 0.16,
      tilt: 0.05,
    };
  }

  private paperDef(rng: Rng): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    for (let i = 0; i < 3; i++) {
      // A curled sheet: two quads with a fold, never flat on the ground.
      const w = rng.range(0.15, 0.24);
      const h = rng.range(0.2, 0.3);
      const g = new THREE.BufferGeometry();
      const lift = rng.range(0.01, 0.05);
      const pos = [
        -w * 0.5, 0.002, -h * 0.5, w * 0.5, 0.002, -h * 0.5, w * 0.5, lift * 0.4, 0,
        -w * 0.5, 0.002, -h * 0.5, w * 0.5, lift * 0.4, 0, -w * 0.5, lift * 0.4, 0,
        -w * 0.5, lift * 0.4, 0, w * 0.5, lift * 0.4, 0, w * 0.45, lift, h * 0.5,
        -w * 0.5, lift * 0.4, 0, w * 0.45, lift, h * 0.5, -w * 0.45, lift, h * 0.5,
      ];
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.computeVertexNormals();
      const f = finalizeGeometry(g);
      f.rotateY(rng.range(0, Math.PI * 2));
      f.translate(rng.jitter(0.16), 0, rng.jitter(0.16));
      parts.push(f);
    }
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 3);
    tintGeometry(merged, 0xd9d2c0);
    return {
      parts: [
        this.part(
          merged,
          this.vault.get('plaster_painted', { seed: 27, color: 0xcfc7b2, side: THREE.DoubleSide }),
          false,
          true,
        ),
      ],
      boxes: [],
      surface: 'fabric',
      radius: 0.5,
      scaleJitter: 0.24,
      tilt: 0.03,
    };
  }

  private canDef(): PropDef {
    const g = cylinderGeo(0.033, 0.033, 0.115, 10);
    g.rotateZ(Math.PI * 0.5);
    g.translate(0, 0.033, 0);
    scaleUv(g, 6);
    return {
      parts: [this.part(g, this.vault.get('painted_metal', { seed: 29, color: 0x9aa0a4 }), false, true)],
      boxes: [],
      surface: 'metal',
      radius: 0.3,
      scaleJitter: 0.1,
      tilt: 0.06,
    };
  }

  private rebarDef(rng: Rng): PropDef {
    const parts: THREE.BufferGeometry[] = [];
    for (let i = 0; i < 3; i++) {
      const b = rebarGeo(rng.range(0.5, 1.15), rng.int(1, 9999), 0.011);
      b.rotateY(rng.range(0, Math.PI * 2));
      b.rotateX(rng.jitter(0.5));
      b.translate(rng.jitter(0.14), 0, rng.jitter(0.14));
      parts.push(b);
    }
    const merged = finalizeGeometry(mergeAll(parts));
    scaleUv(merged, 5);
    return {
      parts: [this.part(merged, this.vault.get('rusted_metal', { seed: 33, color: 0x8a4b2c }), true, true)],
      boxes: [],
      surface: 'metal',
      radius: 0.7,
      scaleJitter: 0.2,
      tilt: 0.08,
    };
  }

  private foliageDef(
    tile: keyof typeof ATLAS,
    width: number,
    height: number,
    planes: number,
    rng: Rng,
  ): PropDef {
    const g = foliageCard(width, height, ATLAS[tile], planes, rng);
    return {
      // Foliage receives shadow but never casts: a shadow pass over thousands of
      // alpha-tested cards is the most expensive thing on this map and the
      // least visible.
      parts: [this.part(g, this.foliage(), false, true)],
      boxes: [],
      surface: 'foliage',
      radius: 0.4,
      scaleJitter: 0.26,
      tilt: 0.09,
    };
  }
}

// ---------------------------------------------------------------------------
// Hero props
// ---------------------------------------------------------------------------

export interface HeroProp {
  object: THREE.Object3D;
  /** Local-space collision, to be transformed by the caller's placement. */
  boxes: BoxSpec[];
}

/**
 * Burnt-out saloon. Built from swept and chamfered masses rather than a shell:
 * the silhouette is what sells a wreck at 30m, and the collapsed roofline,
 * sprung bonnet and missing glass are all silhouette.
 *
 * Returned as a THREE.LOD — the detail level carries the trim, wheels and
 * suspension (~5k triangles), the far level is the four main masses.
 */
export function buildBurntCar(vault: MaterialVault, seed = 0x0cad): HeroProp {
  const rng = new Rng(seed);
  const body: THREE.BufferGeometry[] = [];
  const trim: THREE.BufferGeometry[] = [];
  const rubber: THREE.BufferGeometry[] = [];

  const L = 4.3;
  const W = 1.76;

  // Lower body: sills, floor pan, wheel arches.
  const lower = bevelBox(W, 0.52, L, 0.05);
  lower.translate(0, 0.52, 0);
  body.push(lower);

  const bonnet = bevelBox(W * 0.94, 0.16, 1.28, 0.035);
  bonnet.rotateX(rng.range(-0.26, -0.14));
  bonnet.translate(0, 0.86, L * 0.5 - 0.72);
  body.push(bonnet);

  const boot = bevelBox(W * 0.92, 0.15, 1.0, 0.035);
  boot.rotateX(0.05);
  boot.translate(0, 0.83, -L * 0.5 + 0.58);
  body.push(boot);

  // Cabin: A and C pillars plus a caved-in roof panel.
  for (const sx of [-1, 1]) {
    const a = bevelBox(0.1, 0.66, 0.12, 0.02);
    a.rotateX(-0.5);
    a.translate(sx * (W * 0.5 - 0.08), 1.09, 0.42);
    body.push(a);
    const c = bevelBox(0.11, 0.62, 0.14, 0.02);
    c.rotateX(0.42);
    c.translate(sx * (W * 0.5 - 0.08), 1.07, -0.86);
    body.push(c);
    const bpillar = bevelBox(0.1, 0.6, 0.1, 0.02);
    bpillar.translate(sx * (W * 0.5 - 0.07), 1.06, -0.2);
    body.push(bpillar);
  }
  const roof = bevelBox(W * 0.86, 0.09, 1.5, 0.03);
  roof.rotateZ(rng.range(0.05, 0.13));
  roof.rotateX(rng.jitter(0.05));
  roof.translate(rng.jitter(0.06), 1.33, -0.2);
  body.push(roof);

  // Doors, one hanging open on its hinge.
  for (const sx of [-1, 1]) {
    for (const dz of [0.42, -0.72]) {
      const swing = sx > 0 && dz > 0 ? rng.range(0.5, 0.75) : 0;
      const door = bevelBox(0.07, 0.62, 1.02, 0.02);
      if (swing > 0) {
        door.translate(0, 0, -0.51);
        door.rotateY(-swing * sx);
        door.translate(0, 0, 0.51);
      }
      door.translate(sx * (W * 0.5 - 0.02) + (swing > 0 ? sx * 0.34 : 0), 1.0, dz);
      body.push(door);
    }
  }

  // Bumpers and grille.
  for (const sz of [1, -1]) {
    const bumper = bevelBox(W + 0.06, 0.19, 0.16, 0.03);
    bumper.translate(0, 0.62, sz * (L * 0.5 - 0.02));
    trim.push(bumper);
  }
  const grille = bevelBox(W * 0.72, 0.22, 0.07, 0.015);
  grille.translate(0, 0.83, L * 0.5 - 0.02);
  trim.push(grille);
  for (const sx of [-1, 1]) {
    const lamp = bevelBox(0.3, 0.17, 0.1, 0.02);
    lamp.translate(sx * W * 0.32, 0.85, L * 0.5 - 0.03);
    trim.push(lamp);
    const mirror = bevelBox(0.16, 0.09, 0.06, 0.012);
    mirror.translate(sx * (W * 0.5 + 0.06), 1.14, 0.62);
    trim.push(mirror);
  }

  // Wheels: one burnt down to the rim and sagging.
  const wheelZ = [L * 0.5 - 1.02, -L * 0.5 + 0.94];
  const flatIndex = rng.int(0, 3);
  let wi = 0;
  const wheelBoxes: BoxSpec[] = [];
  for (const sx of [-1, 1]) {
    for (const z of wheelZ) {
      const flat = wi === flatIndex;
      const r = flat ? 0.24 : 0.33;
      const tyre = torusGeo(r, flat ? 0.07 : 0.115, 7, 16);
      tyre.rotateY(Math.PI * 0.5);
      tyre.scale(1, flat ? 0.82 : 1, 1);
      tyre.translate(sx * (W * 0.5 - 0.06), r * (flat ? 0.85 : 1), z);
      rubber.push(tyre);
      const hub = cylinderGeo(r * 0.56, r * 0.56, 0.16, 10);
      hub.rotateZ(Math.PI * 0.5);
      hub.translate(sx * (W * 0.5 - 0.06), r * (flat ? 0.85 : 1), z);
      trim.push(hub);
      wheelBoxes.push({ cx: sx * (W * 0.5 - 0.06), cy: r * 0.5, cz: z, sx: 0.28, sy: r * 2, sz: r * 2 });
      wi++;
    }
  }

  const bodyGeo = finalizeGeometry(mergeAll(body));
  scaleUv(bodyGeo, 1.1);
  // Soot gradient: black at the cabin, scorched paint toward the extremities.
  tintGeometry(bodyGeo, 0x6a6560);

  const trimGeo = finalizeGeometry(mergeAll(trim));
  scaleUv(trimGeo, 2.0);

  const rubberGeo = finalizeGeometry(mergeAll(rubber));
  scaleUv(rubberGeo, 3.0);

  const bodyMat = vault.get('painted_metal', { seed: 41, color: 0x3b342f, roughness: 1.15, metalness: 0.85 });
  const trimMat = vault.get('rusted_metal', { seed: 42, color: 0x6f6259 });
  const rubberMat = vault.get('gun_polymer', { seed: 43, color: 0x201f1e, roughness: 1.1 });

  const detail = new THREE.Group();
  for (const [g, m] of [
    [bodyGeo, bodyMat],
    [trimGeo, trimMat],
    [rubberGeo, rubberMat],
  ] as const) {
    const mesh = new THREE.Mesh(g, m);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.noCollide = true;
    detail.add(mesh);
  }

  // Far LOD: the four masses, no trim, no wheels.
  const farParts: THREE.BufferGeometry[] = [];
  const fl = bevelBox(W, 0.86, L, 0.05);
  fl.translate(0, 0.62, 0);
  farParts.push(fl);
  const fr = bevelBox(W * 0.86, 0.5, 1.7, 0.04);
  fr.translate(0, 1.2, -0.2);
  farParts.push(fr);
  const farGeo = finalizeGeometry(mergeAll(farParts));
  scaleUv(farGeo, 1.1);
  tintGeometry(farGeo, 0x6a6560);
  const far = new THREE.Mesh(farGeo, bodyMat);
  far.castShadow = true;
  far.receiveShadow = true;
  far.userData.noCollide = true;

  const lod = new THREE.LOD();
  lod.name = 'burnt-car';
  lod.addLevel(detail, 0);
  lod.addLevel(far, 38);
  lod.userData.noCollide = true;

  return {
    object: lod,
    boxes: [
      { cx: 0, cy: 0.52, cz: 0, sx: W, sy: 0.62, sz: L },
      { cx: 0, cy: 1.1, cz: -0.2, sx: W * 0.94, sy: 0.62, sz: 1.9 },
      { cx: 0, cy: 0.86, cz: L * 0.5 - 0.72, sx: W * 0.94, sy: 0.3, sz: 1.3 },
      { cx: 0, cy: 0.83, cz: -L * 0.5 + 0.58, sx: W * 0.92, sy: 0.3, sz: 1.0 },
      ...wheelBoxes,
    ],
  };
}

/**
 * Laundry line: a catenary cable with sheets pegged along it. The sheets use
 * the hanging-wind material so they lift and settle instead of hanging like
 * sheet metal.
 */
export function buildLaundryLine(
  vault: MaterialVault,
  clothMat: THREE.MeshStandardMaterial,
  from: THREE.Vector3,
  to: THREE.Vector3,
  sag: number,
  count: number,
  seed: number,
): THREE.Object3D {
  const rng = new Rng(seed);
  const group = new THREE.Group();
  group.name = 'laundry';
  group.userData.noCollide = true;
  // Everything is authored relative to the line's anchor so local y = 0 is the
  // cable: that is exactly the pin the hanging-wind shader expects.
  group.position.copy(from);
  const localTo = new THREE.Vector3().subVectors(to, from);
  const localFrom = new THREE.Vector3(0, 0, 0);

  const pts: THREE.Vector3[] = [];
  const segs = 14;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const p = new THREE.Vector3().lerpVectors(localFrom, localTo, t);
    p.y -= sag * Math.sin(t * Math.PI);
    pts.push(p);
  }
  const cable = finalizeGeometry(
    new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), segs, 0.012, 5, false),
  );
  const cableMesh = new THREE.Mesh(cable, vault.get('rusted_metal', { seed: 51, color: 0x5b5449 }));
  cableMesh.castShadow = true;
  cableMesh.userData.noCollide = true;
  group.add(cableMesh);

  const sheets: THREE.BufferGeometry[] = [];
  for (let i = 0; i < count; i++) {
    const t = (i + 0.5 + rng.jitter(0.22)) / count;
    const p = new THREE.Vector3().lerpVectors(localFrom, localTo, Math.min(0.95, Math.max(0.05, t)));
    p.y -= sag * Math.sin(t * Math.PI);
    const w = rng.range(0.5, 0.95);
    const h = rng.range(0.55, 1.25);
    const dir = localTo.clone().setY(0).normalize();
    const g = new THREE.PlaneGeometry(w, h, 2, 3);
    g.translate(0, -h * 0.5, 0);
    // Ripple across the hanging sheet so it is never a flat rectangle.
    const pos = g.getAttribute('position') as THREE.BufferAttribute;
    for (let v = 0; v < pos.count; v++) {
      const px = pos.getX(v);
      const py = pos.getY(v);
      pos.setZ(v, Math.sin(px * 6.1 + i) * 0.035 * (1 + py / h) + rng.jitter(0.012));
    }
    g.computeVertexNormals();
    const f = finalizeGeometry(g);
    f.rotateY(Math.atan2(dir.x, dir.z) + Math.PI * 0.5 + rng.jitter(0.18));
    f.translate(p.x, p.y - 0.01, p.z);
    tintGeometry(f, rng.pick([0xffffff, 0xd8dbe4, 0xe3d3bb, 0xc3cdc0, 0xe8c9b6]));
    sheets.push(f);
  }
  const sheetGeo = finalizeGeometry(mergeAll(sheets));
  scaleUv(sheetGeo, 1.6);
  const sheetMesh = new THREE.Mesh(sheetGeo, clothMat);
  sheetMesh.castShadow = true;
  sheetMesh.receiveShadow = true;
  sheetMesh.userData.noCollide = true;
  const depth = clothMat.userData.clothDepth as THREE.MeshDepthMaterial | undefined;
  if (depth) sheetMesh.customDepthMaterial = depth;
  group.add(sheetMesh);

  return group;
}
