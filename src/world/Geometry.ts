/**
 * Procedural geometry toolkit for the level.
 *
 * Three rules drive everything in this file:
 *
 *  1. No raw BoxGeometry ever reaches the screen. `bevelBox` chamfers all twelve
 *     edges; a 1-3cm chamfer is what turns a flat-shaded rectangle into
 *     something that catches a specular line and reads as a built object.
 *  2. Every geometry leaves here with the *same* attribute set — position,
 *     normal, uv, color, non-indexed — so any two can be handed to
 *     `mergeGeometries` without a second thought. That is what keeps a whole
 *     town down to a couple of dozen draw calls.
 *  3. Collision is authored separately from visuals. Builders that produce
 *     solid mass also emit `BoxSpec`s; the level registers those (and only
 *     those) with the physics world, so bullets and capsules test ~6k
 *     triangles instead of ~200k, and never snag on a 2cm chamfer.
 */
import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { SurfaceKind } from '../core/Contracts';

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------

/** mulberry32. Deterministic across reloads so the map never re-rolls. */
export class Rng {
  private s: number;

  constructor(seed: number) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }

  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  /** Inclusive integer range. */
  int(a: number, b: number): number {
    return a + Math.min(b - a, Math.floor(this.next() * (b - a + 1)));
  }

  chance(p: number): boolean {
    return this.next() < p;
  }

  sign(): number {
    return this.next() < 0.5 ? -1 : 1;
  }

  pick<T>(arr: readonly T[]): T {
    return arr[Math.min(arr.length - 1, Math.floor(this.next() * arr.length))];
  }

  /** Symmetric jitter around zero. */
  jitter(amount: number): number {
    return (this.next() * 2 - 1) * amount;
  }
}

// ---------------------------------------------------------------------------
// Attribute plumbing
// ---------------------------------------------------------------------------

const _v0 = new THREE.Vector3();
const _e1 = new THREE.Vector3();
const _e2 = new THREE.Vector3();
const _gn = new THREE.Vector3();
const _col = new THREE.Color();

/**
 * Guarantees the attribute contract: non-indexed, position + normal + uv +
 * color. Anything that comes out of three's own generators (Extrude, Cylinder,
 * Torus) goes through here before it is allowed near a merge.
 */
export function finalizeGeometry(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  let g = geo;
  if (g.index) {
    const nonIndexed = g.toNonIndexed();
    g.dispose();
    g = nonIndexed;
  }
  if (!g.getAttribute('normal')) g.computeVertexNormals();

  const pos = g.getAttribute('position') as THREE.BufferAttribute;
  const count = pos.count;

  if (!g.getAttribute('uv')) {
    const nor = g.getAttribute('normal') as THREE.BufferAttribute;
    const uv = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const nx = Math.abs(nor.getX(i));
      const ny = Math.abs(nor.getY(i));
      const nz = Math.abs(nor.getZ(i));
      const px = pos.getX(i);
      const py = pos.getY(i);
      const pz = pos.getZ(i);
      if (nx >= ny && nx >= nz) {
        uv[i * 2] = pz;
        uv[i * 2 + 1] = py;
      } else if (ny >= nz) {
        uv[i * 2] = px;
        uv[i * 2 + 1] = pz;
      } else {
        uv[i * 2] = px;
        uv[i * 2 + 1] = py;
      }
    }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  }

  if (!g.getAttribute('color')) {
    const c = new Float32Array(count * 3).fill(1);
    g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  }

  // Nothing downstream uses these and they break merges.
  for (const name of Object.keys(g.attributes)) {
    if (name !== 'position' && name !== 'normal' && name !== 'uv' && name !== 'color') {
      g.deleteAttribute(name);
    }
  }
  g.morphAttributes = {};
  g.clearGroups();
  return g;
}

/** Re-projects UVs from *current* (usually world) position, in metres. */
export function worldPlanarUv(geo: THREE.BufferGeometry, tilesPerMetre = 1): void {
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const nor = geo.getAttribute('normal') as THREE.BufferAttribute;
  const uv = geo.getAttribute('uv') as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const nx = Math.abs(nor.getX(i));
    const ny = Math.abs(nor.getY(i));
    const nz = Math.abs(nor.getZ(i));
    const px = pos.getX(i);
    const py = pos.getY(i);
    const pz = pos.getZ(i);
    if (nx >= ny && nx >= nz) uv.setXY(i, pz * tilesPerMetre, py * tilesPerMetre);
    else if (ny >= nz) uv.setXY(i, px * tilesPerMetre, pz * tilesPerMetre);
    else uv.setXY(i, px * tilesPerMetre, py * tilesPerMetre);
  }
  uv.needsUpdate = true;
}

export function scaleUv(geo: THREE.BufferGeometry, s: number): THREE.BufferGeometry {
  const uv = geo.getAttribute('uv') as THREE.BufferAttribute;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * s, uv.getY(i) * s);
  uv.needsUpdate = true;
  return geo;
}

/** Multiplies every vertex colour by a tint. */
export function tintGeometry(geo: THREE.BufferGeometry, color: THREE.ColorRepresentation): THREE.BufferGeometry {
  _col.set(color);
  const c = geo.getAttribute('color') as THREE.BufferAttribute;
  for (let i = 0; i < c.count; i++) {
    c.setXYZ(i, c.getX(i) * _col.r, c.getY(i) * _col.g, c.getZ(i) * _col.b);
  }
  c.needsUpdate = true;
  return geo;
}

/**
 * Aerial perspective, baked in. Vertices lift toward `color` (the sky at the
 * horizon) with horizontal distance from `x,z`, which is what stops a building
 * 180m out reading at the same contrast and saturation as the one 12m out.
 * Fog does part of this job in the pipeline; doing the albedo lift here as well
 * is what makes the far massing sit *behind* the near massing rather than
 * merely being tinted by something in front of it.
 */
export function paintAerial(
  geo: THREE.BufferGeometry,
  x: number,
  z: number,
  near: number,
  far: number,
  color: THREE.ColorRepresentation,
  maxStrength = 0.75,
): void {
  _col.set(color);
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const c = geo.getAttribute('color') as THREE.BufferAttribute;
  const span = Math.max(1e-3, far - near);
  for (let i = 0; i < pos.count; i++) {
    const dx = pos.getX(i) - x;
    const dz = pos.getZ(i) - z;
    const t = Math.min(1, Math.max(0, (Math.hypot(dx, dz) - near) / span));
    const k = t * t * (3 - 2 * t) * maxStrength;
    if (k <= 0.001) continue;
    c.setXYZ(
      i,
      c.getX(i) * (1 - k) + _col.r * k,
      c.getY(i) * (1 - k) + _col.g * k,
      c.getZ(i) * (1 - k) + _col.b * k,
    );
  }
  c.needsUpdate = true;
}

/**
 * Slack cable between two points. A real catenary, not a straight line: the sag
 * is what makes a wire read as a wire, and a run of them across a street is the
 * cheapest strong silhouette element there is against a bright sky.
 */
export function cableGeo(
  from: THREE.Vector3,
  to: THREE.Vector3,
  sag: number,
  radius = 0.018,
  segments = 10,
): THREE.BufferGeometry {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const p = from.clone().lerp(to, t);
    // cosh-shaped droop, normalised so the ends stay pinned.
    p.y -= sag * (Math.cosh((t - 0.5) * 3.2) - Math.cosh(1.6)) / (1 - Math.cosh(1.6));
    pts.push(p);
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  return finalizeGeometry(new THREE.TubeGeometry(curve, segments, radius, 4, false));
}

/**
 * Multiplies vertex colour toward `color` inside a sphere with a smooth falloff.
 * Used for scorch haloes around breaches and grime pooling in corners.
 */
export function paintSphere(
  geo: THREE.BufferGeometry,
  center: THREE.Vector3,
  radius: number,
  color: THREE.ColorRepresentation,
  strength = 1,
  falloffPower = 1.6,
): void {
  _col.set(color);
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const c = geo.getAttribute('color') as THREE.BufferAttribute;
  const r2 = radius * radius;
  for (let i = 0; i < pos.count; i++) {
    const dx = pos.getX(i) - center.x;
    const dy = pos.getY(i) - center.y;
    const dz = pos.getZ(i) - center.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= r2) continue;
    const k = Math.pow(1 - Math.sqrt(d2) / radius, falloffPower) * strength;
    c.setXYZ(
      i,
      c.getX(i) * (1 - k + k * _col.r),
      c.getY(i) * (1 - k + k * _col.g),
      c.getZ(i) * (1 - k + k * _col.b),
    );
  }
  c.needsUpdate = true;
}

/**
 * Vertical grime streak: darkens downward from `yTop` over `length`, strongest
 * on the centre line. This is what puts water runoff below every window sill.
 */
export function paintRunoff(
  geo: THREE.BufferGeometry,
  x: number,
  z: number,
  yTop: number,
  length: number,
  halfWidth: number,
  color: THREE.ColorRepresentation,
  strength = 0.7,
): void {
  _col.set(color);
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const c = geo.getAttribute('color') as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const py = pos.getY(i);
    if (py > yTop + 0.05 || py < yTop - length) continue;
    const dx = pos.getX(i) - x;
    const dz = pos.getZ(i) - z;
    const lateral = Math.sqrt(dx * dx + dz * dz);
    if (lateral > halfWidth) continue;
    const down = (yTop - py) / length;
    const k = strength * (1 - lateral / halfWidth) * Math.min(1, down * 3) * (1 - down * 0.55);
    if (k <= 0) continue;
    c.setXYZ(
      i,
      c.getX(i) * (1 - k + k * _col.r),
      c.getY(i) * (1 - k + k * _col.g),
      c.getZ(i) * (1 - k + k * _col.b),
    );
  }
  c.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Triangle sink
// ---------------------------------------------------------------------------

/**
 * Accumulates flat-shaded triangles with auto-corrected winding: the caller
 * supplies the outward normal and the sink flips the triangle if the geometric
 * normal disagrees. That removes a whole category of "why is my box inside
 * out" bugs from every builder below.
 */
class TriSink {
  private readonly pos: number[] = [];
  private readonly nor: number[] = [];
  private readonly uv: number[] = [];

  tri(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, n: THREE.Vector3): void {
    _e1.subVectors(b, a);
    _e2.subVectors(c, a);
    _gn.crossVectors(_e1, _e2);
    if (_gn.lengthSq() < 1e-12) return;
    if (_gn.dot(n) < 0) {
      this.push(a, n);
      this.push(c, n);
      this.push(b, n);
    } else {
      this.push(a, n);
      this.push(b, n);
      this.push(c, n);
    }
  }

  quad(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, d: THREE.Vector3, n: THREE.Vector3): void {
    this.tri(a, b, c, n);
    this.tri(a, c, d, n);
  }

  private push(p: THREE.Vector3, n: THREE.Vector3): void {
    this.pos.push(p.x, p.y, p.z);
    this.nor.push(n.x, n.y, n.z);
    const ax = Math.abs(n.x);
    const ay = Math.abs(n.y);
    const az = Math.abs(n.z);
    if (ax >= ay && ax >= az) this.uv.push(p.z, p.y);
    else if (ay >= az) this.uv.push(p.x, p.z);
    else this.uv.push(p.x, p.y);
  }

  build(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array((this.pos.length / 3) * 3).fill(1), 3));
    return g;
  }
}

/** Empty geometry that still satisfies the attribute contract. */
function emptyGeometry(): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute([], 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute([], 2));
  g.setAttribute('color', new THREE.Float32BufferAttribute([], 3));
  return g;
}

/**
 * mergeGeometries dereferences `geometries[0]` and throws on an empty list, and
 * a merge of one is a pointless copy. Every builder in this file goes through
 * here so neither case can reach it.
 */
export function mergeAll(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  if (parts.length === 0) return emptyGeometry();
  if (parts.length === 1) return parts[0];
  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  return merged;
}

// ---------------------------------------------------------------------------
// Boxes
// ---------------------------------------------------------------------------

/** Axis-aligned local-space box, used for collision authoring. */
export interface BoxSpec {
  cx: number;
  cy: number;
  cz: number;
  sx: number;
  sy: number;
  sz: number;
}

const EDGE_PAIRS: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 2],
  [0, 2],
];

/**
 * Chamfered box: 6 inset faces, 12 edge chamfers, 8 corner triangles.
 * 44 triangles. Yes, that is 3.6x a BoxGeometry; it is also the difference
 * between "an object" and "a screenshot of a cube".
 */
export function bevelBox(w: number, h: number, d: number, bevel = 0.02): THREE.BufferGeometry {
  const ax = w * 0.5;
  const ay = h * 0.5;
  const az = d * 0.5;
  const t = Math.max(0.0015, Math.min(bevel, Math.min(w, h, d) * 0.45));

  // v[corner][axis] — the three vertices that replace each sharp corner.
  const v: THREE.Vector3[][] = new Array(8);
  for (let c = 0; c < 8; c++) {
    const sx = c & 1 ? 1 : -1;
    const sy = c & 2 ? 1 : -1;
    const sz = c & 4 ? 1 : -1;
    v[c] = [
      new THREE.Vector3(sx * ax, sy * (ay - t), sz * (az - t)),
      new THREE.Vector3(sx * (ax - t), sy * ay, sz * (az - t)),
      new THREE.Vector3(sx * (ax - t), sy * (ay - t), sz * az),
    ];
  }
  const corner = (s: readonly number[]): number => (s[0] > 0 ? 1 : 0) | (s[1] > 0 ? 2 : 0) | (s[2] > 0 ? 4 : 0);

  const sink = new TriSink();
  const n = new THREE.Vector3();
  const sg = [0, 0, 0];

  // Faces.
  for (let axis = 0; axis < 3; axis++) {
    const m = (axis + 1) % 3;
    const k = (axis + 2) % 3;
    for (const s of [-1, 1]) {
      const ring: THREE.Vector3[] = [];
      const order: readonly (readonly [number, number])[] = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ];
      for (const [sm, sk] of order) {
        sg[axis] = s;
        sg[m] = sm;
        sg[k] = sk;
        ring.push(v[corner(sg)][axis]);
      }
      n.set(0, 0, 0).setComponent(axis, s);
      sink.quad(ring[0], ring[1], ring[2], ring[3], n);
    }
  }

  // Edge chamfers.
  for (const [i, j] of EDGE_PAIRS) {
    const k = 3 - i - j;
    for (const si of [-1, 1]) {
      for (const sj of [-1, 1]) {
        sg[i] = si;
        sg[j] = sj;
        sg[k] = -1;
        const ca = corner(sg);
        sg[k] = 1;
        const cb = corner(sg);
        n.set(0, 0, 0).setComponent(i, si).setComponent(j, sj).normalize();
        sink.quad(v[ca][i], v[ca][j], v[cb][j], v[cb][i], n);
      }
    }
  }

  // Corner facets.
  for (let c = 0; c < 8; c++) {
    n.set(c & 1 ? 1 : -1, c & 2 ? 1 : -1, c & 4 ? 1 : -1).normalize();
    sink.tri(v[c][0], v[c][1], v[c][2], n);
  }

  return sink.build();
}

/** 12-triangle box. Only for geometry that is never seen (collision proxies). */
export function plainBox(w: number, h: number, d: number): THREE.BufferGeometry {
  return finalizeGeometry(new THREE.BoxGeometry(w, h, d));
}

// ---------------------------------------------------------------------------
// Extrusion
// ---------------------------------------------------------------------------

export interface ExtrudeOpts {
  /** Chamfer applied to the extruded silhouette. */
  bevel?: number;
  curveSegments?: number;
  /** Recentre the extrusion on z=0. */
  center?: boolean;
}

/**
 * Sweeps a closed 2D profile (XY) along Z. This is how every cornice, sill,
 * kerb, coping, handrail and jersey barrier in the map gets its section — a
 * profile with three or four steps in it reads as moulded stone from 30m and
 * costs a few dozen triangles.
 *
 * Convention for wall-mounted trim: rotateY(-PI/2) afterwards maps profile +X
 * (the projection direction) onto world +Z, and sweeps the run along X.
 */
export function extrudeProfile(
  profile: readonly THREE.Vector2[],
  length: number,
  opts: ExtrudeOpts = {},
): THREE.BufferGeometry {
  const bevel = Math.max(0, Math.min(opts.bevel ?? 0.012, length * 0.24));
  const shape = new THREE.Shape();
  shape.moveTo(profile[0].x, profile[0].y);
  for (let i = 1; i < profile.length; i++) shape.lineTo(profile[i].x, profile[i].y);
  shape.closePath();

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.001, length - bevel * 2),
    bevelEnabled: bevel > 0,
    bevelSize: bevel,
    bevelThickness: bevel,
    bevelSegments: 1,
    curveSegments: opts.curveSegments ?? 6,
    steps: 1,
  });
  // Bevelled extrusions span -bevel .. length-bevel; recentre exactly.
  if (opts.center !== false) geo.translate(0, 0, -(length * 0.5 - bevel));
  return finalizeGeometry(geo);
}

/** Classic stepped cornice: fascia, cyma, drip. Origin at the wall face. */
export function corniceProfile(project = 0.26, height = 0.34): THREE.Vector2[] {
  const p = project;
  const h = height;
  return [
    new THREE.Vector2(0, 0),
    new THREE.Vector2(p * 0.34, 0),
    new THREE.Vector2(p * 0.4, -h * 0.16),
    new THREE.Vector2(p * 0.78, -h * 0.2),
    new THREE.Vector2(p, -h * 0.44),
    new THREE.Vector2(p * 0.92, -h * 0.62),
    new THREE.Vector2(p * 0.52, -h * 0.78),
    new THREE.Vector2(p * 0.46, -h),
    new THREE.Vector2(0, -h),
  ];
}

/** Window sill with a weathering slope and a drip groove underneath. */
export function sillProfile(project = 0.14, height = 0.09): THREE.Vector2[] {
  return [
    new THREE.Vector2(-0.06, 0),
    new THREE.Vector2(project, -height * 0.28),
    new THREE.Vector2(project, -height * 0.62),
    new THREE.Vector2(project * 0.74, -height * 0.62),
    new THREE.Vector2(project * 0.7, -height * 0.82),
    new THREE.Vector2(project * 0.6, -height),
    new THREE.Vector2(-0.06, -height),
  ];
}

/** Road kerb: sloped face, rounded top arris. */
export function kerbProfile(height = 0.15, width = 0.16): THREE.Vector2[] {
  return [
    new THREE.Vector2(0, height),
    new THREE.Vector2(width * 0.82, height),
    new THREE.Vector2(width, height - 0.03),
    new THREE.Vector2(width, -0.32),
    new THREE.Vector2(0, -0.32),
  ];
}

/** Parapet coping stone with overhang both sides. */
export function copingProfile(width = 0.42, height = 0.11): THREE.Vector2[] {
  const w = width * 0.5;
  return [
    new THREE.Vector2(-w, 0),
    new THREE.Vector2(-w * 0.86, height * 0.62),
    new THREE.Vector2(0, height),
    new THREE.Vector2(w * 0.86, height * 0.62),
    new THREE.Vector2(w, 0),
    new THREE.Vector2(w, -height * 0.5),
    new THREE.Vector2(-w, -height * 0.5),
  ];
}

/** New-jersey barrier section, half-width 0.30 at the base. */
export function jerseyProfile(height = 0.82): THREE.Vector2[] {
  const h = height;
  return [
    new THREE.Vector2(-0.3, 0),
    new THREE.Vector2(-0.3, h * 0.09),
    new THREE.Vector2(-0.19, h * 0.31),
    new THREE.Vector2(-0.115, h),
    new THREE.Vector2(0.115, h),
    new THREE.Vector2(0.19, h * 0.31),
    new THREE.Vector2(0.3, h * 0.09),
    new THREE.Vector2(0.3, 0),
  ];
}

// ---------------------------------------------------------------------------
// Openings
// ---------------------------------------------------------------------------

/** Rectangular opening in a wall. `y` is the *bottom* edge, `x` the centre. */
export interface Opening {
  x: number;
  y: number;
  w: number;
  h: number;
  /**
   * Suppresses the sill and lintel. A shell hole is not joinery: it has no
   * head, no cill and no reveal, and dressing one with moulded stone is the
   * fastest way to make damage read as architecture.
   */
  raw?: boolean;
}

export interface WallResult {
  geometry: THREE.BufferGeometry;
  /** Local-space solid mass, for collision. */
  boxes: BoxSpec[];
}

export interface WallOpts {
  openings?: readonly Opening[];
  /** How far the inner leaf steps into the opening on each side. */
  reveal?: number;
  bevel?: number;
  /** Protruding base course. */
  plinth?: number;
  /** Horizontal string course at this height, if given. */
  bandY?: number;
  /** Emit sills under openings. */
  sills?: boolean;
  /** Emit lintels over openings. */
  lintels?: boolean;
  /** Split the slab into two leaves so openings get a stepped reveal. */
  twoLeaf?: boolean;
}

interface CellRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function uniqueSorted(values: number[], lo: number, hi: number): number[] {
  const out: number[] = [];
  for (const v of values) {
    if (v <= lo + 1e-4 || v >= hi - 1e-4) continue;
    out.push(v);
  }
  out.sort((a, b) => a - b);
  const dedup = [lo];
  for (const v of out) if (v - dedup[dedup.length - 1] > 1e-3) dedup.push(v);
  if (hi - dedup[dedup.length - 1] > 1e-3) dedup.push(hi);
  else dedup[dedup.length - 1] = hi;
  return dedup;
}

/** Greedy maximal-rectangle cover of the solid cells. Keeps box count low. */
function coverRects(solid: Uint8Array, nx: number, ny: number): CellRect[] {
  const used = new Uint8Array(nx * ny);
  const rects: CellRect[] = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const i = y * nx + x;
      if (!solid[i] || used[i]) continue;
      let x1 = x;
      while (x1 + 1 < nx && solid[y * nx + x1 + 1] && !used[y * nx + x1 + 1]) x1++;
      let y1 = y;
      let grow = true;
      while (grow && y1 + 1 < ny) {
        for (let xx = x; xx <= x1; xx++) {
          const j = (y1 + 1) * nx + xx;
          if (!solid[j] || used[j]) {
            grow = false;
            break;
          }
        }
        if (grow) y1++;
      }
      for (let yy = y; yy <= y1; yy++) for (let xx = x; xx <= x1; xx++) used[yy * nx + xx] = 1;
      rects.push({ x0: x, y0: y, x1: x1 + 1, y1: y1 + 1 });
    }
  }
  return rects;
}

/**
 * Effective hole rect for one opening at a given rebate. The bottom edge is
 * never rebated for a door-height opening — a 9cm lip across a threshold would
 * be an invisible trip hazard for both the player capsule and the AI.
 */
function effRect(o: Opening, shrink: number): { x0: number; x1: number; y0: number; y1: number } {
  const bottom = o.y <= 0.02 ? 0 : shrink;
  return {
    x0: o.x - o.w * 0.5 + shrink,
    x1: o.x + o.w * 0.5 - shrink,
    y0: o.y + bottom,
    y1: o.y + o.h - shrink,
  };
}

/**
 * One solid cell of a slab, plus how far each of its four sides may be grown.
 *
 * Growth is the seam fix. Two abutting chamfered panels meet in a V-groove whose
 * facets catch the key light and alias into a bright dashed hairline — the
 * single most engine-looking artefact on a facade. Growing each panel across an
 * edge it *shares with another solid panel* by one chamfer width buries both
 * chamfers behind the neighbour's flat face, so the two front faces meet
 * exactly, coplanar, with no groove and no overlap to z-fight. Sides that face
 * an opening, or the outside of the wall, are left alone: those are genuine
 * outside corners and the chamfer is what makes them read.
 */
interface SlabCell {
  rect: CellRect;
  xs: number[];
  ys: number[];
  /** Metres of growth on -x, +x, -y, +y. */
  grow: [number, number, number, number];
}

function slabRects(
  width: number,
  height: number,
  openings: readonly Opening[],
  shrink: number,
  weld = 0,
): SlabCell[] {
  const halfW = width * 0.5;
  const rects = openings.map((o) => effRect(o, shrink)).filter((r) => r.x1 > r.x0 && r.y1 > r.y0);
  const xCuts: number[] = [];
  const yCuts: number[] = [];
  for (const r of rects) {
    xCuts.push(r.x0, r.x1);
    yCuts.push(r.y0, r.y1);
  }
  const xs = uniqueSorted(xCuts, -halfW, halfW);
  const ys = uniqueSorted(yCuts, 0, height);
  const nx = xs.length - 1;
  const ny = ys.length - 1;
  const solid = new Uint8Array(nx * ny);
  for (let y = 0; y < ny; y++) {
    const cy = (ys[y] + ys[y + 1]) * 0.5;
    for (let x = 0; x < nx; x++) {
      const cx = (xs[x] + xs[x + 1]) * 0.5;
      let open = false;
      for (const r of rects) {
        if (cx > r.x0 && cx < r.x1 && cy > r.y0 && cy < r.y1) {
          open = true;
          break;
        }
      }
      solid[y * nx + x] = open ? 0 : 1;
    }
  }
  return coverRects(solid, nx, ny).map((rect) => {
    const grow: [number, number, number, number] = [0, 0, 0, 0];
    if (weld > 0) {
      for (let y = rect.y0; y < rect.y1; y++) {
        if (rect.x0 > 0 && solid[y * nx + rect.x0 - 1]) grow[0] = weld;
        if (rect.x1 < nx && solid[y * nx + rect.x1]) grow[1] = weld;
      }
      for (let x = rect.x0; x < rect.x1; x++) {
        if (rect.y0 > 0 && solid[(rect.y0 - 1) * nx + x]) grow[2] = weld;
        if (rect.y1 < ny && solid[rect.y1 * nx + x]) grow[3] = weld;
      }
    }
    return { rect, xs, ys, grow };
  });
}

/**
 * A wall slab with real holes in it.
 *
 * Local space: centred on X, base at y=0, thickness along Z. Openings are cut
 * by subdividing the slab on every opening edge and greedily re-merging the
 * solid cells, so any arrangement of overlapping or abutting holes works.
 *
 * With `twoLeaf` the slab is built as an outer and an inner leaf, the inner one
 * stepped into the opening by `reveal`. That rebate is what makes a window read
 * as a hole in half a metre of masonry instead of a hole in a sheet of card.
 */
export function wallWithOpenings(
  width: number,
  height: number,
  thickness: number,
  opts: WallOpts = {},
): WallResult {
  if (width <= 1e-3 || height <= 1e-3 || thickness <= 1e-3) {
    return { geometry: emptyGeometry(), boxes: [] };
  }
  const openings = opts.openings ?? [];
  const bevel = opts.bevel ?? 0.02;
  const reveal = opts.reveal ?? 0.09;
  const twoLeaf = opts.twoLeaf !== false && openings.length > 0 && thickness > 0.18;

  const parts: THREE.BufferGeometry[] = [];
  const boxes: BoxSpec[] = [];

  const emit = (cells: SlabCell[], z0: number, z1: number, collide: boolean): void => {
    for (const { rect, xs, ys, grow } of cells) {
      const cx0 = xs[rect.x0];
      const cx1 = xs[rect.x1];
      const cy0 = ys[rect.y0];
      const cy1 = ys[rect.y1];
      // Grown extents for the visual box; the collision box stays on the true
      // cell so a 2cm weld never shows up as a lip under the player capsule.
      const x0 = cx0 - grow[0];
      const x1 = cx1 + grow[1];
      const y0 = cy0 - grow[2];
      const y1 = cy1 + grow[3];
      const w = x1 - x0;
      const h = y1 - y0;
      const d = z1 - z0;
      if (w < 1e-3 || h < 1e-3) continue;
      const g = bevelBox(w, h, d, bevel);
      g.translate((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5);
      parts.push(g);
      if (collide) {
        boxes.push({
          cx: (cx0 + cx1) * 0.5,
          cy: (cy0 + cy1) * 0.5,
          cz: 0,
          sx: cx1 - cx0,
          sy: cy1 - cy0,
          sz: thickness,
        });
      }
    }
  };

  // One chamfer width of overlap: the neighbour's flat face starts exactly
  // where this panel's chamfer does, so the chamfer is occluded and the two
  // front planes abut without overlapping.
  const weld = bevel;
  const halfT = thickness * 0.5;
  if (twoLeaf) {
    const outerT = thickness * 0.56;
    emit(slabRects(width, height, openings, 0, weld), halfT - outerT, halfT, false);
    emit(slabRects(width, height, openings, reveal, weld), -halfT, halfT - outerT, true);
  } else {
    emit(slabRects(width, height, openings, 0, weld), -halfT, halfT, true);
  }

  // Plinth.
  if (opts.plinth && opts.plinth > 0) {
    const ph = opts.plinth;
    const g = bevelBox(width + 0.02, ph, thickness + 0.09, 0.022);
    g.translate(0, ph * 0.5, 0.02);
    parts.push(g);
    boxes.push({ cx: 0, cy: ph * 0.5, cz: 0.02, sx: width, sy: ph, sz: thickness + 0.09 });
  }

  // String course.
  if (opts.bandY !== undefined) {
    const g = extrudeProfile(
      [
        new THREE.Vector2(0, 0.09),
        new THREE.Vector2(0.11, 0.075),
        new THREE.Vector2(0.13, 0.02),
        new THREE.Vector2(0.1, -0.06),
        new THREE.Vector2(0, -0.075),
      ],
      width,
      { bevel: 0.008 },
    );
    g.rotateY(-Math.PI * 0.5);
    g.translate(0, opts.bandY, halfT);
    parts.push(g);
  }

  // Sills and lintels.
  for (const o of openings) {
    if (o.raw) continue;
    if (opts.sills !== false && o.y > 0.35) {
      // A sill that projects properly is a horizontal shadow line across the
      // elevation and the anchor for the runoff stain below it. 15cm was not
      // enough to throw one.
      const s = extrudeProfile(sillProfile(0.2, 0.1), o.w + 0.3, { bevel: 0.008 });
      s.rotateY(-Math.PI * 0.5);
      s.translate(o.x, o.y + 0.005, halfT);
      parts.push(s);
      boxes.push({ cx: o.x, cy: o.y - 0.05, cz: halfT * 0.5, sx: o.w + 0.3, sy: 0.1, sz: thickness });
    }
    if (opts.lintels !== false) {
      const l = bevelBox(o.w + 0.3, 0.15, thickness + 0.07, 0.018);
      l.translate(o.x, o.y + o.h + 0.075, 0.01);
      parts.push(l);
    }
  }

  return { geometry: finalizeGeometry(mergeAll(parts)), boxes };
}

/**
 * A wall panel with a segmental-arched opening cut clean through it. Built from
 * a Shape with a hole so the extrusion walls *are* the reveal — no faked depth.
 */
export function archway(
  width: number,
  height: number,
  thickness: number,
  openW: number,
  springHeight: number,
  rise: number,
  bevel = 0.018,
): WallResult {
  const hw = width * 0.5;
  const ow = openW * 0.5;
  const shape = new THREE.Shape();
  shape.moveTo(-hw, 0);
  shape.lineTo(hw, 0);
  shape.lineTo(hw, height);
  shape.lineTo(-hw, height);
  shape.closePath();

  const hole = new THREE.Path();
  hole.moveTo(-ow, 0);
  hole.lineTo(-ow, springHeight);
  hole.absellipse(0, springHeight, ow, rise, Math.PI, 0, true, 0);
  hole.lineTo(ow, 0);
  hole.closePath();
  shape.holes.push(hole);

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.02, thickness - bevel * 2),
    bevelEnabled: bevel > 0,
    bevelSize: bevel,
    bevelThickness: bevel,
    bevelSegments: 1,
    curveSegments: 14,
    steps: 1,
  });
  geo.translate(0, 0, -thickness * 0.5);

  // Collision: piers plus the spandrel above the arch crown.
  const boxes: BoxSpec[] = [];
  const pierW = hw - ow;
  if (pierW > 0.02) {
    boxes.push({ cx: -(ow + pierW * 0.5), cy: height * 0.5, cz: 0, sx: pierW, sy: height, sz: thickness });
    boxes.push({ cx: ow + pierW * 0.5, cy: height * 0.5, cz: 0, sx: pierW, sy: height, sz: thickness });
  }
  const crown = springHeight + rise;
  if (height - crown > 0.02) {
    boxes.push({ cx: 0, cy: (crown + height) * 0.5, cz: 0, sx: openW, sy: height - crown, sz: thickness });
  }
  // Two haunch blocks so you cannot walk through the corners of the arch head.
  const haunch = rise * 0.42;
  boxes.push({ cx: -(ow - openW * 0.14), cy: crown - haunch * 0.5, cz: 0, sx: openW * 0.28, sy: haunch, sz: thickness });
  boxes.push({ cx: ow - openW * 0.14, cy: crown - haunch * 0.5, cz: 0, sx: openW * 0.28, sy: haunch, sz: thickness });

  return { geometry: finalizeGeometry(geo), boxes };
}

/**
 * Blast breach: an irregular hole punched straight through a wall panel. The
 * outline is a noisy polygon so no two breaches share a silhouette, and the
 * extrusion gives the ragged edge real depth you can see the far side through.
 */
export function breachedWall(
  width: number,
  height: number,
  thickness: number,
  holes: readonly { x: number; y: number; r: number; seed: number }[],
  openings: readonly Opening[] = [],
): WallResult {
  const hw = width * 0.5;
  const shape = new THREE.Shape();
  shape.moveTo(-hw, 0);
  shape.lineTo(hw, 0);
  shape.lineTo(hw, height);
  shape.lineTo(-hw, height);
  shape.closePath();

  const boxes: BoxSpec[] = [];

  for (const h of holes) {
    const rng = new Rng(h.seed);
    const path = new THREE.Path();
    const segs = 15;
    // Two-octave radial noise: a big lobe plus chipped edges.
    const lobe = rng.range(0, Math.PI * 2);
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      const r =
        h.r *
        (0.72 +
          0.3 * Math.sin(a * 2 + lobe) * 0.5 +
          0.28 * rng.next() +
          0.14 * Math.sin(a * 5 + lobe * 2.3));
      const px = h.x + Math.cos(a) * r * 1.12;
      const py = h.y + Math.sin(a) * r * 0.86;
      if (i === 0) path.moveTo(px, py);
      else path.lineTo(px, py);
    }
    path.closePath();
    shape.holes.push(path);
  }
  for (const o of openings) {
    const p = new THREE.Path();
    p.moveTo(o.x - o.w * 0.5, o.y);
    p.lineTo(o.x + o.w * 0.5, o.y);
    p.lineTo(o.x + o.w * 0.5, o.y + o.h);
    p.lineTo(o.x - o.w * 0.5, o.y + o.h);
    p.closePath();
    shape.holes.push(p);
  }

  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.02, thickness - 0.03),
    bevelEnabled: true,
    bevelSize: 0.015,
    bevelThickness: 0.015,
    bevelSegments: 1,
    curveSegments: 8,
    steps: 1,
  });
  geo.translate(0, 0, -thickness * 0.5);

  // Collision: cell-split the panel treating each hole as its AABB and each
  // opening as itself, so you can genuinely shoot and see through the breach.
  const asOpenings: Opening[] = openings.slice();
  for (const h of holes) {
    asOpenings.push({ x: h.x, y: Math.max(0, h.y - h.r * 0.78), w: h.r * 1.68, h: h.r * 1.56 });
  }
  for (const { rect, xs, ys } of slabRects(width, height, asOpenings, 0.06)) {
    const x0 = xs[rect.x0];
    const x1 = xs[rect.x1];
    const y0 = ys[rect.y0];
    const y1 = ys[rect.y1];
    boxes.push({
      cx: (x0 + x1) * 0.5,
      cy: (y0 + y1) * 0.5,
      cz: 0,
      sx: x1 - x0,
      sy: y1 - y0,
      sz: thickness,
    });
  }

  return { geometry: finalizeGeometry(geo), boxes };
}

// ---------------------------------------------------------------------------
// Architectural elements
// ---------------------------------------------------------------------------

/**
 * A staircase as a single swept sawtooth: one extrusion, correct bevel on every
 * nosing, a fraction of the triangles of stacked boxes.
 * Local space: base at y=0, first tread at z=0, ascending toward +z.
 */
export function stairsGeo(
  width: number,
  totalRise: number,
  totalRun: number,
  steps: number,
): WallResult {
  const rise = totalRise / steps;
  const run = totalRun / steps;
  const profile: THREE.Vector2[] = [new THREE.Vector2(0, 0)];
  for (let i = 0; i < steps; i++) {
    profile.push(new THREE.Vector2(i * run, (i + 1) * rise));
    profile.push(new THREE.Vector2((i + 1) * run, (i + 1) * rise));
  }
  profile.push(new THREE.Vector2(totalRun, totalRise - 0.02));
  profile.push(new THREE.Vector2(totalRun, -0.22));
  profile.push(new THREE.Vector2(0, -0.22));

  const geo = extrudeProfile(profile, width, { bevel: 0.014 });
  // Profile is in XY swept along Z; rotate so the run climbs +Z and the tread
  // width sweeps along X.
  geo.rotateY(-Math.PI * 0.5);

  const boxes: BoxSpec[] = [];
  for (let i = 0; i < steps; i++) {
    boxes.push({
      cx: 0,
      cy: (i + 1) * rise * 0.5,
      cz: i * run + run * 0.5,
      sx: width,
      sy: (i + 1) * rise,
      sz: run,
    });
  }
  return { geometry: geo, boxes };
}

export interface RailingOpts {
  postSpacing?: number;
  balusterSpacing?: number;
  bevel?: number;
}

/**
 * Balcony/stair railing: swept top and bottom rails, posts at intervals,
 * balusters between. Length runs along X, base at y=0.
 */
export function railingGeo(length: number, height = 1.02, opts: RailingOpts = {}): THREE.BufferGeometry {
  const postSpacing = opts.postSpacing ?? 1.5;
  const balusterSpacing = opts.balusterSpacing ?? 0.17;
  const bevel = opts.bevel ?? 0.006;
  const parts: THREE.BufferGeometry[] = [];

  const topRail = extrudeProfile(
    [
      new THREE.Vector2(-0.032, 0),
      new THREE.Vector2(0.032, 0),
      new THREE.Vector2(0.042, 0.03),
      new THREE.Vector2(0, 0.056),
      new THREE.Vector2(-0.042, 0.03),
    ],
    length,
    { bevel: 0.005 },
  );
  topRail.rotateY(-Math.PI * 0.5);
  topRail.translate(0, height - 0.056, 0);
  parts.push(topRail);

  const bottomRail = bevelBox(length, 0.05, 0.05, bevel);
  bottomRail.translate(0, 0.1, 0);
  parts.push(bottomRail);

  const nPosts = Math.max(2, Math.round(length / postSpacing) + 1);
  for (let i = 0; i < nPosts; i++) {
    const x = -length * 0.5 + (length * i) / (nPosts - 1);
    const p = bevelBox(0.062, height, 0.062, 0.009);
    p.translate(x, height * 0.5, 0);
    parts.push(p);
  }

  // Balusters are 22mm bars: their chamfer is under a pixel past two metres, so
  // they are the one place a plain box is the right call.
  const nBal = Math.max(1, Math.floor(length / balusterSpacing) - 1);
  for (let i = 1; i <= nBal; i++) {
    const x = -length * 0.5 + (length * i) / (nBal + 1);
    const b = plainBox(0.022, height - 0.2, 0.022);
    b.translate(x, 0.12 + (height - 0.2) * 0.5, 0);
    parts.push(b);
  }

  return finalizeGeometry(mergeAll(parts));
}

/**
 * Wall face with recessed panel courses. Each panel plane sits a few millimetres
 * off its neighbour so the sun grades across the elevation instead of hitting
 * one uniform value.
 */
export function panelledWall(
  width: number,
  height: number,
  thickness: number,
  cols: number,
  rows: number,
  seed: number,
): WallResult {
  const rng = new Rng(seed);
  const parts: THREE.BufferGeometry[] = [];
  const back = bevelBox(width, height, thickness * 0.6, 0.02);
  back.translate(0, height * 0.5, -thickness * 0.2);
  parts.push(back);

  const margin = 0.14;
  const cw = (width - margin * (cols + 1)) / cols;
  const ch = (height - margin * (rows + 1)) / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = -width * 0.5 + margin + cw * 0.5 + c * (cw + margin);
      const y = margin + ch * 0.5 + r * (ch + margin);
      const d = thickness * 0.4 * rng.range(0.55, 1.0);
      const g = bevelBox(cw, ch, d, 0.024);
      g.translate(x + rng.jitter(0.01), y + rng.jitter(0.01), thickness * 0.5 - d * 0.5);
      parts.push(g);
    }
  }
  // Frame between the panels, flush with the outer face.
  const frame = bevelBox(width, height, thickness * 0.42, 0.02);
  frame.translate(0, height * 0.5, thickness * 0.29 - thickness * 0.21);
  parts.push(frame);

  return {
    geometry: finalizeGeometry(mergeAll(parts)),
    boxes: [{ cx: 0, cy: height * 0.5, cz: 0, sx: width, sy: height, sz: thickness }],
  };
}

/** Window joinery: outer frame, mullion, transom. Sits in the XY plane. */
export function windowFrameGeo(width: number, height: number, depth = 0.07): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const t = 0.055;
  const sides: [number, number, number, number][] = [
    [0, height * 0.5 - t * 0.5, width, t],
    [0, -height * 0.5 + t * 0.5, width, t],
    [-width * 0.5 + t * 0.5, 0, t, height - t * 2],
    [width * 0.5 - t * 0.5, 0, t, height - t * 2],
  ];
  for (const [x, y, w, h] of sides) {
    const g = bevelBox(w, h, depth, 0.008);
    g.translate(x, y, 0);
    parts.push(g);
  }
  // Mullion and transom are 3.8cm members read at a couple of pixels through a
  // dirty pane; the chamfer on them is not resolvable and they are the single
  // most repeated pair of boxes on the map.
  const mullion = plainBox(0.038, height - t * 2, depth * 0.82);
  parts.push(mullion);
  const transom = plainBox(width - t * 2, 0.036, depth * 0.82);
  transom.translate(0, height * 0.18, 0);
  parts.push(transom);

  return finalizeGeometry(mergeAll(parts));
}

/** Louvred timber shutter. `open` swings it about its hinge edge. */
export function shutterGeo(width: number, height: number, slats = 6): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const styleW = 0.055;
  // Stiles keep their chamfer: they are the vertical members the low sun rakes
  // across and the ones you read the shutter's thickness from. The two rails
  // are horizontal, always in the stile's own shadow, and there are two of them
  // on every leaf of every shutter on the map — much the cheapest place to
  // spend a hundred thousand triangles is not there.
  const stiles: [number, number, number, number][] = [
    [-width * 0.5 + styleW * 0.5, 0, styleW, height],
    [width * 0.5 - styleW * 0.5, 0, styleW, height],
  ];
  for (const [x, y, w, h] of stiles) {
    const g = bevelBox(w, h, 0.034, 0.005);
    g.translate(x, y, 0);
    parts.push(g);
  }
  for (const y of [height * 0.5 - styleW * 0.5, -height * 0.5 + styleW * 0.5]) {
    const g = plainBox(width - styleW * 2, styleW, 0.034);
    g.translate(0, y, 0);
    parts.push(g);
  }
  const inner = height - styleW * 2.4;
  const pitch = inner / slats;
  for (let i = 0; i < slats; i++) {
    const y = -inner * 0.5 + pitch * (i + 0.5);
    const g = plainBox(width - styleW * 2, pitch * 0.72, 0.014);
    g.rotateX(-0.42);
    g.translate(0, y, 0.004);
    parts.push(g);
  }
  return finalizeGeometry(mergeAll(parts));
}

/** Panelled timber door, four raised panels. */
export function doorGeo(width: number, height: number): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const slab = bevelBox(width, height, 0.05, 0.008);
  parts.push(slab);
  const mx = width * 0.22;
  const my = height * 0.2;
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      const g = bevelBox(width * 0.34, height * 0.3, 0.022, 0.008);
      g.translate(sx * mx, sy * my, 0.03);
      parts.push(g);
    }
  }
  const handle = finalizeGeometry(new THREE.CylinderGeometry(0.017, 0.017, 0.09, 8));
  handle.rotateX(Math.PI * 0.5);
  handle.translate(width * 0.36, 0, 0.06);
  parts.push(handle);
  return finalizeGeometry(mergeAll(parts));
}

/**
 * Scattered surface relief on a plane facing +Z. Panels of varying depth and
 * size; the point is to stop a 20m facade reading as one flat rectangle at
 * grazing angles.
 */
export function greebleFace(
  width: number,
  height: number,
  count: number,
  seed: number,
  maxDepth = 0.06,
): THREE.BufferGeometry {
  const rng = new Rng(seed);
  const parts: THREE.BufferGeometry[] = [];
  for (let i = 0; i < count; i++) {
    const w = rng.range(0.25, Math.min(1.9, width * 0.32));
    const h = rng.range(0.2, Math.min(1.4, height * 0.3));
    const d = rng.range(maxDepth * 0.35, maxDepth);
    const g = bevelBox(w, h, d, Math.min(0.014, d * 0.4));
    g.translate(
      rng.range(-width * 0.5 + w * 0.6, width * 0.5 - w * 0.6),
      rng.range(h * 0.6, height - h * 0.6),
      d * 0.5,
    );
    parts.push(g);
  }
  return finalizeGeometry(mergeAll(parts));
}

// ---------------------------------------------------------------------------
// Debris
// ---------------------------------------------------------------------------

/**
 * A cone of broken masonry. Density falls off with radius and pieces sink into
 * the pile, so the silhouette is a real talus slope rather than a heap of
 * floating cubes.
 */
export interface RubbleOpts {
  bevelScale?: number;
  /**
   * Direction the blast travelled, in the cone's local XZ. When set, the pile
   * stops being a neat radial heap: mass is thrown downrange, the throw fans
   * out and thins with distance, and the pieces closest to the blast point are
   * the largest. This is the difference between "debris" and "a tidy pile of
   * debris someone swept up".
   */
  throwX?: number;
  throwZ?: number;
  /** How far downrange the throw reaches, as a multiple of `radius`. */
  reach?: number;
  /** Half-angle of the fan, radians. */
  spread?: number;
}

export function rubbleCone(
  radius: number,
  height: number,
  count: number,
  seed: number,
  opts: RubbleOpts | number = {},
): THREE.BufferGeometry {
  const o: RubbleOpts = typeof opts === 'number' ? { bevelScale: opts } : opts;
  const bevelScale = o.bevelScale ?? 0.14;
  const tx = o.throwX ?? 0;
  const tz = o.throwZ ?? 0;
  const throwLen = Math.hypot(tx, tz);
  const dirA = throwLen > 1e-4 ? Math.atan2(tz, tx) : 0;
  const reach = (o.reach ?? 2.6) * radius;
  const spread = o.spread ?? 0.85;

  const rng = new Rng(seed);
  const parts: THREE.BufferGeometry[] = [];
  for (let i = 0; i < count; i++) {
    let x: number;
    let z: number;
    let r: number;
    let s: number;
    if (throwLen > 1e-4 && rng.chance(0.62)) {
      // Downrange throw: density falls as t^-1 and piece size with it, so the
      // cone tapers instead of ending on a hard edge.
      const t = Math.pow(rng.next(), 0.55);
      const a = dirA + rng.jitter(spread) * (0.35 + t * 0.9);
      const d = radius * 0.35 + t * reach;
      x = Math.cos(a) * d;
      z = Math.sin(a) * d;
      r = d;
      s = rng.range(0.07, 0.3) * (1 - t * 0.62);
    } else {
      // sqrt for uniform area density, then biased inward so the peak is dense.
      r = radius * Math.pow(rng.next(), 0.62);
      const a = rng.range(0, Math.PI * 2);
      x = Math.cos(a) * r;
      z = Math.sin(a) * r;
      s = rng.range(0.09, 0.34) * (1 - (r / radius) * 0.35);
    }
    const surface = height * Math.max(0, 1 - r / radius) * rng.range(0.55, 1.0);
    const g =
      bevelScale > 0
        ? bevelBox(s * rng.range(0.7, 1.6), s * rng.range(0.5, 1.1), s * rng.range(0.7, 1.5), s * bevelScale)
        : plainBox(s * rng.range(0.7, 1.6), s * rng.range(0.5, 1.1), s * rng.range(0.7, 1.5));
    g.rotateY(rng.range(0, Math.PI * 2));
    g.rotateX(rng.jitter(0.7));
    g.rotateZ(rng.jitter(0.7));
    g.translate(x, surface + s * 0.2, z);
    parts.push(g);
  }
  return finalizeGeometry(mergeAll(parts));
}

/**
 * Irregular plate for a patch of render that has come off a wall, exposing the
 * brick underneath. Sits a few millimetres proud with a chamfered edge, which
 * is what a spall reads as at any distance you can see it from — and unlike a
 * decal it catches the sun on its lip.
 */
export function spallPatch(width: number, height: number, depth: number, seed: number): THREE.BufferGeometry {
  const rng = new Rng(seed);
  const shape = new THREE.Shape();
  const segs = 13;
  const lobe = rng.range(0, Math.PI * 2);
  for (let i = 0; i <= segs; i++) {
    const a = (i / segs) * Math.PI * 2;
    const r = 0.5 * (0.68 + 0.36 * rng.next() + 0.18 * Math.sin(a * 3 + lobe));
    const px = Math.cos(a) * r * width;
    const py = Math.sin(a) * r * height;
    if (i === 0) shape.moveTo(px, py);
    else shape.lineTo(px, py);
  }
  shape.closePath();
  const bevel = depth * 0.45;
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelSize: bevel,
    bevelThickness: bevel,
    bevelSegments: 1,
    curveSegments: 4,
    steps: 1,
  });
  geo.translate(0, 0, -depth);
  return finalizeGeometry(geo);
}

/**
 * A sheet draped over something: sags between its corners, ripples across, and
 * lifts slightly at one edge. Never a flat quad.
 */
export function drapedSheet(
  width: number,
  depth: number,
  sag: number,
  seed: number,
  segs = 6,
): THREE.BufferGeometry {
  const rng = new Rng(seed);
  const geo = new THREE.PlaneGeometry(width, depth, segs, segs);
  geo.rotateX(-Math.PI * 0.5);
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const phase = rng.range(0, 6.28);
  for (let i = 0; i < pos.count; i++) {
    const u = pos.getX(i) / width;
    const v = pos.getZ(i) / depth;
    const edge = Math.min(1, (0.5 - Math.abs(u)) * 4) * Math.min(1, (0.5 - Math.abs(v)) * 4);
    const drop = -sag * (1 - edge) + Math.sin(u * 9 + phase) * 0.018 + Math.sin(v * 7.3 - phase) * 0.014;
    pos.setY(i, drop * (edge > 0 ? 1 : 1.25));
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return finalizeGeometry(geo);
}

/** Bent reinforcement bar poking out of broken concrete. */
export function rebarGeo(length: number, seed: number, radius = 0.011): THREE.BufferGeometry {
  const rng = new Rng(seed);
  const pts: THREE.Vector3[] = [];
  const segs = 4;
  let dir = new THREE.Vector3(rng.jitter(0.35), 1, rng.jitter(0.35)).normalize();
  let p = new THREE.Vector3(0, 0, 0);
  pts.push(p.clone());
  for (let i = 0; i < segs; i++) {
    dir = dir.clone().add(new THREE.Vector3(rng.jitter(0.5), rng.jitter(0.35), rng.jitter(0.5))).normalize();
    p = p.clone().addScaledVector(dir, length / segs);
    pts.push(p.clone());
  }
  const curve = new THREE.CatmullRomCurve3(pts);
  return finalizeGeometry(new THREE.TubeGeometry(curve, 5, radius, 4, false));
}

/** Cylinder helper with the attribute contract already applied. */
export function cylinderGeo(
  rTop: number,
  rBottom: number,
  height: number,
  radial = 12,
  open = false,
): THREE.BufferGeometry {
  return finalizeGeometry(new THREE.CylinderGeometry(rTop, rBottom, height, radial, 1, open));
}

export function torusGeo(radius: number, tube: number, radial = 8, tubular = 16): THREE.BufferGeometry {
  return finalizeGeometry(new THREE.TorusGeometry(radius, tube, radial, tubular));
}

/** Straight pipe run with wall brackets — downpipes, conduit, handrails. */
export function pipeRun(length: number, radius: number, brackets: number): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  const body = cylinderGeo(radius, radius, length, 10);
  parts.push(body);
  for (let i = 0; i < brackets; i++) {
    const y = -length * 0.5 + (length * (i + 0.5)) / brackets;
    const b = bevelBox(radius * 2.9, 0.035, radius * 2.6, 0.006);
    b.translate(0, y, -radius * 0.9);
    parts.push(b);
  }
  // Collar joints break the pipe up so it is not one uniform tube.
  for (let i = 1; i < brackets; i++) {
    const y = -length * 0.5 + (length * i) / brackets + 0.12;
    const c = cylinderGeo(radius * 1.18, radius * 1.18, 0.055, 10);
    c.translate(0, y, 0);
    parts.push(c);
  }
  return finalizeGeometry(mergeAll(parts));
}

// ---------------------------------------------------------------------------
// Accumulators
// ---------------------------------------------------------------------------

/** Collects geometry into one merged buffer. Takes ownership of what it is given. */
export class GeoBuilder {
  private readonly parts: THREE.BufferGeometry[] = [];
  private tris = 0;

  get triangles(): number {
    return this.tris;
  }

  get empty(): boolean {
    return this.parts.length === 0;
  }

  /** Consumes `geo`: applies `matrix` in place and keeps the buffer. */
  add(geo: THREE.BufferGeometry, matrix?: THREE.Matrix4): this {
    if (matrix) geo.applyMatrix4(matrix);
    this.tris += (geo.getAttribute('position') as THREE.BufferAttribute).count / 3;
    this.parts.push(geo);
    return this;
  }

  /** Copies `geo` first, so a template can be reused. */
  addCopy(geo: THREE.BufferGeometry, matrix?: THREE.Matrix4): this {
    return this.add(geo.clone(), matrix);
  }

  build(): THREE.BufferGeometry | null {
    if (this.parts.length === 0) return null;
    const merged = mergeAll(this.parts);
    this.parts.length = 0;
    return finalizeGeometry(merged);
  }

  dispose(): void {
    for (const p of this.parts) p.dispose();
    this.parts.length = 0;
  }
}

export interface CollisionSolid {
  surface: SurfaceKind;
  geometry: THREE.BufferGeometry;
  boxes: { center: THREE.Vector3; half: THREE.Vector3 }[];
}

const _mtmp = new THREE.Matrix4();
const _ptmp = new THREE.Vector3();

/**
 * Collision authoring. Everything solid registers a box here; the level then
 * builds one invisible mesh per surface kind and hands those to the physics
 * world. Visual meshes are never registered, which is why bullet queries stay
 * cheap and never catch on a chamfer.
 */
export class ColliderSet {
  private readonly bySurface = new Map<SurfaceKind, THREE.BufferGeometry[]>();
  private readonly aabbs: { center: THREE.Vector3; half: THREE.Vector3 }[] = [];
  private readonly bySurfaceBoxes = new Map<SurfaceKind, { center: THREE.Vector3; half: THREE.Vector3 }[]>();
  private count = 0;

  get boxCount(): number {
    return this.count;
  }

  /** All boxes added so far, for the occlusion bake and cover-point search. */
  get boxes(): readonly { center: THREE.Vector3; half: THREE.Vector3 }[] {
    return this.aabbs;
  }

  addBox(surface: SurfaceKind, spec: BoxSpec, matrix?: THREE.Matrix4): void {
    if (spec.sx <= 1e-3 || spec.sy <= 1e-3 || spec.sz <= 1e-3) return;
    const g = new THREE.BoxGeometry(spec.sx, spec.sy, spec.sz);
    _mtmp.makeTranslation(spec.cx, spec.cy, spec.cz);
    g.applyMatrix4(_mtmp);
    if (matrix) g.applyMatrix4(matrix);
    g.deleteAttribute('uv');
    g.deleteAttribute('normal');

    let list = this.bySurface.get(surface);
    if (!list) {
      list = [];
      this.bySurface.set(surface, list);
    }
    list.push(g);
    this.count++;

    g.computeBoundingBox();
    const bb = g.boundingBox;
    if (bb) {
      const rec = {
        center: bb.getCenter(new THREE.Vector3()),
        half: bb.getSize(new THREE.Vector3()).multiplyScalar(0.5),
      };
      this.aabbs.push(rec);
      let sl = this.bySurfaceBoxes.get(surface);
      if (!sl) {
        sl = [];
        this.bySurfaceBoxes.set(surface, sl);
      }
      sl.push(rec);
    }
  }

  addBoxes(surface: SurfaceKind, specs: readonly BoxSpec[], matrix?: THREE.Matrix4): void {
    for (const s of specs) this.addBox(surface, s, matrix);
  }

  build(): CollisionSolid[] {
    const out: CollisionSolid[] = [];
    for (const [surface, geos] of this.bySurface) {
      if (geos.length === 0) continue;
      const merged = geos.length === 1 ? geos[0] : mergeGeometries(geos, false);
      if (geos.length > 1) for (const g of geos) g.dispose();
      out.push({ surface, geometry: merged, boxes: this.bySurfaceBoxes.get(surface) ?? [] });
    }
    this.bySurface.clear();
    return out;
  }
}

// ---------------------------------------------------------------------------
// Baked occlusion
// ---------------------------------------------------------------------------

/** 14 directions: 6 face + 8 corner, normalized. Cheap, isotropic enough. */
const AO_DIRS: THREE.Vector3[] = (() => {
  const d: THREE.Vector3[] = [];
  d.push(new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0));
  d.push(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0));
  d.push(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1));
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    d.push(new THREE.Vector3(sx, sy, sz).normalize());
  }
  return d;
})();

const AO_STEPS = [0.42, 1.05, 2.1];

export interface AoOptions {
  /** How much the occupancy term can darken, 0..1. */
  strength?: number;
  /** Grime that creeps up from the ground, in metres. */
  groundHeight?: number;
  groundStrength?: number;
  dirtColor?: THREE.ColorRepresentation;
  seed?: number;
}

/**
 * Bakes ambient occlusion and ground grime into vertex colours from the level's
 * own collision volumes.
 *
 * Real hemisphere sampling against a voxelised occupancy grid — corners darken,
 * the undersides of balconies and arcades darken, and the base of every wall
 * picks up a dirt tint. It costs one pass at load and it is the single cheapest
 * thing that stops geometry looking like it is floating on the ground plane.
 */
export class OcclusionBaker {
  private readonly cell: number;
  private readonly inv: number;
  private readonly min = new THREE.Vector3();
  private readonly nx: number;
  private readonly ny: number;
  private readonly nz: number;
  private readonly grid: Uint8Array;
  private readonly cache = new Map<number, number>();

  constructor(bounds: THREE.Box3, cell = 0.55) {
    this.cell = cell;
    this.inv = 1 / cell;
    this.min.copy(bounds.min);
    const size = bounds.getSize(new THREE.Vector3());
    this.nx = Math.max(1, Math.ceil(size.x * this.inv));
    this.ny = Math.max(1, Math.ceil(size.y * this.inv));
    this.nz = Math.max(1, Math.ceil(size.z * this.inv));
    this.grid = new Uint8Array(this.nx * this.ny * this.nz);
  }

  /**
   * Rasterises by cell *centre*, not by overlap. Overlap rasterisation inflates
   * every volume by up to a full cell, and a floor slab inflated upward buries
   * everything standing on it — the whole map comes out uniformly black. When
   * centre sampling misses entirely (a 0.4m wall in a 0.6m grid) the box's own
   * centre cell is marked so thin walls still occlude.
   */
  addBox(center: THREE.Vector3, half: THREE.Vector3): void {
    const span = (
      lo: number,
      hi: number,
      origin: number,
      count: number,
      mid: number,
    ): [number, number] => {
      let a = Math.ceil((lo - origin) * this.inv - 0.5);
      let b = Math.floor((hi - origin) * this.inv - 0.5);
      if (b < a) {
        const c = Math.floor((mid - origin) * this.inv);
        a = c;
        b = c;
      }
      return [Math.max(0, a), Math.min(count - 1, b)];
    };

    const [x0, x1] = span(center.x - half.x, center.x + half.x, this.min.x, this.nx, center.x);
    const [y0, y1] = span(center.y - half.y, center.y + half.y, this.min.y, this.ny, center.y);
    const [z0, z1] = span(center.z - half.z, center.z + half.z, this.min.z, this.nz, center.z);

    for (let y = y0; y <= y1; y++) {
      for (let z = z0; z <= z1; z++) {
        const row = (y * this.nz + z) * this.nx;
        for (let x = x0; x <= x1; x++) this.grid[row + x] = 1;
      }
    }
  }

  /** Cell index for a world point, or -1 outside the grid. */
  private cellAt(x: number, y: number, z: number): number {
    const ix = Math.floor((x - this.min.x) * this.inv);
    if (ix < 0 || ix >= this.nx) return -1;
    const iy = Math.floor((y - this.min.y) * this.inv);
    if (iy < 0 || iy >= this.ny) return -1;
    const iz = Math.floor((z - this.min.z) * this.inv);
    if (iz < 0 || iz >= this.nz) return -1;
    return (iy * this.nz + iz) * this.nx + ix;
  }

  private sample(px: number, py: number, pz: number, nx: number, ny: number, nz: number): number {
    let occ = 0;
    let wsum = 1e-4;
    const ox = px + nx * 0.18;
    const oy = py + ny * 0.18;
    const oz = pz + nz * 0.18;
    // Taps landing back in the surface's own cell are self-occlusion, not
    // occlusion, and must not count.
    const home = this.cellAt(ox, oy, oz);
    for (let i = 0; i < AO_DIRS.length; i++) {
      const d = AO_DIRS[i];
      const c = d.x * nx + d.y * ny + d.z * nz;
      if (c <= 0.12) continue;
      wsum += c;
      for (let s = 0; s < AO_STEPS.length; s++) {
        const t = AO_STEPS[s];
        const cell = this.cellAt(ox + d.x * t, oy + d.y * t, oz + d.z * t);
        if (cell < 0 || cell === home) continue;
        if (this.grid[cell] === 1) {
          occ += c * (1 - s * 0.33);
          break;
        }
      }
    }
    return occ / wsum;
  }

  /**
   * Multiplies the geometry's vertex colours by the baked term. `geo` must be
   * in world space. Results are memoised on a 12cm/normal-octant key because
   * non-indexed geometry repeats every corner four to six times.
   */
  shade(geo: THREE.BufferGeometry, opts: AoOptions = {}): void {
    const strength = opts.strength ?? 0.72;
    const groundH = opts.groundHeight ?? 0.75;
    const groundS = opts.groundStrength ?? 0.5;
    _col.set(opts.dirtColor ?? 0x6d5f4a);
    const seed = opts.seed ?? 1;

    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    const nor = geo.getAttribute('normal') as THREE.BufferAttribute;
    const col = geo.getAttribute('color') as THREE.BufferAttribute;
    const cache = this.cache;

    for (let i = 0; i < pos.count; i++) {
      const px = pos.getX(i);
      const py = pos.getY(i);
      const pz = pos.getZ(i);
      const nx = nor.getX(i);
      const ny = nor.getY(i);
      const nz = nor.getZ(i);

      // 12cm spatial key + 3-bit normal octant. Collisions here are invisible.
      const key =
        ((((px * 8.3) | 0) & 1023) << 20) ^
        ((((py * 8.3) | 0) & 1023) << 10) ^
        (((pz * 8.3) | 0) & 1023) ^
        (((nx > 0 ? 1 : 0) | (ny > 0 ? 2 : 0) | (nz > 0 ? 4 : 0)) << 30);

      let ao = cache.get(key) ?? -1;
      if (ao < 0) {
        ao = 1 - strength * this.sample(px, py, pz, nx, ny, nz);
        cache.set(key, ao);
      }

      // Ground grime: vertical surfaces pick it up, floors much less.
      const up = Math.max(0, ny);
      const h = Math.max(0, py);
      const noise = 0.72 + 0.28 * fract(Math.sin((px * 12.9898 + pz * 78.233 + seed) * 1.0) * 43758.5453);
      const g = groundS * Math.exp(-h / groundH) * (1 - up * 0.72) * noise;

      const r = ao * (1 - g + g * _col.r);
      const gg = ao * (1 - g + g * _col.g);
      const b = ao * (1 - g + g * _col.b);
      col.setXYZ(i, col.getX(i) * r, col.getY(i) * gg, col.getZ(i) * b);
    }
    col.needsUpdate = true;
  }

  dispose(): void {
    this.cache.clear();
  }
}

function fract(x: number): number {
  return x - Math.floor(x);
}

// ---------------------------------------------------------------------------
// Contact occlusion
// ---------------------------------------------------------------------------

export interface ContactOptions {
  /** Peak darkening where a horizontal surface runs into something vertical. */
  ground?: number;
  /** Peak darkening on a horizontal surface that has something over it. */
  overhead?: number;
  /** Flat darkening on downward-facing surfaces: soffits, sills, slab undersides. */
  soffit?: number;
  /** Colour the occluded vertices are pulled toward. */
  tint?: THREE.ColorRepresentation;
  /** Hard ceiling on total darkening, so nothing crushes to black. */
  max?: number;
}

/**
 * The contact shadow, baked into vertex colour from the level's own collision
 * volumes.
 *
 * `OcclusionBaker` samples a 3D hemisphere and is good at rooms and corners, but
 * it is voxelised at 0.6m and a wall's base shares a cell with the road it
 * stands on — the one place occlusion matters most is exactly the place a
 * coarse grid throws away as self-occlusion. So this does it in 2D instead: a
 * chamfer distance transform of every standing volume's footprint gives, for
 * any point on the ground, the metres to the nearest thing sticking up out of
 * it. A tight double-exponential off that distance is the dark line where a
 * building meets the street, and it is the difference between geometry sitting
 * in a scene and geometry pasted onto it.
 *
 * A second grid records the underside height of anything overhead, which is
 * what darkens the floor of an arcade, the ground under a balcony, and the
 * inside of every room.
 */
export class ContactField {
  private readonly cell: number;
  private readonly inv: number;
  private readonly ox: number;
  private readonly oz: number;
  private readonly nx: number;
  private readonly nz: number;
  private readonly dist: Float32Array;
  private readonly ceiling: Float32Array;

  constructor(bounds: THREE.Box3, cell = 0.4) {
    this.cell = cell;
    this.inv = 1 / cell;
    this.ox = bounds.min.x;
    this.oz = bounds.min.z;
    this.nx = Math.max(1, Math.ceil((bounds.max.x - bounds.min.x) * this.inv));
    this.nz = Math.max(1, Math.ceil((bounds.max.z - bounds.min.z) * this.inv));
    this.dist = new Float32Array(this.nx * this.nz).fill(1e9);
    this.ceiling = new Float32Array(this.nx * this.nz).fill(1e9);
  }

  /**
   * Classifies one collision volume. Ground slabs, floor plates and roof decks
   * are all boxes too, and if they were treated as occluders the distance field
   * would be zero everywhere; the height tests are what separate "stands on the
   * ground" from "is the ground".
   */
  addOccluder(center: THREE.Vector3, half: THREE.Vector3): void {
    const bottom = center.y - half.y;
    const top = center.y + half.y;
    const stands = top >= 0.55 && half.y * 2 >= 0.45 && bottom <= 1.4;
    const overhead = bottom >= 1.8;
    if (!stands && !overhead) return;

    const x0 = Math.max(0, Math.floor((center.x - half.x - this.ox) * this.inv));
    const x1 = Math.min(this.nx - 1, Math.floor((center.x + half.x - this.ox) * this.inv));
    const z0 = Math.max(0, Math.floor((center.z - half.z - this.oz) * this.inv));
    const z1 = Math.min(this.nz - 1, Math.floor((center.z + half.z - this.oz) * this.inv));
    if (x1 < x0 || z1 < z0) return;

    for (let z = z0; z <= z1; z++) {
      const row = z * this.nx;
      for (let x = x0; x <= x1; x++) {
        if (stands) this.dist[row + x] = 0;
        if (overhead && bottom < this.ceiling[row + x]) this.ceiling[row + x] = bottom;
      }
    }
  }

  /** Two-pass chamfer distance transform. Call once, after every occluder. */
  finalize(): void {
    const { nx, nz, dist, cell } = this;
    const D = 1;
    const Q = Math.SQRT2;
    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const i = z * nx + x;
        let d = dist[i];
        if (d === 0) continue;
        if (x > 0) d = Math.min(d, dist[i - 1] + D);
        if (z > 0) {
          d = Math.min(d, dist[i - nx] + D);
          if (x > 0) d = Math.min(d, dist[i - nx - 1] + Q);
          if (x < nx - 1) d = Math.min(d, dist[i - nx + 1] + Q);
        }
        dist[i] = d;
      }
    }
    for (let z = nz - 1; z >= 0; z--) {
      for (let x = nx - 1; x >= 0; x--) {
        const i = z * nx + x;
        let d = dist[i];
        if (d === 0) continue;
        if (x < nx - 1) d = Math.min(d, dist[i + 1] + D);
        if (z < nz - 1) {
          d = Math.min(d, dist[i + nx] + D);
          if (x < nx - 1) d = Math.min(d, dist[i + nx + 1] + Q);
          if (x > 0) d = Math.min(d, dist[i + nx - 1] + Q);
        }
        dist[i] = d;
      }
    }
    for (let i = 0; i < dist.length; i++) dist[i] = Math.min(dist[i] * cell, 40);
  }

  /** Bilinear metres-to-nearest-standing-volume. Clamped at the grid edge. */
  distanceAt(x: number, z: number): number {
    const fx = Math.min(this.nx - 1.001, Math.max(0, (x - this.ox) * this.inv - 0.5));
    const fz = Math.min(this.nz - 1.001, Math.max(0, (z - this.oz) * this.inv - 0.5));
    const ix = fx | 0;
    const iz = fz | 0;
    const tx = fx - ix;
    const tz = fz - iz;
    const r0 = iz * this.nx + ix;
    const r1 = r0 + this.nx;
    const a = this.dist[r0] * (1 - tx) + this.dist[r0 + 1] * tx;
    const b = this.dist[r1] * (1 - tx) + this.dist[r1 + 1] * tx;
    return a * (1 - tz) + b * tz;
  }

  /** Underside height of the nearest thing overhead, or 1e9 for open sky. */
  ceilingAt(x: number, z: number): number {
    const ix = Math.floor((x - this.ox) * this.inv);
    const iz = Math.floor((z - this.oz) * this.inv);
    if (ix < 0 || ix >= this.nx || iz < 0 || iz >= this.nz) return 1e9;
    return this.ceiling[iz * this.nx + ix];
  }

  /**
   * Multiplies world-space geometry's vertex colours by the contact term.
   *
   * Only horizontal and downward-facing surfaces are touched. Vertical faces are
   * deliberately left alone: a wall panel is one box with vertices only at its
   * top and bottom, so a per-vertex base term would spread a 40cm contact line
   * across a whole storey. Walls get their height gradient in the shader, where
   * it is evaluated per pixel.
   */
  shade(geo: THREE.BufferGeometry, opts: ContactOptions = {}): void {
    const ground = opts.ground ?? 0.52;
    const overhead = opts.overhead ?? 0.34;
    const soffit = opts.soffit ?? 0.3;
    const cap = opts.max ?? 0.8;
    _col.set(opts.tint ?? 0x3b3229);

    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    const nor = geo.getAttribute('normal') as THREE.BufferAttribute;
    const col = geo.getAttribute('color') as THREE.BufferAttribute;

    for (let i = 0; i < pos.count; i++) {
      const ny = nor.getY(i);
      const up = ny > 0 ? ny : 0;
      const down = ny < 0 ? -ny : 0;
      if (up < 0.12 && down < 0.15) continue;

      const px = pos.getX(i);
      const py = pos.getY(i);
      const pz = pos.getZ(i);
      let k = 0;

      if (up >= 0.12) {
        const d = this.distanceAt(px, pz);
        // Two lobes: a tight one that is the actual contact line, a wide one
        // that is the ambient the wall steals from the ground beside it.
        const c = 0.55 * Math.exp(-d * 2.6) + 0.45 * Math.exp(-d * 0.62);
        k += up * ground * c;

        const ceil = this.ceilingAt(px, pz);
        const gap = ceil - py;
        if (gap > 0.25 && gap < 60) {
          const t = Math.min(1, Math.max(0, (gap - 0.5) / 5.0));
          k += up * overhead * (1 - t * t * (3 - 2 * t));
        }
      }
      if (down >= 0.15) k += down * soffit;

      if (k <= 0.002) continue;
      if (k > cap) k = cap;
      col.setXYZ(
        i,
        col.getX(i) * (1 - k + k * _col.r),
        col.getY(i) * (1 - k + k * _col.g),
        col.getZ(i) * (1 - k + k * _col.b),
      );
    }
    col.needsUpdate = true;
  }
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

export function matrixOf(
  x: number,
  y: number,
  z: number,
  yaw = 0,
  pitch = 0,
  roll = 0,
  scale = 1,
): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));
  m.compose(_ptmp.set(x, y, z), q, _v0.set(scale, scale, scale));
  return m;
}

/** Uniformly-scaled placement matrix, YXZ order to match the camera rig. */
export function placement(x: number, y: number, z: number, yaw = 0, scale = 1): THREE.Matrix4 {
  return matrixOf(x, y, z, yaw, 0, 0, scale);
}
