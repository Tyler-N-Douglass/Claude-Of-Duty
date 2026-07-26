/**
 * PhysicsSystem — the single source of truth for "what is solid".
 *
 * Three responsibilities:
 *   1. Static world queries (ray / capsule sweep / overlap / ground / LOS) against
 *      a baked triangle soup in a uniform grid (see Broadphase.ts).
 *   2. Character hitboxes: oriented boxes that follow bone world matrices, so a
 *      bullet can report *which* box it struck and headshots work.
 *   3. Rigid-body debris: shell casings, gibs and rubble, integrated at the fixed
 *      120Hz step with real impulse response and a sleep threshold.
 *
 * Conventions this system assumes and every consumer must match:
 *   - A capsule position is the point *between the character's feet* (base), not
 *     the centre. `height` is the full standing height, `radius` the cylinder
 *     radius; the internal segment therefore runs from y+radius to y+height-radius.
 *   - RayHit.normal is the outward surface normal: for an entry hit it faces the
 *     shooter, for an exit hit (raycastAll only) it faces along the bullet.
 *   - Statics are baked at addStatic time. Moving a registered object afterwards
 *     does not move its collision; remove and re-add it.
 */
import * as THREE from 'three';
import {
  RayMask,
  UNITS,
  type CapsuleSweepHit,
  type CharacterHitbox,
  type FrameTime,
  type GameContext,
  type HitboxKind,
  type PhysicsWorld,
  type RayHit,
  type SurfaceKind,
  type System,
} from '../core/Contracts';
import {
  CST_SEG,
  CST_TRI,
  TriangleGrid,
  closestPointTriangle,
  closestSegmentTriangle,
  extractTriangles,
  rayTriangle,
  surfaceFromIndex,
} from './Broadphase';

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

const DEFAULT_RAY_MASK = RayMask.World | RayMask.Props | RayMask.Characters;

/** Gap left between a swept capsule and the surface it stops against. */
const SWEEP_SKIN = 0.005;
/**
 * Overlap shallower than this does not stop a sweep. A character standing on the
 * floor is permanently in sub-millimetre contact with it (float32 vertices alone
 * guarantee it); without this tolerance every lateral move would report a hit at
 * t=0 against the ground and the player would be welded in place.
 */
const PENETRATION_SLOP = 0.0015;
/** Sub-step length as a fraction of capsule radius during a sweep. */
const SWEEP_STEP_FRACTION = 0.6;
const SWEEP_MAX_STEPS = 32;
/** Bisection passes used to refine the contact fraction. */
const SWEEP_REFINE = 6;
/** Passes of sequential contact resolution when pushing a capsule out. */
const DEPEN_PASSES = 5;

/** Extra distance raycastAll probes past maxDistance purely to find exit faces. */
const THICKNESS_PROBE = 6;
/** Records a single raycastAll may collect before it stops adding more. */
const MAX_RAY_RECORDS = 192;

/**
 * Fallback material thickness in metres, used by `raycast` (which does not walk
 * through geometry) and by `raycastAll` when a surface has no back face inside
 * the probe distance — i.e. single-sided level geometry. Numbers are the real
 * construction thickness of the thing the surface represents.
 */
const SURFACE_THICKNESS: Record<SurfaceKind, number> = {
  concrete: 0.2,
  metal: 0.012,
  wood: 0.045,
  dirt: 0.6,
  sand: 0.6,
  glass: 0.006,
  water: 1.0,
  flesh: 0.28,
  foliage: 0.04,
  plaster: 0.014,
  rubber: 0.02,
  fabric: 0.004,
};

/** Debris: below this normal speed a bounce is fully absorbed. */
const BOUNCE_DEAD_SPEED = 0.16;
/** Impacts slower than this scale restitution down, so debris settles instead of buzzing. */
const BOUNCE_SOFT_SPEED = 0.8;
const SLEEP_LINEAR = 0.09;
/**
 * Angular sleep is judged by *surface* speed, not rad/s: a 6mm casing turning at
 * 4 rad/s is moving its skin 2.4cm/s and reads as stopped, while a 20cm gib at
 * the same rate is visibly spinning. The absolute cap keeps a fast spin awake
 * however small the body is.
 */
const SLEEP_SURFACE_SPEED = 0.06;
const SLEEP_ANGULAR_ABS = 4.0;
const SLEEP_TIME = 0.3;
/** A body may only fall asleep if it touched something this recently. */
const SLEEP_CONTACT_GRACE = 0.1;
/** Inverse angular inertia factor for a solid sphere contact (see resolveContact). */
const SPHERE_TANGENT_DENOM = 3.5;
/**
 * Rolling resistance coefficient, applied only while touching a surface. A
 * perfect sphere on a perfect plane rolls forever; a brass casing has flats and
 * stops dead. Modelled Coulomb-style (a fixed velocity decrement per step rather
 * than exponential decay) so it produces a real threshold: debris stays put on
 * any slope shallower than atan(ROLL_RESIST) ~= 15.6 degrees and rolls down
 * anything steeper.
 */
const ROLL_RESIST = 0.28;
/** Mild extra damping on top of the resistance, for the tumbling phase. */
const ROLL_ANGULAR_DAMP = 3.0;
const ROLL_LINEAR_DAMP = 1.2;

// ---------------------------------------------------------------------------
// Module-scope scratch — nothing in a hot path may allocate
// ---------------------------------------------------------------------------

const _cp = new Float64Array(3);

// ---------------------------------------------------------------------------
// Characters
// ---------------------------------------------------------------------------

interface HitboxCache {
  kind: HitboxKind;
  damageMultiplier: number;
  attachTo: THREE.Object3D;
  /** Local offset from attachTo's origin. */
  ox: number; oy: number; oz: number;
  /** Local half extents. */
  lhx: number; lhy: number; lhz: number;
  /** World centre. */
  cx: number; cy: number; cz: number;
  /** World orthonormal axes. */
  a0x: number; a0y: number; a0z: number;
  a1x: number; a1y: number; a1z: number;
  a2x: number; a2y: number; a2z: number;
  /** World half extents (local * axis scale). */
  hx: number; hy: number; hz: number;
  /** Bounding sphere radius around the box centre. */
  r: number;
}

interface CharacterBody {
  entityId: number;
  root: THREE.Object3D;
  boxes: HitboxCache[];
  /** Bounding sphere over every hitbox, refreshed with them. */
  bx: number; by: number; bz: number; br: number;
}

// ---------------------------------------------------------------------------
// Rigid bodies
// ---------------------------------------------------------------------------

export interface DebrisOptions {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;
  radius: number;
  restitution: number;
  friction: number;
  /** Seconds before the body retires. */
  lifetime: number;
  object: THREE.Object3D;
  /**
   * Called when the body retires so a pool can reclaim the object. If omitted the
   * object is simply hidden — physics never disposes geometry it did not create.
   */
  onRetire?: (object: THREE.Object3D) => void;
}

class RigidBody {
  handle = 0;
  active = false;
  sleeping = false;
  sleepTimer = 0;
  /** Set by resolveContact during the step; drives rolling resistance and sleep. */
  contact = false;
  sinceContact = 99;
  /** Normal of the last contact this step; the rolling-resistance tangent plane. */
  cnx = 0; cny = 1; cnz = 0;
  /** Set while the body moves; lets settled debris skip its matrix update. */
  poseDirty = true;
  age = 0;
  lifetime = 4;
  radius = 0.01;
  restitution = 0.3;
  friction = 0.5;
  px = 0; py = 0; pz = 0;
  vx = 0; vy = 0; vz = 0;
  wx = 0; wy = 0; wz = 0;
  readonly quat = new THREE.Quaternion();
  object: THREE.Object3D | null = null;
  onRetire: ((object: THREE.Object3D) => void) | null = null;
}

// ---------------------------------------------------------------------------

export class PhysicsSystem implements System, PhysicsWorld {
  readonly name = 'physics';

  private ctx: GameContext | null = null;
  private readonly grid = new TriangleGrid();

  private nextStaticId = 1;
  private readonly staticIds = new Map<THREE.Object3D, number>();

  private readonly characters: CharacterBody[] = [];
  private readonly charById = new Map<number, CharacterBody>();
  private charDirty = true;

  // --- ray query state (shared by the bound visitors below) ----------------
  private qOx = 0; private qOy = 0; private qOz = 0;
  private qDx = 0; private qDy = 0; private qDz = 1;
  private qLimit = 0;
  private qBestTri = -1;
  private qMask = 0;
  /** Rejects triangles whose |normal.y| is below this — used by groundHeight. */
  private qMinAbsNy = 0;

  // --- raycastAll record buffers -------------------------------------------
  private rCount = 0;
  private rT = new Float64Array(MAX_RAY_RECORDS);
  private rKind = new Int32Array(MAX_RAY_RECORDS);
  private rA = new Int32Array(MAX_RAY_RECORDS);
  private rB = new Int32Array(MAX_RAY_RECORDS);
  private rN = new Float64Array(MAX_RAY_RECORDS * 3);
  private rOrder = new Int32Array(MAX_RAY_RECORDS);

  // --- capsule contact state ------------------------------------------------
  private cSegAx = 0; private cSegAy = 0; private cSegAz = 0;
  private cSegBx = 0; private cSegBy = 0; private cSegBz = 0;
  private cRadius = 0;
  private cHit = false;
  private cDepth = 0;
  private cNx = 0; private cNy = 1; private cNz = 0;
  private cPx = 0; private cPy = 0; private cPz = 0;
  private cSurf = 0;

  // --- sphere (debris) contact state ---------------------------------------
  private sPx = 0; private sPy = 0; private sPz = 0;
  private sRadius = 0;
  private sHit = false;
  private sDepth = 0;
  private sNx = 0; private sNy = 1; private sNz = 0;

  // --- OBB test outputs -----------------------------------------------------
  private obbTMin = 0;
  private obbTMax = 0;
  private obbNx = 0; private obbNy = 0; private obbNz = 0;

  // --- debris ---------------------------------------------------------------
  private readonly bodies: RigidBody[] = [];
  private readonly freeBodies: RigidBody[] = [];
  private readonly activeBodies: RigidBody[] = [];
  private readonly bodyByHandle = new Map<number, RigidBody>();
  private nextHandle = 1;
  private maxDebris = 256;
  private debrisSubsteps = 4;

  private unsubQuality: (() => void) | null = null;

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;
    this.applyQuality();
    this.unsubQuality = ctx.events.on('quality.changed', () => this.applyQuality());
  }

  private applyQuality(): void {
    const q = this.ctx?.quality;
    if (!q) return;
    // Debris is a small slice of the particle budget; casings must never be the
    // thing that costs the frame. An airborne body costs ~1.4us to step, so the
    // cap is set where a full pool in flight stays under ~0.35ms per fixed step.
    this.maxDebris = Math.max(48, Math.min(256, Math.round(q.particleBudget / 40)));
    this.debrisSubsteps = q.preset === 'low' ? 2 : q.preset === 'medium' ? 3 : 4;
    while (this.activeBodies.length > this.maxDebris) {
      this.retire(this.activeBodies[0]);
    }
  }

  update(_time: FrameTime, _ctx: GameContext): void {
    // Hitbox transforms are refreshed lazily on the first character query of the
    // frame, so whoever asks gets the freshest pose regardless of system order.
    this.charDirty = true;
    this.writeDebrisTransforms();
  }

  dispose(): void {
    this.unsubQuality?.();
    this.unsubQuality = null;
    for (let i = this.activeBodies.length - 1; i >= 0; i--) this.retire(this.activeBodies[i]);
    this.bodies.length = 0;
    this.freeBodies.length = 0;
    this.activeBodies.length = 0;
    this.bodyByHandle.clear();
    this.characters.length = 0;
    this.charById.clear();
    this.staticIds.clear();
    this.grid.clear();
    this.ctx = null;
  }

  // =========================================================================
  // Static registration
  // =========================================================================

  addStatic(object: THREE.Object3D, surface: SurfaceKind): void {
    if (this.staticIds.has(object)) this.removeStatic(object);
    const verts = extractTriangles(object);
    const triCount = (verts.length / 9) | 0;
    if (triCount === 0) return;

    const id = this.nextStaticId++;
    this.staticIds.set(object, id);
    // Water is queryable but never blocks movement or bullets unless explicitly
    // asked for; everything else answers to both World and Props so the shared
    // RayMask.Solid works without the level builder having to think about it.
    const mask = surface === 'water'
      ? RayMask.Water
      : (RayMask.World | RayMask.Props);
    this.grid.add({ id, root: object, surface, mask, verts, triCount });
  }

  removeStatic(object: THREE.Object3D): void {
    const id = this.staticIds.get(object);
    if (id === undefined) return;
    this.staticIds.delete(object);
    this.grid.removeById(id);
  }

  // =========================================================================
  // Characters
  // =========================================================================

  addCharacter(entityId: number, root: THREE.Object3D, boxes: CharacterHitbox[]): void {
    this.removeCharacter(entityId);
    const cached: HitboxCache[] = [];
    for (const b of boxes) {
      if (!b.attachTo) continue;
      cached.push({
        kind: b.kind,
        damageMultiplier: b.damageMultiplier,
        attachTo: b.attachTo,
        ox: b.offset.x, oy: b.offset.y, oz: b.offset.z,
        lhx: Math.abs(b.size.x), lhy: Math.abs(b.size.y), lhz: Math.abs(b.size.z),
        cx: 0, cy: 0, cz: 0,
        a0x: 1, a0y: 0, a0z: 0,
        a1x: 0, a1y: 1, a1z: 0,
        a2x: 0, a2y: 0, a2z: 1,
        hx: Math.abs(b.size.x), hy: Math.abs(b.size.y), hz: Math.abs(b.size.z),
        r: Math.hypot(b.size.x, b.size.y, b.size.z),
      });
    }
    if (cached.length === 0) return;
    const body: CharacterBody = { entityId, root, boxes: cached, bx: 0, by: 0, bz: 0, br: 0 };
    this.characters.push(body);
    this.charById.set(entityId, body);
    this.syncCharacter(body);
  }

  removeCharacter(entityId: number): void {
    const body = this.charById.get(entityId);
    if (!body) return;
    this.charById.delete(entityId);
    const i = this.characters.indexOf(body);
    if (i >= 0) this.characters.splice(i, 1);
  }

  /** Damage multiplier registered for a hitbox, for ballistics. Falls back to 1. */
  hitboxMultiplier(entityId: number, kind: HitboxKind): number {
    const body = this.charById.get(entityId);
    if (!body) return 1;
    for (const b of body.boxes) if (b.kind === kind) return b.damageMultiplier;
    return 1;
  }

  private syncCharacters(): void {
    if (!this.charDirty) return;
    this.charDirty = false;
    for (let i = 0; i < this.characters.length; i++) this.syncCharacter(this.characters[i]);
  }

  private syncCharacter(body: CharacterBody): void {
    body.root.updateMatrixWorld(false);
    let minR = 0;
    let sx = 0, sy = 0, sz = 0;
    const boxes = body.boxes;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const m = b.attachTo.matrixWorld.elements;

      // Columns are axis * scale; separate them so half extents scale with the rig.
      let x0 = m[0], y0 = m[1], z0 = m[2];
      let x1 = m[4], y1 = m[5], z1 = m[6];
      let x2 = m[8], y2 = m[9], z2 = m[10];
      const s0 = Math.hypot(x0, y0, z0) || 1;
      const s1 = Math.hypot(x1, y1, z1) || 1;
      const s2 = Math.hypot(x2, y2, z2) || 1;
      x0 /= s0; y0 /= s0; z0 /= s0;
      x1 /= s1; y1 /= s1; z1 /= s1;
      x2 /= s2; y2 /= s2; z2 /= s2;

      b.a0x = x0; b.a0y = y0; b.a0z = z0;
      b.a1x = x1; b.a1y = y1; b.a1z = z1;
      b.a2x = x2; b.a2y = y2; b.a2z = z2;
      b.hx = b.lhx * s0;
      b.hy = b.lhy * s1;
      b.hz = b.lhz * s2;
      b.r = Math.hypot(b.hx, b.hy, b.hz);

      b.cx = m[12] + m[0] * b.ox + m[4] * b.oy + m[8] * b.oz;
      b.cy = m[13] + m[1] * b.ox + m[5] * b.oy + m[9] * b.oz;
      b.cz = m[14] + m[2] * b.ox + m[6] * b.oy + m[10] * b.oz;

      sx += b.cx; sy += b.cy; sz += b.cz;
    }
    const n = boxes.length || 1;
    body.bx = sx / n; body.by = sy / n; body.bz = sz / n;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      const d = Math.hypot(b.cx - body.bx, b.cy - body.by, b.cz - body.bz) + b.r;
      if (d > minR) minR = d;
    }
    body.br = minR;
  }

  // =========================================================================
  // Ray queries
  // =========================================================================

  private readonly visitClosest = (tri: number): number => {
    if ((this.grid.triMask[tri] & this.qMask) === 0) return this.qLimit;
    if (this.qMinAbsNy > 0) {
      const ny = this.grid.normals[tri * 3 + 1];
      if (ny > -this.qMinAbsNy && ny < this.qMinAbsNy) return this.qLimit;
    }
    const v = this.grid.verts;
    const o = tri * 9;
    const t = rayTriangle(
      this.qOx, this.qOy, this.qOz, this.qDx, this.qDy, this.qDz,
      v[o], v[o + 1], v[o + 2],
      v[o + 3], v[o + 4], v[o + 5],
      v[o + 6], v[o + 7], v[o + 8],
    );
    if (t >= 0 && t < this.qLimit) {
      this.qLimit = t;
      this.qBestTri = tri;
    }
    return this.qLimit;
  };

  private readonly visitAny = (tri: number): number => {
    if ((this.grid.triMask[tri] & this.qMask) === 0) return this.qLimit;
    const v = this.grid.verts;
    const o = tri * 9;
    const t = rayTriangle(
      this.qOx, this.qOy, this.qOz, this.qDx, this.qDy, this.qDz,
      v[o], v[o + 1], v[o + 2],
      v[o + 3], v[o + 4], v[o + 5],
      v[o + 6], v[o + 7], v[o + 8],
    );
    if (t >= 0 && t < this.qLimit) {
      this.qBestTri = tri;
      this.qLimit = -1; // aborts the DDA
      return -1;
    }
    return this.qLimit;
  };

  private readonly visitGather = (tri: number): number => {
    if ((this.grid.triMask[tri] & this.qMask) === 0) return this.qLimit;
    if (this.rCount >= MAX_RAY_RECORDS) return this.qLimit;
    const v = this.grid.verts;
    const o = tri * 9;
    const t = rayTriangle(
      this.qOx, this.qOy, this.qOz, this.qDx, this.qDy, this.qDz,
      v[o], v[o + 1], v[o + 2],
      v[o + 3], v[o + 4], v[o + 5],
      v[o + 6], v[o + 7], v[o + 8],
    );
    if (t >= 0 && t <= this.qLimit) {
      const i = this.rCount++;
      this.rT[i] = t;
      this.rKind[i] = 0;
      this.rA[i] = tri;
      this.rB[i] = -1;
      const no = tri * 3;
      this.rN[i * 3] = this.grid.normals[no];
      this.rN[i * 3 + 1] = this.grid.normals[no + 1];
      this.rN[i * 3 + 2] = this.grid.normals[no + 2];
    }
    return this.qLimit;
  };

  /** Closest static-world hit. Returns Infinity for a miss; leaves qBestTri set. */
  private rayWorld(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number, mask: number, minAbsNy: number,
  ): number {
    this.qOx = ox; this.qOy = oy; this.qOz = oz;
    this.qDx = dx; this.qDy = dy; this.qDz = dz;
    this.qLimit = maxDist;
    this.qBestTri = -1;
    this.qMask = mask;
    this.qMinAbsNy = minAbsNy;
    this.grid.traverseRay(ox, oy, oz, dx, dy, dz, maxDist, this.visitClosest);
    this.qMinAbsNy = 0;
    return this.qBestTri >= 0 ? this.qLimit : Infinity;
  }

  /** True if anything static blocks the segment. Aborts at the first hit. */
  private rayWorldBlocked(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number, mask: number,
  ): boolean {
    if (maxDist <= 0) return false;
    this.qOx = ox; this.qOy = oy; this.qOz = oz;
    this.qDx = dx; this.qDy = dy; this.qDz = dz;
    this.qLimit = maxDist;
    this.qBestTri = -1;
    this.qMask = mask;
    this.grid.traverseRay(ox, oy, oz, dx, dy, dz, maxDist, this.visitAny);
    return this.qBestTri >= 0;
  }

  /**
   * Ray vs oriented box. Fills obbTMin/obbTMax and the entry face normal.
   * Returns false on a miss or if the box is entirely behind the origin.
   */
  private rayOBB(
    b: HitboxCache,
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    maxDist: number,
  ): boolean {
    const px = b.cx - ox, py = b.cy - oy, pz = b.cz - oz;
    let tMin = 0;
    let tMax = maxDist;
    let nAxis = -1;
    let nSign = 1;

    for (let axis = 0; axis < 3; axis++) {
      let axx: number, axy: number, axz: number, h: number;
      if (axis === 0) { axx = b.a0x; axy = b.a0y; axz = b.a0z; h = b.hx; }
      else if (axis === 1) { axx = b.a1x; axy = b.a1y; axz = b.a1z; h = b.hy; }
      else { axx = b.a2x; axy = b.a2y; axz = b.a2z; h = b.hz; }

      const e = axx * px + axy * py + axz * pz;
      const f = axx * dx + axy * dy + axz * dz;

      if (f > 1e-9 || f < -1e-9) {
        let t1 = (e - h) / f;
        let t2 = (e + h) / f;
        let sign = -1;
        if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; sign = 1; }
        if (t1 > tMin) { tMin = t1; nAxis = axis; nSign = sign; }
        if (t2 < tMax) tMax = t2;
        if (tMin > tMax) return false;
      } else if (-e - h > 0 || -e + h < 0) {
        return false;
      }
    }
    if (tMax < 0) return false;

    this.obbTMin = tMin;
    this.obbTMax = tMax;
    if (nAxis === 0) { this.obbNx = b.a0x * nSign; this.obbNy = b.a0y * nSign; this.obbNz = b.a0z * nSign; }
    else if (nAxis === 1) { this.obbNx = b.a1x * nSign; this.obbNy = b.a1y * nSign; this.obbNz = b.a1z * nSign; }
    else if (nAxis === 2) { this.obbNx = b.a2x * nSign; this.obbNy = b.a2y * nSign; this.obbNz = b.a2z * nSign; }
    else { this.obbNx = -dx; this.obbNy = -dy; this.obbNz = -dz; }
    return true;
  }

  raycast(
    origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number,
    mask: RayMask = DEFAULT_RAY_MASK,
  ): RayHit | null {
    const ox = origin.x, oy = origin.y, oz = origin.z;
    let dx = direction.x, dy = direction.y, dz = direction.z;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dl < 1e-9 || !(maxDistance > 0)) return null;
    dx /= dl; dy /= dl; dz /= dl;

    let bestT = Infinity;
    let bestTri = -1;
    if (mask & (RayMask.World | RayMask.Props | RayMask.Water)) {
      bestT = this.rayWorld(ox, oy, oz, dx, dy, dz, maxDistance, mask, 0);
      bestTri = this.qBestTri;
    }

    let bestChar: CharacterBody | null = null;
    let bestBox: HitboxCache | null = null;
    let cnx = 0, cny = 0, cnz = 0;
    let charExit = 0;

    if (mask & RayMask.Characters) {
      this.syncCharacters();
      const limit = Math.min(bestT, maxDistance);
      for (let i = 0; i < this.characters.length; i++) {
        const body = this.characters[i];
        if (!this.raySphereReject(ox, oy, oz, dx, dy, dz, body.bx, body.by, body.bz, body.br, limit)) continue;
        for (let j = 0; j < body.boxes.length; j++) {
          const box = body.boxes[j];
          if (!this.rayOBB(box, ox, oy, oz, dx, dy, dz, Math.min(bestT, maxDistance))) continue;
          if (this.obbTMin < bestT) {
            bestT = this.obbTMin;
            charExit = this.obbTMax;
            bestChar = body;
            bestBox = box;
            cnx = this.obbNx; cny = this.obbNy; cnz = this.obbNz;
            bestTri = -1;
          }
        }
      }
    }

    if (!Number.isFinite(bestT) || bestT > maxDistance) return null;

    const point = new THREE.Vector3(ox + dx * bestT, oy + dy * bestT, oz + dz * bestT);

    if (bestChar && bestBox) {
      const nl = Math.hypot(cnx, cny, cnz) || 1;
      let nx = cnx / nl, ny = cny / nl, nz = cnz / nl;
      if (nx * dx + ny * dy + nz * dz > 0) { nx = -nx; ny = -ny; nz = -nz; }
      return {
        point,
        normal: new THREE.Vector3(nx, ny, nz),
        distance: bestT,
        surface: 'flesh',
        entityId: bestChar.entityId,
        hitbox: bestBox.kind,
        object: bestChar.root,
        thickness: Math.max(0.02, charExit - bestT),
      };
    }

    const entry = this.grid.entryOf(bestTri);
    const surface = entry ? entry.surface : surfaceFromIndex(this.grid.triSurface[bestTri]);
    const no = bestTri * 3;
    let nx = this.grid.normals[no];
    let ny = this.grid.normals[no + 1];
    let nz = this.grid.normals[no + 2];
    if (nx * dx + ny * dy + nz * dz > 0) { nx = -nx; ny = -ny; nz = -nz; }

    return {
      point,
      normal: new THREE.Vector3(nx, ny, nz),
      distance: bestT,
      surface,
      entityId: -1,
      object: entry ? entry.root : null,
      thickness: SURFACE_THICKNESS[surface],
    };
  }

  private raySphereReject(
    ox: number, oy: number, oz: number,
    dx: number, dy: number, dz: number,
    sx: number, sy: number, sz: number, r: number, maxDist: number,
  ): boolean {
    const mx = sx - ox, my = sy - oy, mz = sz - oz;
    const proj = mx * dx + my * dy + mz * dz;
    if (proj < -r) return false;
    if (proj > maxDist + r) return false;
    const d2 = mx * mx + my * my + mz * mz - proj * proj;
    return d2 <= r * r;
  }

  raycastAll(
    origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number,
    mask: RayMask = DEFAULT_RAY_MASK,
  ): RayHit[] {
    const out: RayHit[] = [];
    const ox = origin.x, oy = origin.y, oz = origin.z;
    let dx = direction.x, dy = direction.y, dz = direction.z;
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dl < 1e-9 || !(maxDistance > 0)) return out;
    dx /= dl; dy /= dl; dz /= dl;

    // Probe past the requested distance so the exit face of the last wall the
    // bullet entered is still found and the thickness is real, not a guess.
    const probe = maxDistance + THICKNESS_PROBE;
    this.rCount = 0;

    if (mask & (RayMask.World | RayMask.Props | RayMask.Water)) {
      this.qOx = ox; this.qOy = oy; this.qOz = oz;
      this.qDx = dx; this.qDy = dy; this.qDz = dz;
      this.qLimit = probe;
      this.qMask = mask;
      this.grid.traverseRay(ox, oy, oz, dx, dy, dz, probe, this.visitGather);
    }

    if (mask & RayMask.Characters) {
      this.syncCharacters();
      for (let i = 0; i < this.characters.length; i++) {
        const body = this.characters[i];
        if (!this.raySphereReject(ox, oy, oz, dx, dy, dz, body.bx, body.by, body.bz, body.br, probe)) continue;
        for (let j = 0; j < body.boxes.length; j++) {
          if (this.rCount + 2 > MAX_RAY_RECORDS) break;
          const box = body.boxes[j];
          if (!this.rayOBB(box, ox, oy, oz, dx, dy, dz, probe)) continue;
          const nl = Math.hypot(this.obbNx, this.obbNy, this.obbNz) || 1;
          const enter = this.rCount++;
          this.rT[enter] = this.obbTMin;
          this.rKind[enter] = 1;
          this.rA[enter] = i;
          this.rB[enter] = j;
          this.rN[enter * 3] = this.obbNx / nl;
          this.rN[enter * 3 + 1] = this.obbNy / nl;
          this.rN[enter * 3 + 2] = this.obbNz / nl;

          const exit = this.rCount++;
          this.rT[exit] = this.obbTMax;
          this.rKind[exit] = 2;
          this.rA[exit] = i;
          this.rB[exit] = j;
          this.rN[exit * 3] = -this.obbNx / nl;
          this.rN[exit * 3 + 1] = -this.obbNy / nl;
          this.rN[exit * 3 + 2] = -this.obbNz / nl;
        }
      }
    }

    const n = this.rCount;
    if (n === 0) return out;

    // Insertion sort: n is tens, not thousands, and this allocates nothing.
    const order = this.rOrder;
    for (let i = 0; i < n; i++) order[i] = i;
    for (let i = 1; i < n; i++) {
      const key = order[i];
      const kv = this.rT[key];
      let j = i - 1;
      while (j >= 0 && this.rT[order[j]] > kv) { order[j + 1] = order[j]; j--; }
      order[j + 1] = key;
    }

    let dupT = -1;
    let dupEntry = -2;
    let dupFacing = 0;

    for (let oi = 0; oi < n; oi++) {
      const i = order[oi];
      const t = this.rT[i];
      if (t > maxDistance) break;
      const kind = this.rKind[i];

      const nx = this.rN[i * 3], ny = this.rN[i * 3 + 1], nz = this.rN[i * 3 + 2];
      const facing = nx * dx + ny * dy + nz * dz;

      if (kind === 0) {
        const tri = this.rA[i];
        const entry = this.grid.entryOf(tri);
        const surface = entry ? entry.surface : surfaceFromIndex(this.grid.triSurface[tri]);
        const entryIdx = this.grid.triEntry[tri];
        const isEntry = facing < 0;

        // A ray crossing the shared edge of two coplanar triangles hits both
        // (the barycentric slack that seals cracks guarantees it). Collapse the
        // pair, or ballistics would charge the bullet for two walls.
        const facingSign = isEntry ? -1 : 1;
        if (entryIdx === dupEntry && facingSign === dupFacing && t - dupT < 2e-4) continue;
        dupT = t; dupEntry = entryIdx; dupFacing = facingSign;

        // The stored normal is already the outward one in both cases: a front
        // face points back at the shooter, a back face points along the bullet.
        let thickness = 0;
        if (isEntry) {
          // Pair with the next back-facing hit belonging to the same mesh.
          let exitT = -1;
          for (let oj = oi + 1; oj < n; oj++) {
            const j = order[oj];
            if (this.rKind[j] !== 0) continue;
            const tj = this.rA[j];
            if (this.grid.triEntry[tj] !== entryIdx) continue;
            const fj = this.rN[j * 3] * dx + this.rN[j * 3 + 1] * dy + this.rN[j * 3 + 2] * dz;
            if (fj > 0 && this.rT[j] > t + 1e-5) { exitT = this.rT[j]; break; }
          }
          // Single-sided level geometry has no back face; fall back to the real
          // construction thickness of the material rather than "solid forever".
          thickness = exitT > 0 ? exitT - t : SURFACE_THICKNESS[surface];
        }

        out.push({
          point: new THREE.Vector3(ox + dx * t, oy + dy * t, oz + dz * t),
          normal: new THREE.Vector3(nx, ny, nz),
          distance: t,
          surface,
          entityId: -1,
          object: entry ? entry.root : null,
          thickness,
        });
      } else {
        const body = this.characters[this.rA[i]];
        if (!body) continue;
        const box = body.boxes[this.rB[i]];
        if (!box) continue;
        let thickness = 0;
        if (kind === 1) {
          for (let oj = oi + 1; oj < n; oj++) {
            const j = order[oj];
            if (this.rKind[j] === 2 && this.rA[j] === this.rA[i] && this.rB[j] === this.rB[i]) {
              thickness = Math.max(0.02, this.rT[j] - t);
              break;
            }
          }
          if (thickness === 0) thickness = SURFACE_THICKNESS.flesh;
        }
        out.push({
          point: new THREE.Vector3(ox + dx * t, oy + dy * t, oz + dz * t),
          normal: new THREE.Vector3(nx, ny, nz),
          distance: t,
          surface: 'flesh',
          entityId: body.entityId,
          hitbox: box.kind,
          object: body.root,
          thickness,
        });
      }
    }

    return out;
  }

  hasLineOfSight(from: THREE.Vector3, to: THREE.Vector3, mask: RayMask = RayMask.Solid): boolean {
    const dx = to.x - from.x, dy = to.y - from.y, dz = to.z - from.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 1e-5) return true;
    const inv = 1 / dist;
    // Pull the far end back a few centimetres: a target standing flush against a
    // wall must not occlude itself with the wall it is touching.
    const span = dist - 0.04;
    if (span <= 0) return true;
    return !this.rayWorldBlocked(from.x, from.y, from.z, dx * inv, dy * inv, dz * inv, span, mask);
  }

  groundHeight(at: THREE.Vector3, maxDrop = 4): number | null {
    const lift = 0.6;
    // |n.y| rather than n.y: level geometry with flipped winding is common and a
    // floor is still a floor.
    const t = this.rayWorld(
      at.x, at.y + lift, at.z, 0, -1, 0,
      maxDrop + lift, RayMask.Solid, 0.2,
    );
    if (!Number.isFinite(t)) return null;
    return at.y + lift - t;
  }

  // =========================================================================
  // Capsule queries
  // =========================================================================

  private readonly visitCapsule = (tri: number): void => {
    if ((this.grid.triMask[tri] & this.qMask) === 0) return;
    const v = this.grid.verts;
    const o = tri * 9;
    const r = this.cRadius;
    const d2 = closestSegmentTriangle(
      this.cSegAx, this.cSegAy, this.cSegAz,
      this.cSegBx, this.cSegBy, this.cSegBz,
      v[o], v[o + 1], v[o + 2],
      v[o + 3], v[o + 4], v[o + 5],
      v[o + 6], v[o + 7], v[o + 8],
    );
    if (d2 >= r * r) return;

    const d = Math.sqrt(d2);
    let nx: number, ny: number, nz: number;
    let depth: number;
    if (d > 1e-6) {
      nx = (CST_SEG[0] - CST_TRI[0]) / d;
      ny = (CST_SEG[1] - CST_TRI[1]) / d;
      nz = (CST_SEG[2] - CST_TRI[2]) / d;
      depth = r - d;
    } else {
      // Axis pierces the face: push out along the face normal, oriented by which
      // side the capsule centre sits on.
      const no = tri * 3;
      nx = this.grid.normals[no]; ny = this.grid.normals[no + 1]; nz = this.grid.normals[no + 2];
      const mx = (this.cSegAx + this.cSegBx) * 0.5 - CST_TRI[0];
      const my = (this.cSegAy + this.cSegBy) * 0.5 - CST_TRI[1];
      const mz = (this.cSegAz + this.cSegBz) * 0.5 - CST_TRI[2];
      if (nx * mx + ny * my + nz * mz < 0) { nx = -nx; ny = -ny; nz = -nz; }
      depth = r;
    }

    if (!this.cHit || depth > this.cDepth) {
      this.cHit = true;
      this.cDepth = depth;
      this.cNx = nx; this.cNy = ny; this.cNz = nz;
      this.cPx = CST_TRI[0]; this.cPy = CST_TRI[1]; this.cPz = CST_TRI[2];
      this.cSurf = this.grid.triSurface[tri];
    }
  };

  /**
   * Deepest overlap of a vertical capsule with the static world.
   * `x/y/z` is the capsule base (feet). Leaves the contact in cNx/cPx/cDepth.
   */
  private capsuleContact(
    x: number, y: number, z: number,
    radius: number, height: number, mask: number, inflate: number,
  ): boolean {
    const r = radius + inflate;
    const ay = y + radius;
    const by = y + Math.max(height - radius, radius);

    this.cSegAx = x; this.cSegAy = ay; this.cSegAz = z;
    this.cSegBx = x; this.cSegBy = by; this.cSegBz = z;
    this.cRadius = r;
    this.cHit = false;
    this.cDepth = 0;
    this.qMask = mask;

    // Top of the swept volume: the upper cap can sit above `height` when the
    // capsule is squat (crouch heights approach 2r).
    this.grid.queryAABB(
      x - r, y - inflate, z - r,
      x + r, by + r, z + r,
      this.visitCapsule,
    );
    return this.cHit;
  }

  /**
   * True if the capsule is meaningfully inside geometry. Uses the same
   * PENETRATION_SLOP as sweepCapsule on purpose: the two must agree, or a
   * character resting on the floor reads as "blocked" to an overlap test while a
   * sweep happily moves it, and stand/crouch transitions jitter.
   */
  overlapCapsule(at: THREE.Vector3, radius: number, height: number, mask: RayMask = RayMask.Solid): boolean {
    if (!(mask & (RayMask.World | RayMask.Props | RayMask.Water))) return false;
    return this.capsuleContact(at.x, at.y, at.z, radius, height, mask, -PENETRATION_SLOP);
  }

  sweepCapsule(
    from: THREE.Vector3, to: THREE.Vector3, radius: number, height: number,
    mask: RayMask = RayMask.Solid,
  ): CapsuleSweepHit | null {
    if (!(mask & (RayMask.World | RayMask.Props | RayMask.Water))) return null;

    const fx = from.x, fy = from.y, fz = from.z;
    const dx = to.x - fx, dy = to.y - fy, dz = to.z - fz;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Already penetrating past the slop: report t=0 with the escape normal so the
    // caller can depenetrate rather than sliding deeper into the wall.
    if (this.capsuleContact(fx, fy, fz, radius, height, mask, -PENETRATION_SLOP)) {
      return this.makeSweepHit(0);
    }
    if (dist < 1e-6) return null;

    // Sub-step shorter than the radius: the capsule always overlaps anything it
    // passes, so a sprint into a corner cannot tunnel however thin the wall is.
    const steps = Math.max(1, Math.min(
      SWEEP_MAX_STEPS,
      Math.ceil(dist / Math.max(1e-3, radius * SWEEP_STEP_FRACTION)),
    ));
    // The stop-short gap is applied to the *result*, never to the overlap test:
    // inflating the test would make sliding along a wall read as a head-on hit
    // and pin the player in place.
    const pull = Math.min(0.5, SWEEP_SKIN / dist);

    const slop = -PENETRATION_SLOP;
    let lastFree = 0;
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      if (!this.capsuleContact(fx + dx * t, fy + dy * t, fz + dz * t, radius, height, mask, slop)) {
        lastFree = t;
        continue;
      }
      let lo = lastFree;
      let hi = t;
      for (let k = 0; k < SWEEP_REFINE; k++) {
        const mid = (lo + hi) * 0.5;
        if (this.capsuleContact(fx + dx * mid, fy + dy * mid, fz + dz * mid, radius, height, mask, slop)) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      // Re-evaluate at the contact fraction so the reported normal/point belong
      // to the triangle actually being touched, not to the bisection's last probe.
      this.capsuleContact(fx + dx * hi, fy + dy * hi, fz + dz * hi, radius, height, mask, slop);
      return this.makeSweepHit(Math.max(0, lo - pull));
    }
    return null;
  }

  private makeSweepHit(t: number): CapsuleSweepHit {
    const l = Math.hypot(this.cNx, this.cNy, this.cNz) || 1;
    return {
      t,
      normal: new THREE.Vector3(this.cNx / l, this.cNy / l, this.cNz / l),
      point: new THREE.Vector3(this.cPx, this.cPy, this.cPz),
      surface: surfaceFromIndex(this.cSurf),
    };
  }

  /**
   * Pushes a capsule out of anything it is intersecting, writing the corrected
   * base position into `out`. Sequential resolution over a few passes handles
   * inside corners (two walls at once) without the oscillation a single
   * max-depth push produces.
   */
  depenetrateCapsule(
    at: THREE.Vector3, radius: number, height: number, out: THREE.Vector3,
    mask: RayMask = RayMask.Solid,
  ): boolean {
    let x = at.x, y = at.y, z = at.z;
    let moved = false;
    for (let pass = 0; pass < DEPEN_PASSES; pass++) {
      if (!this.capsuleContact(x, y, z, radius, height, mask, 0)) break;
      const l = Math.hypot(this.cNx, this.cNy, this.cNz) || 1;
      const push = this.cDepth + 1e-4;
      x += (this.cNx / l) * push;
      y += (this.cNy / l) * push;
      z += (this.cNz / l) * push;
      moved = true;
    }
    out.set(x, y, z);
    return moved;
  }

  // =========================================================================
  // Rigid-body debris
  // =========================================================================

  private readonly visitSphere = (tri: number): void => {
    if ((this.grid.triMask[tri] & RayMask.Solid) === 0) return;
    const v = this.grid.verts;
    const o = tri * 9;
    const r = this.sRadius;
    closestPointTriangle(
      this.sPx, this.sPy, this.sPz,
      v[o], v[o + 1], v[o + 2],
      v[o + 3], v[o + 4], v[o + 5],
      v[o + 6], v[o + 7], v[o + 8],
      _cp, 0,
    );
    const dx = this.sPx - _cp[0], dy = this.sPy - _cp[1], dz = this.sPz - _cp[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= r * r) return;
    const d = Math.sqrt(d2);
    let nx: number, ny: number, nz: number;
    if (d > 1e-7) {
      nx = dx / d; ny = dy / d; nz = dz / d;
    } else {
      const no = tri * 3;
      nx = this.grid.normals[no]; ny = this.grid.normals[no + 1]; nz = this.grid.normals[no + 2];
    }
    const depth = r - d;
    if (!this.sHit || depth > this.sDepth) {
      this.sHit = true;
      this.sDepth = depth;
      this.sNx = nx; this.sNy = ny; this.sNz = nz;
    }
  };

  /**
   * Spawns a rigid debris body (shell casing, gib, rubble chunk). The body is a
   * sphere for collision purposes but carries a full orientation, so a casing
   * tumbles, bites the floor on contact and spins down to rest.
   * Returns a handle usable with `despawnDebris`.
   */
  spawnDebris(opts: DebrisOptions): number {
    if (!(opts.radius > 0)) return 0;

    let body: RigidBody | undefined = this.freeBodies.pop();
    if (!body) {
      if (this.activeBodies.length >= this.maxDebris) {
        // Recycle the oldest rather than refusing: the muzzle must keep ejecting.
        body = this.activeBodies[0];
        this.retire(body);
        body = this.freeBodies.pop();
      } else {
        body = new RigidBody();
        this.bodies.push(body);
      }
    }
    if (!body) return 0;

    body.handle = this.nextHandle++;
    body.active = true;
    body.sleeping = false;
    body.sleepTimer = 0;
    body.age = 0;
    body.lifetime = Math.max(0.1, opts.lifetime);
    body.radius = opts.radius;
    body.restitution = Math.max(0, Math.min(0.95, opts.restitution));
    body.friction = Math.max(0, Math.min(1.5, opts.friction));
    body.px = opts.position.x; body.py = opts.position.y; body.pz = opts.position.z;
    body.vx = opts.velocity.x; body.vy = opts.velocity.y; body.vz = opts.velocity.z;
    body.wx = opts.angularVelocity.x; body.wy = opts.angularVelocity.y; body.wz = opts.angularVelocity.z;
    body.object = opts.object;
    body.onRetire = opts.onRetire ?? null;
    body.quat.copy(opts.object.quaternion);
    body.poseDirty = true;
    opts.object.position.set(body.px, body.py, body.pz);
    opts.object.visible = true;

    this.activeBodies.push(body);
    this.bodyByHandle.set(body.handle, body);
    return body.handle;
  }

  despawnDebris(handle: number): void {
    const body = this.bodyByHandle.get(handle);
    if (body) this.retire(body);
  }

  get debrisCount(): number {
    return this.activeBodies.length;
  }

  /** Bodies currently asleep, i.e. costing nothing. For the debug overlay. */
  get debrisSleepingCount(): number {
    let n = 0;
    for (let i = 0; i < this.activeBodies.length; i++) if (this.activeBodies[i].sleeping) n++;
    return n;
  }

  private retire(body: RigidBody): void {
    if (!body.active) return;
    body.active = false;
    const i = this.activeBodies.indexOf(body);
    if (i >= 0) this.activeBodies.splice(i, 1);
    this.bodyByHandle.delete(body.handle);
    const obj = body.object;
    body.object = null;
    const cb = body.onRetire;
    body.onRetire = null;
    if (obj) {
      if (cb) cb(obj);
      else obj.visible = false;
    }
    this.freeBodies.push(body);
  }

  fixedUpdate(step: number, _ctx: GameContext): void {
    const bodies = this.activeBodies;
    if (bodies.length === 0) return;

    const g = UNITS.gravity;
    // Air drag as an exponential so it is framerate independent.
    const linDamp = Math.exp(-0.4 * step);
    const angDamp = Math.exp(-1.1 * step);
    const floorY = this.grid.worldMinY - 40;

    for (let i = bodies.length - 1; i >= 0; i--) {
      const b = bodies[i];
      b.age += step;
      if (b.age >= b.lifetime) { this.retire(b); continue; }
      if (b.sleeping) continue;
      b.poseDirty = true;

      b.vy -= g * step;
      b.vx *= linDamp; b.vy *= linDamp; b.vz *= linDamp;

      const speed = Math.sqrt(b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
      const sub = Math.max(1, Math.min(
        this.debrisSubsteps,
        Math.ceil((speed * step) / Math.max(1e-4, b.radius * 0.9)),
      ));
      const h = step / sub;

      // The support impulse a resting body receives from the ground over one
      // step. Without it Coulomb friction is clamped to zero at rest and debris
      // spins in place forever instead of biting and settling.
      const supportImpulse = g * step;
      b.contact = false;
      for (let s = 0; s < sub; s++) {
        this.integrateBody(b, h, supportImpulse);
      }

      if (b.contact) {
        b.sinceContact = 0;
        const rollA = Math.exp(-ROLL_ANGULAR_DAMP * step);
        const rollL = Math.exp(-ROLL_LINEAR_DAMP * step);
        b.wx *= rollA; b.wy *= rollA; b.wz *= rollA;
        b.vx *= rollL; b.vz *= rollL;

        // Coulomb rolling resistance: shed a fixed slice of tangential speed,
        // never reversing it. This is what gives debris a resting threshold.
        const dec = ROLL_RESIST * g * step;
        const nx = b.cnx, ny = b.cny, nz = b.cnz;
        const vn = b.vx * nx + b.vy * ny + b.vz * nz;
        const tx = b.vx - nx * vn, ty = b.vy - ny * vn, tz = b.vz - nz * vn;
        const tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (tl > 1e-7) {
          const k = Math.max(0, tl - dec) / tl;
          b.vx = nx * vn + tx * k;
          b.vy = ny * vn + ty * k;
          b.vz = nz * vn + tz * k;
        }
        const wl = Math.sqrt(b.wx * b.wx + b.wy * b.wy + b.wz * b.wz);
        // dw = Crr * N * r * I^-1, with I = 2/5 m r^2, so the r cancels to 2.5/r.
        const dw = (dec * 2.5) / b.radius;
        if (wl > 1e-7) {
          const k = Math.max(0, wl - dw) / wl;
          b.wx *= k; b.wy *= k; b.wz *= k;
        }
      } else {
        b.sinceContact += step;
        b.wx *= angDamp; b.wy *= angDamp; b.wz *= angDamp;
      }

      // Quaternion integration: q += 0.5 * w * q * dt.
      const hx = b.wx * 0.5 * step, hy = b.wy * 0.5 * step, hz = b.wz * 0.5 * step;
      const q = b.quat;
      const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
      q.set(
        qx + (hx * qw + hy * qz - hz * qy),
        qy + (hy * qw + hz * qx - hx * qz),
        qz + (hz * qw + hx * qy - hy * qx),
        qw + (-hx * qx - hy * qy - hz * qz),
      ).normalize();

      if (b.py < floorY) { this.retire(b); continue; }

      const lin2 = b.vx * b.vx + b.vy * b.vy + b.vz * b.vz;
      const ang2 = b.wx * b.wx + b.wy * b.wy + b.wz * b.wz;
      // The contact grace stops a body sleeping at the apex of a lob, where its
      // velocity legitimately passes through zero in mid-air.
      const angLimit = Math.min(SLEEP_ANGULAR_ABS, SLEEP_SURFACE_SPEED / b.radius);
      if (b.sinceContact <= SLEEP_CONTACT_GRACE
        && lin2 < SLEEP_LINEAR * SLEEP_LINEAR && ang2 < angLimit * angLimit) {
        b.sleepTimer += step;
        if (b.sleepTimer >= SLEEP_TIME) {
          b.sleeping = true;
          b.vx = b.vy = b.vz = 0;
          b.wx = b.wy = b.wz = 0;
        }
      } else {
        b.sleepTimer = 0;
      }
    }
  }

  private integrateBody(b: RigidBody, h: number, supportImpulse: number): void {
    const r = b.radius;
    const vlen = Math.sqrt(b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
    const moveLen = vlen * h;

    if (moveLen > 1e-7) {
      // Continuous check first: a 6mm casing at 8 m/s would otherwise punch
      // straight through a floor between two discrete positions.
      const idx = 1 / vlen;
      const dx = b.vx * idx, dy = b.vy * idx, dz = b.vz * idx;
      const t = this.rayWorld(b.px, b.py, b.pz, dx, dy, dz, moveLen + r, RayMask.Solid, 0);
      if (Number.isFinite(t) && this.qBestTri >= 0) {
        const no = this.qBestTri * 3;
        let nx = this.grid.normals[no], ny = this.grid.normals[no + 1], nz = this.grid.normals[no + 2];
        if (nx * dx + ny * dy + nz * dz > 0) { nx = -nx; ny = -ny; nz = -nz; }
        const approach = dx * nx + dy * ny + dz * nz;
        if (approach < -0.05) {
          const hitX = b.px + dx * t, hitY = b.py + dy * t, hitZ = b.pz + dz * t;
          b.px = hitX + nx * r * 1.02;
          b.py = hitY + ny * r * 1.02;
          b.pz = hitZ + nz * r * 1.02;
          this.resolveContact(b, nx, ny, nz, supportImpulse);
        } else {
          b.px += b.vx * h; b.py += b.vy * h; b.pz += b.vz * h;
        }
      } else {
        b.px += b.vx * h; b.py += b.vy * h; b.pz += b.vz * h;
      }
    }

    // Resting/penetration pass.
    this.sPx = b.px; this.sPy = b.py; this.sPz = b.pz;
    this.sRadius = r;
    this.sHit = false;
    this.sDepth = 0;
    this.grid.queryAABB(b.px - r, b.py - r, b.pz - r, b.px + r, b.py + r, b.pz + r, this.visitSphere);
    if (this.sHit) {
      const l = Math.hypot(this.sNx, this.sNy, this.sNz) || 1;
      const nx = this.sNx / l, ny = this.sNy / l, nz = this.sNz / l;
      b.px += nx * (this.sDepth + 1e-5);
      b.py += ny * (this.sDepth + 1e-5);
      b.pz += nz * (this.sDepth + 1e-5);
      this.resolveContact(b, nx, ny, nz, supportImpulse);
    }
  }

  /**
   * Impulse response for a unit-mass solid sphere against a static plane.
   * The normal impulse arm is zero (contact is through the centre) so only the
   * tangential impulse feeds angular velocity — which is exactly what makes a
   * casing skitter and roll instead of sliding like a puck.
   */
  private resolveContact(
    b: RigidBody, nx: number, ny: number, nz: number, supportImpulse: number,
  ): void {
    b.contact = true;
    b.cnx = nx; b.cny = ny; b.cnz = nz;
    const r = b.radius;
    // Contact point relative to centre.
    const rcx = -nx * r, rcy = -ny * r, rcz = -nz * r;
    // v_contact = v + w x rc
    const cvx = b.vx + (b.wy * rcz - b.wz * rcy);
    const cvy = b.vy + (b.wz * rcx - b.wx * rcz);
    const cvz = b.vz + (b.wx * rcy - b.wy * rcx);

    const vn = cvx * nx + cvy * ny + cvz * nz;
    if (vn >= 0) return;

    let e = b.restitution;
    const approach = -vn;
    // Only a real impact restarts the sleep countdown. A body at rest still
    // re-approaches the surface by one step of free fall every step, so the
    // threshold is expressed as a multiple of exactly that — which also makes it
    // independent of the fixed step length.
    if (approach > supportImpulse * 2.5) b.sleepTimer = 0;
    if (approach < BOUNCE_DEAD_SPEED) e = 0;
    else if (approach < BOUNCE_SOFT_SPEED) e *= approach / BOUNCE_SOFT_SPEED;

    const jn = -(1 + e) * vn;
    b.vx += nx * jn; b.vy += ny * jn; b.vz += nz * jn;

    // Tangential (Coulomb friction, clamped to mu * jn).
    let tx = cvx - nx * vn;
    let ty = cvy - ny * vn;
    let tz = cvz - nz * vn;
    const tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (tl < 1e-6) return;
    tx /= tl; ty /= tl; tz /= tl;

    let jt = -tl / SPHERE_TANGENT_DENOM;
    // Clamp against the *total* normal impulse: the bounce impulse plus the
    // support the ground is providing against gravity this step.
    const maxJt = b.friction * (jn + supportImpulse);
    if (jt < -maxJt) jt = -maxJt;

    b.vx += tx * jt; b.vy += ty * jt; b.vz += tz * jt;

    // dw = I^-1 (rc x j_t), with I = 2/5 m r^2 for a solid sphere.
    const invI = 1 / (0.4 * r * r);
    const jx = tx * jt, jy = ty * jt, jz = tz * jt;
    b.wx += (rcy * jz - rcz * jy) * invI;
    b.wy += (rcz * jx - rcx * jz) * invI;
    b.wz += (rcx * jy - rcy * jx) * invI;
  }

  private writeDebrisTransforms(): void {
    const bodies = this.activeBodies;
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      const obj = b.object;
      if (!obj || !b.poseDirty) continue;
      b.poseDirty = false;
      // Writing a settled body's transform every frame would keep three
      // recomputing its world matrix for nothing.
      obj.position.set(b.px, b.py, b.pz);
      obj.quaternion.copy(b.quat);
    }
  }

  // =========================================================================
  // Introspection (debug overlays / QA harness)
  // =========================================================================

  get staticTriangleCount(): number {
    this.grid.ensureBuilt();
    return this.grid.triCount;
  }

  get characterCount(): number {
    return this.characters.length;
  }

  /** World-space centre of a registered character's hitbox, for AI aiming. */
  getHitboxCenter(entityId: number, kind: HitboxKind, out: THREE.Vector3): boolean {
    const body = this.charById.get(entityId);
    if (!body) return false;
    this.syncCharacters();
    for (const b of body.boxes) {
      if (b.kind === kind) {
        out.set(b.cx, b.cy, b.cz);
        return true;
      }
    }
    return false;
  }
}
