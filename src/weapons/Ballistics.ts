/**
 * BallisticsSystem — everything between the muzzle and the target.
 *
 * Owns four things:
 *   1. Round resolution. `shot.fired` arrives with an aim ray; this system adds
 *      spread, then either hitscans it or spawns a simulated projectile that is
 *      integrated at the fixed 120Hz step with gravity and quadratic drag.
 *   2. Terminal ballistics: wall penetration budgeted in "concrete-centimetres",
 *      grazing-angle ricochet, damage falloff, hitbox multipliers.
 *   3. Melee: a lunge with a short anticipation window and forgiving reach cone.
 *   4. Frags: cookable, physically thrown, bouncing, with line-of-sight
 *      attenuated blast damage.
 *
 * Conventions this relies on from the physics contract:
 *   - `raycastAll` returns hits near-to-far and includes back faces. An *entry*
 *     record carries a positive `thickness` (the distance to its own back face);
 *     an *exit* record carries `thickness === 0`. This system therefore ignores
 *     exit records entirely and synthesises the exit point from the entry's
 *     thickness, which keeps a wall's two faces bound to one decision.
 *   - Normals are outward: an entry normal faces the shooter.
 *
 * Allocation policy: nothing in a shot, a projectile step or a blast allocates.
 * Vectors are module-scope scratch, projectiles/grenades are pooled, and event
 * payloads come from ring buffers so a listener that holds a reference for a
 * frame or two (a decal spawner, say) still sees valid data.
 */
import * as THREE from 'three';
import {
  RayMask,
  UNITS,
  type DamageEvent,
  type ExplosionEvent,
  type FrameTime,
  type GameContext,
  type HitboxKind,
  type ImpactEvent,
  type RayHit,
  type SurfaceKind,
  type System,
  type WeaponSpec,
} from '../core/Contracts';
import { getSpec } from './WeaponSpecs';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const TAU = Math.PI * 2;

/** Nothing is resolved past this; the level is far smaller. */
const MAX_TRACE_RANGE = 480;
/** Nudge past a surface after penetrating or reflecting off it. */
const SURFACE_EPSILON = 0.0025;
/** Safety valve on the penetrate/ricochet restart loop. */
const MAX_TRACE_SEGMENTS = 12;

/**
 * Bullets fall under real gravity. UNITS.gravity is 19.6 — deliberately double
 * real, because arcade jump arcs need it — but applying it to a projectile
 * doubles bullet drop and turns a 300m sniper shot into a mortar. Player
 * movement and bullet flight are allowed to disagree here; every shipped
 * shooter does the same.
 */
const BULLET_GRAVITY = 9.81;

/**
 * Quadratic drag constant k in dv/dx = -k*v, per metre. Calibrated from real
 * retained-velocity tables: 5.56 falls 880 -> ~600 m/s over 300 m, which is
 * ln(880/600)/300 = 0.00128. The rest are scaled by sectional density.
 */
const DRAG_BY_CATEGORY: Record<WeaponSpec['category'], number> = {
  ar: 0.00128,
  smg: 0.00190,
  lmg: 0.00118,
  sniper: 0.00074,
  shotgun: 0.00620,
  pistol: 0.00205,
};

/**
 * Cost multiplier per centimetre of material, relative to concrete. A weapon's
 * `penetration` is a budget in concrete-centimetres, so cost = thickness_cm * k.
 *
 * Sanity for the shipped roster: an AR (12) walks through plaster (0.16),
 * plywood (0.34) and sheet steel — thin, so cheap in total — but dies on 20cm
 * of concrete (20.0). The sniper (42) defeats that concrete and still exits a
 * torso behind it. Sandbags and packed earth stop everything, which is the
 * point of sandbags.
 */
const PEN_DENSITY: Record<SurfaceKind, number> = {
  concrete: 1.00,
  metal: 1.65,
  wood: 0.34,
  plaster: 0.16,
  glass: 0.10,
  dirt: 0.95,
  sand: 1.45,
  foliage: 0.03,
  fabric: 0.05,
  rubber: 0.55,
  water: 2.40,
  flesh: 0.55,
};

/** Only rigid, homogeneous surfaces deflect a round instead of eating it. */
const RICOCHET_SURFACE: Partial<Record<SurfaceKind, true>> = {
  concrete: true,
  metal: true,
};

/** cos(78 deg): below this the incidence is grazing enough to skip off. */
const RICOCHET_COS = 0.2079;

/** Damage retained after a wall, from "barely scratched it" to "only just through". */
const PEN_RETAIN_EASY = 0.90;
const PEN_RETAIN_HARD = 0.40;

/** Post-penetration deflection cone, radians. A few milliradians, tumble-dependent. */
const DEFLECT_MIN = 0.0008;
const DEFLECT_SPAN = 0.0042;
/** Ricochets scatter far more than penetrations do. */
const RICOCHET_SCATTER = 0.022;

const HITBOX_MULTIPLIER: Record<HitboxKind, number> = {
  head: 1.0, // spec.headshotMultiplier is applied on top
  chest: 1.0,
  stomach: 1.05,
  arm: 0.90,
  leg: 0.85,
};

/** One in this many rounds leaves a visible trace. Every LMG round does. */
const TRACER_EVERY = 3;

const MELEE_RANGE = 2.35;
const MELEE_DAMAGE = 155;
const MELEE_COOLDOWN = 0.72;
/** Anticipation before the hit registers. Under 100ms, so input still feels instant. */
const MELEE_WINDUP = 0.075;

const GRENADE_FUSE = 3.4;
const GRENADE_RADIUS = 0.035;
const GRENADE_BLAST_RADIUS = 7.0;
const GRENADE_BLAST_DAMAGE = 145;
const GRENADE_THROW_SPEED = 21.5;
const GRENADE_LOB = 3.6;
const GRENADE_RESTITUTION = 0.34;
const GRENADE_FRICTION = 0.70;
const GRENADE_MAX = 3;
const GRENADE_RESUPPLY = 30;
/** Blast damage behind cover. Not zero — a frag round a corner still hurts. */
const BLAST_OCCLUDED = 0.26;

const BULLET_MASK = RayMask.All;

// ---------------------------------------------------------------------------
// Module-scope scratch. Nothing below allocates in a hot path.
// ---------------------------------------------------------------------------

const _traceOrigin = new THREE.Vector3();
const _traceDir = new THREE.Vector3();
const _exitPoint = new THREE.Vector3();
const _exitNormal = new THREE.Vector3();
const _reflect = new THREE.Vector3();
const _basisR = new THREE.Vector3();
const _basisU = new THREE.Vector3();
const _spread = new THREE.Vector3();
const _muzzle = new THREE.Vector3();
const _aim = new THREE.Vector3();
const _stepDir = new THREE.Vector3();
const _accel = new THREE.Vector3();
const _camPos = new THREE.Vector3();
const _camDir = new THREE.Vector3();
const _camRight = new THREE.Vector3();
const _probe = new THREE.Vector3();
const _probeDir = new THREE.Vector3();
const _targetCentre = new THREE.Vector3();
const _blastPoint = new THREE.Vector3();
const _blastFrom = new THREE.Vector3();
const _spinAxis = new THREE.Vector3();
const _spinQuat = new THREE.Quaternion();
const _down = new THREE.Vector3(0, -1, 0);
const _worldUp = new THREE.Vector3(0, 1, 0);

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Orthonormal basis around `dir`. The seed axis is chosen away from `dir` so the
 * cross product never degenerates when shooting straight up or down.
 */
function buildBasis(dir: THREE.Vector3, right: THREE.Vector3, up: THREE.Vector3): void {
  if (Math.abs(dir.y) < 0.985) {
    right.set(-dir.z, 0, dir.x).normalize();
  } else {
    right.set(1, 0, 0);
  }
  up.crossVectors(right, dir).normalize();
}

/** Rotates `dir` by `angle` radians about an azimuth `phi`, writing into `dir`. */
function tiltDirection(dir: THREE.Vector3, angle: number, phi: number): void {
  if (angle <= 0) return;
  buildBasis(dir, _basisR, _basisU);
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  dir.multiplyScalar(c)
    .addScaledVector(_basisR, s * Math.cos(phi))
    .addScaledVector(_basisU, s * Math.sin(phi))
    .normalize();
}

/** Uniform-area sample inside a cone: radius must be sqrt-distributed. */
function scatterInCone(dir: THREE.Vector3, halfAngle: number): void {
  if (halfAngle <= 0) return;
  tiltDirection(dir, Math.sqrt(Math.random()) * halfAngle, Math.random() * TAU);
}

// ---------------------------------------------------------------------------
// Event payload rings
// ---------------------------------------------------------------------------

class Ring<T> {
  private readonly items: T[];
  private i = 0;
  constructor(size: number, make: () => T) {
    this.items = new Array<T>(size);
    for (let k = 0; k < size; k++) this.items[k] = make();
  }
  next(): T {
    const v = this.items[this.i]!;
    this.i = (this.i + 1) % this.items.length;
    return v;
  }
}

// ---------------------------------------------------------------------------
// Pooled records
// ---------------------------------------------------------------------------

class Projectile {
  active = false;
  spec: WeaponSpec | null = null;
  weaponId = '';
  shooterId = 0;
  local = false;
  readonly pos = new THREE.Vector3();
  readonly vel = new THREE.Vector3();
  drag = 0.0013;
  age = 0;
  travelled = 0;
  pen = 0;
  damageScale = 1;
  bounces = 0;
  walls = 0;
}

class Grenade {
  active = false;
  ownerId = 0;
  local = false;
  fuse = GRENADE_FUSE;
  readonly pos = new THREE.Vector3();
  readonly vel = new THREE.Vector3();
  readonly spin = new THREE.Vector3();
  restTime = 0;
  group: THREE.Group | null = null;
}

/** Per-round state carried across penetration/ricochet restarts. */
class TraceState {
  spec!: WeaponSpec;
  weaponId = '';
  shooterId = 0;
  local = false;
  /** Metres already flown before this trace call (projectile flight). */
  travelled = 0;
  pen = 0;
  damageScale = 1;
  bounces = 0;
  walls = 0;
  /** Velocity retention produced by *this* trace call only. */
  stepSpeedScale = 1;
  /** Entity already damaged by this round; a bullet only bills a target once. */
  damagedEntity = -1;
  terminated = false;
  readonly endPoint = new THREE.Vector3();
  readonly endDir = new THREE.Vector3();
}

// ---------------------------------------------------------------------------
// Procedural frag-body normal map
// ---------------------------------------------------------------------------

/**
 * Value-noise + groove-lattice bump, converted to a tangent-space normal map.
 * 64px is plenty: the object is 7cm across and never fills more than a few
 * hundred pixels, but without it the grenade reads as a flat plastic ball.
 */
function makeFragNormalMap(size: number): THREE.CanvasTexture | null {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const g = canvas.getContext('2d');
  if (!g) return null;

  const lattice = 8;
  const seed = new Float32Array(lattice * lattice);
  for (let i = 0; i < seed.length; i++) seed[i] = Math.random();

  const smooth = (t: number): number => t * t * (3 - 2 * t);
  const noise = (u: number, v: number): number => {
    const x = u * lattice;
    const y = v * lattice;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = smooth(x - x0);
    const fy = smooth(y - y0);
    const ix = (a: number, b: number): number => ((a % lattice) + lattice) % lattice
      + (((b % lattice) + lattice) % lattice) * lattice;
    const a = seed[ix(x0, y0)]!;
    const b = seed[ix(x0 + 1, y0)]!;
    const c = seed[ix(x0, y0 + 1)]!;
    const d = seed[ix(x0 + 1, y0 + 1)]!;
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
  };

  const h = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size;
      const v = y / size;
      // Cast-iron grain: three octaves.
      let n = noise(u, v) * 0.55 + noise(u * 2.7, v * 2.7) * 0.30 + noise(u * 6.1, v * 6.1) * 0.15;
      // Fragmentation grooves — the reason a frag looks like a pineapple.
      const gu = Math.abs(((u * 7) % 1) - 0.5) * 2;
      const gv = Math.abs(((v * 9) % 1) - 0.5) * 2;
      n -= (1 - smooth(clamp01(gu * 5))) * 0.55;
      n -= (1 - smooth(clamp01(gv * 5))) * 0.55;
      h[y * size + x] = n;
    }
  }

  const img = g.createImageData(size, size);
  const data = img.data;
  const strength = 2.6;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const xm = (x - 1 + size) % size;
      const xp = (x + 1) % size;
      const ym = (y - 1 + size) % size;
      const yp = (y + 1) % size;
      const dx = (h[y * size + xp]! - h[y * size + xm]!) * strength;
      const dy = (h[yp * size + x]! - h[ym * size + x]!) * strength;
      const len = Math.hypot(dx, dy, 1);
      const o = (y * size + x) * 4;
      data[o] = Math.round((-dx / len * 0.5 + 0.5) * 255);
      data[o + 1] = Math.round((-dy / len * 0.5 + 0.5) * 255);
      data[o + 2] = Math.round((1 / len * 0.5 + 0.5) * 255);
      data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------

export class BallisticsSystem implements System {
  readonly name = 'ballistics';

  private ctx!: GameContext;
  private readonly unsubs: (() => void)[] = [];

  private readonly projectiles: Projectile[] = [];
  private readonly grenades: Grenade[] = [];
  private readonly state = new TraceState();

  /** Local player's ADS state, tracked from `weapon.ads`. */
  private aiming = false;
  /** Round counter per weapon id, so the 1-in-3 tracer cadence is per gun. */
  private readonly tracerCount = new Map<string, number>();

  private meleeCooldown = 0;
  private meleePending = -1;
  private grenadesLeft = GRENADE_MAX;
  private grenadeResupply = 0;
  private cooking = false;
  private cookTime = 0;

  private maxWalls = 4;
  private maxBounces = 2;
  private maxProjectiles = 192;

  // Shared render resources for thrown frags.
  private fragBody: THREE.BufferGeometry | null = null;
  private fragLever: THREE.BufferGeometry | null = null;
  private fragMat: THREE.MeshStandardMaterial | null = null;
  private fragLeverMat: THREE.MeshStandardMaterial | null = null;
  private fragNormal: THREE.CanvasTexture | null = null;

  private readonly impactRing = new Ring<ImpactEvent>(48, () => ({
    point: new THREE.Vector3(),
    normal: new THREE.Vector3(),
    direction: new THREE.Vector3(),
    surface: 'concrete',
    energy: 1,
  }));
  private readonly tracerRing = new Ring<{
    from: THREE.Vector3; to: THREE.Vector3; speed: number; weaponId: string;
  }>(32, () => ({ from: new THREE.Vector3(), to: new THREE.Vector3(), speed: 0, weaponId: '' }));
  private readonly damageRing = new Ring<DamageEvent>(32, () => ({
    targetId: 0, attackerId: 0, amount: 0,
    point: new THREE.Vector3(), normal: new THREE.Vector3(), direction: new THREE.Vector3(),
    hitbox: 'chest', lethal: false, weaponId: '', distance: 0,
  }));
  private readonly hitmarkerRing = new Ring<{ headshot: boolean; lethal: boolean }>(
    16, () => ({ headshot: false, lethal: false }),
  );
  private readonly killRing = new Ring<{
    entityId: number; killerId: number; weaponId: string; headshot: boolean;
  }>(16, () => ({ entityId: 0, killerId: 0, weaponId: '', headshot: false }));
  private readonly explosionRing = new Ring<ExplosionEvent>(8, () => ({
    point: new THREE.Vector3(), radius: 0, damage: 0, attackerId: 0,
  }));
  private readonly shakeRing = new Ring<{ amount: number; duration: number; frequency?: number }>(
    16, () => ({ amount: 0, duration: 0, frequency: 24 }),
  );

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  init(ctx: GameContext): void {
    this.ctx = ctx;

    const low = ctx.quality.preset === 'low';
    this.maxWalls = low ? 2 : 4;
    this.maxBounces = low ? 1 : 2;
    this.maxProjectiles = low ? 64 : ctx.quality.preset === 'medium' ? 128 : 192;

    for (let i = 0; i < this.maxProjectiles; i++) this.projectiles.push(new Projectile());

    this.buildFragResources();
    for (let i = 0; i < 6; i++) {
      const gr = new Grenade();
      gr.group = this.makeFragMesh();
      gr.group.visible = false;
      ctx.scene.add(gr.group);
      this.grenades.push(gr);
    }

    this.unsubs.push(ctx.events.on('shot.fired', this.onShotFired));
    this.unsubs.push(ctx.events.on('weapon.ads', this.onAds));
  }

  private readonly onAds = (p: { aiming: boolean; fovScale: number }): void => {
    this.aiming = p.aiming;
  };

  // -------------------------------------------------------------------------
  // Round resolution
  // -------------------------------------------------------------------------

  private readonly onShotFired = (ev: {
    weaponId: string; origin: THREE.Vector3; direction: THREE.Vector3; local: boolean; shooterId: number;
  }): void => {
    const spec = getSpec(ev.weaponId);
    if (!spec) return;
    if (!(ev.direction.lengthSq() > 1e-8)) return;

    _muzzle.copy(ev.origin);
    _aim.copy(ev.direction).normalize();

    const cone = this.coneFor(spec, ev.local);
    const pellets = Math.max(1, spec.pellets | 0);
    const drag = DRAG_BY_CATEGORY[spec.category];

    // Ring phase is randomised per shot but the *shape* is fixed, so a shotgun
    // pattern is recognisable from one shot to the next. Real chokes behave the
    // same way; pure random pellets feel broken.
    const shotPhase = Math.random() * TAU;
    const tracerThisShot = this.wantsTracer(spec);

    for (let i = 0; i < pellets; i++) {
      _spread.copy(_aim);
      if (pellets > 1) {
        this.shotgunOffset(_spread, i, pellets, cone, shotPhase);
      } else {
        scatterInCone(_spread, cone);
      }

      const wantTracer = tracerThisShot && i === 0;

      if (spec.muzzleVelocity <= 0) {
        this.resolveHitscan(spec, ev.weaponId, ev.shooterId, ev.local, _muzzle, _spread, wantTracer);
      } else {
        const p = this.acquireProjectile();
        if (!p) {
          // Pool exhausted (a wall of shotgun fire). Never drop the round —
          // fall back to hitscan so the shot still counts.
          this.resolveHitscan(spec, ev.weaponId, ev.shooterId, ev.local, _muzzle, _spread, wantTracer);
          continue;
        }
        // The streak is emitted now, not on impact: a tracer that only appears
        // once the round lands is a tracer arriving 100ms late. The FX system
        // animates it from the muzzle at muzzleVelocity, so the visual and the
        // simulated round leave together and land together.
        if (wantTracer) this.emitProjectileTracer(spec, ev.weaponId, _muzzle, _spread);

        p.active = true;
        p.spec = spec;
        p.weaponId = ev.weaponId;
        p.shooterId = ev.shooterId;
        p.local = ev.local;
        p.pos.copy(_muzzle);
        p.vel.copy(_spread).multiplyScalar(spec.muzzleVelocity);
        // Pellets leave the muzzle at slightly different speeds; without this a
        // shotgun's pattern stays a rigid disc all the way downrange.
        if (pellets > 1) p.vel.multiplyScalar(0.9 + Math.random() * 0.2);
        p.drag = drag * (pellets > 1 ? 0.9 + Math.random() * 0.25 : 1);
        p.age = 0;
        p.travelled = 0;
        p.pen = spec.penetration;
        p.damageScale = 1;
        p.bounces = 0;
        p.walls = 0;
      }
    }
  };

  /** Cone half-angle for this shooter, in radians. */
  private coneFor(spec: WeaponSpec, local: boolean): number {
    if (!local) {
      // AI sits between hip and ADS: accurate enough to threaten, loose enough
      // to lose a straight duel against a player who is actually aiming.
      return spec.spreadAds + (spec.spreadHip - spec.spreadAds) * 0.18;
    }
    return this.aiming ? spec.spreadAds : spec.spreadHip;
  }

  /**
   * Designed buckshot pattern: a tight core, a mid ring and an outer ring, each
   * jittered. Matches how a real cylinder-bore pattern board looks and, more
   * importantly, makes the weapon's effective range learnable.
   */
  private shotgunOffset(
    dir: THREE.Vector3, index: number, count: number, cone: number, phase: number,
  ): void {
    if (count <= 3) {
      scatterInCone(dir, cone * 0.5);
      return;
    }
    const core = Math.max(1, Math.floor(count / 3));
    const inner = Math.max(1, Math.floor((count - core) / 2));

    let ring: number;
    let k: number;
    let n: number;
    if (index < core) {
      ring = 0; k = index; n = core;
    } else if (index < core + inner) {
      ring = 1; k = index - core; n = inner;
    } else {
      ring = 2; k = index - core - inner; n = count - core - inner;
    }

    const radiusFactor = ring === 0 ? 0.13 : ring === 1 ? 0.50 : 0.93;
    // 2.399963 rad is the golden angle: staggering rings by it stops the three
    // rings from lining up into visible spokes.
    const base = phase + ring * 2.399963 + (k / Math.max(1, n)) * TAU;
    const jitterA = (Math.random() * 2 - 1) * (TAU / Math.max(1, n)) * 0.30;
    const jitterR = (Math.random() * 2 - 1) * 0.10;
    const r = Math.max(0, radiusFactor + jitterR) * cone;
    tiltDirection(dir, r, base + jitterA);
  }

  /**
   * Cheap straight-line probe purely for the tracer's end point. Over 100m the
   * true trajectory drops ~6cm, which no one can see on a streak that lives for
   * 120ms; a full simulated path is not worth a second trace.
   */
  private emitProjectileTracer(
    spec: WeaponSpec, weaponId: string, origin: THREE.Vector3, dir: THREE.Vector3,
  ): void {
    const hit = this.ctx.physics.raycast(origin, dir, MAX_TRACE_RANGE, BULLET_MASK);
    _probe.copy(origin).addScaledVector(dir, hit ? hit.distance : MAX_TRACE_RANGE);
    this.emitTracer(origin, _probe, spec.muzzleVelocity, weaponId);
  }

  private wantsTracer(spec: WeaponSpec): boolean {
    if (spec.category === 'lmg') return true;
    const n = (this.tracerCount.get(spec.id) ?? 0) + 1;
    this.tracerCount.set(spec.id, n);
    return n % TRACER_EVERY === 0;
  }

  private resolveHitscan(
    spec: WeaponSpec, weaponId: string, shooterId: number, local: boolean,
    origin: THREE.Vector3, dir: THREE.Vector3, tracer: boolean,
  ): void {
    const st = this.state;
    st.spec = spec;
    st.weaponId = weaponId;
    st.shooterId = shooterId;
    st.local = local;
    st.travelled = 0;
    st.pen = spec.penetration;
    st.damageScale = 1;
    st.bounces = 0;
    st.walls = 0;
    st.stepSpeedScale = 1;
    st.damagedEntity = -1;

    this.trace(origin, dir, MAX_TRACE_RANGE, st);
    if (tracer) this.emitTracer(origin, st.endPoint, 1650, weaponId);
  }

  // -------------------------------------------------------------------------
  // The trace: penetration, ricochet, damage
  // -------------------------------------------------------------------------

  /**
   * Walks a ray through the world, spending penetration power and emitting
   * impacts. On return `st.endPoint` is where the round finished this segment,
   * `st.endDir` its direction there, and `st.terminated` whether it stopped.
   */
  private trace(origin: THREE.Vector3, dir: THREE.Vector3, maxDist: number, st: TraceState): void {
    const physics = this.ctx.physics;
    _traceOrigin.copy(origin);
    _traceDir.copy(dir).normalize();
    st.stepSpeedScale = 1;
    st.terminated = false;

    let remaining = maxDist;
    let segments = 0;

    while (remaining > 1e-4 && segments++ < MAX_TRACE_SEGMENTS) {
      const hits = physics.raycastAll(_traceOrigin, _traceDir, remaining, BULLET_MASK);
      let restart = false;

      for (let i = 0; i < hits.length; i++) {
        const h = hits[i]!;
        // Exit records carry no thickness; exits are synthesised from the entry
        // that produced them so a wall's two faces are one decision.
        if (h.thickness <= 0) continue;
        if (h.entityId >= 0 && h.entityId === st.shooterId) continue;

        const isChar = h.entityId >= 0;
        const surface: SurfaceKind = isChar ? 'flesh' : h.surface;
        const distance = st.travelled + h.distance;
        const energy = this.impactEnergy(st, distance);

        this.emitImpact(h.point, h.normal, _traceDir, surface, energy);

        if (isChar) this.applyHit(h, st, distance);

        const penBefore = st.pen;
        const cost = h.thickness * 100 * (PEN_DENSITY[surface] ?? 1);

        if (penBefore <= 0 || cost > penBefore || st.walls >= this.maxWalls) {
          // Not going through. Can it skip off?
          const cosInc = -h.normal.dot(_traceDir);
          if (
            !isChar && st.bounces < this.maxBounces
            && RICOCHET_SURFACE[surface] === true
            && cosInc > 1e-3 && cosInc < RICOCHET_COS
          ) {
            const graze = 1 - cosInc / RICOCHET_COS; // 0 at the threshold, 1 at pure grazing
            const retain = 0.30 + 0.35 * graze;
            st.damageScale *= retain;
            st.stepSpeedScale *= Math.sqrt(retain);
            st.bounces++;

            _reflect.copy(_traceDir).addScaledVector(h.normal, 2 * cosInc).normalize();
            tiltDirection(_reflect, Math.sqrt(Math.random()) * RICOCHET_SCATTER, Math.random() * TAU);
            _traceDir.copy(_reflect);
            _traceOrigin.copy(h.point).addScaledVector(_traceDir, SURFACE_EPSILON);

            st.travelled += h.distance;
            remaining -= h.distance;
            // A round that has skipped is no longer a threat to a target it did
            // not hit head on; drop the ability to also punch walls.
            st.pen *= 0.35;
            restart = true;
            break;
          }

          st.endPoint.copy(h.point);
          st.endDir.copy(_traceDir);
          st.terminated = true;
          return;
        }

        // --- Through it goes -------------------------------------------------
        st.pen = penBefore - cost;
        const used = clamp01(cost / Math.max(0.001, penBefore));
        const retain = PEN_RETAIN_EASY + (PEN_RETAIN_HARD - PEN_RETAIN_EASY) * used;
        st.damageScale *= retain;
        st.stepSpeedScale *= Math.sqrt(retain);
        st.walls++;

        const exitDist = h.distance + h.thickness;
        _exitPoint.copy(_traceOrigin).addScaledVector(_traceDir, exitDist);
        _exitNormal.copy(h.normal).negate();
        this.emitImpact(_exitPoint, _exitNormal, _traceDir, surface, energy * retain);

        // Yaw off-axis by a few milliradians — the deeper the bite, the more it
        // tumbles. This is why through-wall shots need a wider aim.
        tiltDirection(
          _traceDir,
          Math.sqrt(Math.random()) * (DEFLECT_MIN + DEFLECT_SPAN * used),
          Math.random() * TAU,
        );
        _traceOrigin.copy(_exitPoint).addScaledVector(_traceDir, SURFACE_EPSILON);
        st.travelled += exitDist;
        remaining -= exitDist;
        restart = true;
        break;
      }

      if (!restart) {
        st.endPoint.copy(_traceOrigin).addScaledVector(_traceDir, remaining);
        st.endDir.copy(_traceDir);
        st.terminated = false;
        return;
      }
      if (remaining <= 1e-4) break;
    }

    st.endPoint.copy(_traceOrigin);
    st.endDir.copy(_traceDir);
    st.terminated = st.terminated || segments >= MAX_TRACE_SEGMENTS;
  }

  /** 0..1 impact energy for spark/dust intensity. */
  private impactEnergy(st: TraceState, distance: number): number {
    const spec = st.spec;
    const span = Math.max(1, spec.damageRangeFar - spec.damageRangeNear);
    const rangeLoss = 0.45 * clamp01((distance - spec.damageRangeNear) / span);
    return clamp01(st.damageScale * (1 - rangeLoss));
  }

  private damageAtRange(spec: WeaponSpec, distance: number): number {
    if (distance <= spec.damageRangeNear) return spec.damage;
    if (distance >= spec.damageRangeFar) return spec.damage * spec.damageFalloff;
    const t = (distance - spec.damageRangeNear)
      / Math.max(1e-3, spec.damageRangeFar - spec.damageRangeNear);
    return spec.damage * (1 + (spec.damageFalloff - 1) * t);
  }

  private applyHit(h: RayHit, st: TraceState, distance: number): void {
    if (h.entityId === st.damagedEntity) return;
    const target = this.ctx.entities.get(h.entityId);
    if (!target || !target.alive) return;
    st.damagedEntity = h.entityId;

    const hitbox: HitboxKind = h.hitbox ?? 'chest';
    const spec = st.spec;
    const head = hitbox === 'head';
    let amount = this.damageAtRange(spec, distance)
      * (HITBOX_MULTIPLIER[hitbox] ?? 1)
      * st.damageScale;
    if (head) amount *= spec.headshotMultiplier;
    if (amount <= 0) return;

    const wasAlive = target.alive;
    target.applyDamage(amount, hitbox, h.point, st.shooterId);
    const lethal = wasAlive && !target.alive;

    const dmg = this.damageRing.next();
    dmg.targetId = h.entityId;
    dmg.attackerId = st.shooterId;
    dmg.amount = amount;
    dmg.point.copy(h.point);
    dmg.normal.copy(h.normal);
    dmg.direction.copy(_traceDir);
    dmg.hitbox = hitbox;
    dmg.lethal = lethal;
    dmg.weaponId = st.weaponId;
    dmg.distance = distance;
    this.ctx.events.emit('damage.dealt', dmg);

    if (st.local) {
      const hm = this.hitmarkerRing.next();
      hm.headshot = head;
      hm.lethal = lethal;
      this.ctx.events.emit('ui.hitmarker', hm);
    }

    if (lethal) {
      const k = this.killRing.next();
      k.entityId = h.entityId;
      k.killerId = st.shooterId;
      k.weaponId = st.weaponId;
      k.headshot = head;
      this.ctx.events.emit('entity.killed', k);
    }
  }

  private emitImpact(
    point: THREE.Vector3, normal: THREE.Vector3, dir: THREE.Vector3,
    surface: SurfaceKind, energy: number,
  ): void {
    const e = this.impactRing.next();
    e.point.copy(point);
    e.normal.copy(normal);
    e.direction.copy(dir);
    e.surface = surface;
    e.energy = clamp01(energy);
    this.ctx.events.emit('shot.impact', e);
  }

  private emitTracer(from: THREE.Vector3, to: THREE.Vector3, speed: number, weaponId: string): void {
    const t = this.tracerRing.next();
    t.from.copy(from);
    t.to.copy(to);
    t.speed = speed;
    t.weaponId = weaponId;
    this.ctx.events.emit('shot.tracer', t);
  }

  private emitShake(amount: number, duration: number, frequency: number): void {
    if (amount <= 0.001) return;
    const s = this.shakeRing.next();
    s.amount = amount;
    s.duration = duration;
    s.frequency = frequency;
    this.ctx.events.emit('camera.shake', s);
  }

  // -------------------------------------------------------------------------
  // Projectile pool
  // -------------------------------------------------------------------------

  private acquireProjectile(): Projectile | null {
    for (let i = 0; i < this.projectiles.length; i++) {
      const p = this.projectiles[i]!;
      if (!p.active) return p;
    }
    return null;
  }

  // -------------------------------------------------------------------------
  // Fixed step
  // -------------------------------------------------------------------------

  fixedUpdate(step: number, _ctx: GameContext): void {
    this.stepProjectiles(step);
    this.stepGrenades(step);
  }

  private stepProjectiles(step: number): void {
    const st = this.state;
    for (let i = 0; i < this.projectiles.length; i++) {
      const p = this.projectiles[i]!;
      if (!p.active || !p.spec) continue;

      p.age += step;

      // Semi-implicit Euler with quadratic drag: a = -k|v|v - g.
      const speed = p.vel.length();
      if (speed > 1e-4) {
        _accel.copy(p.vel).multiplyScalar(-p.drag * speed);
      } else {
        _accel.set(0, 0, 0);
      }
      _accel.y -= BULLET_GRAVITY;
      p.vel.addScaledVector(_accel, step);

      const speedNow = p.vel.length();
      const segLen = speedNow * step;
      if (segLen <= 1e-6) {
        this.retireProjectile(p);
        continue;
      }
      _stepDir.copy(p.vel).multiplyScalar(1 / speedNow);

      st.spec = p.spec;
      st.weaponId = p.weaponId;
      st.shooterId = p.shooterId;
      st.local = p.local;
      st.travelled = p.travelled;
      st.pen = p.pen;
      st.damageScale = p.damageScale;
      st.bounces = p.bounces;
      st.walls = p.walls;
      st.damagedEntity = -1;

      this.trace(p.pos, _stepDir, segLen, st);

      p.pos.copy(st.endPoint);
      p.travelled += segLen;
      p.pen = st.pen;
      p.damageScale = st.damageScale;
      p.bounces = st.bounces;
      p.walls = st.walls;

      if (st.terminated) {
        this.retireProjectile(p);
        continue;
      }

      // Direction and speed may both have changed inside the segment.
      const newSpeed = speedNow * st.stepSpeedScale;
      p.vel.copy(st.endDir).multiplyScalar(newSpeed);

      // Below ~25 m/s a round has no terminal effect worth simulating.
      if (p.age > 5 || p.travelled > MAX_TRACE_RANGE || newSpeed < 25) {
        this.retireProjectile(p);
      }
    }
  }

  private retireProjectile(p: Projectile): void {
    p.active = false;
    p.spec = null;
  }

  // -------------------------------------------------------------------------
  // Melee & grenades
  // -------------------------------------------------------------------------

  update(time: FrameTime, ctx: GameContext): void {
    const dt = time.dt;
    if (this.meleeCooldown > 0) this.meleeCooldown = Math.max(0, this.meleeCooldown - dt);

    if (this.grenadesLeft < GRENADE_MAX) {
      this.grenadeResupply -= dt;
      if (this.grenadeResupply <= 0) {
        this.grenadesLeft++;
        this.grenadeResupply = GRENADE_RESUPPLY;
      }
    }

    if (this.meleePending >= 0) {
      this.meleePending -= dt;
      if (this.meleePending <= 0) {
        this.meleePending = -1;
        this.resolveMelee();
      }
    }

    if (ctx.paused) {
      // Never leave a cooked frag frozen in hand across a pause.
      this.cooking = false;
      return;
    }

    const input = ctx.input;

    if (input.wasPressed('melee') && this.meleeCooldown <= 0 && this.meleePending < 0) {
      this.meleeCooldown = MELEE_COOLDOWN;
      this.meleePending = MELEE_WINDUP;
      this.emitShake(0.05, 0.18, 30);
    }

    if (input.wasPressed('grenade') && !this.cooking) {
      if (this.grenadesLeft > 0) {
        this.cooking = true;
        this.cookTime = 0;
      } else {
        ctx.events.emit('ui.notify', { text: 'NO FRAGS', kind: 'info' });
      }
    }

    if (this.cooking) {
      this.cookTime += dt;
      // Held too long: it goes off in your hand. That is the deal you made.
      if (this.cookTime >= GRENADE_FUSE) {
        this.cooking = false;
        this.grenadesLeft--;
        this.grenadeResupply = GRENADE_RESUPPLY;
        ctx.camera.getWorldPosition(_camPos);
        this.detonate(_camPos, ctx.localPlayerId, true);
      } else if (input.wasReleased('grenade')) {
        this.cooking = false;
        if (this.throwGrenade(GRENADE_FUSE - this.cookTime)) {
          this.grenadesLeft--;
          this.grenadeResupply = GRENADE_RESUPPLY;
        }
      }
    }
  }

  private resolveMelee(): void {
    const ctx = this.ctx;
    ctx.camera.getWorldPosition(_camPos);
    ctx.camera.getWorldDirection(_camDir);

    // Forgiving reach cone: centre first, then four offsets. Sub-degree misses
    // should not cost a knife kill.
    let best: RayHit | null = null;
    let bestDot = -1;
    for (let i = 0; i < 5; i++) {
      _probeDir.copy(_camDir);
      if (i > 0) {
        const yaw = (i === 1 ? -1 : i === 2 ? 1 : 0) * 0.13;
        const pitch = (i === 3 ? -1 : i === 4 ? 1 : 0) * 0.10;
        tiltDirection(_probeDir, Math.hypot(yaw, pitch), Math.atan2(pitch, yaw));
      }
      const hit = ctx.physics.raycast(_camPos, _probeDir, MELEE_RANGE, BULLET_MASK);
      if (!hit) continue;
      if (hit.entityId < 0 || hit.entityId === ctx.localPlayerId) {
        if (!best && i === 0) best = hit; // remember the wall for the centre swing
        continue;
      }
      const d = _probeDir.dot(_camDir);
      if (!best || best.entityId < 0 || d > bestDot) {
        best = hit;
        bestDot = d;
      }
    }

    if (best && best.entityId >= 0) {
      const target = ctx.entities.get(best.entityId);
      if (target && target.alive) {
        const hitbox: HitboxKind = best.hitbox ?? 'chest';
        const wasAlive = target.alive;
        target.applyDamage(MELEE_DAMAGE, hitbox, best.point, ctx.localPlayerId);
        const lethal = wasAlive && !target.alive;

        const dmg = this.damageRing.next();
        dmg.targetId = best.entityId;
        dmg.attackerId = ctx.localPlayerId;
        dmg.amount = MELEE_DAMAGE;
        dmg.point.copy(best.point);
        dmg.normal.copy(best.normal);
        dmg.direction.copy(_camDir);
        dmg.hitbox = hitbox;
        dmg.lethal = lethal;
        dmg.weaponId = 'melee';
        dmg.distance = best.distance;
        ctx.events.emit('damage.dealt', dmg);

        const hm = this.hitmarkerRing.next();
        hm.headshot = hitbox === 'head';
        hm.lethal = lethal;
        ctx.events.emit('ui.hitmarker', hm);

        if (lethal) {
          const k = this.killRing.next();
          k.entityId = best.entityId;
          k.killerId = ctx.localPlayerId;
          k.weaponId = 'melee';
          k.headshot = false;
          ctx.events.emit('entity.killed', k);
        }

        this.emitImpact(best.point, best.normal, _camDir, 'flesh', 0.9);
        this.emitShake(0.16, 0.22, 26);
        return;
      }
    }

    if (best && best.distance < 1.4) {
      this.emitImpact(best.point, best.normal, _camDir, best.surface, 0.35);
      this.emitShake(0.10, 0.16, 34);
    }
  }

  /** Returns false if no frag could be spawned, so the caller can refund it. */
  private throwGrenade(fuse: number): boolean {
    const g = this.acquireGrenade();
    if (!g || !g.group) return false;
    const ctx = this.ctx;
    ctx.camera.getWorldPosition(_camPos);
    ctx.camera.getWorldDirection(_camDir);
    _camRight.crossVectors(_camDir, _worldUp).normalize();

    g.active = true;
    g.ownerId = ctx.localPlayerId;
    g.local = true;
    g.fuse = Math.max(0.15, fuse);
    g.restTime = 0;
    g.pos.copy(_camPos).addScaledVector(_camDir, 0.42).addScaledVector(_camRight, 0.16);
    g.pos.y -= 0.06;
    g.vel.copy(_camDir).multiplyScalar(GRENADE_THROW_SPEED);
    g.vel.y += GRENADE_LOB;
    // Tumble comes off the wrist: mostly pitch, a little yaw.
    g.spin.set(
      (Math.random() * 2 - 1) * 6 + 14,
      (Math.random() * 2 - 1) * 5,
      (Math.random() * 2 - 1) * 6,
    );
    g.group.position.copy(g.pos);
    g.group.quaternion.identity();
    g.group.visible = true;
    return true;
  }

  private acquireGrenade(): Grenade | null {
    for (let i = 0; i < this.grenades.length; i++) {
      const g = this.grenades[i]!;
      if (!g.active) return g;
    }
    return null;
  }

  private stepGrenades(step: number): void {
    const physics = this.ctx.physics;
    for (let i = 0; i < this.grenades.length; i++) {
      const g = this.grenades[i]!;
      if (!g.active || !g.group) continue;

      g.fuse -= step;
      if (g.fuse <= 0) {
        g.active = false;
        g.group.visible = false;
        this.detonate(g.pos, g.ownerId, g.local);
        continue;
      }

      g.vel.y -= UNITS.gravity * step;
      // Light air drag so a long throw arcs rather than flying like a bullet.
      g.vel.multiplyScalar(1 - Math.min(0.5, 0.22 * step));

      let travel = g.vel.length() * step;
      if (travel > 1e-6) {
        _stepDir.copy(g.vel).multiplyScalar(1 / (travel / step));
        // Sub-step so a fast frag cannot tunnel a thin wall.
        let guard = 0;
        while (travel > 1e-5 && guard++ < 4) {
          const hit = physics.raycast(g.pos, _stepDir, travel + GRENADE_RADIUS, RayMask.Solid);
          if (!hit || hit.distance > travel + GRENADE_RADIUS) {
            g.pos.addScaledVector(_stepDir, travel);
            break;
          }
          const advance = Math.max(0, hit.distance - GRENADE_RADIUS);
          g.pos.addScaledVector(_stepDir, advance);
          travel -= advance;

          const vn = g.vel.dot(hit.normal);
          if (vn < 0) {
            // Split into normal and tangential: bounce one, scrub the other.
            g.vel.addScaledVector(hit.normal, -(1 + GRENADE_RESTITUTION) * vn);
            const vnAfter = g.vel.dot(hit.normal);
            _reflect.copy(g.vel).addScaledVector(hit.normal, -vnAfter);
            g.vel.copy(hit.normal).multiplyScalar(vnAfter)
              .addScaledVector(_reflect, GRENADE_FRICTION);
            // Impact torque: the tangential scrub becomes spin.
            _spinAxis.crossVectors(hit.normal, _reflect);
            const sl = _spinAxis.length();
            if (sl > 1e-4) g.spin.addScaledVector(_spinAxis.multiplyScalar(1 / sl), Math.min(24, _reflect.length() * 5));
            g.spin.multiplyScalar(0.55);

            const speed = Math.abs(vn);
            if (speed > 1.2) {
              this.emitImpact(g.pos, hit.normal, _stepDir, hit.surface, clamp01(speed * 0.08));
            }
          }
          g.pos.addScaledVector(hit.normal, 0.002);

          const remainingSpeed = g.vel.length();
          if (remainingSpeed < 0.45) {
            g.vel.multiplyScalar(0.2);
            g.spin.multiplyScalar(0.25);
            break;
          }
          _stepDir.copy(g.vel).normalize();
          travel = Math.min(travel, remainingSpeed * step);
        }
      }

      // Rolling resistance once it is basically on the deck.
      if (g.vel.lengthSq() < 4) {
        g.vel.multiplyScalar(1 - Math.min(0.9, 2.4 * step));
        g.spin.multiplyScalar(1 - Math.min(0.9, 3.0 * step));
      }

      g.group.position.copy(g.pos);
      const spinMag = g.spin.length();
      if (spinMag > 1e-4) {
        _spinAxis.copy(g.spin).multiplyScalar(1 / spinMag);
        _spinQuat.setFromAxisAngle(_spinAxis, spinMag * step);
        g.group.quaternion.premultiply(_spinQuat);
      }
    }
  }

  private detonate(at: THREE.Vector3, attackerId: number, local: boolean): void {
    const ctx = this.ctx;
    // Snapshot first: callers legitimately pass scratch vectors (a cook-off
    // passes the camera position) that this method goes on to overwrite.
    const point = _blastPoint.copy(at);

    const ex = this.explosionRing.next();
    ex.point.copy(point);
    ex.radius = GRENADE_BLAST_RADIUS;
    ex.damage = GRENADE_BLAST_DAMAGE;
    ex.attackerId = attackerId;
    ctx.events.emit('explosion', ex);

    // Scorch the ground under the blast so the crater is not just particles.
    _probe.copy(point);
    _probe.y += 0.1;
    const ground = ctx.physics.raycast(_probe, _down, 2.2, RayMask.Solid);
    if (ground) this.emitImpact(ground.point, ground.normal, _down, ground.surface, 1);

    _blastFrom.copy(point);
    _blastFrom.y += 0.18; // lift out of the floor so LOS is not self-blocked

    const targets = ctx.entities.all;
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i]!;
      if (!t.alive) continue;
      _targetCentre.copy(t.position);
      _targetCentre.y += UNITS.eyeOffset * 0.55;
      const d = _targetCentre.distanceTo(point);
      if (d > GRENADE_BLAST_RADIUS) continue;

      // Slightly super-linear falloff: lethal in the pocket, survivable at the lip.
      let atten = Math.pow(1 - d / GRENADE_BLAST_RADIUS, 1.35);
      if (d > 0.6 && !ctx.physics.hasLineOfSight(_blastFrom, _targetCentre, RayMask.Solid)) {
        atten *= BLAST_OCCLUDED;
      }
      const amount = GRENADE_BLAST_DAMAGE * atten;
      if (amount < 1) continue;

      _probeDir.copy(_targetCentre).sub(point);
      const dist = _probeDir.length() || 1;
      _probeDir.multiplyScalar(1 / dist);

      const wasAlive = t.alive;
      t.applyDamage(amount, 'chest', point, attackerId);
      const lethal = wasAlive && !t.alive;

      const dmg = this.damageRing.next();
      dmg.targetId = t.entityId;
      dmg.attackerId = attackerId;
      dmg.amount = amount;
      dmg.point.copy(_targetCentre);
      dmg.normal.copy(_probeDir).negate();
      dmg.direction.copy(_probeDir);
      dmg.hitbox = 'chest';
      dmg.lethal = lethal;
      dmg.weaponId = 'frag';
      dmg.distance = d;
      ctx.events.emit('damage.dealt', dmg);

      if (local && t.entityId !== ctx.localPlayerId) {
        const hm = this.hitmarkerRing.next();
        hm.headshot = false;
        hm.lethal = lethal;
        ctx.events.emit('ui.hitmarker', hm);
      }

      if (lethal) {
        const k = this.killRing.next();
        k.entityId = t.entityId;
        k.killerId = attackerId;
        k.weaponId = 'frag';
        k.headshot = false;
        ctx.events.emit('entity.killed', k);
      }
    }

    ctx.camera.getWorldPosition(_camPos);
    const camDist = _camPos.distanceTo(point);
    if (camDist < GRENADE_BLAST_RADIUS * 3) {
      const falloff = clamp01(1 - camDist / (GRENADE_BLAST_RADIUS * 3));
      this.emitShake(0.35 + 0.95 * falloff * falloff, 0.55 + 0.35 * falloff, 18);
    }
  }

  // -------------------------------------------------------------------------
  // Frag mesh
  // -------------------------------------------------------------------------

  private buildFragResources(): void {
    // Ovoid body: an M67 is 64mm across and 90mm tall including the fuse.
    const body = new THREE.IcosahedronGeometry(0.032, 2);
    body.scale(1, 1.16, 1);
    this.fragBody = body;

    const lever = new THREE.BoxGeometry(0.007, 0.052, 0.011);
    lever.translate(0.0, 0.030, 0.028);
    this.fragLever = lever;

    this.fragNormal = makeFragNormalMap(64);

    this.fragMat = new THREE.MeshStandardMaterial({
      color: 0x39432c,
      roughness: 0.68,
      metalness: 0.42,
      normalMap: this.fragNormal,
    });
    if (this.fragNormal) {
      this.fragNormal.repeat.set(2, 2);
      this.fragMat.normalScale.set(1.35, 1.35);
    }
    this.fragLeverMat = new THREE.MeshStandardMaterial({
      color: 0x8a8f96,
      roughness: 0.34,
      metalness: 0.88,
    });
  }

  private makeFragMesh(): THREE.Group {
    const group = new THREE.Group();
    group.name = 'frag';
    const body = new THREE.Mesh(this.fragBody!, this.fragMat!);
    body.castShadow = true;
    group.add(body);
    const lever = new THREE.Mesh(this.fragLever!, this.fragLeverMat!);
    lever.castShadow = true;
    group.add(lever);
    group.matrixAutoUpdate = true;
    return group;
  }

  // -------------------------------------------------------------------------

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;

    for (const g of this.grenades) {
      if (g.group) {
        g.group.removeFromParent();
        g.group.clear();
      }
      g.group = null;
      g.active = false;
    }
    this.grenades.length = 0;

    for (const p of this.projectiles) {
      p.active = false;
      p.spec = null;
    }
    this.projectiles.length = 0;

    this.fragBody?.dispose();
    this.fragLever?.dispose();
    this.fragMat?.dispose();
    this.fragLeverMat?.dispose();
    this.fragNormal?.dispose();
    this.fragBody = null;
    this.fragLever = null;
    this.fragMat = null;
    this.fragLeverMat = null;
    this.fragNormal = null;

    this.tracerCount.clear();
  }
}
