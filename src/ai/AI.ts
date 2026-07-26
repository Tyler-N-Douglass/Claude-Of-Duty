/**
 * AISystem — the opposing squad.
 *
 * Responsibilities:
 *   1. Bakes a navigation grid off the physics world once the level exists.
 *   2. Spawns, steers, animates and renders every enemy soldier.
 *   3. Runs perception (cone + line of sight + awareness ramp), a combat FSM,
 *      and fires through the shared ballistics pipeline by emitting
 *      `shot.fired`, so enemy rounds penetrate, ricochet and fall off exactly
 *      like the player's.
 *
 * Nothing here owns its own damage maths, its own tracers or its own impact
 * effects — those all belong to other systems and arrive over the event bus.
 *
 * Cost model: agents are limited by quality preset; pathfinding is staggered so
 * at most two A* searches run per frame; perception runs on a 4-frame rotation;
 * rendering is ten instanced draw calls total.
 */
import * as THREE from 'three';
import {
  RayMask,
  type DamageTarget,
  type FrameTime,
  type GameContext,
  type HitboxKind,
  type CharacterHitbox,
  type System,
  type WeaponSpec,
} from '../core/Contracts';
import { allocEntityId } from '../core/Entities';
import { NavGrid, PathFollower } from './Navigation';
import { PART, SoldierRenderer, mulberry32 } from './Soldier';
import { getSpec } from '../weapons/WeaponSpecs';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const AGENT_HEIGHT = 1.8;
/**
 * Hip-joint height. Set so the sole of the boot geometry lands on y=0 with the
 * leg chain extended: 0.055 (hip drop) + 0.42 (thigh) + 0.42 (shin) + 0.529
 * (boot + sole below the knee) measured off the authored parts.
 */
const HIP_HEIGHT = 1.004;
const AGENT_RADIUS = 0.34;
const EYE_HEIGHT = 1.62;
const CHEST_HEIGHT = 1.36;
const MUZZLE_HEIGHT = 1.38;

const MAX_HEALTH = 100;

const WALK_SPEED = 2.05;
const RUN_SPEED = 4.35;
const ACCEL = 14;
const TURN_RATE = 6.5;

const SIGHT_RANGE = 72;
const FOV_COS = Math.cos(1.05); // ±60°, so a 120° cone
const PERIPHERAL_COS = Math.cos(1.6);

const REACTION_MIN = 0.26;
const REACTION_MAX = 0.52;

const AIM_ERROR_INITIAL = 0.055; // radians, ~3.1°
const AIM_ERROR_SETTLED = 0.011;
const AIM_SETTLE_TIME = 1.35;

const LOSE_TARGET_TIME = 2.4;
const CORPSE_TIME = 18;
const RESPAWN_DELAY = 6;

const REPATH_INTERVAL = 0.85;
const PATHS_PER_FRAME = 2;

const SEPARATION_RADIUS = 1.15;

const ENEMY_WEAPONS = ['mk4_ranger', 'vks9_wasp', 'm16br'] as const;

type AgentState = 'idle' | 'patrol' | 'alert' | 'engage' | 'reposition' | 'dead';

// Module-scope scratch. Nothing in the per-frame path allocates.
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _eye = new THREE.Vector3();
const _muz = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _colorTmp = new THREE.Color();

function angleDelta(a: number, b: number): number {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

/** Critically-damped spring step. Frame-rate independent, no overshoot ringing. */
function damp(current: number, target: number, lambda: number, dt: number): number {
  return current + (target - current) * (1 - Math.exp(-lambda * dt));
}

function easeOutBack(t: number): number {
  const c = 1.70158;
  const u = t - 1;
  return 1 + (c + 1) * u * u * u + c * u * u;
}

interface LevelLike extends System {
  getSpawnPoints(): { position: THREE.Vector3; yaw: number }[];
  getCoverPoints(): { position: THREE.Vector3; normal: THREE.Vector3 }[];
  getNavBounds(): THREE.Box3;
}

interface PlayerLike extends System {
  readonly position: THREE.Vector3;
  readonly alive: boolean;
}

// ---------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------

/**
 * A soldier's joint hierarchy. Each named node is the parent space its body
 * part's geometry was authored in, so posing is pure joint rotation and the
 * physics hitboxes ride along for free.
 */
class Rig {
  readonly root = new THREE.Group();
  readonly hips = new THREE.Group();
  readonly pelvis = new THREE.Group();
  readonly spine = new THREE.Group();
  readonly neck = new THREE.Group();
  readonly head = new THREE.Group();
  readonly helmet = new THREE.Group();
  readonly shoulder: [THREE.Group, THREE.Group] = [new THREE.Group(), new THREE.Group()];
  readonly elbow: [THREE.Group, THREE.Group] = [new THREE.Group(), new THREE.Group()];
  readonly hip: [THREE.Group, THREE.Group] = [new THREE.Group(), new THREE.Group()];
  readonly knee: [THREE.Group, THREE.Group] = [new THREE.Group(), new THREE.Group()];
  readonly weapon = new THREE.Group();
  readonly pack = new THREE.Group();
  readonly muzzle = new THREE.Object3D();

  constructor() {
    this.root.matrixAutoUpdate = true;
    this.root.add(this.hips);
    this.hips.position.set(0, HIP_HEIGHT, 0);
    this.hips.add(this.pelvis);
    this.pelvis.add(this.spine);
    this.spine.position.set(0, 0.115, 0);

    this.spine.add(this.neck);
    this.neck.position.set(0, 0.555, 0.012);
    this.neck.add(this.head);
    this.head.add(this.helmet);
    this.helmet.position.set(0, 0.028, -0.006);

    for (let s = 0; s < 2; s++) {
      const sign = s === 0 ? -1 : 1;
      this.spine.add(this.shoulder[s]);
      this.shoulder[s].position.set(sign * 0.178, 0.44, 0);
      this.shoulder[s].add(this.elbow[s]);
      this.elbow[s].position.set(0, -0.27, 0);

      this.hips.add(this.hip[s]);
      this.hip[s].position.set(sign * 0.098, -0.055, 0);
      this.hip[s].add(this.knee[s]);
      this.knee[s].position.set(0, -0.42, 0);
    }

    this.spine.add(this.pack);
    this.pack.position.set(0, 0.28, -0.185);

    this.spine.add(this.weapon);
    this.weapon.add(this.muzzle);
    this.muzzle.position.set(0, 0.05, -0.74);
  }

  hitboxes(): CharacterHitbox[] {
    const box = (
      kind: HitboxKind, attachTo: THREE.Object3D,
      ox: number, oy: number, oz: number,
      hx: number, hy: number, hz: number, mult: number,
    ): CharacterHitbox => ({
      kind,
      attachTo,
      offset: new THREE.Vector3(ox, oy, oz),
      size: new THREE.Vector3(hx, hy, hz),
      damageMultiplier: mult,
    });
    return [
      box('head', this.head, 0, 0.005, 0.005, 0.098, 0.115, 0.105, 4.2),
      box('chest', this.spine, 0, 0.315, 0, 0.21, 0.16, 0.135, 1.0),
      box('stomach', this.spine, 0, 0.10, 0, 0.175, 0.125, 0.12, 1.15),
      box('arm', this.shoulder[0], 0, -0.19, 0, 0.075, 0.235, 0.075, 0.7),
      box('arm', this.shoulder[1], 0, -0.19, 0, 0.075, 0.235, 0.075, 0.7),
      box('leg', this.hip[0], 0, -0.36, 0, 0.10, 0.40, 0.10, 0.75),
      box('leg', this.hip[1], 0, -0.36, 0, 0.10, 0.40, 0.10, 0.75),
    ];
  }
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

class Agent implements DamageTarget {
  readonly entityId: number;
  readonly index: number;
  readonly rig = new Rig();
  readonly path = new PathFollower();

  alive = true;
  health = MAX_HEALTH;

  /** Chest-height world position — what other systems aim at. */
  readonly position = new THREE.Vector3();
  /** Capsule base (feet). */
  readonly feet = new THREE.Vector3();
  readonly velocity = new THREE.Vector3();
  readonly home = new THREE.Vector3();
  readonly destination = new THREE.Vector3();
  readonly lastKnown = new THREE.Vector3();

  state: AgentState = 'idle';
  stateTime = 0;

  yaw = 0;
  targetYaw = 0;
  aimYaw = 0;
  aimPitch = 0;
  upperTwist = 0;
  headTwist = 0;

  awareness = 0;
  hasLos = false;
  losTime = 0;
  noLosTime = 999;
  reaction = 0;
  reactionLeft = 0;

  spec: WeaponSpec;
  weaponId: string;
  ammo: number;
  burstLeft = 0;
  fireCooldown = 0;
  burstCooldown = 0;
  reloadLeft = 0;

  stridePhase = 0;
  strideSpeed = 0;
  speedSmooth = 0;
  breath = 0;
  flinch = 0;
  flinchAxis = 0;
  crouch = 0;
  crouchTarget = 0;

  deathTime = 0;
  deathPitch = 0;
  deathRoll = 0;
  deathSpin = 0;
  respawnLeft = 0;

  repathIn = 0;
  wantsPath = false;

  readonly tint = new THREE.Color(1, 1, 1);
  readonly scale: number;

  private readonly onKilled: (a: Agent, attackerId: number) => void;

  constructor(index: number, spec: WeaponSpec, weaponId: string, scale: number, onKilled: (a: Agent, attackerId: number) => void) {
    this.index = index;
    this.entityId = allocEntityId();
    this.spec = spec;
    this.weaponId = weaponId;
    this.ammo = spec.magSize;
    this.scale = scale;
    this.onKilled = onKilled;
    this.rig.root.scale.setScalar(scale);
  }

  applyDamage(amount: number, hitbox: HitboxKind, from: THREE.Vector3, attackerId: number): void {
    if (!this.alive) return;
    this.health -= amount;

    // Flinch is signed by which side of the body took the round, so a hit from
    // the left rocks the torso right. Head hits snap harder and recover faster.
    _v0.copy(from).sub(this.position);
    const lateral = _v0.x * Math.cos(this.yaw) - _v0.z * Math.sin(this.yaw);
    this.flinchAxis = lateral >= 0 ? -1 : 1;
    this.flinch = Math.min(1, this.flinch + (hitbox === 'head' ? 0.9 : 0.55));

    // Being shot at from an unseen angle still tells you where the shooter is.
    this.awareness = 1;
    if (this.noLosTime > 0.2) this.lastKnown.copy(from);

    if (this.health <= 0) {
      this.health = 0;
      this.alive = false;
      this.onKilled(this, attackerId);
    }
  }
}

// ---------------------------------------------------------------------------
// Muzzle flash pool (enemy fire needs to read at a glance across the street)
// ---------------------------------------------------------------------------

function flashTexture(): THREE.CanvasTexture {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const g = c.getContext('2d');
  if (g) {
    const img = g.createImageData(size, size);
    const px = img.data;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = (x + 0.5) / size - 0.5;
        const dy = (y + 0.5) / size - 0.5;
        const r = Math.hypot(dx, dy) * 2;
        // Core + four-point star: a bare radial blob reads as a bokeh circle.
        const ang = Math.atan2(dy, dx);
        const star = 0.55 + 0.45 * Math.abs(Math.cos(ang * 2));
        const a = Math.max(0, 1 - r / star);
        const v = a * a * a;
        const i = (y * size + x) * 4;
        px[i] = 255;
        px[i + 1] = Math.round(190 + 65 * v);
        px[i + 2] = Math.round(110 + 80 * v);
        px[i + 3] = Math.round(Math.min(1, v * 1.6) * 255);
      }
    }
    g.putImageData(img, 0, 0);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

interface Flash {
  sprite: THREE.Sprite;
  life: number;
  duration: number;
  size: number;
  roll: number;
}

// ---------------------------------------------------------------------------

export class AISystem implements System {
  readonly name = 'ai';

  private ctx: GameContext | null = null;
  private nav: NavGrid | null = null;
  private renderer: SoldierRenderer | null = null;
  private readonly agents: Agent[] = [];
  private readonly rigRoot = new THREE.Group();
  private readonly cover: { position: THREE.Vector3; normal: THREE.Vector3 }[] = [];
  private readonly spawnPoints: { position: THREE.Vector3; yaw: number }[] = [];

  private readonly rng = mulberry32(0xa19f00d);
  private readonly pathScratch: THREE.Vector3[] = [];
  private pathCursor = 0;
  private perceptionCursor = 0;

  private flashes: Flash[] = [];
  private flashTex: THREE.Texture | null = null;
  private flashMat: THREE.SpriteMaterial | null = null;
  private flashPool = 0;

  private player: PlayerLike | null = null;
  private playerPos = new THREE.Vector3();
  private playerAlive = true;

  private readonly unsubs: (() => void)[] = [];
  private navReady = false;

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;
    this.rigRoot.name = 'ai-rigs';
    // The rig hierarchy carries hitboxes and drives instancing; the meshes are
    // drawn by the instanced renderer, so the joints themselves never render.
    this.rigRoot.visible = false;
    ctx.scene.add(this.rigRoot);

    const level = ctx.system<LevelLike>('level');
    const bounds = level?.getNavBounds() ?? new THREE.Box3(
      new THREE.Vector3(-42, -2, -42), new THREE.Vector3(42, 26, 42),
    );

    // 0.55m cells: below the 0.68m agent diameter so doorways stay open, coarse
    // enough that the bake stays under ~25k probes on this level footprint.
    const nav = new NavGrid();
    nav.build(ctx.physics, bounds, 0.55);
    this.nav = nav;
    this.navReady = nav.ready;

    if (level) {
      for (const c of level.getCoverPoints()) this.cover.push(c);
      for (const s of level.getSpawnPoints()) this.spawnPoints.push(s);
    }

    const count = this.agentBudget(ctx);
    this.renderer = new SoldierRenderer(count, ctx.quality);
    ctx.scene.add(this.renderer.root);

    for (let i = 0; i < count; i++) {
      const weaponId = ENEMY_WEAPONS[i % ENEMY_WEAPONS.length];
      const spec = getSpec(weaponId);
      if (!spec) continue;
      // Real squads are not one height. ±6% reads clearly without breaking the
      // shared hitbox proportions.
      const scale = 0.955 + this.rng() * 0.11;
      const agent = new Agent(i, spec, weaponId, scale, this.onAgentKilled);
      this.agents.push(agent);
      this.rigRoot.add(agent.rig.root);

      // Per-soldier fabric tint: a small hue/value spread across the squad.
      const warm = 0.9 + this.rng() * 0.22;
      _colorTmp.setRGB(warm, 0.92 + this.rng() * 0.16, 0.86 + this.rng() * 0.2);
      agent.tint.copy(_colorTmp);
      this.renderer.setTint(i, _colorTmp.r, _colorTmp.g, _colorTmp.b);

      ctx.entities.register(agent);
      ctx.physics.addCharacter(agent.entityId, agent.rig.root, agent.rig.hitboxes());
      this.placeAtSpawn(agent, true);
    }

    // Open the round with the squad already converging. A level that takes half
    // a minute to become a firefight reads as an empty diorama.
    const player = ctx.system<PlayerLike>('player');
    if (player) this.playerPos.copy(player.position);
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      a.lastKnown.copy(this.playerPos);
      a.awareness = 0.45;
      a.state = 'alert';
      a.repathIn = i * 0.07;
    }

    this.buildFlashPool(ctx);

    this.unsubs.push(ctx.events.on('shot.fired', this.onShotFired));
    this.unsubs.push(ctx.events.on('explosion', this.onExplosion));

    // One pose pass so the very first rendered frame has soldiers in it rather
    // than ten zero-scale instances.
    this.poseAll(0.016, 0);
  }

  dispose(): void {
    for (const off of this.unsubs) off();
    this.unsubs.length = 0;

    const ctx = this.ctx;
    for (const a of this.agents) {
      ctx?.entities.unregister(a.entityId);
      ctx?.physics.removeCharacter(a.entityId);
      a.rig.root.parent?.remove(a.rig.root);
    }
    this.agents.length = 0;

    for (const f of this.flashes) {
      f.sprite.parent?.remove(f.sprite);
      f.sprite.material.dispose();
    }
    this.flashes.length = 0;
    this.flashMat?.dispose();
    this.flashTex?.dispose();
    this.flashMat = null;
    this.flashTex = null;

    this.renderer?.dispose();
    this.renderer = null;
    this.nav?.dispose();
    this.nav = null;
    this.rigRoot.parent?.remove(this.rigRoot);
    this.ctx = null;
  }

  private agentBudget(ctx: GameContext): number {
    switch (ctx.quality.preset) {
      case 'low': return 6;
      case 'medium': return 8;
      case 'ultra': return 14;
      default: return 11;
    }
  }

  // =========================================================================
  // Spawning
  // =========================================================================

  private placeAtSpawn(agent: Agent, initial: boolean): void {
    const ctx = this.ctx;
    if (!ctx) return;

    let best: THREE.Vector3 | null = null;
    let bestScore = -Infinity;
    const ref = initial ? _v0.set(0, 1, 8) : this.playerPos;

    for (let attempt = 0; attempt < 16; attempt++) {
      let candidate: THREE.Vector3 | null = null;
      if (this.spawnPoints.length > 0) {
        const sp = this.spawnPoints[(this.rng() * this.spawnPoints.length) | 0];
        candidate = _v1.copy(sp.position);
      } else if (this.nav?.ready && this.nav.randomPoint(_v1)) {
        candidate = _v1;
      }
      if (!candidate) continue;

      const d = candidate.distanceTo(ref);
      // Far enough that nobody materialises in the player's face, near enough
      // that contact happens in the first few seconds rather than after a walk.
      let score = d < 11 ? d - 40 : -Math.abs(d - 21);
      // Squads that all pile onto the single best-scoring spawn read as one
      // enemy with a rendering bug. Penalise anywhere a living agent already is.
      for (const other of this.agents) {
        if (other === agent || !other.alive) continue;
        const od = candidate.distanceTo(other.feet);
        if (od < 9) score -= (9 - od) * 3.5;
      }
      score += this.rng() * 2.5;
      if (score > bestScore) {
        bestScore = score;
        best = (best ?? new THREE.Vector3()).copy(candidate);
      }
    }

    // Scatter within the spawn's own footprint so two soldiers never stand in
    // exactly the same boot prints.
    if (best && this.nav?.ready && this.nav.randomPointNear(best, 3.5, _v2)) best.copy(_v2);

    if (!best) {
      best = new THREE.Vector3(
        (this.rng() - 0.5) * 30,
        2,
        -12 - this.rng() * 18,
      );
    }

    const ground = this.nav?.groundAt(best.x, best.z);
    const y = ground !== null && ground !== undefined && Number.isFinite(ground)
      ? ground
      : (ctx.physics.groundHeight(_v2.set(best.x, best.y + 6, best.z), 30) ?? best.y);

    agent.feet.set(best.x, y, best.z);
    agent.velocity.set(0, 0, 0);
    agent.home.copy(agent.feet);
    agent.destination.copy(agent.feet);
    agent.lastKnown.copy(agent.feet);
    agent.alive = true;
    agent.health = MAX_HEALTH;
    agent.state = 'idle';
    agent.stateTime = 0;
    agent.awareness = 0;
    agent.hasLos = false;
    agent.losTime = 0;
    agent.noLosTime = 999;
    agent.ammo = agent.spec.magSize;
    agent.burstLeft = 0;
    agent.fireCooldown = 0;
    agent.burstCooldown = 0.6;
    agent.reloadLeft = 0;
    agent.flinch = 0;
    agent.crouch = 0;
    agent.crouchTarget = 0;
    agent.deathTime = 0;
    agent.respawnLeft = 0;
    agent.path.clear();
    agent.repathIn = this.rng() * 0.6;
    agent.yaw = Math.atan2(ref.x - best.x, ref.z - best.z) + Math.PI;
    agent.targetYaw = agent.yaw;
    agent.aimYaw = agent.yaw;
    agent.aimPitch = 0;
    agent.stridePhase = this.rng() * Math.PI * 2;
    agent.breath = this.rng() * Math.PI * 2;
    agent.position.set(agent.feet.x, agent.feet.y + CHEST_HEIGHT * agent.scale, agent.feet.z);

    this.ctx?.physics.addCharacter(agent.entityId, agent.rig.root, agent.rig.hitboxes());
  }

  private readonly onAgentKilled = (agent: Agent, _attackerId: number): void => {
    agent.state = 'dead';
    agent.stateTime = 0;
    agent.deathTime = 0;
    agent.respawnLeft = CORPSE_TIME + RESPAWN_DELAY;
    agent.velocity.multiplyScalar(0.35);
    // A body does not fall the same way twice: the collapse direction is seeded
    // from the momentum it had plus a random tumble bias.
    agent.deathPitch = this.rng() < 0.55 ? 1 : -1;
    agent.deathRoll = (this.rng() - 0.5) * 1.6;
    agent.deathSpin = (this.rng() - 0.5) * 2.4;
    agent.path.clear();
    this.ctx?.physics.removeCharacter(agent.entityId);
  };

  // =========================================================================
  // Event reactions
  // =========================================================================

  private readonly onShotFired = (ev: {
    origin: THREE.Vector3; direction: THREE.Vector3; local: boolean; shooterId: number;
  }): void => {
    if (!ev.local) return;
    // Gunfire is the loudest thing in the level: everyone within earshot turns
    // toward it, and anyone already suspicious commits.
    for (const a of this.agents) {
      if (!a.alive) continue;
      const d = a.position.distanceTo(ev.origin);
      if (d > 55) continue;
      a.awareness = Math.min(1, a.awareness + (d < 22 ? 0.85 : 0.45));
      if (a.noLosTime > 0.4) a.lastKnown.copy(ev.origin);
      if (a.state === 'idle' || a.state === 'patrol') {
        a.state = 'alert';
        a.stateTime = 0;
        a.repathIn = 0;
      }
    }
  };

  private readonly onExplosion = (ev: { point: THREE.Vector3; radius: number }): void => {
    for (const a of this.agents) {
      if (!a.alive) continue;
      const d = a.position.distanceTo(ev.point);
      if (d > ev.radius * 2.5) continue;
      a.awareness = 1;
      a.lastKnown.copy(ev.point);
      a.crouchTarget = 1;
      a.flinch = Math.min(1, a.flinch + 0.5);
    }
  };

  // =========================================================================
  // Frame
  // =========================================================================

  update(time: FrameTime, ctx: GameContext): void {
    if (ctx.paused) return;
    const dt = Math.min(time.dt, 1 / 20);

    if (!this.player) this.player = ctx.system<PlayerLike>('player') ?? null;
    if (this.player) {
      this.playerPos.copy(this.player.position);
      this.playerAlive = this.player.alive;
    } else {
      this.playerPos.copy(ctx.camera.position);
      this.playerAlive = true;
    }

    this.pathCursor = 0;
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      a.stateTime += dt;
      if (a.state === 'dead') {
        this.updateDead(a, dt);
        continue;
      }
      // Perception is expensive (a raycast each); rotate through the squad so
      // only a quarter of them pay for it on any given frame.
      if ((time.frame + i) % 4 === 0) this.perceive(a, dt * 4);
      this.think(a, dt);
      this.steer(a, dt, ctx);
      this.combat(a, dt, ctx);
    }

    this.updateFlashes(dt);
    this.poseAll(dt, time.elapsed);
  }

  // =========================================================================
  // Perception
  // =========================================================================

  private perceive(a: Agent, dt: number): void {
    const ctx = this.ctx;
    if (!ctx || !this.playerAlive) {
      a.hasLos = false;
      a.noLosTime += dt;
      a.awareness = Math.max(0, a.awareness - dt * 0.25);
      return;
    }

    _eye.set(a.feet.x, a.feet.y + EYE_HEIGHT * a.scale - a.crouch * 0.42, a.feet.z);
    _dir.copy(this.playerPos).sub(_eye);
    const dist = _dir.length();

    let visible = false;
    if (dist < SIGHT_RANGE && dist > 0.001) {
      _dir.multiplyScalar(1 / dist);
      const facingX = -Math.sin(a.aimYaw);
      const facingZ = -Math.cos(a.aimYaw);
      const dot = _dir.x * facingX + _dir.z * facingZ;
      const cone = a.awareness > 0.5 ? PERIPHERAL_COS : FOV_COS;
      if (dot > cone) {
        visible = ctx.physics.hasLineOfSight(_eye, this.playerPos, RayMask.Solid);
      }
    }

    a.hasLos = visible;
    if (visible) {
      a.noLosTime = 0;
      a.losTime += dt;
      a.lastKnown.copy(this.playerPos);
      // Detection speed: close and centred is near-instant, far and peripheral
      // takes a beat. This is the difference between "fair" and "aimbot".
      const rate = THREE.MathUtils.clamp(2.6 - dist * 0.022, 0.55, 2.6);
      if (a.awareness < 1) {
        a.awareness = Math.min(1, a.awareness + rate * dt);
        if (a.awareness >= 1) {
          a.reaction = REACTION_MIN + this.rng() * (REACTION_MAX - REACTION_MIN);
          a.reactionLeft = a.reaction;
        }
      }
    } else {
      a.losTime = 0;
      a.noLosTime += dt;
      a.awareness = Math.max(0, a.awareness - dt * 0.16);
    }
  }

  // =========================================================================
  // Decision
  // =========================================================================

  private think(a: Agent, dt: number): void {
    a.repathIn -= dt;

    switch (a.state) {
      case 'idle':
        if (a.awareness >= 1) this.enter(a, 'engage');
        else if (a.awareness > 0.15) this.enter(a, 'alert');
        else if (a.stateTime > 2.5 + this.rng() * 3) this.enter(a, 'patrol');
        break;

      case 'patrol':
        if (a.awareness >= 1) this.enter(a, 'engage');
        else if (a.awareness > 0.2) this.enter(a, 'alert');
        else if (a.path.finished || a.stateTime > 16) this.enter(a, 'idle');
        break;

      case 'alert':
        if (a.awareness >= 1 && a.hasLos) this.enter(a, 'engage');
        else if (a.awareness <= 0.02 && a.stateTime > 6) this.enter(a, 'idle');
        break;

      case 'engage':
        if (a.noLosTime > LOSE_TARGET_TIME) this.enter(a, 'alert');
        // Wounded soldiers break contact and find something solid.
        else if (a.health < 42 && a.stateTime > 1.4 && this.cover.length > 0) this.enter(a, 'reposition');
        // Standing in the open trading shots is how a bot dies; move every few
        // seconds even when winning.
        else if (a.stateTime > 4.5 + this.rng() * 3) this.enter(a, 'reposition');
        break;

      case 'reposition':
        if (a.path.finished || a.stateTime > 5) this.enter(a, a.hasLos ? 'engage' : 'alert');
        break;

      default:
        break;
    }

    // Repath, budgeted. `pathCursor` is reset each frame so at most
    // PATHS_PER_FRAME A* searches run no matter how many agents want one.
    if (a.repathIn <= 0 && this.pathCursor < PATHS_PER_FRAME) {
      a.repathIn = REPATH_INTERVAL * (0.75 + this.rng() * 0.6);
      if (this.chooseDestination(a)) {
        this.pathCursor++;
        this.repath(a);
      }
    }
    a.path.age += dt;
  }

  private enter(a: Agent, state: AgentState): void {
    if (a.state === state) return;
    a.state = state;
    a.stateTime = 0;
    a.repathIn = 0;
    if (state === 'engage') {
      a.burstCooldown = Math.max(a.burstCooldown, 0.12 + this.rng() * 0.2);
      a.crouchTarget = this.rng() < 0.25 ? 1 : 0;
    } else if (state === 'reposition') {
      a.crouchTarget = 0;
    }
  }

  /** Returns true if the destination changed enough to justify a new search. */
  private chooseDestination(a: Agent): boolean {
    const nav = this.nav;
    _v0.copy(a.destination);

    switch (a.state) {
      case 'idle':
        a.destination.copy(a.feet);
        break;

      case 'patrol':
        if (nav?.ready && nav.randomPointNear(a.home, 14, _v1)) a.destination.copy(_v1);
        else a.destination.copy(a.home);
        break;

      case 'alert':
        // Push to the last known position, but stop short of standing on it.
        _v1.copy(a.lastKnown).sub(a.feet);
        _v1.y = 0;
        if (_v1.lengthSq() > 9) {
          _v1.setLength(Math.max(0, _v1.length() - 3.5));
          a.destination.copy(a.feet).add(_v1);
        } else {
          a.destination.copy(a.lastKnown);
        }
        break;

      case 'engage': {
        // Hold ground but shuffle: strafe within a small radius of where the
        // fight started so the silhouette never freezes.
        const side = this.rng() < 0.5 ? -1 : 1;
        const ang = a.aimYaw + Math.PI * 0.5 * side;
        const step = 1.4 + this.rng() * 2.2;
        _v1.set(a.feet.x + Math.sin(ang) * step, a.feet.y, a.feet.z + Math.cos(ang) * step);
        if (nav?.ready && nav.snap(_v1, _v2, 4)) a.destination.copy(_v2);
        else a.destination.copy(a.feet);
        break;
      }

      case 'reposition': {
        const spot = this.pickCover(a);
        if (spot) a.destination.copy(spot);
        else if (nav?.ready && nav.randomPointNear(a.feet, 9, _v1)) a.destination.copy(_v1);
        else return false;
        break;
      }

      default:
        return false;
    }

    return a.destination.distanceToSquared(_v0) > 0.36 || !a.path.valid;
  }

  /** Nearest cover that actually breaks the player's line, not just nearest. */
  private pickCover(a: Agent): THREE.Vector3 | null {
    let best: THREE.Vector3 | null = null;
    let bestScore = -Infinity;
    for (let i = 0; i < this.cover.length; i++) {
      const c = this.cover[i];
      const d = c.position.distanceTo(a.feet);
      if (d > 22 || d < 1.2) continue;
      // The normal points toward the exposed side; we want the cover between us
      // and the threat, so reward a normal pointing at the player.
      _v3.copy(this.playerPos).sub(c.position);
      _v3.y = 0;
      const toward = _v3.lengthSq() > 1e-4 ? _v3.normalize().dot(c.normal) : 0;
      const score = toward * 6 - d * 0.35 + this.rng() * 1.4;
      if (score > bestScore) {
        bestScore = score;
        best = c.position;
      }
    }
    return best;
  }

  private repath(a: Agent): void {
    const nav = this.nav;
    if (!nav?.ready) {
      // No navmesh (degenerate level): steer straight at the destination.
      this.pathScratch.length = 0;
      this.pathScratch.push(a.destination.clone());
      a.path.set(this.pathScratch, a.destination);
      return;
    }
    this.pathScratch.length = 0;
    if (nav.findPath(a.feet, a.destination, this.pathScratch) && this.pathScratch.length > 0) {
      a.path.set(this.pathScratch, a.destination);
    } else {
      a.path.clear();
    }
  }

  // =========================================================================
  // Steering + movement
  // =========================================================================

  private steer(a: Agent, dt: number, ctx: GameContext): void {
    const wantSpeed = a.state === 'idle' ? 0
      : a.state === 'patrol' ? WALK_SPEED
        : a.state === 'engage' ? WALK_SPEED * 1.15
          : RUN_SPEED;

    let desiredX = 0;
    let desiredZ = 0;

    if (wantSpeed > 0 && a.path.target(a.feet, _v0, 0.6)) {
      _v0.y = a.feet.y;
      _v1.copy(_v0).sub(a.feet);
      _v1.y = 0;
      const len = _v1.length();
      if (len > 1e-4) {
        _v1.multiplyScalar(1 / len);
        desiredX = _v1.x * wantSpeed;
        desiredZ = _v1.z * wantSpeed;
      }
    }

    // Separation: agents that stack up look like one broken agent.
    for (let i = 0; i < this.agents.length; i++) {
      const o = this.agents[i];
      if (o === a || !o.alive) continue;
      const dx = a.feet.x - o.feet.x;
      const dz = a.feet.z - o.feet.z;
      const d2 = dx * dx + dz * dz;
      if (d2 > SEPARATION_RADIUS * SEPARATION_RADIUS || d2 < 1e-6) continue;
      const d = Math.sqrt(d2);
      const push = (1 - d / SEPARATION_RADIUS) * 2.6;
      desiredX += (dx / d) * push;
      desiredZ += (dz / d) * push;
    }

    a.velocity.x = damp(a.velocity.x, desiredX, ACCEL, dt);
    a.velocity.z = damp(a.velocity.z, desiredZ, ACCEL, dt);
    if (Math.abs(a.velocity.x) < 0.01) a.velocity.x = 0;
    if (Math.abs(a.velocity.z) < 0.01) a.velocity.z = 0;

    const moveX = a.velocity.x * dt;
    const moveZ = a.velocity.z * dt;
    if (moveX !== 0 || moveZ !== 0) {
      _v0.copy(a.feet);
      _v1.set(a.feet.x + moveX, a.feet.y, a.feet.z + moveZ);
      const hit = ctx.physics.sweepCapsule(_v0, _v1, AGENT_RADIUS, AGENT_HEIGHT, RayMask.Solid);
      if (hit) {
        // Slide along the wall with the remaining motion instead of stopping
        // dead — a soldier that pins itself on a doorframe is the classic tell.
        _v2.set(moveX, 0, moveZ).multiplyScalar(Math.max(0, hit.t - 0.01));
        _v3.copy(hit.normal);
        _v3.y = 0;
        if (_v3.lengthSq() > 1e-6) {
          _v3.normalize();
          const remainX = moveX - _v2.x;
          const remainZ = moveZ - _v2.z;
          const dot = remainX * _v3.x + remainZ * _v3.z;
          _v2.x += remainX - _v3.x * dot;
          _v2.z += remainZ - _v3.z * dot;
        }
        _v1.set(a.feet.x + _v2.x, a.feet.y, a.feet.z + _v2.z);
        if (!ctx.physics.overlapCapsule(_v1, AGENT_RADIUS, AGENT_HEIGHT, RayMask.Solid)) {
          a.feet.x = _v1.x;
          a.feet.z = _v1.z;
        }
        a.velocity.x *= 0.6;
        a.velocity.z *= 0.6;
      } else {
        a.feet.x = _v1.x;
        a.feet.z = _v1.z;
      }
    }

    // Ground: navmesh height where it exists (free, smooth), physics elsewhere.
    let ground = this.navReady ? this.nav?.groundAt(a.feet.x, a.feet.z) ?? null : null;
    if (ground === null || !Number.isFinite(ground) || Math.abs(ground - a.feet.y) > 1.6) {
      ground = ctx.physics.groundHeight(_v0.set(a.feet.x, a.feet.y + 1.2, a.feet.z), 6);
    }
    if (ground !== null && Number.isFinite(ground)) {
      // Feet track the floor with a spring, so a kerb is stepped rather than
      // teleported over.
      a.feet.y = damp(a.feet.y, ground, 14, dt);
    }

    a.position.set(a.feet.x, a.feet.y + CHEST_HEIGHT * a.scale - a.crouch * 0.32, a.feet.z);

    // Facing: body turns toward motion, or toward the threat while engaged.
    const speed = Math.hypot(a.velocity.x, a.velocity.z);
    a.speedSmooth = damp(a.speedSmooth, speed, 9, dt);
    if (a.state === 'engage' || a.awareness >= 1) {
      _v0.copy(a.hasLos ? this.playerPos : a.lastKnown).sub(a.position);
      a.targetYaw = Math.atan2(-_v0.x, -_v0.z);
    } else if (speed > 0.25) {
      a.targetYaw = Math.atan2(-a.velocity.x, -a.velocity.z);
    }
    a.yaw += angleDelta(a.yaw, a.targetYaw) * (1 - Math.exp(-TURN_RATE * dt));

    // Aim tracks faster than the body turns; the difference is the upper-body
    // twist, which is what makes a soldier read as tracking you while walking.
    _v0.copy(a.hasLos || a.awareness > 0.4 ? (a.hasLos ? this.playerPos : a.lastKnown) : a.feet).sub(
      _v1.set(a.feet.x, a.feet.y + MUZZLE_HEIGHT * a.scale, a.feet.z),
    );
    if (_v0.lengthSq() > 0.04 && (a.hasLos || a.awareness > 0.4)) {
      const wantAimYaw = Math.atan2(-_v0.x, -_v0.z);
      const flat = Math.hypot(_v0.x, _v0.z);
      const wantPitch = Math.atan2(_v0.y, Math.max(0.05, flat));
      a.aimYaw += angleDelta(a.aimYaw, wantAimYaw) * (1 - Math.exp(-11 * dt));
      a.aimPitch = damp(a.aimPitch, THREE.MathUtils.clamp(wantPitch, -0.85, 0.85), 11, dt);
    } else {
      a.aimYaw += angleDelta(a.aimYaw, a.yaw) * (1 - Math.exp(-4 * dt));
      a.aimPitch = damp(a.aimPitch, 0, 4, dt);
    }
    a.upperTwist = THREE.MathUtils.clamp(angleDelta(a.yaw, a.aimYaw), -0.95, 0.95);

    a.crouch = damp(a.crouch, a.crouchTarget, 6.5, dt);
    a.flinch = Math.max(0, a.flinch - dt * 3.4);
  }

  // =========================================================================
  // Combat
  // =========================================================================

  private combat(a: Agent, dt: number, ctx: GameContext): void {
    a.fireCooldown -= dt;
    a.burstCooldown -= dt;
    if (a.reactionLeft > 0) a.reactionLeft -= dt;

    if (a.reloadLeft > 0) {
      a.reloadLeft -= dt;
      if (a.reloadLeft <= 0) a.ammo = a.spec.magSize;
      return;
    }

    if (a.state !== 'engage' || !a.hasLos || !this.playerAlive) return;
    if (a.reactionLeft > 0) return;

    if (a.ammo <= 0) {
      a.reloadLeft = a.spec.reloadEmptyTime;
      a.burstLeft = 0;
      return;
    }

    if (a.burstLeft <= 0) {
      if (a.burstCooldown > 0) return;
      const auto = a.spec.fireMode === 'auto';
      a.burstLeft = auto ? 3 + ((this.rng() * 4) | 0) : (a.spec.burstCount ?? 1);
      // Longer pauses at distance: a squad that never stops shooting is noise.
      const dist = a.position.distanceTo(this.playerPos);
      a.burstCooldown = 0.35 + this.rng() * 0.55 + dist * 0.012;
    }

    if (a.fireCooldown > 0) return;

    this.fire(a, ctx);
    a.burstLeft--;
    a.fireCooldown = (60 / Math.max(1, a.spec.rpm)) * (0.96 + this.rng() * 0.1);
  }

  private fire(a: Agent, ctx: GameContext): void {
    a.ammo--;

    a.rig.muzzle.updateWorldMatrix(true, false);
    _muz.setFromMatrixPosition(a.rig.muzzle.matrixWorld);
    if (!Number.isFinite(_muz.x)) {
      _muz.set(a.feet.x, a.feet.y + MUZZLE_HEIGHT * a.scale, a.feet.z);
    }

    _dir.copy(this.playerPos).sub(_muz);
    // Lead the player slightly at range so a moving target is still threatened.
    const dist = _dir.length();
    if (dist < 1e-3) return;
    _dir.multiplyScalar(1 / dist);

    // Aim converges the longer contact is held, and opens back up when the
    // soldier is hurt or moving.
    const settle = THREE.MathUtils.clamp(a.losTime / AIM_SETTLE_TIME, 0, 1);
    let err = AIM_ERROR_INITIAL + (AIM_ERROR_SETTLED - AIM_ERROR_INITIAL) * settle;
    err *= 1 + a.speedSmooth * 0.16 + (1 - a.health / MAX_HEALTH) * 0.5;
    // Deliberately generous at very close range so a rush is survivable.
    if (dist < 6) err *= 1.5;

    // Random direction in a cone, uniform on the cap.
    const theta = Math.sqrt(this.rng()) * err;
    const phi = this.rng() * Math.PI * 2;
    _v0.set(0, 1, 0);
    if (Math.abs(_dir.y) > 0.95) _v0.set(1, 0, 0);
    _v1.crossVectors(_dir, _v0).normalize();
    _v2.crossVectors(_dir, _v1).normalize();
    _dir.multiplyScalar(Math.cos(theta))
      .addScaledVector(_v1, Math.sin(theta) * Math.cos(phi))
      .addScaledVector(_v2, Math.sin(theta) * Math.sin(phi))
      .normalize();

    ctx.events.emit('shot.fired', {
      weaponId: a.weaponId,
      origin: _muz,
      direction: _dir,
      local: false,
      shooterId: a.entityId,
    });

    this.spawnFlash(_muz, _dir, a.spec);

    // Recoil the rig: the shoulder absorbs it and the muzzle climbs, both
    // recovering over the next few hundred milliseconds.
    a.flinch = Math.min(0.55, a.flinch + 0.14);
    a.aimPitch += a.spec.recoilVertical * 0.5;
  }

  // =========================================================================
  // Muzzle flashes
  // =========================================================================

  private buildFlashPool(ctx: GameContext): void {
    this.flashTex = flashTexture();
    this.flashMat = new THREE.SpriteMaterial({
      map: this.flashTex,
      color: 0xffffff,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      transparent: true,
      fog: false,
      toneMapped: false,
    });
    this.flashPool = ctx.quality.preset === 'low' ? 5 : 10;
    for (let i = 0; i < this.flashPool; i++) {
      // Each flash owns its material: rotation and opacity are per-instance and
      // a shared SpriteMaterial would make every flash flicker in lockstep.
      const s = new THREE.Sprite(this.flashMat.clone());
      s.visible = false;
      s.frustumCulled = false;
      s.renderOrder = 6;
      ctx.scene.add(s);
      this.flashes.push({ sprite: s, life: 0, duration: 0.055, size: 0.4, roll: 0 });
    }
  }

  private spawnFlash(at: THREE.Vector3, dir: THREE.Vector3, spec: WeaponSpec): void {
    let slot: Flash | null = null;
    for (const f of this.flashes) {
      if (f.life <= 0) { slot = f; break; }
    }
    if (!slot) slot = this.flashes[(this.rng() * this.flashes.length) | 0];
    if (!slot) return;

    // Flash sits a little past the crown so it never intersects the barrel.
    slot.sprite.position.copy(at).addScaledVector(dir, 0.06);
    slot.sprite.visible = true;
    slot.size = (spec.category === 'smg' ? 0.30 : spec.category === 'sniper' ? 0.55 : 0.40)
      * (0.85 + this.rng() * 0.4);
    slot.duration = 0.045 + this.rng() * 0.022;
    slot.life = slot.duration;
    slot.roll = this.rng() * Math.PI * 2;
    slot.sprite.material.rotation = slot.roll;
    slot.sprite.scale.setScalar(slot.size);
  }

  private updateFlashes(dt: number): void {
    for (const f of this.flashes) {
      if (f.life <= 0) continue;
      f.life -= dt;
      if (f.life <= 0) {
        f.sprite.visible = false;
        continue;
      }
      const t = f.life / f.duration;
      // Bloom out then collapse: peak on the first frame, gone by the third.
      f.sprite.scale.setScalar(f.size * (0.55 + 0.75 * t) * (1 + (1 - t) * 0.6));
      const mat = f.sprite.material;
      mat.opacity = t * t;
    }
  }

  // =========================================================================
  // Death
  // =========================================================================

  private updateDead(a: Agent, dt: number): void {
    a.deathTime += dt;
    a.respawnLeft -= dt;

    // Settle onto the floor over ~0.9s with a bit of overshoot, then stop.
    const t = THREE.MathUtils.clamp(a.deathTime / 0.9, 0, 1);
    const k = t < 1 ? easeOutBack(t) : 1;

    const ground = this.navReady ? this.nav?.groundAt(a.feet.x, a.feet.z) ?? null : null;
    if (ground !== null && Number.isFinite(ground)) a.feet.y = damp(a.feet.y, ground, 9, dt);

    // Slide a short distance in the direction of travel, decelerating hard.
    a.feet.x += a.velocity.x * dt;
    a.feet.z += a.velocity.z * dt;
    a.velocity.multiplyScalar(Math.max(0, 1 - dt * 5.5));

    a.yaw += a.deathSpin * dt * (1 - k) * 0.6;
    a.position.set(a.feet.x, a.feet.y + 0.28, a.feet.z);

    if (a.respawnLeft <= 0) this.placeAtSpawn(a, false);
  }

  // =========================================================================
  // Pose + instancing
  // =========================================================================

  private poseAll(dt: number, elapsed: number): void {
    const r = this.renderer;
    if (!r) return;

    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      this.poseAgent(a, dt, elapsed);

      const rig = a.rig;

      // Sink the corpse out of sight over the last 1.5s of its lifetime rather
      // than popping it. Applied before the matrix flush or it lands a frame
      // late and is immediately overwritten by the next pose.
      if (a.state === 'dead' && a.respawnLeft < 1.5) {
        rig.root.position.y -= (1.5 - Math.max(0, a.respawnLeft)) * 0.62;
      }
      rig.root.updateMatrixWorld(true);

      r.write(PART.pelvis, i, 0, rig.pelvis.matrixWorld);
      r.write(PART.torso, i, 0, rig.spine.matrixWorld);
      r.write(PART.head, i, 0, rig.head.matrixWorld);
      r.write(PART.helmet, i, 0, rig.helmet.matrixWorld);
      r.write(PART.pack, i, 0, rig.pack.matrixWorld);
      r.write(PART.rifle, i, 0, rig.weapon.matrixWorld);
      for (let s = 0; s < 2; s++) {
        r.write(PART.upperArm, i, s, rig.shoulder[s].matrixWorld);
        r.write(PART.foreArm, i, s, rig.elbow[s].matrixWorld);
        r.write(PART.thigh, i, s, rig.hip[s].matrixWorld);
        r.write(PART.shin, i, s, rig.knee[s].matrixWorld);
      }
    }
    r.flush();
  }

  private poseAgent(a: Agent, dt: number, elapsed: number): void {
    const rig = a.rig;
    const dead = a.state === 'dead';

    rig.root.position.copy(a.feet);
    rig.root.rotation.set(0, a.yaw, 0);

    if (dead) {
      this.poseDeath(a);
      return;
    }

    // --- Locomotion -------------------------------------------------------
    const gait = THREE.MathUtils.clamp(a.speedSmooth / RUN_SPEED, 0, 1);
    // Stride frequency rises with speed but not linearly: real gait cadence
    // saturates around 3 Hz and the rest of the speed comes from stride length.
    const cadence = 1.35 + gait * 1.55;
    a.stridePhase += dt * cadence * Math.PI * 2 * (a.speedSmooth > 0.08 ? 1 : 0);
    if (a.stridePhase > Math.PI * 4) a.stridePhase -= Math.PI * 4;
    a.breath += dt * (1.1 + gait * 1.4);

    const phase = a.stridePhase;
    const swing = 0.16 + gait * 0.62;
    const idleBreath = Math.sin(a.breath) * 0.018;

    // Hips: bob at twice stride frequency, roll into the stance leg, drop when
    // crouching. The 0.5 phase offset puts the low point at foot strike.
    const bob = Math.cos(phase * 2) * (0.012 + gait * 0.042);
    rig.hips.position.set(
      Math.sin(phase) * (0.006 + gait * 0.022),
      HIP_HEIGHT - a.crouch * 0.34 + bob + idleBreath,
      0,
    );
    rig.hips.rotation.set(
      a.crouch * 0.16 + gait * 0.06,
      Math.sin(phase) * (0.03 + gait * 0.16),
      Math.sin(phase) * (0.02 + gait * 0.075),
    );

    // Pelvis counter-rotates against the shoulders — the counter-rotation is
    // most of what sells a walk.
    rig.pelvis.rotation.set(0, -Math.sin(phase) * (0.02 + gait * 0.1), 0);

    // Spine: lean into the run, twist toward the aim, absorb flinch.
    const flinchAmt = a.flinch * a.flinch;
    rig.spine.rotation.set(
      gait * 0.20 + a.crouch * 0.22 + flinchAmt * 0.16 + idleBreath * 0.4,
      a.upperTwist * 0.62 + Math.sin(phase) * (0.02 + gait * 0.09),
      flinchAmt * 0.20 * a.flinchAxis - Math.sin(phase) * gait * 0.04,
    );

    // Head stabilises: counters the spine so the eyes stay level, then adds the
    // remaining twist toward the target.
    const spinePitch = gait * 0.20 + a.crouch * 0.22;
    rig.neck.rotation.set(
      THREE.MathUtils.clamp(-a.aimPitch * 0.35 - spinePitch * 0.75, -0.5, 0.5),
      THREE.MathUtils.clamp(a.upperTwist * 0.38, -0.6, 0.6),
      -Math.sin(phase) * gait * 0.05,
    );
    rig.head.rotation.set(0, 0, 0);

    // --- Legs -------------------------------------------------------------
    for (let s = 0; s < 2; s++) {
      const p = phase + (s === 0 ? 0 : Math.PI);
      const sw = Math.sin(p);
      const lift = Math.max(0, -Math.cos(p));
      rig.hip[s].rotation.set(
        sw * swing - a.crouch * 0.85 + gait * 0.06,
        0,
        (s === 0 ? -1 : 1) * (0.035 + a.crouch * 0.14),
      );
      // Knee only bends one way; the extra bend during swing-through is what
      // stops the classic "skating" look.
      rig.knee[s].rotation.set(
        Math.max(0, -sw * 0.5) + lift * (0.25 + gait * 0.95) + a.crouch * 1.5 + 0.06,
        0, 0,
      );
    }

    // --- Arms + weapon ----------------------------------------------------
    const ready = a.state === 'engage' || a.awareness > 0.55 ? 1 : 0;
    const aimBlend = damp((rig.weapon.userData.aim as number | undefined) ?? 0, ready, 6, dt);
    rig.weapon.userData.aim = aimBlend;

    // Low ready -> shouldered. The weapon rides in front of the chest and rises
    // to eye level; the arms are posed to match rather than IK'd, which at 6m+
    // is indistinguishable and costs nothing.
    const wy = 0.30 + aimBlend * 0.19;
    const wz = 0.16 + aimBlend * 0.02;
    rig.weapon.position.set(0.055 + aimBlend * 0.03, wy, -wz);
    rig.weapon.rotation.set(
      -0.55 + aimBlend * 0.55 + a.aimPitch * (0.25 + aimBlend * 0.75) - flinchAmt * 0.22,
      -0.28 + aimBlend * 0.20,
      0.10 - aimBlend * 0.09,
    );

    // Right hand (index 1) on the grip, left hand (0) forward on the handguard.
    rig.shoulder[1].rotation.set(
      -0.62 - aimBlend * 0.42 + a.aimPitch * 0.28,
      -0.30 - aimBlend * 0.10,
      0.42 - aimBlend * 0.14 - Math.sin(phase) * gait * 0.06,
    );
    rig.elbow[1].rotation.set(-1.32 + aimBlend * 0.22, 0.16, 0.20);

    rig.shoulder[0].rotation.set(
      -0.95 - aimBlend * 0.35 + a.aimPitch * 0.30,
      0.44 + aimBlend * 0.14,
      -0.50 + aimBlend * 0.12 + Math.sin(phase) * gait * 0.06,
    );
    rig.elbow[0].rotation.set(-1.05 - aimBlend * 0.30, -0.22, -0.16);

    rig.pack.rotation.set(0.06 + gait * 0.05, 0, Math.sin(phase) * gait * 0.03);
    rig.pack.position.set(0, 0.28 - gait * 0.01, -0.185);

    // Muzzle marker follows the weapon; recompute each frame so `fire()` can
    // read a world position without a second traversal.
    rig.muzzle.position.set(0, 0.05, -0.74);
    void elapsed;
  }

  private poseDeath(a: Agent): void {
    const rig = a.rig;
    const t = THREE.MathUtils.clamp(a.deathTime / 0.9, 0, 1);
    const k = t < 1 ? easeOutBack(t) : 1;
    const slack = THREE.MathUtils.clamp(a.deathTime / 0.55, 0, 1);

    // The whole body rotates about the ankles into the ground plane, then the
    // joints go slack. Two-stage is what separates a death from a T-pose flop.
    rig.root.rotation.set(a.deathPitch * k * 1.42, a.yaw, a.deathRoll * k * 0.7);
    rig.hips.position.set(0, HIP_HEIGHT - k * 0.60, k * (a.deathPitch > 0 ? -0.12 : 0.10));
    rig.hips.rotation.set(-a.deathPitch * k * 0.30, 0, a.deathRoll * k * 0.25);
    rig.pelvis.rotation.set(0, 0, 0);
    rig.spine.rotation.set(-a.deathPitch * k * 0.34, a.deathRoll * k * 0.4, -a.deathRoll * k * 0.3);
    rig.neck.rotation.set(0.42 * slack * a.deathPitch, a.deathRoll * 0.5 * slack, 0.3 * slack);
    rig.head.rotation.set(0, 0, 0);

    for (let s = 0; s < 2; s++) {
      const sign = s === 0 ? -1 : 1;
      rig.hip[s].rotation.set(
        -0.30 * slack + a.deathPitch * 0.42 * slack,
        sign * 0.16 * slack,
        sign * (0.22 + a.deathRoll * 0.2) * slack,
      );
      rig.knee[s].rotation.set(0.65 * slack + (s === 0 ? 0.35 : 0.1) * slack, 0, 0);
      rig.shoulder[s].rotation.set(
        -0.35 * slack,
        sign * 0.3 * slack,
        sign * (0.85 + 0.25 * (s === 0 ? 1 : -1)) * slack,
      );
      rig.elbow[s].rotation.set(-0.55 * slack, 0, 0);
    }

    // The rifle is dropped: it separates from the hands and lands flat.
    rig.weapon.position.set(0.24 * slack, 0.30 - 0.28 * slack, -0.16 - 0.30 * slack);
    rig.weapon.rotation.set(-0.55 + 1.9 * slack, -0.28 - 0.9 * slack, 0.10 + 1.3 * slack);
    rig.pack.rotation.set(0, 0, 0);
  }
}
