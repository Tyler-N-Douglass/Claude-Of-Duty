/**
 * PlayerSystem — the local player: capsule, camera, health.
 *
 * Ordering matters and is deliberate:
 *   fixedUpdate  runs the movement simulation at the engine's 120Hz step, so
 *                collision response is frame-rate independent and reproducible.
 *   update       interpolates the visual position between the last two fixed
 *                states, integrates every secondary camera motion, and writes
 *                the camera. Doing it here (not in lateUpdate) means the shadow
 *                cascade system — which reads ctx.camera in its own lateUpdate,
 *                and is registered before this one — sees this frame's aim.
 *   lateUpdate   re-writes the camera without advancing time, so recoil the
 *                weapon system applied during its update lands on the frame the
 *                shot was fired rather than the next one.
 */
import * as THREE from 'three';
import {
  RayMask,
  UNITS,
  type DamageTarget,
  type FrameTime,
  type GameContext,
  type HitboxKind,
  type SurfaceKind,
  type System,
} from '../core/Contracts';
import { CameraRig, RIG, type DebugPose, type RigMotion } from './CameraRig';
import { MOVE, MovementController, type MoveCommand } from './Movement';

/** Must match Engine's FIXED_STEP; used only to reconstruct the render alpha. */
const FIXED_STEP = 1 / 120;

const MAX_HEALTH = 100;
const REGEN_DELAY = 4.5;
const REGEN_RATE = 25;
const RESPAWN_DELAY = 4.0;
/** Below this the player has fallen out of the world and is put back. */
const VOID_Y = -60;

const _v = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _down = new THREE.Vector3(0, -1, 0);

export class PlayerSystem implements System, DamageTarget {
  readonly name = 'player';
  readonly entityId = 0;

  readonly rig = new CameraRig();
  readonly movement = new MovementController();

  health = MAX_HEALTH;
  readonly maxHealth = MAX_HEALTH;

  /** Chest-height world position; this is what the AI aims at. */
  readonly position = new THREE.Vector3();
  /** Interpolated capsule base for the current frame. */
  readonly renderPosition = new THREE.Vector3();

  private ctx: GameContext | null = null;
  private aiming = false;
  private dead = false;
  private sinceHit = 999;
  private respawnTimer = 0;
  private accumulator = 0;
  private alpha = 1;
  /**
   * The fixed step can run twice in a frame (or zero times above 120fps), while
   * InputState's edge sets are per-frame. Edges are therefore polled exactly
   * once per frame — by whichever of fixedUpdate/update reaches the frame first
   * — latched, and consumed by the next simulation step. Without this a single
   * crouch tap reads as a double-tap and drops you straight to prone.
   */
  private inputFrame = -1;
  private pendingJump = false;
  private pendingCrouch = false;
  private pendingSprint = false;
  private readonly spawn = new THREE.Vector3(0, 1, 8);
  private spawnYaw = 0;

  private readonly cmd: MoveCommand = {
    moveX: 0, moveY: 0, yaw: 0,
    jumpPressed: false, jumpHeld: false,
    crouchPressed: false, crouchHeld: false,
    sprintPressed: false, sprintHeld: false,
    walkHeld: false, aiming: false, enabled: true,
  };

  private readonly motion: RigMotion = {
    eyeY: UNITS.eyeOffset, x: 0, z: 0,
    speed: 0, gaitSpeed: MOVE.speedWalk, stridePhase: 0,
    grounded: true, sprinting: false, tacSprinting: false,
    sliding: false, slideT: 0, slideLateral: 0,
    mantleActive: false, mantleT: 0,
    crouched: false, aiming: false,
    moveIntent: 0, strafeIntent: 0,
    leanLeft: false, leanRight: false,
  };

  /** Reused so the per-frame state broadcast allocates nothing. */
  private readonly stateEvent = {
    health: MAX_HEALTH, maxHealth: MAX_HEALTH,
    sprinting: false, crouched: false, grounded: true,
  };

  private readonly unsubs: Array<() => void> = [];
  private debugHookInstalled = false;

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;
    this.rig.init(ctx);

    this.findSpawn(ctx);
    this.movement.teleport(this.spawn.x, this.spawn.y, this.spawn.z);
    this.rig.setAim(this.spawnYaw, 0);
    this.rig.setEyeAnchor(this.spawn.x, this.spawn.z);
    this.renderPosition.copy(this.spawn);
    this.syncMotion(1);
    this.rig.advance(1 / 60, this.motion, ctx);
    this.rig.apply(ctx);

    ctx.entities.register(this);

    this.unsubs.push(
      ctx.events.on('weapon.ads', (p) => {
        this.aiming = p.aiming;
        this.rig.setAdsFovScale(p.aiming ? p.fovScale : 1);
      }),
      // Explosions shove the player around; the damage itself belongs to
      // whoever raised the explosion, so this only applies the impulse.
      ctx.events.on('explosion', (p) => this.onExplosion(p.point, p.radius)),
    );

    this.installDebugHook();
  }

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;
    this.ctx?.entities.unregister(this.entityId);
    this.rig.dispose();
    this.removeDebugHook();
    this.ctx = null;
  }

  /**
   * Drops the spawn onto whatever the level builder put down. Falls back to a
   * fixed height if there is no geometry yet, and nudges out of anything solid.
   */
  private findSpawn(ctx: GameContext): void {
    const candidates: Array<[number, number]> = [[0, 8], [0, 4], [3, 6], [-3, 6], [0, 0], [6, 0], [-6, 0]];
    for (const [x, z] of candidates) {
      _v.set(x, 6, z);
      const g = ctx.physics.groundHeight(_v, 24);
      if (g === null) continue;
      _v.set(x, g + 0.02, z);
      if (ctx.physics.overlapCapsule(_v, MOVE.radius, MOVE.standHeight, RayMask.Solid)) continue;
      this.spawn.copy(_v);
      // Face the middle of the map; the QA hero shot looks down -Z from +Z.
      this.spawnYaw = Math.atan2(x, z) + Math.PI;
      if (Math.abs(x) < 0.01 && Math.abs(z) < 0.01) this.spawnYaw = 0;
      return;
    }
    this.spawn.set(0, 1.0, 8);
    this.spawnYaw = 0;
  }

  // =========================================================================
  // Simulation
  // =========================================================================

  /**
   * Runs once per frame, from whichever hook gets there first. Aim is applied
   * here rather than in update() so that the movement simulation for this frame
   * already turns with the mouse — a frame of yaw lag on the move direction is
   * exactly what makes a controller feel like it is dragging behind the view.
   */
  private pollFrameInput(ctx: GameContext): void {
    if (this.inputFrame === ctx.time.frame) return;
    this.inputFrame = ctx.time.frame;

    const input = ctx.input;
    const live = !this.rig.debugFrozen && !ctx.paused;
    if (live) this.rig.look(input.lookDelta.x, input.lookDelta.y);

    if (live && !this.dead) {
      if (input.wasPressed('jump')) this.pendingJump = true;
      if (input.wasPressed('crouch')) this.pendingCrouch = true;
      if (input.wasPressed('sprint')) this.pendingSprint = true;
    } else {
      this.pendingJump = false;
      this.pendingCrouch = false;
      this.pendingSprint = false;
    }
  }

  fixedUpdate(step: number, ctx: GameContext): void {
    this.pollFrameInput(ctx);
    this.accumulator -= step;

    const frozen = this.rig.debugFrozen;
    const input = ctx.input;
    const c = this.cmd;

    if (frozen || this.dead) {
      c.moveX = 0; c.moveY = 0;
      c.jumpPressed = false; c.jumpHeld = false;
      c.crouchPressed = false; c.crouchHeld = false;
      c.sprintPressed = false; c.sprintHeld = false;
      c.walkHeld = false;
      c.aiming = false;
      c.enabled = false;
    } else {
      c.moveX = input.moveAxis.x;
      c.moveY = input.moveAxis.y;
      c.jumpPressed = this.pendingJump;
      c.crouchPressed = this.pendingCrouch;
      c.sprintPressed = this.pendingSprint;
      this.pendingJump = false;
      this.pendingCrouch = false;
      this.pendingSprint = false;
      c.jumpHeld = input.isDown('jump');
      c.crouchHeld = input.isDown('crouch');
      c.sprintHeld = input.isDown('sprint');
      c.walkHeld = input.isDown('walk');
      c.aiming = this.aiming;
      c.enabled = true;
    }
    c.yaw = this.rig.yaw;

    if (frozen) return;

    const m = this.movement;
    m.step(c, ctx.physics, step);

    if (m.footstepPending) this.emitFootstep(ctx, false);
    if (m.landPending >= 0) this.onLanded(ctx, m.landPending);
    if (m.slideStarted) {
      this.emitFootstep(ctx, true);
      ctx.events.emit('camera.shake', { amount: 0.12, duration: 0.22, frequency: 21 });
    }
    if (m.mantleStarted) {
      ctx.events.emit('camera.shake', { amount: 0.08, duration: 0.25, frequency: 15 });
    }

    if (m.position.y < VOID_Y) this.respawn(ctx);
  }

  update(time: FrameTime, ctx: GameContext): void {
    // Above 120fps a frame can run zero fixed steps; this catches those.
    this.pollFrameInput(ctx);

    // Mirror the engine's fixed-step accumulator so the render position can be
    // interpolated between the last two simulated states instead of popping at
    // whatever sub-step boundary the frame happened to land on.
    this.accumulator += time.dt;
    if (this.accumulator < -FIXED_STEP * 4 || this.accumulator > FIXED_STEP * 4) this.accumulator = 0;
    this.alpha = THREE.MathUtils.clamp(this.accumulator / FIXED_STEP, 0, 1);

    const m = this.movement;
    this.renderPosition.lerpVectors(m.prevPosition, m.position, this.alpha);

    this.syncMotion(time.dt);
    this.rig.setEyeAnchor(this.renderPosition.x, this.renderPosition.z);
    // Paused freezes the secondary motion too, otherwise the springs keep
    // ringing behind the menu and the view has drifted when you come back.
    if (!ctx.paused) this.rig.advance(time.dt, this.motion, ctx);
    this.rig.apply(ctx);

    this.position.set(
      this.renderPosition.x,
      this.renderPosition.y + (m.stance === 'stand' ? 1.25 : m.stance === 'crouch' ? 0.75 : 0.32),
      this.renderPosition.z,
    );

    this.updateHealth(time.dt, ctx);
    this.broadcastState(ctx);
  }

  lateUpdate(_time: FrameTime, ctx: GameContext): void {
    // No time advance: this only folds in impulses raised after our update.
    this.rig.apply(ctx);
  }

  private syncMotion(_dt: number): void {
    const m = this.movement;
    const mo = this.motion;
    mo.eyeY = this.renderPosition.y + m.eyeHeight - m.stepSmooth;
    mo.x = this.renderPosition.x;
    mo.z = this.renderPosition.z;
    mo.speed = m.horizontalSpeed;
    mo.gaitSpeed = m.sliding
      ? MOVE.slideSpeed
      : m.tacSprinting ? MOVE.speedTacSprint
      : m.sprinting ? MOVE.speedSprint
      : m.stance === 'prone' ? MOVE.speedProne
      : m.stance === 'crouch' ? MOVE.speedCrouch
      : MOVE.speedWalk;
    mo.stridePhase = m.stridePhase;
    mo.grounded = m.grounded;
    mo.sprinting = m.sprinting;
    mo.tacSprinting = m.tacSprinting;
    mo.sliding = m.sliding;
    mo.slideT = m.slideT;
    mo.slideLateral = m.slideLateral;
    mo.mantleActive = m.mantleActive;
    mo.mantleT = m.mantleT;
    mo.crouched = m.stance !== 'stand';
    mo.aiming = this.aiming;
    mo.moveIntent = m.moveIntent;
    mo.strafeIntent = m.strafeIntent;

    const input = this.ctx?.input;
    const canLean = !!input && !this.dead && !this.rig.debugFrozen;
    mo.leanLeft = canLean && input.isDown('leanLeft');
    mo.leanRight = canLean && input.isDown('leanRight');
  }

  // =========================================================================
  // Events out
  // =========================================================================

  private surfaceUnderFoot(ctx: GameContext): SurfaceKind {
    _v.set(this.movement.position.x, this.movement.position.y + 0.25, this.movement.position.z);
    const hit = ctx.physics.raycast(_v, _down, 0.9, RayMask.Solid);
    return hit ? hit.surface : this.movement.groundSurface;
  }

  private emitFootstep(ctx: GameContext, forceRunning: boolean): void {
    const m = this.movement;
    const running = forceRunning || m.sprinting || m.horizontalSpeed > MOVE.speedWalk + 0.6;
    ctx.events.emit('player.footstep', {
      position: m.position.clone(),
      surface: this.surfaceUnderFoot(ctx),
      running,
    });
  }

  private onLanded(ctx: GameContext, impactSpeed: number): void {
    const surface = this.surfaceUnderFoot(ctx);
    // 0..1 for consumers; 12 m/s is a hard landing, 4.9 is a flat jump.
    const impact = THREE.MathUtils.clamp(impactSpeed / 12, 0, 1);
    ctx.events.emit('player.land', {
      position: this.movement.position.clone(),
      impact,
      surface,
    });
    this.rig.applyLanding(impactSpeed);

    const dmg = MovementController.fallDamage(impactSpeed);
    if (dmg > 0) {
      _dir.set(0, 1, 0);
      this.applyDamage(dmg, 'leg', _dir, this.entityId);
    }
  }

  private onExplosion(point: THREE.Vector3, radius: number): void {
    const m = this.movement;
    _dir.set(m.position.x - point.x, m.position.y + 0.9 - point.y, m.position.z - point.z);
    const d = _dir.length();
    if (d > radius || radius <= 0) return;
    const falloff = 1 - d / radius;
    if (d < 1e-3) _dir.set(0, 1, 0);
    else _dir.multiplyScalar(1 / d);
    const impulse = 9 * falloff * falloff;
    m.velocity.addScaledVector(_dir, impulse);
    m.velocity.y = Math.max(m.velocity.y, impulse * 0.45);
    m.grounded = false;
    this.rig.addTrauma(0.55 * falloff + 0.2, 0.7, 15);
  }

  private broadcastState(ctx: GameContext): void {
    const s = this.stateEvent;
    const m = this.movement;
    s.health = this.health;
    s.sprinting = m.sprinting;
    s.crouched = m.stance !== 'stand';
    s.grounded = m.grounded;
    ctx.events.emit('player.state', s);
  }

  // =========================================================================
  // Health
  // =========================================================================

  get alive(): boolean {
    return !this.dead;
  }

  applyDamage(amount: number, _hitbox: HitboxKind, from: THREE.Vector3, attackerId: number): void {
    if (this.dead || amount <= 0) return;
    const ctx = this.ctx;
    this.health = Math.max(0, this.health - amount);
    this.sinceHit = 0;

    _dir.copy(from).sub(this.position);
    const l = _dir.length();
    if (l > 1e-4) _dir.multiplyScalar(1 / l);
    else _dir.set(0, 0, -1);
    this.rig.applyDamageKick(_dir, amount);

    if (ctx) {
      ctx.events.emit('player.damaged', {
        amount,
        fromDirection: _dir.clone(),
        health: this.health,
      });
    }

    if (this.health <= 0) this.die(attackerId);
  }

  private die(killerId: number): void {
    this.dead = true;
    this.respawnTimer = RESPAWN_DELAY;
    this.movement.velocity.set(0, 0, 0);
    this.rig.addTrauma(0.8, 0.9, 11);
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.events.emit('entity.killed', {
      entityId: this.entityId, killerId, weaponId: '', headshot: false,
    });
    ctx.events.emit('game.over', { won: false });
  }

  private updateHealth(dt: number, ctx: GameContext): void {
    if (this.dead) {
      this.respawnTimer -= dt;
      if (this.respawnTimer <= 0) this.respawn(ctx);
      return;
    }
    this.sinceHit += dt;
    if (this.health < this.maxHealth && this.sinceHit >= REGEN_DELAY) {
      this.health = Math.min(this.maxHealth, this.health + REGEN_RATE * dt);
    }
  }

  respawn(ctx: GameContext): void {
    this.findSpawn(ctx);
    this.health = this.maxHealth;
    this.dead = false;
    this.sinceHit = REGEN_DELAY;
    this.movement.teleport(this.spawn.x, this.spawn.y, this.spawn.z);
    this.renderPosition.copy(this.spawn);
    this.rig.setAim(this.spawnYaw, 0);
    this.rig.setEyeAnchor(this.spawn.x, this.spawn.z);
    this.syncMotion(1 / 60);
    ctx.events.emit('ui.notify', { text: 'Respawned', kind: 'info' });
  }

  // =========================================================================
  // Public surface for other systems
  // =========================================================================

  /** Called by the weapon system on every shot. */
  applyRecoil(pitch: number, yaw: number, recovery?: number): void {
    this.rig.applyRecoil(pitch, yaw, recovery);
  }

  get velocity(): THREE.Vector3 {
    return this.movement.velocity;
  }

  get grounded(): boolean {
    return this.movement.grounded;
  }

  get sprinting(): boolean {
    return this.movement.sprinting;
  }

  get speed(): number {
    return this.movement.horizontalSpeed;
  }

  get stance(): string {
    return this.movement.stance;
  }

  get gait(): string {
    return this.movement.gait;
  }

  /** 0..1 progress through the stride cycle; weapon sway can lock onto this. */
  get stridePhase(): number {
    return this.movement.stridePhase;
  }

  get yaw(): number {
    return this.rig.yaw;
  }

  get pitch(): number {
    return this.rig.pitch;
  }

  getForward(out: THREE.Vector3): THREE.Vector3 {
    return this.rig.getForward(out);
  }

  getEyePosition(out: THREE.Vector3): THREE.Vector3 {
    return out.set(this.renderPosition.x, this.motion.eyeY, this.renderPosition.z);
  }

  teleport(x: number, y: number, z: number, yaw?: number): void {
    this.movement.teleport(x, y, z);
    this.renderPosition.set(x, y, z);
    if (yaw !== undefined) this.rig.setAim(yaw, this.rig.pitch);
    this.rig.setEyeAnchor(x, z);
  }

  // =========================================================================
  // Debug hook for the visual-QA harness
  // =========================================================================

  setDebugPose(pose: DebugPose): void {
    this.rig.setDebugPose(pose);
    const p = this.rig.currentEyeY;
    // Park the capsule under the frozen camera so anything tracking the player
    // (AI, audio listener, occlusion probes) stays consistent with the view.
    const pos = Array.isArray(pose.pos)
      ? { x: pose.pos[0] ?? 0, y: pose.pos[1] ?? p, z: pose.pos[2] ?? 0 }
      : pose.pos ?? { x: 0, y: p, z: 0 };
    this.movement.teleport(pos.x, pos.y - UNITS.eyeOffset, pos.z);
    this.renderPosition.copy(this.movement.position);
    this.position.set(pos.x, pos.y - 0.4, pos.z);
    if (this.ctx) {
      this.syncMotion(1 / 60);
      this.rig.apply(this.ctx);
    }
  }

  clearDebugPose(): void {
    this.rig.clearDebugPose();
  }

  /**
   * main.ts assigns `window.GAME` after every system has initialised, replacing
   * whatever was there. Rather than fight it, install an accessor that merges
   * our hooks into whatever object gets assigned — the harness then finds
   * setDebugPose regardless of who wrote GAME last.
   */
  private installDebugHook(): void {
    const target = window as unknown as Record<string, unknown>;
    const attach = (obj: unknown): unknown => {
      if (obj && typeof obj === 'object') {
        const o = obj as Record<string, unknown>;
        o.setDebugPose = (pose: DebugPose) => this.setDebugPose(pose);
        o.clearDebugPose = () => this.clearDebugPose();
        o.player = this;
      }
      return obj;
    };

    let store: unknown = target.GAME;
    try {
      Object.defineProperty(window, 'GAME', {
        configurable: true,
        enumerable: true,
        get: () => store,
        set: (v: unknown) => {
          store = attach(v);
        },
      });
      this.debugHookInstalled = true;
    } catch {
      // Property is locked down; fall back to attaching to whatever exists.
    }
    if (store !== undefined) attach(store);
  }

  private removeDebugHook(): void {
    if (!this.debugHookInstalled) return;
    this.debugHookInstalled = false;
    const value = (window as unknown as Record<string, unknown>).GAME;
    try {
      Object.defineProperty(window, 'GAME', {
        configurable: true, enumerable: true, writable: true, value,
      });
    } catch {
      /* nothing more we can do */
    }
  }
}

export { CameraRig, RIG };
