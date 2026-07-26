/**
 * Shared contracts for Claude of Duty.
 *
 * Every subsystem is written against these types and nothing else. Modules must
 * not import from each other's internals — only from this file and from three.
 * This is what lets the subsystems be built independently and still compose.
 */
import type * as THREE from 'three';

// ---------------------------------------------------------------------------
// Core loop
// ---------------------------------------------------------------------------

/** Frame timing, produced once per frame by the engine. */
export interface FrameTime {
  /** Seconds since last frame, clamped to a sane max (0.1s). */
  dt: number;
  /** Unclamped seconds since last frame. */
  rawDt: number;
  /** Seconds since engine start. */
  elapsed: number;
  /** Monotonically increasing frame index. */
  frame: number;
  /** Smoothed frames-per-second estimate. */
  fps: number;
}

/**
 * A subsystem with a lifecycle. Systems are updated in registration order.
 * `fixedUpdate` runs at a fixed 120Hz step for physics-sensitive work and may
 * be called zero or more times per frame.
 */
export interface System {
  readonly name: string;
  init?(ctx: GameContext): void | Promise<void>;
  fixedUpdate?(step: number, ctx: GameContext): void;
  update?(time: FrameTime, ctx: GameContext): void;
  /** Runs after all updates, before render. Use for camera-dependent work. */
  lateUpdate?(time: FrameTime, ctx: GameContext): void;
  resize?(width: number, height: number): void;
  dispose?(): void;
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

export type ActionName =
  | 'forward' | 'back' | 'left' | 'right'
  | 'jump' | 'crouch' | 'sprint' | 'walk'
  | 'fire' | 'aim' | 'reload' | 'melee' | 'grenade'
  | 'swapWeapon' | 'interact' | 'leanLeft' | 'leanRight'
  | 'pause' | 'scoreboard' | 'flashlight';

export interface InputState {
  /** True while the action is held. */
  isDown(action: ActionName): boolean;
  /** True only on the frame the action went down. */
  wasPressed(action: ActionName): boolean;
  /** True only on the frame the action went up. */
  wasReleased(action: ActionName): boolean;
  /** Accumulated mouse delta for this frame, in radians (already sensitivity-scaled). */
  readonly lookDelta: { x: number; y: number };
  /** Raw wheel delta for this frame. */
  readonly wheelDelta: number;
  /** Normalized movement intent, magnitude clamped to 1. */
  readonly moveAxis: { x: number; y: number };
  readonly pointerLocked: boolean;
  requestPointerLock(): void;
  exitPointerLock(): void;
  setSensitivity(v: number): void;
  setAdsSensitivityScale(v: number): void;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

export interface DamageEvent {
  targetId: number;
  attackerId: number;
  amount: number;
  /** Where the hit landed, world space. */
  point: THREE.Vector3;
  normal: THREE.Vector3;
  /** Direction the bullet was travelling. */
  direction: THREE.Vector3;
  hitbox: HitboxKind;
  /** True if the damage killed the target. */
  lethal: boolean;
  weaponId: string;
  /** Distance from muzzle in metres. */
  distance: number;
}

export type SurfaceKind =
  | 'concrete' | 'metal' | 'wood' | 'dirt' | 'sand' | 'glass'
  | 'water' | 'flesh' | 'foliage' | 'plaster' | 'rubber' | 'fabric';

export type HitboxKind = 'head' | 'chest' | 'stomach' | 'arm' | 'leg';

export interface ImpactEvent {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  /** Incoming bullet direction (normalized). */
  direction: THREE.Vector3;
  surface: SurfaceKind;
  /** 0..1 energy remaining, drives spark/dust intensity. */
  energy: number;
}

export interface ExplosionEvent {
  point: THREE.Vector3;
  radius: number;
  damage: number;
  attackerId: number;
}

export interface ShotEvent {
  weaponId: string;
  /** World-space muzzle position. */
  origin: THREE.Vector3;
  direction: THREE.Vector3;
  /** True when fired by the local player. */
  local: boolean;
  shooterId: number;
}

export interface GameEvents {
  'shot.fired': ShotEvent;
  'shot.impact': ImpactEvent;
  'shot.tracer': { from: THREE.Vector3; to: THREE.Vector3; speed: number; weaponId: string };
  'damage.dealt': DamageEvent;
  'entity.killed': { entityId: number; killerId: number; weaponId: string; headshot: boolean };
  'explosion': ExplosionEvent;
  'weapon.equipped': { weaponId: string; ammo: number; reserve: number };
  'weapon.ammo': { ammo: number; reserve: number };
  'weapon.reload.start': { weaponId: string; duration: number };
  'weapon.reload.end': { weaponId: string };
  'weapon.ads': { aiming: boolean; fovScale: number };
  'player.state': { health: number; maxHealth: number; sprinting: boolean; crouched: boolean; grounded: boolean };
  'player.footstep': { position: THREE.Vector3; surface: SurfaceKind; running: boolean };
  'player.land': { position: THREE.Vector3; impact: number; surface: SurfaceKind };
  'player.damaged': { amount: number; fromDirection: THREE.Vector3; health: number };
  'camera.shake': { amount: number; duration: number; frequency?: number };
  'ui.hitmarker': { headshot: boolean; lethal: boolean };
  'ui.notify': { text: string; kind?: 'info' | 'kill' | 'objective' };
  'game.pause': { paused: boolean };
  'game.over': { won: boolean };
  'quality.changed': { preset: QualityPreset };
}

export interface EventBus {
  on<K extends keyof GameEvents>(type: K, fn: (payload: GameEvents[K]) => void): () => void;
  once<K extends keyof GameEvents>(type: K, fn: (payload: GameEvents[K]) => void): () => void;
  off<K extends keyof GameEvents>(type: K, fn: (payload: GameEvents[K]) => void): void;
  emit<K extends keyof GameEvents>(type: K, payload: GameEvents[K]): void;
}

// ---------------------------------------------------------------------------
// Collision / world queries
// ---------------------------------------------------------------------------

export interface RayHit {
  point: THREE.Vector3;
  normal: THREE.Vector3;
  distance: number;
  surface: SurfaceKind;
  /** Entity id if the ray hit a character, otherwise -1. */
  entityId: number;
  hitbox?: HitboxKind;
  object: THREE.Object3D | null;
  /** Material thickness in metres for penetration maths; Infinity for solid world. */
  thickness: number;
}

export interface CapsuleSweepHit {
  /** Fraction of the sweep completed before contact, 0..1. */
  t: number;
  normal: THREE.Vector3;
  point: THREE.Vector3;
  surface: SurfaceKind;
}

/**
 * The single source of truth for "what is solid". Implemented by the physics
 * system; consumed by the player controller, AI, ballistics and VFX.
 */
export interface PhysicsWorld {
  /** Closest hit along the ray, or null. `mask` filters entity classes. */
  raycast(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number, mask?: RayMask): RayHit | null;
  /** All hits along the ray, sorted near-to-far. Used for wall penetration. */
  raycastAll(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number, mask?: RayMask): RayHit[];
  /** Sweeps a vertical capsule; returns first contact. */
  sweepCapsule(
    from: THREE.Vector3, to: THREE.Vector3, radius: number, height: number, mask?: RayMask,
  ): CapsuleSweepHit | null;
  /** True if a capsule at this position overlaps world geometry. */
  overlapCapsule(at: THREE.Vector3, radius: number, height: number, mask?: RayMask): boolean;
  /** Registers static level collision. Called by the world builder. */
  addStatic(object: THREE.Object3D, surface: SurfaceKind): void;
  removeStatic(object: THREE.Object3D): void;
  /** Registers a character's hitbox rig for bullet queries. */
  addCharacter(entityId: number, root: THREE.Object3D, boxes: CharacterHitbox[]): void;
  removeCharacter(entityId: number): void;
  /** True if there is nothing solid between two points. */
  hasLineOfSight(from: THREE.Vector3, to: THREE.Vector3, mask?: RayMask): boolean;
  /** Height of the ground under a point, or null if none within `maxDrop`. */
  groundHeight(at: THREE.Vector3, maxDrop?: number): number | null;
}

export interface CharacterHitbox {
  kind: HitboxKind;
  /** Local-space half extents. */
  size: THREE.Vector3;
  /** Object the box is parented to (bone or root). */
  attachTo: THREE.Object3D;
  offset: THREE.Vector3;
  damageMultiplier: number;
}

export const enum RayMask {
  World = 1 << 0,
  Characters = 1 << 1,
  Props = 1 << 2,
  Water = 1 << 3,
  /** World + props: what a camera or movement capsule should collide with. */
  Solid = (1 << 0) | (1 << 2),
  All = 0xffff,
}

// ---------------------------------------------------------------------------
// Weapons
// ---------------------------------------------------------------------------

export type FireMode = 'auto' | 'semi' | 'burst' | 'bolt';

export interface WeaponSpec {
  id: string;
  displayName: string;
  category: 'ar' | 'smg' | 'lmg' | 'sniper' | 'shotgun' | 'pistol';
  fireMode: FireMode;
  /** Rounds per minute. */
  rpm: number;
  burstCount?: number;
  magSize: number;
  reserveAmmo: number;
  /** Damage at point blank. */
  damage: number;
  /** Damage multiplier falls off linearly between these ranges (metres). */
  damageRangeNear: number;
  damageRangeFar: number;
  /** Multiplier applied at/after damageRangeFar. */
  damageFalloff: number;
  /** Muzzle velocity m/s. 0 means pure hitscan. */
  muzzleVelocity: number;
  /** Pellets per shot (shotguns). */
  pellets: number;
  /** Cone half-angle in radians when hip firing / aiming. */
  spreadHip: number;
  spreadAds: number;
  /** Vertical/horizontal recoil impulse in radians per shot. */
  recoilVertical: number;
  recoilHorizontal: number;
  /** How fast the view returns to centre, 0..1 per frame at 60fps. */
  recoilRecovery: number;
  /** Seconds. */
  adsTime: number;
  reloadTime: number;
  reloadEmptyTime: number;
  drawTime: number;
  /** FOV multiplier when aiming (e.g. 0.6 for a 1.6x scope feel). */
  adsFovScale: number;
  /** Penetration power in "concrete-centimetres". */
  penetration: number;
  headshotMultiplier: number;
  /** Optic style for the viewmodel builder. */
  optic: 'irons' | 'reddot' | 'holo' | 'acog' | 'sniper';
}

export interface WeaponRuntime {
  readonly spec: WeaponSpec;
  readonly ammo: number;
  readonly reserve: number;
  readonly aiming: boolean;
  readonly reloading: boolean;
  /** Viewmodel root, parented under the view camera by the weapon system. */
  readonly viewmodel: THREE.Object3D;
  /** World-space muzzle transform, valid after lateUpdate. */
  getMuzzleWorld(out: THREE.Vector3): THREE.Vector3;
}

// ---------------------------------------------------------------------------
// Characters / AI
// ---------------------------------------------------------------------------

export interface DamageTarget {
  readonly entityId: number;
  readonly alive: boolean;
  readonly position: THREE.Vector3;
  applyDamage(amount: number, hitbox: HitboxKind, from: THREE.Vector3, attackerId: number): void;
}

/** Registry so ballistics can resolve an entity id to something damageable. */
export interface EntityRegistry {
  register(target: DamageTarget): void;
  unregister(entityId: number): void;
  get(entityId: number): DamageTarget | undefined;
  readonly all: readonly DamageTarget[];
}

// ---------------------------------------------------------------------------
// Quality / settings
// ---------------------------------------------------------------------------

export type QualityPreset = 'low' | 'medium' | 'high' | 'ultra';

export interface QualitySettings {
  preset: QualityPreset;
  renderScale: number;
  shadowMapSize: number;
  shadowCascades: number;
  ssao: boolean;
  ssr: boolean;
  bloom: boolean;
  motionBlur: boolean;
  taa: boolean;
  volumetrics: boolean;
  anisotropy: number;
  particleBudget: number;
  decalBudget: number;
  /** Target frames per second the adaptive resolution aims for. */
  targetFps: number;
}

// ---------------------------------------------------------------------------
// The context handed to every system
// ---------------------------------------------------------------------------

export interface GameContext {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  /** The camera actually rendered. Owned by the camera rig. */
  readonly camera: THREE.PerspectiveCamera;
  /** Separate scene + camera for the viewmodel, rendered on top without clipping. */
  readonly viewScene: THREE.Scene;
  readonly viewCamera: THREE.PerspectiveCamera;
  readonly input: InputState;
  readonly events: EventBus;
  readonly physics: PhysicsWorld;
  readonly entities: EntityRegistry;
  readonly quality: QualitySettings;
  readonly time: FrameTime;
  /** Viewport size in CSS pixels. */
  readonly width: number;
  readonly height: number;
  /** Look up a registered system by name, e.g. ctx.system('weapons'). */
  system<T extends System>(name: string): T | undefined;
  /** Environment map for IBL, set by the sky system. */
  environment: THREE.Texture | null;
  /** Local player's entity id. */
  readonly localPlayerId: number;
  paused: boolean;
}

/** Systems are constructed with no arguments and receive ctx in init(). */
export type SystemFactory = () => System;

// ---------------------------------------------------------------------------
// Constants shared across subsystems
// ---------------------------------------------------------------------------

export const UNITS = {
  /** One world unit is one metre. */
  metre: 1,
  playerHeight: 1.8,
  playerCrouchHeight: 1.1,
  playerRadius: 0.32,
  eyeOffset: 1.65,
  crouchEyeOffset: 0.95,
  gravity: 19.6,
} as const;

export const LAYERS = {
  default: 0,
  viewmodel: 1,
  /** Objects excluded from reflections/SSR. */
  noReflect: 2,
} as const;
