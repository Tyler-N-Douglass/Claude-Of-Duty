/**
 * Viewmodel animation.
 *
 * Everything that moves here moves on a spring or a keyframe track with per-key
 * easing. There is not a single `lerp(a, b, 0.1)` in this file, because that is
 * the tell: an exponential chase has no overshoot, no settle and no weight, and
 * a weapon that has no weight reads as a prop pasted onto the screen.
 *
 * Conventions. The weapon lives in "rig space", which is the camera's frame:
 * -Z forward, +Y up, +X right. Rotations are applied ZXY-free — poses are
 * composed as quaternions, so blending an ADS pose against a sprint pose never
 * gimbal-locks. Angle signs, once and for all:
 *   +pitch (about X) points the muzzle up
 *   +yaw   (about Y) points the muzzle left
 *   +roll  (about Z) tips the weapon's top edge left
 */
import * as THREE from 'three';
import type { WeaponVisual, WeaponFeel } from './WeaponSpecs';

// ---------------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------------

export type Ease = 'linear' | 'in' | 'out' | 'inout' | 'out28' | 'back' | 'snap' | 'settle';

/**
 * `out28` is the ADS curve: t^0.28 moves 62% of the way in the first 20% of the
 * time and then eases into the sight picture, which is what makes aiming feel
 * instant without the pose snapping.
 */
export function ease(kind: Ease, t: number): number {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  switch (kind) {
    case 'linear': return x;
    case 'in': return x * x * x;
    case 'out': return 1 - Math.pow(1 - x, 3);
    case 'inout': return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    case 'out28': return Math.pow(x, 0.28);
    case 'back': {
      const c1 = 1.70158;
      const c3 = c1 + 1;
      return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
    }
    case 'snap': return 1 - Math.pow(1 - x, 6);
    case 'settle': {
      // Damped cosine: arrives, overshoots ~9%, settles. Weight, in one line.
      if (x >= 1) return 1;
      return 1 - Math.exp(-6.2 * x) * Math.cos(9.4 * x);
    }
  }
}

// ---------------------------------------------------------------------------
// Springs
// ---------------------------------------------------------------------------

const MAX_SUBSTEP = 1 / 240;

/**
 * Damped harmonic oscillator. `zeta` below 1 overshoots (recoil, flicks), at 1
 * it is critically damped (poses that must not wobble). Substepped so a frame
 * spike cannot make it explode.
 */
export class Spring {
  value = 0;
  velocity = 0;
  target = 0;

  constructor(public stiffness: number, public zeta = 1) {}

  step(dt: number): number {
    let remaining = dt;
    const k = this.stiffness;
    const c = 2 * this.zeta * Math.sqrt(k);
    while (remaining > 1e-6) {
      const h = Math.min(remaining, MAX_SUBSTEP);
      const a = -k * (this.value - this.target) - c * this.velocity;
      this.velocity += a * h;
      this.value += this.velocity * h;
      remaining -= h;
    }
    return this.value;
  }

  kick(impulse: number): void {
    this.velocity += impulse;
  }

  reset(v = 0): void {
    this.value = v;
    this.velocity = 0;
    this.target = v;
  }
}

/** Three independent springs sharing a tuning; used for positional lag. */
export class Spring3 {
  readonly x: Spring;
  readonly y: Spring;
  readonly z: Spring;

  constructor(stiffness: number, zeta = 1) {
    this.x = new Spring(stiffness, zeta);
    this.y = new Spring(stiffness, zeta);
    this.z = new Spring(stiffness, zeta);
  }

  setTarget(x: number, y: number, z: number): void {
    this.x.target = x;
    this.y.target = y;
    this.z.target = z;
  }

  step(dt: number, out: THREE.Vector3): THREE.Vector3 {
    return out.set(this.x.step(dt), this.y.step(dt), this.z.step(dt));
  }

  kick(x: number, y: number, z: number): void {
    this.x.kick(x);
    this.y.kick(y);
    this.z.kick(z);
  }

  reset(): void {
    this.x.reset();
    this.y.reset();
    this.z.reset();
  }
}

/**
 * An angle that chases another angle on a spring, exposing the *error* — the
 * amount by which it is behind. That error is the sway: the weapon is not
 * offset by a magic number, it is genuinely lagging the camera, so a hard flick
 * throws it off centre by exactly as much as the flick was hard.
 */
class LagAngle {
  private value = 0;
  private velocity = 0;
  private primed = false;

  constructor(private readonly stiffness: number, private readonly zeta: number) {}

  update(target: number, dt: number, maxError: number): number {
    if (!this.primed) {
      this.value = target;
      this.primed = true;
      return 0;
    }
    const k = this.stiffness;
    const c = 2 * this.zeta * Math.sqrt(k);
    let remaining = dt;
    while (remaining > 1e-6) {
      const h = Math.min(remaining, MAX_SUBSTEP);
      const a = -k * (this.value - target) - c * this.velocity;
      this.velocity += a * h;
      this.value += this.velocity * h;
      remaining -= h;
    }
    let err = this.value - target;
    if (err > maxError) { err = maxError; this.value = target + maxError; this.velocity *= 0.5; }
    else if (err < -maxError) { err = -maxError; this.value = target - maxError; this.velocity *= 0.5; }
    return err;
  }

  reset(): void {
    this.primed = false;
    this.velocity = 0;
  }
}

// ---------------------------------------------------------------------------
// Keyframe tracks
// ---------------------------------------------------------------------------

export interface Key {
  /** Normalised time, 0..1 through the clip. */
  t: number;
  v: number;
  /** Easing applied on the segment arriving at this key. */
  e?: Ease;
}

export type Track = readonly Key[];

/** Evaluates a track at normalised time. Tracks must be sorted by `t`. */
export function evalTrack(track: Track, t: number): number {
  const n = track.length;
  if (n === 0) return 0;
  const first = track[0]!;
  if (t <= first.t) return first.v;
  const last = track[n - 1]!;
  if (t >= last.t) return last.v;
  for (let i = 1; i < n; i++) {
    const b = track[i]!;
    if (t <= b.t) {
      const a = track[i - 1]!;
      const span = b.t - a.t;
      const local = span > 1e-6 ? (t - a.t) / span : 1;
      return a.v + (b.v - a.v) * ease(b.e ?? 'inout', local);
    }
  }
  return last.v;
}

/** Every channel a reload drives. All values are additive offsets. */
export interface ReloadClip {
  duration: number;
  /** Weapon-local pose offsets. */
  px: Track; py: Track; pz: Track;
  rx: Track; ry: Track; rz: Track;
  /** Magazine offsets in weapon space, plus its cant as it drops/seats. */
  magY: Track; magZ: Track; magRot: Track;
  /** 0 while the old mag is gone and the new one has not arrived. */
  magVisible: Track;
  /** Support-hand offsets. */
  handY: Track; handZ: Track; handX: Track;
  /** Charging handle pull, 0..1 (empty reloads only). */
  charge: Track;
  /** Normalised times at which discrete events fire. */
  tMagRelease: number;
  tMagSeat: number;
  tCharge: number;
}

/**
 * Builds the reload clip. Times are normalised so the same choreography reads
 * correctly whether it plays over 1.5 s or 3.9 s; the shapes were tuned against
 * a 2.1 s rifle reload and then checked at both extremes.
 */
export function buildReloadClip(duration: number, empty: boolean): ReloadClip {
  const tRelease = 0.17;
  const tSeat = empty ? 0.56 : 0.62;
  const tCharge = empty ? 0.76 : 1.1;

  return {
    duration,
    // Weapon drops and rolls toward the shooter so the magwell is presented,
    // then returns with a small overshoot as the hand comes off it.
    px: [{ t: 0, v: 0 }, { t: 0.16, v: 0.012, e: 'out' }, { t: 0.70, v: 0.010 }, { t: 1, v: 0, e: 'settle' }],
    py: [{ t: 0, v: 0 }, { t: 0.18, v: -0.040, e: 'out' }, { t: 0.66, v: -0.034 }, { t: 1, v: 0, e: 'settle' }],
    pz: [{ t: 0, v: 0 }, { t: 0.18, v: 0.026, e: 'out' }, { t: 0.70, v: 0.022 }, { t: 1, v: 0, e: 'settle' }],
    rx: [{ t: 0, v: 0 }, { t: 0.20, v: -0.30, e: 'out' }, { t: 0.66, v: -0.26 }, { t: 1, v: 0, e: 'settle' }],
    ry: [{ t: 0, v: 0 }, { t: 0.20, v: -0.16, e: 'out' }, { t: 0.70, v: -0.14 }, { t: 1, v: 0, e: 'settle' }],
    rz: [{ t: 0, v: 0 }, { t: 0.20, v: 0.42, e: 'out' }, { t: 0.66, v: 0.38 }, { t: 1, v: 0, e: 'settle' }],

    // The old magazine is released, falls a few centimetres and is handed off
    // to the physics debris pool; the fresh one rises into the well and seats.
    magY: [
      { t: 0, v: 0 },
      { t: tRelease, v: 0, e: 'linear' },
      { t: tRelease + 0.10, v: -0.075, e: 'in' },
      { t: tSeat - 0.16, v: -0.085, e: 'linear' },
      { t: tSeat, v: 0, e: 'out' },
      { t: 1, v: 0 },
    ],
    magZ: [
      { t: 0, v: 0 },
      { t: tRelease, v: 0, e: 'linear' },
      { t: tSeat - 0.16, v: 0.020, e: 'out' },
      { t: tSeat, v: 0, e: 'snap' },
      { t: 1, v: 0 },
    ],
    magRot: [
      { t: 0, v: 0 },
      { t: tRelease, v: 0, e: 'linear' },
      { t: tSeat - 0.16, v: 0.22, e: 'out' },
      { t: tSeat, v: 0, e: 'snap' },
      { t: 1, v: 0 },
    ],
    magVisible: [
      { t: 0, v: 1 },
      { t: tRelease + 0.11, v: 1, e: 'linear' },
      { t: tRelease + 0.111, v: 0, e: 'linear' },
      { t: tSeat - 0.17, v: 0, e: 'linear' },
      { t: tSeat - 0.169, v: 1, e: 'linear' },
      { t: 1, v: 1 },
    ],

    // Support hand leaves the handguard, fetches a magazine, slaps it home,
    // then (on an empty reload) goes up for the charging handle.
    handX: [
      { t: 0, v: 0 },
      { t: tRelease, v: -0.010, e: 'out' },
      { t: tSeat - 0.14, v: -0.030, e: 'inout' },
      { t: tSeat, v: -0.004, e: 'snap' },
      { t: empty ? tCharge : 1, v: empty ? 0.014 : 0, e: 'inout' },
      { t: 1, v: 0, e: 'settle' },
    ],
    handY: [
      { t: 0, v: 0 },
      { t: tRelease, v: -0.030, e: 'out' },
      { t: tSeat - 0.14, v: -0.105, e: 'inout' },
      { t: tSeat, v: -0.012, e: 'snap' },
      { t: empty ? tCharge : 1, v: empty ? 0.052 : 0, e: 'inout' },
      { t: 1, v: 0, e: 'settle' },
    ],
    handZ: [
      { t: 0, v: 0 },
      { t: tRelease, v: 0.030, e: 'out' },
      { t: tSeat - 0.14, v: 0.086, e: 'inout' },
      { t: tSeat, v: 0.012, e: 'snap' },
      { t: empty ? tCharge : 1, v: empty ? 0.060 : 0, e: 'inout' },
      { t: 1, v: 0, e: 'settle' },
    ],

    charge: empty
      ? [
        { t: 0, v: 0 },
        { t: tCharge, v: 0, e: 'linear' },
        { t: tCharge + 0.09, v: 1, e: 'out' },
        { t: tCharge + 0.13, v: 0, e: 'snap' },
        { t: 1, v: 0 },
      ]
      : [{ t: 0, v: 0 }, { t: 1, v: 0 }],

    tMagRelease: tRelease,
    tMagSeat: tSeat,
    tCharge: empty ? tCharge + 0.09 : -1,
  };
}

// ---------------------------------------------------------------------------
// Pose output
// ---------------------------------------------------------------------------

export interface AnimPose {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  /** Magazine offset in weapon space and its cant about X. */
  magOffset: THREE.Vector3;
  magRot: number;
  magVisible: boolean;
  /** Bolt / slide travel back along +Z, metres. */
  boltBack: number;
  /** Charging handle travel back along +Z, metres. */
  chargeBack: number;
  /** 0..1 trigger blade rotation. */
  triggerPull: number;
  /** 0..1 dust-cover open. */
  dustOpen: number;
  /** Support hand offset in weapon space. */
  handOffset: THREE.Vector3;
  /** 0..1 eased ADS blend. */
  adsBlend: number;
  /** 0..1 eased sprint blend. */
  sprintBlend: number;
}

export interface MotionInput {
  /** Camera yaw and pitch, radians, this frame. */
  yaw: number;
  pitch: number;
  /** Horizontal speed and the speed of the current gait, m/s. */
  speed: number;
  gaitSpeed: number;
  /** 0..1 through the two-footfall stride cycle. */
  stridePhase: number;
  grounded: boolean;
  sprinting: boolean;
  crouched: boolean;
  /** Vertical velocity, for the airborne float. */
  verticalVelocity: number;
}

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const SWAY = {
  yawStiffness: 130,
  yawZeta: 0.62,
  pitchStiffness: 155,
  pitchZeta: 0.66,
  maxYawError: 0.20,
  maxPitchError: 0.16,
  /** Metres of positional lag per radian of angular error. */
  posPerYaw: 0.115,
  posPerPitch: 0.098,
  /** Radians of counter-rotation per radian of angular error. */
  rotPerYaw: 0.58,
  rotPerPitch: 0.52,
  /** Roll induced by a horizontal flick — the weapon banks into the turn. */
  rollPerYaw: 0.42,
  /** Sway is heavily suppressed while aiming; the sight picture must hold. */
  adsScale: 0.22,
} as const;

const BREATH = {
  /** Two incommensurate rates, so the cycle never visibly repeats. */
  slow: 0.223,
  fast: 0.3541,
  ampY: 0.0022,
  ampX: 0.0013,
  ampPitch: 0.0075,
  ampRoll: 0.0052,
  driftRate: 0.0713,
  driftAmp: 0.0016,
  adsScale: 0.42,
} as const;

const WALK = {
  ampX: 0.0130,
  ampY: 0.0105,
  ampRoll: 0.030,
  ampPitch: 0.016,
  /** Bob rises with speed but saturates: sprinting is not four times walking. */
  saturate: 0.85,
  adsScale: 0.24,
} as const;

/** The sprint pose. Canted 35 degrees, lowered, pulled right, muzzle up-left. */
const SPRINT = {
  px: 0.030,
  py: -0.062,
  pz: 0.052,
  pitch: 0.30,
  yaw: 0.34,
  roll: -0.61,
  blendIn: 0.22,
  blendOut: 0.22,
  swingAmpX: 0.020,
  swingAmpY: 0.016,
  swingAmpRoll: 0.075,
} as const;

const AIR = {
  posY: 0.020,
  pitch: 0.075,
  stiffness: 90,
  zeta: 0.7,
} as const;

const _tmpQ = new THREE.Quaternion();
const _tmpQ2 = new THREE.Quaternion();
const _tmpE = new THREE.Euler(0, 0, 0, 'XYZ');
/** Metres the hip pose is pushed down the bore, away from the eye. */
const HIP_DEPTH_PUSH = 0.185;
/** The lateral/vertical hold widens a little to match the extra depth. */
const HIP_LATERAL_SCALE = 1.18;

const _tmpV = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();

function quatFromEuler(x: number, y: number, z: number, out: THREE.Quaternion): THREE.Quaternion {
  _tmpE.set(x, y, z, 'XYZ');
  return out.setFromEuler(_tmpE);
}

// ---------------------------------------------------------------------------
// The animator
// ---------------------------------------------------------------------------

export type AnimEvent = 'mag.release' | 'mag.seat' | 'charge.pull' | 'reload.done' | 'draw.done';

export class WeaponAnimator {
  readonly pose: AnimPose = {
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    magOffset: new THREE.Vector3(),
    magRot: 0,
    magVisible: true,
    boltBack: 0,
    chargeBack: 0,
    triggerPull: 0,
    dustOpen: 0,
    handOffset: new THREE.Vector3(),
    adsBlend: 0,
    sprintBlend: 0,
  };

  /** Set by the weapon system when the weapon changes. */
  private visual: WeaponVisual | null = null;
  private feel: WeaponFeel | null = null;

  // --- sway ---------------------------------------------------------------
  private readonly lagYaw = new LagAngle(SWAY.yawStiffness, SWAY.yawZeta);
  private readonly lagPitch = new LagAngle(SWAY.pitchStiffness, SWAY.pitchZeta);

  // --- recoil -------------------------------------------------------------
  private readonly recoilPos = new Spring3(260, 0.44);
  private readonly recoilRot = new Spring3(260, 0.40);

  // --- blends -------------------------------------------------------------
  private adsRaw = 0;
  private adsTime = 0.25;
  private aimingWanted = false;
  private sprintRaw = 0;
  private sprintWanted = false;
  /** Counts down the raise-to-fire delay after sprinting. */
  private sprintOut = 0;

  private readonly airSpring = new Spring(AIR.stiffness, AIR.zeta);

  // --- clip playback ------------------------------------------------------
  private clip: ReloadClip | null = null;
  private clipTime = 0;
  private firedRelease = false;
  private firedSeat = false;
  private firedCharge = false;

  private drawTime = 0;
  private drawDuration = 0;
  private holsterTime = 0;
  private holsterDuration = 0;

  // --- mechanism ----------------------------------------------------------
  private boltTimer = 0;
  private boltDuration = 0;
  private boltThrow = 0;
  /** True when the charging handle is fixed to the bolt and strokes with it. */
  private reciprocating = false;
  private triggerTimer = 0;
  private dustTimer = 0;

  private elapsed = 0;
  private seedPhase = Math.random() * 100;

  /** ADS target pose, recomputed each frame by the weapon system. */
  private readonly adsPos = new THREE.Vector3();
  private readonly adsQuat = new THREE.Quaternion();
  private adsValid = false;

  private readonly hipPos = new THREE.Vector3();
  private readonly hipQuat = new THREE.Quaternion();

  private listener: ((e: AnimEvent) => void) | null = null;

  onEvent(fn: (e: AnimEvent) => void): void {
    this.listener = fn;
  }

  private emit(e: AnimEvent): void {
    this.listener?.(e);
  }

  // -- configuration -------------------------------------------------------

  setWeapon(visual: WeaponVisual, feel: WeaponFeel, adsTime: number): void {
    this.visual = visual;
    this.feel = feel;
    this.adsTime = Math.max(0.06, adsTime);
    // The authored hip offsets place the grip ~0.15m from the eye, which puts a
    // 0.7m rifle's buttstock behind the near plane: the receiver then projects
    // at extreme perspective and eats the bottom-right quadrant. Real games sit
    // the grip at roughly a third of a metre so the whole weapon is in front of
    // the camera and the near/far foreshortening across its length stays around
    // 2.5:1 rather than 10:1. The per-weapon variation is preserved; only the
    // common depth baseline moves.
    this.hipPos.set(
      visual.hipPos[0] * HIP_LATERAL_SCALE,
      visual.hipPos[1] * HIP_LATERAL_SCALE,
      visual.hipPos[2] - HIP_DEPTH_PUSH,
    );
    quatFromEuler(visual.hipRot[0], visual.hipRot[1], visual.hipRot[2], this.hipQuat);
    this.boltThrow = feel.boltThrow;
    // An AR's charging handle is non-reciprocating; an AK-pattern side lever and
    // a pump forend are physically part of the moving group.
    this.reciprocating = visual.charging === 'side' || visual.charging === 'pump';
    for (const sp of [this.recoilPos.x, this.recoilPos.y, this.recoilPos.z]) sp.stiffness = feel.kickStiffness;
    for (const sp of [this.recoilRot.x, this.recoilRot.y, this.recoilRot.z]) sp.stiffness = feel.kickStiffness * 1.15;
    this.recoilPos.reset();
    this.recoilRot.reset();
    this.lagYaw.reset();
    this.lagPitch.reset();
    this.clip = null;
    this.boltTimer = 0;
    this.triggerTimer = 0;
    this.dustTimer = 0;
    this.adsRaw = 0;
    this.aimingWanted = false;
    this.seedPhase = Math.random() * 100;
  }

  /** Called every frame with the world-space-independent ADS target. */
  setAdsTarget(position: THREE.Vector3, quaternion: THREE.Quaternion): void {
    this.adsPos.copy(position);
    this.adsQuat.copy(quaternion);
    this.adsValid = true;
  }

  // -- state -------------------------------------------------------------

  setAiming(v: boolean): void {
    this.aimingWanted = v;
  }

  setSprinting(v: boolean): void {
    if (v && !this.sprintWanted) this.sprintOut = 0;
    if (!v && this.sprintWanted && this.feel) this.sprintOut = this.feel.sprintOutTime;
    this.sprintWanted = v;
  }

  /**
   * True while the weapon is still coming up out of the sprint pose. The weapon
   * system refuses to fire during this window — CoD's raise-to-fire delay, and
   * the single biggest reason sprinting feels like a commitment.
   */
  get raising(): boolean {
    return this.sprintOut > 0;
  }

  get aiming(): boolean {
    return this.aimingWanted && this.adsRaw > 0.02;
  }

  get adsProgress(): number {
    return this.pose.adsBlend;
  }

  get reloading(): boolean {
    return this.clip !== null;
  }

  get busy(): boolean {
    return this.clip !== null || this.drawTime > 0 || this.holsterTime > 0;
  }

  // -- one-shot triggers ---------------------------------------------------

  startReload(duration: number, empty: boolean): void {
    this.clip = buildReloadClip(duration, empty);
    this.clipTime = 0;
    this.firedRelease = false;
    this.firedSeat = false;
    this.firedCharge = false;
    this.aimingWanted = false;
  }

  cancelReload(): void {
    this.clip = null;
    this.pose.magVisible = true;
  }

  startDraw(duration: number): void {
    this.drawDuration = Math.max(0.05, duration);
    this.drawTime = this.drawDuration;
    this.holsterTime = 0;
    this.clip = null;
  }

  startHolster(duration: number): void {
    this.holsterDuration = Math.max(0.05, duration);
    this.holsterTime = this.holsterDuration;
    this.clip = null;
  }

  /** Applies one shot's worth of impulse to the viewmodel springs. */
  fire(patternH: number, patternV: number, interval: number): void {
    const f = this.feel;
    if (!f) return;
    const ads = this.pose.adsBlend;
    // Aiming plants the stock in the shoulder: the same round moves the
    // viewmodel much less, which is what makes ADS feel controllable.
    const damp = 1 - ads * 0.42;

    this.recoilPos.kick(
      -patternH * f.kickBack * 26 * damp,
      f.kickUp * 42 * damp,
      f.kickBack * 46 * damp,
    );
    this.recoilRot.kick(
      f.kickPitch * 30 * damp * (0.85 + patternV * 0.25),
      -patternH * f.kickYaw * 34 * damp,
      -patternH * f.kickRoll * 26 * damp - f.kickRoll * 6 * damp,
    );

    this.triggerTimer = Math.min(0.09, interval * 0.55);
    this.dustTimer = 0.55;
    this.boltDuration = Math.max(0.035, Math.min(interval * f.boltCycleFraction, 0.34));
    this.boltTimer = this.boltDuration;
  }

  /** Manual bolt cycle for bolt-action rifles and pump guns. */
  cycleAction(duration: number): void {
    this.boltDuration = Math.max(0.08, duration);
    this.boltTimer = this.boltDuration;
    this.dustTimer = 0.5;
  }

  // -- per-frame -----------------------------------------------------------

  update(dt: number, m: MotionInput): AnimPose {
    const v = this.visual;
    const f = this.feel;
    const pose = this.pose;
    if (!v || !f) return pose;

    this.elapsed += dt;

    // --- blend states ----------------------------------------------------
    if (this.sprintOut > 0) this.sprintOut = Math.max(0, this.sprintOut - dt);

    const sprintTarget = this.sprintWanted && !this.aimingWanted && this.clip === null ? 1 : 0;
    const sprintRate = 1 / (sprintTarget > this.sprintRaw ? SPRINT.blendIn : SPRINT.blendOut);
    this.sprintRaw = approach(this.sprintRaw, sprintTarget, sprintRate * dt);
    const sprintBlend = ease('inout', this.sprintRaw);
    pose.sprintBlend = sprintBlend;

    const canAim = this.adsValid && this.clip === null && this.drawTime <= 0 && this.holsterTime <= 0;
    const adsTarget = this.aimingWanted && canAim && !this.sprintWanted ? 1 : 0;
    // Coming out of ADS is faster than going in, as it is in every shooter.
    const adsRate = 1 / (adsTarget > this.adsRaw ? this.adsTime : this.adsTime * 0.78);
    this.adsRaw = approach(this.adsRaw, adsTarget, adsRate * dt);
    const adsBlend = ease('out28', this.adsRaw);
    pose.adsBlend = adsBlend;

    // --- base pose: hip, ADS, sprint -------------------------------------
    const basePos = _tmpV.copy(this.hipPos);
    const baseQuat = _tmpQ.copy(this.hipQuat);

    if (adsBlend > 0.0005 && this.adsValid) {
      basePos.lerp(this.adsPos, adsBlend);
      baseQuat.slerp(this.adsQuat, adsBlend);
    }

    if (sprintBlend > 0.0005) {
      // Arm swing rides the footstep cadence, a quarter cycle behind the feet.
      const swing = Math.sin((m.stridePhase - 0.25) * Math.PI * 2);
      const swing2 = Math.sin((m.stridePhase - 0.25) * Math.PI * 4);
      const sx = SPRINT.px + swing * SPRINT.swingAmpX;
      const sy = SPRINT.py + Math.abs(swing2) * SPRINT.swingAmpY - SPRINT.swingAmpY * 0.5;
      const sprintPos = _tmpV2.set(this.hipPos.x + sx, this.hipPos.y + sy, this.hipPos.z + SPRINT.pz);
      quatFromEuler(
        v.hipRot[0] + SPRINT.pitch,
        v.hipRot[1] + SPRINT.yaw,
        v.hipRot[2] + SPRINT.roll + swing * SPRINT.swingAmpRoll,
        _tmpQ2,
      );
      basePos.lerp(sprintPos, sprintBlend);
      baseQuat.slerp(_tmpQ2, sprintBlend);
    }

    // --- sway: the weapon lags the camera --------------------------------
    const swayScale = (1 - adsBlend * (1 - SWAY.adsScale)) * (1 - sprintBlend * 0.45);
    const yawErr = this.lagYaw.update(m.yaw, dt, SWAY.maxYawError);
    const pitchErr = this.lagPitch.update(m.pitch, dt, SWAY.maxPitchError);

    let px = yawErr * SWAY.posPerYaw * swayScale;
    let py = -pitchErr * SWAY.posPerPitch * swayScale;
    let pz = 0;
    let rx = -pitchErr * SWAY.rotPerPitch * swayScale;
    let ry = yawErr * SWAY.rotPerYaw * swayScale;
    let rz = -yawErr * SWAY.rollPerYaw * swayScale;

    // --- idle breathing --------------------------------------------------
    const bScale = (1 - adsBlend * (1 - BREATH.adsScale)) * (1 - sprintBlend);
    const t = this.elapsed + this.seedPhase;
    const b1 = Math.sin(t * Math.PI * 2 * BREATH.slow);
    const b2 = Math.sin(t * Math.PI * 2 * BREATH.fast + 1.21);
    const drift = Math.sin(t * Math.PI * 2 * BREATH.driftRate + 0.62);
    px += (b2 * BREATH.ampX + drift * BREATH.driftAmp) * bScale;
    py += (b1 * BREATH.ampY + b2 * BREATH.ampY * 0.38) * bScale;
    rx += b1 * BREATH.ampPitch * bScale;
    rz += b2 * BREATH.ampRoll * bScale;

    // --- walk / sprint bob ----------------------------------------------
    const gait = Math.max(0.5, m.gaitSpeed);
    const speedT = Math.min(1, m.speed / gait);
    const bobAmt = Math.pow(speedT, WALK.saturate)
      * (m.grounded ? 1 : 0.15)
      * (1 - adsBlend * (1 - WALK.adsScale))
      * (m.crouched ? 0.7 : 1);
    if (bobAmt > 0.001) {
      const ph = m.stridePhase * Math.PI * 2;
      px += Math.sin(ph) * WALK.ampX * bobAmt;
      py += (Math.abs(Math.cos(ph)) - 0.6) * WALK.ampY * bobAmt;
      rz += Math.sin(ph) * WALK.ampRoll * bobAmt;
      rx += Math.cos(ph * 2) * WALK.ampPitch * bobAmt;
    }

    // --- airborne float --------------------------------------------------
    this.airSpring.target = m.grounded ? 0 : THREE.MathUtils.clamp(-m.verticalVelocity / 8, -1, 1);
    const air = this.airSpring.step(dt);
    py += air * AIR.posY * (1 - adsBlend * 0.6);
    rx += air * AIR.pitch * (1 - adsBlend * 0.6);

    // --- reload / draw / holster clips -----------------------------------
    pose.magOffset.set(0, 0, 0);
    pose.magRot = 0;
    pose.magVisible = true;
    pose.handOffset.set(0, 0, 0);
    let clipCharge = 0;

    if (this.clip) {
      const c = this.clip;
      this.clipTime += dt;
      const nt = Math.min(1, this.clipTime / c.duration);

      px += evalTrack(c.px, nt);
      py += evalTrack(c.py, nt);
      pz += evalTrack(c.pz, nt);
      rx += evalTrack(c.rx, nt);
      ry += evalTrack(c.ry, nt);
      rz += evalTrack(c.rz, nt);

      pose.magOffset.set(0, evalTrack(c.magY, nt), evalTrack(c.magZ, nt));
      pose.magRot = evalTrack(c.magRot, nt);
      pose.magVisible = evalTrack(c.magVisible, nt) > 0.5;
      pose.handOffset.set(evalTrack(c.handX, nt), evalTrack(c.handY, nt), evalTrack(c.handZ, nt));
      clipCharge = evalTrack(c.charge, nt);

      if (!this.firedRelease && nt >= c.tMagRelease) { this.firedRelease = true; this.emit('mag.release'); }
      if (!this.firedSeat && nt >= c.tMagSeat) {
        this.firedSeat = true;
        this.emit('mag.seat');
        // The slap is a real impulse, not a canned shake.
        this.recoilPos.kick(0, -0.34, 0.18);
        this.recoilRot.kick(-0.9, 0, 0.5);
      }
      if (!this.firedCharge && c.tCharge > 0 && nt >= c.tCharge) {
        this.firedCharge = true;
        this.emit('charge.pull');
        this.boltDuration = 0.10;
        this.boltTimer = this.boltDuration;
        this.recoilRot.kick(-0.5, 0, 0);
      }
      if (nt >= 1) {
        this.clip = null;
        this.emit('reload.done');
      }
    }

    if (this.drawTime > 0) {
      this.drawTime = Math.max(0, this.drawTime - dt);
      const k = 1 - this.drawTime / this.drawDuration;
      const e = ease('settle', k);
      // Comes up from below and to the right with a settle at the top.
      px += (1 - e) * 0.075;
      py += (1 - e) * -0.155;
      pz += (1 - e) * 0.045;
      rx += (1 - e) * -0.62;
      ry += (1 - e) * 0.30;
      rz += (1 - e) * -0.42;
      if (this.drawTime === 0) this.emit('draw.done');
    } else if (this.holsterTime > 0) {
      this.holsterTime = Math.max(0, this.holsterTime - dt);
      const k = 1 - this.holsterTime / this.holsterDuration;
      const e = ease('in', k);
      px += e * 0.070;
      py += e * -0.170;
      pz += e * 0.040;
      rx += e * -0.68;
      ry += e * 0.32;
      rz += e * -0.46;
    }

    // --- recoil springs --------------------------------------------------
    this.recoilPos.setTarget(0, 0, 0);
    this.recoilRot.setTarget(0, 0, 0);
    const rp = this.recoilPos.step(dt, _tmpV2);
    px += rp.x;
    py += rp.y;
    pz += rp.z;
    const rr = this.recoilRot;
    rx += rr.x.step(dt);
    ry += rr.y.step(dt);
    rz += rr.z.step(dt);

    // --- compose ---------------------------------------------------------
    pose.position.copy(basePos).add(_tmpV2.set(px, py, pz));
    quatFromEuler(rx, ry, rz, _tmpQ2);
    pose.quaternion.copy(baseQuat).multiply(_tmpQ2);

    // --- mechanism -------------------------------------------------------
    if (this.boltTimer > 0) {
      this.boltTimer = Math.max(0, this.boltTimer - dt);
      const k = 1 - this.boltTimer / this.boltDuration;
      // Back hard, forward under spring: 38% of the cycle out, 62% back.
      const travel = k < 0.38
        ? ease('out', k / 0.38)
        : 1 - ease('snap', (k - 0.38) / 0.62);
      pose.boltBack = travel * this.boltThrow;
    } else {
      pose.boltBack = 0;
    }

    pose.chargeBack = clipCharge * this.boltThrow * 1.6
      + (this.reciprocating ? pose.boltBack : 0);

    if (this.triggerTimer > 0) {
      this.triggerTimer = Math.max(0, this.triggerTimer - dt);
      pose.triggerPull = ease('out', Math.min(1, this.triggerTimer * 22));
    } else {
      pose.triggerPull = Math.max(0, pose.triggerPull - dt * 9);
    }

    if (this.dustTimer > 0) this.dustTimer = Math.max(0, this.dustTimer - dt);
    pose.dustOpen = Math.min(1, this.dustTimer * 6);

    return pose;
  }

  reset(): void {
    this.recoilPos.reset();
    this.recoilRot.reset();
    this.lagYaw.reset();
    this.lagPitch.reset();
    this.airSpring.reset();
    this.clip = null;
    this.drawTime = 0;
    this.holsterTime = 0;
    this.boltTimer = 0;
    this.adsRaw = 0;
    this.sprintRaw = 0;
    this.sprintOut = 0;
  }
}

function approach(current: number, target: number, maxDelta: number): number {
  const d = target - current;
  if (Math.abs(d) <= maxDelta) return target;
  return current + Math.sign(d) * maxDelta;
}
