/**
 * CameraRig — everything the player's eye does.
 *
 * The single rule that everything else bends around: raw aim is never smoothed,
 * filtered, accelerated or interpolated. `yaw` and `pitch` take the mouse delta
 * verbatim on the frame it arrives. Every other motion in this file — bob, sway,
 * lean, landing, recoil recovery, shake — is a *secondary* offset composed on
 * top of that, and only those are allowed to have inertia.
 *
 * Composition happens twice a frame: once in the player's update (so the shadow
 * cascades and anything else reading the camera see this frame's aim), and again
 * in lateUpdate (so recoil the weapon system applied during its own update lands
 * on the same frame it was fired, not the next one).
 */
import * as THREE from 'three';
import { RayMask, type GameContext } from '../core/Contracts';

const DEG = THREE.MathUtils.DEG2RAD;

export const RIG = {
  fovBase: 80,
  /** Viewmodel camera FOV at rest. */
  fovView: 60,
  /**
   * How much of a world-FOV change bleeds into the viewmodel camera. Keeping
   * this well under 1 is the classic separate-viewmodel-FOV trick: the world
   * opens up when you sprint but the weapon in your hands barely distorts.
   */
  fovViewFollow: 0.25,
  fovViewAdsFollow: 0.35,
  fovSprintPush: 8,
  fovSprintIn: 0.25,
  fovSprintOut: 0.15,
  fovSlidePush: 5.5,
  fovLandPunch: 3.4,

  pitchLimit: 88 * DEG,

  /** Bob amplitudes at full walk speed, metres / radians. */
  bobVertical: 0.012,
  bobLateral: 0.009,
  bobRoll: 0.35 * DEG,
  bobPitch: 0.16 * DEG,
  bobAdsScale: 0.25,

  swaySprintLateral: 0.028,
  swaySprintRoll: 1.15 * DEG,
  swaySprintPitch: 0.5 * DEG,

  /** Landing dip spring. Underdamped on purpose — it must settle, not glide. */
  landStiffness: 140,
  landDamping: 14,
  /**
   * Metres of dip per unit of landing strength. Tuned so a flat 0.62m jump dips
   * about 6cm and a bone-rattler tops out near 20cm — read off frame-stepped
   * CoD landings, which are far subtler than they feel in motion.
   */
  landScale: 0.017,
  landMaxDip: 0.34,

  leanAngle: 12 * DEG,
  leanOffset: 0.40,
  leanStiffness: 130,
  leanDamping: 20,
  /** Probe capsule used to stop a lean from clipping through a wall. */
  leanProbeRadius: 0.24,
  leanProbeHeight: 0.42,

  /**
   * Fraction of a recoil impulse that never comes back. This single number is
   * the difference between "recoil" and "screen shake": the transient part
   * decays to zero, the retained part is folded into the aim so a sustained
   * burst genuinely climbs and the player has to pull down through it.
   */
  recoilRetainPitch: 0.30,
  recoilRetainYaw: 0.45,
  /** Rate the transient target bleeds off, per second. */
  recoilRecovery: 9.0,
  /** Spring the visible kick follows the target with — gives a ~30ms rise. */
  recoilStiffness: 900,
  recoilDamping: 55,
  /** Camera pushback along view forward, metres per radian of kick. */
  recoilPushback: 0.10,

  shakeFrequency: 19,
  shakeRotation: 3.0 * DEG,
  shakePosition: 0.07,

  breathPitch: 0.13 * DEG,
  breathYaw: 0.10 * DEG,

  strafeRoll: 0.55 * DEG,
  /** Exponential follow rate for the eye height when the stance changes. */
  eyeFollow: 11,
} as const;

/** Everything the rig needs to know about the body it is bolted to. */
export interface RigMotion {
  /** World-space eye height, already step-smoothed by the movement controller. */
  eyeY: number;
  x: number;
  z: number;
  /** Horizontal speed, m/s. */
  speed: number;
  /** Speed the current gait tops out at, for normalising bob. */
  gaitSpeed: number;
  /** 0..1 through the two-footfall stride cycle; 0 and 0.5 are footfalls. */
  stridePhase: number;
  grounded: boolean;
  sprinting: boolean;
  tacSprinting: boolean;
  sliding: boolean;
  /** 0..1 through the slide. */
  slideT: number;
  /** -1..1 steering bias during a slide. */
  slideLateral: number;
  mantleActive: boolean;
  /** 0..1 through the mantle. */
  mantleT: number;
  crouched: boolean;
  aiming: boolean;
  /** 0..1 magnitude of the movement stick. */
  moveIntent: number;
  /** -1..1 strafe component of the movement stick. */
  strafeIntent: number;
  leanLeft: boolean;
  leanRight: boolean;
}

export interface DebugPose {
  pos?: number[] | { x: number; y: number; z: number };
  yaw?: number;
  pitch?: number;
  fov?: number;
}

// ---------------------------------------------------------------------------
// 1D gradient noise. Smooth by construction, so the shake reads as a physical
// wobble rather than the per-frame random jitter that screams "WebGL demo".
// ---------------------------------------------------------------------------

function hash(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

function gradNoise(x: number, seed: number): number {
  const i = Math.floor(x);
  const f = x - i;
  const ga = hash(i + seed) * 2 - 1;
  const gb = hash(i + 1 + seed) * 2 - 1;
  // Quintic fade (Perlin's improved curve): C2 continuous, no visible creases.
  const u = f * f * f * (f * (f * 6 - 15) + 10);
  return (ga * f + (gb * (f - 1) - ga * f) * u) * 2;
}

/**
 * Two octaves, normalised. 1D gradient noise is zero at every lattice point and
 * only reaches ~0.21 RMS, so the 1.8 puts the practical peak near 1 and lets the
 * shake amplitudes below be read as "degrees at trauma 1" rather than as an
 * arbitrary scale factor.
 */
function fbm(x: number, seed: number): number {
  return (gradNoise(x, seed) * 0.68 + gradNoise(x * 2.17 + 13.7, seed + 91) * 0.32) * 1.8;
}

function moveTowards(current: number, target: number, maxDelta: number): number {
  const d = target - current;
  if (Math.abs(d) <= maxDelta) return target;
  return current + Math.sign(d) * maxDelta;
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

// ---------------------------------------------------------------------------

const _right = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _probe = new THREE.Vector3();
const _dirTmp = new THREE.Vector3();

export class CameraRig {
  /** Raw aim. Nothing filters these. */
  yaw = 0;
  pitch = 0;

  /** Frozen by the QA harness through window.GAME.setDebugPose. */
  debugFrozen = false;

  private ctx: GameContext | null = null;
  private unsubShake: (() => void) | null = null;

  // eye
  private eyeY = 0;
  private eyeInit = false;
  /** Horizontal eye anchor, written by the player each frame. */
  private baseX = 0;
  private baseZ = 0;

  // recoil
  private kickPitch = 0;
  private kickYaw = 0;
  private kickPitchVel = 0;
  private kickYawVel = 0;
  private kickTargetPitch = 0;
  private kickTargetYaw = 0;
  private recoveryRate: number = RIG.recoilRecovery;

  // landing spring
  private landOffset = 0;
  private landVel = 0;
  private landPunch = 0;

  // lean
  private lean = 0;
  private leanVel = 0;

  // fov
  private sprintT = 0;
  private adsScale = 1;
  private adsBlend = 1;
  private adsRate = 1;
  private worldFov: number = RIG.fovBase;
  private viewFov: number = RIG.fovView;
  private appliedFov = -1;
  private appliedViewFov = -1;

  // shake
  private trauma = 0;
  private traumaDecay = 1;
  private traumaFreq: number = RIG.shakeFrequency;
  private noiseT = 0;

  // damage punch
  private punchPitch = 0;
  private punchYaw = 0;

  // composed offsets, written by advance() and consumed by apply()
  private oPitch = 0;
  private oYaw = 0;
  private oRoll = 0;
  private oUp = 0;
  private oSide = 0;
  private oFwd = 0;

  private time = 0;
  private lastMotionSpeed = 0;

  private readonly debugPos = new THREE.Vector3();
  private debugFov: number = RIG.fovBase;

  // -- lifecycle -----------------------------------------------------------

  init(ctx: GameContext): void {
    this.ctx = ctx;
    ctx.camera.rotation.order = 'YXZ';
    ctx.viewCamera.rotation.order = 'YXZ';
    this.worldFov = RIG.fovBase;
    this.viewFov = RIG.fovView;
    this.unsubShake = ctx.events.on('camera.shake', (p) => {
      this.addTrauma(p.amount, p.duration, p.frequency);
    });
  }

  dispose(): void {
    this.unsubShake?.();
    this.unsubShake = null;
    this.ctx = null;
  }

  // -- aim -----------------------------------------------------------------

  /**
   * Raw look. No smoothing, no acceleration, no clamped delta — whatever the
   * mouse reported this frame is what the view does this frame.
   */
  look(dx: number, dy: number): void {
    if (this.debugFrozen) return;
    this.yaw -= dx;
    this.pitch -= dy;
    if (this.pitch > RIG.pitchLimit) this.pitch = RIG.pitchLimit;
    else if (this.pitch < -RIG.pitchLimit) this.pitch = -RIG.pitchLimit;
    // Keep yaw in range so float precision never degrades in a long session.
    if (this.yaw > Math.PI) this.yaw -= Math.PI * 2;
    else if (this.yaw < -Math.PI) this.yaw += Math.PI * 2;
  }

  setAim(yaw: number, pitch: number): void {
    this.yaw = yaw;
    this.pitch = THREE.MathUtils.clamp(pitch, -RIG.pitchLimit, RIG.pitchLimit);
  }

  /** World-space forward the player is aiming along. */
  getForward(out: THREE.Vector3): THREE.Vector3 {
    const cp = Math.cos(this.pitch);
    return out.set(-Math.sin(this.yaw) * cp, Math.sin(this.pitch), -Math.cos(this.yaw) * cp);
  }

  // -- impulses ------------------------------------------------------------

  /**
   * Called by the weapon system, once per shot. `pitch` is the upward kick in
   * radians, `yaw` the horizontal component (signed).
   */
  applyRecoil(pitch: number, yaw: number, recovery?: number): void {
    if (recovery !== undefined && recovery > 0) {
      // WeaponSpec.recoilRecovery is a 0..1 per-frame-at-60fps figure; convert
      // it into the continuous rate this rig integrates with.
      this.recoveryRate = THREE.MathUtils.clamp(-Math.log(Math.max(1e-3, 1 - recovery)) * 60, 3, 26);
    }
    const retainedP = pitch * RIG.recoilRetainPitch;
    const retainedY = yaw * RIG.recoilRetainYaw;
    // The retained part becomes real aim: the barrel is genuinely pointing
    // higher now and stays there until the player corrects it.
    this.pitch = THREE.MathUtils.clamp(this.pitch + retainedP, -RIG.pitchLimit, RIG.pitchLimit);
    this.yaw += retainedY;
    this.kickTargetPitch += pitch - retainedP;
    this.kickTargetYaw += yaw - retainedY;
  }

  addTrauma(amount: number, duration = 0.4, frequency?: number): void {
    if (amount <= 0) return;
    const d = Math.max(0.05, duration);
    // Blend the decay rate by how much each source contributes, so a small
    // ping during a long rumble does not cut the rumble short.
    const w = THREE.MathUtils.clamp(amount / (amount + this.trauma), 0, 1);
    this.traumaDecay = THREE.MathUtils.lerp(this.traumaDecay, 1 / d, w);
    if (frequency) this.traumaFreq = THREE.MathUtils.lerp(this.traumaFreq, frequency, w);
    this.trauma = Math.min(1, this.trauma + amount);
  }

  /** Directional flinch when the player is hit. */
  applyDamageKick(dirWorld: THREE.Vector3, amount: number): void {
    const mag = THREE.MathUtils.clamp(amount / 45, 0.15, 1);
    _dirTmp.copy(dirWorld).setY(0);
    const l = _dirTmp.length();
    let lateral = 0;
    if (l > 1e-4) {
      _dirTmp.multiplyScalar(1 / l);
      _right.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
      lateral = _dirTmp.dot(_right);
    }
    this.punchPitch += mag * 3.6 * DEG;
    this.punchYaw -= lateral * mag * 2.4 * DEG;
    this.addTrauma(mag * 0.42, 0.35, 26);
  }

  /** Landing dip. `impact` is the vertical contact speed in m/s. */
  applyLanding(impactSpeed: number): void {
    const i = Math.max(0, impactSpeed - 1.4);
    if (i <= 0) return;
    const strength = Math.min(1.6, i * 0.115);
    this.landVel -= strength * RIG.landStiffness * RIG.landScale;
    this.landPunch = Math.min(1, this.landPunch + strength * 0.8);
    if (strength > 0.55) this.addTrauma((strength - 0.5) * 0.5, 0.28, 24);
  }

  // -- per-frame -----------------------------------------------------------

  /**
   * Integrates every secondary motion. Called once per frame from update();
   * `apply` may then be called any number of times without advancing time.
   */
  advance(dt: number, m: RigMotion, ctx: GameContext): void {
    this.time += dt;
    if (this.debugFrozen) {
      this.zeroSecondary();
      return;
    }

    // -- eye height: a fast exponential follow so crouching reads as a smooth
    // duck rather than a teleport, but never slow enough to feel floaty.
    if (!this.eyeInit) {
      this.eyeY = m.eyeY;
      this.eyeInit = true;
    } else {
      this.eyeY += (m.eyeY - this.eyeY) * (1 - Math.exp(-RIG.eyeFollow * dt));
    }

    // -- recoil: the target bleeds toward zero while the visible kick springs
    // toward it. The spring is what gives the kick a ~30ms rise instead of a
    // one-frame snap, and the bleed is what brings the muzzle back down.
    const bleed = Math.exp(-this.recoveryRate * dt);
    this.kickTargetPitch *= bleed;
    this.kickTargetYaw *= bleed;
    const kp = -RIG.recoilStiffness * (this.kickPitch - this.kickTargetPitch) - RIG.recoilDamping * this.kickPitchVel;
    this.kickPitchVel += kp * dt;
    this.kickPitch += this.kickPitchVel * dt;
    const ky = -RIG.recoilStiffness * (this.kickYaw - this.kickTargetYaw) - RIG.recoilDamping * this.kickYawVel;
    this.kickYawVel += ky * dt;
    this.kickYaw += this.kickYawVel * dt;

    // -- damage punch decays on its own, faster than recoil.
    const punchDecay = Math.exp(-7.5 * dt);
    this.punchPitch *= punchDecay;
    this.punchYaw *= punchDecay;

    // -- landing spring.
    const acc = -RIG.landStiffness * this.landOffset - RIG.landDamping * this.landVel;
    this.landVel += acc * dt;
    this.landOffset += this.landVel * dt;
    if (this.landOffset < -RIG.landMaxDip) {
      this.landOffset = -RIG.landMaxDip;
      if (this.landVel < 0) this.landVel = 0;
    }
    this.landPunch *= Math.exp(-6 * dt);

    // -- lean, blocked by geometry so you cannot peek through a wall.
    const canLean = !m.sprinting && !m.sliding && !m.mantleActive && m.grounded;
    let leanTarget = canLean ? (m.leanRight ? 1 : 0) - (m.leanLeft ? 1 : 0) : 0;
    if (leanTarget !== 0) leanTarget *= this.probeLean(ctx, leanTarget);
    const la = -RIG.leanStiffness * (this.lean - leanTarget) - RIG.leanDamping * this.leanVel;
    this.leanVel += la * dt;
    this.lean += this.leanVel * dt;
    this.lean = THREE.MathUtils.clamp(this.lean, -1, 1);

    // -- bob. Phase-locked to the stride so the vertical bottoms out exactly on
    // the footfall; the lateral runs at half that rate, giving the figure-eight.
    const gait = Math.max(1, m.gaitSpeed);
    const speedN = THREE.MathUtils.clamp(m.speed / gait, 0, 1.15);
    const airborne = !m.grounded || m.mantleActive;
    // Bob strength eases in and out rather than tracking speed instantly.
    this.lastMotionSpeed += ((airborne ? 0 : speedN) - this.lastMotionSpeed) * (1 - Math.exp(-9 * dt));
    const amp = this.lastMotionSpeed * (m.aiming ? RIG.bobAdsScale : 1) * (m.crouched ? 0.7 : 1);

    const phase = m.stridePhase * Math.PI * 2;
    // x = sin(t), y = -cos(2t): a lissajous figure-eight whose y minima sit at
    // t = 0 and t = pi, i.e. on each footfall.
    const bobLat = Math.sin(phase) * RIG.bobLateral * amp;
    const bobUp = -Math.cos(phase * 2) * RIG.bobVertical * amp;
    const bobRoll = -Math.sin(phase) * RIG.bobRoll * amp;
    const bobPitch = -Math.cos(phase * 2) * RIG.bobPitch * amp;

    // -- sprint sway: slower, wider, and mostly lateral + roll.
    const sprintAmt = m.sprinting ? THREE.MathUtils.clamp(m.speed / gait, 0, 1) : 0;
    const sway = Math.sin(phase * 0.5);
    const swayLat = sway * RIG.swaySprintLateral * sprintAmt;
    const swayRoll = -sway * RIG.swaySprintRoll * sprintAmt;
    const swayPitch = Math.cos(phase) * RIG.swaySprintPitch * sprintAmt;

    // -- idle breathing: sub-degree, two incommensurate rates so it never loops.
    const idle = (1 - this.lastMotionSpeed) * (m.aiming ? 0.35 : 1);
    const breathPitch =
      (Math.sin(this.time * 1.77) * 0.62 + Math.sin(this.time * 0.93 + 1.3) * 0.38) * RIG.breathPitch * idle;
    const breathYaw =
      (Math.sin(this.time * 1.31 + 2.1) * 0.6 + Math.sin(this.time * 0.61) * 0.4) * RIG.breathYaw * idle;

    // -- strafe roll: the body leans into a lateral change of direction.
    const strafeRoll = -m.strafeIntent * RIG.strafeRoll * (airborne ? 0.4 : 1) * (m.aiming ? 0.4 : 1);

    // -- slide: dip and roll into the direction you are steering.
    const slideEnv = m.sliding ? Math.sin(Math.PI * Math.min(1, 0.15 + m.slideT * 0.85)) : 0;
    const slideRoll = -m.slideLateral * 5.5 * DEG * slideEnv - (m.sliding ? 2.2 * DEG * slideEnv : 0);
    const slideDip = m.sliding ? -0.055 * slideEnv : 0;
    const slidePitch = m.sliding ? -1.6 * DEG * slideEnv : 0;

    // -- mantle: the view drops as you load the legs, then rises over the lip.
    let mantlePitch = 0;
    let mantleUp = 0;
    let mantleRoll = 0;
    if (m.mantleActive) {
      const t = m.mantleT;
      mantlePitch = -Math.sin(Math.PI * Math.min(1, t * 1.4)) * 5.5 * DEG;
      mantleUp = -Math.sin(Math.PI * t) * 0.075;
      mantleRoll = Math.sin(Math.PI * t) * 1.8 * DEG;
    }

    // -- shake: trauma^2 amplitude over smooth noise, so it ramps and fades.
    // Clamped both ends: trauma is a normalised 0..1 quantity and the shake
    // amplitude is quadratic in it, so any drift above 1 is instantly violent.
    this.trauma = THREE.MathUtils.clamp(this.trauma - this.traumaDecay * dt, 0, 1);
    this.noiseT += dt * this.traumaFreq;
    let shakePitch = 0;
    let shakeYaw = 0;
    let shakeRoll = 0;
    let shakeUp = 0;
    let shakeSide = 0;
    if (this.trauma > 0.0005) {
      const s = this.trauma * this.trauma;
      shakePitch = fbm(this.noiseT, 11) * RIG.shakeRotation * s;
      shakeYaw = fbm(this.noiseT, 37) * RIG.shakeRotation * s;
      shakeRoll = fbm(this.noiseT * 0.8, 73) * RIG.shakeRotation * 1.4 * s;
      shakeUp = fbm(this.noiseT * 1.13, 101) * RIG.shakePosition * s;
      shakeSide = fbm(this.noiseT * 1.07, 149) * RIG.shakePosition * s;
    }

    // -- compose.
    this.oPitch = this.kickPitch + this.punchPitch + bobPitch + swayPitch + breathPitch
      + shakePitch + slidePitch + mantlePitch + this.landOffset * 0.9;
    this.oYaw = this.kickYaw + this.punchYaw + breathYaw + shakeYaw;
    this.oRoll = bobRoll + swayRoll + strafeRoll + slideRoll + shakeRoll + mantleRoll
      - this.lean * RIG.leanAngle;
    this.oUp = bobUp + this.landOffset + slideDip + mantleUp + shakeUp;
    this.oSide = bobLat + swayLat + shakeSide + this.lean * RIG.leanOffset;
    // Kick pushes the camera back along its own forward axis — a couple of
    // centimetres, but it is what sells the impulse as coming from the weapon.
    this.oFwd = -Math.abs(this.kickPitch) * RIG.recoilPushback;

    // -- field of view.
    const sprintTarget = m.sprinting && !m.aiming ? 1 : 0;
    const rate = sprintTarget > this.sprintT ? 1 / RIG.fovSprintIn : 1 / RIG.fovSprintOut;
    this.sprintT = moveTowards(this.sprintT, sprintTarget, rate * dt);
    const sprintPush = smoothstep(this.sprintT) * RIG.fovSprintPush * (m.tacSprinting ? 1.2 : 1);
    const slidePush = m.sliding ? RIG.fovSlidePush * slideEnv : 0;
    const landPush = this.landPunch * RIG.fovLandPunch;

    this.adsBlend = moveTowards(this.adsBlend, this.adsScale, this.adsRate * dt);
    const base = (RIG.fovBase + sprintPush + slidePush - landPush) * this.adsBlend;
    this.worldFov = THREE.MathUtils.clamp(base, 20, 120);
    const adsDelta = (this.adsBlend - 1) * RIG.fovBase * RIG.fovViewAdsFollow;
    this.viewFov = THREE.MathUtils.clamp(
      RIG.fovView + (sprintPush + slidePush - landPush) * RIG.fovViewFollow + adsDelta,
      30, 85,
    );
  }

  /**
   * Weapon system reports its ADS state through the 'weapon.ads' event.
   * `time` is the weapon's ADS duration so the FOV pull matches the gun raise.
   */
  setAdsFovScale(scale: number, time = 0.18): void {
    this.adsScale = THREE.MathUtils.clamp(scale, 0.2, 1.4);
    this.adsRate = Math.max(0.6, Math.abs(this.adsScale - this.adsBlend) / Math.max(0.04, time));
  }

  /** Writes the composed transform onto both cameras. Cheap; safe to repeat. */
  apply(ctx: GameContext): void {
    const cam = ctx.camera;
    const view = ctx.viewCamera;

    if (this.debugFrozen) {
      cam.position.copy(this.debugPos);
      cam.rotation.set(this.pitch, this.yaw, 0);
      this.writeFov(cam, view, this.debugFov, RIG.fovView);
      view.position.copy(cam.position);
      view.rotation.copy(cam.rotation);
      cam.updateMatrixWorld();
      view.updateMatrixWorld();
      return;
    }

    const pitch = THREE.MathUtils.clamp(this.pitch + this.oPitch, -89.5 * DEG, 89.5 * DEG);
    const yaw = this.yaw + this.oYaw;

    const sy = Math.sin(yaw);
    const cy = Math.cos(yaw);
    // Matches three.js' rotation.y basis: +X is (cos, 0, -sin), -Z is (-sin, 0, -cos).
    _right.set(cy, 0, -sy);
    _fwd.set(-sy, 0, -cy);

    cam.position.set(
      this.baseX + _right.x * this.oSide + _fwd.x * this.oFwd,
      this.eyeY + this.oUp,
      this.baseZ + _right.z * this.oSide + _fwd.z * this.oFwd,
    );
    cam.rotation.set(pitch, yaw, this.oRoll);

    this.writeFov(cam, view, this.worldFov, this.viewFov);

    view.position.copy(cam.position);
    view.rotation.copy(cam.rotation);
    cam.updateMatrixWorld();
    view.updateMatrixWorld();
  }

  setEyeAnchor(x: number, z: number): void {
    this.baseX = x;
    this.baseZ = z;
  }

  /** Current world-space eye height after stance smoothing. */
  get currentEyeY(): number {
    return this.eyeY;
  }

  private writeFov(
    cam: THREE.PerspectiveCamera, view: THREE.PerspectiveCamera, world: number, viewFov: number,
  ): void {
    if (this.appliedFov !== world) {
      cam.fov = world;
      cam.updateProjectionMatrix();
      this.appliedFov = world;
    }
    if (this.appliedViewFov !== viewFov) {
      view.fov = viewFov;
      view.updateProjectionMatrix();
      this.appliedViewFov = viewFov;
    }
  }

  // -- lean occlusion -------------------------------------------------------

  /**
   * Returns 0..1: how far the lean is allowed to go before the head would be
   * inside geometry. Probes a few fractions rather than binary-searching, which
   * is both cheaper and gives the lean a natural feel of stopping short.
   */
  private probeLean(ctx: GameContext, dir: number): number {
    const y = this.eyeY - RIG.leanProbeHeight * 0.5;
    const sy = Math.sin(this.yaw);
    const cy = Math.cos(this.yaw);
    for (let i = 0; i < 3; i++) {
      const f = 1 - i * 0.34;
      const off = dir * f * RIG.leanOffset;
      _probe.set(this.baseX + cy * off, y, this.baseZ - sy * off);
      if (!ctx.physics.overlapCapsule(_probe, RIG.leanProbeRadius, RIG.leanProbeHeight, RayMask.Solid)) {
        return f;
      }
    }
    return 0;
  }

  // -- debug ---------------------------------------------------------------

  setDebugPose(pose: DebugPose): void {
    const p = pose.pos;
    if (Array.isArray(p)) this.debugPos.set(p[0] ?? 0, p[1] ?? 1.65, p[2] ?? 0);
    else if (p) this.debugPos.set(p.x, p.y, p.z);
    if (typeof pose.yaw === 'number') this.yaw = pose.yaw;
    if (typeof pose.pitch === 'number') this.pitch = pose.pitch;
    this.debugFov = typeof pose.fov === 'number' ? pose.fov : RIG.fovBase;
    this.debugFrozen = true;
    this.zeroSecondary();
    this.eyeY = this.debugPos.y;
    this.baseX = this.debugPos.x;
    this.baseZ = this.debugPos.z;
  }

  clearDebugPose(): void {
    this.debugFrozen = false;
    this.eyeInit = false;
  }

  private zeroSecondary(): void {
    this.kickPitch = 0; this.kickYaw = 0;
    this.kickPitchVel = 0; this.kickYawVel = 0;
    this.kickTargetPitch = 0; this.kickTargetYaw = 0;
    this.landOffset = 0; this.landVel = 0; this.landPunch = 0;
    this.lean = 0; this.leanVel = 0;
    this.trauma = 0;
    this.punchPitch = 0; this.punchYaw = 0;
    this.sprintT = 0;
    this.oPitch = 0; this.oYaw = 0; this.oRoll = 0;
    this.oUp = 0; this.oSide = 0; this.oFwd = 0;
    this.lastMotionSpeed = 0;
  }
}
