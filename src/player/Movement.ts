/**
 * Movement.ts — the capsule controller.
 *
 * Pure simulation: it owns a position/velocity pair and talks to the world only
 * through `PhysicsWorld`. No three.js scene objects, no events, no camera. The
 * PlayerSystem drives it from `fixedUpdate` and reads the flags it raises.
 *
 * Conventions (matching PhysicsSystem):
 *   - `position` is the capsule BASE — the point between the feet.
 *   - `height` is the full stance height, `radius` the cylinder radius.
 *
 * The feel model is Quake-lineage (separate friction and acceleration terms,
 * projected sliding, air-strafe control) retuned to Call of Duty numbers: much
 * higher ground acceleration so the character starts and stops inside ~120ms,
 * a hard speed ceiling per gait rather than a soft one, and a step offset large
 * enough that stairs never interrupt a sprint.
 */
import * as THREE from 'three';
import {
  RayMask,
  UNITS,
  type PhysicsWorld,
  type SurfaceKind,
} from '../core/Contracts';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Stance = 'stand' | 'crouch' | 'prone';

export type Gait =
  | 'idle' | 'walk' | 'sprint' | 'tacsprint'
  | 'crouch' | 'prone' | 'air' | 'slide' | 'mantle';

export interface MoveCommand {
  /** Strafe intent, -1..1 (right positive). */
  moveX: number;
  /** Forward intent, -1..1 (forward positive). */
  moveY: number;
  /** View yaw in radians; movement is relative to it. */
  yaw: number;
  jumpPressed: boolean;
  jumpHeld: boolean;
  crouchPressed: boolean;
  crouchHeld: boolean;
  sprintPressed: boolean;
  sprintHeld: boolean;
  /** Tactical (slow) walk modifier. */
  walkHeld: boolean;
  aiming: boolean;
  /** False freezes the simulation (death, debug pose, menus). */
  enabled: boolean;
}

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

export const MOVE = {
  radius: UNITS.playerRadius,

  standHeight: UNITS.playerHeight,
  crouchHeight: UNITS.playerCrouchHeight,
  proneHeight: 0.62,

  standEye: UNITS.eyeOffset,
  crouchEye: UNITS.crouchEyeOffset,
  proneEye: 0.42,
  slideEye: 0.80,

  speedWalk: 3.2,
  speedSprint: 6.1,
  speedTacSprint: 7.3,
  speedCrouch: 1.9,
  speedProne: 0.9,
  /** Held-Alt tactical walk. */
  speedTactical: 1.75,
  /** Movement while aiming down sights. */
  speedAds: 2.45,

  groundAccel: 55,
  /** Turning around is quicker than getting going. */
  groundDecel: 82,
  /** Bleeds the sideways component when you change direction mid-stride. */
  friction: 10,
  /** Applied when the stick is centred — stopping beats starting. */
  frictionStop: 15,
  /** Bleeds speed above the current gait's ceiling (sprint -> walk, slide exit). */
  frictionOverspeed: 13,
  /** Below this speed friction uses a constant drop, so you actually stop. */
  frictionFloor: 1.35,

  airAccel: 12,
  /**
   * Classic air-strafe: the acceleration target along the wish direction is
   * clamped to this, so pushing straight forward in the air does nothing once
   * you are moving, while turning the view across the wish direction keeps
   * adding a perpendicular component. That asymmetry is the whole trick.
   */
  airWishCap: 1.6,
  /** Absolute lid on airborne horizontal speed, as a multiple of gait speed. */
  airSpeedCeiling: 1.3,

  gravity: UNITS.gravity,
  terminalVelocity: 55,
  /**
   * Take-off speed for a 0.62m apex. Semi-implicit Euler undershoots the
   * continuous solution by v*dt/2 (about 2cm at 120Hz), so this solves
   * v^2/2g - v*dt/2 = 0.62 rather than using the textbook sqrt(2gh) and
   * shipping a jump that is measurably short.
   */
  jumpVelocity: (() => {
    const g = UNITS.gravity;
    const half = (g * (1 / 120)) / 2;
    return Math.sqrt(2 * g * 0.62 + half * half) + half;
  })(),
  coyoteTime: 0.09,
  jumpBuffer: 0.12,
  /** Blocks the instant re-jump that otherwise happens on the landing frame. */
  jumpCooldown: 0.16,

  stepOffset: 0.35,
  /** Seconds the ground snap is held off the lower tread after a step lift. */
  stepLiftHold: 0.24,
  /** cos(46 degrees) — anything steeper is not standable. */
  walkableCos: Math.cos(46 * THREE.MathUtils.DEG2RAD),
  /** How fast the visual eye catches up after a step up/down, per second. */
  stepSmoothRate: 12,

  slideSpeed: 8.5,
  slideDuration: 0.8,
  /** Exponential drag during a slide; 8.5 * e^(-1.8*0.8) ~= 2.0 m/s at the end. */
  slideDrag: 1.8,
  slideSteer: 5.5,
  slideMinEntrySpeed: 4.4,
  slideCooldown: 0.55,
  /** Downhill acceleration that keeps a slide alive on a ramp. */
  slideSlopeAccel: 11,

  mantleMinRise: 0.34,
  mantleMaxRise: 1.72,
  mantleReach: 0.62,
  mantleTimeShort: 0.45,
  mantleTimeTall: 0.62,
  /** Forward speed handed back to the player as the mantle releases. */
  mantleExitSpeed: 1.6,

  tacSprintDuration: 4.0,
  tacSprintRecharge: 3.0,

  /**
   * Metres of travel per full two-footfall stride cycle. Fitted to real gait
   * data: ~2.5m per cycle at a 3.2m/s jog rising to ~3.7m at a 6.1m/s sprint,
   * i.e. 2.6 and 3.3 footfalls per second. Getting this wrong is immediately
   * audible — too short and the player sounds like they are scurrying.
   */
  strideBase: 1.09,
  stridePerSpeed: 0.428,
  strideCrouchScale: 1.0,
  strideProneScale: 1.3,

  /** Impact speed at which fall damage starts / becomes lethal, m/s. */
  fallDamageSpeed: 11,
  fallLethalSpeed: 22,
} as const;

// ---------------------------------------------------------------------------
// Module scratch — the fixed step must not allocate
// ---------------------------------------------------------------------------

const _wish = new THREE.Vector3();
const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _delta = new THREE.Vector3();
const _remain = new THREE.Vector3();
const _from = new THREE.Vector3();
const _to = new THREE.Vector3();
const _probe = new THREE.Vector3();
const _probeB = new THREE.Vector3();
const _flat = new THREE.Vector3();
const _down = new THREE.Vector3(0, -1, 0);

function easeOutCubic(t: number): number {
  const u = 1 - t;
  return 1 - u * u * u;
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) * 0.5;
}

// ---------------------------------------------------------------------------

export class MovementController {
  /** Capsule base, simulation-authoritative. */
  readonly position = new THREE.Vector3();
  /** Base at the start of the current fixed step, for render interpolation. */
  readonly prevPosition = new THREE.Vector3();
  readonly velocity = new THREE.Vector3();
  readonly groundNormal = new THREE.Vector3(0, 1, 0);

  groundSurface: SurfaceKind = 'concrete';
  grounded = false;
  /** True when standing on something too steep to hold. */
  onSteep = false;

  stance: Stance = 'stand';
  gait: Gait = 'idle';
  sprinting = false;
  tacSprinting = false;

  sliding = false;
  /** 0..1 progress through the slide. */
  slideT = 0;
  /** -1..1 steering bias during the slide, drives the camera roll. */
  slideLateral = 0;

  mantleActive = false;
  /** 0..1 progress through the mantle. */
  mantleT = 0;

  /**
   * Vertical offset between the capsule and where the eye should render. Steps
   * add to it instantly and it decays exponentially, so stairs do not strobe
   * the camera. Positive means the capsule went up and the eye is lagging.
   */
  stepSmooth = 0;
  /** 0..1 through the two-footfall stride cycle; 0 and 0.5 are footfalls. */
  stridePhase = 0;

  /** Raised for exactly one fixed step; the owner consumes and clears nothing. */
  footstepPending = false;
  /** Impact speed in m/s on the step the player landed, else -1. */
  landPending = -1;
  landSurface: SurfaceKind = 'concrete';
  /** Set on the step a mantle begins. */
  mantleStarted = false;
  /** Set on the step a slide begins. */
  slideStarted = false;

  /** Speed the current gait is asking for, m/s. */
  targetSpeed = 0;
  /** 0..1 magnitude of the movement stick. */
  moveIntent = 0;
  /** -1..1 strafe component of the current intent, for camera roll. */
  strafeIntent = 0;

  private elapsed = 0;
  private coyote = 0;
  private jumpBufferTimer = 0;
  private jumpCooldownTimer = 0;
  private slideTimer = 0;
  private slideCooldownTimer = 0;
  private proneLatched = false;
  private lastCrouchPressAt = -10;
  private tacSprintTimer = 0;
  private tacSprintCooldown = 0;
  private mantleTimer = 0;
  private mantleDuration: number = MOVE.mantleTimeShort;
  private mantleProbeCooldown = 0;
  private readonly mantleStart = new THREE.Vector3();
  private readonly mantleTarget = new THREE.Vector3();
  private readonly mantleDir = new THREE.Vector3(0, 0, -1);
  private mantleToCrouch = false;
  private strideAccum = 0;
  private wasGrounded = true;
  private blocked = false;
  private blockedVX = 0;
  private blockedVZ = 0;
  private liftHold = 0;
  private liftFloor = -Infinity;

  // -- queries -------------------------------------------------------------

  get height(): number {
    if (this.stance === 'prone') return MOVE.proneHeight;
    if (this.stance === 'crouch') return MOVE.crouchHeight;
    return MOVE.standHeight;
  }

  /** Eye height above the capsule base for the current stance. */
  get eyeHeight(): number {
    if (this.sliding) return MOVE.slideEye;
    if (this.stance === 'prone') return MOVE.proneEye;
    if (this.stance === 'crouch') return MOVE.crouchEye;
    return MOVE.standEye;
  }

  get horizontalSpeed(): number {
    return Math.hypot(this.velocity.x, this.velocity.z);
  }

  teleport(x: number, y: number, z: number): void {
    this.position.set(x, y, z);
    this.prevPosition.copy(this.position);
    this.velocity.set(0, 0, 0);
    this.stepSmooth = 0;
    this.grounded = false;
    this.onSteep = false;
    this.sliding = false;
    this.slideTimer = 0;
    this.slideT = 0;
    this.mantleActive = false;
    this.mantleTimer = 0;
    this.mantleT = 0;
    this.proneLatched = false;
    this.blocked = false;
    this.liftHold = 0;
    this.liftFloor = -Infinity;
    this.stance = 'stand';
    this.stridePhase = 0;
    this.strideAccum = 0;
    this.velocity.set(0, 0, 0);
  }

  // -- the step ------------------------------------------------------------

  step(cmd: MoveCommand, phys: PhysicsWorld, dt: number): void {
    this.prevPosition.copy(this.position);
    this.footstepPending = false;
    this.landPending = -1;
    this.mantleStarted = false;
    this.slideStarted = false;
    this.elapsed += dt;

    if (!cmd.enabled) {
      this.velocity.set(0, 0, 0);
      this.gait = 'idle';
      this.moveIntent = 0;
      this.strafeIntent = 0;
      this.decayStepSmooth(dt);
      return;
    }

    if (this.jumpCooldownTimer > 0) this.jumpCooldownTimer -= dt;
    if (this.slideCooldownTimer > 0) this.slideCooldownTimer -= dt;
    if (this.mantleProbeCooldown > 0) this.mantleProbeCooldown -= dt;
    this.jumpBufferTimer = cmd.jumpPressed ? MOVE.jumpBuffer : Math.max(0, this.jumpBufferTimer - dt);

    if (this.mantleActive) {
      this.advanceMantle(phys, dt);
      this.decayStepSmooth(dt);
      return;
    }

    // Movement basis from the view yaw. Camera-forward is -Z at yaw 0.
    const sy = Math.sin(cmd.yaw);
    const cy = Math.cos(cmd.yaw);
    // three.js: an object with rotation.y = yaw has local -Z at (-sin, 0, -cos)
    // and local +X at (cos, 0, -sin). Getting the sign of right wrong silently
    // swaps A and D, which is the kind of bug you feel before you can name it.
    _fwd.set(-sy, 0, -cy);
    _right.set(cy, 0, -sy);

    let ix = cmd.moveX;
    let iy = cmd.moveY;
    let intent = Math.hypot(ix, iy);
    if (intent > 1) {
      ix /= intent;
      iy /= intent;
      intent = 1;
    }
    this.moveIntent = intent;
    this.strafeIntent = ix;

    _wish.set(0, 0, 0).addScaledVector(_right, ix).addScaledVector(_fwd, iy);
    const wishLen = _wish.length();
    if (wishLen > 1e-5) _wish.multiplyScalar(1 / wishLen);

    // Forwardness gates sprinting; 0.6 is roughly a 53 degree cone.
    const forwardness = intent > 1e-4 ? iy / intent : 0;

    this.updateStance(cmd, phys, forwardness, dt);
    this.updateSprint(cmd, forwardness, dt);
    this.updateSlide(cmd, phys, dt);

    // Mantle is tested before the jump so that jumping into a waist-high wall
    // vaults it instead of bouncing off — the auto-mantle every modern CoD has.
    // Falling with forward intent also grabs, which is the ledge catch.
    if (this.mantleProbeCooldown <= 0) {
      const wantsMantle =
        this.jumpBufferTimer > 0 ||
        (!this.grounded && this.velocity.y < 0.5 && intent > 0.5);
      if (wantsMantle && this.tryMantle(phys, _wish, wishLen > 0.1)) {
        this.decayStepSmooth(dt);
        return;
      }
    }

    this.tryJump(cmd, phys);

    const speed = this.resolveTargetSpeed(cmd, intent, forwardness);
    this.targetSpeed = speed;

    if (this.sliding) {
      this.slideMotion(_wish, wishLen, dt);
    } else if (this.grounded) {
      this.groundMove(_wish, wishLen, speed, intent, dt);
    } else {
      this.airMove(_wish, wishLen, speed, dt);
    }

    // Gravity. On a too-steep contact the surface normal has already clipped the
    // downhill component out of the velocity, so this is what slides you off.
    this.velocity.y -= MOVE.gravity * dt;
    if (this.velocity.y < -MOVE.terminalVelocity) this.velocity.y = -MOVE.terminalVelocity;

    const fallSpeed = -this.velocity.y;

    // Consume last step's block report before slideMove raises a new one.
    if (this.liftHold > 0) this.liftHold -= dt;
    if (this.blocked && !this.sliding && !this.mantleActive) this.tryStepLift(phys);
    this.blocked = false;

    _delta.copy(this.velocity).multiplyScalar(dt);
    // On a walkable surface the frame's motion is rotated into the ground plane
    // before it is swept. Letting collide-and-slide discover the slope instead
    // converts forward speed into vertical speed on contact, which unsticks the
    // ground check and makes the player bunny-hop their way up every ramp.
    if (this.grounded && this.groundNormal.y >= MOVE.walkableCos && this.groundNormal.y > 1e-3) {
      const n = this.groundNormal;
      // Scaling the horizontal part by cos(slope) first keeps the along-surface
      // distance equal to the flat-ground distance, so a gradient costs you
      // ground speed instead of handing you a free 30% on the climb.
      _delta.x *= n.y;
      _delta.z *= n.y;
      _delta.y = -(_delta.x * n.x + _delta.z * n.z) / n.y;
    }
    this.slideMove(phys, _delta);
    this.groundCheck(phys);

    if (this.grounded && !this.wasGrounded) {
      // Impact strength is the vertical speed at the moment of contact.
      this.landPending = Math.max(0, fallSpeed);
      this.landSurface = this.groundSurface;
      this.stridePhase = 0.08;
      this.strideAccum = 0.08;
      // Deliberately no landing cooldown here: it would outlive the jump buffer
      // and eat the very input buffering is there to preserve.
    }
    this.wasGrounded = this.grounded;

    this.updateStride(dt);
    this.updateGait();
    this.decayStepSmooth(dt);

    if (this.grounded) this.coyote = MOVE.coyoteTime;
    else this.coyote = Math.max(0, this.coyote - dt);
  }

  // -- stance --------------------------------------------------------------

  private canFit(phys: PhysicsWorld, height: number): boolean {
    // A hair of extra height so a stand-up never ends flush against a ceiling.
    return !phys.overlapCapsule(this.position, MOVE.radius, height + 0.02, RayMask.Solid);
  }

  private updateStance(cmd: MoveCommand, phys: PhysicsWorld, forwardness: number, _dt: number): void {
    if (cmd.crouchPressed) {
      const dbl = this.elapsed - this.lastCrouchPressAt < 0.3;
      this.lastCrouchPressAt = this.elapsed;
      if (this.proneLatched) {
        this.proneLatched = false;
      } else if (dbl && this.grounded && !this.sliding) {
        this.proneLatched = true;
      }
    }

    // Sprinting overrides a held crouch — you stand up to run, like every CoD.
    const wantsSprint = cmd.sprintHeld && this.moveIntent > 0.5 && forwardness > 0.6 && !cmd.aiming;
    if (wantsSprint && this.proneLatched) this.proneLatched = false;

    let want: Stance;
    if (this.proneLatched) want = 'prone';
    else if (this.sliding || (cmd.crouchHeld && !wantsSprint)) want = 'crouch';
    else want = 'stand';

    if (want === this.stance) return;

    const targetHeight = want === 'prone'
      ? MOVE.proneHeight
      : want === 'crouch' ? MOVE.crouchHeight : MOVE.standHeight;

    // Shrinking is always legal; growing needs headroom.
    if (targetHeight <= this.height || this.canFit(phys, targetHeight)) {
      this.stance = want;
    } else if (want === 'stand' && this.stance === 'prone' && this.canFit(phys, MOVE.crouchHeight)) {
      this.stance = 'crouch';
      this.proneLatched = false;
    }
  }

  // -- sprint --------------------------------------------------------------

  private updateSprint(cmd: MoveCommand, forwardness: number, dt: number): void {
    const eligible =
      cmd.sprintHeld &&
      this.moveIntent > 0.5 &&
      forwardness > 0.6 &&
      !cmd.aiming &&
      !this.sliding &&
      this.stance === 'stand';

    if (!eligible) {
      this.sprinting = false;
      this.tacSprinting = false;
      this.tacSprintCooldown += dt;
      if (this.tacSprintCooldown >= MOVE.tacSprintRecharge) this.tacSprintTimer = 0;
      return;
    }

    // A fresh press out of a rested state opens the tactical-sprint window.
    if (!this.sprinting && cmd.sprintPressed && this.tacSprintTimer <= 0) {
      this.tacSprintTimer = MOVE.tacSprintDuration;
    }
    this.sprinting = true;
    this.tacSprintCooldown = 0;

    if (this.tacSprintTimer > 0) {
      this.tacSprintTimer -= dt;
      this.tacSprinting = this.tacSprintTimer > 0;
      if (!this.tacSprinting) this.tacSprintTimer = -1; // spent until recharge
    } else {
      this.tacSprinting = false;
    }
  }

  private resolveTargetSpeed(cmd: MoveCommand, intent: number, forwardness: number): number {
    let base: number;
    if (this.stance === 'prone') base = MOVE.speedProne;
    else if (this.stance === 'crouch') base = MOVE.speedCrouch;
    else if (this.tacSprinting) base = MOVE.speedTacSprint;
    else if (this.sprinting) base = MOVE.speedSprint;
    else if (cmd.aiming) base = MOVE.speedAds;
    else if (cmd.walkHeld) base = MOVE.speedTactical;
    else base = MOVE.speedWalk;

    // Directional penalty: backpedalling and strafing are slower than advancing.
    if (!this.sprinting && intent > 1e-4) {
      const side = Math.sqrt(Math.max(0, 1 - forwardness * forwardness));
      base *= 1 - 0.2 * Math.max(0, -forwardness) - 0.07 * side;
    }
    return base * intent;
  }

  // -- acceleration model ---------------------------------------------------

  /**
   * Ground movement. Friction is applied to the component of velocity ACROSS
   * the wish direction, never along it — the naive Quake form (blanket friction
   * plus a constant acceleration cap) has a fixed point at accel/friction and
   * silently clamps top speed to 5.5 m/s no matter what gait asks for. Splitting
   * the axes gives exact per-gait top speeds and still bleeds sideways drift
   * when the player changes direction mid-stride, which is what makes a hard
   * strafe swap feel like it has a body behind it.
   */
  private groundMove(
    wish: THREE.Vector3, wishLen: number, targetSpeed: number, intent: number, dt: number,
  ): void {
    const vx = this.velocity.x;
    const vz = this.velocity.z;

    if (wishLen < 1e-5 || intent < 0.08 || targetSpeed <= 0) {
      const speed = Math.hypot(vx, vz);
      if (speed < 1e-4) {
        this.velocity.x = 0;
        this.velocity.z = 0;
        return;
      }
      // Constant drop below the floor speed: exponential decay alone never
      // actually reaches zero and the player creeps for half a second.
      const control = speed < MOVE.frictionFloor ? MOVE.frictionFloor : speed;
      const scale = Math.max(0, speed - control * MOVE.frictionStop * dt) / speed;
      this.velocity.x = vx * scale;
      this.velocity.z = vz * scale;
      return;
    }

    let along = vx * wish.x + vz * wish.z;
    let px = vx - wish.x * along;
    let pz = vz - wish.z * along;

    const pmag = Math.hypot(px, pz);
    if (pmag > 1e-5) {
      const control = pmag < MOVE.frictionFloor ? MOVE.frictionFloor : pmag;
      const scale = Math.max(0, pmag - control * MOVE.friction * dt) / pmag;
      px *= scale;
      pz *= scale;
    }

    if (along < targetSpeed) {
      // Reversing decelerates harder than starting from rest accelerates.
      const rate = along < 0 ? MOVE.groundDecel : MOVE.groundAccel;
      along = Math.min(targetSpeed, along + rate * dt);
    } else {
      const control = along < MOVE.frictionFloor ? MOVE.frictionFloor : along;
      along = Math.max(targetSpeed, along - control * MOVE.frictionOverspeed * dt);
    }

    this.velocity.x = wish.x * along + px;
    this.velocity.z = wish.z * along + pz;
  }

  private airMove(wish: THREE.Vector3, wishLen: number, targetSpeed: number, dt: number): void {
    if (wishLen < 1e-5) return;
    // The clamped wish speed is what produces air-strafing: forward input adds
    // nothing once you exceed the cap, but input across your velocity always
    // can, because the projection onto that axis is still near zero.
    const capped = Math.min(targetSpeed, MOVE.airWishCap);
    const current = this.velocity.x * wish.x + this.velocity.z * wish.z;
    const add = capped - current;
    if (add > 0) {
      const gain = Math.min(MOVE.airAccel * dt, add);
      this.velocity.x += wish.x * gain;
      this.velocity.z += wish.z * gain;
    }

    const ceiling = Math.max(MOVE.speedWalk, this.targetSpeed, MOVE.speedSprint) * MOVE.airSpeedCeiling;
    const h = Math.hypot(this.velocity.x, this.velocity.z);
    if (h > ceiling) {
      const s = ceiling / h;
      this.velocity.x *= s;
      this.velocity.z *= s;
    }
  }

  // -- jump ----------------------------------------------------------------

  private tryJump(cmd: MoveCommand, phys: PhysicsWorld): boolean {
    if (this.jumpBufferTimer <= 0) return false;
    if (this.jumpCooldownTimer > 0) return false;
    if (this.coyote <= 0 && !this.grounded) return false;
    if (this.onSteep) return false;

    if (this.stance === 'prone') {
      // First jump press out of prone just gets you up.
      this.proneLatched = false;
      this.jumpBufferTimer = 0;
      return false;
    }
    if (this.stance === 'crouch' && !this.sliding) {
      if (!this.canFit(phys, MOVE.standHeight)) return false;
      this.stance = 'stand';
    }
    if (this.sliding) {
      // Slide cancel: keep the speed you built, jump out of it.
      this.endSlide(0.25);
      if (!this.canFit(phys, MOVE.standHeight)) {
        this.stance = 'crouch';
      } else {
        this.stance = 'stand';
      }
    }

    this.velocity.y = MOVE.jumpVelocity;
    this.grounded = false;
    this.liftHold = 0;
    this.coyote = 0;
    this.jumpBufferTimer = 0;
    this.jumpCooldownTimer = MOVE.jumpCooldown;
    this.wasGrounded = false;
    return true;
  }

  // -- slide ---------------------------------------------------------------

  private updateSlide(cmd: MoveCommand, phys: PhysicsWorld, dt: number): void {
    if (this.sliding) {
      this.slideTimer -= dt;
      this.slideT = 1 - Math.max(0, this.slideTimer) / MOVE.slideDuration;
      const stillFast = this.horizontalSpeed > MOVE.speedCrouch + 0.15;
      if (this.slideTimer <= 0 || !stillFast || !this.grounded || !cmd.crouchHeld) {
        this.endSlide(MOVE.slideCooldown);
      }
      return;
    }

    const canStart =
      cmd.crouchPressed &&
      this.grounded &&
      this.slideCooldownTimer <= 0 &&
      this.stance !== 'prone' &&
      this.horizontalSpeed >= MOVE.slideMinEntrySpeed &&
      (this.sprinting || this.horizontalSpeed >= MOVE.speedSprint * 0.85);

    if (!canStart) return;
    if (!phys.overlapCapsule(this.position, MOVE.radius, MOVE.crouchHeight, RayMask.Solid)) {
      this.beginSlide();
    }
  }

  private beginSlide(): void {
    const h = this.horizontalSpeed;
    if (h > 1e-4) {
      const s = MOVE.slideSpeed / h;
      this.velocity.x *= s;
      this.velocity.z *= s;
    }
    this.sliding = true;
    this.slideStarted = true;
    this.slideTimer = MOVE.slideDuration;
    this.slideT = 0;
    this.slideLateral = 0;
    this.stance = 'crouch';
    this.sprinting = false;
    this.tacSprinting = false;
    this.proneLatched = false;
  }

  private endSlide(cooldown: number): void {
    this.sliding = false;
    this.slideT = 0;
    this.slideLateral = 0;
    this.slideTimer = 0;
    this.slideCooldownTimer = cooldown;
  }

  private slideMotion(wish: THREE.Vector3, wishLen: number, dt: number): void {
    // Exponential drag, plus downhill gravity along the contact plane so a slide
    // down a ramp carries and a slide up one dies early.
    const decay = Math.exp(-MOVE.slideDrag * dt);
    this.velocity.x *= decay;
    this.velocity.z *= decay;

    if (this.grounded && this.groundNormal.y < 0.9995) {
      _flat.set(this.groundNormal.x, 0, this.groundNormal.z);
      const l = _flat.length();
      if (l > 1e-4) {
        _flat.multiplyScalar(1 / l);
        const grade = Math.sqrt(Math.max(0, 1 - this.groundNormal.y * this.groundNormal.y));
        this.velocity.x += _flat.x * MOVE.slideSlopeAccel * grade * dt;
        this.velocity.z += _flat.z * MOVE.slideSlopeAccel * grade * dt;
      }
    }

    if (wishLen > 1e-5) {
      // Only the component across the slide direction steers; you cannot
      // accelerate a slide by holding forward.
      const h = Math.hypot(this.velocity.x, this.velocity.z);
      if (h > 1e-4) {
        const dx = this.velocity.x / h;
        const dz = this.velocity.z / h;
        const lateral = wish.x * -dz + wish.z * dx;
        this.slideLateral += (lateral - this.slideLateral) * Math.min(1, dt * 8);
        const steer = MOVE.slideSteer * dt * lateral;
        this.velocity.x += -dz * steer;
        this.velocity.z += dx * steer;
        // Re-normalise so steering turns the slide instead of adding energy.
        const h2 = Math.hypot(this.velocity.x, this.velocity.z);
        if (h2 > 1e-4) {
          this.velocity.x *= h / h2;
          this.velocity.z *= h / h2;
        }
      }
    } else {
      this.slideLateral *= Math.max(0, 1 - dt * 8);
    }
  }

  // -- mantle / vault -------------------------------------------------------

  private tryMantle(phys: PhysicsWorld, wish: THREE.Vector3, hasWish: boolean): boolean {
    this.mantleProbeCooldown = 0.12;
    if (this.stance === 'prone' || this.sliding) return false;

    // Direction of travel: movement intent if any, otherwise current velocity.
    let dx: number;
    let dz: number;
    if (hasWish) {
      dx = wish.x;
      dz = wish.z;
    } else {
      const h = Math.hypot(this.velocity.x, this.velocity.z);
      if (h < 0.8) return false;
      dx = this.velocity.x / h;
      dz = this.velocity.z / h;
    }

    const reach = MOVE.radius + MOVE.mantleReach;

    // Hip ray must find a near-vertical face; a walkable ramp is not a ledge.
    _probe.set(this.position.x, this.position.y + 0.52, this.position.z);
    _probeB.set(dx, 0, dz);
    const hip = phys.raycast(_probe, _probeB, reach, RayMask.Solid);
    if (!hip || Math.abs(hip.normal.y) > 0.45) return false;

    const contactDist = hip.distance;

    // Search for a walkable top surface just past the face.
    const landX = this.position.x + dx * (contactDist + MOVE.radius + 0.06);
    const landZ = this.position.z + dz * (contactDist + MOVE.radius + 0.06);
    _probe.set(landX, this.position.y + MOVE.mantleMaxRise + 0.45, landZ);
    const top = phys.raycast(_probe, _down, MOVE.mantleMaxRise + 0.45 - MOVE.mantleMinRise, RayMask.Solid);
    if (!top || top.normal.y < MOVE.walkableCos) return false;

    const rise = top.point.y - this.position.y;
    if (rise < MOVE.mantleMinRise || rise > MOVE.mantleMaxRise) return false;

    // Chest ray must be clear: something overhanging the ledge blocks the pull-up.
    _probe.set(this.position.x, this.position.y + Math.min(rise + 0.32, MOVE.standHeight - 0.1), this.position.z);
    _probeB.set(dx, 0, dz);
    const chest = phys.raycast(_probe, _probeB, contactDist + 0.02, RayMask.Solid);
    if (chest) return false;

    _to.set(landX, top.point.y + 0.015, landZ);
    let toCrouch = false;
    if (phys.overlapCapsule(_to, MOVE.radius, MOVE.standHeight, RayMask.Solid)) {
      if (phys.overlapCapsule(_to, MOVE.radius, MOVE.crouchHeight, RayMask.Solid)) return false;
      toCrouch = true;
    }

    this.mantleStart.copy(this.position);
    this.mantleTarget.copy(_to);
    this.mantleDir.set(dx, 0, dz);
    this.mantleToCrouch = toCrouch;
    this.mantleActive = true;
    this.mantleStarted = true;
    this.mantleTimer = 0;
    this.mantleT = 0;
    this.mantleDuration = THREE.MathUtils.lerp(
      MOVE.mantleTimeShort, MOVE.mantleTimeTall,
      THREE.MathUtils.clamp((rise - 0.6) / (MOVE.mantleMaxRise - 0.6), 0, 1),
    );
    this.velocity.set(0, 0, 0);
    this.jumpBufferTimer = 0;
    this.sliding = false;
    this.sprinting = false;
    this.tacSprinting = false;
    this.grounded = false;
    this.stance = toCrouch ? 'crouch' : 'stand';
    return true;
  }

  private advanceMantle(phys: PhysicsWorld, dt: number): void {
    this.mantleTimer += dt;
    const t = THREE.MathUtils.clamp(this.mantleTimer / this.mantleDuration, 0, 1);
    this.mantleT = t;
    this.gait = 'mantle';
    this.moveIntent = 0;
    this.strafeIntent = 0;

    // Vertical finishes first and the horizontal follows: you rise onto the
    // ledge and then step over it, which is what reads as weight.
    const vert = easeOutCubic(Math.min(1, t / 0.55));
    const horiz = easeInOutCubic(THREE.MathUtils.clamp((t - 0.16) / 0.84, 0, 1));
    // A few centimetres of heave that returns — the shoulders clearing the lip.
    const heave = Math.sin(Math.PI * Math.min(1, t / 0.62)) * 0.045;

    this.position.x = THREE.MathUtils.lerp(this.mantleStart.x, this.mantleTarget.x, horiz);
    this.position.z = THREE.MathUtils.lerp(this.mantleStart.z, this.mantleTarget.z, horiz);
    this.position.y = THREE.MathUtils.lerp(this.mantleStart.y, this.mantleTarget.y, vert) + heave;

    if (t >= 1) {
      this.mantleActive = false;
      this.mantleT = 0;
      this.position.copy(this.mantleTarget);
      this.velocity.set(this.mantleDir.x, 0, this.mantleDir.z).multiplyScalar(MOVE.mantleExitSpeed);
      this.jumpCooldownTimer = MOVE.jumpCooldown;
      this.wasGrounded = true;
      if (this.mantleToCrouch) this.stance = 'crouch';
      this.groundCheck(phys);
      this.stridePhase = 0.5;
      this.strideAccum = 0.5;
    }
  }

  // -- collision ------------------------------------------------------------

  /**
   * Pushes the capsule out of geometry it is genuinely inside. `overlapCapsule`
   * is the only reliable penetration test here — a sweep fraction of zero does
   * NOT mean penetration, because the sweep pulls its result back by a skin gap
   * that exceeds one 120Hz step of travel whenever a surface is close.
   */
  private escape(phys: PhysicsWorld, h: number): void {
    for (let i = 0; i < 4; i++) {
      if (!phys.overlapCapsule(this.position, MOVE.radius, h, RayMask.Solid)) return;
      _from.copy(this.position);
      _to.set(this.position.x, this.position.y + 0.001, this.position.z);
      const hit = phys.sweepCapsule(_from, _to, MOVE.radius, h, RayMask.Solid);
      if (!hit || hit.t > 0) return;
      this.position.addScaledVector(hit.normal, 0.02);
    }
  }

  /**
   * Stairs, kerbs, kickplates. Fired the step after a horizontal move was
   * stopped by something too steep to walk, which is what keeps it off ramps
   * (a walkable slope never blocks) and off flat ground (nothing blocks).
   *
   * The lift is instantaneous on the capsule and entirely absorbed by
   * `stepSmooth`, so the collision shape hops while the eye rises on a 12/s
   * exponential. Riding the capsule's bottom sphere over the edge instead —
   * which is what falls out of plain collide-and-slide — converts forward speed
   * into vertical speed, and the player visibly bounces up every stair.
   */
  private tryStepLift(phys: PhysicsWorld): void {
    const vx = this.blockedVX;
    const vz = this.blockedVZ;
    const speed = Math.hypot(vx, vz);
    if (speed < 0.35) return;
    const dx = vx / speed;
    const dz = vz / speed;

    const ahead = MOVE.radius + 0.10;
    _probe.set(
      this.position.x + dx * ahead,
      this.position.y + MOVE.stepOffset + 0.06,
      this.position.z + dz * ahead,
    );
    const hit = phys.raycast(_probe, _down, MOVE.stepOffset + 0.10, RayMask.Solid);
    if (!hit || hit.normal.y < MOVE.walkableCos) return;

    const rise = hit.point.y - this.position.y;
    if (rise <= 0.02 || rise > MOVE.stepOffset) return;

    // Overlap, not sweep: the capsule is already in marginal contact with the
    // edge it is trying to climb (that is why it blocked), so a sweep from here
    // reports a hit at t=0 and would refuse every legitimate step.
    const targetY = hit.point.y + 0.008;
    _to.set(this.position.x, targetY, this.position.z);
    if (phys.overlapCapsule(_to, MOVE.radius, this.height, RayMask.Solid)) return;

    this.stepSmooth += targetY - this.position.y;
    this.position.y = targetY;
    // Hold the ground snap off the lower tread until the capsule has travelled
    // far enough forward to actually be over the new one.
    this.liftHold = MOVE.stepLiftHold;
    this.liftFloor = targetY - 0.04;
    if (this.velocity.y < 0) this.velocity.y = 0;
    // Give back the speed the contact stole, so a staircase does not read as a
    // series of small stumbles.
    const cur = Math.hypot(this.velocity.x, this.velocity.z);
    if (speed > cur) {
      this.velocity.x = dx * speed;
      this.velocity.z = dz * speed;
    }
  }

  private slideMove(phys: PhysicsWorld, delta: THREE.Vector3): void {
    const h = this.height;
    this.escape(phys, h);
    _remain.copy(delta);
    this.onSteep = false;

    for (let planes = 0; planes < 4; planes++) {
      if (_remain.lengthSq() < 1e-12) return;

      _from.copy(this.position);
      _to.copy(this.position).add(_remain);
      const hit = phys.sweepCapsule(_from, _to, MOVE.radius, h, RayMask.Solid);
      if (!hit) {
        this.position.copy(_to);
        return;
      }

      if (hit.t > 0) {
        this.position.addScaledVector(_remain, hit.t);
        _remain.multiplyScalar(1 - hit.t);
      }

      const n = hit.normal;
      const walkable = n.y >= MOVE.walkableCos;
      if (!walkable) {
        if (n.y > 0.05) this.onSteep = true;
        // Remember what stopped us, and at what speed, for next step's lift.
        if (this.grounded && n.y > -0.3 && (_remain.x * n.x + _remain.z * n.z) < 0) {
          this.blocked = true;
          this.blockedVX = this.velocity.x;
          this.blockedVZ = this.velocity.z;
        }
      }

      // Which plane to clip against. A grounded player meeting something they
      // cannot walk on gets clipped against the VERTICAL projection of the
      // normal: the true normal of a stair nose points up and out, and
      // projecting onto it converts forward speed into a launch — the player
      // pogos up the staircase and never stays grounded long enough for the
      // step lift to fire.
      let cx = n.x, cy = n.y, cz = n.z;
      if (!walkable && this.grounded) {
        const hl = Math.hypot(n.x, n.z);
        if (hl > 1e-4) {
          cx = n.x / hl; cy = 0; cz = n.z / hl;
        }
      } else if (walkable && this.grounded) {
        // Brushing the surface we are already walking on: kill the downward
        // component only, the ground-plane projection already handled the rest.
        cx = 0; cy = 1; cz = 0;
      }

      const vd = this.velocity.x * cx + this.velocity.y * cy + this.velocity.z * cz;
      if (vd < 0) {
        this.velocity.x -= cx * vd;
        this.velocity.y -= cy * vd;
        this.velocity.z -= cz * vd;
      }
      const rd = _remain.x * cx + _remain.y * cy + _remain.z * cz;
      if (rd < 0) {
        _remain.x -= cx * rd;
        _remain.y -= cy * rd;
        _remain.z -= cz * rd;
      }

      // Creases: if the projection has reversed us relative to the original
      // motion, stop rather than sliding backwards out of the corner.
      if (_remain.dot(delta) <= 0) return;
    }
  }

  private groundCheck(phys: PhysicsWorld): void {
    const holding = this.liftHold > 0;
    if (this.velocity.y > 0.12 && !holding) {
      this.grounded = false;
      return;
    }

    // Reach further when already grounded so walking off a small lip or down a
    // stair keeps you glued instead of launching you into a fall state.
    const probe = this.grounded || this.wasGrounded ? MOVE.stepOffset + 0.03 : 0.05;
    const h = this.height;
    _from.copy(this.position);
    _to.set(this.position.x, this.position.y - probe, this.position.z);
    const hit = phys.sweepCapsule(_from, _to, MOVE.radius, h, RayMask.Solid);

    if (hit && hit.normal.y >= MOVE.walkableCos) {
      const dist = probe * hit.t;
      const newY = this.position.y - dist;
      // While a step lift is in flight, only accept ground at or above the
      // tread we climbed onto. Anything lower is the tread we just left.
      if (!holding || newY >= this.liftFloor) {
        if (dist > 0.001) {
          this.position.y = newY;
          // Dropping onto a lower step must not yank the eye down either.
          if (this.wasGrounded && dist > 0.015) this.stepSmooth -= dist;
        }
        this.liftHold = 0;
        this.grounded = true;
        this.onSteep = false;
        this.groundNormal.copy(hit.normal);
        this.groundSurface = hit.surface;
        if (this.velocity.y < 0) this.velocity.y = 0;
        const lim = MOVE.stepOffset + 0.08;
        this.stepSmooth = THREE.MathUtils.clamp(this.stepSmooth, -lim, lim);
        return;
      }
    }

    if (holding) {
      // Suspended between treads. Staying grounded here is the entire point:
      // the alternative is a fall state on every stair, which means a landing
      // dip, a landing sound and a broken footstep cadence on every stair.
      this.grounded = true;
      this.onSteep = false;
      this.groundNormal.set(0, 1, 0);
      this.velocity.y = 0;
      return;
    }

    if (!hit) {
      this.grounded = false;
      return;
    }

    // Too steep to stand on: not grounded, but clip the velocity into the plane
    // so we ride down it instead of jittering against it.
    this.grounded = false;
    this.onSteep = true;
    const vd = this.velocity.dot(hit.normal);
    if (vd < 0) this.velocity.addScaledVector(hit.normal, -vd);
    const dist = probe * hit.t;
    if (dist < 0.02) this.position.y -= dist;
  }

  private decayStepSmooth(dt: number): void {
    if (this.stepSmooth === 0) return;
    const k = Math.exp(-MOVE.stepSmoothRate * dt);
    this.stepSmooth *= k;
    if (Math.abs(this.stepSmooth) < 1e-4) this.stepSmooth = 0;
  }

  // -- cadence --------------------------------------------------------------

  private updateStride(dt: number): void {
    if (!this.grounded || this.sliding) {
      // Hold the phase so the first step after landing lands on the beat.
      return;
    }
    const dx = this.position.x - this.prevPosition.x;
    const dz = this.position.z - this.prevPosition.z;
    const travelled = Math.hypot(dx, dz);
    const speed = travelled / Math.max(dt, 1e-5);
    if (speed < 0.35) return; // idle: the cadence holds where it is

    let stride = MOVE.strideBase + speed * MOVE.stridePerSpeed;
    if (this.stance === 'crouch') stride *= MOVE.strideCrouchScale;
    else if (this.stance === 'prone') stride *= MOVE.strideProneScale;

    const before = this.strideAccum;
    this.strideAccum += travelled / stride;
    // Two footfalls per cycle: phase 0 and phase 0.5.
    if (Math.floor(before * 2) !== Math.floor(this.strideAccum * 2)) {
      this.footstepPending = true;
    }
    this.stridePhase = this.strideAccum - Math.floor(this.strideAccum);
  }

  private updateGait(): void {
    if (this.mantleActive) this.gait = 'mantle';
    else if (this.sliding) this.gait = 'slide';
    else if (!this.grounded) this.gait = 'air';
    else if (this.stance === 'prone') this.gait = 'prone';
    else if (this.stance === 'crouch') this.gait = 'crouch';
    else if (this.tacSprinting) this.gait = 'tacsprint';
    else if (this.sprinting) this.gait = 'sprint';
    else this.gait = this.horizontalSpeed > 0.4 ? 'walk' : 'idle';
  }

  /** Fall damage for an impact speed, 0 when the landing is survivable. */
  static fallDamage(impactSpeed: number): number {
    if (impactSpeed <= MOVE.fallDamageSpeed) return 0;
    const t = (impactSpeed - MOVE.fallDamageSpeed) / (MOVE.fallLethalSpeed - MOVE.fallDamageSpeed);
    return Math.min(100, Math.pow(Math.max(0, t), 1.5) * 100);
  }
}
