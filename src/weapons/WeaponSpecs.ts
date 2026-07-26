/**
 * Weapon data.
 *
 * Three layers live here, deliberately separated:
 *
 *  - `WeaponSpec`  — the shared contract other systems read (ballistics, HUD,
 *                    audio). Numbers are internally consistent: sustained DPS
 *                    lands in a class band, ADS time tracks weight, and range
 *                    falloff tracks muzzle velocity.
 *  - `WeaponFeel`  — tuning that only the weapon system cares about: bloom,
 *                    move penalties, burst cadence, animation timings.
 *  - `WeaponVisual`— proportions and finish for the procedural viewmodel.
 *
 * Everything is authored in "viewmodel metres". A real M4 is 0.84 m long; the
 * models here are foreshortened to ~0.62 m the way every shipped FPS does it,
 * so that when the optic is on the camera axis the buttpad sits just in front
 * of the near plane instead of a metre behind your eye.
 */
import type { WeaponSpec } from '../core/Contracts';

// ---------------------------------------------------------------------------
// Recoil patterns
// ---------------------------------------------------------------------------

/**
 * A learnable recoil path. The first `path.length` shots of a magazine follow
 * the designed curve exactly (multipliers on the spec impulse), after which the
 * weapon randomises inside a cone around `sustain`. This is the CoD/CS model:
 * the opening burst is memorisable, the tail is not.
 */
export interface RecoilPattern {
  /** [horizontal, vertical] multipliers, shot 0 first. */
  readonly path: readonly (readonly [number, number])[];
  /** Random half-width added once the path is exhausted. */
  readonly coneH: number;
  readonly coneV: number;
  /** Vertical multiplier during the randomised tail — climb always flattens. */
  readonly sustainV: number;
  /** Sinusoidal horizontal drift in the tail; period in shots. */
  readonly driftH: number;
  readonly driftPeriod: number;
}

// ---------------------------------------------------------------------------
// Feel / tuning
// ---------------------------------------------------------------------------

export interface WeaponFeel {
  /** Extra cone (radians) added per shot fired. */
  bloomPerShot: number;
  /** Bloom ceiling, radians. */
  bloomMax: number;
  /** Seconds for bloom to decay to 1/e. */
  bloomDecay: number;
  /** Spread multiplier at full sprint speed. */
  spreadMoveMult: number;
  /** Spread multiplier while crouched and still. */
  spreadCrouchMult: number;
  /** Spread multiplier while airborne. */
  spreadAirMult: number;
  /** Seconds between bursts, on top of the in-burst interval. */
  burstCooldown: number;
  /** Camera shake amount per shot. */
  shake: number;
  /** Viewmodel kick, metres back along the bore. */
  kickBack: number;
  /** Viewmodel kick, metres up. */
  kickUp: number;
  /** Viewmodel rotational kick, radians (pitch up, yaw, roll). */
  kickPitch: number;
  kickYaw: number;
  kickRoll: number;
  /** Spring stiffness for recoil recovery on the viewmodel. */
  kickStiffness: number;
  /** Bolt/slide travel, metres. */
  boltThrow: number;
  /** Fraction of the fire interval the bolt spends cycling (clamped ≤ 0.9). */
  boltCycleFraction: number;
  /** Casing ejection speed, m/s, and spin, rad/s. */
  ejectSpeed: number;
  ejectSpin: number;
  /** Muzzle flash radius (m) and point-light intensity (candela-ish). */
  flashSize: number;
  flashIntensity: number;
  /** Seconds of sprint-to-fire delay. Real CoD mechanic. */
  sprintOutTime: number;
  /** Multiplier on hip spread while sprinting out of the sprint pose. */
  sprintOutSpread: number;
}

// ---------------------------------------------------------------------------
// Visual description consumed by the viewmodel builder
// ---------------------------------------------------------------------------

export type HandguardStyle = 'mlok' | 'vented' | 'pump' | 'slab' | 'none';
export type StockStyle = 'collapsible' | 'fixed' | 'skeleton' | 'sniper' | 'none';
export type MagStyle = 'stanag' | 'stick' | 'pistol' | 'tube' | 'none';
export type MuzzleStyle = 'birdcage' | 'brake' | 'comp' | 'choke' | 'plain';
export type ChargingStyle = 'ar' | 'side' | 'boltaction' | 'slide' | 'pump';

export interface WeaponVisual {
  /** Receiver block: length along the bore, height, width, and its centre. */
  receiverLength: number;
  receiverHeight: number;
  receiverWidth: number;
  receiverZ: number;
  receiverY: number;
  /** Round AR-style upper riding on top of a flat lower. */
  upperTube: boolean;
  barrelLength: number;
  barrelRadius: number;
  handguard: HandguardStyle;
  handguardLength: number;
  handguardRadius: number;
  /** Number of M-LOK / vent slot stations along the handguard. */
  handguardSlots: number;
  muzzle: MuzzleStyle;
  stock: StockStyle;
  stockLength: number;
  stockDrop: number;
  gripAngle: number;
  gripLength: number;
  mag: MagStyle;
  magLength: number;
  magWidth: number;
  magDepth: number;
  /** Curvature of a banana mag, radians of arc over its length. */
  magCurve: number;
  /** +1 ejects to the shooter's right. */
  ejectSide: 1 | -1;
  charging: ChargingStyle;
  /** Height of the optic's sight line above the bore. */
  opticHeight: number;
  /** Distance from eye to the optic when fully aimed, metres. */
  adsDistance: number;
  /** Hip-fire pose: position then euler XYZ. */
  hipPos: readonly [number, number, number];
  hipRot: readonly [number, number, number];
  polymerColor: number;
  metalColor: number;
  accentColor: number;
  /** Reticle emissive colour. */
  reticleColor: number;
  hands: boolean;
}

// ---------------------------------------------------------------------------
// The roster
// ---------------------------------------------------------------------------

const DEG = Math.PI / 180;

export interface WeaponEntry {
  spec: WeaponSpec;
  feel: WeaponFeel;
  visual: WeaponVisual;
  recoil: RecoilPattern;
}

/**
 * Base feel, overridden per weapon. Keeping a single base means a change to
 * "how guns feel" is one edit, not six.
 */
const BASE_FEEL: WeaponFeel = {
  bloomPerShot: 0.0016,
  bloomMax: 0.030,
  bloomDecay: 0.28,
  spreadMoveMult: 2.1,
  spreadCrouchMult: 0.68,
  spreadAirMult: 3.0,
  burstCooldown: 0,
  shake: 0.10,
  kickBack: 0.014,
  kickUp: 0.0035,
  kickPitch: 0.052,
  kickYaw: 0.016,
  kickRoll: 0.030,
  kickStiffness: 260,
  boltThrow: 0.030,
  boltCycleFraction: 0.55,
  ejectSpeed: 2.8,
  ejectSpin: 26,
  flashSize: 0.075,
  flashIntensity: 34,
  sprintOutTime: 0.15,
  sprintOutSpread: 2.6,
};

function feel(over: Partial<WeaponFeel>): WeaponFeel {
  return { ...BASE_FEEL, ...over };
}

// --- MK4 Ranger: the reference assault rifle -------------------------------

const MK4: WeaponEntry = {
  spec: {
    id: 'mk4_ranger',
    displayName: 'MK4 Ranger',
    category: 'ar',
    fireMode: 'auto',
    rpm: 730,
    magSize: 30,
    reserveAmmo: 210,
    damage: 34,
    damageRangeNear: 28,
    damageRangeFar: 55,
    damageFalloff: 0.62,
    muzzleVelocity: 880,
    pellets: 1,
    spreadHip: 0.0320,
    spreadAds: 0.0034,
    recoilVertical: 0.0104,
    recoilHorizontal: 0.0034,
    recoilRecovery: 0.28,
    adsTime: 0.26,
    reloadTime: 2.10,
    reloadEmptyTime: 2.82,
    drawTime: 0.55,
    adsFovScale: 0.72,
    penetration: 12,
    headshotMultiplier: 1.5,
    optic: 'reddot',
  },
  feel: feel({ shake: 0.11, kickBack: 0.015, boltThrow: 0.028 }),
  visual: {
    receiverLength: 0.235, receiverHeight: 0.062, receiverWidth: 0.040,
    receiverZ: -0.055, receiverY: 0.0,
    upperTube: true,
    barrelLength: 0.250, barrelRadius: 0.0090,
    handguard: 'mlok', handguardLength: 0.190, handguardRadius: 0.0225, handguardSlots: 5,
    muzzle: 'birdcage',
    stock: 'collapsible', stockLength: 0.150, stockDrop: 0.008,
    gripAngle: 26 * DEG, gripLength: 0.084,
    mag: 'stanag', magLength: 0.130, magWidth: 0.024, magDepth: 0.050, magCurve: 0.20,
    ejectSide: 1,
    charging: 'ar',
    opticHeight: 0.050,
    adsDistance: 0.248,
    hipPos: [0.112, -0.108, -0.150],
    hipRot: [-1.5 * DEG, -3.6 * DEG, 1.2 * DEG],
    polymerColor: 0x22262a,
    metalColor: 0x2c2f33,
    accentColor: 0x14161a,
    reticleColor: 0xff2a18,
    hands: true,
  },
  recoil: {
    // Opens straight up, drifts left through shots 5-8: a classic AR path.
    path: [
      [0.00, 1.00], [0.12, 1.05], [-0.10, 1.02], [-0.28, 0.98],
      [-0.44, 0.92], [-0.52, 0.86], [-0.36, 0.82], [-0.08, 0.80],
    ],
    coneH: 0.85, coneV: 0.26, sustainV: 0.76, driftH: 0.55, driftPeriod: 9,
  },
};

// --- VKS-9 Wasp: high-rate SMG --------------------------------------------

const VKS9: WeaponEntry = {
  spec: {
    id: 'vks9_wasp',
    displayName: 'VKS-9 Wasp',
    category: 'smg',
    fireMode: 'auto',
    rpm: 950,
    magSize: 32,
    reserveAmmo: 224,
    damage: 26,
    damageRangeNear: 14,
    damageRangeFar: 30,
    damageFalloff: 0.55,
    muzzleVelocity: 400,
    pellets: 1,
    spreadHip: 0.0268,
    spreadAds: 0.0052,
    recoilVertical: 0.0082,
    recoilHorizontal: 0.0046,
    recoilRecovery: 0.34,
    adsTime: 0.20,
    reloadTime: 1.82,
    reloadEmptyTime: 2.44,
    drawTime: 0.44,
    adsFovScale: 0.80,
    penetration: 6,
    headshotMultiplier: 1.4,
    optic: 'holo',
  },
  feel: feel({
    bloomPerShot: 0.0019, bloomMax: 0.034, shake: 0.085,
    kickBack: 0.011, kickPitch: 0.042, kickYaw: 0.020, kickStiffness: 300,
    boltThrow: 0.022, boltCycleFraction: 0.60, ejectSpeed: 2.4, flashSize: 0.062,
    flashIntensity: 26, spreadMoveMult: 1.7,
  }),
  visual: {
    receiverLength: 0.200, receiverHeight: 0.058, receiverWidth: 0.038,
    receiverZ: -0.040, receiverY: 0.0,
    upperTube: false,
    barrelLength: 0.150, barrelRadius: 0.0080,
    handguard: 'vented', handguardLength: 0.110, handguardRadius: 0.0200, handguardSlots: 4,
    muzzle: 'comp',
    stock: 'skeleton', stockLength: 0.118, stockDrop: 0.004,
    gripAngle: 21 * DEG, gripLength: 0.080,
    mag: 'stick', magLength: 0.148, magWidth: 0.021, magDepth: 0.038, magCurve: 0.10,
    ejectSide: 1,
    charging: 'side',
    opticHeight: 0.046,
    adsDistance: 0.215,
    hipPos: [0.108, -0.100, -0.135],
    hipRot: [-1.0 * DEG, -4.4 * DEG, 1.6 * DEG],
    polymerColor: 0x1e2124,
    metalColor: 0x303338,
    accentColor: 0x101215,
    reticleColor: 0xff3a20,
    hands: true,
  },
  recoil: {
    path: [
      [0.00, 1.00], [-0.16, 0.98], [0.22, 0.94], [0.40, 0.90],
      [0.30, 0.86], [0.02, 0.84], [-0.26, 0.82], [-0.40, 0.80],
    ],
    coneH: 1.15, coneV: 0.30, sustainV: 0.74, driftH: 0.62, driftPeriod: 7,
  },
};

// --- M16-BR: burst-fire marksman rifle with an ACOG -------------------------

const M16BR: WeaponEntry = {
  spec: {
    id: 'm16br',
    displayName: 'M16-BR',
    category: 'ar',
    fireMode: 'burst',
    rpm: 800,
    burstCount: 3,
    magSize: 30,
    reserveAmmo: 180,
    damage: 38,
    damageRangeNear: 34,
    damageRangeFar: 64,
    damageFalloff: 0.68,
    muzzleVelocity: 950,
    pellets: 1,
    spreadHip: 0.0362,
    spreadAds: 0.0026,
    recoilVertical: 0.0126,
    recoilHorizontal: 0.0030,
    recoilRecovery: 0.30,
    adsTime: 0.30,
    reloadTime: 2.24,
    reloadEmptyTime: 2.96,
    drawTime: 0.60,
    adsFovScale: 0.55,
    penetration: 16,
    headshotMultiplier: 1.55,
    optic: 'acog',
  },
  feel: feel({
    burstCooldown: 0.26, bloomPerShot: 0.0021, shake: 0.13,
    kickBack: 0.017, kickPitch: 0.060, kickStiffness: 240,
    boltThrow: 0.030, flashSize: 0.082, flashIntensity: 38,
  }),
  visual: {
    receiverLength: 0.250, receiverHeight: 0.064, receiverWidth: 0.041,
    receiverZ: -0.058, receiverY: 0.0,
    upperTube: true,
    barrelLength: 0.300, barrelRadius: 0.0092,
    handguard: 'slab', handguardLength: 0.205, handguardRadius: 0.0250, handguardSlots: 6,
    muzzle: 'birdcage',
    stock: 'fixed', stockLength: 0.168, stockDrop: 0.006,
    gripAngle: 27 * DEG, gripLength: 0.086,
    mag: 'stanag', magLength: 0.132, magWidth: 0.024, magDepth: 0.050, magCurve: 0.18,
    ejectSide: 1,
    charging: 'ar',
    opticHeight: 0.058,
    adsDistance: 0.275,
    hipPos: [0.114, -0.112, -0.156],
    hipRot: [-1.6 * DEG, -3.2 * DEG, 1.0 * DEG],
    polymerColor: 0x2a2822,
    metalColor: 0x2b2d30,
    accentColor: 0x16171a,
    reticleColor: 0xff5a12,
    hands: true,
  },
  recoil: {
    // Burst weapon: each three-round burst walks up hard, then resets.
    path: [
      [0.00, 1.00], [0.18, 1.12], [0.34, 1.22],
      [-0.06, 1.00], [-0.24, 1.10], [-0.40, 1.20],
      [0.10, 1.00], [0.30, 1.10],
    ],
    coneH: 0.70, coneV: 0.22, sustainV: 1.0, driftH: 0.40, driftPeriod: 6,
  },
};

// --- KV-800 Ballista: bolt-action sniper ------------------------------------

const KV800: WeaponEntry = {
  spec: {
    id: 'kv800_ballista',
    displayName: 'KV-800 Ballista',
    category: 'sniper',
    fireMode: 'bolt',
    rpm: 48,
    magSize: 5,
    reserveAmmo: 40,
    damage: 145,
    damageRangeNear: 60,
    damageRangeFar: 120,
    damageFalloff: 0.85,
    muzzleVelocity: 1000,
    pellets: 1,
    spreadHip: 0.0900,
    spreadAds: 0.0002,
    recoilVertical: 0.0350,
    recoilHorizontal: 0.0062,
    recoilRecovery: 0.22,
    adsTime: 0.42,
    reloadTime: 3.10,
    reloadEmptyTime: 3.85,
    drawTime: 0.85,
    adsFovScale: 0.32,
    penetration: 42,
    headshotMultiplier: 2.2,
    optic: 'sniper',
  },
  feel: feel({
    bloomPerShot: 0.0, bloomMax: 0.0, shake: 0.42,
    kickBack: 0.038, kickUp: 0.010, kickPitch: 0.150, kickYaw: 0.028, kickRoll: 0.070,
    kickStiffness: 150,
    boltThrow: 0.060, boltCycleFraction: 0.72, ejectSpeed: 3.4, ejectSpin: 18,
    flashSize: 0.125, flashIntensity: 68,
    spreadMoveMult: 2.6, spreadAirMult: 4.5,
  }),
  visual: {
    receiverLength: 0.230, receiverHeight: 0.052, receiverWidth: 0.036,
    receiverZ: -0.050, receiverY: 0.0,
    upperTube: false,
    barrelLength: 0.330, barrelRadius: 0.0115,
    handguard: 'slab', handguardLength: 0.185, handguardRadius: 0.0245, handguardSlots: 3,
    muzzle: 'brake',
    stock: 'sniper', stockLength: 0.185, stockDrop: 0.010,
    gripAngle: 24 * DEG, gripLength: 0.088,
    mag: 'stick', magLength: 0.070, magWidth: 0.026, magDepth: 0.054, magCurve: 0.0,
    ejectSide: 1,
    charging: 'boltaction',
    opticHeight: 0.062,
    adsDistance: 0.235,
    hipPos: [0.128, -0.118, -0.168],
    hipRot: [-2.0 * DEG, -3.0 * DEG, 1.4 * DEG],
    polymerColor: 0x2e3128,
    metalColor: 0x26282b,
    accentColor: 0x121316,
    reticleColor: 0x1cff9a,
    hands: true,
  },
  recoil: {
    path: [[0.00, 1.00], [0.10, 1.00], [-0.12, 1.00], [0.06, 1.00],
      [-0.08, 1.00], [0.04, 1.00], [-0.05, 1.00], [0.07, 1.00]],
    coneH: 0.25, coneV: 0.10, sustainV: 1.0, driftH: 0.0, driftPeriod: 1,
  },
};

// --- M870 Breacher: pump shotgun --------------------------------------------

const M870: WeaponEntry = {
  spec: {
    id: 'm870_breacher',
    displayName: 'M870 Breacher',
    category: 'shotgun',
    fireMode: 'semi',
    rpm: 76,
    magSize: 6,
    reserveAmmo: 36,
    damage: 22,
    damageRangeNear: 8,
    damageRangeFar: 18,
    damageFalloff: 0.35,
    muzzleVelocity: 380,
    pellets: 9,
    spreadHip: 0.0620,
    spreadAds: 0.0420,
    recoilVertical: 0.0420,
    recoilHorizontal: 0.0100,
    recoilRecovery: 0.20,
    adsTime: 0.34,
    reloadTime: 3.40,
    reloadEmptyTime: 3.85,
    drawTime: 0.62,
    adsFovScale: 0.86,
    penetration: 4,
    headshotMultiplier: 1.35,
    optic: 'irons',
  },
  feel: feel({
    bloomPerShot: 0.0, bloomMax: 0.0, shake: 0.34,
    kickBack: 0.034, kickUp: 0.009, kickPitch: 0.130, kickYaw: 0.024, kickRoll: 0.058,
    kickStiffness: 170,
    boltThrow: 0.070, boltCycleFraction: 0.80, ejectSpeed: 2.2, ejectSpin: 14,
    flashSize: 0.150, flashIntensity: 72,
  }),
  visual: {
    receiverLength: 0.180, receiverHeight: 0.056, receiverWidth: 0.042,
    receiverZ: -0.030, receiverY: 0.0,
    upperTube: false,
    barrelLength: 0.290, barrelRadius: 0.0125,
    handguard: 'pump', handguardLength: 0.130, handguardRadius: 0.0250, handguardSlots: 8,
    muzzle: 'choke',
    stock: 'fixed', stockLength: 0.175, stockDrop: 0.014,
    gripAngle: 30 * DEG, gripLength: 0.082,
    mag: 'tube', magLength: 0.250, magWidth: 0.0130, magDepth: 0.0130, magCurve: 0,
    ejectSide: 1,
    charging: 'pump',
    opticHeight: 0.034,
    adsDistance: 0.230,
    hipPos: [0.118, -0.112, -0.150],
    hipRot: [-1.8 * DEG, -3.8 * DEG, 1.6 * DEG],
    polymerColor: 0x241f1a,
    metalColor: 0x212326,
    accentColor: 0x131416,
    reticleColor: 0xff8c22,
    hands: true,
  },
  recoil: {
    path: [[0.00, 1.00], [0.14, 0.98], [-0.16, 1.00], [0.10, 0.96],
      [-0.12, 1.00], [0.08, 0.98], [-0.09, 1.00], [0.11, 0.98]],
    coneH: 0.55, coneV: 0.18, sustainV: 1.0, driftH: 0.0, driftPeriod: 1,
  },
};

// --- P226 Sidearm -----------------------------------------------------------

const P226: WeaponEntry = {
  spec: {
    id: 'p226_sidearm',
    displayName: 'P226 Sidearm',
    category: 'pistol',
    fireMode: 'semi',
    rpm: 430,
    magSize: 15,
    reserveAmmo: 90,
    damage: 32,
    damageRangeNear: 16,
    damageRangeFar: 34,
    damageFalloff: 0.50,
    muzzleVelocity: 380,
    pellets: 1,
    spreadHip: 0.0240,
    spreadAds: 0.0046,
    recoilVertical: 0.0136,
    recoilHorizontal: 0.0056,
    recoilRecovery: 0.40,
    adsTime: 0.17,
    reloadTime: 1.54,
    reloadEmptyTime: 2.06,
    drawTime: 0.34,
    adsFovScale: 0.88,
    penetration: 5,
    headshotMultiplier: 1.6,
    optic: 'irons',
  },
  feel: feel({
    bloomPerShot: 0.0026, bloomMax: 0.028, bloomDecay: 0.22, shake: 0.09,
    kickBack: 0.012, kickPitch: 0.068, kickYaw: 0.022, kickRoll: 0.036,
    kickStiffness: 330,
    boltThrow: 0.030, boltCycleFraction: 0.42, ejectSpeed: 2.6, ejectSpin: 30,
    flashSize: 0.058, flashIntensity: 24,
    spreadMoveMult: 1.6, sprintOutTime: 0.11,
  }),
  visual: {
    receiverLength: 0.150, receiverHeight: 0.040, receiverWidth: 0.030,
    receiverZ: -0.030, receiverY: 0.010,
    upperTube: false,
    barrelLength: 0.030, barrelRadius: 0.0072,
    handguard: 'none', handguardLength: 0, handguardRadius: 0, handguardSlots: 0,
    muzzle: 'plain',
    stock: 'none', stockLength: 0, stockDrop: 0,
    gripAngle: 18 * DEG, gripLength: 0.108,
    mag: 'pistol', magLength: 0.104, magWidth: 0.019, magDepth: 0.032, magCurve: 0,
    ejectSide: 1,
    charging: 'slide',
    opticHeight: 0.048,
    adsDistance: 0.290,
    hipPos: [0.098, -0.128, -0.180],
    hipRot: [-2.4 * DEG, -5.0 * DEG, 2.2 * DEG],
    polymerColor: 0x1c1e21,
    metalColor: 0x2a2c30,
    accentColor: 0x0f1013,
    reticleColor: 0xffffff,
    hands: true,
  },
  recoil: {
    path: [[0.00, 1.00], [0.20, 0.98], [-0.24, 0.96], [0.16, 0.94],
      [-0.18, 0.92], [0.22, 0.90], [-0.14, 0.90], [0.12, 0.88]],
    coneH: 1.0, coneV: 0.28, sustainV: 0.86, driftH: 0.3, driftPeriod: 5,
  },
};

// ---------------------------------------------------------------------------

export const WEAPON_ENTRIES: readonly WeaponEntry[] = [MK4, VKS9, M16BR, KV800, M870, P226];

export const WEAPON_SPECS: readonly WeaponSpec[] = WEAPON_ENTRIES.map((e) => e.spec);

export const WEAPON_SPEC_MAP: Readonly<Record<string, WeaponSpec>> = Object.freeze(
  WEAPON_ENTRIES.reduce<Record<string, WeaponSpec>>((acc, e) => {
    acc[e.spec.id] = e.spec;
    return acc;
  }, {}),
);

const ENTRY_MAP = new Map<string, WeaponEntry>(WEAPON_ENTRIES.map((e) => [e.spec.id, e]));

export function getEntry(id: string): WeaponEntry | undefined {
  return ENTRY_MAP.get(id);
}

/** Spec lookup for systems that only know a weapon id (ballistics, HUD, audio). */
export function getSpec(id: string): WeaponSpec | undefined {
  return ENTRY_MAP.get(id)?.spec;
}

export function getFeel(id: string): WeaponFeel | undefined {
  return ENTRY_MAP.get(id)?.feel;
}

export function getVisual(id: string): WeaponVisual | undefined {
  return ENTRY_MAP.get(id)?.visual;
}

export function getRecoilPattern(id: string): RecoilPattern | undefined {
  return ENTRY_MAP.get(id)?.recoil;
}

/** Order the swap key cycles through. */
export const DEFAULT_LOADOUT: readonly string[] = [
  'mk4_ranger', 'vks9_wasp', 'm16br', 'kv800_ballista', 'm870_breacher', 'p226_sidearm',
];

/**
 * Samples the recoil path. `shotIndex` is shots fired since the trigger was
 * last released for semi/burst weapons, or since the magazine was inserted for
 * automatics — the caller decides which counter is authoritative.
 */
export function sampleRecoil(
  pattern: RecoilPattern,
  shotIndex: number,
  out: { h: number; v: number },
): { h: number; v: number } {
  const path = pattern.path;
  if (shotIndex < path.length) {
    const p = path[shotIndex]!;
    out.h = p[0];
    out.v = p[1];
    return out;
  }
  const k = shotIndex - path.length;
  const drift = Math.sin((k / pattern.driftPeriod) * Math.PI * 2) * pattern.driftH;
  out.h = drift + (Math.random() * 2 - 1) * pattern.coneH;
  out.v = pattern.sustainV + (Math.random() * 2 - 1) * pattern.coneV;
  return out;
}

/** Seconds between rounds for a weapon, from its rpm. */
export function fireInterval(spec: WeaponSpec): number {
  return 60 / Math.max(1, spec.rpm);
}
