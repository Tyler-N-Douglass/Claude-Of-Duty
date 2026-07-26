/**
 * EffectsSystem — every particle, decal, tracer, explosion and piece of debris.
 *
 * Impacts are composites: no surface gets a single puff. Concrete throws dust,
 * chips and a spall flash; metal throws a stretched spark shower that cools
 * from white through orange to red, plus bounce sparks staged into the future.
 * Staging is free because the particle simulation is driven by spawn time — a
 * particle with a delay simply stays clipped until its clock starts.
 */
import * as THREE from 'three';
import {
  type ExplosionEvent,
  type FrameTime,
  type GameContext,
  type ImpactEvent,
  type SurfaceKind,
  type System,
  RayMask,
} from '../core/Contracts';
import { CELL, Particles, mulberry32 } from './Particles';
import { DECAL, Decals, decalCellForSurface } from './Decals';
import { Tracers } from './Tracers';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

const DEBRIS_GRAVITY = 19.6;
const MAX_DECALS_PER_FRAME = 4;
const MUZZLE_HEAT_PER_SHOT = 0.09;
const MUZZLE_HEAT_DECAY = 0.34;

/** Linear-space colours. Everything authored here is already scene-referred. */
function srgb(hex: number): THREE.Color {
  return new THREE.Color().setHex(hex, THREE.SRGBColorSpace);
}

interface SurfaceProfile {
  /** Dust/smoke tint. */
  dust: THREE.Color;
  /** Solid fragment tint. */
  debris: THREE.Color;
  dustAmount: number;
  chipAmount: number;
  sparkAmount: number;
  /** Metres, bullet hole decal size. */
  holeSize: number;
  /** Chip particle atlas cell. */
  chipCell: number;
  smokeAlpha: number;
}

const SURFACES: Record<SurfaceKind, SurfaceProfile> = {
  concrete: {
    dust: srgb(0xa8a49b), debris: srgb(0x8b877e), dustAmount: 1, chipAmount: 1, sparkAmount: 0.12,
    holeSize: 0.1, chipCell: CELL.chip, smokeAlpha: 0.36,
  },
  plaster: {
    dust: srgb(0xd8d4cb), debris: srgb(0xcfcbc2), dustAmount: 1.35, chipAmount: 1.1, sparkAmount: 0,
    holeSize: 0.13, chipCell: CELL.chip, smokeAlpha: 0.42,
  },
  metal: {
    dust: srgb(0x6e6e72), debris: srgb(0x9a9a9f), dustAmount: 0.34, chipAmount: 0.35, sparkAmount: 1,
    holeSize: 0.065, chipCell: CELL.chip, smokeAlpha: 0.2,
  },
  wood: {
    dust: srgb(0xb08a5c), debris: srgb(0xa37b52), dustAmount: 0.6, chipAmount: 1.2, sparkAmount: 0.05,
    holeSize: 0.09, chipCell: CELL.splinter, smokeAlpha: 0.3,
  },
  dirt: {
    dust: srgb(0x6d5a42), debris: srgb(0x53442f), dustAmount: 1.5, chipAmount: 0.9, sparkAmount: 0,
    holeSize: 0.14, chipCell: CELL.chip, smokeAlpha: 0.5,
  },
  sand: {
    dust: srgb(0xc4ab7e), debris: srgb(0xb09a70), dustAmount: 1.7, chipAmount: 0.7, sparkAmount: 0,
    holeSize: 0.15, chipCell: CELL.chip, smokeAlpha: 0.52,
  },
  glass: {
    dust: srgb(0xcfe0e8), debris: srgb(0xdff0f8), dustAmount: 0.35, chipAmount: 1.4, sparkAmount: 0.1,
    holeSize: 0.11, chipCell: CELL.shard, smokeAlpha: 0.18,
  },
  water: {
    dust: srgb(0xbfd4dc), debris: srgb(0xd6e6ee), dustAmount: 0.8, chipAmount: 1.6, sparkAmount: 0,
    holeSize: 0.3, chipCell: CELL.droplet, smokeAlpha: 0.28,
  },
  flesh: {
    dust: srgb(0x4a0d0a), debris: srgb(0x6b1410), dustAmount: 0.7, chipAmount: 0.8, sparkAmount: 0,
    holeSize: 0.06, chipCell: CELL.droplet, smokeAlpha: 0.4,
  },
  foliage: {
    dust: srgb(0x4e6b32), debris: srgb(0x5f7d3a), dustAmount: 0.45, chipAmount: 1.1, sparkAmount: 0,
    holeSize: 0.07, chipCell: CELL.splinter, smokeAlpha: 0.22,
  },
  rubber: {
    dust: srgb(0x2c2c2e), debris: srgb(0x3a3a3d), dustAmount: 0.4, chipAmount: 0.6, sparkAmount: 0,
    holeSize: 0.07, chipCell: CELL.chip, smokeAlpha: 0.3,
  },
  fabric: {
    dust: srgb(0x8a8378), debris: srgb(0x6e685e), dustAmount: 0.6, chipAmount: 0.5, sparkAmount: 0,
    holeSize: 0.08, chipCell: CELL.chip, smokeAlpha: 0.32,
  },
};

// ---------------------------------------------------------------------------
// Shimmer (heat haze fallback)
// ---------------------------------------------------------------------------

const SHIMMER_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SHIMMER_FRAG = /* glsl */ `
precision highp float;
uniform float uTime;
uniform float uOpacity;
uniform float uRing;
uniform vec3 uTint;
varying vec2 vUv;

void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float r = length(p);
  // uRing = 0 gives a filled shimmer patch (barrel), 1 gives a hollow ring
  // (blast wave). Both wobble on the same cheap two-axis interference field.
  float shape = mix(smoothstep(1.0, 0.25, r), exp(-pow((r - 0.74) / 0.24, 2.0)), uRing);
  float n = sin(p.x * 21.0 + uTime * 8.5) * sin(p.y * 17.0 - uTime * 6.7);
  float n2 = sin((p.x + p.y) * 33.0 - uTime * 12.0);
  float ripple = 0.5 + 0.5 * (n * 0.7 + n2 * 0.3);
  float a = shape * uOpacity * (0.25 + 0.75 * ripple);
  if (a < 0.002) discard;
  gl_FragColor = vec4(uTint * (0.55 + 0.9 * ripple), a);
}
`;

interface ShimmerInstance {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  active: boolean;
  age: number;
  life: number;
  radius0: number;
  radius1: number;
  strength: number;
}

interface Debris {
  active: boolean;
  pos: THREE.Vector3;
  vel: THREE.Vector3;
  spin: THREE.Vector3;
  quat: THREE.Quaternion;
  scale: number;
  age: number;
  life: number;
  resting: boolean;
  bounces: number;
  dust: THREE.Color;
}

interface PendingDecal {
  active: boolean;
  at: number;
  point: THREE.Vector3;
  normal: THREE.Vector3;
  cell: number;
  size: number;
  life: number;
  opacity: number;
  alignUp: boolean;
  drip: number;
  parallax: number;
}

interface ExplosionLight {
  light: THREE.PointLight;
  active: boolean;
  age: number;
  peak: number;
  linger: number;
  range: number;
}

interface DepthProvider {
  getSceneDepthTexture?(): THREE.Texture | null;
  sceneDepthTexture?: THREE.Texture | null;
  getDepthTexture?(): THREE.Texture | null;
}

interface DistortionProvider {
  addDistortion?(object: THREE.Object3D): void;
  registerDistortion?(object: THREE.Object3D): void;
}

interface SunProvider {
  sunDirection?: THREE.Vector3;
  sunColor?: THREE.Color;
  sunIntensity?: number;
  zenithColor?: THREE.Color;
  horizonColor?: THREE.Color;
}

interface LooseEmitter {
  emit(type: string, payload: unknown): void;
}

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3();
const _tangent = new THREE.Vector3();
const _bitangent = new THREE.Vector3();
const _matrix = new THREE.Matrix4();
const _quat = new THREE.Quaternion();
const _scaleVec = new THREE.Vector3();
const _color = new THREE.Color();
const _colorB = new THREE.Color();
const _down = new THREE.Vector3(0, -1, 0);
const _up = new THREE.Vector3(0, 1, 0);

const SPARK_HOT = srgb(0xfff6e0);
const SPARK_MID = srgb(0xff9a3c);
const SPARK_COLD = srgb(0x8c1e05);
const FIRE_CORE = srgb(0xfff2c4);
const FIRE_MID = srgb(0xff7a1e);
const FIRE_COLD = srgb(0x501004);
const SMOKE_DARK = srgb(0x232326);
const SMOKE_LIGHT = srgb(0x8e8b86);
const BLOOD_COLOR = srgb(0x6e0a06);
const TRACER_FRIENDLY = srgb(0xffc36a);
const TRACER_HOSTILE = srgb(0xff5a2a);

export class EffectsSystem implements System {
  readonly name = 'fx';

  private ctx: GameContext | null = null;
  private particles: Particles | null = null;
  private decals: Decals | null = null;
  private tracers: Tracers | null = null;

  private readonly unsubscribe: (() => void)[] = [];
  private readonly rng = mulberry32(0x0df077);
  private density = 1;
  private decalsThisFrame = 0;
  private clock = 0;

  // Debris ------------------------------------------------------------------
  private debrisMesh: THREE.InstancedMesh | null = null;
  private debrisGeometry: THREE.BufferGeometry | null = null;
  private debrisMaterial: THREE.MeshStandardMaterial | null = null;
  private debrisTexture: THREE.Texture | null = null;
  private readonly debris: Debris[] = [];
  private debrisCursor = 0;

  // Lights ------------------------------------------------------------------
  private readonly lights: ExplosionLight[] = [];

  // Shimmer -----------------------------------------------------------------
  private shimmerMaterialTemplate: THREE.ShaderMaterial | null = null;
  private readonly shimmers: ShimmerInstance[] = [];
  private barrelShimmer: ShimmerInstance | null = null;
  private shimmerGeometry: THREE.PlaneGeometry | null = null;

  // Muzzle ------------------------------------------------------------------
  private muzzleHeat = 0;
  private readonly muzzleLocal = new THREE.Vector3(0.16, -0.1, -0.55);
  private hasMuzzleLocal = false;
  private tracerCounter = 0;

  // Deferred ----------------------------------------------------------------
  private readonly pending: PendingDecal[] = [];
  private depthProbeTimer = 0;

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  init(ctx: GameContext): void {
    this.ctx = ctx;
    const q = ctx.quality;
    this.density = q.preset === 'low' ? 0.45 : q.preset === 'medium' ? 0.72 : q.preset === 'ultra' ? 1.25 : 1;

    const cellSize = q.particleBudget >= 4000 ? 256 : 128;
    this.particles = new Particles(ctx.scene, {
      budget: q.particleBudget,
      cellSize,
      soft: this.resolveDepthTexture() !== null,
    });
    this.particles.setSizeScale(q.preset === 'low' ? 0.9 : 1);

    this.decals = new Decals(ctx.scene, ctx.physics, q.decalBudget, q.decalBudget >= 256 ? 256 : 128);
    this.decals.setFadeRange(q.preset === 'low' ? 18 : 30, q.preset === 'low' ? 30 : 55);

    this.tracers = new Tracers(ctx.scene, q.preset === 'low' ? 48 : 128);
    this.tracers.onCrack = this.onTracerCrack;

    this.buildDebris(ctx);
    this.buildLights(ctx);
    this.buildShimmer(ctx);

    for (let i = 0; i < 24; i++) {
      this.pending.push({
        active: false, at: 0, point: new THREE.Vector3(), normal: new THREE.Vector3(0, 1, 0),
        cell: 0, size: 0.2, life: 0, opacity: 1, alignUp: false, drip: 0, parallax: 0,
      });
    }

    const ev = ctx.events;
    this.unsubscribe.push(
      ev.on('shot.impact', this.onImpact),
      ev.on('shot.tracer', this.onTracer),
      ev.on('shot.fired', this.onShotFired),
      ev.on('explosion', this.onExplosion),
      ev.on('entity.killed', this.onKilled),
      ev.on('player.footstep', this.onFootstep),
      ev.on('player.land', this.onLand),
      ev.on('quality.changed', this.onQualityChanged),
    );
  }

  // -------------------------------------------------------------------------
  // Construction helpers
  // -------------------------------------------------------------------------

  private buildDebris(ctx: GameContext): void {
    const count = ctx.quality.preset === 'low' ? 24 : ctx.quality.preset === 'ultra' ? 96 : 56;
    const geo = new THREE.IcosahedronGeometry(0.5, 0);
    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    // Displace by a hash of the *direction*, so vertices shared between faces
    // move together and the chunk stays closed instead of tearing apart.
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      const h = Math.sin(x * 91.7 + y * 47.3 + z * 13.9) * 43758.5453;
      const f = 0.62 + (h - Math.floor(h)) * 0.72;
      pos.setXYZ(i, x * f, y * f * 0.82, z * f);
    }
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
    this.debrisGeometry = geo;

    this.debrisTexture = buildRockTexture(128);
    this.debrisMaterial = new THREE.MeshStandardMaterial({
      map: this.debrisTexture,
      roughness: 0.94,
      metalness: 0.02,
      envMapIntensity: 0.6,
    });

    const mesh = new THREE.InstancedMesh(geo, this.debrisMaterial, count);
    mesh.name = 'fx-debris';
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    mesh.count = 0;
    mesh.visible = false;
    const colors = new THREE.InstancedBufferAttribute(new Float32Array(count * 3).fill(1), 3);
    mesh.instanceColor = colors;
    ctx.scene.add(mesh);
    this.debrisMesh = mesh;

    _matrix.makeScale(0, 0, 0);
    for (let i = 0; i < count; i++) {
      mesh.setMatrixAt(i, _matrix);
      this.debris.push({
        active: false,
        pos: new THREE.Vector3(),
        vel: new THREE.Vector3(),
        spin: new THREE.Vector3(),
        quat: new THREE.Quaternion(),
        scale: 0.06,
        age: 0,
        life: 10,
        resting: false,
        bounces: 0,
        dust: new THREE.Color(0.5, 0.5, 0.5),
      });
    }
    mesh.instanceMatrix.needsUpdate = true;
  }

  private buildLights(ctx: GameContext): void {
    // Created once and kept in the scene forever: the light count is baked into
    // every compiled program, so adding one mid-fight would recompile the world.
    const count = ctx.quality.preset === 'low' ? 2 : 3;
    for (let i = 0; i < count; i++) {
      const light = new THREE.PointLight(0xffb060, 0, 30, 2);
      light.name = `fx-explosion-light-${i}`;
      light.castShadow = false;
      light.visible = false;
      ctx.scene.add(light);
      this.lights.push({ light, active: false, age: 0, peak: 0, linger: 0, range: 30 });
    }
  }

  private buildShimmer(ctx: GameContext): void {
    this.shimmerGeometry = new THREE.PlaneGeometry(1, 1);
    this.shimmerMaterialTemplate = new THREE.ShaderMaterial({
      vertexShader: SHIMMER_VERT,
      fragmentShader: SHIMMER_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uOpacity: { value: 0 },
        uRing: { value: 1 },
        uTint: { value: new THREE.Color(0.92, 0.88, 0.82) },
      },
      transparent: true,
      depthWrite: false,
      depthTest: true,
      toneMapped: false,
      side: THREE.DoubleSide,
    });

    const host = ctx.system<System & DistortionProvider>('render');
    const attach = (mesh: THREE.Mesh, viewmodel: boolean): void => {
      if (!viewmodel && host) {
        if (typeof host.addDistortion === 'function') {
          host.addDistortion(mesh);
          return;
        }
        if (typeof host.registerDistortion === 'function') {
          host.registerDistortion(mesh);
          return;
        }
      }
      (viewmodel ? ctx.viewScene : ctx.scene).add(mesh);
    };

    const makeShimmer = (viewmodel: boolean, ring: number): ShimmerInstance => {
      const material = this.shimmerMaterialTemplate!.clone();
      material.uniforms.uRing.value = ring;
      const mesh = new THREE.Mesh(this.shimmerGeometry!, material);
      mesh.name = viewmodel ? 'fx-barrel-shimmer' : 'fx-heat-ring';
      mesh.frustumCulled = false;
      mesh.renderOrder = 14;
      mesh.visible = false;
      attach(mesh, viewmodel);
      return { mesh, material, active: false, age: 0, life: 1, radius0: 1, radius1: 2, strength: 0.1 };
    };

    const ringCount = ctx.quality.preset === 'low' ? 1 : 2;
    for (let i = 0; i < ringCount; i++) this.shimmers.push(makeShimmer(false, 1));
    this.barrelShimmer = makeShimmer(true, 0);
  }

  // -------------------------------------------------------------------------
  // Event handlers (payloads are pooled by the emitters — copy, never retain)
  // -------------------------------------------------------------------------

  private readonly onImpact = (e: ImpactEvent): void => {
    this.impact(e.point, e.normal, e.direction, e.surface, e.energy);
  };

  private readonly onTracer = (e: { from: THREE.Vector3; to: THREE.Vector3; speed: number; weaponId: string }): void => {
    const ctx = this.ctx;
    const tracers = this.tracers;
    if (!ctx || !tracers) return;

    const fromCamera = e.from.distanceTo(ctx.camera.position);
    const local = fromCamera < 2.5;
    this.tracerCounter++;
    // Real belts run one tracer in three to five; every round glowing reads as
    // a laser show, none at all reads as nothing happening.
    const bright = local ? this.tracerCounter % 3 === 0 : this.tracerCounter % 2 === 0;

    _tracerSpawn.from.copy(e.from);
    _tracerSpawn.to.copy(e.to);
    _tracerSpawn.speed = e.speed > 1 ? e.speed : 880;
    _tracerSpawn.intensity = bright ? (local ? 2.6 : 3.4) : 0.55;
    _tracerSpawn.color.copy(local ? TRACER_FRIENDLY : TRACER_HOSTILE);
    _tracerSpawn.width = bright ? 0.035 : 0.02;
    _tracerSpawn.tail = 0;
    tracers.spawn(_tracerSpawn, ctx.camera.position);
  };

  private readonly onShotFired = (e: { origin: THREE.Vector3; direction: THREE.Vector3; local: boolean }): void => {
    this.muzzleSmoke(e.origin, e.direction, e.local);
    if (e.local) {
      this.muzzleHeat = Math.min(1.6, this.muzzleHeat + MUZZLE_HEAT_PER_SHOT);
      const ctx = this.ctx;
      if (ctx) {
        // Cache where the muzzle sits relative to the eye so the barrel haze can
        // ride the viewmodel without reaching into the weapon system.
        _v1.copy(e.origin);
        ctx.camera.worldToLocal(_v1);
        if (_v1.z < -0.05 && _v1.length() < 3) {
          this.muzzleLocal.lerp(_v1, this.hasMuzzleLocal ? 0.25 : 1);
          this.hasMuzzleLocal = true;
        }
      }
    }
  };

  private readonly onExplosion = (e: ExplosionEvent): void => {
    this.explosion(e.point, e.radius);
  };

  private readonly onKilled = (e: { entityId: number; headshot: boolean }): void => {
    const ctx = this.ctx;
    if (!ctx) return;
    const target = ctx.entities.get(e.entityId);
    if (!target || e.entityId === ctx.localPlayerId) return;
    _v1.copy(target.position);
    _v1.y += e.headshot ? 1.6 : 1.15;
    _v2.copy(_v1).sub(ctx.camera.position).normalize().multiplyScalar(-1);
    this.bloodBurst(_v1, _v2, e.headshot ? 1.5 : 0.85);

    // Pool under the body once it has had time to fall.
    _v3.copy(target.position);
    _v3.y += 0.6;
    const hit = ctx.physics.raycast(_v3, _down, 3.2, RayMask.Solid);
    if (hit) {
      this.queueDecal(hit.point, hit.normal, DECAL.bloodPool, e.headshot ? 1.05 : 0.8, 1.4, 55, 0.92, false, 0);
    }
  };

  private readonly onFootstep = (e: { position: THREE.Vector3; surface: SurfaceKind; running: boolean }): void => {
    const p = this.particles;
    if (!p) return;
    const profile = SURFACES[e.surface] ?? SURFACES.concrete;
    if (profile.dustAmount < 0.5 && !e.running) return;
    const n = Math.max(1, Math.round((e.running ? 3 : 2) * profile.dustAmount * this.density));
    for (let i = 0; i < n; i++) {
      const w = p.write();
      w.pos.copy(e.position);
      w.pos.x += (this.rng() - 0.5) * 0.24;
      w.pos.z += (this.rng() - 0.5) * 0.24;
      w.pos.y += 0.02 + this.rng() * 0.05;
      w.vel.set((this.rng() - 0.5) * 0.7, 0.25 + this.rng() * 0.5, (this.rng() - 0.5) * 0.7);
      w.life = 0.7 + this.rng() * 0.8;
      w.drag = 2.4;
      w.gravity = 0.08;
      w.size0 = 0.06;
      w.size1 = 0.34 + this.rng() * 0.2;
      w.sizeCurve = 0.55;
      w.rotRate = (this.rng() - 0.5) * 1.4;
      w.cell = CELL.dust;
      w.turbulence = 0.12;
      w.colorA.copy(profile.dust);
      w.colorB.copy(profile.dust).multiplyScalar(0.8);
      w.alphaA = (e.running ? 0.2 : 0.12) * profile.dustAmount;
      w.alphaB = 0;
      p.commit();
    }
  };

  private readonly onLand = (e: { position: THREE.Vector3; impact: number; surface: SurfaceKind }): void => {
    if (e.impact < 0.25) return;
    const profile = SURFACES[e.surface] ?? SURFACES.concrete;
    const p = this.particles;
    if (!p) return;
    const n = Math.max(2, Math.round(8 * e.impact * profile.dustAmount * this.density));
    for (let i = 0; i < n; i++) {
      const a = this.rng() * Math.PI * 2;
      const speed = 1.1 + this.rng() * 2.2 * e.impact;
      const w = p.write();
      w.pos.copy(e.position);
      w.pos.y += 0.04;
      w.vel.set(Math.cos(a) * speed, 0.35 + this.rng() * 0.5, Math.sin(a) * speed);
      w.life = 0.9 + this.rng() * 0.9;
      w.drag = 3.1;
      w.gravity = 0.1;
      w.size0 = 0.08;
      w.size1 = 0.5 + this.rng() * 0.35;
      w.sizeCurve = 0.5;
      w.rotRate = (this.rng() - 0.5) * 1.2;
      w.cell = CELL.plume;
      w.turbulence = 0.18;
      w.colorA.copy(profile.dust);
      w.colorB.copy(profile.dust).multiplyScalar(0.75);
      w.alphaA = 0.22 * profile.dustAmount * e.impact;
      w.alphaB = 0;
      p.commit();
    }
  };

  private readonly onQualityChanged = (): void => {
    const ctx = this.ctx;
    if (!ctx) return;
    const q = ctx.quality;
    this.density = q.preset === 'low' ? 0.45 : q.preset === 'medium' ? 0.72 : q.preset === 'ultra' ? 1.25 : 1;
    this.decals?.setFadeRange(q.preset === 'low' ? 18 : 30, q.preset === 'low' ? 30 : 55);
    this.particles?.setSizeScale(q.preset === 'low' ? 0.9 : 1);
  };

  private readonly onTracerCrack = (x: number, y: number, z: number, distance: number): void => {
    const ctx = this.ctx;
    if (!ctx) return;
    // The audio system owns the sound; the typed bus has no channel for it, so
    // this goes out as a loose event that a listener can opt into.
    (ctx.events as unknown as LooseEmitter).emit('fx.tracer.crack', {
      position: { x, y, z },
      distance,
    });
    if (distance < 2.2) {
      ctx.events.emit('camera.shake', { amount: 0.045 * (1 - distance / 2.2), duration: 0.1, frequency: 34 });
    }
  };

  // -------------------------------------------------------------------------
  // Public effect API (other systems reach this via ctx.system('fx'))
  // -------------------------------------------------------------------------

  impact(point: THREE.Vector3, normal: THREE.Vector3, direction: THREE.Vector3, surface: SurfaceKind, energy: number): void {
    const p = this.particles;
    if (!p) return;
    const profile = SURFACES[surface] ?? SURFACES.concrete;
    const e = THREE.MathUtils.clamp(energy, 0.15, 1);
    const scale = this.density * (0.55 + e * 0.65);

    orthoBasis(normal, _tangent, _bitangent);
    // Ricochet direction: the incoming ray mirrored about the surface normal,
    // biased back along the normal so ejecta leaves the wall.
    _v4.copy(direction).reflect(normal).normalize().addScaledVector(normal, 0.55).normalize();

    switch (surface) {
      case 'water':
        this.waterSplash(point, normal, e);
        break;
      case 'flesh':
        this.bloodBurst(point, direction, e);
        break;
      case 'glass':
        this.glassShatter(point, normal, _v4, e, profile, scale);
        break;
      default:
        this.genericImpact(point, normal, _v4, e, profile, scale, surface);
        break;
    }

    if (profile.sparkAmount > 0.05) this.sparkShower(point, _v4, e, profile.sparkAmount, scale);

    if (surface !== 'flesh' && surface !== 'water' && surface !== 'foliage') {
      if (this.decalsThisFrame < MAX_DECALS_PER_FRAME) {
        this.decalsThisFrame++;
        const size = profile.holeSize * (0.78 + this.rng() * 0.5) * (0.7 + e * 0.45);
        this.decals?.add({
          point,
          normal,
          cell: decalCellForSurface(surface),
          size,
          depth: Math.max(0.06, size * 1.1),
          life: 0,
          parallax: surface === 'glass' ? 0.008 : 0.028,
          opacity: 0.75 + e * 0.25,
        });
      }
    }
  }

  explosion(point: THREE.Vector3, radius: number): void {
    const ctx = this.ctx;
    const p = this.particles;
    if (!ctx || !p) return;
    const r = Math.max(1, radius);
    const d = this.density;

    // 1. Core flash — white hot, gone in ~90ms.
    for (let i = 0; i < 2; i++) {
      const w = p.write();
      w.additive = true;
      w.pos.copy(point);
      w.vel.set(0, 0, 0);
      w.life = 0.075 + i * 0.045;
      w.drag = 4;
      w.gravity = 0;
      w.size0 = r * (0.5 + i * 0.4);
      w.size1 = r * (1.7 + i * 1.3);
      w.sizeCurve = 0.45;
      w.cell = i === 0 ? CELL.flash : CELL.glow;
      w.colorA.setRGB(9, 8.2, 6.4);
      w.colorB.copy(FIRE_MID).multiplyScalar(3.2);
      w.alphaA = 1;
      w.alphaB = 0;
      w.rotRate = (this.rng() - 0.5) * 3;
      p.commit();
    }

    // 2. Fireball — staged puffs so it blooms outward instead of appearing.
    const fireCount = Math.max(5, Math.round(13 * d));
    for (let i = 0; i < fireCount; i++) {
      const w = p.write();
      w.additive = true;
      w.pos.copy(point);
      randomInSphere(this.rng, _v1, r * 0.32);
      w.pos.add(_v1);
      randomInSphere(this.rng, _v2, 1);
      w.vel.copy(_v2).multiplyScalar(r * (1.4 + this.rng() * 2.6));
      w.vel.y += r * 0.9;
      w.delay = this.rng() * 0.085;
      w.life = 0.42 + this.rng() * 0.55;
      w.drag = 3.4;
      w.gravity = -0.32;
      w.size0 = r * 0.34;
      w.size1 = r * (1.05 + this.rng() * 0.7);
      w.sizeCurve = 0.62;
      w.rotRate = (this.rng() - 0.5) * 2.6;
      w.cell = CELL.fire;
      w.turbulence = r * 0.5;
      w.colorA.copy(FIRE_CORE).multiplyScalar(5.5 + this.rng() * 3);
      w.colorB.copy(FIRE_COLD).multiplyScalar(0.8);
      w.alphaA = 0.95;
      w.alphaB = 0;
      p.commit();
    }

    // 3. Ground-hugging dust shockwave.
    const ringCount = Math.max(8, Math.round(20 * d));
    for (let i = 0; i < ringCount; i++) {
      const a = (i / ringCount) * Math.PI * 2 + this.rng() * 0.3;
      const speed = r * (2.6 + this.rng() * 1.8);
      const w = p.write();
      w.pos.copy(point);
      w.pos.y += 0.12;
      w.vel.set(Math.cos(a) * speed, 0.4 + this.rng() * 0.9, Math.sin(a) * speed);
      w.life = 1.5 + this.rng() * 1.4;
      w.drag = 2.9;
      w.gravity = 0.05;
      w.size0 = r * 0.28;
      w.size1 = r * (1.5 + this.rng() * 0.8);
      w.sizeCurve = 0.45;
      w.rotRate = (this.rng() - 0.5) * 0.9;
      w.cell = CELL.plume;
      w.turbulence = 0.3;
      w.colorA.copy(SMOKE_LIGHT);
      w.colorB.copy(SMOKE_LIGHT).multiplyScalar(0.62);
      w.alphaA = 0.4;
      w.alphaB = 0;
      p.commit();
    }

    // Fast bright ring that sells the pressure front.
    {
      const w = p.write();
      w.additive = true;
      w.pos.copy(point);
      w.pos.y += 0.1;
      w.life = 0.32;
      w.drag = 6;
      w.gravity = 0;
      w.size0 = r * 0.5;
      w.size1 = r * 3.1;
      w.sizeCurve = 0.5;
      w.cell = CELL.ring;
      w.colorA.copy(FIRE_MID).multiplyScalar(2.4);
      w.colorB.copy(SMOKE_LIGHT).multiplyScalar(0.4);
      w.alphaA = 0.85;
      w.alphaB = 0;
      p.commit();
    }

    // 4. Lingering black smoke that rises and thins.
    const smokeCount = Math.max(5, Math.round(12 * d));
    for (let i = 0; i < smokeCount; i++) {
      randomInSphere(this.rng, _v1, r * 0.5);
      const w = p.write();
      w.pos.copy(point).add(_v1);
      w.vel.set((this.rng() - 0.5) * r * 0.8, 0.9 + this.rng() * 1.9, (this.rng() - 0.5) * r * 0.8);
      w.delay = 0.08 + this.rng() * 0.35;
      w.life = 3.4 + this.rng() * 3;
      w.drag = 1.1;
      w.gravity = -0.09;
      w.size0 = r * 0.4;
      w.size1 = r * (2.1 + this.rng() * 1.2);
      w.sizeCurve = 0.5;
      w.rotRate = (this.rng() - 0.5) * 0.5;
      w.cell = i % 3 === 0 ? CELL.smoke : CELL.smokeDense;
      w.turbulence = 0.34;
      w.colorA.copy(SMOKE_DARK);
      w.colorB.copy(SMOKE_LIGHT).multiplyScalar(0.85);
      w.alphaA = 0.82;
      w.alphaB = 0;
      p.commit();
    }

    // 5. Embers riding the plume.
    const emberCount = Math.max(6, Math.round(26 * d));
    for (let i = 0; i < emberCount; i++) {
      randomInSphere(this.rng, _v1, 1);
      const w = p.write();
      w.additive = true;
      w.pos.copy(point);
      w.vel.copy(_v1).multiplyScalar(r * (1.6 + this.rng() * 3.4));
      w.vel.y += r * 1.2;
      w.life = 0.9 + this.rng() * 1.9;
      w.drag = 1.3;
      w.gravity = 0.55;
      w.size0 = 0.05 + this.rng() * 0.05;
      w.size1 = 0.012;
      w.sizeCurve = 1.4;
      w.cell = CELL.ember;
      w.stretch = 0.02;
      w.turbulence = 0.55;
      w.colorA.copy(SPARK_HOT).multiplyScalar(6);
      w.colorB.copy(SPARK_COLD).multiplyScalar(1.2);
      w.alphaA = 1;
      w.alphaB = 0;
      p.commit();
    }

    // 6. Light spike.
    this.spawnLight(point, r);

    // 7. Physical debris.
    const chunks = Math.max(4, Math.round(14 * d));
    for (let i = 0; i < chunks; i++) {
      randomInSphere(this.rng, _v1, 1);
      _v1.y = Math.abs(_v1.y) * 0.9 + 0.35;
      _v2.copy(point).addScaledVector(_v1, r * 0.2);
      _v3.copy(_v1).multiplyScalar(r * (1.6 + this.rng() * 3.2));
      _color.copy(SMOKE_LIGHT).multiplyScalar(0.5 + this.rng() * 0.4);
      this.spawnDebris(_v2, _v3, 0.055 + this.rng() * 0.13, _color, 7 + this.rng() * 6);
    }

    // 8. Scorch on the ground beneath the blast.
    _v1.copy(point);
    _v1.y += 0.35;
    const ground = ctx.physics.raycast(_v1, _down, r * 1.4, RayMask.Solid);
    if (ground) {
      this.queueDecal(
        ground.point, ground.normal, r > 3 ? DECAL.scorchLarge : DECAL.scorchSmall,
        r * (1.1 + this.rng() * 0.35), 0, 0, 0.9, false, 0,
      );
    }

    // 9. Heat haze.
    this.spawnShimmer(point, r * 0.9, r * 3.4, 0.55, 0.62);

    // 10. Camera shake, falling off with distance.
    const dist = ctx.camera.position.distanceTo(point);
    const falloff = THREE.MathUtils.clamp(1 - dist / (r * 6 + 6), 0, 1);
    if (falloff > 0.01) {
      ctx.events.emit('camera.shake', {
        amount: 0.28 + 1.05 * falloff * falloff,
        duration: 0.32 + falloff * 0.45,
        frequency: 26 - falloff * 8,
      });
    }
  }

  /** Fine blood mist with a tight falloff, plus a wall splatter behind the hit. */
  bloodBurst(point: THREE.Vector3, direction: THREE.Vector3, amount: number): void {
    const p = this.particles;
    const ctx = this.ctx;
    if (!p || !ctx) return;
    const n = Math.max(4, Math.round(22 * amount * this.density));
    _v1.copy(direction).normalize();
    for (let i = 0; i < n; i++) {
      p.cone(_v2, _v1, 0.95, 1.6 + this.rng() * 4.5 * amount, this.rng);
      const w = p.write();
      w.pos.copy(point);
      w.vel.copy(_v2);
      w.vel.y += 0.7;
      w.life = 0.28 + this.rng() * 0.5;
      // Mist decelerates hard: droplets are tiny, so drag dominates gravity.
      w.drag = 5.5;
      w.gravity = 0.85;
      w.size0 = 0.03 + this.rng() * 0.05;
      w.size1 = 0.12 + this.rng() * 0.14;
      w.sizeCurve = 0.6;
      w.rotRate = (this.rng() - 0.5) * 5;
      w.cell = CELL.blood;
      w.turbulence = 0.2;
      w.colorA.copy(BLOOD_COLOR).multiplyScalar(1.25);
      w.colorB.copy(BLOOD_COLOR).multiplyScalar(0.6);
      w.alphaA = 0.72;
      w.alphaB = 0;
      p.commit();
    }

    // Heavier droplets that arc and fall.
    const drops = Math.max(2, Math.round(7 * amount * this.density));
    for (let i = 0; i < drops; i++) {
      p.cone(_v2, _v1, 1.1, 2.5 + this.rng() * 4, this.rng);
      const w = p.write();
      w.pos.copy(point);
      w.vel.copy(_v2);
      w.vel.y += 1.6;
      w.life = 0.6 + this.rng() * 0.6;
      w.drag = 0.4;
      w.gravity = 1;
      w.size0 = 0.035;
      w.size1 = 0.02;
      w.sizeCurve = 1;
      w.cell = CELL.droplet;
      w.stretch = 0.012;
      w.colorA.copy(BLOOD_COLOR).multiplyScalar(1.1);
      w.colorB.copy(BLOOD_COLOR).multiplyScalar(0.75);
      w.alphaA = 0.9;
      w.alphaB = 0.5;
      p.commit();
    }

    // Splatter on whatever is behind the target.
    if (amount > 0.5 && this.decalsThisFrame < MAX_DECALS_PER_FRAME) {
      _v3.copy(point).addScaledVector(_v1, 0.15);
      const hit = ctx.physics.raycast(_v3, _v1, 3.4, RayMask.Solid);
      if (hit) {
        this.decalsThisFrame++;
        this.decals?.add({
          point: hit.point,
          normal: hit.normal,
          cell: this.rng() < 0.5 ? DECAL.bloodA : DECAL.bloodB,
          size: 0.5 + amount * 0.75,
          depth: 0.35,
          alignUp: true,
          dripLength: 0.44,
          life: 60,
          opacity: 0.85,
          parallax: 0.006,
        });
      }
    }
  }

  // -------------------------------------------------------------------------
  // Impact composites
  // -------------------------------------------------------------------------

  private genericImpact(
    point: THREE.Vector3, normal: THREE.Vector3, ricochet: THREE.Vector3,
    energy: number, profile: SurfaceProfile, scale: number, surface: SurfaceKind,
  ): void {
    const p = this.particles!;

    // Spall flash: the instant of contact, one frame of white.
    if (profile.sparkAmount > 0.02 || surface === 'concrete' || surface === 'plaster') {
      const w = p.write();
      w.additive = true;
      w.pos.copy(point).addScaledVector(normal, 0.02);
      w.life = 0.045 + this.rng() * 0.03;
      w.drag = 6;
      w.gravity = 0;
      w.size0 = 0.1 + energy * 0.16;
      w.size1 = 0.32 + energy * 0.3;
      w.sizeCurve = 0.5;
      w.cell = CELL.flash;
      w.rotRate = (this.rng() - 0.5) * 8;
      w.colorA.copy(SPARK_HOT).multiplyScalar(3.4 + energy * 3);
      w.colorB.copy(SPARK_MID).multiplyScalar(0.6);
      w.alphaA = 0.9;
      w.alphaB = 0;
      p.commit();
    }

    // Dust puff, pushed out along the surface normal and drifting up.
    const dustCount = Math.max(2, Math.round(6 * profile.dustAmount * scale));
    for (let i = 0; i < dustCount; i++) {
      p.cone(_v1, normal, 0.85, 0.7 + this.rng() * 2.1 * energy, this.rng);
      const w = p.write();
      w.pos.copy(point).addScaledVector(normal, 0.02 + this.rng() * 0.05);
      w.vel.copy(_v1);
      w.vel.y += 0.35;
      w.life = 0.55 + this.rng() * 1.05;
      w.drag = 3.2;
      w.gravity = 0.1;
      w.size0 = 0.04 + this.rng() * 0.04;
      w.size1 = (0.28 + this.rng() * 0.34) * (0.6 + profile.dustAmount * 0.6);
      w.sizeCurve = 0.5;
      w.rotRate = (this.rng() - 0.5) * 2.2;
      w.cell = i % 3 === 0 ? CELL.smoke : CELL.dust;
      w.turbulence = 0.16;
      w.colorA.copy(profile.dust);
      w.colorB.copy(profile.dust).multiplyScalar(0.72);
      w.alphaA = profile.smokeAlpha * (0.6 + energy * 0.6);
      w.alphaB = 0;
      p.commit();
    }

    // Chips / splinters: solid fragments with real ballistic arcs.
    const chipCount = Math.max(2, Math.round(9 * profile.chipAmount * scale));
    for (let i = 0; i < chipCount; i++) {
      p.cone(_v1, ricochet, 0.85, 2.4 + this.rng() * 7 * energy, this.rng);
      const w = p.write();
      w.pos.copy(point).addScaledVector(normal, 0.015);
      w.vel.copy(_v1);
      w.life = 0.5 + this.rng() * 0.9;
      w.drag = 0.55;
      w.gravity = 1;
      w.size0 = 0.018 + this.rng() * 0.03;
      w.size1 = 0.012 + this.rng() * 0.02;
      w.sizeCurve = 1;
      w.rotRate = (this.rng() - 0.5) * 22;
      w.cell = profile.chipCell;
      w.colorA.copy(profile.debris);
      w.colorB.copy(profile.debris).multiplyScalar(0.75);
      w.alphaA = 1;
      w.alphaB = 0.85;
      p.commit();
    }

    // A few grains linger and settle: this is what makes dust feel heavy.
    const settleCount = Math.max(1, Math.round(3 * profile.dustAmount * scale));
    for (let i = 0; i < settleCount; i++) {
      p.cone(_v1, normal, 1.2, 0.25 + this.rng() * 0.6, this.rng);
      const w = p.write();
      w.pos.copy(point).addScaledVector(normal, 0.03);
      w.vel.copy(_v1);
      w.delay = 0.05 + this.rng() * 0.18;
      w.life = 1.5 + this.rng() * 1.6;
      w.drag = 1.6;
      w.gravity = 0.32;
      w.size0 = 0.1;
      w.size1 = 0.42 + this.rng() * 0.3;
      w.sizeCurve = 0.6;
      w.rotRate = (this.rng() - 0.5) * 0.8;
      w.cell = CELL.smoke;
      w.turbulence = 0.14;
      w.colorA.copy(profile.dust).multiplyScalar(0.9);
      w.colorB.copy(profile.dust).multiplyScalar(0.6);
      w.alphaA = profile.smokeAlpha * 0.42;
      w.alphaB = 0;
      p.commit();
    }
  }

  private sparkShower(point: THREE.Vector3, ricochet: THREE.Vector3, energy: number, amount: number, scale: number): void {
    const p = this.particles!;
    const count = Math.max(4, Math.round(26 * amount * scale * (0.5 + energy)));
    for (let i = 0; i < count; i++) {
      p.cone(_v1, ricochet, 0.75, 5 + this.rng() * 16 * (0.4 + energy), this.rng);
      const w = p.write();
      w.additive = true;
      w.pos.copy(point);
      w.vel.copy(_v1);
      w.life = 0.22 + this.rng() * 0.55;
      w.drag = 1.5;
      w.gravity = 1.05;
      w.size0 = 0.035 + this.rng() * 0.03;
      w.size1 = 0.006;
      w.sizeCurve = 1.7;
      w.cell = CELL.spark;
      // Velocity stretch turns the point sprite into a streak whose length
      // tracks how fast it is actually moving on screen.
      w.stretch = 0.028;
      w.colorA.copy(SPARK_HOT).multiplyScalar(7 + this.rng() * 6);
      w.colorB.copy(SPARK_COLD).multiplyScalar(1.4);
      w.alphaA = 1;
      w.alphaB = 0;
      p.commit();
    }

    // Bounce sparks: staged into the future from the impact point, which reads
    // as the first shower skipping off the surface.
    const bounces = Math.max(2, Math.round(count * 0.35));
    for (let i = 0; i < bounces; i++) {
      p.cone(_v1, ricochet, 1.25, 1.8 + this.rng() * 5, this.rng);
      const w = p.write();
      w.additive = true;
      w.pos.copy(point).addScaledVector(_v1, 0.12);
      w.vel.copy(_v1).multiplyScalar(0.55);
      w.vel.y += 0.6;
      w.delay = 0.06 + this.rng() * 0.22;
      w.life = 0.25 + this.rng() * 0.4;
      w.drag = 2.2;
      w.gravity = 1.1;
      w.size0 = 0.025;
      w.size1 = 0.005;
      w.sizeCurve = 1.6;
      w.cell = CELL.spark;
      w.stretch = 0.022;
      w.colorA.copy(SPARK_MID).multiplyScalar(4);
      w.colorB.copy(SPARK_COLD).multiplyScalar(0.9);
      w.alphaA = 1;
      w.alphaB = 0;
      p.commit();
    }

    // The impact glow: white, cooling through orange to red as it dies.
    const w = p.write();
    w.additive = true;
    w.pos.copy(point);
    w.life = 0.3 + energy * 0.25;
    w.drag = 5;
    w.gravity = 0;
    w.size0 = 0.11 + energy * 0.12;
    w.size1 = 0.03;
    w.sizeCurve = 1.3;
    w.cell = CELL.glow;
    w.colorA.setRGB(6.5, 5.4, 3.6);
    w.colorB.copy(SPARK_COLD).multiplyScalar(1.1);
    w.alphaA = 1;
    w.alphaB = 0;
    p.commit();
  }

  private glassShatter(
    point: THREE.Vector3, normal: THREE.Vector3, ricochet: THREE.Vector3,
    energy: number, profile: SurfaceProfile, scale: number,
  ): void {
    const p = this.particles!;

    // Expanding fracture ring on the pane.
    const ring = p.write();
    ring.additive = true;
    ring.pos.copy(point).addScaledVector(normal, 0.005);
    ring.life = 0.22;
    ring.drag = 6;
    ring.gravity = 0;
    ring.size0 = 0.05;
    ring.size1 = 0.5 + energy * 0.5;
    ring.sizeCurve = 0.5;
    ring.cell = CELL.ring;
    ring.colorA.setRGB(2.4, 2.7, 3.1);
    ring.colorB.setRGB(0.2, 0.24, 0.3);
    ring.alphaA = 0.85;
    ring.alphaB = 0;
    p.commit();

    const shards = Math.max(4, Math.round(18 * scale));
    for (let i = 0; i < shards; i++) {
      p.cone(_v1, ricochet, 1.0, 2.5 + this.rng() * 8 * energy, this.rng);
      const w = p.write();
      w.pos.copy(point);
      w.vel.copy(_v1);
      w.life = 0.7 + this.rng() * 0.9;
      w.drag = 0.35;
      w.gravity = 1;
      w.size0 = 0.03 + this.rng() * 0.05;
      w.size1 = 0.025 + this.rng() * 0.04;
      w.sizeCurve = 1;
      w.rotRate = (this.rng() - 0.5) * 18;
      w.cell = CELL.shard;
      w.colorA.copy(profile.debris).multiplyScalar(1.6);
      w.colorB.copy(profile.debris).multiplyScalar(1.1);
      w.alphaA = 0.9;
      w.alphaB = 0.55;
      p.commit();
    }

    const dust = Math.max(2, Math.round(5 * scale));
    for (let i = 0; i < dust; i++) {
      p.cone(_v1, normal, 0.9, 0.6 + this.rng() * 1.6, this.rng);
      const w = p.write();
      w.pos.copy(point);
      w.vel.copy(_v1);
      w.life = 0.5 + this.rng() * 0.5;
      w.drag = 4;
      w.gravity = 0.35;
      w.size0 = 0.03;
      w.size1 = 0.18 + this.rng() * 0.14;
      w.sizeCurve = 0.55;
      w.cell = CELL.dust;
      w.colorA.copy(profile.dust);
      w.colorB.copy(profile.dust).multiplyScalar(0.7);
      w.alphaA = 0.2;
      w.alphaB = 0;
      p.commit();
    }
  }

  private waterSplash(point: THREE.Vector3, normal: THREE.Vector3, energy: number): void {
    const p = this.particles!;
    const scale = this.density;

    // Crown: a ring of droplets thrown up and outward from the entry point.
    const crown = Math.max(6, Math.round(20 * scale));
    for (let i = 0; i < crown; i++) {
      const a = (i / crown) * Math.PI * 2 + this.rng() * 0.4;
      const out = 0.9 + this.rng() * 1.9 * energy;
      const w = p.write();
      w.pos.copy(point);
      w.pos.x += Math.cos(a) * 0.04;
      w.pos.z += Math.sin(a) * 0.04;
      w.vel.set(Math.cos(a) * out, 2.6 + this.rng() * 3.4 * energy, Math.sin(a) * out);
      w.life = 0.42 + this.rng() * 0.5;
      w.drag = 0.9;
      w.gravity = 1;
      w.size0 = 0.03 + this.rng() * 0.04;
      w.size1 = 0.02;
      w.sizeCurve = 1.2;
      w.cell = CELL.droplet;
      w.stretch = 0.014;
      w.rotRate = (this.rng() - 0.5) * 3;
      w.colorA.setRGB(0.62, 0.72, 0.78);
      w.colorB.setRGB(0.4, 0.5, 0.56);
      w.alphaA = 0.75;
      w.alphaB = 0;
      p.commit();
    }

    // Central column.
    for (let i = 0; i < Math.max(2, Math.round(5 * scale)); i++) {
      const w = p.write();
      w.pos.copy(point);
      w.vel.set((this.rng() - 0.5) * 0.5, 2.2 + this.rng() * 2.4 * energy, (this.rng() - 0.5) * 0.5);
      w.life = 0.5 + this.rng() * 0.4;
      w.drag = 2.4;
      w.gravity = 0.9;
      w.size0 = 0.07;
      w.size1 = 0.22;
      w.sizeCurve = 0.6;
      w.cell = CELL.wisp;
      w.colorA.setRGB(0.72, 0.8, 0.84);
      w.colorB.setRGB(0.5, 0.58, 0.62);
      w.alphaA = 0.5;
      w.alphaB = 0;
      p.commit();
    }

    // Mist hanging over the surface.
    for (let i = 0; i < Math.max(2, Math.round(4 * scale)); i++) {
      const w = p.write();
      w.pos.copy(point);
      w.pos.y += 0.05;
      w.vel.set((this.rng() - 0.5) * 0.9, 0.35 + this.rng() * 0.5, (this.rng() - 0.5) * 0.9);
      w.life = 0.9 + this.rng() * 0.8;
      w.drag = 3.4;
      w.gravity = 0.05;
      w.size0 = 0.08;
      w.size1 = 0.45;
      w.sizeCurve = 0.5;
      w.cell = CELL.smoke;
      w.turbulence = 0.12;
      w.colorA.setRGB(0.66, 0.72, 0.76);
      w.colorB.setRGB(0.5, 0.56, 0.6);
      w.alphaA = 0.22;
      w.alphaB = 0;
      p.commit();
    }

    // Expanding ripple that flattens onto the water.
    const ripple = p.write();
    ripple.pos.copy(point).addScaledVector(normal, 0.01);
    ripple.life = 0.85;
    ripple.drag = 4;
    ripple.gravity = 0;
    ripple.size0 = 0.1;
    ripple.size1 = 1.2 + energy * 0.8;
    ripple.sizeCurve = 0.5;
    ripple.cell = CELL.ring;
    ripple.colorA.setRGB(0.72, 0.8, 0.86);
    ripple.colorB.setRGB(0.45, 0.52, 0.58);
    ripple.alphaA = 0.5;
    ripple.alphaB = 0;
    p.commit();
  }

  private muzzleSmoke(origin: THREE.Vector3, direction: THREE.Vector3, local: boolean): void {
    const p = this.particles;
    if (!p) return;
    const heatBoost = local ? 1 + this.muzzleHeat * 1.6 : 1;
    const count = Math.max(1, Math.round(2.5 * this.density * heatBoost));
    for (let i = 0; i < count; i++) {
      p.cone(_v1, direction, 0.42, 1.1 + this.rng() * 2.4, this.rng);
      const w = p.write();
      w.pos.copy(origin).addScaledVector(direction, 0.04 + this.rng() * 0.1);
      w.vel.copy(_v1);
      w.vel.y += 0.35;
      w.delay = this.rng() * 0.03;
      w.life = 0.55 + this.rng() * 1.1 * heatBoost;
      w.drag = 3.6;
      w.gravity = -0.06;
      w.size0 = 0.035;
      w.size1 = 0.24 + this.rng() * 0.3 * heatBoost;
      w.sizeCurve = 0.5;
      w.rotRate = (this.rng() - 0.5) * 2.4;
      w.cell = i % 2 === 0 ? CELL.wisp : CELL.smoke;
      w.turbulence = 0.22;
      w.colorA.copy(SMOKE_LIGHT).multiplyScalar(0.95);
      w.colorB.copy(SMOKE_LIGHT).multiplyScalar(0.6);
      w.alphaA = 0.11 + 0.09 * this.muzzleHeat;
      w.alphaB = 0;
      p.commit();
    }

    // Unburnt powder flecks out of the barrel.
    const flecks = Math.max(1, Math.round(4 * this.density));
    for (let i = 0; i < flecks; i++) {
      p.cone(_v1, direction, 0.3, 6 + this.rng() * 9, this.rng);
      const w = p.write();
      w.additive = true;
      w.pos.copy(origin);
      w.vel.copy(_v1);
      w.life = 0.1 + this.rng() * 0.22;
      w.drag = 3.5;
      w.gravity = 0.8;
      w.size0 = 0.022;
      w.size1 = 0.004;
      w.sizeCurve = 1.5;
      w.cell = CELL.spark;
      w.stretch = 0.02;
      w.colorA.copy(SPARK_HOT).multiplyScalar(5);
      w.colorB.copy(SPARK_MID).multiplyScalar(1.2);
      w.alphaA = 1;
      w.alphaB = 0;
      p.commit();
    }
  }

  // -------------------------------------------------------------------------
  // Pools
  // -------------------------------------------------------------------------

  private spawnLight(point: THREE.Vector3, radius: number): void {
    let slot: ExplosionLight | null = null;
    for (const l of this.lights) {
      if (!l.active) { slot = l; break; }
    }
    if (!slot) {
      // Steal the oldest.
      let oldest = this.lights[0]!;
      for (const l of this.lights) if (l.age > oldest.age) oldest = l;
      slot = oldest;
    }
    slot.active = true;
    slot.age = 0;
    // Candela: a 5m fireball is genuinely blinding at night and still reads at
    // noon, so the peak is deliberately far above any level light.
    slot.peak = 2200 * radius * radius;
    slot.linger = 0.75;
    slot.range = radius * 9;
    slot.light.position.copy(point);
    slot.light.distance = slot.range;
    slot.light.color.copy(FIRE_CORE);
    slot.light.intensity = slot.peak;
    slot.light.visible = true;
  }

  private spawnDebris(
    position: THREE.Vector3, velocity: THREE.Vector3, size: number, tint: THREE.Color, life: number,
  ): void {
    const mesh = this.debrisMesh;
    if (!mesh || this.debris.length === 0) return;
    const i = this.debrisCursor;
    this.debrisCursor = (i + 1) % this.debris.length;
    const d = this.debris[i]!;
    d.active = true;
    d.pos.copy(position);
    d.vel.copy(velocity);
    d.spin.set((this.rng() - 0.5) * 26, (this.rng() - 0.5) * 26, (this.rng() - 0.5) * 26);
    d.quat.set(this.rng() - 0.5, this.rng() - 0.5, this.rng() - 0.5, this.rng() - 0.5).normalize();
    d.scale = size;
    d.age = 0;
    d.life = life;
    d.resting = false;
    d.bounces = 0;
    d.dust.copy(tint);
    if (mesh.instanceColor) {
      mesh.instanceColor.setXYZ(i, tint.r, tint.g, tint.b);
      mesh.instanceColor.needsUpdate = true;
    }
    if (i >= mesh.count) mesh.count = i + 1;
    mesh.visible = true;
  }

  private queueDecal(
    point: THREE.Vector3, normal: THREE.Vector3, cell: number, size: number,
    delay: number, life: number, opacity: number, alignUp: boolean, drip: number,
  ): void {
    for (const slot of this.pending) {
      if (slot.active) continue;
      slot.active = true;
      slot.at = this.clock + delay;
      slot.point.copy(point);
      slot.normal.copy(normal);
      slot.cell = cell;
      slot.size = size;
      slot.life = life;
      slot.opacity = opacity;
      slot.alignUp = alignUp;
      slot.drip = drip;
      slot.parallax = 0.01;
      return;
    }
  }

  private spawnShimmer(point: THREE.Vector3, r0: number, r1: number, life: number, strength: number): void {
    let slot: ShimmerInstance | null = null;
    for (const s of this.shimmers) {
      if (!s.active) { slot = s; break; }
    }
    if (!slot) slot = this.shimmers[0] ?? null;
    if (!slot) return;
    slot.active = true;
    slot.age = 0;
    slot.life = life;
    slot.radius0 = r0;
    slot.radius1 = r1;
    slot.strength = strength;
    slot.mesh.position.copy(point);
    slot.mesh.visible = true;
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  update(time: FrameTime, ctx: GameContext): void {
    const dt = time.dt;
    this.clock += dt;
    this.decalsThisFrame = 0;

    this.particles?.update(dt);
    this.tracers?.update(dt);
    this.decals?.update(dt, ctx.camera.position);

    this.updatePending();
    this.updateDebris(dt, ctx);
    this.updateLights(dt);
    this.updateShimmer(dt, ctx);

    this.muzzleHeat = Math.max(0, this.muzzleHeat - MUZZLE_HEAT_DECAY * dt);

    this.depthProbeTimer -= dt;
    if (this.depthProbeTimer <= 0) {
      this.depthProbeTimer = 0.5;
      this.syncDepth(ctx);
    }
  }

  lateUpdate(_time: FrameTime, ctx: GameContext): void {
    const p = this.particles;
    if (!p) return;
    const sun = ctx.system<System & SunProvider>('lighting') ?? ctx.system<System & SunProvider>('sky');
    if (sun?.sunDirection && sun.sunColor) {
      _color.copy(sun.sunColor);
      _colorB.copy(sun.zenithColor ?? sun.horizonColor ?? _skyFallback);
      p.setSun(ctx.camera.matrixWorldInverse, sun.sunDirection, _color, _colorB, sun.sunIntensity ?? 3.5);
    } else {
      p.setSun(ctx.camera.matrixWorldInverse, _sunFallbackDir, _sunFallback, _skyFallback, 3.5);
    }
  }

  private updatePending(): void {
    const decals = this.decals;
    if (!decals) return;
    for (const slot of this.pending) {
      if (!slot.active || this.clock < slot.at) continue;
      slot.active = false;
      if (this.decalsThisFrame >= MAX_DECALS_PER_FRAME) {
        // Try again next frame rather than blowing the per-frame clip budget.
        slot.active = true;
        slot.at = this.clock + 0.05;
        continue;
      }
      this.decalsThisFrame++;
      decals.add({
        point: slot.point,
        normal: slot.normal,
        cell: slot.cell,
        size: slot.size,
        depth: Math.max(0.1, slot.size * 0.5),
        life: slot.life,
        opacity: slot.opacity,
        alignUp: slot.alignUp,
        dripLength: slot.drip,
        parallax: slot.parallax,
      });
    }
  }

  private updateDebris(dt: number, ctx: GameContext): void {
    const mesh = this.debrisMesh;
    if (!mesh || mesh.count === 0) return;
    const step = Math.min(dt, 1 / 30);
    let anyActive = false;

    for (let i = 0; i < mesh.count; i++) {
      const d = this.debris[i]!;
      if (!d.active) continue;
      anyActive = true;
      d.age += dt;

      if (!d.resting) {
        d.vel.y -= DEBRIS_GRAVITY * step;
        _v1.copy(d.vel).multiplyScalar(step);
        const travel = _v1.length();
        if (travel > 1e-4) {
          _v2.copy(_v1).multiplyScalar(1 / travel);
          const hit = ctx.physics.raycast(d.pos, _v2, travel + d.scale, RayMask.Solid);
          if (hit) {
            d.pos.copy(hit.point).addScaledVector(hit.normal, d.scale * 0.55);
            const vn = d.vel.dot(hit.normal);
            // Split into normal and tangential: bounce one, scrub the other.
            _v3.copy(hit.normal).multiplyScalar(vn);
            d.vel.sub(_v3.multiplyScalar(1.62)); // restitution 0.62
            _v3.copy(hit.normal).multiplyScalar(d.vel.dot(hit.normal));
            _v4.copy(d.vel).sub(_v3).multiplyScalar(0.62); // friction
            d.vel.copy(_v3).add(_v4);
            d.spin.multiplyScalar(0.55);
            d.bounces++;
            if (d.bounces <= 2 && d.vel.lengthSq() > 4) {
              this.debrisDust(d.pos, hit.normal, d.dust);
            }
            if (d.vel.lengthSq() < 0.5 || d.bounces > 5) {
              d.resting = true;
              d.vel.set(0, 0, 0);
              d.spin.set(0, 0, 0);
            }
          } else {
            d.pos.addScaledVector(_v2, travel);
          }
        }
        if (d.spin.lengthSq() > 1e-6) {
          _quat.setFromAxisAngle(_v3.copy(d.spin).normalize(), d.spin.length() * step);
          d.quat.premultiply(_quat).normalize();
        }
      }

      // Shrink out in the last second instead of vanishing.
      const remaining = d.life - d.age;
      const fade = remaining < 1 ? Math.max(0, remaining) : 1;
      if (remaining <= 0) {
        d.active = false;
        _matrix.makeScale(0, 0, 0);
        mesh.setMatrixAt(i, _matrix);
        continue;
      }
      _scaleVec.setScalar(d.scale * fade);
      _matrix.compose(d.pos, d.quat, _scaleVec);
      mesh.setMatrixAt(i, _matrix);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (!anyActive) {
      mesh.visible = false;
      mesh.count = 0;
      this.debrisCursor = 0;
    }
  }

  private debrisDust(point: THREE.Vector3, normal: THREE.Vector3, tint: THREE.Color): void {
    const p = this.particles;
    if (!p) return;
    for (let i = 0; i < 2; i++) {
      p.cone(_v1, normal, 1.1, 0.4 + this.rng() * 0.9, this.rng);
      const w = p.write();
      w.pos.copy(point);
      w.vel.copy(_v1);
      w.life = 0.5 + this.rng() * 0.5;
      w.drag = 3.5;
      w.gravity = 0.1;
      w.size0 = 0.04;
      w.size1 = 0.24;
      w.sizeCurve = 0.5;
      w.cell = CELL.dust;
      w.colorA.copy(tint);
      w.colorB.copy(tint).multiplyScalar(0.7);
      w.alphaA = 0.18;
      w.alphaB = 0;
      p.commit();
    }
  }

  private updateLights(dt: number): void {
    for (const l of this.lights) {
      if (!l.active) continue;
      l.age += dt;
      // 120ms spike, then a fireball glow that decays over ~0.75s.
      const spike = Math.exp(-l.age / 0.045);
      const glow = Math.exp(-l.age / (l.linger * 0.42));
      const flicker = 0.82 + 0.18 * Math.sin(l.age * 47 + l.peak);
      const intensity = l.peak * (spike * 0.85 + glow * 0.3 * flicker);
      if (l.age > l.linger * 3 || intensity < l.peak * 0.002) {
        l.active = false;
        l.light.intensity = 0;
        l.light.visible = false;
        continue;
      }
      l.light.intensity = intensity;
      // Colour cools with the fireball.
      const t = THREE.MathUtils.clamp(l.age / (l.linger * 1.2), 0, 1);
      l.light.color.copy(FIRE_CORE).lerp(FIRE_MID, t).lerp(FIRE_COLD, t * t);
    }
  }

  private updateShimmer(dt: number, ctx: GameContext): void {
    for (const s of this.shimmers) {
      if (!s.active) continue;
      s.age += dt;
      const u = s.age / s.life;
      if (u >= 1) {
        s.active = false;
        s.mesh.visible = false;
        continue;
      }
      const radius = s.radius0 + (s.radius1 - s.radius0) * Math.pow(u, 0.55);
      s.mesh.scale.setScalar(radius * 2);
      s.mesh.quaternion.copy(ctx.camera.quaternion);
      s.material.uniforms.uTime.value = this.clock;
      s.material.uniforms.uOpacity.value = s.strength * (1 - u) * (1 - u);
    }

    const barrel = this.barrelShimmer;
    if (barrel) {
      const heat = THREE.MathUtils.clamp((this.muzzleHeat - 0.34) / 0.7, 0, 1);
      if (heat <= 0.001) {
        barrel.mesh.visible = false;
      } else {
        barrel.mesh.visible = true;
        // The muzzle offset was captured in main-camera space; the viewmodel
        // camera has a different FOV, so rescale to keep the same screen spot.
        const k = Math.tan(THREE.MathUtils.degToRad(ctx.camera.fov * 0.5))
          / Math.tan(THREE.MathUtils.degToRad(ctx.viewCamera.fov * 0.5));
        _v1.set(this.muzzleLocal.x * k, this.muzzleLocal.y * k, this.muzzleLocal.z);
        _v1.applyQuaternion(ctx.viewCamera.quaternion).add(ctx.viewCamera.position);
        barrel.mesh.position.copy(_v1);
        barrel.mesh.quaternion.copy(ctx.viewCamera.quaternion);
        barrel.mesh.scale.setScalar(0.18 + heat * 0.12);
        barrel.material.uniforms.uTime.value = this.clock;
        barrel.material.uniforms.uOpacity.value = heat * 0.1;
      }
    }
  }

  private resolveDepthTexture(): THREE.Texture | null {
    const ctx = this.ctx;
    if (!ctx) return null;
    const provider = ctx.system<System & DepthProvider>('render');
    if (!provider) return null;
    if (typeof provider.getSceneDepthTexture === 'function') return provider.getSceneDepthTexture();
    if (typeof provider.getDepthTexture === 'function') return provider.getDepthTexture();
    return provider.sceneDepthTexture ?? null;
  }

  private syncDepth(ctx: GameContext): void {
    const p = this.particles;
    if (!p) return;
    const tex = this.resolveDepthTexture();
    if (!tex) return;
    const image = tex.image as { width?: number; height?: number } | undefined;
    p.setDepthTexture(tex, image?.width ?? ctx.width, image?.height ?? ctx.height, ctx.camera.near, ctx.camera.far);
  }

  dispose(): void {
    for (const off of this.unsubscribe) off();
    this.unsubscribe.length = 0;

    this.particles?.dispose();
    this.decals?.dispose();
    this.tracers?.dispose();
    this.particles = null;
    this.decals = null;
    this.tracers = null;

    const scene = this.ctx?.scene;
    if (this.debrisMesh) {
      scene?.remove(this.debrisMesh);
      this.debrisMesh.dispose();
      this.debrisMesh = null;
    }
    this.debrisGeometry?.dispose();
    this.debrisMaterial?.dispose();
    this.debrisTexture?.dispose();
    this.debrisGeometry = null;
    this.debrisMaterial = null;
    this.debrisTexture = null;
    this.debris.length = 0;

    for (const l of this.lights) {
      l.light.parent?.remove(l.light);
      l.light.dispose();
    }
    this.lights.length = 0;

    for (const s of this.shimmers) {
      s.mesh.parent?.remove(s.mesh);
      s.material.dispose();
    }
    this.shimmers.length = 0;
    if (this.barrelShimmer) {
      this.barrelShimmer.mesh.parent?.remove(this.barrelShimmer.mesh);
      this.barrelShimmer.material.dispose();
      this.barrelShimmer = null;
    }
    this.shimmerGeometry?.dispose();
    this.shimmerGeometry = null;
    this.shimmerMaterialTemplate?.dispose();
    this.shimmerMaterialTemplate = null;

    this.pending.length = 0;
    this.ctx = null;
  }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const _tracerSpawn = {
  from: new THREE.Vector3(),
  to: new THREE.Vector3(),
  speed: 900,
  intensity: 1,
  color: new THREE.Color(),
  width: 0.03,
  tail: 0,
};

const _sunFallback = srgb(0xffd9a8);
const _sunFallbackDir = new THREE.Vector3(0.4, 0.78, 0.48).normalize();
const _skyFallback = srgb(0x7ea6d8);

function orthoBasis(normal: THREE.Vector3, tangent: THREE.Vector3, bitangent: THREE.Vector3): void {
  if (Math.abs(normal.y) > 0.94) tangent.set(1, 0, 0);
  else tangent.copy(_up);
  bitangent.crossVectors(normal, tangent).normalize();
  tangent.crossVectors(bitangent, normal).normalize();
}

function randomInSphere(rng: () => number, out: THREE.Vector3, radius: number): THREE.Vector3 {
  // Rejection-free: normalise a gaussian-ish triple, then scale by cbrt for a
  // uniform volume distribution.
  let x = rng() * 2 - 1;
  let y = rng() * 2 - 1;
  let z = rng() * 2 - 1;
  const len = Math.hypot(x, y, z) || 1;
  x /= len; y /= len; z /= len;
  const r = Math.cbrt(rng()) * radius;
  return out.set(x * r, y * r, z * r);
}

/** Grainy rock albedo for debris chunks. */
function buildRockTexture(size: number): THREE.Texture {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const c2d = canvas.getContext('2d');
  if (!c2d) throw new Error('[fx] 2d context unavailable');
  const img = c2d.createImageData(size, size);
  const rng = mulberry32(0x9a71c3);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const n = 0.55 + 0.45 * rng();
      const blotch = 0.75 + 0.35 * Math.sin(x * 0.13 + y * 0.09);
      const v = Math.min(255, 255 * n * blotch * 0.82);
      img.data[i] = v;
      img.data[i + 1] = v * 0.97;
      img.data[i + 2] = v * 0.92;
      img.data[i + 3] = 255;
    }
  }
  c2d.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}
