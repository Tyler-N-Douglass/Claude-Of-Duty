import * as THREE from 'three';
import type { FrameTime, GameContext, System } from '../core/Contracts';
import { SKY_FRAG, SKY_VERT } from './Shaders';

/**
 * Late afternoon. Sun sits at ~19 degrees: long shadows, warm key, cool sky
 * fill, and every vertical surface gets a readable light-to-dark gradient.
 */
export const DEFAULT_TIME_OF_DAY = 17.35;

/** Sun is above the horizon between these hours. */
const DAY_START = 6;
const DAY_END = 20;
const MAX_ELEVATION = 34 * (Math.PI / 180);

export interface SkyParams {
  turbidity: number;
  rayleigh: number;
  mieCoefficient: number;
  mieG: number;
  intensity: number;
  cloudCover: number;
}

// Calibrated against the pipeline's exposure so that after the ACES curve the
// horizon lands near sRGB 0.87 — bright but still holding detail — and the
// zenith near 0.5, giving the sky a real value range instead of a white band.
// Turbidity is pushed past clean air so Mie scatter desaturates the blue into
// something gradeable rather than a cyan poster.
export const DEFAULT_SKY_PARAMS: SkyParams = {
  turbidity: 5.4,
  rayleigh: 1.75,
  mieCoefficient: 0.0058,
  mieG: 0.8,
  intensity: 0.95,
  cloudCover: 0.54,
};

// Preetham fits for sea-level air; identical values drive the GPU dome so the
// CPU-side colours used for fog, IBL tinting and reflections cannot drift.
const TOTAL_RAYLEIGH = [5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5] as const;
const MIE_CONST = [1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14] as const;

const _dir = new THREE.Vector3();
const _tmpColor = new THREE.Color();

/** Direction pointing *towards* the sun, normalised. */
export function sunDirectionForTime(hours: number, out = new THREE.Vector3()): THREE.Vector3 {
  const t = THREE.MathUtils.clamp((hours - DAY_START) / (DAY_END - DAY_START), 0.001, 0.999);
  const elevation = MAX_ELEVATION * Math.sin(Math.PI * t);
  const azimuth = Math.PI * 0.5 + Math.PI * t;
  const ce = Math.cos(elevation);
  return out.set(ce * Math.sin(azimuth), Math.sin(elevation), ce * Math.cos(azimuth)).normalize();
}

function rayleighPhase(c: number): number {
  return (3 / (16 * Math.PI)) * (1 + c * c);
}

function miePhase(c: number, g: number): number {
  const g2 = g * g;
  const inv = 1 / Math.pow(Math.max(1e-4, 1 - 2 * g * c + g2), 1.5);
  return (1 / (4 * Math.PI)) * ((1 - g2) * inv);
}

/**
 * CPU evaluation of the same analytic sky the dome shader renders. Used for
 * fog colour, the SSR/atmosphere fallback uniforms and the hemisphere fill.
 */
export function evaluateSkyRadiance(
  dir: THREE.Vector3,
  sunDir: THREE.Vector3,
  p: SkyParams,
  out = new THREE.Color(),
): THREE.Color {
  const sunE = 1000 * Math.max(0, 1 - Math.exp(-((Math.PI / 2 - Math.acos(THREE.MathUtils.clamp(sunDir.y, -1, 1))) / 1.5)));
  const c = 0.2 * p.turbidity * 10e-18;

  const zenithAngle = Math.acos(Math.max(0, dir.y));
  const denom = Math.cos(zenithAngle) + 0.15 * Math.pow(Math.max(1e-3, 93.885 - (zenithAngle * 180) / Math.PI), -1.253);
  const inverse = 1 / Math.max(1e-4, denom);
  const sR = 8.4e3 * inverse;
  const sM = 1.25e3 * inverse;

  const cosTheta = dir.dot(sunDir);
  const rPhase = rayleighPhase(cosTheta * 0.5 + 0.5);
  const mPhase = miePhase(cosTheta, p.mieG);
  const mixFactor = THREE.MathUtils.clamp(Math.pow(1 - sunDir.y, 5), 0, 1);
  const bias = [0, 0.00035, 0.00085];

  const rgb = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const betaR = TOTAL_RAYLEIGH[i] * p.rayleigh;
    const betaM = 0.434 * c * MIE_CONST[i] * p.mieCoefficient;
    const fex = Math.exp(-(betaR * sR + betaM * sM));
    const base = (betaR * rPhase + betaM * mPhase) / (betaR + betaM);
    let lin = Math.pow(Math.max(0, sunE * base * (1 - fex)), 1.5);
    lin *= 1 + (Math.pow(Math.max(0, sunE * base * fex), 0.5) - 1) * mixFactor;
    rgb[i] = ((lin + 0.1 * fex) * 0.04 + bias[i]) * p.intensity;
  }
  return out.setRGB(Math.max(0, rgb[0]), Math.max(0, rgb[1]), Math.max(0, rgb[2]), THREE.LinearSRGBColorSpace);
}

/** Atmospheric extinction along the sun ray, normalised to a light colour. */
export function sunLightColor(sunDir: THREE.Vector3, p: SkyParams, out = new THREE.Color()): THREE.Color {
  const zenithAngle = Math.acos(THREE.MathUtils.clamp(sunDir.y, -1, 1));
  const denom = Math.cos(zenithAngle) + 0.15 * Math.pow(Math.max(1e-3, 93.885 - (zenithAngle * 180) / Math.PI), -1.253);
  const inverse = 1 / Math.max(1e-4, denom);
  const sR = 8.4e3 * inverse;
  const sM = 1.25e3 * inverse;
  const c = 0.2 * p.turbidity * 10e-18;

  const rgb = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const betaR = TOTAL_RAYLEIGH[i] * p.rayleigh;
    const betaM = 0.434 * c * MIE_CONST[i] * p.mieCoefficient;
    rgb[i] = Math.exp(-(betaR * sR + betaM * sM));
  }
  const peak = Math.max(rgb[0], rgb[1], rgb[2], 1e-4);
  out.setRGB(rgb[0] / peak, rgb[1] / peak, rgb[2] / peak, THREE.LinearSRGBColorSpace);
  // Physical extinction alone drifts too orange to grade; pull it back toward
  // the art-directed golden-hour key.
  _tmpColor.setHex(0xffd9a8, THREE.SRGBColorSpace);
  return out.lerp(_tmpColor, 0.45);
}

export class SkySystem implements System {
  readonly name = 'sky';

  readonly params: SkyParams = { ...DEFAULT_SKY_PARAMS };
  readonly sunDirection = new THREE.Vector3();

  /** Representative dome colours, in linear space, kept in sync with the shader. */
  readonly zenithColor = new THREE.Color();
  readonly horizonColor = new THREE.Color();
  readonly groundColor = new THREE.Color();
  readonly sunColor = new THREE.Color();
  /** Ambient/IBL scale so the visible dome can be bright without washing PBR. */
  /**
   * IBL weight. The sun runs at 5.0 and a street canyon bounces far more than a
   * hemisphere light alone provides; below ~0.8 every shadowed facade and the
   * whole road surface crush to black.
   */
  environmentIntensity = 0.92;

  private ctx: GameContext | null = null;
  private timeOfDay = DEFAULT_TIME_OF_DAY;
  private material: THREE.ShaderMaterial | null = null;
  private geometry: THREE.SphereGeometry | null = null;
  private domeMesh: THREE.Mesh | null = null;
  private envMesh: THREE.Mesh | null = null;
  private envScene: THREE.Scene | null = null;
  private pmrem: THREE.PMREMGenerator | null = null;
  private envTarget: THREE.WebGLRenderTarget | null = null;
  private fog: THREE.FogExp2 | null = null;
  private envDirty = true;

  init(ctx: GameContext): void {
    this.ctx = ctx;

    sunDirectionForTime(this.timeOfDay, this.sunDirection);
    this.recomputeColors();

    const groundLinear = new THREE.Color().setHex(0x4a4239, THREE.SRGBColorSpace);

    this.material = new THREE.ShaderMaterial({
      name: 'SkyDome',
      uniforms: {
        uSunDir: { value: this.sunDirection.clone() },
        uTurbidity: { value: this.params.turbidity },
        uRayleigh: { value: this.params.rayleigh },
        uMieCoefficient: { value: this.params.mieCoefficient },
        uMieG: { value: this.params.mieG },
        uIntensity: { value: this.params.intensity },
        uTime: { value: 0 },
        uCloudCover: { value: this.params.cloudCover },
        uGroundColor: { value: groundLinear },
        uExposureClamp: { value: 12 },
      },
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      fog: false,
      toneMapped: false,
    });

    this.geometry = new THREE.SphereGeometry(1, 48, 32);

    this.domeMesh = new THREE.Mesh(this.geometry, this.material);
    this.domeMesh.name = 'sky-dome';
    this.domeMesh.scale.setScalar(1200);
    this.domeMesh.frustumCulled = false;
    this.domeMesh.renderOrder = -10000;
    this.domeMesh.matrixAutoUpdate = false;
    this.domeMesh.onBeforeRender = (_r, _s, camera) => {
      // The dome is a direction field, so it simply rides the camera.
      this.domeMesh!.position.copy(camera.position);
      this.domeMesh!.updateMatrix();
      this.domeMesh!.updateMatrixWorld(true);
    };
    ctx.scene.add(this.domeMesh);

    this.envMesh = new THREE.Mesh(this.geometry, this.material);
    this.envMesh.scale.setScalar(1200);
    this.envMesh.frustumCulled = false;
    this.envScene = new THREE.Scene();
    this.envScene.add(this.envMesh);

    this.pmrem = new THREE.PMREMGenerator(ctx.renderer);

    this.fog = new THREE.FogExp2(0x000000, 0.0009);
    this.fog.color.copy(this.horizonColor);
    ctx.scene.fog = this.fog;

    this.applyUniforms();
    this.regenerateEnvironment();
  }

  /** 0..24. Regenerates the IBL and re-derives every dependent colour. */
  setTimeOfDay(hours: number): void {
    const h = ((hours % 24) + 24) % 24;
    if (Math.abs(h - this.timeOfDay) < 1e-4) return;
    this.timeOfDay = h;
    sunDirectionForTime(this.timeOfDay, this.sunDirection);
    this.recomputeColors();
    this.applyUniforms();
    this.envDirty = true;
  }

  getTimeOfDay(): number {
    return this.timeOfDay;
  }

  update(time: FrameTime): void {
    if (this.material) this.material.uniforms.uTime.value = time.elapsed;
    if (this.envDirty) {
      this.envDirty = false;
      this.regenerateEnvironment();
    }
  }

  private applyUniforms(): void {
    const m = this.material;
    if (!m) return;
    (m.uniforms.uSunDir.value as THREE.Vector3).copy(this.sunDirection);
    m.uniforms.uTurbidity.value = this.params.turbidity;
    m.uniforms.uRayleigh.value = this.params.rayleigh;
    m.uniforms.uMieCoefficient.value = this.params.mieCoefficient;
    m.uniforms.uMieG.value = this.params.mieG;
    m.uniforms.uIntensity.value = this.params.intensity;
    m.uniforms.uCloudCover.value = this.params.cloudCover;
    if (this.fog) this.fog.color.copy(this.horizonColor);
  }

  private recomputeColors(): void {
    evaluateSkyRadiance(_dir.set(0, 1, 0), this.sunDirection, this.params, this.zenithColor);

    // Horizon sampled across from the sun so the fog colour is not dominated
    // by the sun's forward-scattering lobe.
    const away = _dir.set(-this.sunDirection.x, 0, -this.sunDirection.z);
    if (away.lengthSq() < 1e-6) away.set(1, 0, 0);
    away.normalize().setY(0.035).normalize();
    evaluateSkyRadiance(away, this.sunDirection, this.params, this.horizonColor);

    sunLightColor(this.sunDirection, this.params, this.sunColor);

    // The dome's lower hemisphere: warm dust lit by the same sun.
    this.groundColor
      .setHex(0x4a4239, THREE.SRGBColorSpace)
      .multiplyScalar(0.35 + 0.65 * Math.max(0, this.sunDirection.y))
      .add(_tmpColor.copy(this.horizonColor).multiplyScalar(0.35));
  }

  private regenerateEnvironment(): void {
    const ctx = this.ctx;
    if (!ctx || !this.pmrem || !this.envScene) return;

    const previous = this.envTarget;
    try {
      this.envTarget = this.pmrem.fromScene(this.envScene, 0, 0.5, 5000);
    } catch (err) {
      console.error('[sky] PMREM generation failed', err);
      this.envTarget = previous;
      return;
    }
    if (previous && previous !== this.envTarget) previous.dispose();

    const tex = this.envTarget.texture;
    ctx.environment = tex;
    ctx.scene.environment = tex;
    ctx.scene.environmentIntensity = this.environmentIntensity;
  }

  dispose(): void {
    const ctx = this.ctx;
    if (ctx) {
      if (this.domeMesh) ctx.scene.remove(this.domeMesh);
      if (ctx.scene.environment === this.envTarget?.texture) ctx.scene.environment = null;
      if (ctx.environment === this.envTarget?.texture) ctx.environment = null;
      if (ctx.scene.fog === this.fog) ctx.scene.fog = null;
    }
    if (this.envMesh && this.envScene) this.envScene.remove(this.envMesh);
    this.envTarget?.dispose();
    this.pmrem?.dispose();
    this.geometry?.dispose();
    this.material?.dispose();
    this.envTarget = null;
    this.pmrem = null;
    this.geometry = null;
    this.material = null;
    this.domeMesh = null;
    this.envMesh = null;
    this.envScene = null;
  }
}
