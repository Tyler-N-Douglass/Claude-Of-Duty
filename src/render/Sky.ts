import * as THREE from 'three';
import type { FrameTime, GameContext, System } from '../core/Contracts';
import { SKY_FRAG, SKY_VERT } from './Shaders';

/**
 * Late afternoon. At this hour the rig below puts the sun at ~21 degrees of
 * elevation and an azimuth of ~154 degrees — which, on this map, is 26 degrees
 * off the main street's axis. That number is the whole art direction:
 *
 *  - The street runs along Z between facades ~12m tall and ~14m apart. A 21
 *    degree sun throws a 31m shadow; 26 degrees off-axis puts 13.7m of that
 *    across the street, so the east terrace shades almost the full carriageway
 *    but leaves a lit strip at the west kerb and floods every alley mouth and
 *    roofline gap with a long wedge of light. Those wedges, and the shadows of
 *    balconies and awnings raking down them, are the directional read.
 *  - It is ahead of the hero camera, so the establishing shot is backlit:
 *    silhouetted masses, rim light along every parapet, and shafts wherever the
 *    sun rakes between buildings.
 *
 * A sun straight down the street washes out; a sun square across it leaves the
 * whole canyon in flat shade. 26 degrees is the band where both happen at once.
 */
export const DEFAULT_TIME_OF_DAY = 16.92;

/** Sun is above the horizon between these hours. */
const DAY_START = 6;
const DAY_END = 20;
/**
 * Elevation and azimuth are deliberately decoupled. A single-parameter arc ties
 * the two together and there is then no way to ask for "low sun raking across
 * that street" — you get whatever elevation the azimuth you needed implies.
 */
const MAX_ELEVATION = 32.6 * (Math.PI / 180);
const AZIMUTH_START = 68.2 * (Math.PI / 180);
const AZIMUTH_SWEEP = 110 * (Math.PI / 180);

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
//
// Turbidity is pushed well past clean air and rayleigh pulled back below the
// textbook 2.0. Both moves do the same job: they trade Rayleigh's saturated
// blue for Mie's near-white forward scatter. A clean-air sky renders a zenith
// that is far too chromatic to be a light source — with an IBL driven off this
// dome, every shadow in the level inherits that blue and the asphalt turns
// navy. This is a dusty town in summer, and dusty air is what makes the
// shadows read as grey-blue instead of as ink.
export const DEFAULT_SKY_PARAMS: SkyParams = {
  turbidity: 6.0,
  rayleigh: 1.14,
  // Mie was carrying the frame's whole failure of contrast. At 0.0044 with a
  // g of 0.76 the forward lobe put several hundred pixels of sky at the top of
  // the range around the sun, and — because this same dome is the IBL — every
  // shadowed surface in the level was being filled by it. Backing both off
  // narrows the aureole to something a solar disc can punch through and takes
  // the sky's share of the fill budget down with it.
  mieCoefficient: 0.0032,
  mieG: 0.7,
  // The sky was rendering brighter than sunlit plaster, which is backwards: a
  // white wall in direct sun is two to four times the luminance of clear blue
  // sky, not a third of it. That inversion is most of why shadows and highlights
  // measured a tenth of a stop apart — the dome was flooding the whole street.
  intensity: 0.16,
  cloudCover: 0.5,
};

/**
 * Where the dome's radiance starts to roll, and what it asymptotes to.
 *
 * Not a clamp. `min( sky, 4.0 )` is what turned the sun into a flat cream
 * plateau three hundred pixels across — the disc and the aureole around it were
 * being written the same constant. A hyperbolic roll leaves everything under
 * the knee exact, keeps a gradient all the way up, and lets the solar disc,
 * added after the roll, sit three orders of magnitude clear of its surroundings.
 */
export const SKY_ROLL_KNEE = 0.095;
export const SKY_ROLL_MAX = 0.16;
/**
 * Solar disc radiance, as a multiplier on the Preetham `sunE` term.
 *
 * Absurdly large on purpose: the sun is about 0.53 degrees across, which is
 * four pixels at this field of view, and it has to survive an ACES shoulder as
 * a hard white core with a bloom skirt rather than as a slightly brighter patch
 * of sky. It is excluded from the IBL — the sun is already a directional light.
 */
export const SUN_DISC_INTENSITY = 46.0;

// Preetham fits for sea-level air; identical values drive the GPU dome so the
// CPU-side colours used for fog, IBL tinting and reflections cannot drift.
const TOTAL_RAYLEIGH = [5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5] as const;
const MIE_CONST = [1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14] as const;
/**
 * Dust chroma pull. Must stay identical to ATM_DUST / ATM_DUST_TINT in
 * ATMOSPHERE_GLSL — see the note there for why clean-air Preetham is the wrong
 * hue for this map. These are the numbers the fog colour, the hemisphere fill
 * and the reflection fallback are derived from, and if they drift from the
 * dome's the far end of the street stops matching the pixel above it.
 */
const ATM_DUST = 0.60;
const ATM_DUST_TINT = [1.075, 1.0, 0.925] as const;

const _dir = new THREE.Vector3();
const _tmpColor = new THREE.Color();

/** Direction pointing *towards* the sun, normalised. */
export function sunDirectionForTime(hours: number, out = new THREE.Vector3()): THREE.Vector3 {
  const t = THREE.MathUtils.clamp((hours - DAY_START) / (DAY_END - DAY_START), 0.001, 0.999);
  const elevation = MAX_ELEVATION * Math.sin(Math.PI * t);
  const azimuth = AZIMUTH_START + AZIMUTH_SWEEP * t;
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
    rgb[i] = (lin + 0.1 * fex) * 0.04 + bias[i];
  }
  // Same dust pull the dome applies, before the intensity scale — which is
  // where the shader applies it too, since uSkyIntensity multiplies the call.
  const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  for (let i = 0; i < 3; i++) {
    rgb[i] = (rgb[i] * (1 - ATM_DUST) + lum * ATM_DUST_TINT[i] * ATM_DUST) * p.intensity;
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
  //
  // At 0.45 this key ran a red-over-blue of 1.79, and since it is the key it set
  // the hue of every sunlit surface in the level: the sand measured 1.35 to 1.44
  // across the whole carriageway, which is a pigment, not a light. Golden hour is
  // warm light falling on materials that still have their own colour. Backed off
  // to 0.33 the key sits at 1.68 and the separation the frame reads as sunlight —
  // this against a sky fill the dust pull now puts near 0.4 — is untouched.
  _tmpColor.setHex(0xffd9a8, THREE.SRGBColorSpace);
  return out.lerp(_tmpColor, 0.33);
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
  /** Lower hemisphere albedo, in linear space. Shared with every pass. */
  readonly groundLinear = new THREE.Color().setHex(0x4a4239, THREE.SRGBColorSpace);
  /** Roll parameters, published so the composite's haze rolls identically. */
  readonly rollKnee = SKY_ROLL_KNEE;
  readonly rollMax = SKY_ROLL_MAX;
  /**
   * IBL weight, and the single most important number in the frame.
   *
   * The dome is a *sky*: bright, and — even at this turbidity — chromatic. Fed
   * to the PBR ambient at anything near unity it becomes the dominant light on
   * every surface the sun cannot reach, which is most of a street canyon at 21
   * degrees. That is what makes engine screenshots look like they were shot
   * through a blue gel.
   *
   * At 0.78, against a dome that was itself two stops hot, it was not a fill —
   * it was the key. A sunlit facade measured 0.12 of scene radiance and the
   * shadowed carriageway in front of it measured 0.064: a ratio of under two to
   * one, when the rig was authored for seven. No tone curve can put a sunlit
   * wall at sRGB 0.75 and a shadow at 0.10 out of that; the range has to exist
   * in the scene first. Halved, and the dome behind it dimmed as well.
   */
  environmentIntensity = 0.82;

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

    this.material = new THREE.ShaderMaterial({
      name: 'SkyDome',
      uniforms: {
        uSunDir: { value: this.sunDirection.clone() },
        uTurbidity: { value: this.params.turbidity },
        uRayleigh: { value: this.params.rayleigh },
        uMieCoefficient: { value: this.params.mieCoefficient },
        uMieG: { value: this.params.mieG },
        uSkyIntensity: { value: this.params.intensity },
        uTime: { value: 0 },
        uCloudCover: { value: this.params.cloudCover },
        uGroundColor: { value: this.groundLinear },
        uSkyRollKnee: { value: SKY_ROLL_KNEE },
        uSkyRollMax: { value: SKY_ROLL_MAX },
        uSunDiscIntensity: { value: SUN_DISC_INTENSITY },
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

    // Deliberately faint. The real aerial perspective is a screen-space pass in
    // the composite, which integrates height-attenuated density along the view
    // ray and tints with the sky radiance *in the direction of view* — warm
    // toward the sun, cool away from it. A uniform FogExp2 cannot do that and
    // would only flatten the result, so it is left in solely to catch the
    // transparent surfaces the depth-driven pass never sees.
    this.fog = new THREE.FogExp2(0x000000, 0.0018);
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
    m.uniforms.uSkyIntensity.value = this.params.intensity;
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
    // The disc is a directional light already. Baking a ~7000-unit fireball
    // into the PMREM as well double-counts the key and puts a second, blurrier
    // sun into every specular highlight in the level.
    const discUniform = this.material?.uniforms.uSunDiscIntensity;
    if (discUniform) discUniform.value = 0;
    try {
      this.envTarget = this.pmrem.fromScene(this.envScene, 0, 0.5, 5000);
    } catch (err) {
      console.error('[sky] PMREM generation failed', err);
      this.envTarget = previous;
      return;
    } finally {
      if (discUniform) discUniform.value = SUN_DISC_INTENSITY;
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
