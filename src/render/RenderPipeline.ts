import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import type { RenderSystem } from '../core/Engine';
import type { GameContext } from '../core/Contracts';
import type { LightingSystem } from './Lighting';
import type { SkySystem } from './Sky';
import {
  BILATERAL_FRAG,
  BLOOM_DOWN_FRAG,
  BLOOM_PREFILTER_FRAG,
  BLOOM_UP_FRAG,
  CAMERA_VELOCITY_FRAG,
  COMPOSITE_FRAG,
  DOF_FRAG,
  FINAL_FRAG,
  FULLSCREEN_VERT,
  GTAO_FRAG,
  MOTION_BLUR_FRAG,
  OBJECT_VELOCITY_FRAG,
  OBJECT_VELOCITY_VERT,
  SSR_FRAG,
  TAA_FRAG,
  TONEMAP_FRAG,
  VOLUMETRIC_FRAG,
} from './Shaders';

const BLOOM_LEVELS = 6;
/** How much of the GTAO term reaches the composite at full weight. */
const AO_STRENGTH = 0.8;
const MIN_ADAPTIVE_SCALE = 0.6;
const MAX_DYNAMIC_MESHES = 16;
const DYNAMIC_REFRESH_FRAMES = 30;

function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}

/** 8-tap Halton(2,3), centred on the pixel. */
const JITTER: ReadonlyArray<readonly [number, number]> = Array.from({ length: 8 }, (_, i) => {
  return [halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5] as const;
});

const _v3a = new THREE.Vector3();
const _v3b = new THREE.Vector3();
const _matA = new THREE.Matrix4();

interface DynamicEntry {
  mesh: THREE.Mesh;
  /** World matrix as of the last frame this mesh wrote velocity. */
  prev: THREE.Matrix4;
}

export class RenderPipeline implements RenderSystem {
  readonly name = 'render';

  private ctx: GameContext | null = null;
  private renderer!: THREE.WebGLRenderer;
  private quad!: FullScreenQuad;
  private ready = false;

  // Sizing -------------------------------------------------------------------
  private cssWidth = 1;
  private cssHeight = 1;
  private bufferWidth = 1;
  private bufferHeight = 1;
  private adaptiveScale = 1;
  private targetsDirty = true;
  private lowFpsTimer = 0;
  private highFpsTimer = 0;
  private warmupFrames = 0;

  // Targets ------------------------------------------------------------------
  private sceneRT: THREE.WebGLRenderTarget | null = null;
  private velocityRT: THREE.WebGLRenderTarget | null = null;
  private aoRT: THREE.WebGLRenderTarget | null = null;
  private aoTmpRT: THREE.WebGLRenderTarget | null = null;
  private ssrRT: THREE.WebGLRenderTarget | null = null;
  private ssrTmpRT: THREE.WebGLRenderTarget | null = null;
  private volumeRT: THREE.WebGLRenderTarget | null = null;
  private postA: THREE.WebGLRenderTarget | null = null;
  private postB: THREE.WebGLRenderTarget | null = null;
  private history: [THREE.WebGLRenderTarget | null, THREE.WebGLRenderTarget | null] = [null, null];
  private bloom: (THREE.WebGLRenderTarget | null)[] = [];
  private historyIndex = 0;
  private historyValid = false;
  private hasPrevViewProj = false;

  // Materials ----------------------------------------------------------------
  private mCameraVelocity!: THREE.ShaderMaterial;
  private mObjectVelocity!: THREE.ShaderMaterial;
  private mViewmodelVelocity!: THREE.ShaderMaterial;
  private mGtao!: THREE.ShaderMaterial;
  private mBlur!: THREE.ShaderMaterial;
  private mSsr!: THREE.ShaderMaterial;
  private mVolume: THREE.ShaderMaterial | null = null;
  private mComposite!: THREE.ShaderMaterial;
  private mBloomPrefilter!: THREE.ShaderMaterial;
  private mBloomDown!: THREE.ShaderMaterial;
  private mBloomUp!: THREE.ShaderMaterial;
  private mMotionBlur!: THREE.ShaderMaterial;
  private mDof!: THREE.ShaderMaterial;
  private mTaa!: THREE.ShaderMaterial;
  private mTonemap!: THREE.ShaderMaterial;
  private mFinal!: THREE.ShaderMaterial;

  // Camera bookkeeping -------------------------------------------------------
  private readonly projSaved = new THREE.Matrix4();
  private readonly projInvSaved = new THREE.Matrix4();
  private readonly viewProjSaved = new THREE.Matrix4();
  private readonly viewProjInvSaved = new THREE.Matrix4();
  private readonly viewProj = new THREE.Matrix4();
  private readonly viewProjInv = new THREE.Matrix4();
  private readonly prevViewProj = new THREE.Matrix4();
  private jitterIndex = 0;

  // Post state ---------------------------------------------------------------
  /**
   * The lighting rig's sun went up and its ambient came down by more, which is
   * a contrast change rather than a brightness change — so the exposure has to
   * absorb the difference or the whole frame simply gets hotter. Set by the
   * highlight end, because that is the end with no headroom: sunlit plaster
   * lands around 0.75, which leaves the sky's diffuse dome under the shoulder
   * and lets the shadows fall where the ratio puts them, near 0.08.
   */
  private exposure = 1.3;
  private focusDistance = 8;
  private focusTarget = 8;
  private focusVelocity = 0;
  private aperture = 0.18;
  private apertureTarget = 0.18;
  private manualFocusTimer = 0;
  private aiming = false;
  private volumeCascadeCount = -1;

  private dynamics: DynamicEntry[] = [];
  private dynamicRefresh = 0;
  /**
   * FXAA is the only anti-aliasing when TAA is off, but it is a full-resolution
   * 13-tap pass and a machine already missing frame time cannot pay for it.
   * Latched off frame-time with hysteresis, and off to begin with so the very
   * first frame — always the most expensive one — never carries it.
   */
  private fxaaOn = false;

  // -------------------------------------------------------------------------
  // Lifecycle
  // -------------------------------------------------------------------------

  init(ctx: GameContext): void {
    this.ctx = ctx;
    this.renderer = ctx.renderer;
    this.cssWidth = Math.max(1, ctx.width);
    this.cssHeight = Math.max(1, ctx.height);

    // The pipeline owns tonemapping and the display transform; the renderer
    // must hand us untouched linear HDR.
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.toneMappingExposure = 1;

    try {
      this.quad = new FullScreenQuad();
      this.buildMaterials(ctx);
      this.allocate();
      this.ready = true;
    } catch (err) {
      console.error('[render] post pipeline unavailable, falling back to forward rendering', err);
      this.ready = false;
    }

    ctx.events.on('weapon.ads', (p) => {
      this.aiming = p.aiming;
      this.apertureTarget = p.aiming ? 0.46 : 0.18;
      this.manualFocusTimer = 0;
    });
  }

  resize(width: number, height: number): void {
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    if (w === this.cssWidth && h === this.cssHeight && !this.targetsDirty) return;
    this.cssWidth = w;
    this.cssHeight = h;
    this.targetsDirty = true;
    this.historyValid = false;
  }

  // -------------------------------------------------------------------------
  // Public controls
  // -------------------------------------------------------------------------

  /** Explicit focus override. Holds for ~1.5s before autofocus resumes. */
  setDofFocus(distance: number, aperture: number): void {
    this.focusTarget = Math.max(0.15, distance);
    this.apertureTarget = THREE.MathUtils.clamp(aperture, 0, 2);
    this.manualFocusTimer = 1.5;
  }

  setExposure(v: number): void {
    this.exposure = THREE.MathUtils.clamp(v, 0.05, 8);
  }

  /** Current internal render scale, 0.6..1. Useful for HUD diagnostics. */
  get renderScale(): number {
    return this.adaptiveScale;
  }

  /**
   * The scene pass's depth attachment, in internal buffer resolution. The FX
   * system reads this to soften particle intersections against world geometry;
   * it is null until the targets are allocated and again after dispose.
   */
  getSceneDepthTexture(): THREE.Texture | null {
    return this.sceneRT?.depthTexture ?? null;
  }

  // -------------------------------------------------------------------------
  // Materials
  // -------------------------------------------------------------------------

  private makeMaterial(
    fragmentShader: string,
    uniforms: Record<string, THREE.IUniform>,
    defines?: Record<string, string | number>,
  ): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      uniforms,
      defines: defines ?? {},
      vertexShader: FULLSCREEN_VERT,
      fragmentShader,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
  }

  private buildMaterials(ctx: GameContext): void {
    const q = ctx.quality;
    const heavy = q.preset === 'high' || q.preset === 'ultra';

    this.mCameraVelocity = this.makeMaterial(CAMERA_VELOCITY_FRAG, {
      tDepth: { value: null },
      uInvViewProj: { value: new THREE.Matrix4() },
      uPrevViewProj: { value: new THREE.Matrix4() },
    });

    this.mObjectVelocity = new THREE.ShaderMaterial({
      uniforms: {
        tSceneDepth: { value: null },
        uInvRes: { value: new THREE.Vector2(1, 1) },
        uPrevModelMatrix: { value: new THREE.Matrix4() },
        uPrevViewProj: { value: new THREE.Matrix4() },
        uCurrViewProj: { value: new THREE.Matrix4() },
      },
      vertexShader: OBJECT_VELOCITY_VERT,
      fragmentShader: OBJECT_VELOCITY_FRAG,
      side: THREE.FrontSide,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });

    this.mViewmodelVelocity = new THREE.ShaderMaterial({
      defines: { ZERO_VELOCITY: 1 },
      uniforms: {
        uPrevModelMatrix: { value: new THREE.Matrix4() },
        uPrevViewProj: { value: new THREE.Matrix4() },
        uCurrViewProj: { value: new THREE.Matrix4() },
      },
      vertexShader: OBJECT_VELOCITY_VERT,
      fragmentShader: OBJECT_VELOCITY_FRAG,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });

    this.mGtao = this.makeMaterial(
      GTAO_FRAG,
      {
        tDepth: { value: null },
        uInvProj: { value: new THREE.Matrix4() },
        uInvFullRes: { value: new THREE.Vector2() },
        uProjScaleY: { value: 1 },
        // World-space, in metres. 0.85 was tuned for surface crevices; the
        // occlusion that actually sells a frame is architectural — the metre or
        // so where a wall, a barrier or a crate meets the ground. Below ~0.6m
        // the pass simply cannot see that contact and every object in the level
        // looks pasted onto the road.
        uRadius: { value: 1.0 },
        uPower: { value: 1.5 },
        uFrame: { value: 0 },
        uFadeRange: { value: new THREE.Vector2(34, 70) },
      },
      // Even the low preset gets three slices: with two, a 1m radius at half
      // resolution turns contact darkening into three grey lobes per corner.
      { GTAO_SLICES: heavy ? 4 : 3, GTAO_STEPS: q.preset === 'ultra' ? 4 : q.preset === 'low' ? 2 : 3 },
    );

    this.mBlur = this.makeMaterial(BILATERAL_FRAG, {
      tDiffuse: { value: null },
      tDepth: { value: null },
      uDir: { value: new THREE.Vector2() },
      uNearFar: { value: new THREE.Vector2(0.05, 2000) },
      uDepthSigma: { value: 6 },
    });

    this.mSsr = this.makeMaterial(
      SSR_FRAG,
      {
        tScene: { value: null },
        tDepth: { value: null },
        uProj: { value: new THREE.Matrix4() },
        uInvProj: { value: new THREE.Matrix4() },
        uCamWorld: { value: new THREE.Matrix4() },
        uInvFullRes: { value: new THREE.Vector2() },
        uNearFar: { value: new THREE.Vector2(0.05, 2000) },
        uFrame: { value: 0 },
        uThickness: { value: 0.65 },
        uMaxDistance: { value: 26 },
        uReflectivity: { value: 0.55 },
        uSkyZenith: { value: new THREE.Color(0.15, 0.4, 1.0) },
        uSkyHorizon: { value: new THREE.Color(1.2, 1.1, 1.0) },
        uSkyGround: { value: new THREE.Color(0.1, 0.09, 0.08) },
        uSunColor: { value: new THREE.Color(1, 0.85, 0.65) },
        uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      },
      { SSR_STEPS: q.preset === 'ultra' ? 28 : 18 },
    );

    this.mComposite = this.makeMaterial(COMPOSITE_FRAG, {
      tScene: { value: null },
      tDepth: { value: null },
      tAo: { value: null },
      tSsr: { value: null },
      tVolume: { value: null },
      uInvProj: { value: new THREE.Matrix4() },
      uCamWorld: { value: new THREE.Matrix4() },
      uCamPos: { value: new THREE.Vector3() },
      uAoStrength: { value: AO_STRENGTH },
      // Ambient occlusion occludes *ambient*. Applying it to a surface the sun
      // is hitting is a lighting error that reads as dirt smeared into every
      // corner. There is no separate indirect buffer to multiply here, so the
      // pass uses scene luminance as the proxy: with the sun now eight-ish
      // times the fill, anything above this band is by definition sunlit and
      // keeps only uAoDirectKeep of its occlusion — enough for the bounce it
      // does block, not enough to dirty the key.
      uAoLitRange: { value: new THREE.Vector2(0.3, 0.85) },
      uAoDirectKeep: { value: 0.3 },
      uSsrEnabled: { value: 0 },
      uVolumeEnabled: { value: 0 },
      // Aerial perspective. Roughly three times the old density with twice the
      // height falloff: at 90m a ground-level facade now takes ~40% of the sky
      // radiance along its view ray, while a roofline 14m up takes a fraction
      // of that. Distance therefore reads as *air*, with the near-sun side of
      // the frame hazing warm and the away side hazing cool, and the ground
      // plane hazing harder than anything standing on it.
      uFogDensity: { value: 0.0072 },
      uFogHeightFalloff: { value: 0.1 },
      uFogBaseHeight: { value: -1.0 },
      uFogStart: { value: 13 },
      uFogDesaturate: { value: 0.4 },
      uSkyZenith: { value: new THREE.Color(0.15, 0.4, 1.0) },
      uSkyHorizon: { value: new THREE.Color(1.2, 1.1, 1.0) },
      uSkyGround: { value: new THREE.Color(0.1, 0.09, 0.08) },
      uSunColor: { value: new THREE.Color(1, 0.85, 0.65) },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
    });

    this.mBloomPrefilter = this.makeMaterial(BLOOM_PREFILTER_FRAG, {
      tDiffuse: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uThreshold: { value: 1.35 },
      uKnee: { value: 0.6 },
      uClamp: { value: 3.0 },
      uExposure: { value: 1.6 },
    });

    this.mBloomDown = this.makeMaterial(BLOOM_DOWN_FRAG, {
      tDiffuse: { value: null },
      uTexel: { value: new THREE.Vector2() },
    });

    this.mBloomUp = this.makeMaterial(BLOOM_UP_FRAG, {
      tDiffuse: { value: null },
      uTexel: { value: new THREE.Vector2() },
      uRadius: { value: 0.85 },
      uLevelWeight: { value: 0.72 },
    });
    this.mBloomUp.blending = THREE.AdditiveBlending;

    this.mMotionBlur = this.makeMaterial(
      MOTION_BLUR_FRAG,
      {
        tDiffuse: { value: null },
        tVelocity: { value: null },
        tDepth: { value: null },
        uResolution: { value: new THREE.Vector2() },
        uShutter: { value: 0.5 },
        uMaxRadius: { value: 0.032 },
        uFrame: { value: 0 },
      },
      // 8 iterations = 16 taps either side of the pixel. Fewer than this and
      // the jittered tap pattern shows as a comb along fast-moving silhouettes.
      { MB_TAPS: heavy ? 8 : 4 },
    );

    this.mDof = this.makeMaterial(
      DOF_FRAG,
      {
        tDiffuse: { value: null },
        tDepth: { value: null },
        uResolution: { value: new THREE.Vector2() },
        uNearFar: { value: new THREE.Vector2(0.05, 2000) },
        uFocusDistance: { value: 8 },
        uAperture: { value: 0.18 },
        uFocalLength: { value: 0.05 },
        uMaxCoc: { value: 5 },
        uFrame: { value: 0 },
      },
      { DOF_RINGS: heavy ? 2 : 1 },
    );

    this.mTaa = this.makeMaterial(TAA_FRAG, {
      tCurrent: { value: null },
      tHistory: { value: null },
      tVelocity: { value: null },
      tDepth: { value: null },
      uResolution: { value: new THREE.Vector2() },
      uHistoryValid: { value: 0 },
      uMinBlend: { value: 0.08 },
      uMaxBlend: { value: 0.42 },
    });

    this.mTonemap = this.makeMaterial(TONEMAP_FRAG, {
      tDiffuse: { value: null },
      tBloom: { value: null },
      uExposure: { value: 1 },
      uBloomStrength: { value: q.bloom ? 0.055 : 0 },
      uLift: { value: new THREE.Vector3(0.017, 0.020, 0.029) },
      uGamma: { value: new THREE.Vector3(1.0, 1.0, 1.03) },
      uGain: { value: new THREE.Vector3(1.025, 1.0, 0.972) },
      // Pushed well past the previous split. Shadowed faces go cool, sunlit
      // faces go warm, and the eye reads the difference as light rather than as
      // pigment — which is the whole trick, because the pigment is then free to
      // come down.
      uShadowTint: { value: new THREE.Vector3(0.9, 0.965, 1.085) },
      uHighlightTint: { value: new THREE.Vector3(1.12, 1.005, 0.85) },
      // Call of Duty's palette is far more desaturated than anyone remembers.
      // It earns its colour from the light, not from the materials.
      uSaturation: { value: 0.86 },
      uContrast: { value: 1.09 },
      // Filmic toe and shoulder. ACES has its own, but the pivot contrast above
      // it used to be a straight line into a hard clamp, which is what put the
      // frame in a narrow mid band with no black and no roll-off at the top.
      uToe: { value: 0.14 },
      uShoulder: { value: 0.75 },
    });

    this.mFinal = this.makeMaterial(FINAL_FRAG, {
      tDiffuse: { value: null },
      uResolution: { value: new THREE.Vector2() },
      uTime: { value: 0 },
      uAberration: { value: 0.0021 },
      uDistortion: { value: 0.035 },
      uVignette: { value: 0.4 },
      uGrain: { value: 0.026 },
      uSharpen: { value: 0.24 },
      uFxaa: { value: 0 },
    });

  }

  private ensureVolumeMaterial(cascades: number): void {
    if (this.mVolume && this.volumeCascadeCount === cascades) return;
    this.mVolume?.dispose();
    this.volumeCascadeCount = cascades;

    const uniforms: Record<string, THREE.IUniform> = {
      tDepth: { value: null },
      uInvProj: { value: new THREE.Matrix4() },
      uCamWorld: { value: new THREE.Matrix4() },
      uCamPos: { value: new THREE.Vector3() },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uSunColor: { value: new THREE.Color(1, 0.85, 0.65) },
      // High enough that the march reads as shafts where the sun rakes past an
      // occluder, low enough that it never becomes a uniform veil. With the sun
      // ahead of the hero camera and 26 degrees off the street's axis, every
      // alley mouth and roofline gap on the east terrace throws one.
      uDensity: { value: 0.0085 },
      // Shallower than before so the shafts survive up to roof height instead
      // of dying at head height; the medium is dust, and dust is well mixed.
      uHeightFalloff: { value: 0.055 },
      uBaseHeight: { value: -2 },
      uG: { value: 0.74 },
      uMaxDistance: { value: 64 },
      uFrame: { value: 0 },
      uCascadeFar: { value: new THREE.Vector4(1e6, 1e6, 1e6, 1e6) },
    };
    for (let i = 0; i < cascades; i++) {
      uniforms[`uShadow${i}`] = { value: null };
      uniforms[`uShadowMat${i}`] = { value: new THREE.Matrix4() };
    }

    const preset = this.ctx?.quality.preset;
    this.mVolume = this.makeMaterial(VOLUMETRIC_FRAG, uniforms, {
      CSM_COUNT: cascades,
      // Half resolution, dithered, and the step count only sets how smooth the
      // shaft edge is — 8 is cheap enough for a software rasteriser and still
      // reads as light rather than as banding.
      VOL_STEPS: preset === 'ultra' ? 16 : preset === 'low' ? 6 : 12,
    });
  }

  // -------------------------------------------------------------------------
  // Targets
  // -------------------------------------------------------------------------

  private makeTarget(
    w: number,
    h: number,
    type: THREE.TextureDataType,
    depthBuffer: boolean,
  ): THREE.WebGLRenderTarget {
    const rt = new THREE.WebGLRenderTarget(w, h, {
      type,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    rt.texture.colorSpace = THREE.LinearSRGBColorSpace;
    rt.texture.wrapS = THREE.ClampToEdgeWrapping;
    rt.texture.wrapT = THREE.ClampToEdgeWrapping;
    return rt;
  }

  private disposeTargets(): void {
    const all = [
      this.sceneRT, this.velocityRT, this.aoRT, this.aoTmpRT, this.ssrRT, this.ssrTmpRT,
      this.volumeRT, this.postA, this.postB, this.history[0], this.history[1], ...this.bloom,
    ];
    for (const rt of all) {
      if (!rt) continue;
      rt.depthTexture?.dispose();
      rt.dispose();
    }
    this.sceneRT = this.velocityRT = this.aoRT = this.aoTmpRT = null;
    this.ssrRT = this.ssrTmpRT = this.volumeRT = this.postA = this.postB = null;
    this.history = [null, null];
    this.bloom = [];
  }

  private allocate(): void {
    this.disposeTargets();

    const q = this.ctx?.quality;
    const dpr = this.renderer.getPixelRatio();
    const scale = Math.max(0.4, (q?.renderScale ?? 1) * this.adaptiveScale);
    const w = Math.max(2, Math.floor(this.cssWidth * dpr * scale));
    const h = Math.max(2, Math.floor(this.cssHeight * dpr * scale));
    this.bufferWidth = w;
    this.bufferHeight = h;

    const half = THREE.HalfFloatType;

    this.sceneRT = this.makeTarget(w, h, half, true);
    const depth = new THREE.DepthTexture(w, h, THREE.UnsignedIntType);
    depth.format = THREE.DepthFormat;
    depth.minFilter = THREE.NearestFilter;
    depth.magFilter = THREE.NearestFilter;
    this.sceneRT.depthTexture = depth;

    this.velocityRT = this.makeTarget(w, h, half, false);
    this.postA = this.makeTarget(w, h, half, true);
    this.postB = this.makeTarget(w, h, half, true);
    this.history = [this.makeTarget(w, h, half, false), this.makeTarget(w, h, half, false)];

    const hw = Math.max(1, Math.floor(w / 2));
    const hh = Math.max(1, Math.floor(h / 2));
    this.aoRT = this.makeTarget(hw, hh, THREE.UnsignedByteType, false);
    this.aoTmpRT = this.makeTarget(hw, hh, THREE.UnsignedByteType, false);
    this.ssrRT = this.makeTarget(hw, hh, half, false);
    this.ssrTmpRT = this.makeTarget(hw, hh, half, false);
    this.volumeRT = this.makeTarget(hw, hh, half, false);

    this.bloom = [];
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      const bw = Math.max(1, Math.floor(w / Math.pow(2, i + 1)));
      const bh = Math.max(1, Math.floor(h / Math.pow(2, i + 1)));
      this.bloom.push(this.makeTarget(bw, bh, half, false));
    }

    this.historyValid = false;
    this.targetsDirty = false;
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private blit(mat: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget | null, clear = true): void {
    this.quad.material = mat;
    this.renderer.setRenderTarget(target);
    this.renderer.autoClear = clear;
    this.quad.render(this.renderer);
    this.renderer.autoClear = true;
  }

  private updateAdaptiveScale(ctx: GameContext): void {
    if (this.warmupFrames < 90) {
      this.warmupFrames++;
      return;
    }
    const dt = Math.min(ctx.time.dt, 0.1);
    const target = Math.max(24, ctx.quality.targetFps);
    const fps = ctx.time.fps;

    if (fps < target * 0.92) {
      this.lowFpsTimer += dt;
      this.highFpsTimer = 0;
    } else if (fps > target * 1.03) {
      this.highFpsTimer += dt;
      this.lowFpsTimer = 0;
    } else {
      this.lowFpsTimer = Math.max(0, this.lowFpsTimer - dt * 0.5);
      this.highFpsTimer = Math.max(0, this.highFpsTimer - dt * 0.5);
    }

    if (this.lowFpsTimer >= 1 && this.adaptiveScale > MIN_ADAPTIVE_SCALE) {
      this.adaptiveScale = Math.max(MIN_ADAPTIVE_SCALE, this.adaptiveScale - 0.1);
      this.targetsDirty = true;
      this.lowFpsTimer = 0;
      this.highFpsTimer = 0;
    } else if (this.highFpsTimer >= 2 && this.adaptiveScale < 1) {
      this.adaptiveScale = Math.min(1, this.adaptiveScale + 0.05);
      this.targetsDirty = true;
      this.lowFpsTimer = 0;
      this.highFpsTimer = 0;
    }
  }

  private updateFocus(ctx: GameContext): void {
    const dt = Math.min(ctx.time.dt, 0.05);
    if (this.manualFocusTimer > 0) {
      this.manualFocusTimer -= dt;
    } else {
      let hitDistance = this.aiming ? 30 : 9;
      try {
        ctx.camera.getWorldDirection(_v3a);
        const hit = ctx.physics.raycast(ctx.camera.position, _v3a, 120);
        if (hit) hitDistance = Math.max(0.6, hit.distance);
      } catch {
        /* physics may not be populated yet */
      }
      this.focusTarget = hitDistance;
    }

    // Critically damped spring: racks focus with weight instead of snapping.
    const omega = this.aiming ? 22 : 14;
    const delta = this.focusTarget - this.focusDistance;
    this.focusVelocity += (delta * omega * omega - 2 * omega * this.focusVelocity) * dt;
    this.focusDistance += this.focusVelocity * dt;
    this.focusDistance = THREE.MathUtils.clamp(this.focusDistance, 0.2, 400);

    this.aperture += (this.apertureTarget - this.aperture) * Math.min(1, dt * 9);
  }

  private refreshDynamics(ctx: GameContext): void {
    this.dynamics.length = 0;
    const camPos = ctx.camera.position;
    const found: { entry: DynamicEntry; d: number }[] = [];
    ctx.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh || Array.isArray(mesh.material)) return;
      const isDynamic = (mesh as THREE.SkinnedMesh).isSkinnedMesh === true || o.userData.dynamic === true;
      if (!isDynamic) return;
      o.getWorldPosition(_v3b);
      found.push({
        entry: { mesh, prev: new THREE.Matrix4().copy(mesh.matrixWorld) },
        d: _v3b.distanceToSquared(camPos),
      });
    });
    found.sort((a, b) => a.d - b.d);
    for (let i = 0; i < Math.min(found.length, MAX_DYNAMIC_MESHES); i++) this.dynamics.push(found[i].entry);
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  render(ctx: GameContext): void {
    if (!this.ready) {
      this.renderFallback(ctx);
      return;
    }

    const r = this.renderer;
    const q = ctx.quality;

    this.updateAdaptiveScale(ctx);
    if (this.targetsDirty) this.allocate();

    const lighting = ctx.system<LightingSystem>('lighting');
    const sky = ctx.system<SkySystem>('sky');
    lighting?.prepare(ctx);
    lighting?.ensureViewmodelLighting(ctx);
    this.updateFocus(ctx);

    const scene = this.sceneRT!;
    const velocity = this.velocityRT!;
    const postA = this.postA!;
    const postB = this.postB!;
    const w = this.bufferWidth;
    const h = this.bufferHeight;

    const camera = ctx.camera;
    const viewCamera = ctx.viewCamera;
    camera.updateMatrixWorld();
    viewCamera.updateMatrixWorld();

    // Unjittered matrices drive velocity and reprojection; the jitter exists
    // only to move the sampling grid, and TAA must not chase it.
    this.projSaved.copy(camera.projectionMatrix);
    this.projInvSaved.copy(camera.projectionMatrixInverse);
    this.viewProjSaved.copy(viewCamera.projectionMatrix);
    this.viewProjInvSaved.copy(viewCamera.projectionMatrixInverse);
    this.viewProj.multiplyMatrices(this.projSaved, camera.matrixWorldInverse);
    this.viewProjInv.copy(this.viewProj).invert();

    const taaOn = q.taa;
    if (taaOn) {
      const j = JITTER[this.jitterIndex % JITTER.length];
      this.jitterIndex++;
      this.applyJitter(camera, j[0], j[1], w, h);
      this.applyJitter(viewCamera, j[0], j[1], w, h);
    }

    // --- 1. Scene into the HDR target -------------------------------------
    r.setRenderTarget(scene);
    r.autoClear = true;
    r.render(ctx.scene, camera);

    // --- 2. Velocity -------------------------------------------------------
    const needVelocity = q.motionBlur || taaOn;
    if (needVelocity) this.renderVelocity(ctx, velocity, scene, camera, viewCamera, w, h);

    // --- 3/4/5. Depth-driven effects on the world only ---------------------
    //
    // Ambient occlusion and light shafts are not scalable luxuries here, they
    // are load-bearing. Without AO nothing in the level makes contact with the
    // ground and the whole frame reads as decals on a backdrop; without the
    // volumetric march a backlit street has no air in it. The quality preset
    // switches them off at 'low' — which is also the preset every software
    // rasteriser and every phone gets, and therefore the preset the frame is
    // most often judged at. Both are half-resolution single passes costing two
    // triangles apiece; what scales with the preset is their tap count, set in
    // the material defines. So they run whenever their targets exist.
    const aoOn = this.aoRT !== null;
    if (aoOn) this.renderAo(scene, w, h, camera, ctx);

    const ssrOn = q.ssr && this.ssrRT !== null;
    if (ssrOn) this.renderSsr(scene, w, h, camera, ctx, sky);

    let volumeOn = false;
    if (lighting && this.volumeRT) {
      volumeOn = this.renderVolumetrics(scene, camera, ctx, lighting);
    }

    // --- 6. Composite ------------------------------------------------------
    this.updateCompositeUniforms(camera, ctx, sky, aoOn, ssrOn, volumeOn);
    this.mComposite.uniforms.tScene.value = scene.texture;
    this.mComposite.uniforms.tDepth.value = scene.depthTexture;
    this.mComposite.uniforms.tAo.value = aoOn ? this.aoRT!.texture : null;
    this.mComposite.uniforms.tSsr.value = ssrOn ? this.ssrRT!.texture : null;
    this.mComposite.uniforms.tVolume.value = volumeOn ? this.volumeRT!.texture : null;
    this.blit(this.mComposite, postA);

    // --- 7. Viewmodel, composited into the HDR frame before the rest of post
    r.setRenderTarget(postA);
    r.autoClear = false;
    r.clearDepth();
    r.render(ctx.viewScene, viewCamera);
    r.autoClear = true;

    if (taaOn) {
      camera.projectionMatrix.copy(this.projSaved);
      camera.projectionMatrixInverse.copy(this.projInvSaved);
      viewCamera.projectionMatrix.copy(this.viewProjSaved);
      viewCamera.projectionMatrixInverse.copy(this.viewProjInvSaved);
    }

    let src: THREE.WebGLRenderTarget = postA;
    let dst: THREE.WebGLRenderTarget = postB;

    // --- 8. Motion blur ----------------------------------------------------
    if (q.motionBlur) {
      const u = this.mMotionBlur.uniforms;
      u.tDiffuse.value = src.texture;
      u.tVelocity.value = velocity.texture;
      u.tDepth.value = scene.depthTexture;
      (u.uResolution.value as THREE.Vector2).set(w, h);
      // 180 degree shutter: velocity is already a per-frame displacement,
      // so half of it is exactly how far a point travels while the shutter is
      // open, independent of frame rate.
      u.uShutter.value = 0.5;
      u.uFrame.value = ctx.time.frame % 64;
      this.blit(this.mMotionBlur, dst);
      const t = src; src = dst; dst = t;
    }

    // --- 9. Depth of field -------------------------------------------------
    {
      const u = this.mDof.uniforms;
      u.tDiffuse.value = src.texture;
      u.tDepth.value = scene.depthTexture;
      (u.uResolution.value as THREE.Vector2).set(w, h);
      (u.uNearFar.value as THREE.Vector2).set(camera.near, camera.far);
      u.uFocusDistance.value = this.focusDistance;
      u.uAperture.value = this.aperture;
      u.uMaxCoc.value = this.aiming ? 9 : 5;
      u.uFrame.value = ctx.time.frame % 64;
      this.blit(this.mDof, dst);
      const t = src; src = dst; dst = t;
    }

    // --- 10. TAA -----------------------------------------------------------
    let resolved: THREE.Texture = src.texture;
    if (taaOn && this.history[0] && this.history[1]) {
      const write = this.history[this.historyIndex]!;
      const read = this.history[1 - this.historyIndex]!;
      const u = this.mTaa.uniforms;
      u.tCurrent.value = src.texture;
      u.tHistory.value = read.texture;
      u.tVelocity.value = velocity.texture;
      u.tDepth.value = scene.depthTexture;
      (u.uResolution.value as THREE.Vector2).set(w, h);
      u.uHistoryValid.value = this.historyValid ? 1 : 0;
      this.blit(this.mTaa, write);
      resolved = write.texture;
      this.historyIndex = 1 - this.historyIndex;
      this.historyValid = true;
      // Both ping-pong buffers are free again once the history owns the frame.
      dst = src === postA ? postB : postA;
    }

    // --- 11. Bloom ---------------------------------------------------------
    let bloomTexture: THREE.Texture | null = null;
    if (q.bloom && this.bloom.length === BLOOM_LEVELS) bloomTexture = this.renderBloom(resolved, w, h);

    // --- 12. Tonemap + grade ----------------------------------------------
    const tonemapTarget = dst;
    {
      const u = this.mTonemap.uniforms;
      u.tDiffuse.value = resolved;
      u.tBloom.value = bloomTexture;
      u.uExposure.value = this.exposure;
      u.uBloomStrength.value = bloomTexture ? 0.055 : 0;
      this.blit(this.mTonemap, tonemapTarget);
    }

    // --- 13. Final display pass -------------------------------------------
    {
      const u = this.mFinal.uniforms;
      u.tDiffuse.value = tonemapTarget.texture;
      (u.uResolution.value as THREE.Vector2).set(w, h);
      u.uTime.value = ctx.time.elapsed;
      // MSAA is unavailable through render targets and the renderer is created
      // with antialias:false, so with TAA off FXAA is the only thing standing
      // between the player and crawling geometry edges — when it is affordable.
      const frameMs = ctx.time.rawDt;
      if (frameMs > 0.045) this.fxaaOn = false;
      else if (frameMs > 0 && frameMs < 0.020) this.fxaaOn = true;
      u.uFxaa.value = !taaOn && this.fxaaOn ? 1 : 0;
      // Unsharp mask amplifies whatever the upscale reconstructed. Backing it
      // off with the internal resolution keeps a 0.7-scale frame from turning
      // every chamfer into a bright dash.
      u.uSharpen.value = 0.24 * (0.42 + 0.58 * this.adaptiveScale);
      u.uAberration.value = 0.0021 * (0.45 + 0.55 * this.adaptiveScale);
      this.blit(this.mFinal, null);
    }

    this.prevViewProj.copy(this.viewProj);
    this.hasPrevViewProj = true;
    r.setRenderTarget(null);
  }

  private applyJitter(
    camera: THREE.PerspectiveCamera,
    jx: number,
    jy: number,
    w: number,
    h: number,
  ): void {
    const e = camera.projectionMatrix.elements;
    e[8] += (jx * 2) / w;
    e[9] += (jy * 2) / h;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }

  // -------------------------------------------------------------------------
  // Passes
  // -------------------------------------------------------------------------

  private renderVelocity(
    ctx: GameContext,
    velocity: THREE.WebGLRenderTarget,
    scene: THREE.WebGLRenderTarget,
    camera: THREE.PerspectiveCamera,
    viewCamera: THREE.PerspectiveCamera,
    w: number,
    h: number,
  ): void {
    const r = this.renderer;

    const cu = this.mCameraVelocity.uniforms;
    cu.tDepth.value = scene.depthTexture;
    (cu.uInvViewProj.value as THREE.Matrix4).copy(this.viewProjInv);
    (cu.uPrevViewProj.value as THREE.Matrix4).copy(this.hasPrevViewProj ? this.prevViewProj : this.viewProj);
    this.blit(this.mCameraVelocity, velocity);

    // Object motion. Rendering each mesh on its own keeps us out of the other
    // systems' scene graphs — nothing is reparented, only the material is
    // swapped for the duration of one draw.
    if (ctx.quality.motionBlur && ctx.quality.preset !== 'low') {
      if (--this.dynamicRefresh <= 0) {
        this.dynamicRefresh = DYNAMIC_REFRESH_FRAMES;
        this.refreshDynamics(ctx);
      }

      if (this.dynamics.length > 0) {
        const u = this.mObjectVelocity.uniforms;
        u.tSceneDepth.value = scene.depthTexture;
        (u.uInvRes.value as THREE.Vector2).set(1 / w, 1 / h);
        (u.uCurrViewProj.value as THREE.Matrix4).copy(this.viewProj);
        (u.uPrevViewProj.value as THREE.Matrix4).copy(this.hasPrevViewProj ? this.prevViewProj : this.viewProj);

        r.setRenderTarget(velocity);
        r.autoClear = false;
        for (const entry of this.dynamics) {
          const mesh = entry.mesh;
          if (!mesh.parent || !mesh.visible || Array.isArray(mesh.material)) continue;
          _matA.copy(mesh.matrixWorld);
          if (_matA.equals(entry.prev)) {
            entry.prev.copy(_matA);
            continue;
          }
          (u.uPrevModelMatrix.value as THREE.Matrix4).copy(entry.prev);
          this.mObjectVelocity.uniformsNeedUpdate = true;
          const original = mesh.material;
          mesh.material = this.mObjectVelocity;
          try {
            r.render(mesh, camera);
          } finally {
            mesh.material = original;
          }
          entry.prev.copy(_matA);
        }
        r.autoClear = true;
      }
    }

    // The viewmodel rides the camera, so its screen-space motion is ~zero.
    // Writing that explicitly stops the world's velocity leaking onto the gun.
    const viewScene = ctx.viewScene;
    const previousOverride = viewScene.overrideMaterial;
    viewScene.overrideMaterial = this.mViewmodelVelocity;
    r.setRenderTarget(velocity);
    r.autoClear = false;
    r.render(viewScene, viewCamera);
    r.autoClear = true;
    viewScene.overrideMaterial = previousOverride;
  }

  private renderAo(
    scene: THREE.WebGLRenderTarget,
    w: number,
    h: number,
    camera: THREE.PerspectiveCamera,
    ctx: GameContext,
  ): void {
    const ao = this.aoRT!;
    const tmp = this.aoTmpRT!;

    const u = this.mGtao.uniforms;
    u.tDepth.value = scene.depthTexture;
    (u.uInvProj.value as THREE.Matrix4).copy(this.projInvSaved);
    (u.uInvFullRes.value as THREE.Vector2).set(1 / w, 1 / h);
    // projectionMatrix[1][1] * 0.5 * height = pixels per world unit at z = 1.
    u.uProjScaleY.value = this.projSaved.elements[5] * 0.5 * h;
    u.uFrame.value = ctx.time.frame % 64;
    this.blit(this.mGtao, ao);

    const b = this.mBlur.uniforms;
    b.tDepth.value = scene.depthTexture;
    (b.uNearFar.value as THREE.Vector2).set(camera.near, camera.far);
    b.uDepthSigma.value = 8;

    b.tDiffuse.value = ao.texture;
    (b.uDir.value as THREE.Vector2).set(1 / ao.width, 0);
    this.blit(this.mBlur, tmp);

    b.tDiffuse.value = tmp.texture;
    (b.uDir.value as THREE.Vector2).set(0, 1 / ao.height);
    this.blit(this.mBlur, ao);
  }

  private renderSsr(
    scene: THREE.WebGLRenderTarget,
    w: number,
    h: number,
    camera: THREE.PerspectiveCamera,
    ctx: GameContext,
    sky: SkySystem | undefined,
  ): void {
    const ssr = this.ssrRT!;
    const tmp = this.ssrTmpRT!;

    const u = this.mSsr.uniforms;
    u.tScene.value = scene.texture;
    u.tDepth.value = scene.depthTexture;
    (u.uProj.value as THREE.Matrix4).copy(this.projSaved);
    (u.uInvProj.value as THREE.Matrix4).copy(this.projInvSaved);
    (u.uCamWorld.value as THREE.Matrix4).copy(camera.matrixWorld);
    (u.uInvFullRes.value as THREE.Vector2).set(1 / w, 1 / h);
    (u.uNearFar.value as THREE.Vector2).set(camera.near, camera.far);
    u.uFrame.value = ctx.time.frame % 64;
    if (sky) {
      (u.uSkyZenith.value as THREE.Color).copy(sky.zenithColor);
      (u.uSkyHorizon.value as THREE.Color).copy(sky.horizonColor);
      (u.uSkyGround.value as THREE.Color).copy(sky.groundColor);
      (u.uSunColor.value as THREE.Color).copy(sky.sunColor);
      (u.uSunDir.value as THREE.Vector3).copy(sky.sunDirection);
    }
    this.blit(this.mSsr, ssr);

    const b = this.mBlur.uniforms;
    b.tDepth.value = scene.depthTexture;
    (b.uNearFar.value as THREE.Vector2).set(camera.near, camera.far);
    b.uDepthSigma.value = 12;

    b.tDiffuse.value = ssr.texture;
    (b.uDir.value as THREE.Vector2).set(1 / ssr.width, 0);
    this.blit(this.mBlur, tmp);

    b.tDiffuse.value = tmp.texture;
    (b.uDir.value as THREE.Vector2).set(0, 1 / ssr.height);
    this.blit(this.mBlur, ssr);
  }

  private renderVolumetrics(
    scene: THREE.WebGLRenderTarget,
    camera: THREE.PerspectiveCamera,
    ctx: GameContext,
    lighting: LightingSystem,
  ): boolean {
    const count = Math.min(4, lighting.cascadeCount);
    if (count <= 0) return false;
    for (let i = 0; i < count; i++) {
      if (!lighting.getCascadeMap(i)) return false;
    }

    this.ensureVolumeMaterial(count);
    const mat = this.mVolume;
    if (!mat) return false;

    const u = mat.uniforms;
    u.tDepth.value = scene.depthTexture;
    (u.uInvProj.value as THREE.Matrix4).copy(this.projInvSaved);
    (u.uCamWorld.value as THREE.Matrix4).copy(camera.matrixWorld);
    (u.uCamPos.value as THREE.Vector3).copy(camera.position);
    (u.uSunDir.value as THREE.Vector3).copy(lighting.sunDirection);
    (u.uSunColor.value as THREE.Color).copy(lighting.sunColor).multiplyScalar(lighting.sunIntensity * 0.5);
    u.uFrame.value = ctx.time.frame % 64;

    const far = u.uCascadeFar.value as THREE.Vector4;
    far.set(
      lighting.getCascadeFar(0),
      lighting.getCascadeFar(Math.min(1, count - 1)),
      lighting.getCascadeFar(Math.min(2, count - 1)),
      lighting.getCascadeFar(count - 1),
    );

    for (let i = 0; i < count; i++) {
      u[`uShadow${i}`].value = lighting.getCascadeMap(i);
      const m = lighting.getCascadeMatrix(i);
      if (m) (u[`uShadowMat${i}`].value as THREE.Matrix4).copy(m);
    }

    this.blit(mat, this.volumeRT!);
    return true;
  }

  private renderBloom(source: THREE.Texture, w: number, h: number): THREE.Texture | null {
    const levels = this.bloom;
    const first = levels[0];
    if (!first) return null;

    const pre = this.mBloomPrefilter.uniforms;
    pre.tDiffuse.value = source;
    pre.uExposure.value = this.exposure;
    (pre.uTexel.value as THREE.Vector2).set(1 / w, 1 / h);
    this.blit(this.mBloomPrefilter, first);

    for (let i = 1; i < levels.length; i++) {
      const from = levels[i - 1]!;
      const to = levels[i]!;
      const u = this.mBloomDown.uniforms;
      u.tDiffuse.value = from.texture;
      (u.uTexel.value as THREE.Vector2).set(1 / from.width, 1 / from.height);
      this.blit(this.mBloomDown, to);
    }

    // Additive upsample straight back into the finer level: no extra buffers,
    // and each level keeps the tight core of the one below it.
    for (let i = levels.length - 2; i >= 0; i--) {
      const from = levels[i + 1]!;
      const to = levels[i]!;
      const u = this.mBloomUp.uniforms;
      u.tDiffuse.value = from.texture;
      (u.uTexel.value as THREE.Vector2).set(1 / from.width, 1 / from.height);
      this.blit(this.mBloomUp, to, false);
    }

    return first.texture;
  }

  private updateCompositeUniforms(
    camera: THREE.PerspectiveCamera,
    ctx: GameContext,
    sky: SkySystem | undefined,
    aoOn: boolean,
    ssrOn: boolean,
    volumeOn: boolean,
  ): void {
    const u = this.mComposite.uniforms;
    (u.uInvProj.value as THREE.Matrix4).copy(this.projInvSaved);
    (u.uCamWorld.value as THREE.Matrix4).copy(camera.matrixWorld);
    (u.uCamPos.value as THREE.Vector3).copy(camera.position);
    u.uAoStrength.value = aoOn ? AO_STRENGTH : 0;
    u.uSsrEnabled.value = ssrOn ? 1 : 0;
    u.uVolumeEnabled.value = volumeOn ? 1 : 0;
    if (sky) {
      (u.uSkyZenith.value as THREE.Color).copy(sky.zenithColor);
      (u.uSkyHorizon.value as THREE.Color).copy(sky.horizonColor);
      (u.uSkyGround.value as THREE.Color).copy(sky.groundColor);
      (u.uSunColor.value as THREE.Color).copy(sky.sunColor);
      (u.uSunDir.value as THREE.Vector3).copy(sky.sunDirection);
    }
  }

  private renderFallback(ctx: GameContext): void {
    const r = this.renderer;
    r.setRenderTarget(null);
    r.autoClear = true;
    r.render(ctx.scene, ctx.camera);
    r.autoClear = false;
    r.clearDepth();
    r.render(ctx.viewScene, ctx.viewCamera);
    r.autoClear = true;
  }

  // -------------------------------------------------------------------------

  dispose(): void {
    this.disposeTargets();
    const materials = [
      this.mCameraVelocity, this.mObjectVelocity, this.mViewmodelVelocity, this.mGtao, this.mBlur,
      this.mSsr, this.mVolume, this.mComposite, this.mBloomPrefilter, this.mBloomDown, this.mBloomUp,
      this.mMotionBlur, this.mDof, this.mTaa, this.mTonemap, this.mFinal,
    ];
    for (const m of materials) m?.dispose();
    this.quad?.dispose();
    this.dynamics.length = 0;
    this.ctx = null;
  }
}
