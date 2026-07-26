import * as THREE from 'three';
import type { FrameTime, GameContext, System } from '../core/Contracts';
import { CSM_COVERAGE_GLSL, csmLoopSnippet } from './Shaders';
import { DEFAULT_SKY_PARAMS, DEFAULT_TIME_OF_DAY, sunDirectionForTime, sunLightColor, type SkySystem } from './Sky';

/** Nearest cascade starts here; closer than this nothing casts a visible shadow. */
const SHADOW_NEAR = 0.25;
/** Beyond this the cascades stop and geometry is lit but unshadowed. */
const SHADOW_DISTANCE = 165;
/** 0 = uniform splits, 1 = logarithmic. 0.5 is the classic practical blend. */
const SPLIT_LAMBDA = 0.5;
/** How far back along the sun ray each cascade's ortho camera sits. */
const CASCADE_BACK_DISTANCE = 95;
const MAX_SHADOW_LOCALS = 4;

const UP = new THREE.Vector3(0, 1, 0);
const _origin = new THREE.Vector3();
const _eye = new THREE.Vector3();
const _center = new THREE.Vector3();
const _centerLs = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _rot = new THREE.Matrix4();
const _rotInv = new THREE.Matrix4();
const _lightWorld = new THREE.Vector3();

export interface CascadeInfo {
  readonly light: THREE.DirectionalLight;
  /** View-space distance (metres) this cascade covers out to. */
  far: number;
  /** World radius of the cascade's ortho box. */
  radius: number;
}

interface LocalLightEntry {
  light: THREE.PointLight | THREE.SpotLight;
  baseIntensity: number;
  range: number;
  shadowed: boolean;
  distance: number;
}

/**
 * Sun, sky fill and local lights.
 *
 * Cascades are implemented as N directional lights sharing one colour and
 * intensity. A patch injected into three's directional-light loop weights each
 * light by whether the fragment falls inside that cascade's box, so exactly one
 * cascade (or a blend of two across the fade band) contributes per pixel and
 * the total never exceeds a single sun.
 */
export class LightingSystem implements System {
  readonly name = 'lighting';

  /** Direction pointing *towards* the sun. */
  readonly sunDirection = new THREE.Vector3();
  readonly sunColor = new THREE.Color(0xffd9a8);
  sunIntensity = 5.0;

  readonly cascades: CascadeInfo[] = [];

  private ctx: GameContext | null = null;
  private hemi: THREE.HemisphereLight | null = null;
  /**
   * Fake single-bounce fill. There is no GI here, and without it a street in a
   * building's shadow is lit only by a blue sky and renders as navy. Real
   * shadowed asphalt between sunlit sandstone facades picks up a warm bounce
   * roughly a stop and a half under the key. Shadowless, so it costs one extra
   * light term and nothing else.
   */
  private bounce: THREE.DirectionalLight | null = null;
  private readonly locals: LocalLightEntry[] = [];
  private readonly splits: number[] = [];
  private csmInstalled = false;
  private originalLightsChunk: string | null = null;
  private originalCommonChunk: string | null = null;
  private lastPreparedFrame = -1;
  private shadowsEnabled = true;
  private viewRigChecked = false;
  private viewSun: THREE.DirectionalLight | null = null;
  private viewFill: THREE.HemisphereLight | null = null;
  /**
   * Viewmodel lights the weapon system brought with it. This system does not
   * own their positions or their ratios — that rig is authored around the gun —
   * but it does own how much light the player is standing in, so it scales
   * them all by the same sun-exposure factor the built-in rig uses.
   */
  private readonly borrowedViewLights: { light: THREE.Light; base: number }[] = [];
  private viewSunExposure = 1;
  private viewSunTarget = 1;
  private viewSunProbeCountdown = 0;

  init(ctx: GameContext): void {
    this.ctx = ctx;

    const sky = ctx.system<SkySystem>('sky');
    if (sky) {
      this.sunDirection.copy(sky.sunDirection);
      this.sunColor.copy(sky.sunColor);
    } else {
      sunDirectionForTime(DEFAULT_TIME_OF_DAY, this.sunDirection);
      sunLightColor(this.sunDirection, DEFAULT_SKY_PARAMS, this.sunColor);
    }

    const q = ctx.quality;
    this.shadowsEnabled = q.shadowMapSize > 0;
    ctx.renderer.shadowMap.enabled = this.shadowsEnabled;
    // PCFSoftShadowMap is deprecated in this three build and silently degrades
    // to hard shadows; PCF is the hardware-filtered Vogel-disc path.
    ctx.renderer.shadowMap.type = THREE.PCFShadowMap;

    const requested = this.shadowsEnabled ? Math.max(1, Math.min(4, Math.floor(q.shadowCascades))) : 1;
    const count = requested > 1 && this.installCsmPatch(requested) ? requested : 1;

    this.buildSplits(count);

    const mapSize = this.shadowsEnabled ? Math.max(512, q.shadowMapSize) : 512;
    for (let i = 0; i < count; i++) {
      const light = new THREE.DirectionalLight(0xffffff, this.sunIntensity);
      light.name = `sun-cascade-${i}`;
      light.color.copy(this.sunColor);
      light.castShadow = this.shadowsEnabled;
      light.shadow.mapSize.setScalar(mapSize);
      light.shadow.bias = -0.00012;
      light.shadow.normalBias = 0.02;
      // Tight filter up close so contacts stay crisp, wider further out where
      // the penumbra should be soft anyway.
      light.shadow.radius = 1.5 + i * 0.9;
      light.shadow.camera.near = 0.5;
      light.shadow.camera.far = CASCADE_BACK_DISTANCE * 2;
      light.matrixAutoUpdate = true;
      ctx.scene.add(light);
      ctx.scene.add(light.target);
      this.cascades.push({ light, far: this.splits[i + 1], radius: 1 });
    }

    // Cool sky fill. The IBL already carries most of the ambient, so this is
    // deliberately restrained — it exists to keep shadow interiors from going
    // to a flat neutral.
    // Sky fill, pulled off full saturation so shadows read as cool, not navy.
    const skyFill = new THREE.Color(0xcdd4de);
    const groundFill = sky ? new THREE.Color().copy(sky.groundColor) : new THREE.Color(0x6b5a44);
    const gm = Math.max(groundFill.r, groundFill.g, groundFill.b, 1e-3);
    if (gm > 1) groundFill.multiplyScalar(1 / gm);
    this.hemi = new THREE.HemisphereLight(skyFill, groundFill, 0.55);
    this.hemi.position.set(0, 60, 0);
    ctx.scene.add(this.hemi);

    const bounce = new THREE.DirectionalLight(0xffd6a6, this.sunIntensity * 0.20);
    bounce.name = 'sun-bounce';
    bounce.castShadow = false;
    ctx.scene.add(bounce);
    ctx.scene.add(bounce.target);
    this.bounce = bounce;

    this.prepare(ctx);
  }

  // -------------------------------------------------------------------------
  // Cascades
  // -------------------------------------------------------------------------

  private buildSplits(count: number): void {
    this.splits.length = 0;
    this.splits.push(SHADOW_NEAR);
    for (let i = 1; i < count; i++) {
      const s = i / count;
      const uniform = SHADOW_NEAR + (SHADOW_DISTANCE - SHADOW_NEAR) * s;
      const logarithmic = SHADOW_NEAR * Math.pow(SHADOW_DISTANCE / SHADOW_NEAR, s);
      this.splits.push(SPLIT_LAMBDA * logarithmic + (1 - SPLIT_LAMBDA) * uniform);
    }
    this.splits.push(SHADOW_DISTANCE);
  }

  private installCsmPatch(cascades: number): boolean {
    const chunk = THREE.ShaderChunk.lights_fragment_begin;
    const anchor = 'getDirectionalLightInfo( directionalLight, directLight );';
    if (typeof chunk !== 'string' || chunk.indexOf(anchor) < 0) {
      console.warn('[lighting] directional light loop not found; falling back to a single shadow cascade');
      return false;
    }
    this.originalLightsChunk = chunk;
    this.originalCommonChunk = THREE.ShaderChunk.common;
    THREE.ShaderChunk.common = `${this.originalCommonChunk}\n${CSM_COVERAGE_GLSL}`;
    THREE.ShaderChunk.lights_fragment_begin = chunk.replace(anchor, `${anchor}\n${csmLoopSnippet(cascades)}`);
    this.csmInstalled = true;
    return true;
  }

  private uninstallCsmPatch(): void {
    if (!this.csmInstalled) return;
    if (this.originalLightsChunk !== null) THREE.ShaderChunk.lights_fragment_begin = this.originalLightsChunk;
    if (this.originalCommonChunk !== null) THREE.ShaderChunk.common = this.originalCommonChunk;
    this.csmInstalled = false;
  }

  /**
   * Refits every cascade to the current camera. Idempotent per frame so the
   * render pipeline can force it immediately before drawing, after the camera
   * rig has had the last word on where the eye actually is.
   */
  prepare(ctx: GameContext): void {
    const frame = ctx.time.frame;
    if (frame === this.lastPreparedFrame) return;
    this.lastPreparedFrame = frame;

    const sky = ctx.system<SkySystem>('sky');
    if (sky && !sky.sunDirection.equals(this.sunDirection)) {
      this.sunDirection.copy(sky.sunDirection);
      this.sunColor.copy(sky.sunColor);
    }

    const camera = ctx.camera;
    camera.updateMatrixWorld();

    // The cascade rotation only depends on the sun, so it is computed once and
    // reused as the snapping frame for every cascade.
    _eye.copy(this.sunDirection);
    if (Math.abs(_eye.y) > 0.9995) _eye.y = Math.sign(_eye.y) * 0.9995;
    _rot.lookAt(_eye, _origin.set(0, 0, 0), UP);
    _rotInv.copy(_rot).transpose();

    _forward.set(0, 0, -1).applyQuaternion(camera.quaternion).normalize();

    const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
    const tanH = tanV * camera.aspect;
    const k2 = tanH * tanH + tanV * tanV;

    for (let i = 0; i < this.cascades.length; i++) {
      const cascade = this.cascades[i];
      const near = this.splits[i];
      const far = this.splits[i + 1];
      cascade.far = far;

      const mid = (near + far) * 0.5;
      const half = (far - near) * 0.5;
      // Bounding sphere of the frustum slice; depends only on the slice shape,
      // so it stays constant while the camera translates.
      let radius = Math.sqrt(half * half + far * far * k2);
      // Quantise so a changing FOV (ADS) cannot make shadows crawl.
      radius = Math.ceil(radius * 8) / 8;
      cascade.radius = radius;

      _center.copy(camera.position).addScaledVector(_forward, mid);

      const light = cascade.light;
      const mapSize = light.shadow.mapSize.x;
      const texelWorld = (radius * 2) / mapSize;

      // Snap the box origin to whole shadow texels in light space. Without
      // this the ortho box slides continuously and every shadow edge crawls.
      _centerLs.copy(_center).applyMatrix4(_rotInv);
      _centerLs.x = Math.round(_centerLs.x / texelWorld) * texelWorld;
      _centerLs.y = Math.round(_centerLs.y / texelWorld) * texelWorld;
      _centerLs.applyMatrix4(_rot);

      light.target.position.copy(_centerLs);
      light.position.copy(_centerLs).addScaledVector(this.sunDirection, CASCADE_BACK_DISTANCE + radius);
      light.color.copy(this.sunColor);
      light.intensity = this.sunIntensity;

      const cam = light.shadow.camera;
      cam.left = -radius;
      cam.right = radius;
      cam.top = radius;
      cam.bottom = -radius;
      cam.near = 0.5;
      cam.far = CASCADE_BACK_DISTANCE + radius * 3 + 40;
      cam.updateProjectionMatrix();

      // Normal-offset bias in world units: one and a half texels of the
      // cascade that will actually sample this fragment. Constant bias alone
      // either acnes on grazing surfaces or peter-pans on flat ones.
      light.shadow.normalBias = texelWorld * 1.6;
      light.shadow.bias = -0.00012;

      light.target.updateMatrixWorld(true);
      light.updateMatrixWorld(true);
    }

    // Bounce comes back from the sunlit side of the street, low and warm: the
    // sun direction mirrored through the vertical, tilted a little above
    // horizontal so it reaches the road as well as the facades opposite.
    const bounce = this.bounce;
    if (bounce) {
      _eye.set(-this.sunDirection.x, Math.max(0.42, this.sunDirection.y * 0.5), -this.sunDirection.z)
        .normalize();
      bounce.position.copy(_eye).multiplyScalar(30).add(camera.position);
      bounce.target.position.copy(camera.position);
      bounce.intensity = this.sunIntensity * 0.20;
      bounce.updateMatrixWorld(true);
      bounce.target.updateMatrixWorld(true);
    }

    this.updateLocals(camera);
    this.updateViewmodelRig(ctx);
  }

  /**
   * The viewmodel lives in its own scene with its own camera, so unless it is
   * given a matching rig it ends up lit by nothing and reads as a sticker.
   * Only installs a rig if the weapon system did not bring its own.
   */
  ensureViewmodelLighting(ctx: GameContext): void {
    if (this.viewRigChecked) return;
    this.viewRigChecked = true;

    const sky = ctx.system<SkySystem>('sky');
    if (ctx.viewScene.environment === null) ctx.viewScene.environment = ctx.environment;
    ctx.viewScene.environmentIntensity = (sky ? sky.environmentIntensity : 0.92) * 0.7;

    ctx.viewScene.traverse((o) => {
      const l = o as THREE.Light;
      if (l.isLight === true) this.borrowedViewLights.push({ light: l, base: l.intensity });
    });
    if (this.borrowedViewLights.length > 0) {
      this.updateViewmodelRig(ctx);
      return;
    }

    const key = new THREE.DirectionalLight(0xffffff, this.sunIntensity);
    key.name = 'viewmodel-key';
    key.color.copy(this.sunColor);
    key.castShadow = false;
    ctx.viewScene.add(key);
    ctx.viewScene.add(key.target);
    this.viewSun = key;

    const fill = new THREE.HemisphereLight(0x8fb3ff, 0x6b5a44, 0.3);
    ctx.viewScene.add(fill);
    this.viewFill = fill;

    this.updateViewmodelRig(ctx);
  }

  private updateViewmodelRig(ctx: GameContext): void {
    const key = this.viewSun;
    if (!key && this.borrowedViewLights.length === 0) return;

    // The viewmodel casts and receives no world shadows, so instead the key is
    // dimmed when the player themselves is out of the sun. Without this the
    // gun stays sunlit inside buildings, which reads instantly as wrong.
    if (--this.viewSunProbeCountdown <= 0) {
      this.viewSunProbeCountdown = 4;
      let lit = 1;
      try {
        _eye.copy(ctx.camera.position).addScaledVector(this.sunDirection, 60);
        lit = ctx.physics.hasLineOfSight(ctx.camera.position, _eye) ? 1 : 0.22;
      } catch {
        lit = this.viewSunExposure;
      }
      this.viewSunTarget = lit;
    }
    this.viewSunExposure += (this.viewSunTarget - this.viewSunExposure) * 0.12;

    if (this.borrowedViewLights.length > 0) {
      // Never fully dark: a weapon in shade is still lit by the sky and by
      // bounce off the ground, which is what the 0.34 floor stands in for.
      const k = 0.34 + 0.66 * this.viewSunExposure;
      for (const entry of this.borrowedViewLights) entry.light.intensity = entry.base * k;
    }
    if (!key) return;

    key.position.copy(this.sunDirection).multiplyScalar(24).add(ctx.camera.position);
    key.target.position.copy(ctx.camera.position);
    key.color.copy(this.sunColor);
    key.intensity = this.sunIntensity * this.viewSunExposure;
    key.updateMatrixWorld(true);
    key.target.updateMatrixWorld(true);

    if (this.viewFill) this.viewFill.intensity = 0.3 - 0.09 * this.viewSunExposure;
  }

  private updateLocals(camera: THREE.PerspectiveCamera): void {
    if (this.locals.length === 0) return;

    for (const entry of this.locals) {
      entry.light.getWorldPosition(_lightWorld);
      entry.distance = _lightWorld.distanceTo(camera.position);
      const cull = entry.range * 1.6 + 12;
      const fade = 1 - THREE.MathUtils.smoothstep(entry.distance, cull * 0.75, cull);
      entry.light.intensity = entry.baseIntensity * fade;
    }

    // Only the nearest few shadow casters get their maps re-rendered; the rest
    // keep the last map they were given, which is free and rarely visible.
    const shadowed = this.locals.filter((e) => e.shadowed && e.light.intensity > 0.001);
    shadowed.sort((a, b) => a.distance - b.distance);
    for (let i = 0; i < shadowed.length; i++) {
      shadowed[i].light.shadow.needsUpdate = i < MAX_SHADOW_LOCALS;
    }
  }

  lateUpdate(_time: FrameTime, ctx: GameContext): void {
    this.prepare(ctx);
  }

  // -------------------------------------------------------------------------
  // Public queries for the render pipeline
  // -------------------------------------------------------------------------

  get cascadeCount(): number {
    return this.cascades.length;
  }

  /** World -> shadow-map [0,1]^3 matrix for a cascade. */
  getCascadeMatrix(index: number): THREE.Matrix4 | null {
    const c = this.cascades[index];
    return c ? c.light.shadow.matrix : null;
  }

  /** The cascade's depth texture, or null until the first shadow render. */
  getCascadeMap(index: number): THREE.Texture | null {
    const c = this.cascades[index];
    const map = c?.light.shadow.map;
    return map ? (map.depthTexture ?? map.texture) : null;
  }

  getCascadeFar(index: number): number {
    return this.cascades[index]?.far ?? SHADOW_DISTANCE;
  }

  // -------------------------------------------------------------------------
  // Local lights
  // -------------------------------------------------------------------------

  addPointLight(
    position: THREE.Vector3,
    color: THREE.ColorRepresentation,
    intensity: number,
    range: number,
    castShadow = false,
  ): THREE.PointLight {
    const light = new THREE.PointLight(color, intensity, range, 2);
    light.position.copy(position);
    this.registerLocal(light, intensity, range, castShadow);
    return light;
  }

  addSpotLight(
    position: THREE.Vector3,
    target: THREE.Vector3,
    color: THREE.ColorRepresentation,
    intensity: number,
    range: number,
    angle = Math.PI / 5,
    penumbra = 0.45,
    castShadow = false,
  ): THREE.SpotLight {
    const light = new THREE.SpotLight(color, intensity, range, angle, penumbra, 2);
    light.position.copy(position);
    light.target.position.copy(target);
    this.ctx?.scene.add(light.target);
    this.registerLocal(light, intensity, range, castShadow);
    return light;
  }

  private registerLocal(
    light: THREE.PointLight | THREE.SpotLight,
    intensity: number,
    range: number,
    castShadow: boolean,
  ): void {
    const alreadyShadowed = this.locals.reduce((n, e) => n + (e.shadowed ? 1 : 0), 0);
    // Shadow casting is fixed at creation: toggling it later changes the light
    // counts baked into every program and would recompile the whole scene.
    const shadowed = castShadow && this.shadowsEnabled && alreadyShadowed < MAX_SHADOW_LOCALS;
    light.castShadow = shadowed;
    if (shadowed) {
      light.shadow.mapSize.setScalar(Math.min(1024, Math.max(512, this.ctx?.quality.shadowMapSize ?? 1024)));
      light.shadow.bias = -0.0016;
      light.shadow.normalBias = 0.035;
      light.shadow.radius = 2.5;
      light.shadow.autoUpdate = false;
      light.shadow.needsUpdate = true;
      light.shadow.camera.near = 0.12;
    }
    this.locals.push({ light, baseIntensity: intensity, range, shadowed, distance: 0 });
    this.ctx?.scene.add(light);
  }

  removeLight(light: THREE.Light): void {
    const i = this.locals.findIndex((e) => e.light === light);
    if (i < 0) return;
    const entry = this.locals[i];
    this.locals.splice(i, 1);
    entry.light.parent?.remove(entry.light);
    if ((entry.light as THREE.SpotLight).target) {
      const t = (entry.light as THREE.SpotLight).target;
      t.parent?.remove(t);
    }
    entry.light.shadow?.dispose();
    entry.light.dispose();
  }

  dispose(): void {
    const ctx = this.ctx;
    for (const cascade of this.cascades) {
      cascade.light.shadow.dispose();
      ctx?.scene.remove(cascade.light.target);
      ctx?.scene.remove(cascade.light);
      cascade.light.dispose();
    }
    this.cascades.length = 0;

    for (const entry of this.locals.slice()) this.removeLight(entry.light);
    this.locals.length = 0;

    if (this.hemi) {
      ctx?.scene.remove(this.hemi);
      this.hemi.dispose();
      this.hemi = null;
    }

    if (this.viewSun && ctx) {
      ctx.viewScene.remove(this.viewSun.target);
      ctx.viewScene.remove(this.viewSun);
    }
    this.viewSun?.dispose();
    this.viewSun = null;
    if (this.viewFill && ctx) ctx.viewScene.remove(this.viewFill);
    this.viewFill?.dispose();
    this.viewFill = null;

    this.uninstallCsmPatch();
    this.ctx = null;
  }
}
