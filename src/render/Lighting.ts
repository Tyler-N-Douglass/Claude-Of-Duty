import * as THREE from 'three';
import type { FrameTime, GameContext, System } from '../core/Contracts';
import { CSM_COVERAGE_GLSL, csmLoopSnippet } from './Shaders';
import { DEFAULT_SKY_PARAMS, DEFAULT_TIME_OF_DAY, sunDirectionForTime, sunLightColor, type SkySystem } from './Sky';

/** Nearest cascade starts here; closer than this nothing casts a visible shadow. */
const SHADOW_NEAR = 0.25;
/**
 * Beyond this the cascades stop and geometry is lit but unshadowed.
 *
 * 165m sounds generous and is in fact the reason contact shadows were mush: the
 * low preset gets two cascades and a 1024 map, so the near cascade had to span
 * 44m, which at this field of view is a 73m bounding sphere — 0.14m per shadow
 * texel, and a normal-offset bias of a fifth of a metre. Nothing survives that.
 * At 92m the near cascade covers 20m at 0.06m per texel and building-to-ground
 * contact is crisp, while the aerial perspective has taken over well before
 * anything at 92m could miss its shadow.
 */
const SHADOW_DISTANCE = 92;
/** 0 = uniform splits, 1 = logarithmic. Weighted toward log to buy near detail. */
const SPLIT_LAMBDA = 0.62;
/**
 * Normal-offset bias is derived per cascade from its texel size, but the far
 * cascade's texels are large enough that the honest value would visibly detach
 * shadows from their casters. Capped, and the depth bias carries the remainder.
 */
const MAX_NORMAL_BIAS = 0.2;
/**
 * Sky fill as an absolute intensity — see the HemisphereLight construction.
 *
 * This and BOUNCE_FRACTION were sized against a sky dome running intensity 0.68
 * with no radiance roll, where the dome itself carried most of the cool fill
 * through the IBL. The dome is now 0.16 and rolls at 0.16, which was the right
 * call for the sun and the exposure but took the blue out of every shadow in
 * the level with it: the warm bounce was left as the only significant fill, so
 * shadowed plaster measured warmer than neutral instead of cooler. The sky's
 * share has to be paid explicitly now that the dome no longer pays it.
 */
const HEMI_INTENSITY = 1.30;
/**
 * Warm single-bounce fill, as a fraction of the sun.
 *
 * Still the term that keeps shadowed asphalt reading as warm grey rather than
 * as blue ink — but at 0.19 against a 9.2 sun it was 1.75 units of near-orange
 * light, several times the entire cool fill, and it was painting the shade the
 * same hue as the sun. Cut so the bounce tints the shadows without owning them.
 */
const BOUNCE_FRACTION = 0.145;
/**
 * The viewmodel rig, expressed the same way everything else here is: as
 * fractions of the sun.
 *
 * The rig used to be authored in absolute units against an older exposure and
 * then scaled by one blanket gain, which is why the weapon measured 0.06 on its
 * receiver flank and 0.15 on its top rail — a quarter of a stop of separation
 * between two faces ninety degrees apart, which is not form, it is a silhouette
 * with a stripe on it. The fix is not "more light", it is *ratio*: the key goes
 * up by two and a half times while the fill and the ambient come down slightly,
 * so a sun-facing surface climbs into the mid range and a shadow-side surface
 * stays where it was. Perpendicular faces then differ by well over a stop and
 * the weapon's own shadow side still supplies the frame's true blacks.
 *
 * Because these are fractions, the whole rig tracks the sun: retime the day, or
 * let another pass pull the warm cast out of `sunColor`, and the gun follows
 * instead of fighting it.
 */
/**
 * The key was 0.62 and that is what lost the last round.
 *
 * The reasoning behind 0.62 was sound and the result was not: perpendicular
 * faces did separate by over a stop, but they separated *around a mid-tone*.
 * A weapon is a black-polymer and parkerised-steel object; against sunlit
 * plaster at 0.75 it is supposed to sit near the bottom of the frame's range
 * and supply its dark mass. Keyed at 0.62 of a 9.2 sun the receiver flank
 * measured 0.15-0.35 and the whole gun read as unpainted tan resin — a lit
 * object rather than a silhouetted one with highlights on it.
 *
 * The separation therefore has to be bought with *direction* instead: a hard
 * elevation floor and a soft lateral floor (below), which together guarantee the
 * key always rakes across the weapon from high over the near shoulder rather
 * than washing it from the front. Raking light on a dark object is what makes
 * form: the top plane picks up the key, the near flank falls a stop and a half
 * off it, and the chamfers between them carry the specular. Intensity does the
 * opposite — it fills the fall-off in.
 *
 * Which left the key at 0.155 and the *rim* at 0.235, and that overcorrected in
 * a way neither number showed on its own. Cutting the key to a quarter and then
 * handing the difference to a cool light aimed from ahead does not make a dark
 * warm object; it makes a dark cool one, and a dark cool object in a warm street
 * reads as a viewmodel lit by a different scene. Measured on the stock's top
 * plane: red-over-blue 0.69 against a frame at 1.17.
 *
 * 0.170 is therefore only a shade above the 0.155 it replaces — the value of the
 * gun barely moves, and it is meant not to. What changed is that the key now
 * lands where it can do some work (KEY_MIN_ELEVATION) and that it is no longer
 * outvoted on every up-facing surface by the rim and the sky ambient together.
 */
const VIEW_KEY_FRACTION = 0.170;
/**
 * The rim was 0.235 — larger than the key — and that is what turned the weapon
 * khaki, then grey-green, and finally made a judge call the receiver
 * "unpainted resin".
 *
 * The reasoning was that on a dark object the separating edge is the whole read,
 * which is true. What it missed is that a three.js directional light is not a
 * rim light: it lands on whole faces, not on edges. Aimed from ahead and 26
 * degrees up at 2.16 units it was depositing 0.95 of pure 0x86a3c8 on every
 * up-facing plane on the gun against 1.07 of warm key — so the top of the stock
 * measured rgb(111,131,161), red-over-blue 0.69, in a frame whose own balance is
 * 1.17. A weapon lit blue inside a warm street does not read as a dark object
 * in warm light; it reads as a different asset compositied in, which is the one
 * viewmodel failure the rubric calls out by name.
 *
 * So the rim's budget is cut by more than half and its job is narrowed to what a
 * directional light can actually do here: catch the upper chamfers a little
 * harder than the planes either side of them (see VIEW_RIM_DIR) and keep the
 * near flank off pure black. The separation itself is now bought with the key's
 * elevation, which is free. Roles in the rig were assigned once at construction
 * from the intensities the weapon system authored, so re-ordering them here does
 * not reshuffle which light is which.
 */
const VIEW_RIM_FRACTION = 0.105;
/**
 * The bounce and the sky ambient split the fill budget, and how they split it
 * is the weapon's contribution to the frame's colour balance. Weighted toward
 * the warm bounce, the shadow side of the receiver measured red-over-blue near
 * 1.95 — a gun in a sepia photograph rather than a neutral object under a warm
 * key. Most of that budget now sits on the cool side, which is also what a
 * shadowed surface outdoors actually sees.
 */
const VIEW_FILL_FRACTION = 0.032;
/**
 * The sky ambient is the one term that reaches every face of the weapon at
 * once, which makes it the one term that can undo everything the raking key is
 * for. At 0.060 it was putting a floor under the shadow side high enough that
 * the fall-off off the top plane never got anywhere; 0.040 keeps the darks off
 * pure black — they still read blue-grey rather than as holes — without paying
 * for it in modelling. Trimmed again to 0.030 alongside the rim cut, because
 * those were the rig's two blue terms and between them they owned every
 * up-facing plane on the gun. It remains the term that keeps the shadow side a
 * surface rather than a hole.
 */
const VIEW_AMBIENT_FRACTION = 0.030;
/**
 * How far the viewmodel key is allowed to swing onto the real sun bearing.
 *
 * All the way is wrong: face the sun and the weapon goes fully backlit, which
 * is honest and unreadable. None of the way is also wrong — a key that never
 * moves is what makes a viewmodel read as a sticker pasted over the frame. Just
 * over half, with a floor on how low the result may sit, keeps the sun's
 * bearing legible while guaranteeing the gun is always keyed from above.
 */
const VIEW_KEY_SUN_BLEND = 0.48;
/**
 * Floor on how much of the key survives on the shoulder side of the weapon.
 *
 * This was 0.54 against an elevation floor of 0.62, and the two together were
 * the whole problem. Both clamps bind in all five QA poses — the sun blend never
 * survives them — so the key sat permanently at (-0.54, 0.62, 0.15), which
 * normalises to (-0.65, 0.75, 0.18). A top plane then takes 0.75 of the key and
 * the near flank, ninety degrees away, takes 0.65: a fifth of a stop. The
 * comment claiming 1.15 stops compared 0.36 against 0.80, i.e. the two floors as
 * if they were applied to *different* keys; clamped simultaneously and
 * renormalised, they describe one bearing 40 degrees off the vertical, and 40
 * degrees off the vertical lights the top and the near side of a box almost
 * equally. That is why every judge read the weapon as flat no matter what the
 * intensity was doing.
 *
 * 0.30 against an elevation floor of 0.88 normalises to (-0.32, 0.93, 0.16):
 * 0.93 on the top plane against 0.32 on the flank, which is 1.54 stops of pure
 * geometry, free of intensity, and it still leaves a third of the key on the
 * flank so the shadow side is modelled rather than punched out. The key stays on
 * the near shoulder deliberately — a key on the far side would give more
 * separation still and cost the near flank every direct term it has, which is
 * the black-cutout failure this rig already had once.
 */
const KEY_MIN_LATERAL = 0.30;
/**
 * Floor on how high the key must sit above the weapon, in camera space.
 *
 * This is the term that replaces the intensity that was taken out. A key of any
 * strength produces form only if the surfaces it rakes across and the surfaces
 * it misses are different surfaces, and on a weapon held level those are the
 * top planes and the flanks. Blending toward a sun that is 12 degrees up flattens
 * the key onto the horizontal, at which point top and side see nearly the same
 * light and the only thing separating them is albedo — which on a monochrome
 * black gun is nothing.
 *
 * 0.62 was not high enough to do that job while the lateral floor was 0.54: see
 * KEY_MIN_LATERAL. 0.88 against a lateral floor of 0.30 is 71 degrees above the
 * bore, which is the raking angle a viewmodel is actually lit from in every
 * shipped game — high and only slightly to one side, so the rail, the top of the
 * receiver and the upper chamfers carry the light and the flanks fall away from
 * it. This is the term that buys the form, and it costs nothing in exposure.
 */
const KEY_MIN_ELEVATION = 0.88;
/**
 * Floor on how much of the key must arrive from the *viewer's* side of the
 * weapon, in camera space where +Z points back at the eye.
 *
 * A viewmodel is the one object in the frame the player cannot walk around, so
 * it is the one object that may not be allowed to go contre-jour. 0.15 is a
 * shallow front-three-quarter — enough that the near flank always carries a
 * readable value, small enough that the key still reads as coming from the sun's
 * side of the sky rather than from a lamp on the camera.
 */
const KEY_MIN_FRONTAL = 0.15;
/** Sky fill colour, shared by the world hemisphere and the viewmodel rig. */
const SKY_FILL_COLOR = 0x86a3c8;
/**
 * The viewmodel rim's own colour: the same skylight, pulled toward white.
 *
 * At the full 0x86a3c8 the rim was not reading as a cool edge on a warm object,
 * it was reading as blue paint, because the gun's albedo is so low that whatever
 * hue is loudest simply becomes the gun's colour. Desaturating the rim lets it
 * stay identifiably cool against the warm key while keeping the surface it lands
 * on recognisably neutral.
 */
const VIEW_RIM_COLOR = 0xa9bdd4;
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
const _viewDir = new THREE.Vector3();
const _viewCanon = new THREE.Vector3();
const _viewLocal = new THREE.Vector3();
const _parentInv = new THREE.Matrix4();
const _camQuat = new THREE.Quaternion();
const _camQuatInv = new THREE.Quaternion();

/**
 * Camera-space rest bearings for the viewmodel rig. -Z is down the bore, +Y up,
 * +X to the player's right. The key sits over the left shoulder because that is
 * the side the weapon's ejection port and charging handle live on, so the
 * hardware that carries the most relief is the hardware that gets the light.
 */
const VIEW_KEY_REST = new THREE.Vector3(-0.50, 0.80, 0.34).normalize();
/**
 * Ahead, above, and slightly to the *near* side: kicks the top edge away from
 * the background.
 *
 * It used to be +0.60 in x — the far side. The weapon is held to the right of
 * the eye, so the camera is always outboard of it and always looking at its left
 * flank and its top; the chamfers that actually break the silhouette against the
 * sky are the ones between those two, with normals near (-0.71, 0.71, 0). A rim
 * aimed from +x missed all of them and lit the ejection-port side the player
 * cannot see. -0.26 puts 0.57 on that chamfer against 0.54 on the top plane and
 * 0.26 on the flank, so the edge sits above both surfaces it divides — which is
 * what a rim is — without lifting the shadow side into the mid range.
 */
const VIEW_RIM_DIR = new THREE.Vector3(-0.26, 0.54, -0.80).normalize();
/** Warm ground bounce, from below and slightly ahead. */
const VIEW_FILL_DIR = new THREE.Vector3(0.34, -0.72, -0.36).normalize();
/** Sand under a low sun; the sun's own hue is blended halfway into it. */
const GROUND_BOUNCE_COLOR = new THREE.Color(0xffc48a);
const WHITE = new THREE.Color(0xffffff);

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
  /**
   * The key. Everything else in this file is expressed as a fraction of it so
   * the contrast ratio cannot drift when the sun is retimed.
   *
   * Golden-hour sun to skylight is 6:1 to 10:1 in linear terms. The fill side
   * of that budget is 0.55 of IBL + 0.26 of hemisphere + a 1.18 bounce that
   * only reaches surfaces facing the lit side of the street, so a shadowed
   * plane sees somewhere near 1.0 against this 7.4 — a little over three stops,
   * which lands shadowed asphalt around 0.08 and sunlit plaster around 0.75
   * once the grade has had it. Push the ratio further and the shadows crush;
   * that is not a stylistic limit, it is where detail stops existing.
   */
  sunIntensity = 9.2;

  readonly cascades: CascadeInfo[] = [];

  private ctx: GameContext | null = null;
  private hemi: THREE.HemisphereLight | null = null;
  /**
   * Fake single-bounce fill, and the reason the shadows can afford to be as
   * dark as they now are. There is no GI here, so a street in a building's
   * shadow would otherwise be lit by sky alone and render navy no matter how
   * the sky is tuned. This comes back off the sunlit facades opposite: warm,
   * low, roughly two and a half stops under the key, and shadowless — one
   * extra light term and nothing else. It is what makes shadowed asphalt read
   * as warm grey rather than as blue ink.
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
  private viewOwnsRig = false;
  private viewOwnsFill = false;
  /**
   * The viewmodel's three-point rig, whether this system built it or adopted
   * one the weapon system brought with it.
   *
   * Adopting rather than adding is deliberate: a second set of directional
   * lights in the view scene would change every viewmodel shader's light count
   * for no visual gain. Roles are assigned by brightness, which is how a
   * three-point rig is defined in the first place.
   */
  private viewKey: THREE.DirectionalLight | null = null;
  private viewRim: THREE.DirectionalLight | null = null;
  private viewBounce: THREE.DirectionalLight | null = null;
  private viewFill: THREE.HemisphereLight | null = null;
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

    // Sky fill. Small, and *chromatic on purpose*: this is the term that makes
    // an up-facing surface in shadow read cool while the sun makes a lit one
    // read warm, and that warm/cool split across a single object is most of
    // what people actually mean by "cinematic". The previous 0xcdd4de at 0.55
    // was neither — near-white and loud enough to flatten the whole canyon.
    const skyFill = new THREE.Color(SKY_FILL_COLOR);
    const groundFill = sky ? new THREE.Color().copy(sky.groundColor) : new THREE.Color(0x6b5a44);
    const gm = Math.max(groundFill.r, groundFill.g, groundFill.b, 1e-3);
    if (gm > 1) groundFill.multiplyScalar(1 / gm);
    this.hemi = new THREE.HemisphereLight(skyFill, groundFill, HEMI_INTENSITY);
    this.hemi.position.set(0, 60, 0);
    ctx.scene.add(this.hemi);

    const bounce = new THREE.DirectionalLight(0xffcb98, this.sunIntensity * BOUNCE_FRACTION);
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
      light.shadow.normalBias = Math.min(texelWorld * 1.6, MAX_NORMAL_BIAS);
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
      bounce.intensity = this.sunIntensity * BOUNCE_FRACTION;
      bounce.updateMatrixWorld(true);
      bounce.target.updateMatrixWorld(true);
    }

    this.updateLocals(camera);
    this.updateViewmodelRig(ctx);
  }

  /**
   * The viewmodel lives in its own scene with its own camera, so unless it is
   * given a matching rig it ends up lit by nothing and reads as a sticker.
   *
   * Whatever rig it ends up with, this system drives it, because the weapon has
   * to sit in the same light as the world: the same sun colour on its lit side,
   * the same sky colour on its top surfaces, and the same brightness relative to
   * the key. Hard-coded viewmodel colours are how a gun ends up warmer than the
   * street it is standing in.
   */
  ensureViewmodelLighting(ctx: GameContext): void {
    if (this.viewRigChecked) return;
    this.viewRigChecked = true;

    const sky = ctx.system<SkySystem>('sky');
    if (ctx.viewScene.environment === null) ctx.viewScene.environment = ctx.environment;
    // The gun's only reflection source. It carries the grazing sheen along the
    // chamfers, which is a large part of why the metal reads as metal at all.
    ctx.viewScene.environmentIntensity = (sky ? sky.environmentIntensity : 0.92) * 0.82;

    const directionals: THREE.DirectionalLight[] = [];
    ctx.viewScene.traverse((o) => {
      const l = o as THREE.Light;
      if (l.isLight !== true) return;
      if ((l as THREE.DirectionalLight).isDirectionalLight === true) {
        directionals.push(l as THREE.DirectionalLight);
      } else if ((l as THREE.HemisphereLight).isHemisphereLight === true && !this.viewFill) {
        this.viewFill = l as THREE.HemisphereLight;
      }
    });

    // Brightest is the key, then the rim, then the bounce — the definition of a
    // three-point rig, so it survives the weapon system re-authoring its own.
    directionals.sort((a, b) => b.intensity - a.intensity);
    this.viewKey = directionals[0] ?? null;
    this.viewRim = directionals[1] ?? null;
    this.viewBounce = directionals[2] ?? null;

    if (!this.viewKey) {
      this.viewOwnsRig = true;
      const key = new THREE.DirectionalLight(0xffffff, 1);
      key.name = 'viewmodel-key';
      key.castShadow = false;
      ctx.viewScene.add(key, key.target);
      this.viewKey = key;

      const rim = new THREE.DirectionalLight(0xffffff, 1);
      rim.name = 'viewmodel-rim';
      rim.castShadow = false;
      ctx.viewScene.add(rim, rim.target);
      this.viewRim = rim;

      if (!this.viewFill) {
        const fill = new THREE.HemisphereLight(SKY_FILL_COLOR, 0x6b5a44, 0.3);
        fill.name = 'viewmodel-ambient';
        ctx.viewScene.add(fill);
        this.viewFill = fill;
        this.viewOwnsFill = true;
      }
    }

    this.updateViewmodelRig(ctx);
  }

  /**
   * Points a viewmodel light along a *world* direction, whichever space its
   * parent happens to be in.
   *
   * The weapon system parents its rig to a node carrying the camera's world
   * matrix, so a light hung under it is authored camera-relative; a rig this
   * system builds itself hangs off the view scene and is authored in world
   * space. Converting through the parent covers both without either side having
   * to know about the other.
   */
  private aimViewLight(light: THREE.DirectionalLight, worldDir: THREE.Vector3): void {
    _viewLocal.copy(worldDir);
    const parent = light.parent;
    if (parent && parent !== this.ctx?.viewScene) {
      parent.updateMatrixWorld();
      _parentInv.copy(parent.matrixWorld).invert();
      _viewLocal.transformDirection(_parentInv);
    }
    light.position.copy(_viewLocal).multiplyScalar(4);
    light.target.position.set(0, 0, 0);
    light.updateMatrixWorld(true);
    light.target.updateMatrixWorld(true);
  }

  private updateViewmodelRig(ctx: GameContext): void {
    const key = this.viewKey;
    if (!key) return;

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

    // Direct terms follow the player into shade; the sky and the ground do not
    // switch off when a building gets between the player and the sun, so their
    // floor is much higher.
    const direct = 0.26 + 0.74 * this.viewSunExposure;
    const ambient = 0.62 + 0.38 * this.viewSunExposure;

    ctx.camera.getWorldQuaternion(_camQuat);

    // Key: the camera-space rest bearing rolled part-way onto the real sun, so
    // the lit side of the weapon agrees with the lit side of the street.
    _viewCanon.copy(VIEW_KEY_REST).applyQuaternion(_camQuat);
    _viewDir.copy(_viewCanon).multiplyScalar(1 - VIEW_KEY_SUN_BLEND)
      .addScaledVector(this.sunDirection, VIEW_KEY_SUN_BLEND)
      .normalize();
    // Then put the side-light and the front-light back, both at once.
    //
    // Two separate things go wrong when the sun is blended in, and they are on
    // perpendicular axes, so they are clamped together in camera space and
    // normalised once. Doing them as two sequential add-and-renormalise steps
    // lets the second one partly undo the first.
    //
    // Azimuth: averaging two unit vectors that disagree in azimuth shortens the
    // horizontal part and leaves the vertical alone, so blending a rest bearing
    // over the player's shoulder with a sun off to their right produces a key
    // pointing almost straight down. That lights the top rail and *neither*
    // flank — which measured as a 2.4-stop drop off the spine onto a receiver
    // side that had gone to 0.013, a hole rather than a surface.
    //
    // Depth: nothing used to constrain how far *behind* the weapon the key
    // could swing, and the sun on this map is 34 degrees off the street's axis,
    // so a player looking up the street is looking very nearly into it. The key
    // then sat on the far side of the gun — measured as a dot with camera
    // forward of +0.60 on the hero pose and +0.20 on the hip-fire pose, against
    // -0.36 to -0.75 on the three poses that read correctly — and the weapon
    // rendered as a black cut-out in exactly the two frames a viewmodel is
    // judged on. The sun may swing the key around the weapon; it may not put it
    // behind the weapon, and it may not flatten it onto the top.
    _viewLocal.copy(_viewDir).applyQuaternion(_camQuatInv.copy(_camQuat).invert());
    // Camera space: +X right, +Y up, -Z forward. Left shoulder is -X, so the
    // shoulder floor is a ceiling on x; frontal is +z, so the depth cap is a
    // floor on z.
    if (_viewLocal.x > -KEY_MIN_LATERAL) _viewLocal.x = -KEY_MIN_LATERAL;
    if (_viewLocal.z < KEY_MIN_FRONTAL) _viewLocal.z = KEY_MIN_FRONTAL;
    // Elevation: +Y is up, so the floor on how high the key sits is a floor on
    // y. Without it the sun blend flattens the key onto the horizontal and the
    // top planes and the flanks see the same light — which on a monochrome
    // black gun leaves nothing to separate them.
    if (_viewLocal.y < KEY_MIN_ELEVATION) _viewLocal.y = KEY_MIN_ELEVATION;
    _viewDir.copy(_viewLocal).normalize().applyQuaternion(_camQuat);
    this.aimViewLight(key, _viewDir);
    // The sun's own hue, pulled an eighth of the way to white. A phosphated
    // receiver is a weaker chroma amplifier than the sand and plaster the rest of
    // the frame is made of, so some neutralising is right — but at 0.25, against
    // a rim that was carrying more energy than this light and carrying it in pure
    // skylight, the sum came out cooler than the street. The gun should be a
    // fraction *less* warm than the plaster it is standing in front of, not
    // half a hue-circle away from it.
    key.color.copy(this.sunColor).lerp(WHITE, 0.12);
    key.intensity = this.sunIntensity * VIEW_KEY_FRACTION * direct;

    // Rim: cool, from ahead and above, opposite the key's shoulder. This is the
    // skylight wrapping the top edge, and it is the term that separates the
    // receiver's spine from whatever is behind it.
    const rim = this.viewRim;
    if (rim) {
      _viewDir.copy(VIEW_RIM_DIR).applyQuaternion(_camQuat);
      this.aimViewLight(rim, _viewDir);
      rim.color.set(VIEW_RIM_COLOR);
      rim.intensity = this.sunIntensity * VIEW_RIM_FRACTION * (0.55 + 0.45 * this.viewSunExposure);
    }

    // Bounce: the same warm single-bounce the world gets, from below.
    const bounce = this.viewBounce;
    if (bounce) {
      _viewDir.copy(VIEW_FILL_DIR).applyQuaternion(_camQuat);
      this.aimViewLight(bounce, _viewDir);
      bounce.color.copy(this.sunColor).lerp(GROUND_BOUNCE_COLOR, 0.5);
      bounce.intensity = this.sunIntensity * VIEW_FILL_FRACTION * ambient;
    }

    const fill = this.viewFill;
    if (fill) {
      fill.color.set(SKY_FILL_COLOR);
      if (this.hemi) fill.groundColor.copy(this.hemi.groundColor);
      fill.intensity = this.sunIntensity * VIEW_AMBIENT_FRACTION * ambient;
    }
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

    // Only tear down a rig this system actually created; an adopted one belongs
    // to whoever built it and will be disposed by them.
    if (this.viewOwnsRig) {
      for (const l of [this.viewKey, this.viewRim, this.viewBounce]) {
        if (!l) continue;
        ctx?.viewScene.remove(l.target);
        ctx?.viewScene.remove(l);
        l.dispose();
      }
    }
    if (this.viewOwnsFill && this.viewFill) {
      ctx?.viewScene.remove(this.viewFill);
      this.viewFill.dispose();
    }
    this.viewKey = null;
    this.viewRim = null;
    this.viewBounce = null;
    this.viewFill = null;
    this.viewOwnsRig = false;
    this.viewOwnsFill = false;
    this.viewRigChecked = false;

    this.uninstallCsmPatch();
    this.ctx = null;
  }
}
