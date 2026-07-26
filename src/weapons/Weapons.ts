/**
 * The weapon system: firing logic, ammunition, the viewmodel rig and every
 * event other systems key off.
 *
 * Fire timing is accumulated in `fixedUpdate` at the engine's 120 Hz step and
 * carries its remainder across steps, so a 950 rpm SMG fires 950 rounds per
 * minute whether the frame rate is 144 or 34. Nothing about rate of fire is
 * allowed to depend on how long a frame took.
 *
 * The viewmodel lives in `ctx.viewScene` under a rig node that copies the view
 * camera's world transform each late update. That gives the animator a clean
 * camera-relative frame to write into while leaving the muzzle, ejection port
 * and optic at genuine world coordinates for ballistics and debris.
 */
import * as THREE from 'three';
import {
  LAYERS,
  type FrameTime,
  type GameContext,
  type System,
  type WeaponRuntime,
  type WeaponSpec,
} from '../core/Contracts';
import {
  DEFAULT_LOADOUT,
  fireInterval,
  getEntry,
  sampleRecoil,
  type RecoilPattern,
  type WeaponEntry,
  type WeaponFeel,
  type WeaponVisual,
} from './WeaponSpecs';
import { GunMaterials, WeaponViewmodel, buildScopeOverlay } from './Viewmodel';
import { WeaponAnimator, type MotionInput } from './WeaponAnimation';

// ---------------------------------------------------------------------------
// Duck-typed neighbours. Cross-system access goes through ctx.system(), so the
// shapes we need are declared locally rather than importing another module.
// ---------------------------------------------------------------------------

interface PlayerLike {
  applyRecoil(pitch: number, yaw: number, recovery?: number): void;
  readonly sprinting: boolean;
  readonly grounded: boolean;
  readonly speed: number;
  readonly stridePhase: number;
  readonly stance: string;
  readonly gait: string;
  readonly velocity: THREE.Vector3;
  readonly yaw: number;
  readonly pitch: number;
}

interface DebrisRequest {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  angularVelocity: THREE.Vector3;
  radius: number;
  restitution: number;
  friction: number;
  lifetime: number;
  object: THREE.Object3D;
  onRetire?: (object: THREE.Object3D) => void;
}

interface DebrisSpawner {
  spawnDebris(opts: DebrisRequest): number;
}

function asDebrisSpawner(o: unknown): DebrisSpawner | null {
  return typeof (o as DebrisSpawner)?.spawnDebris === 'function' ? (o as DebrisSpawner) : null;
}

// ---------------------------------------------------------------------------

const FLASH_LIFETIME = 0.034; // two frames at 60 fps; real flashes are faster
const AIM_DISTANCE = 220;
const CASING_POOL = 28;

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3();
const _v5 = new THREE.Vector3();
const _spawnPos = new THREE.Vector3();
const _spawnVel = new THREE.Vector3();
const _spawnAng = new THREE.Vector3();
const _q1 = new THREE.Quaternion();
const _e1 = new THREE.Euler();
const _m1 = new THREE.Matrix4();
const _recoilSample = { h: 0, v: 0 };

/** Brass case: a lathed profile with a rim, an extractor groove and a hollow mouth. */
function buildCasingGeometry(caliberRadius: number, length: number): THREE.BufferGeometry {
  const r = caliberRadius;
  const h = length * 0.5;
  const pts = [
    new THREE.Vector2(0, -h),
    new THREE.Vector2(r * 1.14, -h),
    new THREE.Vector2(r * 1.14, -h + length * 0.055),
    new THREE.Vector2(r * 0.86, -h + length * 0.10),
    new THREE.Vector2(r * 1.02, -h + length * 0.16),
    new THREE.Vector2(r * 0.99, h - length * 0.06),
    new THREE.Vector2(r * 0.93, h),
    new THREE.Vector2(r * 0.78, h),
    new THREE.Vector2(r * 0.80, h - length * 0.30),
    new THREE.Vector2(0, h - length * 0.34),
  ];
  const geo = new THREE.LatheGeometry(pts, 12);
  geo.rotateX(-Math.PI / 2);
  return geo;
}

class WeaponInstance implements WeaponRuntime {
  readonly spec: WeaponSpec;
  readonly feel: WeaponFeel;
  readonly visualSpec: WeaponVisual;
  readonly pattern: RecoilPattern;
  readonly view: WeaponViewmodel;

  ammo: number;
  reserve: number;
  aiming = false;
  reloading = false;

  /** Shots since the last trigger release / magazine change, for the pattern. */
  patternIndex = 0;
  bloom = 0;

  constructor(entry: WeaponEntry, view: WeaponViewmodel) {
    this.spec = entry.spec;
    this.feel = entry.feel;
    this.visualSpec = entry.visual;
    this.pattern = entry.recoil;
    this.view = view;
    this.ammo = entry.spec.magSize;
    this.reserve = entry.spec.reserveAmmo;
  }

  get viewmodel(): THREE.Object3D {
    return this.view.root;
  }

  getMuzzleWorld(out: THREE.Vector3): THREE.Vector3 {
    return this.view.muzzleNode.getWorldPosition(out);
  }
}

export class WeaponSystem implements System {
  readonly name = 'weapons';

  private ctx: GameContext | null = null;
  private materials: GunMaterials | null = null;
  private readonly animator = new WeaponAnimator();

  private readonly rig = new THREE.Object3D();
  private readonly weapons: WeaponInstance[] = [];
  private index = 0;
  private pending = -1;

  private scopeOverlay: THREE.Object3D | null = null;
  private scopeOwned: { materials: THREE.Material[]; geometries: THREE.BufferGeometry[] } | null = null;
  private worldFlash: THREE.PointLight | null = null;
  private readonly viewLights: THREE.Light[] = [];
  private lastEnvironment: THREE.Texture | null = null;

  // firing state
  private sinceShot = 999;
  private burstLeft = 0;
  private burstCooldown = 0;
  private cycleTimer = 0;
  private triggerHeld = false;
  private triggerLatched = false;
  private flashTimer = 0;
  private flashSeed = 0;

  // debris pools
  private casingGeo: THREE.BufferGeometry | null = null;
  private casingMat: THREE.MeshStandardMaterial | null = null;
  private readonly casingPool: THREE.Mesh[] = [];
  private readonly magPool = new Map<string, THREE.Object3D[]>();
  private droppedMagPending: WeaponInstance | null = null;

  private readonly unsubs: Array<() => void> = [];
  private readonly motion: MotionInput = {
    yaw: 0, pitch: 0, speed: 0, gaitSpeed: 5.4, stridePhase: 0,
    grounded: true, sprinting: false, crouched: false, verticalVelocity: 0,
  };

  // -- lifecycle -----------------------------------------------------------

  init(ctx: GameContext): void {
    this.ctx = ctx;
    this.materials = new GunMaterials(ctx.quality);

    this.rig.name = 'viewmodel-rig';
    this.rig.matrixAutoUpdate = false;
    this.rig.layers.enable(LAYERS.viewmodel);
    ctx.viewScene.add(this.rig);
    // The view camera renders only the layers it is on; make sure that includes
    // the viewmodel layer whatever else the pipeline decides to do with masks.
    ctx.viewCamera.layers.enable(LAYERS.viewmodel);

    for (const id of DEFAULT_LOADOUT) {
      const entry = getEntry(id);
      if (!entry) continue;
      const view = new WeaponViewmodel(entry.spec, entry.visual, this.materials);
      this.rig.add(view.root);
      this.weapons.push(new WeaponInstance(entry, view));
    }
    if (this.weapons.length === 0) throw new Error('WeaponSystem: no weapons in loadout');

    this.casingGeo = buildCasingGeometry(0.0046, 0.0195);
    this.casingMat = new THREE.MeshStandardMaterial({
      color: 0xb8862f, metalness: 1.0, roughness: 0.26, envMapIntensity: 1.4,
    });
    for (let i = 0; i < CASING_POOL; i++) {
      const m = new THREE.Mesh(this.casingGeo, this.casingMat);
      m.visible = false;
      m.castShadow = false;
      m.frustumCulled = true;
      ctx.scene.add(m);
      this.casingPool.push(m);
    }

    // A real light at the real muzzle: the world must flash, not just the gun.
    this.worldFlash = new THREE.PointLight(0xffc48a, 0, 14, 2);
    this.worldFlash.castShadow = false;
    ctx.scene.add(this.worldFlash);

    this.buildViewLights();

    this.animator.onEvent((e) => this.onAnimEvent(e));

    this.unsubs.push(ctx.events.on('game.pause', (p) => {
      if (p.paused) this.triggerHeld = false;
    }));

    this.equip(0, true);
  }

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;
    for (const w of this.weapons) w.view.dispose();
    this.weapons.length = 0;
    this.rig.removeFromParent();
    for (const m of this.casingPool) m.removeFromParent();
    this.casingPool.length = 0;
    for (const list of this.magPool.values()) for (const o of list) o.removeFromParent();
    this.magPool.clear();
    this.casingGeo?.dispose();
    this.casingMat?.dispose();
    this.worldFlash?.removeFromParent();
    this.worldFlash = null;
    for (const l of this.viewLights) {
      l.removeFromParent();
      l.dispose();
    }
    this.viewLights.length = 0;
    if (this.scopeOwned) {
      for (const m of this.scopeOwned.materials) m.dispose();
      for (const g of this.scopeOwned.geometries) g.dispose();
      this.scopeOwned = null;
    }
    this.scopeOverlay?.removeFromParent();
    this.scopeOverlay = null;
    this.materials?.dispose();
    this.materials = null;
    this.ctx = null;
  }

  /**
   * The viewmodel renders in its own scene, which means the level's lighting
   * does not reach it. Every shipped FPS solves this the same way: a small
   * dedicated three-point rig locked to the camera, so the weapon reads
   * identically wherever the player is standing, with the world's environment
   * map supplying the reflections that tie it back into the scene.
   *
   * Ratios are the studio-photography convention — key : rim : fill of roughly
   * 1 : 0.6 : 0.3 — with the fill cool and the key slightly warm so the bevels
   * pick up a colour difference between their lit and shadowed faces.
   */
  private buildViewLights(): void {
    // Absolute levels sit just under the world sun (5.0 through a 0.05-albedo
    // street) so the weapon never out-exposes the scene behind it; the 1 : 0.55
    // : 0.27 ratio between them is what carries the read.
    const key = new THREE.DirectionalLight(0xfff2e2, 1.55);
    key.position.set(-0.55, 0.86, 0.42);
    const rim = new THREE.DirectionalLight(0xffd6a8, 0.85);
    rim.position.set(0.62, 0.30, -0.92);
    const fill = new THREE.DirectionalLight(0x93b6dc, 0.42);
    fill.position.set(0.74, -0.46, 0.38);

    for (const l of [key, rim, fill]) {
      l.castShadow = false;
      l.layers.enable(LAYERS.viewmodel);
      // Targets default to the origin; parenting both to the rig makes the
      // whole rig camera-relative without any per-frame work.
      this.rig.add(l, l.target);
      this.viewLights.push(l);
    }
    // A hair of ambient so deep recesses (the port cavity, the magwell) are not
    // pure black in a scene with no bounce.
    const amb = new THREE.HemisphereLight(0x9fb4c8, 0x2a2620, 0.32);
    amb.layers.enable(LAYERS.viewmodel);
    this.rig.add(amb);
    this.viewLights.push(amb);
  }

  // -- public surface ------------------------------------------------------

  get current(): WeaponRuntime | null {
    return this.weapons[this.index] ?? null;
  }

  get ammo(): number {
    return this.weapons[this.index]?.ammo ?? 0;
  }

  get reserve(): number {
    return this.weapons[this.index]?.reserve ?? 0;
  }

  get aiming(): boolean {
    return this.animator.aiming;
  }

  /** Current cone half-angle in radians — the HUD sizes the crosshair off this. */
  get spread(): number {
    const w = this.weapons[this.index];
    return w ? this.computeSpread(w) : 0;
  }

  // -- equipping -----------------------------------------------------------

  private equip(index: number, immediate: boolean): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const next = this.weapons[index];
    if (!next) return;

    for (let i = 0; i < this.weapons.length; i++) {
      this.weapons[i]!.view.setVisible(i === index);
    }
    this.index = index;

    this.animator.setWeapon(next.visualSpec, next.feel, next.spec.adsTime);
    // The ADS pose is fully determined by where the optic actually is: put the
    // sight node on the camera axis and let the rest of the weapon follow.
    this.animator.setAdsTarget(
      _v1.set(-next.view.sightPos.x, -next.view.sightPos.y, -next.visualSpec.adsDistance - next.view.sightPos.z),
      _q1.identity(),
    );
    this.animator.startDraw(immediate ? next.spec.drawTime * 0.6 : next.spec.drawTime);

    this.sinceShot = 999;
    this.burstLeft = 0;
    this.burstCooldown = 0;
    this.cycleTimer = 0;
    next.patternIndex = 0;
    next.bloom = 0;
    next.reloading = false;

    this.rebuildScopeOverlay(next);

    ctx.events.emit('weapon.equipped', { weaponId: next.spec.id, ammo: next.ammo, reserve: next.reserve });
    ctx.events.emit('weapon.ammo', { ammo: next.ammo, reserve: next.reserve });
    ctx.events.emit('weapon.ads', { aiming: false, fovScale: 1 });
  }

  private rebuildScopeOverlay(w: WeaponInstance): void {
    const ctx = this.ctx;
    const materials = this.materials;
    if (!ctx || !materials) return;

    if (this.scopeOverlay) {
      this.scopeOverlay.removeFromParent();
      this.scopeOverlay = null;
    }
    if (this.scopeOwned) {
      for (const m of this.scopeOwned.materials) m.dispose();
      for (const g of this.scopeOwned.geometries) g.dispose();
      this.scopeOwned = null;
    }
    if (!w.view.usesScopeOverlay) return;

    // Project the ocular's angular size onto a plane just off the near clip, so
    // the black surround lines up with the tube however far out the pose sits.
    const overlayZ = 0.05;
    const inner = w.view.scopeRadius * (overlayZ / Math.max(0.05, w.visualSpec.adsDistance));
    const built = buildScopeOverlay(materials, inner);
    built.root.position.set(0, 0, -overlayZ);
    this.rig.add(built.root);
    this.scopeOverlay = built.root;
    this.scopeOwned = { materials: built.materials, geometries: built.geometries };
  }

  // -- fixed step ----------------------------------------------------------

  fixedUpdate(step: number, ctx: GameContext): void {
    const w = this.weapons[this.index];
    if (!w) return;

    this.sinceShot += step;
    if (this.burstCooldown > 0) this.burstCooldown = Math.max(0, this.burstCooldown - step);
    if (this.cycleTimer > 0) this.cycleTimer = Math.max(0, this.cycleTimer - step);

    // Bloom decays toward zero on an exponential whose time constant is the
    // weapon's own; standing still tightens the cone, holding the trigger opens it.
    if (w.bloom > 0) {
      w.bloom *= Math.exp(-step / Math.max(0.02, w.feel.bloomDecay));
      if (w.bloom < 1e-5) w.bloom = 0;
    }
    if (!this.triggerHeld && w.patternIndex > 0 && this.sinceShot > 0.35) w.patternIndex = 0;

    this.serviceFiring(ctx, w);
  }

  /**
   * Consumes accumulated time into shots. Timers only ever advance in
   * `fixedUpdate`, so calling this again right after the input poll costs
   * nothing but lets a trigger pull resolve on the frame it happened instead of
   * waiting for the next fixed step.
   */
  private serviceFiring(ctx: GameContext, w: WeaponInstance): void {
    const interval = fireInterval(w.spec);
    let guard = 0;
    while (this.canFire(w) && this.sinceShot >= interval && guard++ < 8) {
      this.sinceShot -= interval;
      this.shoot(ctx, w, interval);
    }
    // Never bank more than one round of credit. Subtracting exactly `interval`
    // per shot keeps sustained fire below this ceiling, so the clamp only ever
    // bites after an idle period — which is precisely when a queued backlog
    // would otherwise dump half a magazine on the first trigger pull.
    if (this.sinceShot > interval) this.sinceShot = interval;
  }

  private canFire(w: WeaponInstance): boolean {
    if (w.ammo <= 0) return false;
    if (w.reloading || this.animator.busy) return false;
    if (this.animator.raising) return false; // sprint-out delay
    if (this.cycleTimer > 0) return false;

    switch (w.spec.fireMode) {
      case 'auto':
        return this.triggerHeld;
      case 'semi':
      case 'bolt':
        return this.triggerHeld && !this.triggerLatched;
      case 'burst':
        if (this.burstLeft > 0) return true;
        return this.triggerHeld && !this.triggerLatched && this.burstCooldown <= 0;
    }
  }

  private computeSpread(w: WeaponInstance): number {
    const ads = this.animator.adsProgress;
    const base = w.spec.spreadHip + (w.spec.spreadAds - w.spec.spreadHip) * ads;
    const m = this.motion;
    const speedT = Math.min(1, m.speed / Math.max(0.5, m.gaitSpeed));
    let mult = 1 + (w.feel.spreadMoveMult - 1) * speedT;
    if (m.crouched && speedT < 0.15) mult *= w.feel.spreadCrouchMult;
    if (!m.grounded) mult *= w.feel.spreadAirMult;
    if (this.animator.raising) mult *= w.feel.sprintOutSpread;
    return base * mult + w.bloom * (1 - ads * 0.75);
  }

  private shoot(ctx: GameContext, w: WeaponInstance, interval: number): void {
    w.ammo--;
    if (w.spec.fireMode === 'burst') {
      if (this.burstLeft <= 0) this.burstLeft = (w.spec.burstCount ?? 3);
      this.burstLeft--;
      if (this.burstLeft <= 0) this.burstCooldown = w.feel.burstCooldown;
    }
    if (w.spec.fireMode !== 'auto') this.triggerLatched = true;
    if (w.spec.fireMode === 'bolt' || w.visualSpec.charging === 'pump') {
      this.cycleTimer = interval * 0.92;
      this.animator.cycleAction(interval * 0.72);
    }

    sampleRecoil(w.pattern, w.patternIndex, _recoilSample);
    w.patternIndex++;

    this.animator.fire(_recoilSample.h, _recoilSample.v, interval);

    // Camera recoil. The rig turns the 0..1 per-frame recovery figure from the
    // spec into its own continuous rate, so we hand it across unchanged.
    const player = ctx.system<System & PlayerLike>('player');
    const adsDamp = 1 - this.animator.adsProgress * 0.30;
    player?.applyRecoil(
      w.spec.recoilVertical * _recoilSample.v * adsDamp,
      w.spec.recoilHorizontal * _recoilSample.h * adsDamp,
      w.spec.recoilRecovery,
    );

    // The viewmodel matrices are one frame stale here (lateUpdate writes them),
    // which is exactly the frame the player was looking at when they pulled the
    // trigger — so this is the correct muzzle, not a lagging one.
    w.view.muzzleNode.updateWorldMatrix(true, false);
    const origin = w.view.muzzleNode.getWorldPosition(_v1);

    const cam = ctx.camera;
    cam.getWorldDirection(_v2);
    const aimPoint = _v3.copy(cam.position).addScaledVector(_v2, AIM_DISTANCE);
    const dir = aimPoint.sub(origin).normalize();
    this.applySpread(dir, this.computeSpread(w));

    w.bloom = Math.min(w.feel.bloomMax, w.bloom + w.feel.bloomPerShot);

    ctx.events.emit('shot.fired', {
      weaponId: w.spec.id,
      origin: origin.clone(),
      direction: dir.clone(),
      local: true,
      shooterId: ctx.localPlayerId,
    });
    ctx.events.emit('weapon.ammo', { ammo: w.ammo, reserve: w.reserve });
    ctx.events.emit('camera.shake', {
      amount: w.feel.shake * adsDamp,
      duration: Math.max(0.08, interval * 1.6),
      frequency: 26,
    });

    this.flashTimer = FLASH_LIFETIME;
    this.flashSeed = Math.random();
    this.ejectCasing(ctx, w);

    if (w.ammo === 0 && w.reserve > 0) this.beginReload(ctx, w);
  }

  /** Uniform disc inside the cone, applied about two axes orthogonal to `dir`. */
  private applySpread(dir: THREE.Vector3, spread: number): void {
    if (spread <= 1e-6) return;
    // sqrt() of a uniform variate gives a uniform *area* distribution inside
    // the cone; without it the shots pile up in the middle.
    const r = spread * Math.sqrt(Math.random());
    const a = Math.random() * Math.PI * 2;
    _v4.set(0, 1, 0);
    if (Math.abs(dir.y) > 0.98) _v4.set(1, 0, 0);
    const right = _v5.crossVectors(dir, _v4).normalize();
    _q1.setFromAxisAngle(right, r * Math.cos(a));
    dir.applyQuaternion(_q1);
    const up = _v4.crossVectors(right, dir).normalize();
    _q1.setFromAxisAngle(up, r * Math.sin(a));
    dir.applyQuaternion(_q1).normalize();
  }

  private ejectCasing(ctx: GameContext, w: WeaponInstance): void {
    const spawner = asDebrisSpawner(ctx.physics);
    if (!spawner) return;
    const mesh = this.casingPool.pop();
    if (!mesh) return;

    w.view.ejectNode.updateWorldMatrix(true, false);
    const pos = w.view.ejectNode.getWorldPosition(_v1);
    _m1.copy(w.view.ejectNode.matrixWorld);
    // Weapon-space right / up / back, in world terms.
    const right = _v2.set(_m1.elements[0]!, _m1.elements[1]!, _m1.elements[2]!).normalize();
    const up = _v3.set(_m1.elements[4]!, _m1.elements[5]!, _m1.elements[6]!).normalize();
    const back = _v4.set(_m1.elements[8]!, _m1.elements[9]!, _m1.elements[10]!).normalize();

    const side = w.visualSpec.ejectSide;
    const speed = w.feel.ejectSpeed;
    const vel = _spawnVel.set(0, 0, 0)
      .addScaledVector(right, side * speed * (0.85 + Math.random() * 0.4))
      .addScaledVector(up, speed * (0.42 + Math.random() * 0.3))
      .addScaledVector(back, speed * (0.10 + Math.random() * 0.18));

    // Inherit the shooter's motion, or casings hang in the air when sprinting.
    const player = ctx.system<System & PlayerLike>('player');
    if (player) vel.add(player.velocity);

    const spin = w.feel.ejectSpin;
    _e1.set(Math.random() * 6.28, Math.random() * 6.28, Math.random() * 6.28);
    mesh.quaternion.setFromEuler(_e1);
    mesh.visible = true;

    spawner.spawnDebris({
      position: _spawnPos.copy(pos),
      velocity: vel,
      angularVelocity: _spawnAng.set(
        (Math.random() * 2 - 1) * spin,
        (Math.random() * 2 - 1) * spin * 0.6,
        (Math.random() * 2 - 1) * spin,
      ),
      radius: 0.006,
      restitution: 0.34,
      friction: 0.55,
      lifetime: 7,
      object: mesh,
      onRetire: (o) => {
        o.visible = false;
        if (this.casingPool.length < CASING_POOL) this.casingPool.push(o as THREE.Mesh);
      },
    });
  }

  // -- reloading -----------------------------------------------------------

  private beginReload(ctx: GameContext, w: WeaponInstance): void {
    if (w.reloading || this.animator.busy) return;
    if (w.reserve <= 0 || w.ammo >= w.spec.magSize) return;
    const empty = w.ammo === 0;
    const duration = empty ? w.spec.reloadEmptyTime : w.spec.reloadTime;
    w.reloading = true;
    this.animator.startReload(duration, empty);
    this.droppedMagPending = w;
    ctx.events.emit('weapon.reload.start', { weaponId: w.spec.id, duration });
  }

  private onAnimEvent(e: string): void {
    const ctx = this.ctx;
    const w = this.weapons[this.index];
    if (!ctx || !w) return;

    switch (e) {
      case 'mag.release':
        this.dropMagazine(ctx, w);
        break;
      case 'mag.seat':
        ctx.events.emit('camera.shake', { amount: 0.045, duration: 0.16, frequency: 30 });
        break;
      case 'reload.done': {
        const want = w.spec.magSize - w.ammo;
        const take = Math.min(want, w.reserve);
        w.ammo += take;
        w.reserve -= take;
        w.reloading = false;
        w.patternIndex = 0;
        ctx.events.emit('weapon.ammo', { ammo: w.ammo, reserve: w.reserve });
        ctx.events.emit('weapon.reload.end', { weaponId: w.spec.id });
        break;
      }
      case 'draw.done':
        break;
      default:
        break;
    }
  }

  /** Hands the spent magazine to the physics debris pool as a real rigid body. */
  private dropMagazine(ctx: GameContext, w: WeaponInstance): void {
    if (this.droppedMagPending !== w) return;
    this.droppedMagPending = null;
    const spawner = asDebrisSpawner(ctx.physics);
    if (!spawner) return;

    let pool = this.magPool.get(w.spec.id);
    if (!pool) { pool = []; this.magPool.set(w.spec.id, pool); }
    let obj = pool.pop();
    if (!obj) {
      const src = w.view.nodes.mag;
      if (src.children.length === 0) return;
      obj = src.clone(true);
      obj.traverse((o) => {
        o.layers.set(LAYERS.default);
        o.frustumCulled = true;
      });
      ctx.scene.add(obj);
    }

    w.view.nodes.mag.updateWorldMatrix(true, false);
    const pos = w.view.nodes.mag.getWorldPosition(_v1);
    _m1.copy(w.view.nodes.mag.matrixWorld);
    const down = _v2.set(-_m1.elements[4]!, -_m1.elements[5]!, -_m1.elements[6]!).normalize();
    const back = _v3.set(_m1.elements[8]!, _m1.elements[9]!, _m1.elements[10]!).normalize();

    obj.quaternion.setFromRotationMatrix(_m1);
    obj.visible = true;

    const vel = _spawnVel.set(0, 0, 0)
      .addScaledVector(down, 1.55 + Math.random() * 0.35)
      .addScaledVector(back, 0.30 + Math.random() * 0.25);
    const player = ctx.system<System & PlayerLike>('player');
    if (player) vel.add(player.velocity);

    spawner.spawnDebris({
      position: _spawnPos.copy(pos),
      velocity: vel,
      angularVelocity: _spawnAng.set(
        (Math.random() * 2 - 1) * 5, (Math.random() * 2 - 1) * 3, (Math.random() * 2 - 1) * 5,
      ),
      radius: 0.030,
      restitution: 0.18,
      friction: 0.85,
      lifetime: 12,
      object: obj,
      onRetire: (o) => {
        o.visible = false;
        this.magPool.get(w.spec.id)?.push(o);
      },
    });
  }

  // -- per-frame -----------------------------------------------------------

  update(time: FrameTime, ctx: GameContext): void {
    let w = this.weapons[this.index];
    if (!w) return;

    // The sky system owns the IBL probe but only installs it on the world
    // scene; without this the weapon's metal has nothing to reflect.
    if (ctx.environment !== this.lastEnvironment) {
      this.lastEnvironment = ctx.environment;
      ctx.viewScene.environment = ctx.environment;
    }

    const player = ctx.system<System & PlayerLike>('player');
    const m = this.motion;
    m.yaw = player?.yaw ?? 0;
    m.pitch = player?.pitch ?? 0;
    m.speed = player?.speed ?? 0;
    m.stridePhase = player?.stridePhase ?? 0;
    m.grounded = player?.grounded ?? true;
    m.sprinting = player?.sprinting ?? false;
    m.crouched = (player?.stance ?? '') === 'crouch';
    m.verticalVelocity = player?.velocity.y ?? 0;
    const gait = player?.gait ?? 'run';
    m.gaitSpeed = gait === 'sprint' ? 7.2 : gait === 'walk' ? 2.4 : 5.4;

    if (!ctx.paused) {
      this.pollInput(ctx, w);
      // pollInput can complete a weapon swap, so re-read before posing.
      w = this.weapons[this.index] ?? w;
      this.serviceFiring(ctx, w);
    }

    const wasAiming = this.animator.aiming;
    const pose = this.animator.update(ctx.paused ? 0 : time.dt, m);

    const nowAiming = this.animator.aiming;
    if (nowAiming !== wasAiming) {
      w.aiming = nowAiming;
      ctx.events.emit('weapon.ads', {
        aiming: nowAiming,
        fovScale: nowAiming ? w.spec.adsFovScale : 1,
      });
    }

    // --- apply the pose to the viewmodel ---------------------------------
    const view = w.view;
    view.root.position.copy(pose.position);
    view.root.quaternion.copy(pose.quaternion);

    view.nodes.mag.position.set(
      view.magRest.x + pose.magOffset.x,
      view.magRest.y + pose.magOffset.y,
      view.magRest.z + pose.magOffset.z,
    );
    view.nodes.mag.rotation.x = pose.magRot;
    view.nodes.mag.visible = pose.magVisible;

    view.nodes.bolt.position.set(view.boltRest.x, view.boltRest.y, view.boltRest.z + pose.boltBack);
    view.nodes.charge.position.set(
      view.chargeRest.x, view.chargeRest.y, view.chargeRest.z + pose.chargeBack,
    );
    view.nodes.trigger.rotation.x = -pose.triggerPull * 0.30;
    // The cover hinges along the bore and swings outboard, so it rotates about
    // Z, not X. The sign follows whichever side the port is on.
    view.nodes.dust.rotation.z = -w.visualSpec.ejectSide * pose.dustOpen * 1.25;
    view.nodes.lhand.position.set(
      view.lhandRest.x + pose.handOffset.x,
      view.lhandRest.y + pose.handOffset.y,
      view.lhandRest.z + pose.handOffset.z,
    );

    // --- muzzle flash ----------------------------------------------------
    this.updateFlash(w, time.dt);

    // --- reticle: brightness and the parallax shift ----------------------
    if (view.reticleMaterial && view.reticleNode) {
      const ads = pose.adsBlend;
      view.reticleMaterial.opacity = 0.34 + 0.66 * ads;
      // Off axis, a collimated dot appears to sit against the far target; the
      // classic viewmodel cheat is to shear it by the residual eye offset.
      const sx = pose.position.x + view.sightPos.x;
      const sy = pose.position.y + view.sightPos.y;
      const k = 0.055 * (1 - ads);
      view.reticleNode.position.set(
        view.reticleRest.x - sx * k,
        view.reticleRest.y - sy * k,
        view.reticleRest.z,
      );
    }

    // --- scope surround --------------------------------------------------
    if (this.scopeOverlay) {
      const ads = pose.adsBlend;
      const show = ads > 0.55;
      this.scopeOverlay.visible = show;
      if (show) {
        // Opens from wide to exact over the last 45% of the ADS blend, so the
        // sight picture irises in rather than popping.
        const t = (ads - 0.55) / 0.45;
        this.scopeOverlay.scale.setScalar(1 + (1 - t) * (1 - t) * 2.6);
      }
    }
  }

  private pollInput(ctx: GameContext, w: WeaponInstance): void {
    const input = ctx.input;

    const fireDown = input.isDown('fire');
    if (!fireDown) this.triggerLatched = false;
    this.triggerHeld = fireDown;
    if (input.wasPressed('fire')) this.triggerLatched = false;

    this.animator.setAiming(input.isDown('aim'));
    this.animator.setSprinting(input.isDown('sprint') && this.motion.speed > 1.0 && this.motion.grounded);

    if (input.wasPressed('reload')) this.beginReload(ctx, w);

    if (input.wasPressed('swapWeapon') && this.pending < 0 && !this.animator.busy) {
      this.pending = (this.index + 1) % this.weapons.length;
      this.animator.startHolster(w.spec.drawTime * 0.55);
    }
    if (this.pending >= 0 && !this.animator.busy) {
      const next = this.pending;
      this.pending = -1;
      this.equip(next, false);
    }
  }

  private updateFlash(w: WeaponInstance, dt: number): void {
    const flash = w.view.flash;
    if (this.flashTimer <= 0) {
      if (flash.root.visible) {
        flash.root.visible = false;
        flash.light.intensity = 0;
        if (this.worldFlash) this.worldFlash.intensity = 0;
      }
      return;
    }

    const first = !flash.root.visible;
    this.flashTimer = Math.max(0, this.flashTimer - dt);
    const k = this.flashTimer / FLASH_LIFETIME;
    flash.root.visible = true;

    if (first) {
      // Three lobes, each rolled independently: no two shots look alike.
      const base = this.flashSeed * Math.PI * 2;
      for (let i = 0; i < flash.lobes.length; i++) {
        const lobe = flash.lobes[i]!;
        lobe.rotation.z = base + i * 2.1 + Math.random() * 1.4;
        const s = w.feel.flashSize * (0.72 + Math.random() * 0.55) * (i === 0 ? 1 : 0.62 - i * 0.13);
        lobe.scale.set(s * (0.85 + Math.random() * 0.3), s, 1);
        lobe.position.z = -0.004 - i * 0.007;
      }
    }
    // Flashes do not fade linearly; they collapse.
    const fall = k * k;
    flash.material.opacity = fall;
    flash.light.intensity = w.feel.flashIntensity * fall * 0.25;

    if (this.worldFlash) {
      w.view.muzzleNode.getWorldPosition(_v1);
      this.worldFlash.position.copy(_v1);
      this.worldFlash.intensity = w.feel.flashIntensity * fall;
      this.worldFlash.distance = 6 + w.feel.flashSize * 40;
    }
  }

  /**
   * Runs after the camera rig has written the final view transform, so the rig
   * node lands exactly on the eye with no one-frame lag.
   */
  lateUpdate(_time: FrameTime, ctx: GameContext): void {
    const cam = ctx.viewCamera;
    cam.updateMatrixWorld();
    this.rig.matrix.copy(cam.matrixWorld);
    this.rig.matrix.decompose(this.rig.position, this.rig.quaternion, this.rig.scale);
    this.rig.matrixWorldNeedsUpdate = true;
    this.rig.updateMatrixWorld(true);
  }
}
