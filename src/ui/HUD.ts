/**
 * The player-facing HUD: a DOM overlay above the canvas.
 *
 * DOM rather than a canvas layer because text has to be pixel-crisp at any DPI
 * and the browser's text rasteriser is better than anything we would write. The
 * cost of DOM is layout thrash, so the rule throughout this file is: build the
 * tree once, cache every reference, and only touch a node when the value it
 * displays actually changed. Per-frame writes are confined to `transform`,
 * `opacity` and custom properties, all of which composite without reflow.
 *
 * Everything the HUD knows arrives on the event bus or through `ctx`. It never
 * reaches into another system's internals beyond the structural contracts
 * declared below.
 */
import * as THREE from 'three';
import type { FrameTime, GameContext, System, WeaponRuntime } from '../core/Contracts';
import {
  PALETTE, EASE, clamp, clamp01, damp, el, svgNode, setText, setClass, weaponGlyph,
  bloodDataURI, injectUIStyles, releaseUIStyles,
} from './Style';

// --- structural views onto other systems (never their concrete classes) -----

interface WeaponsLike extends System {
  readonly current: WeaponRuntime | null;
  readonly spread: number;
  readonly aiming: boolean;
}

interface LevelLike extends System {
  getNavBounds(): THREE.Box3;
}

// --- module-scope scratch: nothing in the hot path allocates ---------------

const _box = new THREE.Box3();
const _sphere = new THREE.Sphere();
const _mat = new THREE.Matrix4();
const RAD2DEG = 180 / Math.PI;

const KILLFEED_TTL = 5.0;
const KILLFEED_MAX = 5;
const CONTACT_TTL = 4.5;
const DIR_POOL = 5;
const MM_PIPS = 12;
const COMPASS_PIPS = 8;
const COMPASS_PPD = 4.6; // px per degree of the compass strip
const COMPASS_STEP = 5;

/** Deterministic callsigns so the killfeed reads like a lobby, not like ids. */
const CALLSIGNS = [
  'WRAITH', 'SABLE', 'ONYX', 'MERIDIAN', 'TALON', 'ROOK', 'HALO', 'KESTREL',
  'VECTOR', 'ASHFALL', 'DRAKE', 'NOMAD', 'CINDER', 'HOLLOW', 'VANTAGE', 'RIPTIDE',
];

function callsign(id: number): string {
  const n = CALLSIGNS[Math.abs(id * 2654435761) % CALLSIGNS.length];
  return `${n}-${String(Math.abs(id) % 90 + 10)}`;
}

// ---------------------------------------------------------------------------
// Minimap raster helpers. These run once at init, never per frame.
// ---------------------------------------------------------------------------

function markLine(buf: Uint8Array, w: number, h: number, ax: number, ay: number, bx: number, by: number): void {
  let x0 = Math.round(ax);
  let y0 = Math.round(ay);
  const x1 = Math.round(bx);
  const y1 = Math.round(by);
  const dx = Math.abs(x1 - x0);
  const dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  // Long spans only happen for geometry we already rejected; the guard keeps a
  // pathological transform from locking the tab.
  let guard = dx - dy + 4;
  if (guard > 8192) return;
  for (;;) {
    if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h) buf[y0 * w + x0] = 255;
    if (x0 === x1 && y0 === y1) break;
    if (--guard < 0) break;
    const e2 = err * 2;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

function fillTri(
  buf: Uint8Array, w: number, h: number,
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
): void {
  const minX = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
  const maxX = Math.min(w - 1, Math.ceil(Math.max(ax, bx, cx)));
  const minY = Math.max(0, Math.floor(Math.min(ay, by, cy)));
  const maxY = Math.min(h - 1, Math.ceil(Math.max(ay, by, cy)));
  if (maxX < minX || maxY < minY) return;

  const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(det) < 1e-7) return;
  const inv = 1 / det;

  // Huge triangles (a whole terrain quad) get a coarse fill: 4x fewer samples
  // and a 2x2 write, which is invisible at the scale the map is displayed.
  const area = (maxX - minX + 1) * (maxY - minY + 1);
  const step = area > 24000 ? 2 : 1;

  for (let y = minY; y <= maxY; y += step) {
    const py = y + 0.5;
    for (let x = minX; x <= maxX; x += step) {
      const px = x + 0.5;
      const l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) * inv;
      if (l0 < -0.002) continue;
      const l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) * inv;
      if (l1 < -0.002) continue;
      if (l0 + l1 > 1.002) continue;
      buf[y * w + x] = 255;
      if (step === 2) {
        if (x + 1 < w) buf[y * w + x + 1] = 255;
        if (y + 1 < h) {
          buf[(y + 1) * w + x] = 255;
          if (x + 1 < w) buf[(y + 1) * w + x + 1] = 255;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------

interface KillfeedRow {
  node: HTMLElement;
  born: number;
  anim: Animation | null;
}

interface Contact {
  ttl: number;
  x: number;
  z: number;
}

interface DirIndicator {
  node: HTMLElement;
  anim: Animation | null;
}

export class HUDSystem implements System {
  readonly name = 'hud';

  private ctx: GameContext | null = null;
  private root: HTMLElement | null = null;
  private readonly unsubs: (() => void)[] = [];

  // --- cached elements -----------------------------------------------------
  private xh!: HTMLElement;
  private hm!: HTMLElement;
  private hmRing!: HTMLElement;
  private dmg!: HTMLElement;
  private bloodA!: HTMLElement;
  private bloodB!: HTMLElement;
  private beat!: HTMLElement;
  private dirs!: HTMLElement;
  private compass!: HTMLElement;
  private compassTrack!: HTMLElement;
  private killfeed!: HTMLElement;
  private ammoRoot!: HTMLElement;
  private ammoName!: HTMLElement;
  private ammoGlyphSlot!: HTMLElement;
  private ammoCur!: HTMLElement;
  private ammoRes!: HTMLElement;
  private ammoMag!: HTMLElement;
  private ammoMode!: HTMLElement;
  private reloadBar!: HTMLElement;
  private prompt!: HTMLElement;
  private notify!: HTMLElement;
  private notifyKind!: HTMLElement;
  private notifyText!: HTMLElement;
  private confirm!: HTMLElement;
  private confirmKind!: HTMLElement;
  private confirmPts!: HTMLElement;
  private streak!: HTMLElement;
  private streakNum!: HTMLElement;
  private death!: HTMLElement;
  private deathSub!: HTMLElement;
  private deathBy!: HTMLElement;
  private deathCount!: HTMLElement;
  private mmWorld!: HTMLElement;
  private mmCards!: HTMLElement;
  private mmCardEls: HTMLElement[] = [];
  private mmTag!: HTMLElement;
  private mmCanvas: HTMLCanvasElement | null = null;

  private readonly dirPool: DirIndicator[] = [];
  private dirNext = 0;
  private readonly mmPips: HTMLElement[] = [];
  private readonly compassPips: HTMLElement[] = [];
  private readonly rows: KillfeedRow[] = [];

  // --- animations we own and must cancel on dispose ------------------------
  private hmAnim: Animation | null = null;
  private confirmAnim: Animation | null = null;
  private notifyAnim: Animation | null = null;
  private streakAnim: Animation | null = null;
  private reloadAnim: Animation | null = null;

  // --- state ---------------------------------------------------------------
  private now = 0;
  private weapons: WeaponsLike | null = null;

  private gap = 9;
  private gapKick = 0;
  private lastGapWritten = -1;
  private aiming = false;
  private sprinting = false;
  private xhHidden = false;

  private health = 100;
  private maxHealth = 100;
  private hurt = 0; // 0..1 damage vignette intensity
  private lastHurtWritten = -1;
  private lastBloodWritten = -1;
  private beating = false;

  private ammo = 0;
  private reserve = 0;
  private magSize = 0;
  private magTicks: HTMLElement[] = [];
  private lastAmmoWritten = -1;
  private lastReserveWritten = -1;
  private reloading = false;

  private streakCount = 0;
  private score = 0;

  private dead = false;
  private deathTimer = 0;

  private readonly contacts = new Map<number, Contact>();
  private readonly notifyQueue: { text: string; kind: string }[] = [];
  private notifyHold = 0;

  private heading = 0;
  private eyeX = 0;
  private eyeZ = 0;
  private lastHeadingWritten = 999;
  private lastCompassWritten = 999;
  private compassHalf = 240;

  // minimap world->pixel mapping (display pixels, before rotation)
  private mmReady = false;
  private mmCentreX = 0;
  private mmCentreZ = 0;
  private mmScale = 1; // display px per metre
  private lastMmX = 1e9;
  private lastMmZ = 1e9;

  private hudScale = 1;
  private viewHeight = 1080;
  private tanHalfFov = Math.tan((80 * 0.5 * Math.PI) / 180);
  private slowTimer = 0;

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;
    injectUIStyles();

    const root = el('div', 'cod-ui cod-hud');
    root.setAttribute('aria-hidden', 'true');
    document.body.appendChild(root);
    this.root = root;

    this.buildDamage(root);
    this.buildCrosshair(root);
    this.buildHitmarker(root);
    this.buildDirections(root);
    this.buildCompass(root);
    this.buildKillfeed(root);
    this.buildAmmo(root);
    this.buildMinimap(root, ctx);
    this.buildNotify(root);
    this.buildConfirm(root);
    this.buildDeath(root);

    this.weapons = ctx.system<WeaponsLike>('weapons') ?? null;
    this.syncWeapon();
    this.resize(ctx.width, ctx.height);
    this.bind(ctx);
    this.pushNotify('Clear the corridor', 'objective');
  }

  dispose(): void {
    for (const off of this.unsubs) off();
    this.unsubs.length = 0;

    for (const a of [this.hmAnim, this.confirmAnim, this.notifyAnim, this.streakAnim, this.reloadAnim]) {
      a?.cancel();
    }
    this.hmAnim = this.confirmAnim = this.notifyAnim = this.streakAnim = this.reloadAnim = null;
    for (const d of this.dirPool) d.anim?.cancel();
    for (const r of this.rows) r.anim?.cancel();
    this.rows.length = 0;
    this.dirPool.length = 0;
    this.mmPips.length = 0;
    this.compassPips.length = 0;
    this.magTicks.length = 0;
    this.mmCardEls.length = 0;
    this.contacts.clear();

    if (this.mmCanvas) {
      this.mmCanvas.width = 0;
      this.mmCanvas.height = 0;
      this.mmCanvas = null;
    }
    this.root?.remove();
    this.root = null;
    this.ctx = null;
    releaseUIStyles();
  }

  resize(width: number, height: number): void {
    this.viewHeight = Math.max(1, height);
    this.hudScale = clamp(Math.min(width / 1920, height / 1080), 0.78, 1.4);
    this.root?.style.setProperty('--hs', this.hudScale.toFixed(3));
    this.lastGapWritten = -1;
    // The only layout read in the whole HUD, and it happens once per resize.
    if (this.compass) {
      this.compassHalf = this.compass.clientWidth * 0.5;
      this.lastCompassWritten = 999;
    }
  }

  // =========================================================================
  // Event wiring
  // =========================================================================

  private bind(ctx: GameContext): void {
    const ev = ctx.events;
    const u = this.unsubs;

    u.push(ev.on('weapon.equipped', () => {
      this.syncWeapon();
    }));

    u.push(ev.on('weapon.ammo', (p) => {
      this.ammo = p.ammo;
      this.reserve = p.reserve;
      this.paintAmmo();
    }));

    u.push(ev.on('weapon.reload.start', (p) => {
      this.reloading = true;
      setClass(this.ammoRoot, 'reloading', true);
      this.reloadAnim?.cancel();
      this.reloadAnim = this.reloadBar.animate(
        [{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }],
        { duration: Math.max(120, p.duration * 1000), easing: 'linear', fill: 'forwards' },
      );
      setClass(this.prompt, 'on', false);
    }));

    u.push(ev.on('weapon.reload.end', () => {
      this.reloading = false;
      setClass(this.ammoRoot, 'reloading', false);
      this.reloadAnim?.cancel();
      this.reloadAnim = null;
    }));

    u.push(ev.on('weapon.ads', (p) => {
      this.aiming = p.aiming;
      this.updateCrosshairVisibility();
    }));

    u.push(ev.on('player.state', (p) => {
      this.maxHealth = p.maxHealth || 100;
      const prev = this.health;
      this.health = p.health;
      if (this.health > prev + 0.01 && this.dead) this.clearDeath();
      this.sprinting = p.sprinting;
      this.updateCrosshairVisibility();
    }));

    u.push(ev.on('player.damaged', (p) => {
      this.health = p.health;
      const frac = clamp01(p.amount / 34);
      this.hurt = clamp01(this.hurt + 0.34 + frac * 0.5);
      this.spawnDirIndicator(p.fromDirection.x, p.fromDirection.z);
    }));

    u.push(ev.on('ui.hitmarker', (p) => {
      this.flashHitmarker(p.headshot, p.lethal);
    }));

    u.push(ev.on('shot.fired', (p) => {
      if (p.local) {
        this.gapKick = Math.min(this.gapKick + 4.5, 26);
      } else {
        this.noteContact(p.shooterId, p.origin.x, p.origin.z);
      }
    }));

    u.push(ev.on('damage.dealt', (p) => {
      if (p.attackerId === ctx.localPlayerId && p.targetId !== ctx.localPlayerId) {
        this.noteContact(p.targetId, p.point.x, p.point.z);
      }
    }));

    u.push(ev.on('entity.killed', (p) => {
      this.onKill(ctx, p.entityId, p.killerId, p.weaponId, p.headshot);
    }));

    u.push(ev.on('ui.notify', (p) => {
      this.pushNotify(p.text, p.kind ?? 'info');
    }));

    u.push(ev.on('game.over', (p) => {
      if (!p.won) this.showDeath();
    }));
  }

  // =========================================================================
  // Build
  // =========================================================================

  private buildCrosshair(root: HTMLElement): void {
    const xh = el('div', 'cod-xh', root);
    el('i', 'v up', xh);
    el('i', 'v dn', xh);
    el('i', 'h lf', xh);
    el('i', 'h rt', xh);
    el('b', '', xh);
    this.xh = xh;

    this.prompt = el('div', 'cod-prompt', root);
    this.prompt.innerHTML = 'Reload <em>[R]</em>';
  }

  private buildHitmarker(root: HTMLElement): void {
    const hm = el('div', 'cod-hm', root);
    el('i', 'a', hm);
    el('i', 'b', hm);
    el('i', 'c', hm);
    el('i', 'd', hm);
    this.hmRing = el('u', '', hm);
    this.hm = hm;
  }

  private buildDamage(root: HTMLElement): void {
    this.bloodA = el('div', 'cod-blood', root);
    this.bloodB = el('div', 'cod-blood b', root);
    this.bloodA.style.backgroundImage = `url(${bloodDataURI(1)})`;
    this.bloodB.style.backgroundImage = `url(${bloodDataURI(7)})`;
    this.beat = el('div', 'cod-beat', root);
    this.dmg = el('div', 'cod-dmg', root);
  }

  private buildDirections(root: HTMLElement): void {
    const wrap = el('div', 'cod-dirs', root);
    for (let i = 0; i < DIR_POOL; i++) {
      const node = el('div', 'cod-dir', wrap);
      const svg = svgNode('svg', { viewBox: '0 0 280 280' }, node);
      // 118px radius arc about the 280px box centre, spanning +/-31 degrees:
      // 140 +/- 118*sin31 horizontally, 140 - 118*cos31 vertically.
      // Three stacked strokes: a dark backing for legibility over bright walls,
      // the red body, and a hot inner line so the arc reads at a glance.
      const d = 'M 79.2 38.9 A 118 118 0 0 1 200.8 38.9';
      svgNode('path', { d, class: 'o' }, svg);
      svgNode('path', { d, class: 'i' }, svg);
      svgNode('path', { d, class: 'g' }, svg);
      this.dirPool.push({ node, anim: null });
    }
    this.dirs = wrap;
  }

  private buildCompass(root: HTMLElement): void {
    const c = el('div', 'cod-compass', root);
    el('div', 'rule', c);
    const track = el('div', 'track', c);
    track.style.width = `${360 * 2 * COMPASS_PPD}px`;

    const cards: Record<number, string> = {
      0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW',
    };
    for (let d = 0; d <= 720; d += COMPASS_STEP) {
      const deg = d % 360;
      const x = d * COMPASS_PPD;
      const card = cards[deg];
      if (card) {
        const lb = el('div', `lb card${deg === 0 ? ' n' : ''}`, track, card);
        lb.style.left = `${x}px`;
      } else if (deg % 15 === 0) {
        const lb = el('div', 'lb', track, String(deg));
        lb.style.left = `${x}px`;
        const tk = el('div', 'tk mj', track);
        tk.style.left = `${x}px`;
      } else {
        const tk = el('div', 'tk', track);
        tk.style.left = `${x}px`;
      }
    }
    for (let i = 0; i < COMPASS_PIPS; i++) {
      const pip = el('div', 'pip', track);
      this.compassPips.push(pip);
    }
    el('div', 'caret', c);
    this.compass = c;
    this.compassTrack = track;
  }

  private buildKillfeed(root: HTMLElement): void {
    this.killfeed = el('div', 'cod-corner cod-killfeed', root);
  }

  private buildAmmo(root: HTMLElement): void {
    const a = el('div', 'cod-corner cod-ammo', root);
    const name = el('div', 'name', a);
    this.ammoGlyphSlot = el('span', 'g', name);
    const label = el('span', 'n', name, 'NO WEAPON');
    const nums = el('div', 'nums', a);
    this.ammoCur = el('span', 'cur', nums, '0');
    el('span', 'sep', nums, '/');
    this.ammoRes = el('span', 'res', nums, '0');
    this.ammoMag = el('div', 'mag', a);
    this.ammoMode = el('div', 'mode', a, '');
    const rl = el('div', 'rl', a);
    this.reloadBar = el('i', '', rl);

    const streak = el('div', 'cod-corner cod-streak', root);
    el('div', 'lb', streak, 'STREAK');
    el('div', 'bar', streak);
    this.streakNum = el('div', 'n', streak, '0');
    this.streak = streak;

    this.ammoRoot = a;
    this.ammoName = label;
  }

  private buildNotify(root: HTMLElement): void {
    const n = el('div', 'cod-notify', root);
    this.notifyKind = el('div', 'kind', n, '');
    this.notifyText = el('div', 'txt', n, '');
    el('div', 'rule', n);
    this.notify = n;
  }

  private buildConfirm(root: HTMLElement): void {
    const c = el('div', 'cod-confirm', root);
    this.confirmKind = el('div', 'k', c, 'ELIMINATED');
    this.confirmPts = el('div', 'p', c, '+100');
    this.confirm = c;
  }

  private buildDeath(root: HTMLElement): void {
    const d = el('div', 'cod-death', root);
    this.deathSub = el('div', 's', d, 'KILLED IN ACTION');
    this.deathBy = el('div', 't', d, 'YOU DIED');
    el('hr', '', d);
    this.deathCount = el('div', 'c', d, 'RESPAWNING');
    this.death = d;
  }

  // =========================================================================
  // Minimap
  // =========================================================================

  private buildMinimap(root: HTMLElement, ctx: GameContext): void {
    const wrap = el('div', 'cod-corner cod-minimap', root);
    const frame = el('div', 'cod-mm-frame', wrap);
    const world = el('div', 'cod-mm-world', frame);
    const cards = el('div', 'cod-mm-cards', frame);
    const you = el('div', 'cod-mm-you', frame);
    el('div', 'cone', you);
    el('div', 'arw', you);
    el('div', 'cod-mm-grade', frame);
    const tag = el('div', 'cod-mm-tag', wrap, 'GRID A00');

    const R = 76;
    const dirs: [string, number, number][] = [['N', 0, -R], ['E', R, 0], ['S', 0, R], ['W', -R, 0]];
    for (const [t, dx, dy] of dirs) {
      const s = el('span', t === 'N' ? 'n' : '', cards, t);
      s.dataset.dx = String(dx);
      s.dataset.dy = String(dy);
      s.style.transform = `translate(${dx}px,${dy}px) translate(-50%,-50%)`;
      this.mmCardEls.push(s);
    }

    for (let i = 0; i < MM_PIPS; i++) {
      this.mmPips.push(el('div', 'cod-mm-pip', world));
    }

    this.mmWorld = world;
    this.mmCards = cards;
    this.mmTag = tag;

    this.rasterizeLevel(ctx, world);
  }

  /**
   * Renders the level's top-down silhouette to a canvas exactly once.
   *
   * Rather than asking the level for a floorplan it does not publish, this
   * projects the scene's triangles: near-horizontal ones at ground height
   * become walkable floor, near-horizontal ones above a storey become upper
   * decks, and vertical ones become walls. Walls are drawn as edges as well as
   * fills because a wall seen from directly above is a sliver that pixel-centre
   * coverage would miss entirely.
   */
  private rasterizeLevel(ctx: GameContext, world: HTMLElement): void {
    const level = ctx.system<LevelLike>('level');
    let bounds: THREE.Box3;
    try {
      bounds = level?.getNavBounds ? level.getNavBounds() : _box.setFromObject(ctx.scene).clone();
    } catch {
      bounds = new THREE.Box3(new THREE.Vector3(-50, -2, -50), new THREE.Vector3(50, 30, 50));
    }
    if (!isFinite(bounds.min.x) || bounds.max.x <= bounds.min.x) {
      bounds = new THREE.Box3(new THREE.Vector3(-50, -2, -50), new THREE.Vector3(50, 30, 50));
    }

    const cx = (bounds.min.x + bounds.max.x) * 0.5;
    const cz = (bounds.min.z + bounds.max.z) * 0.5;
    const span = Math.max(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z) * 1.04;

    const q = ctx.quality.preset;
    const res = q === 'low' ? 384 : q === 'medium' ? 512 : q === 'ultra' ? 1024 : 768;
    const ppm = res / span;

    const floorBuf = new Uint8Array(res * res);
    const upperBuf = new Uint8Array(res * res);
    const wallBuf = new Uint8Array(res * res);

    const toX = (x: number): number => (x - cx) * ppm + res * 0.5;
    const toZ = (z: number): number => (z - cz) * ppm + res * 0.5;

    const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const deadline = now() + 240;
    let budget = 900000;

    ctx.scene.updateMatrixWorld(true);

    const stack: THREE.Object3D[] = [ctx.scene];
    while (stack.length > 0) {
      const obj = stack.pop();
      if (!obj || !obj.visible) continue;
      for (let i = obj.children.length - 1; i >= 0; i--) stack.push(obj.children[i]);

      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) continue;
      if (/sky|cloud|sun|moon|star|scope|overlay|viewmodel|soldier|tracer|decal|particle/i.test(obj.name)) continue;

      const geo = mesh.geometry;
      const pos = geo?.getAttribute('position') as THREE.BufferAttribute | undefined;
      if (!pos || pos.itemSize < 3) continue;

      if (!geo.boundingSphere) geo.computeBoundingSphere();
      const bs = geo.boundingSphere;
      if (bs) {
        _sphere.copy(bs).applyMatrix4(mesh.matrixWorld);
        // Sky domes and fog shells dwarf the playable area and carry no map
        // information. The threshold is generous so an oversized terrain plane
        // — which does carry the ground mask — still gets rasterised.
        if (_sphere.radius > span * 5) continue;
      }

      const inst = mesh as THREE.InstancedMesh;
      if (inst.isInstancedMesh) {
        if (!geo.boundingBox) geo.computeBoundingBox();
        const bb = geo.boundingBox;
        if (!bb) continue;
        const count = Math.min(inst.count, 4000);
        for (let i = 0; i < count; i++) {
          inst.getMatrixAt(i, _mat);
          _mat.premultiply(mesh.matrixWorld);
          const e = _mat.elements;
          const yLo = bb.min.y * e[5] + e[13];
          const yHi = bb.max.y * e[5] + e[13];
          if (yHi - yLo < 0.75 || yHi < 0.7) continue;
          // Footprint of the oriented bounding box, bottom face.
          const px: number[] = [];
          const pz: number[] = [];
          const xs = [bb.min.x, bb.max.x];
          const zs = [bb.min.z, bb.max.z];
          const order = [[0, 0], [1, 0], [1, 1], [0, 1]];
          for (const [a, b] of order) {
            const lx = xs[a];
            const lz = zs[b];
            const ly = bb.min.y;
            px.push(toX(e[0] * lx + e[4] * ly + e[8] * lz + e[12]));
            pz.push(toZ(e[2] * lx + e[6] * ly + e[10] * lz + e[14]));
          }
          for (let k = 0; k < 4; k++) {
            const n = (k + 1) & 3;
            markLine(wallBuf, res, res, px[k], pz[k], px[n], pz[n]);
          }
          if (yHi > 2.4) fillTri(upperBuf, res, res, px[0], pz[0], px[1], pz[1], px[2], pz[2]);
        }
        continue;
      }

      const index = geo.getIndex();
      const triCount = Math.floor((index ? index.count : pos.count) / 3);
      if (triCount <= 0) continue;

      const e = mesh.matrixWorld.elements;
      const arr = pos.array as ArrayLike<number>;
      const idx = index ? (index.array as ArrayLike<number>) : null;
      const stride = pos.itemSize;

      for (let t = 0; t < triCount; t++) {
        if (--budget < 0) break;
        if ((t & 16383) === 0 && t > 0 && now() > deadline) break;
        const i0 = (idx ? idx[t * 3] : t * 3) * stride;
        const i1 = (idx ? idx[t * 3 + 1] : t * 3 + 1) * stride;
        const i2 = (idx ? idx[t * 3 + 2] : t * 3 + 2) * stride;

        const lx0 = arr[i0], ly0 = arr[i0 + 1], lz0 = arr[i0 + 2];
        const lx1 = arr[i1], ly1 = arr[i1 + 1], lz1 = arr[i1 + 2];
        const lx2 = arr[i2], ly2 = arr[i2 + 1], lz2 = arr[i2 + 2];

        const ax = e[0] * lx0 + e[4] * ly0 + e[8] * lz0 + e[12];
        const ay = e[1] * lx0 + e[5] * ly0 + e[9] * lz0 + e[13];
        const az = e[2] * lx0 + e[6] * ly0 + e[10] * lz0 + e[14];
        const bx = e[0] * lx1 + e[4] * ly1 + e[8] * lz1 + e[12];
        const by = e[1] * lx1 + e[5] * ly1 + e[9] * lz1 + e[13];
        const bz = e[2] * lx1 + e[6] * ly1 + e[10] * lz1 + e[14];
        const cx2 = e[0] * lx2 + e[4] * ly2 + e[8] * lz2 + e[12];
        const cy2 = e[1] * lx2 + e[5] * ly2 + e[9] * lz2 + e[13];
        const cz2 = e[2] * lx2 + e[6] * ly2 + e[10] * lz2 + e[14];

        const maxY = Math.max(ay, by, cy2);
        const minY = Math.min(ay, by, cy2);
        if (minY > 17 || maxY < -3) continue;

        const ux = bx - ax, uy = by - ay, uz = bz - az;
        const vx = cx2 - ax, vy = cy2 - ay, vz = cz2 - az;
        const nx = uy * vz - uz * vy;
        const ny = uz * vx - ux * vz;
        const nz = ux * vy - uy * vx;
        const nl = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (nl < 1e-9) continue;
        const flat = Math.abs(ny) / nl;

        const sx0 = toX(ax), sz0 = toZ(az);
        const sx1 = toX(bx), sz1 = toZ(bz);
        const sx2 = toX(cx2), sz2 = toZ(cz2);
        if (Math.max(sx0, sx1, sx2) < 0 || Math.min(sx0, sx1, sx2) > res) continue;
        if (Math.max(sz0, sz1, sz2) < 0 || Math.min(sz0, sz1, sz2) > res) continue;

        if (flat > 0.7) {
          if (maxY <= 2.1) {
            fillTri(floorBuf, res, res, sx0, sz0, sx1, sz1, sx2, sz2);
          } else if (maxY <= 16) {
            fillTri(upperBuf, res, res, sx0, sz0, sx1, sz1, sx2, sz2);
          }
        } else if (flat < 0.62 && maxY > 0.85) {
          markLine(wallBuf, res, res, sx0, sz0, sx1, sz1);
          markLine(wallBuf, res, res, sx1, sz1, sx2, sz2);
          markLine(wallBuf, res, res, sx2, sz2, sx0, sz0);
        }
      }
      if (budget < 0 || now() > deadline) break;
    }

    // --- compose -----------------------------------------------------------
    const canvas = document.createElement('canvas');
    canvas.width = res;
    canvas.height = res;
    const g = canvas.getContext('2d');
    if (!g) return;

    const rgb = (hex: string): [number, number, number] => {
      const v = parseInt(hex.slice(1), 16);
      return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
    };
    const cVoid = rgb(PALETTE.mapVoid);
    const cFloor = rgb(PALETTE.mapFloor);
    const cUpper = rgb(PALETTE.mapUpper);
    const cWall = rgb(PALETTE.mapWall);

    const img = g.createImageData(res, res);
    const d = img.data;
    let seed = 0x2545f491;
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const i = y * res + x;
        let r: number, gg: number, b: number, a: number;
        if (wallBuf[i]) {
          r = cWall[0]; gg = cWall[1]; b = cWall[2]; a = 235;
        } else if (upperBuf[i]) {
          r = cUpper[0]; gg = cUpper[1]; b = cUpper[2]; a = 232;
        } else if (floorBuf[i]) {
          r = cFloor[0]; gg = cFloor[1]; b = cFloor[2]; a = 224;
        } else {
          r = cVoid[0]; gg = cVoid[1]; b = cVoid[2]; a = 150;
        }
        if (!wallBuf[i] && (floorBuf[i] || upperBuf[i])) {
          // Drop shadow to the lower-right of walls: gives the plan relief.
          const up = y > 0 ? wallBuf[i - res] : 0;
          const lf = x > 0 ? wallBuf[i - 1] : 0;
          if (up || lf) {
            r *= 0.55; gg *= 0.55; b *= 0.55;
          }
        }
        seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
        const n = ((seed >>> 0) % 1000) / 1000 - 0.5;
        const o = i * 4;
        d[o] = clamp(r + n * 9, 0, 255);
        d[o + 1] = clamp(gg + n * 9, 0, 255);
        d[o + 2] = clamp(b + n * 9, 0, 255);
        d[o + 3] = a;
      }
    }
    g.putImageData(img, 0, 0);

    // Grid every 10 m — the tactical-map read that sells it as a map.
    g.strokeStyle = PALETTE.mapGrid;
    g.lineWidth = 1;
    g.beginPath();
    for (let m = Math.ceil(bounds.min.x / 10) * 10; m <= bounds.max.x; m += 10) {
      const px = Math.round(toX(m)) + 0.5;
      g.moveTo(px, 0);
      g.lineTo(px, res);
    }
    for (let m = Math.ceil(bounds.min.z / 10) * 10; m <= bounds.max.z; m += 10) {
      const pz = Math.round(toZ(m)) + 0.5;
      g.moveTo(0, pz);
      g.lineTo(res, pz);
    }
    g.stroke();

    const display = 3.9; // display px per metre inside the minimap frame
    const shown = res * (display / ppm);
    canvas.style.width = `${shown.toFixed(2)}px`;
    canvas.style.height = `${shown.toFixed(2)}px`;
    canvas.style.left = `${(-shown * 0.5).toFixed(2)}px`;
    canvas.style.top = `${(-shown * 0.5).toFixed(2)}px`;
    world.insertBefore(canvas, world.firstChild);

    this.mmCanvas = canvas;
    this.mmCentreX = cx;
    this.mmCentreZ = cz;
    this.mmScale = display;
    this.mmReady = true;
  }

  // =========================================================================
  // Per-frame
  // =========================================================================

  update(time: FrameTime, ctx: GameContext): void {
    if (!this.root) return;
    this.now += time.dt;
    setClass(this.root, 'muted', ctx.paused);

    this.updateCrosshair(time.dt, ctx);
    this.updateDamage(time.dt);
    this.updateNotify(time.dt);
    this.updateContacts(time.dt, ctx);

    if (this.dead) {
      this.deathTimer += time.dt;
      const s = Math.max(0, 3 - this.deathTimer);
      setText(this.deathCount, s > 0.05 ? `RESPAWNING IN ${s.toFixed(1)}` : 'RESPAWNING');
    }

    this.slowTimer += time.dt;
    if (this.slowTimer >= 0.1) {
      this.slowTimer = 0;
      this.pruneKillfeed();
      this.updatePrompt();
    }
  }

  /** Camera-dependent work: the rig has finished writing the camera by now. */
  lateUpdate(_time: FrameTime, ctx: GameContext): void {
    if (!this.root) return;
    const e = ctx.camera.matrixWorld.elements;
    this.heading = Math.atan2(-e[8], e[10]);
    this.eyeX = e[12];
    this.eyeZ = e[14];
    this.updateCompass();
    this.updateMinimap();
  }

  // --- crosshair -----------------------------------------------------------

  private updateCrosshair(dt: number, ctx: GameContext): void {
    const fov = ctx.camera.fov;
    this.tanHalfFov = Math.tan((fov * 0.5 * Math.PI) / 180);

    let spread = 0;
    const w = this.weapons;
    if (w && typeof w.spread === 'number') spread = w.spread;

    // Angular half-cone -> screen pixels, exactly as the bullet will land.
    const pxPerRad = (this.viewHeight * 0.5) / this.tanHalfFov;
    const target = clamp(Math.tan(spread) * pxPerRad, 4, 190) + this.gapKick;

    // Asymmetric response: bloom snaps open, then settles back slowly. Linear
    // interpolation in both directions feels like rubber.
    const rate = target > this.gap ? 34 : 11;
    this.gap = damp(this.gap, target, rate, dt);
    this.gapKick = damp(this.gapKick, 0, 7.5, dt);

    const q = Math.round(this.gap * 2) * 0.5;
    if (q !== this.lastGapWritten) {
      this.lastGapWritten = q;
      this.xh.style.setProperty('--gap', `${q}px`);
    }
  }

  private updateCrosshairVisibility(): void {
    setClass(this.xh, 'ads', this.aiming);
    const hide = this.sprinting && !this.aiming;
    if (hide !== this.xhHidden) {
      this.xhHidden = hide;
      setClass(this.xh, 'hide', hide);
    }
  }

  // --- hitmarker -----------------------------------------------------------

  private flashHitmarker(headshot: boolean, lethal: boolean): void {
    const hm = this.hm;
    setClass(hm, 'head', headshot && !lethal);
    setClass(hm, 'lethal', lethal);
    this.hmAnim?.cancel();
    this.hmAnim = hm.animate(
      [
        { opacity: 1, transform: 'scale(1.62)' },
        { opacity: 1, transform: 'scale(1)', offset: 0.24 },
        { opacity: 0.85, transform: 'scale(0.96)', offset: 0.62 },
        { opacity: 0, transform: 'scale(0.9)' },
      ],
      { duration: lethal ? 340 : 250, easing: EASE.out },
    );
    if (lethal) {
      this.hmRing.animate(
        [
          { opacity: 0.9, transform: 'scale(0.55)' },
          { opacity: 0, transform: 'scale(1.5)' },
        ],
        { duration: 380, easing: EASE.out },
      );
    }
  }

  // --- damage --------------------------------------------------------------

  private updateDamage(dt: number): void {
    const hp = clamp01(this.health / Math.max(1, this.maxHealth));
    // Baseline pain from being low, plus the decaying hit flashes on top.
    const base = hp < 0.55 ? (0.55 - hp) / 0.55 : 0;
    this.hurt = damp(this.hurt, 0, 1.6, dt);
    const shown = clamp01(Math.max(this.hurt, base * 0.72) * (this.dead ? 1.2 : 1));

    const q = Math.round(shown * 50) / 50;
    if (q !== this.lastHurtWritten) {
      this.lastHurtWritten = q;
      this.dmg.style.opacity = q.toFixed(2);
    }

    const bloodAmt = clamp01((0.42 - hp) / 0.42);
    const bq = Math.round(bloodAmt * 40) / 40;
    if (bq !== this.lastBloodWritten) {
      this.lastBloodWritten = bq;
      this.bloodA.style.opacity = (bq * 0.62).toFixed(2);
      this.bloodB.style.opacity = (bq * bq * 0.38).toFixed(2);
    }

    const beat = hp < 0.34 && !this.dead;
    if (beat !== this.beating) {
      this.beating = beat;
      this.beat.style.animationPlayState = beat ? 'running' : 'paused';
      this.beat.style.opacity = beat ? '' : '0';
    }
    if (beat) {
      // Faster and stronger the closer to death.
      const t = clamp01((0.34 - hp) / 0.34);
      this.beat.style.animationDuration = `${(1.0 - t * 0.42).toFixed(2)}s`;
    }
  }

  private spawnDirIndicator(dx: number, dz: number): void {
    const d = this.dirPool[this.dirNext];
    this.dirNext = (this.dirNext + 1) % this.dirPool.length;
    const bearing = Math.atan2(dx, -dz);
    const rel = (bearing - this.heading) * RAD2DEG;
    d.anim?.cancel();
    d.node.style.transform = `rotate(${rel.toFixed(1)}deg)`;
    d.anim = d.node.animate(
      [
        { opacity: 0, transform: `rotate(${rel.toFixed(1)}deg) scale(1.22)` },
        { opacity: 1, transform: `rotate(${rel.toFixed(1)}deg) scale(1)`, offset: 0.12 },
        { opacity: 0.85, transform: `rotate(${rel.toFixed(1)}deg) scale(1)`, offset: 0.55 },
        { opacity: 0, transform: `rotate(${rel.toFixed(1)}deg) scale(0.97)` },
      ],
      { duration: 1500, easing: EASE.outSoft },
    );
  }

  // --- ammo ----------------------------------------------------------------

  private syncWeapon(): void {
    const w = this.weapons?.current ?? null;
    if (!w) return;
    const spec = w.spec;
    setText(this.ammoName, spec.displayName.toUpperCase());
    setText(this.ammoMode, spec.fireMode === 'auto' ? 'FULL AUTO'
      : spec.fireMode === 'burst' ? `${spec.burstCount ?? 3}-RND BURST`
        : spec.fireMode === 'bolt' ? 'BOLT ACTION' : 'SEMI AUTO');

    this.ammoGlyphSlot.textContent = '';
    weaponGlyph(spec.category === 'pistol' ? 'blade' : 'rifle', this.ammoGlyphSlot);

    if (spec.magSize !== this.magSize) {
      this.magSize = spec.magSize;
      this.ammoMag.textContent = '';
      this.magTicks.length = 0;
      // A 100-round belt as 100 ticks is noise; cap the row and let each tick
      // stand for several rounds.
      const shown = Math.min(spec.magSize, 40);
      for (let i = 0; i < shown; i++) this.magTicks.push(el('i', '', this.ammoMag));
      this.lastAmmoWritten = -1;
    }
    this.ammo = this.weapons?.current?.ammo ?? this.ammo;
    this.reserve = this.weapons?.current?.reserve ?? this.reserve;
    this.paintAmmo();
  }

  private paintAmmo(): void {
    if (this.ammo !== this.lastAmmoWritten) {
      setText(this.ammoCur, String(Math.max(0, Math.round(this.ammo))));
      const n = this.magTicks.length;
      if (n > 0 && this.magSize > 0) {
        const lit = Math.round((this.ammo / this.magSize) * n);
        const prevLit = this.lastAmmoWritten < 0 ? -1 : Math.round((this.lastAmmoWritten / this.magSize) * n);
        if (lit !== prevLit) {
          for (let i = 0; i < n; i++) setClass(this.magTicks[i], 'off', i >= lit);
        }
      }
      setClass(this.ammoRoot, 'low', this.magSize > 0 && this.ammo <= Math.max(1, this.magSize * 0.25));
      this.lastAmmoWritten = this.ammo;
    }
    if (this.reserve !== this.lastReserveWritten) {
      setText(this.ammoRes, String(Math.max(0, Math.round(this.reserve))));
      setClass(this.ammoRoot, 'empty', this.reserve <= 0);
      this.lastReserveWritten = this.reserve;
    }
  }

  private updatePrompt(): void {
    const show = !this.reloading && this.magSize > 0 && this.ammo <= 0 && this.reserve > 0;
    setClass(this.prompt, 'on', show);
  }

  // --- kills ---------------------------------------------------------------

  private onKill(ctx: GameContext, victim: number, killer: number, weaponId: string, headshot: boolean): void {
    const mine = killer === ctx.localPlayerId && victim !== ctx.localPlayerId;
    const died = victim === ctx.localPlayerId;

    const row = el('div', `cod-kf${mine ? ' mine' : ''}${died ? ' victim' : ''}`);
    el('span', 'a', row, killer === ctx.localPlayerId ? 'YOU' : callsign(killer));
    const kind = /nade|frag|launch|rpg|explos/i.test(weaponId) ? 'explosive'
      : /knife|melee|blade/i.test(weaponId) ? 'blade' : 'rifle';
    weaponGlyph(kind, row);
    if (headshot) el('span', 'hs', row, 'HS');
    el('span', 'b', row, died ? 'YOU' : callsign(victim));

    this.killfeed.insertBefore(row, this.killfeed.firstChild);
    const entry: KillfeedRow = { node: row, born: this.now, anim: null };
    entry.anim = row.animate(
      [
        { opacity: 0, transform: 'translateX(26px)' },
        { opacity: 1, transform: 'translateX(0)' },
      ],
      { duration: 260, easing: EASE.out },
    );
    this.rows.unshift(entry);
    while (this.rows.length > KILLFEED_MAX) {
      const old = this.rows.pop();
      old?.anim?.cancel();
      old?.node.remove();
    }

    if (mine) {
      this.streakCount++;
      this.score += headshot ? 150 : 100;
      this.showConfirm(headshot);
      this.paintStreak();
      this.contacts.delete(victim);
    }
    if (died) {
      this.streakCount = 0;
      this.paintStreak();
      this.showDeath(killer >= 0 && killer !== victim ? callsign(killer) : null);
    }
  }

  private showConfirm(headshot: boolean): void {
    setClass(this.confirm, 'head', headshot);
    setText(this.confirmKind, headshot ? 'HEADSHOT' : 'ELIMINATED');
    setText(this.confirmPts, headshot ? '+150' : '+100');
    this.confirmAnim?.cancel();
    this.confirmAnim = this.confirm.animate(
      [
        { opacity: 0, transform: 'translateX(-50%) translateY(8px) scale(1.18)' },
        { opacity: 1, transform: 'translateX(-50%) translateY(0) scale(1)', offset: 0.16 },
        { opacity: 1, transform: 'translateX(-50%) translateY(-2px) scale(1)', offset: 0.62 },
        { opacity: 0, transform: 'translateX(-50%) translateY(-12px) scale(0.98)' },
      ],
      { duration: 1250, easing: EASE.out },
    );
  }

  private paintStreak(): void {
    setClass(this.streak, 'on', this.streakCount > 0);
    setText(this.streakNum, String(this.streakCount).padStart(2, '0'));
    if (this.streakCount > 0) {
      this.streakAnim?.cancel();
      this.streakAnim = this.streakNum.animate(
        [{ transform: 'scale(1.5)' }, { transform: 'scale(1)' }],
        { duration: 320, easing: EASE.punch },
      );
    }
  }

  private pruneKillfeed(): void {
    for (let i = this.rows.length - 1; i >= 0; i--) {
      const r = this.rows[i];
      const age = this.now - r.born;
      if (age > KILLFEED_TTL && r.node.dataset.out !== '1') {
        r.node.dataset.out = '1';
        r.anim?.cancel();
        r.anim = r.node.animate(
          [{ opacity: 1, transform: 'translateX(0)' }, { opacity: 0, transform: 'translateX(18px)' }],
          { duration: 420, easing: EASE.in, fill: 'forwards' },
        );
      }
      if (age > KILLFEED_TTL + 0.45) {
        r.anim?.cancel();
        r.node.remove();
        this.rows.splice(i, 1);
      }
    }
  }

  // --- death ---------------------------------------------------------------

  private showDeath(killer: string | null = null): void {
    if (this.dead) return;
    this.dead = true;
    this.deathTimer = 0;
    setText(this.deathSub, killer ? 'KILLED BY' : 'KILLED IN ACTION');
    setText(this.deathBy, killer ?? 'YOU DIED');
    setClass(this.death, 'on', true);
    setClass(this.xh, 'hide', true);
  }

  private clearDeath(): void {
    this.dead = false;
    setClass(this.death, 'on', false);
    this.xhHidden = false;
    setClass(this.xh, 'hide', false);
    this.hurt = 0;
  }

  // --- notifications -------------------------------------------------------

  private pushNotify(text: string, kind: string): void {
    this.notifyQueue.push({ text, kind });
    if (this.notifyQueue.length > 4) this.notifyQueue.shift();
  }

  private updateNotify(dt: number): void {
    this.notifyHold -= dt;
    if (this.notifyHold > 0 || this.notifyQueue.length === 0) return;
    const n = this.notifyQueue.shift();
    if (!n) return;

    const objective = n.kind === 'objective';
    setClass(this.notify, 'info', n.kind === 'info');
    setText(this.notifyKind, objective ? 'OBJECTIVE' : n.kind === 'kill' ? 'CONFIRMED' : 'INTEL');
    setText(this.notifyText, n.text.toUpperCase());
    setClass(this.notify, 'on', true);

    this.notifyHold = 3.2;
    this.notifyAnim?.cancel();
    this.notifyAnim = this.notify.animate(
      [
        { opacity: 0, transform: 'translateX(-50%) translateY(-10px)', filter: 'blur(4px)' },
        { opacity: 1, transform: 'translateX(-50%) translateY(0)', filter: 'blur(0)', offset: 0.13 },
        { opacity: 1, transform: 'translateX(-50%) translateY(0)', filter: 'blur(0)', offset: 0.82 },
        { opacity: 0, transform: 'translateX(-50%) translateY(-6px)', filter: 'blur(3px)' },
      ],
      { duration: 3100, easing: EASE.out },
    );
    window.setTimeout(() => setClass(this.notify, 'on', false), 2900);
  }

  // --- contacts (enemies the player has reason to see) ---------------------

  private noteContact(id: number, x: number, z: number): void {
    if (!this.ctx || id === this.ctx.localPlayerId || id < 0) return;
    const c = this.contacts.get(id);
    if (c) {
      c.ttl = CONTACT_TTL;
      c.x = x;
      c.z = z;
    } else {
      this.contacts.set(id, { ttl: CONTACT_TTL, x, z });
    }
  }

  private updateContacts(dt: number, ctx: GameContext): void {
    for (const [id, c] of this.contacts) {
      c.ttl -= dt;
      const ent = ctx.entities.get(id);
      if (!ent || !ent.alive || c.ttl <= 0) {
        this.contacts.delete(id);
        continue;
      }
      c.x = ent.position.x;
      c.z = ent.position.z;
    }
  }

  // --- compass -------------------------------------------------------------

  private updateCompass(): void {
    const deg = (this.heading * RAD2DEG + 360) % 360;
    if (Math.abs(deg - this.lastCompassWritten) > 0.08) {
      this.lastCompassWritten = deg;
      const x = this.compassHalf - (deg + 360) * COMPASS_PPD;
      this.compassTrack.style.transform = `translate3d(${x.toFixed(1)}px,0,0)`;
    }

    let i = 0;
    for (const c of this.contacts.values()) {
      if (i >= this.compassPips.length) break;
      const bearing = Math.atan2(c.x - this.eyeX, -(c.z - this.eyeZ));
      const bdeg = (bearing * RAD2DEG + 360) % 360;
      const pip = this.compassPips[i++];
      // Placed in the same 720-degree track space as the tick marks.
      const near = bdeg + (Math.abs(bdeg - deg) > 180 ? (bdeg < deg ? 360 : -360) : 0);
      pip.style.transform = `translate3d(${((near + 360) * COMPASS_PPD).toFixed(1)}px,0,0) rotate(45deg)`;
      pip.style.opacity = clamp01(c.ttl / 1.2).toFixed(2);
    }
    for (; i < this.compassPips.length; i++) {
      const pip = this.compassPips[i];
      if (pip.style.opacity !== '0') pip.style.opacity = '0';
    }
  }

  // --- minimap -------------------------------------------------------------

  private updateMinimap(): void {
    if (!this.mmReady) return;
    const px = this.eyeX;
    const pz = this.eyeZ;
    const s = this.mmScale;
    const lx = (px - this.mmCentreX) * s;
    const lz = (pz - this.mmCentreZ) * s;
    const degHeading = this.heading * RAD2DEG;

    const moved = Math.abs(lx - this.lastMmX) > 0.15 || Math.abs(lz - this.lastMmZ) > 0.15;
    const turned = Math.abs(degHeading - this.lastHeadingWritten) > 0.2;
    if (moved || turned) {
      this.lastMmX = lx;
      this.lastMmZ = lz;
      this.mmWorld.style.transform =
        `rotate(${(-degHeading).toFixed(2)}deg) translate3d(${(-lx).toFixed(2)}px,${(-lz).toFixed(2)}px,0)`;
      if (turned) {
        this.lastHeadingWritten = degHeading;
        this.mmCards.style.transform = `rotate(${(-degHeading).toFixed(2)}deg)`;
        for (const c of this.mmCardEls) {
          c.style.transform =
            `translate(${c.dataset.dx}px,${c.dataset.dy}px) translate(-50%,-50%) rotate(${degHeading.toFixed(2)}deg)`;
        }
      }
      if (moved) {
        // Military-style grid reference, so the corner tag is not decoration.
        const gx = Math.floor((px - this.mmCentreX) / 10) + 10;
        const gz = Math.floor((pz - this.mmCentreZ) / 10) + 10;
        setText(
          this.mmTag,
          `GRID ${String.fromCharCode(65 + clamp(gx, 0, 25))}${String(clamp(gz, 0, 99)).padStart(2, '0')}`,
        );
      }
    }

    let i = 0;
    for (const c of this.contacts.values()) {
      if (i >= this.mmPips.length) break;
      const pip = this.mmPips[i++];
      const x = (c.x - this.mmCentreX) * s;
      const z = (c.z - this.mmCentreZ) * s;
      pip.style.transform = `translate3d(${x.toFixed(1)}px,${z.toFixed(1)}px,0) rotate(45deg)`;
      pip.style.opacity = clamp01(c.ttl / 1.2).toFixed(2);
    }
    for (; i < this.mmPips.length; i++) {
      const pip = this.mmPips[i];
      if (pip.style.opacity !== '0') pip.style.opacity = '0';
    }
  }
}
