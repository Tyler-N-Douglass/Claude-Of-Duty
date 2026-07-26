/**
 * Front end: main menu, pause menu, settings, controls, round end, and the F3
 * developer readout.
 *
 * The menu owns two pieces of global state nobody else touches — `ctx.paused`
 * and pointer lock — and it is the only writer of persisted user settings. All
 * quality changes go through `ctx.quality` plus a `quality.changed` broadcast,
 * so systems that cache derived state can rebuild.
 *
 * Pointer lock is the awkward part of shipping an FPS in a browser: the lock
 * can only be requested from a user gesture, and a lock released with Escape is
 * refused for about a second afterwards. Every transition into gameplay
 * therefore goes through `enterGame()`, which stays on a click-to-continue
 * screen until the lock is actually held.
 */
import type {
  FrameTime, GameContext, QualityPreset, QualitySettings, System,
} from '../core/Contracts';
import {
  EASE, clamp, el, setText, setClass, grainDataURI, injectUIStyles, releaseUIStyles,
} from './Style';

interface RenderLike extends System {
  setExposure(v: number): void;
  readonly renderScale: number;
}

interface PlayerLike extends System {
  respawn(ctx: GameContext): void;
}

type Screen = 'main' | 'pause' | 'settings' | 'controls' | 'end' | 'none';

const STORAGE_KEY = 'cod.settings.v2';
const BASE_SENSITIVITY = 0.0022;
const BASE_FOV = 80;

/**
 * Local copy of the preset table. The UI layer may not import from another
 * module's internals, and duplicating twelve numbers is cheaper than a
 * dependency that would couple the menu to the engine's boot order.
 */
const PRESETS: Record<QualityPreset, Omit<QualitySettings, 'preset'>> = {
  low: {
    renderScale: 0.7, shadowMapSize: 1024, shadowCascades: 2, ssao: false, ssr: false,
    bloom: true, motionBlur: false, taa: false, volumetrics: false, anisotropy: 4,
    particleBudget: 1500, decalBudget: 64, targetFps: 60,
  },
  medium: {
    renderScale: 0.85, shadowMapSize: 2048, shadowCascades: 3, ssao: true, ssr: false,
    bloom: true, motionBlur: true, taa: true, volumetrics: false, anisotropy: 8,
    particleBudget: 4000, decalBudget: 128, targetFps: 60,
  },
  high: {
    renderScale: 1.0, shadowMapSize: 2048, shadowCascades: 4, ssao: true, ssr: true,
    bloom: true, motionBlur: true, taa: true, volumetrics: true, anisotropy: 16,
    particleBudget: 8000, decalBudget: 256, targetFps: 60,
  },
  ultra: {
    renderScale: 1.0, shadowMapSize: 4096, shadowCascades: 4, ssao: true, ssr: true,
    bloom: true, motionBlur: true, taa: true, volumetrics: true, anisotropy: 16,
    particleBudget: 16000, decalBudget: 512, targetFps: 60,
  },
};

interface UserSettings {
  preset: QualityPreset;
  sensitivity: number;
  adsSens: number;
  fov: number;
  exposure: number;
  renderScale: number;
  ssao: boolean;
  ssr: boolean;
  bloom: boolean;
  motionBlur: boolean;
  taa: boolean;
  volumetrics: boolean;
}

function defaults(q: QualitySettings): UserSettings {
  return {
    preset: q.preset,
    sensitivity: 1,
    adsSens: 0.75,
    fov: BASE_FOV,
    exposure: 1,
    renderScale: q.renderScale,
    ssao: q.ssao,
    ssr: q.ssr,
    bloom: q.bloom,
    motionBlur: q.motionBlur,
    taa: q.taa,
    volumetrics: q.volumetrics,
  };
}

const BINDINGS: [string, string[]][] = [
  ['Move', ['W', 'A', 'S', 'D']],
  ['Sprint', ['SHIFT']],
  ['Crouch / Slide', ['CTRL', 'C']],
  ['Jump / Mantle', ['SPACE']],
  ['Tactical Walk', ['ALT']],
  ['Fire', ['MOUSE 1']],
  ['Aim Down Sights', ['MOUSE 2']],
  ['Reload', ['R']],
  ['Melee', ['V']],
  ['Lethal', ['G']],
  ['Swap Weapon', ['Q', '1', '2']],
  ['Interact', ['F', 'E']],
  ['Lean', ['Z', 'X']],
  ['Flashlight', ['T']],
  ['Scoreboard', ['TAB']],
  ['Pause', ['ESC']],
  ['Debug Stats', ['F3']],
];

interface MenuItem {
  node: HTMLElement;
  action: () => void;
}

export class MenuSystem implements System {
  readonly name = 'menu';

  private ctx: GameContext | null = null;
  private root: HTMLElement | null = null;
  private stats: HTMLElement | null = null;
  private lockGate: HTMLElement | null = null;
  private readonly unsubs: (() => void)[] = [];

  private wrap!: HTMLElement;
  private brand!: HTMLElement;
  private body!: HTMLElement;
  private footRight!: HTMLElement;

  private screen: Screen = 'none';
  private returnTo: Screen = 'main';
  private items: MenuItem[] = [];
  private selected = 0;

  private settings!: UserSettings;
  private statsTimer = 0;
  private statsOn = false;
  private settled = false;
  private lastPauseToggle = -1;
  private wantLock = false;
  private lockDeadline = 0;
  private elapsed = 0;

  // Round bookkeeping for the end screen.
  private kills = 0;
  private deaths = 0;
  private headshots = 0;
  private bestStreak = 0;
  private streak = 0;

  // FOV is applied as a multiplier on whatever the camera rig computed, since
  // the rig owns the camera and only writes fov when its own value changes.
  private fovMul = 1;
  private rigFov = BASE_FOV;
  private myFov = -1;
  private rigViewFov = 60;
  private myViewFov = -1;

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;
    injectUIStyles();

    this.settings = this.load(ctx.quality);

    const root = el('div', 'cod-ui cod-menu');
    document.body.appendChild(root);
    this.root = root;

    el('div', 'bg', root);
    const grain = el('div', 'grain', root);
    grain.style.backgroundImage = `url(${grainDataURI()})`;
    el('div', 'scan', root);
    el('div', 'sweep', root);
    el('div', 'vig', root);

    const wrap = el('div', 'wrap', root);
    this.wrap = wrap;
    this.brand = el('div', 'brand', wrap);
    this.body = el('div', 'body', wrap);
    const foot = el('div', 'foot', wrap);
    const keys = el('div', 'keys', foot);
    el('span', '', keys, 'WASD Move');
    el('span', '', keys, 'Mouse Look');
    el('span', '', keys, 'LMB Fire');
    el('span', '', keys, 'F3 Stats');
    this.footRight = el('div', '', foot, 'BUILD 1.0.0 // WEB');

    this.buildBrand();

    this.stats = el('div', 'cod-ui cod-stats');
    document.body.appendChild(this.stats);

    const gate = el('div', 'cod-ui cod-lock');
    el('span', '', gate, 'Click to continue');
    gate.addEventListener('click', this.onGateClick);
    document.body.appendChild(gate);
    this.lockGate = gate;

    document.addEventListener('keydown', this.onKeyDown);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);

    this.applyAll(ctx, true);

    this.unsubs.push(ctx.events.on('game.pause', (p) => {
      if (p.paused && this.screen === 'none') this.open('pause');
    }));
    this.unsubs.push(ctx.events.on('entity.killed', (p) => {
      if (p.entityId === ctx.localPlayerId) {
        this.deaths++;
        this.streak = 0;
      } else if (p.killerId === ctx.localPlayerId) {
        this.kills++;
        if (p.headshot) this.headshots++;
        this.streak++;
        if (this.streak > this.bestStreak) this.bestStreak = this.streak;
      }
    }));
    this.unsubs.push(ctx.events.on('game.over', (p) => {
      if (p.won) this.open('end');
    }));

    // The automated capture harness never clicks Play, so a blocking menu would
    // make every screenshot a screenshot of the menu. Detect automation (or an
    // explicit override) and drop straight into the round.
    if (this.shouldAutostart()) {
      ctx.paused = false;
      this.screen = 'none';
    } else {
      this.open('main');
    }
  }

  dispose(): void {
    for (const off of this.unsubs) off();
    this.unsubs.length = 0;
    if (this.statsTimer !== 0) window.clearInterval(this.statsTimer);
    this.statsTimer = 0;
    document.removeEventListener('keydown', this.onKeyDown);
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    this.lockGate?.removeEventListener('click', this.onGateClick);
    this.root?.remove();
    this.stats?.remove();
    this.lockGate?.remove();
    this.root = null;
    this.stats = null;
    this.lockGate = null;
    this.items.length = 0;
    this.ctx = null;
    releaseUIStyles();
  }

  private shouldAutostart(): boolean {
    try {
      const q = `${window.location.search}${window.location.hash}`;
      if (/[?&#]menu\b/.test(q)) return false;
      if (/[?&#](nomenu|autostart|qa)\b/.test(q)) return true;
      // Two independent signals, because a capture run that lands on the menu
      // produces a folder full of screenshots of the menu.
      return navigator.webdriver === true || /headless/i.test(navigator.userAgent);
    } catch {
      return false;
    }
  }

  // =========================================================================
  // Frame
  // =========================================================================

  update(time: FrameTime, ctx: GameContext): void {
    this.elapsed = time.elapsed;

    if (!this.settled) {
      // Systems that init after this one (the render pipeline is registered
      // last) would clobber values written during init; re-apply once the whole
      // boot sequence has finished.
      this.settled = true;
      this.applyExposure();
      this.applyFov();
    }

    if (ctx.input.wasPressed('pause') && time.elapsed - this.lastPauseToggle > 0.25) {
      this.lastPauseToggle = time.elapsed;
      if (this.screen === 'none') this.open('pause');
      else if (this.screen === 'pause') this.enterGame();
      else if (this.screen === 'settings' || this.screen === 'controls') this.open(this.returnTo);
    }

    if (this.wantLock && this.screen === 'none') {
      if (ctx.input.pointerLocked) {
        this.wantLock = false;
        this.showGate(false);
      } else if (time.elapsed > this.lockDeadline) {
        this.showGate(true);
      }
    }

  }

  /**
   * Applies the user's FOV on top of the rig's own value. Runs after the player
   * system's lateUpdate, and tracks who wrote the field last so repeated
   * multiplication can never compound.
   */
  lateUpdate(_time: FrameTime, ctx: GameContext): void {
    const cam = ctx.camera;
    if (cam.fov !== this.myFov) this.rigFov = cam.fov;
    const want = this.rigFov * this.fovMul;
    if (cam.fov !== want) {
      cam.fov = want;
      cam.updateProjectionMatrix();
    }
    this.myFov = want;

    // The viewmodel camera only takes a third of the change, matching the
    // separate-viewmodel-FOV convention the rig already uses.
    const view = ctx.viewCamera;
    if (view.fov !== this.myViewFov) this.rigViewFov = view.fov;
    const wantView = this.rigViewFov * (1 + (this.fovMul - 1) * 0.34);
    if (view.fov !== wantView) {
      view.fov = wantView;
      view.updateProjectionMatrix();
    }
    this.myViewFov = wantView;
  }

  // =========================================================================
  // Screens
  // =========================================================================

  private open(screen: Screen): void {
    const ctx = this.ctx;
    if (!ctx || !this.root) return;
    this.screen = screen;
    if (screen === 'none') {
      setClass(this.root, 'on', false);
      return;
    }

    const wasPlaying = !ctx.paused;
    ctx.paused = true;
    ctx.input.exitPointerLock();
    if (wasPlaying) ctx.events.emit('game.pause', { paused: true });
    setClass(this.root, 'on', true);
    this.showGate(false);
    this.items = [];
    this.selected = 0;
    this.body.textContent = '';
    this.body.className = 'body';

    switch (screen) {
      case 'main': this.buildMain(); break;
      case 'pause': this.buildPause(); break;
      case 'settings': this.buildSettings(); break;
      case 'controls': this.buildControls(); break;
      case 'end': this.buildEnd(); break;
      default: break;
    }
    this.paintSelection();
    this.wrap.animate(
      [{ opacity: 0, transform: 'translateY(10px)' }, { opacity: 1, transform: 'translateY(0)' }],
      { duration: 260, easing: EASE.out },
    );
  }

  private buildBrand(): void {
    const b = this.brand;
    const eyebrow = el('div', 'eyebrow', b);
    el('s', '', eyebrow);
    el('span', '', eyebrow, 'Task Force // Dust Corridor');
    const h1 = el('h1', '', b);
    h1.append('Claude');
    el('em', '', h1, 'of Duty');
  }

  private addItem(label: string, hint: string, action: () => void, danger = false): void {
    const list = this.body.querySelector('.list') ?? el('div', 'list', this.body);
    const node = el('button', `item${danger ? ' danger' : ''}`, list);
    node.type = 'button';
    el('span', 'idx', node, String(this.items.length + 1).padStart(2, '0'));
    el('span', 'lb', node, label);
    el('span', 'hint', node, hint);
    const item: MenuItem = { node, action };
    node.addEventListener('click', () => {
      this.selected = this.items.indexOf(item);
      action();
    });
    node.addEventListener('mouseenter', () => {
      this.selected = this.items.indexOf(item);
      this.paintSelection();
    });
    this.items.push(item);
  }

  private buildMain(): void {
    this.brand.style.display = '';
    this.body.classList.add('split');
    el('div', 'list', this.body);
    this.addItem('Play', 'Deploy', () => this.startRound());
    this.addItem('Settings', 'Video & Input', () => {
      this.returnTo = 'main';
      this.open('settings');
    });
    this.addItem('Controls', 'Key Bindings', () => {
      this.returnTo = 'main';
      this.open('controls');
    });

    const brief = el('div', 'brief', this.body);
    el('div', 'bt', brief, 'Deployment');
    const row = (k: string, v: string): void => {
      const r = el('div', 'br', brief);
      el('span', '', r, k);
      el('b', '', r, v);
    };
    row('Theatre', 'Dust Corridor');
    row('Contract', 'Search & Destroy');
    row('Threat', 'Veteran');
    row('Primary', 'MK4 Ranger');
    row('Quality', this.settings.preset);
    el('div', 'bn', brief,
      'Contact expected at close to medium range. Elevated positions on the '
      + 'east block cover the whole street — clear them before pushing.');
  }

  private buildPause(): void {
    this.brand.style.display = 'none';
    this.body.classList.add('mid');
    const col = el('div', 'col', this.body);
    el('div', 'plabel', col, 'Mission Paused');
    el('div', 'list', col);
    this.addItem('Resume', 'ESC', () => this.enterGame());
    this.addItem('Settings', 'Video & Input', () => {
      this.returnTo = 'pause';
      this.open('settings');
    });
    this.addItem('Controls', 'Key Bindings', () => {
      this.returnTo = 'pause';
      this.open('controls');
    });
    this.addItem('Restart Round', 'Redeploy', () => this.restart(), true);
    this.addItem('Abort to Menu', '', () => {
      this.open('main');
    }, true);
  }

  private buildEnd(): void {
    this.brand.style.display = 'none';
    const c = el('div', 'center', this.body);
    el('div', 'lead', c, 'Objective Complete');
    el('div', 'big', c, 'Victory');
    const stat = el('div', 'stat', c);
    const cell = (label: string, value: string): void => {
      const d = el('div', '', stat);
      el('b', '', d, value);
      el('span', '', d, label);
    };
    cell('Eliminations', String(this.kills));
    cell('Headshots', String(this.headshots));
    cell('Best Streak', String(this.bestStreak));
    cell('Deaths', String(this.deaths));
    const list = el('div', 'list', c);
    list.style.marginTop = '18px';
    this.addItem('Play Again', 'Redeploy', () => this.restart());
    this.addItem('Main Menu', '', () => this.open('main'));
  }

  private panelHead(parent: HTMLElement, eyebrow: string, title: string): void {
    const head = el('div', 'phead', parent);
    const eb = el('div', 'peyebrow', head);
    el('s', '', eb);
    el('span', '', eb, eyebrow);
    el('div', 'ptitle', head, title);
  }

  private buildControls(): void {
    this.brand.style.display = 'none';
    this.body.classList.add('mid');
    const p = el('div', 'panel', this.body);
    this.panelHead(p, 'Reference', 'Controls');
    el('h2', '', p, 'Key Bindings');
    const grid = el('div', 'binds', p);
    for (const [name, keys] of BINDINGS) {
      const row = el('div', 'bind', grid);
      el('span', 'k', row, name);
      const v = el('span', 'v', row);
      for (const k of keys) el('kbd', '', v, k);
    }
    const back = el('button', 'back', p, 'Back');
    back.type = 'button';
    back.addEventListener('click', () => this.open(this.returnTo));
  }

  // =========================================================================
  // Settings
  // =========================================================================

  private buildSettings(): void {
    this.brand.style.display = 'none';
    this.body.classList.add('mid');
    const p = el('div', 'panel', this.body);
    this.panelHead(p, 'Options', 'Settings');

    el('h2', '', p, 'Video');
    this.addSegment(p, 'Quality Preset', ['low', 'medium', 'high', 'ultra'], this.settings.preset, (v) => {
      this.settings.preset = v as QualityPreset;
      const preset = PRESETS[v as QualityPreset];
      this.settings.renderScale = preset.renderScale;
      this.settings.ssao = preset.ssao;
      this.settings.ssr = preset.ssr;
      this.settings.bloom = preset.bloom;
      this.settings.motionBlur = preset.motionBlur;
      this.settings.taa = preset.taa;
      this.settings.volumetrics = preset.volumetrics;
      this.applyAll(this.ctx, true);
      this.open('settings');
    });

    this.addSlider(p, 'Brightness', 0.55, 1.75, this.settings.exposure, 0.01,
      (v) => `${Math.round(v * 100)}%`, (v) => {
        this.settings.exposure = v;
        this.applyExposure();
        this.save();
      });

    this.addSlider(p, 'Render Scale', 0.55, 1, this.settings.renderScale, 0.05,
      (v) => `${Math.round(v * 100)}%`, (v) => {
        this.settings.renderScale = v;
        this.applyQuality(true);
      });

    this.addToggle(p, 'Ambient Occlusion', this.settings.ssao, (v) => {
      this.settings.ssao = v;
      this.applyQuality(true);
    });
    this.addToggle(p, 'Screen Space Reflections', this.settings.ssr, (v) => {
      this.settings.ssr = v;
      this.applyQuality(true);
    });
    this.addToggle(p, 'Bloom', this.settings.bloom, (v) => {
      this.settings.bloom = v;
      this.applyQuality(true);
    });
    this.addToggle(p, 'Motion Blur', this.settings.motionBlur, (v) => {
      this.settings.motionBlur = v;
      this.applyQuality(true);
    });
    this.addToggle(p, 'Temporal AA', this.settings.taa, (v) => {
      this.settings.taa = v;
      this.applyQuality(true);
    });
    this.addToggle(p, 'Volumetric Light', this.settings.volumetrics, (v) => {
      this.settings.volumetrics = v;
      this.applyQuality(true);
    });

    el('h2', '', p, 'Input');
    this.addSlider(p, 'Sensitivity', 0.2, 3, this.settings.sensitivity, 0.01,
      (v) => v.toFixed(2), (v) => {
        this.settings.sensitivity = v;
        this.applyInput();
        this.save();
      });
    this.addSlider(p, 'ADS Sensitivity', 0.3, 1.2, this.settings.adsSens, 0.01,
      (v) => v.toFixed(2), (v) => {
        this.settings.adsSens = v;
        this.applyInput();
        this.save();
      });
    this.addSlider(p, 'Field of View', 65, 115, this.settings.fov, 1,
      (v) => `${Math.round(v)}`, (v) => {
        this.settings.fov = v;
        this.applyFov();
        this.save();
      });

    const back = el('button', 'back', p, 'Back');
    back.type = 'button';
    back.addEventListener('click', () => this.open(this.returnTo));
  }

  private addRow(parent: HTMLElement, label: string): { row: HTMLElement; value: HTMLElement } {
    const row = el('div', 'row', parent);
    el('div', 'k', row, label);
    const mid = el('div', 'm', row);
    const value = el('div', 'v', row, '');
    return { row: mid, value };
  }

  private addSegment(
    parent: HTMLElement, label: string, options: string[], current: string, onPick: (v: string) => void,
  ): void {
    const { row, value } = this.addRow(parent, label);
    const seg = el('div', 'seg', row);
    setText(value, '');
    for (const o of options) {
      const b = el('button', o === current ? 'on' : '', seg, o.toUpperCase());
      b.type = 'button';
      b.addEventListener('click', () => onPick(o));
    }
  }

  private addToggle(parent: HTMLElement, label: string, current: boolean, onSet: (v: boolean) => void): void {
    const { row, value } = this.addRow(parent, label);
    const sw = el('div', `sw${current ? ' on' : ''}`, row);
    el('i', '', sw);
    let state = current;
    setText(value, state ? 'ON' : 'OFF');
    sw.addEventListener('click', () => {
      state = !state;
      setClass(sw, 'on', state);
      setText(value, state ? 'ON' : 'OFF');
      onSet(state);
    });
  }

  private addSlider(
    parent: HTMLElement, label: string, min: number, max: number, current: number, step: number,
    fmt: (v: number) => string, onSet: (v: number) => void,
  ): void {
    const { row, value } = this.addRow(parent, label);
    const sl = el('div', 'sl', row);
    el('div', 'trk', sl);
    const fil = el('div', 'fil', sl);
    const nub = el('div', 'nub', sl);
    for (let i = 0; i <= 4; i++) {
      const t = el('div', 'tick', sl);
      t.style.left = `${i * 25}%`;
    }

    let v = clamp(current, min, max);
    const paint = (): void => {
      const t = (v - min) / (max - min);
      fil.style.transform = `scaleX(${t.toFixed(4)})`;
      nub.style.left = `${(t * 100).toFixed(2)}%`;
      setText(value, fmt(v));
    };
    paint();

    let dragging = false;
    const setFromEvent = (e: PointerEvent): void => {
      const r = sl.getBoundingClientRect();
      const t = clamp((e.clientX - r.left) / Math.max(1, r.width), 0, 1);
      const raw = min + t * (max - min);
      const snapped = Math.round(raw / step) * step;
      const next = clamp(Number(snapped.toFixed(4)), min, max);
      if (next !== v) {
        v = next;
        paint();
        onSet(v);
      }
    };
    sl.addEventListener('pointerdown', (e) => {
      dragging = true;
      sl.setPointerCapture(e.pointerId);
      setFromEvent(e);
    });
    sl.addEventListener('pointermove', (e) => {
      if (dragging) setFromEvent(e);
    });
    const stop = (e: PointerEvent): void => {
      dragging = false;
      if (sl.hasPointerCapture(e.pointerId)) sl.releasePointerCapture(e.pointerId);
    };
    sl.addEventListener('pointerup', stop);
    sl.addEventListener('pointercancel', stop);
  }

  // =========================================================================
  // Applying settings
  // =========================================================================

  private applyAll(ctx: GameContext | null, broadcast: boolean): void {
    if (!ctx) return;
    this.applyQuality(broadcast);
    this.applyInput();
    this.applyFov();
    this.applyExposure();
    this.save();
  }

  private applyQuality(broadcast: boolean): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const s = this.settings;
    const q = ctx.quality;
    const preset = PRESETS[s.preset];
    q.preset = s.preset;
    q.shadowMapSize = preset.shadowMapSize;
    q.shadowCascades = preset.shadowCascades;
    q.anisotropy = Math.min(preset.anisotropy, ctx.renderer.capabilities.getMaxAnisotropy());
    q.particleBudget = preset.particleBudget;
    q.decalBudget = preset.decalBudget;
    q.targetFps = preset.targetFps;
    q.renderScale = s.renderScale;
    q.ssao = s.ssao;
    q.ssr = s.ssr;
    q.bloom = s.bloom;
    q.motionBlur = s.motionBlur;
    q.taa = s.taa;
    q.volumetrics = s.volumetrics;
    ctx.renderer.shadowMap.enabled = q.shadowMapSize > 0;
    this.save();
    if (broadcast) ctx.events.emit('quality.changed', { preset: q.preset });
  }

  private applyInput(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    ctx.input.setSensitivity(BASE_SENSITIVITY * this.settings.sensitivity);
    ctx.input.setAdsSensitivityScale(this.settings.adsSens);
  }

  private applyFov(): void {
    // lateUpdate keeps running even at 1.0 — the rig caches the last fov it
    // wrote, so handing the field back mid-frame could strand a stale value it
    // would never correct.
    this.fovMul = clamp(this.settings.fov / BASE_FOV, 0.6, 1.6);
  }

  private applyExposure(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const render = ctx.system<RenderLike>('render');
    if (render && typeof render.setExposure === 'function') render.setExposure(this.settings.exposure);
    else ctx.renderer.toneMappingExposure = this.settings.exposure;
  }

  private load(q: QualitySettings): UserSettings {
    const base = defaults(q);
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return base;
      const parsed = JSON.parse(raw) as Partial<UserSettings>;
      const out = { ...base };
      if (parsed.preset && parsed.preset in PRESETS) out.preset = parsed.preset;
      const num = (v: unknown, lo: number, hi: number, fallback: number): number =>
        typeof v === 'number' && isFinite(v) ? clamp(v, lo, hi) : fallback;
      out.sensitivity = num(parsed.sensitivity, 0.2, 3, base.sensitivity);
      out.adsSens = num(parsed.adsSens, 0.3, 1.2, base.adsSens);
      out.fov = num(parsed.fov, 65, 115, base.fov);
      out.exposure = num(parsed.exposure, 0.55, 1.75, base.exposure);
      out.renderScale = num(parsed.renderScale, 0.55, 1, base.renderScale);
      for (const k of ['ssao', 'ssr', 'bloom', 'motionBlur', 'taa', 'volumetrics'] as const) {
        if (typeof parsed[k] === 'boolean') out[k] = parsed[k] as boolean;
      }
      return out;
    } catch {
      return base;
    }
  }

  private save(): void {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(this.settings));
    } catch {
      /* private browsing or a full quota: settings simply do not persist */
    }
  }

  // =========================================================================
  // Gameplay transitions
  // =========================================================================

  private startRound(): void {
    this.kills = 0;
    this.deaths = 0;
    this.headshots = 0;
    this.streak = 0;
    this.bestStreak = 0;
    this.enterGame();
  }

  private restart(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const player = ctx.system<PlayerLike>('player');
    if (player && typeof player.respawn === 'function') player.respawn(ctx);
    this.kills = 0;
    this.deaths = 0;
    this.headshots = 0;
    this.streak = 0;
    this.bestStreak = 0;
    this.enterGame();
    ctx.events.emit('ui.notify', { text: 'Redeployed', kind: 'objective' });
  }

  private enterGame(): void {
    const ctx = this.ctx;
    if (!ctx || !this.root) return;
    this.screen = 'none';
    setClass(this.root, 'on', false);
    ctx.paused = false;
    ctx.events.emit('game.pause', { paused: false });
    this.wantLock = true;
    // Chromium refuses a re-lock for about a second after an Escape release, so
    // give it that long before falling back to the click-to-continue gate.
    this.lockDeadline = this.elapsed + 1.1;
    ctx.input.requestPointerLock();
  }

  private showGate(on: boolean): void {
    if (this.lockGate) setClass(this.lockGate, 'on', on);
  }

  private readonly onGateClick = (): void => {
    this.showGate(false);
    this.lockDeadline = this.elapsed + 1.1;
    this.ctx?.input.requestPointerLock();
  };

  private readonly onPointerLockChange = (): void => {
    const ctx = this.ctx;
    if (!ctx) return;
    const locked = document.pointerLockElement !== null;
    if (locked) {
      this.wantLock = false;
      this.showGate(false);
      return;
    }
    // Lock lost while playing: the player pressed Escape or alt-tabbed.
    if (this.screen === 'none' && !this.wantLock) this.open('pause');
  };

  // =========================================================================
  // Keyboard
  // =========================================================================

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (e.code === 'F3') {
      e.preventDefault();
      this.toggleStats();
      return;
    }
    if (this.screen === 'none' || this.items.length === 0) return;

    if (e.code === 'ArrowDown' || e.code === 'KeyS' || e.code === 'Tab') {
      e.preventDefault();
      this.selected = (this.selected + 1) % this.items.length;
      this.paintSelection();
    } else if (e.code === 'ArrowUp' || e.code === 'KeyW') {
      e.preventDefault();
      this.selected = (this.selected - 1 + this.items.length) % this.items.length;
      this.paintSelection();
    } else if (e.code === 'Enter' || e.code === 'NumpadEnter' || e.code === 'Space') {
      e.preventDefault();
      this.items[this.selected]?.action();
    }
  };

  private paintSelection(): void {
    for (let i = 0; i < this.items.length; i++) {
      setClass(this.items[i].node, 'sel', i === this.selected);
    }
  }

  // =========================================================================
  // Dev stats
  // =========================================================================

  private toggleStats(): void {
    this.statsOn = !this.statsOn;
    if (this.stats) setClass(this.stats, 'on', this.statsOn);
    if (this.statsTimer !== 0) {
      window.clearInterval(this.statsTimer);
      this.statsTimer = 0;
    }
    if (!this.statsOn || !this.ctx) return;
    // Sampled from a timer rather than from update(): the engine resets
    // renderer.info at the top of every frame and fills it during the render
    // pass, so anything read inside update() would always be zero. A macrotask
    // lands after the frame is presented, where the counters are real.
    this.paintStats(this.ctx);
    this.statsTimer = window.setInterval(() => {
      if (this.ctx) this.paintStats(this.ctx);
    }, 250);
  }

  private paintStats(ctx: GameContext): void {
    const node = this.stats;
    if (!node) return;
    const info = ctx.renderer.info;
    const render = ctx.system<RenderLike>('render');
    const scale = render && typeof render.renderScale === 'number' ? render.renderScale : ctx.quality.renderScale;
    const fps = ctx.time.fps;
    const ms = fps > 0 ? 1000 / fps : 0;
    const w = Math.round(ctx.width * scale);
    const h = Math.round(ctx.height * scale);
    // renderer.info is reset at the top of every frame and filled during the
    // render pass, so these are last frame's numbers — which is what we want.
    setText(node, [
      `${fps.toFixed(0).padStart(3)} fps   ${ms.toFixed(2)} ms`,
      `draws  ${String(info.render.calls).padStart(6)}`,
      `tris   ${formatCount(info.render.triangles)}`,
      `geo/tex ${info.memory.geometries} / ${info.memory.textures}`,
      `res    ${w}x${h}  (${(scale * 100).toFixed(0)}%)`,
      `preset ${ctx.quality.preset.toUpperCase()}  fov ${ctx.camera.fov.toFixed(0)}`,
    ].join('\n'));
  }
}

function formatCount(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`.padStart(6);
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`.padStart(6);
  return String(n).padStart(6);
}
