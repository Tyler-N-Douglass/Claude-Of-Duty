import * as THREE from 'three';
import { Emitter } from './Events';
import {
  type FrameTime,
  type GameContext,
  type InputState,
  type PhysicsWorld,
  type EntityRegistry,
  type QualitySettings,
  type System,
  LAYERS,
} from './Contracts';

/**
 * A system that takes over presentation. The engine calls `render` instead of
 * doing a plain forward render. Only one may be registered; the last wins.
 */
export interface RenderSystem extends System {
  render(ctx: GameContext): void;
}

function isRenderSystem(s: System): s is RenderSystem {
  return typeof (s as RenderSystem).render === 'function';
}

/** Input implementations that need per-frame edge bookkeeping. */
interface FramedInput {
  beginFrame(): void;
  endFrame(): void;
}

function isFramedInput(i: unknown): i is FramedInput {
  const f = i as FramedInput;
  return typeof f?.beginFrame === 'function' && typeof f?.endFrame === 'function';
}

const FIXED_STEP = 1 / 120;
const MAX_STEPS_PER_FRAME = 8;

export interface EngineOptions {
  container: HTMLElement;
  input: InputState;
  physics: PhysicsWorld;
  entities: EntityRegistry;
  quality: QualitySettings;
}

export class Engine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly viewScene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly viewCamera: THREE.PerspectiveCamera;
  readonly events = new Emitter();
  readonly ctx: GameContext;

  private readonly systems: System[] = [];
  private readonly byName = new Map<string, System>();
  private renderSystem: RenderSystem | null = null;

  private readonly time: FrameTime & { dt: number; rawDt: number; elapsed: number; frame: number; fps: number };
  private accumulator = 0;
  private lastTs = 0;
  private running = false;
  private rafId = 0;
  private readonly container: HTMLElement;
  private width = 1;
  private height = 1;
  private readonly framedInput: FramedInput | null;

  constructor(opts: EngineOptions) {
    this.container = opts.container;
    this.framedInput = isFramedInput(opts.input) ? opts.input : null;

    const canvas = document.createElement('canvas');
    canvas.tabIndex = 0;
    opts.container.appendChild(canvas);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false, // handled by TAA/FXAA in the post stack
      alpha: false,
      stencil: false,
      depth: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.info.autoReset = false;

    this.camera = new THREE.PerspectiveCamera(80, 1, 0.05, 2000);
    this.camera.layers.enable(LAYERS.default);
    this.camera.rotation.order = 'YXZ';

    // The viewmodel renders in its own pass with a tight near plane so the
    // weapon never intersects world geometry — same trick the real games use.
    this.viewCamera = new THREE.PerspectiveCamera(60, 1, 0.005, 12);
    this.viewCamera.rotation.order = 'YXZ';
    this.viewScene.name = 'viewmodel-scene';

    this.time = { dt: 0, rawDt: 0, elapsed: 0, frame: 0, fps: 60 };

    const self = this;
    this.ctx = {
      renderer: this.renderer,
      scene: this.scene,
      camera: this.camera,
      viewScene: this.viewScene,
      viewCamera: this.viewCamera,
      input: opts.input,
      events: this.events,
      physics: opts.physics,
      entities: opts.entities,
      quality: opts.quality,
      time: this.time,
      environment: null,
      localPlayerId: 0,
      paused: false,
      get width() {
        return self.width;
      },
      get height() {
        return self.height;
      },
      system<T extends System>(name: string): T | undefined {
        return self.byName.get(name) as T | undefined;
      },
    };

    window.addEventListener('resize', this.onResize);
    document.addEventListener('visibilitychange', this.onVisibility);
  }

  add(system: System): this {
    this.systems.push(system);
    this.byName.set(system.name, system);
    if (isRenderSystem(system)) this.renderSystem = system;
    return this;
  }

  async init(): Promise<void> {
    this.applyQuality();
    this.onResize();
    for (const s of this.systems) {
      if (s.init) await s.init(this.ctx);
    }
    // Late-registered render systems (e.g. created during another init).
    for (const s of this.systems) {
      if (isRenderSystem(s)) this.renderSystem = s;
    }
    this.resizeSystems();
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTs = performance.now();
    this.rafId = requestAnimationFrame(this.tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  private readonly tick = (ts: number): void => {
    this.rafId = requestAnimationFrame(this.tick);
    if (!this.running) return;

    const rawDt = (ts - this.lastTs) / 1000;
    this.lastTs = ts;
    const dt = Math.min(rawDt, 0.1);

    this.time.rawDt = rawDt;
    this.time.dt = dt;
    this.time.elapsed += dt;
    this.time.frame++;
    // Exponential moving average; converges in ~0.5s.
    const inst = rawDt > 0 ? 1 / rawDt : 60;
    this.time.fps += (inst - this.time.fps) * 0.08;

    this.renderer.info.reset();
    this.framedInput?.beginFrame();

    if (!this.ctx.paused) {
      this.accumulator += dt;
      let steps = 0;
      while (this.accumulator >= FIXED_STEP && steps < MAX_STEPS_PER_FRAME) {
        for (const s of this.systems) s.fixedUpdate?.(FIXED_STEP, this.ctx);
        this.accumulator -= FIXED_STEP;
        steps++;
      }
      if (steps === MAX_STEPS_PER_FRAME) this.accumulator = 0; // don't spiral
    }

    for (const s of this.systems) s.update?.(this.time, this.ctx);
    for (const s of this.systems) s.lateUpdate?.(this.time, this.ctx);

    if (this.renderSystem) {
      this.renderSystem.render(this.ctx);
    } else {
      this.renderer.setRenderTarget(null);
      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);
      this.renderer.autoClear = false;
      this.renderer.clearDepth();
      this.renderer.render(this.viewScene, this.viewCamera);
      this.renderer.autoClear = true;
    }

    this.framedInput?.endFrame();
  };

  private applyQuality(): void {
    const q = this.ctx.quality;
    const caps = this.renderer.capabilities;
    q.anisotropy = Math.min(q.anisotropy, caps.getMaxAnisotropy());
    this.renderer.shadowMap.enabled = q.shadowMapSize > 0;
  }

  private readonly onResize = (): void => {
    const rect = this.container.getBoundingClientRect();
    this.width = Math.max(1, Math.floor(rect.width || window.innerWidth));
    this.height = Math.max(1, Math.floor(rect.height || window.innerHeight));

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.width, this.height, false);

    const aspect = this.width / this.height;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    this.viewCamera.aspect = aspect;
    this.viewCamera.updateProjectionMatrix();

    this.resizeSystems();
  };

  private resizeSystems(): void {
    for (const s of this.systems) s.resize?.(this.width, this.height);
  }

  private readonly onVisibility = (): void => {
    if (document.hidden) {
      this.events.emit('game.pause', { paused: true });
    }
  };

  dispose(): void {
    this.stop();
    window.removeEventListener('resize', this.onResize);
    document.removeEventListener('visibilitychange', this.onVisibility);
    for (const s of this.systems) s.dispose?.();
    this.events.clear();
    this.renderer.dispose();
  }
}
