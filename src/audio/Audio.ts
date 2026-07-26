/**
 * AudioSystem — every sound in the game, synthesised from scratch.
 *
 * There are no audio files. Each shot, impact, footstep and explosion is built
 * per-trigger out of noise bursts, filtered envelopes and pitch sweeps, then
 * placed in the stereo field and fed to a generated convolution reverb whose
 * impulse response models a narrow street between masonry blocks.
 *
 * Design notes that matter:
 *   - A gunshot is not one sound. It is a click (the case head), a body (the
 *     expanding gas front), a low thump (the recoil impulse in your chest) and
 *     a tail (the street reflecting it back). Layering those four with the right
 *     relative delays is the whole difference between "gunshot" and "beep".
 *   - Distance does three things: attenuates, delays, and low-passes. Skipping
 *     the low-pass is why browser games sound like everything is in your ear.
 *   - Browsers refuse to start an AudioContext without a gesture. The context is
 *     created up front but every trigger is a no-op until it is running, so a
 *     headless capture never throws and never queues a thousand pending voices.
 */
import * as THREE from 'three';
import type {
  FrameTime,
  GameContext,
  SurfaceKind,
  System,
} from '../core/Contracts';

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const SPEED_OF_SOUND = 343;
/** Above this many simultaneous voices new non-critical sounds are dropped. */
const MAX_VOICES = 28;
const MASTER_GAIN = 0.55;
/** Reference distance for the inverse-distance law. */
const REF_DISTANCE = 3;
const MAX_AUDIBLE = 140;

interface WeaponVoice {
  /** Bandpass centre of the muzzle body, Hz. */
  body: number;
  /** Low thump start frequency. */
  thump: number;
  /** Overall level. */
  level: number;
  /** Body decay, seconds. */
  decay: number;
  /** Tail send amount. */
  tail: number;
}

/**
 * Per-weapon voicing. Calibre drives the thump, barrel length drives how much
 * of the report is body versus crack.
 */
const WEAPON_VOICES: Record<string, WeaponVoice> = {
  mk4_ranger: { body: 1150, thump: 168, level: 1.0, decay: 0.11, tail: 0.5 },
  vks9_wasp: { body: 1650, thump: 205, level: 0.82, decay: 0.075, tail: 0.36 },
  m16br: { body: 1280, thump: 176, level: 0.95, decay: 0.1, tail: 0.48 },
  kv800_ballista: { body: 780, thump: 108, level: 1.55, decay: 0.2, tail: 0.95 },
  m870_breacher: { body: 620, thump: 96, level: 1.35, decay: 0.17, tail: 0.8 },
  p226_sidearm: { body: 1450, thump: 220, level: 0.7, decay: 0.07, tail: 0.32 },
};

const DEFAULT_VOICE: WeaponVoice = { body: 1200, thump: 175, level: 0.95, decay: 0.1, tail: 0.5 };

interface SurfaceVoice {
  /** Impact click centre frequency. */
  freq: number;
  q: number;
  decay: number;
  level: number;
  /** How much of the hit is a tonal ring versus broadband noise. */
  ring: number;
}

const SURFACE_VOICES: Record<SurfaceKind, SurfaceVoice> = {
  concrete: { freq: 1900, q: 1.4, decay: 0.075, level: 0.85, ring: 0 },
  plaster: { freq: 1500, q: 1.1, decay: 0.06, level: 0.6, ring: 0 },
  metal: { freq: 3200, q: 7.5, decay: 0.34, level: 0.95, ring: 0.75 },
  wood: { freq: 1050, q: 2.2, decay: 0.1, level: 0.7, ring: 0.22 },
  dirt: { freq: 520, q: 0.9, decay: 0.075, level: 0.55, ring: 0 },
  sand: { freq: 780, q: 0.7, decay: 0.055, level: 0.45, ring: 0 },
  glass: { freq: 5200, q: 5.5, decay: 0.24, level: 0.8, ring: 0.6 },
  water: { freq: 900, q: 0.8, decay: 0.13, level: 0.6, ring: 0 },
  flesh: { freq: 320, q: 1.6, decay: 0.055, level: 0.9, ring: 0 },
  foliage: { freq: 4200, q: 0.6, decay: 0.09, level: 0.4, ring: 0 },
  rubber: { freq: 420, q: 1.8, decay: 0.06, level: 0.5, ring: 0.1 },
  fabric: { freq: 600, q: 0.7, decay: 0.05, level: 0.35, ring: 0 },
};

interface FootVoice {
  freq: number;
  decay: number;
  level: number;
  grit: number;
}

const FOOT_VOICES: Record<SurfaceKind, FootVoice> = {
  concrete: { freq: 900, decay: 0.055, level: 0.5, grit: 0.55 },
  plaster: { freq: 850, decay: 0.05, level: 0.4, grit: 0.4 },
  metal: { freq: 1700, decay: 0.13, level: 0.6, grit: 0.3 },
  wood: { freq: 620, decay: 0.075, level: 0.5, grit: 0.25 },
  dirt: { freq: 380, decay: 0.06, level: 0.45, grit: 0.85 },
  sand: { freq: 480, decay: 0.05, level: 0.4, grit: 1 },
  glass: { freq: 3600, decay: 0.11, level: 0.5, grit: 0.9 },
  water: { freq: 700, decay: 0.14, level: 0.55, grit: 0.7 },
  flesh: { freq: 300, decay: 0.05, level: 0.3, grit: 0.1 },
  foliage: { freq: 2600, decay: 0.09, level: 0.35, grit: 0.95 },
  rubber: { freq: 350, decay: 0.05, level: 0.35, grit: 0.15 },
  fabric: { freq: 500, decay: 0.045, level: 0.28, grit: 0.2 },
};

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _rel = new THREE.Vector3();

// ---------------------------------------------------------------------------

export class AudioSystem implements System {
  readonly name = 'audio';

  private ctx: GameContext | null = null;
  private ac: AudioContext | null = null;

  private master: GainNode | null = null;
  private dry: GainNode | null = null;
  private wet: GainNode | null = null;
  private convolver: ConvolverNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;
  /** Muffles the world when the player is badly hurt or an explosion lands close. */
  private duckFilter: BiquadFilterNode | null = null;

  private noise: AudioBuffer | null = null;
  private voices = 0;
  private started = false;
  private ready = false;

  private readonly listenerPos = new THREE.Vector3();
  private readonly listenerFwd = new THREE.Vector3(0, 0, -1);
  private readonly listenerRight = new THREE.Vector3(1, 0, 0);

  private concussion = 0;
  private lastFootstep = -1;
  private muted = false;

  private readonly unsubs: (() => void)[] = [];
  private readonly gestureEvents = ['pointerdown', 'keydown', 'touchstart'] as const;

  // =========================================================================
  // Lifecycle
  // =========================================================================

  init(ctx: GameContext): void {
    this.ctx = ctx;

    try {
      const Ctor: typeof AudioContext | undefined =
        window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (Ctor) this.ac = new Ctor({ latencyHint: 'interactive' });
    } catch {
      this.ac = null;
    }

    if (this.ac) {
      this.buildGraph(this.ac);
      this.ready = true;
      if (this.ac.state === 'running') this.started = true;
      for (const type of this.gestureEvents) {
        window.addEventListener(type, this.onGesture, { passive: true });
      }
      document.addEventListener('visibilitychange', this.onVisibility);
    }

    this.unsubs.push(
      ctx.events.on('shot.fired', this.onShot),
      ctx.events.on('shot.impact', this.onImpact),
      ctx.events.on('explosion', this.onExplosion),
      ctx.events.on('player.footstep', this.onFootstep),
      ctx.events.on('player.land', this.onLand),
      ctx.events.on('player.damaged', this.onDamaged),
      ctx.events.on('weapon.reload.start', this.onReloadStart),
      ctx.events.on('weapon.reload.end', this.onReloadEnd),
      ctx.events.on('weapon.equipped', this.onEquipped),
      ctx.events.on('ui.hitmarker', this.onHitmarker),
      ctx.events.on('entity.killed', this.onKilled),
      ctx.events.on('game.pause', this.onPause),
    );
  }

  dispose(): void {
    for (const off of this.unsubs) off();
    this.unsubs.length = 0;
    for (const type of this.gestureEvents) window.removeEventListener(type, this.onGesture);
    document.removeEventListener('visibilitychange', this.onVisibility);
    try {
      this.ac?.close();
    } catch {
      /* already closed */
    }
    this.ac = null;
    this.master = null;
    this.dry = null;
    this.wet = null;
    this.convolver = null;
    this.compressor = null;
    this.duckFilter = null;
    this.noise = null;
    this.ready = false;
    this.ctx = null;
  }

  private readonly onGesture = (): void => {
    const ac = this.ac;
    if (!ac) return;
    if (ac.state !== 'running') {
      void ac.resume().catch(() => undefined);
    }
    this.started = true;
  };

  private readonly onVisibility = (): void => {
    const ac = this.ac;
    if (!ac) return;
    if (document.hidden) void ac.suspend().catch(() => undefined);
    else if (this.started) void ac.resume().catch(() => undefined);
  };

  private readonly onPause = (p: { paused: boolean }): void => {
    this.muted = p.paused;
    const m = this.master;
    const ac = this.ac;
    if (!m || !ac) return;
    m.gain.cancelScheduledValues(ac.currentTime);
    m.gain.setTargetAtTime(p.paused ? MASTER_GAIN * 0.18 : MASTER_GAIN, ac.currentTime, 0.08);
  };

  // =========================================================================
  // Graph
  // =========================================================================

  private buildGraph(ac: AudioContext): void {
    this.noise = this.makeNoiseBuffer(ac, 2.0);

    const master = ac.createGain();
    master.gain.value = MASTER_GAIN;

    // Limiter-ish: a fast, hard compressor so a dozen simultaneous rifles do
    // not clip, without pumping on single shots.
    const comp = ac.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.knee.value = 6;
    comp.ratio.value = 9;
    comp.attack.value = 0.002;
    comp.release.value = 0.22;

    // Global muffle: shifted down by concussion and low health.
    const duck = ac.createBiquadFilter();
    duck.type = 'lowpass';
    duck.frequency.value = 20000;
    duck.Q.value = 0.7;

    const dry = ac.createGain();
    dry.gain.value = 1;
    const wet = ac.createGain();
    wet.gain.value = 0.85;

    const conv = ac.createConvolver();
    conv.buffer = this.makeStreetIR(ac);
    conv.normalize = true;

    dry.connect(duck);
    wet.connect(conv);
    conv.connect(duck);
    duck.connect(comp);
    comp.connect(master);
    master.connect(ac.destination);

    this.master = master;
    this.compressor = comp;
    this.duckFilter = duck;
    this.dry = dry;
    this.wet = wet;
    this.convolver = conv;
  }

  private makeNoiseBuffer(ac: AudioContext, seconds: number): AudioBuffer {
    const n = Math.floor(ac.sampleRate * seconds);
    const buf = ac.createBuffer(2, n, ac.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const data = buf.getChannelData(ch);
      let last = 0;
      for (let i = 0; i < n; i++) {
        const w = Math.random() * 2 - 1;
        // A touch of integration gives the noise a -3dB/oct tilt, which is much
        // closer to real acoustic noise than flat white.
        last = last * 0.22 + w * 0.78;
        data[i] = last;
      }
    }
    return buf;
  }

  /**
   * Street reverb: a handful of discrete early reflections off the facades
   * either side, then an exponentially decaying diffuse tail that darkens as it
   * decays (air and plaster both eat treble first).
   */
  private makeStreetIR(ac: AudioContext): AudioBuffer {
    const seconds = 1.6;
    const n = Math.floor(ac.sampleRate * seconds);
    const buf = ac.createBuffer(2, n, ac.sampleRate);

    // Early reflections in metres of extra path length, alternating walls.
    const early = [6.5, 9.2, 13.1, 17.8, 21.4, 26.9, 33.5, 41.2];

    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      let lp = 0;
      for (let i = 0; i < n; i++) {
        const t = i / ac.sampleRate;
        const env = Math.pow(1 - t / seconds, 2.6) * Math.exp(-t * 2.1);
        const w = (Math.random() * 2 - 1) * env;
        // One-pole low pass whose cutoff falls with time.
        const a = THREE.MathUtils.clamp(0.55 - t * 0.28, 0.08, 0.55);
        lp += (w - lp) * a;
        d[i] = lp * 0.9;
      }
      for (let k = 0; k < early.length; k++) {
        const metres = early[k] * (ch === 0 ? 1 : 1.07);
        const idx = Math.floor((metres / SPEED_OF_SOUND) * ac.sampleRate);
        if (idx >= n) continue;
        const amp = (0.6 / (1 + k * 0.85)) * (ch === 0 ? 1 : 0.92);
        d[idx] += amp;
        d[idx + 1] -= amp * 0.45;
      }
    }
    return buf;
  }

  // =========================================================================
  // Frame
  // =========================================================================

  update(time: FrameTime, ctx: GameContext): void {
    const ac = this.ac;
    if (!ac || !this.ready) return;

    ctx.camera.getWorldPosition(this.listenerPos);
    ctx.camera.getWorldDirection(this.listenerFwd);
    this.listenerRight.set(this.listenerFwd.z, 0, -this.listenerFwd.x);
    if (this.listenerRight.lengthSq() < 1e-6) this.listenerRight.set(1, 0, 0);
    else this.listenerRight.normalize();

    // Concussion decays back to a full-bandwidth mix over ~2.5s.
    if (this.concussion > 0) {
      this.concussion = Math.max(0, this.concussion - time.dt * 0.42);
      const duck = this.duckFilter;
      if (duck) {
        const cutoff = 20000 * Math.pow(0.045, this.concussion);
        duck.frequency.setTargetAtTime(Math.max(320, cutoff), ac.currentTime, 0.05);
      }
    }
  }

  // =========================================================================
  // Voice plumbing
  // =========================================================================

  private get live(): boolean {
    return this.ready && this.ac !== null && this.ac.state === 'running' && !this.muted;
  }

  /** Inverse-distance attenuation with a soft rolloff to silence at MAX_AUDIBLE. */
  private attenuation(distance: number): number {
    if (distance >= MAX_AUDIBLE) return 0;
    const g = REF_DISTANCE / Math.max(REF_DISTANCE, distance);
    const fade = 1 - distance / MAX_AUDIBLE;
    return g * fade * fade;
  }

  /**
   * Builds the per-voice output chain: stereo placement, distance low-pass and
   * a reverb send. Returns the node to connect a source into.
   */
  private spatial(at: THREE.Vector3 | null, tailAmount: number, distanceOut: { d: number }): GainNode | null {
    const ac = this.ac;
    const dry = this.dry;
    const wet = this.wet;
    if (!ac || !dry || !wet) return null;

    const input = ac.createGain();
    input.gain.value = 1;

    if (!at) {
      distanceOut.d = 0;
      input.connect(dry);
      if (tailAmount > 0) {
        const send = ac.createGain();
        send.gain.value = tailAmount;
        input.connect(send);
        send.connect(wet);
      }
      return input;
    }

    _rel.copy(at).sub(this.listenerPos);
    const distance = _rel.length();
    distanceOut.d = distance;

    const pan = ac.createStereoPanner();
    if (distance > 1e-4) {
      _fwd.copy(_rel).multiplyScalar(1 / distance);
      _right.copy(this.listenerRight);
      // Softened so sounds directly behind do not collapse to dead centre in a
      // way that makes them impossible to localise.
      pan.pan.value = THREE.MathUtils.clamp(_fwd.dot(_right) * 0.85, -1, 1);
    }

    // Air absorption: about -1dB/10m at 4kHz, modelled as a first-order shelf.
    const air = ac.createBiquadFilter();
    air.type = 'lowpass';
    air.frequency.value = THREE.MathUtils.clamp(20000 * Math.exp(-distance * 0.028), 900, 20000);
    air.Q.value = 0.6;

    const dist = ac.createGain();
    dist.gain.value = this.attenuation(distance);

    input.connect(air);
    air.connect(pan);
    pan.connect(dist);
    dist.connect(dry);

    if (tailAmount > 0) {
      const send = ac.createGain();
      // Further away means proportionally more reflected energy — that ratio is
      // most of the distance cue.
      send.gain.value = tailAmount * THREE.MathUtils.clamp(0.25 + distance * 0.03, 0.25, 1.6);
      dist.connect(send);
      send.connect(wet);
    }

    return input;
  }

  private claim(): boolean {
    if (this.voices >= MAX_VOICES) return false;
    this.voices++;
    return true;
  }

  private release(): void {
    this.voices = Math.max(0, this.voices - 1);
  }

  /** One shaped noise burst. Returns the scheduled stop time. */
  private burst(
    dest: AudioNode, when: number, level: number,
    filterType: BiquadFilterType, freq: number, q: number,
    attack: number, decay: number, rate = 1,
  ): void {
    const ac = this.ac;
    const noise = this.noise;
    if (!ac || !noise) return;

    if (!this.claim()) return;

    const src = ac.createBufferSource();
    src.buffer = noise;
    src.playbackRate.value = rate;
    // Random start offset so repeated shots never phase-cancel into a machine
    // gun of identical clicks.
    const offset = Math.random() * (noise.duration - decay - attack - 0.05);

    const filt = ac.createBiquadFilter();
    filt.type = filterType;
    filt.frequency.value = freq;
    filt.Q.value = q;

    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(Math.max(0.0002, level), when + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, when + attack + decay);

    src.connect(filt);
    filt.connect(g);
    g.connect(dest);

    src.onended = () => this.release();
    src.start(when, Math.max(0, offset), attack + decay + 0.02);
    src.stop(when + attack + decay + 0.02);
  }

  /** A pitch-swept oscillator: the "thump" layer under any percussive event. */
  private sweep(
    dest: AudioNode, when: number, level: number,
    f0: number, f1: number, decay: number, type: OscillatorType = 'sine',
  ): void {
    const ac = this.ac;
    if (!ac) return;
    if (!this.claim()) return;
    const osc = ac.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(f0, when);
    osc.frequency.exponentialRampToValueAtTime(Math.max(20, f1), when + decay);

    const g = ac.createGain();
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(Math.max(0.0002, level), when + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, when + decay);

    osc.connect(g);
    g.connect(dest);
    osc.onended = () => this.release();
    osc.start(when);
    osc.stop(when + decay + 0.02);
  }

  /** Metallic ring for ricochets and mechanism clicks: two detuned partials. */
  private ring(dest: AudioNode, when: number, level: number, freq: number, decay: number): void {
    const ac = this.ac;
    if (!ac) return;
    for (let i = 0; i < 2; i++) {
      if (!this.claim()) return;
      const osc = ac.createOscillator();
      osc.type = 'triangle';
      // A ratio slightly off an octave; exact harmonics sound synthetic.
      const f = freq * (i === 0 ? 1 : 2.07);
      osc.frequency.setValueAtTime(f * (0.97 + Math.random() * 0.06), when);
      osc.frequency.exponentialRampToValueAtTime(f * 0.86, when + decay);
      const g = ac.createGain();
      const lvl = level * (i === 0 ? 1 : 0.42);
      g.gain.setValueAtTime(0.0001, when);
      g.gain.exponentialRampToValueAtTime(Math.max(0.0002, lvl), when + 0.003);
      g.gain.exponentialRampToValueAtTime(0.0001, when + decay);
      osc.connect(g);
      g.connect(dest);
      osc.onended = () => this.release();
      osc.start(when);
      osc.stop(when + decay + 0.02);
    }
  }

  // =========================================================================
  // Event handlers
  // =========================================================================

  private readonly onShot = (ev: {
    weaponId: string; origin: THREE.Vector3; local: boolean;
  }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;

    const v = WEAPON_VOICES[ev.weaponId] ?? DEFAULT_VOICE;
    const d = { d: 0 };
    const input = this.spatial(ev.local ? null : ev.origin, v.tail, d);
    if (!input) return;

    // Sound travels: distant shots arrive late. This is the single cheapest
    // thing that makes a firefight feel like it is happening in a place.
    const delay = ev.local ? 0 : Math.min(0.45, d.d / SPEED_OF_SOUND);
    const t = ac.currentTime + delay + 0.001;
    const near = ev.local ? 1 : THREE.MathUtils.clamp(1 - d.d / 60, 0.1, 1);
    const lvl = v.level * (ev.local ? 0.5 : 0.42);

    if (this.attenuation(d.d) <= 0.0005 && !ev.local) return;
    if (this.voices > MAX_VOICES - 4 && !ev.local) return;

    // 1. Transient: the case-head crack. Very short, very bright.
    this.burst(input, t, lvl * 0.85 * near, 'highpass', 2600, 0.7, 0.0008, 0.012, 1.35);
    // 2. Body: the gas front. This is what identifies the weapon.
    this.burst(input, t + 0.0012, lvl * 1.0, 'bandpass', v.body, 0.85, 0.0025, v.decay, 1);
    // 3. Low: the punch you feel more than hear.
    this.sweep(input, t, lvl * 0.75, v.thump, v.thump * 0.42, v.decay * 1.7);
    // 4. Street tail: a longer, darker burst that mostly feeds the convolver.
    this.burst(input, t + 0.012, lvl * 0.34 * v.tail, 'lowpass', 1400, 0.5, 0.01, 0.20 + v.tail * 0.2, 0.7);

    if (ev.local) {
      // Mechanism: the bolt cycling, a beat behind the report.
      this.burst(input, t + 0.028, lvl * 0.16, 'bandpass', 3400, 3.2, 0.001, 0.03, 1.2);
      this.ring(input, t + 0.052, lvl * 0.05, 1850, 0.06);
    }
  };

  private readonly onImpact = (ev: {
    point: THREE.Vector3; surface: SurfaceKind; energy: number;
  }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const v = SURFACE_VOICES[ev.surface] ?? SURFACE_VOICES.concrete;
    const d = { d: 0 };
    const input = this.spatial(ev.point, 0.28, d);
    if (!input) return;
    if (this.attenuation(d.d) <= 0.001) return;
    if (this.voices > MAX_VOICES - 6) return;

    const e = THREE.MathUtils.clamp(ev.energy, 0.1, 1);
    const t = ac.currentTime + Math.min(0.35, d.d / SPEED_OF_SOUND) + 0.001;
    const lvl = v.level * (0.35 + e * 0.5) * 0.5;

    this.burst(input, t, lvl, 'bandpass', v.freq * (0.85 + Math.random() * 0.3), v.q, 0.001, v.decay);
    if (v.ring > 0.05) {
      this.ring(input, t + 0.004, lvl * v.ring * 0.5, v.freq * (0.9 + Math.random() * 0.5), v.decay * 2.4);
    }
    if (ev.surface !== 'flesh' && ev.surface !== 'foliage') {
      this.sweep(input, t, lvl * 0.35, v.freq * 0.16, v.freq * 0.07, 0.05);
    }
  };

  private readonly onExplosion = (ev: { point: THREE.Vector3; radius: number }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const d = { d: 0 };
    const input = this.spatial(ev.point, 1.4, d);
    if (!input) return;
    const t = ac.currentTime + Math.min(0.5, d.d / SPEED_OF_SOUND) + 0.001;
    const r = Math.max(1, ev.radius);

    this.burst(input, t, 1.0, 'highpass', 1800, 0.6, 0.001, 0.05, 1.2);
    this.burst(input, t + 0.004, 1.3, 'lowpass', 900, 0.5, 0.008, 0.42 + r * 0.02, 0.55);
    this.sweep(input, t, 1.15, 92, 26, 0.6);
    this.burst(input, t + 0.09, 0.5, 'lowpass', 420, 0.4, 0.05, 1.1, 0.4);

    // Close blasts blow out the top end for a couple of seconds.
    const proximity = THREE.MathUtils.clamp(1 - d.d / (r * 3.5), 0, 1);
    if (proximity > 0.05) this.concussion = Math.max(this.concussion, proximity);
  };

  private readonly onFootstep = (ev: {
    position: THREE.Vector3; surface: SurfaceKind; running: boolean;
  }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    // Guard against two systems both reporting the same step.
    if (ac.currentTime - this.lastFootstep < 0.09) return;
    this.lastFootstep = ac.currentTime;

    const v = FOOT_VOICES[ev.surface] ?? FOOT_VOICES.concrete;
    const d = { d: 0 };
    const input = this.spatial(ev.position, 0.16, d);
    if (!input) return;
    if (this.attenuation(d.d) <= 0.002) return;

    const t = ac.currentTime + 0.001;
    const lvl = v.level * (ev.running ? 0.5 : 0.3);
    const jitter = 0.88 + Math.random() * 0.26;

    // Heel strike, then the scuff of the sole rolling forward.
    this.burst(input, t, lvl, 'lowpass', v.freq * jitter, 0.8, 0.0015, v.decay);
    if (v.grit > 0.1) {
      this.burst(input, t + 0.012 + Math.random() * 0.01, lvl * v.grit * 0.42,
        'highpass', 3200 * jitter, 0.5, 0.004, v.decay * 1.6, 0.9);
    }
    this.sweep(input, t, lvl * 0.5, v.freq * 0.28, v.freq * 0.13, 0.045);
  };

  private readonly onLand = (ev: { position: THREE.Vector3; impact: number; surface: SurfaceKind }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const v = FOOT_VOICES[ev.surface] ?? FOOT_VOICES.concrete;
    const d = { d: 0 };
    const input = this.spatial(ev.position, 0.3, d);
    if (!input) return;
    const e = THREE.MathUtils.clamp(ev.impact, 0.1, 1);
    const t = ac.currentTime + 0.001;

    this.burst(input, t, 0.55 * e, 'lowpass', v.freq * 0.8, 0.7, 0.002, v.decay * 2.2);
    this.sweep(input, t, 0.7 * e, 130, 48, 0.14);
    // Kit rattle on a hard landing.
    this.burst(input, t + 0.03, 0.16 * e, 'bandpass', 2800, 1.6, 0.005, 0.09, 1.1);
  };

  private readonly onDamaged = (ev: { amount: number; health: number }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const d = { d: 0 };
    const input = this.spatial(null, 0.1, d);
    if (!input) return;
    const t = ac.currentTime + 0.001;
    const e = THREE.MathUtils.clamp(ev.amount / 40, 0.2, 1);

    this.burst(input, t, 0.5 * e, 'lowpass', 700, 0.8, 0.002, 0.09);
    this.sweep(input, t, 0.55 * e, 160, 55, 0.18);
    // Near death, the mix goes underwater and a ringing tone creeps in.
    if (ev.health < 35) {
      this.concussion = Math.max(this.concussion, 0.55 * (1 - ev.health / 35));
      this.ring(input, t + 0.02, 0.035, 4200, 1.4);
    }
  };

  private readonly onReloadStart = (ev: { duration: number }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const d = { d: 0 };
    const input = this.spatial(null, 0.12, d);
    if (!input) return;
    const t = ac.currentTime + 0.001;
    const dur = Math.max(0.6, ev.duration);

    // Mag release, mag out, mag in. Spaced across the animation so the audio
    // and the viewmodel agree about what is happening.
    this.burst(input, t + dur * 0.08, 0.20, 'bandpass', 2600, 4.5, 0.001, 0.028, 1.3);
    this.ring(input, t + dur * 0.09, 0.05, 2400, 0.05);
    this.burst(input, t + dur * 0.30, 0.16, 'bandpass', 1400, 2.0, 0.003, 0.06, 0.9);
    this.burst(input, t + dur * 0.66, 0.26, 'lowpass', 1600, 1.1, 0.001, 0.05, 1.0);
    this.ring(input, t + dur * 0.67, 0.06, 1750, 0.07);
  };

  private readonly onReloadEnd = (): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const d = { d: 0 };
    const input = this.spatial(null, 0.12, d);
    if (!input) return;
    const t = ac.currentTime + 0.001;
    // Bolt release: a hard metallic slam.
    this.burst(input, t, 0.30, 'bandpass', 3100, 3.0, 0.0008, 0.035, 1.25);
    this.ring(input, t + 0.002, 0.085, 2050, 0.11);
    this.sweep(input, t, 0.12, 320, 150, 0.05);
  };

  private readonly onEquipped = (): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const d = { d: 0 };
    const input = this.spatial(null, 0.1, d);
    if (!input) return;
    const t = ac.currentTime + 0.001;
    this.burst(input, t, 0.16, 'bandpass', 900, 0.9, 0.01, 0.12, 0.85);
    this.burst(input, t + 0.09, 0.14, 'bandpass', 2400, 2.4, 0.002, 0.05, 1.1);
  };

  private readonly onHitmarker = (ev: { headshot: boolean; lethal: boolean }): void => {
    if (!this.live) return;
    const ac = this.ac;
    if (!ac) return;
    const d = { d: 0 };
    const input = this.spatial(null, 0, d);
    if (!input) return;
    const t = ac.currentTime + 0.001;
    // Non-diegetic confirmation blip: short, bright, and pitched by hit quality
    // so the player learns it without looking.
    const f = ev.headshot ? 1750 : 1180;
    this.ring(input, t, ev.lethal ? 0.09 : 0.06, f, ev.headshot ? 0.075 : 0.045);
    this.burst(input, t, 0.05, 'highpass', 4200, 1.2, 0.0008, 0.012, 1.4);
  };

  private readonly onKilled = (ev: { entityId: number }): void => {
    if (!this.live) return;
    const ctx = this.ctx;
    if (!ctx || ev.entityId === ctx.localPlayerId) return;
    const ac = this.ac;
    if (!ac) return;
    const target = ctx.entities.get(ev.entityId);
    const d = { d: 0 };
    const input = this.spatial(target ? target.position : null, 0.4, d);
    if (!input) return;
    const t = ac.currentTime + Math.min(0.25, d.d / SPEED_OF_SOUND) + 0.001;
    // Body and kit hitting the ground, a beat after the round lands.
    this.burst(input, t + 0.34, 0.42, 'lowpass', 520, 0.7, 0.004, 0.16, 0.7);
    this.sweep(input, t + 0.34, 0.4, 110, 42, 0.2);
    this.burst(input, t + 0.40, 0.14, 'bandpass', 2600, 1.4, 0.006, 0.11, 1.0);
  };
}
