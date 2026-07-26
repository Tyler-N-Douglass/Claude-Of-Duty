import type { ActionName, InputState } from './Contracts';

/** Default bindings. Codes are KeyboardEvent.code, or Mouse0/1/2, or Wheel±. */
const DEFAULT_BINDINGS: Record<ActionName, string[]> = {
  forward: ['KeyW', 'ArrowUp'],
  back: ['KeyS', 'ArrowDown'],
  left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'],
  jump: ['Space'],
  crouch: ['ControlLeft', 'KeyC'],
  sprint: ['ShiftLeft'],
  walk: ['AltLeft'],
  fire: ['Mouse0'],
  aim: ['Mouse2'],
  reload: ['KeyR'],
  melee: ['KeyV', 'Mouse1'],
  grenade: ['KeyG'],
  swapWeapon: ['KeyQ', 'Digit1', 'Digit2'],
  interact: ['KeyF', 'KeyE'],
  leanLeft: ['KeyZ'],
  leanRight: ['KeyX'],
  pause: ['Escape'],
  scoreboard: ['Tab'],
  flashlight: ['KeyT'],
};

/** Keys we swallow so the browser doesn't scroll/focus behind the game. */
const SWALLOW = new Set(['Space', 'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);

export class Input implements InputState {
  readonly lookDelta = { x: 0, y: 0 };
  readonly moveAxis = { x: 0, y: 0 };
  wheelDelta = 0;
  pointerLocked = false;

  private readonly bindings = DEFAULT_BINDINGS;
  private readonly down = new Set<string>();
  private readonly pressed = new Set<string>();
  private readonly released = new Set<string>();
  private sensitivity = 0.0022;
  private adsScale = 0.75;
  private aiming = false;
  private pendingLook = { x: 0, y: 0 };
  private pendingWheel = 0;
  private readonly target: HTMLElement;

  constructor(target: HTMLElement) {
    this.target = target;
    window.addEventListener('keydown', this.onKeyDown, { passive: false });
    window.addEventListener('keyup', this.onKeyUp);
    target.addEventListener('mousedown', this.onMouseDown);
    window.addEventListener('mouseup', this.onMouseUp);
    window.addEventListener('mousemove', this.onMouseMove);
    target.addEventListener('wheel', this.onWheel, { passive: false });
    target.addEventListener('contextmenu', this.onContextMenu);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);
    window.addEventListener('blur', this.onBlur);
  }

  // -- public API ----------------------------------------------------------

  isDown(action: ActionName): boolean {
    for (const code of this.bindings[action]) if (this.down.has(code)) return true;
    return false;
  }

  wasPressed(action: ActionName): boolean {
    for (const code of this.bindings[action]) if (this.pressed.has(code)) return true;
    return false;
  }

  wasReleased(action: ActionName): boolean {
    for (const code of this.bindings[action]) if (this.released.has(code)) return true;
    return false;
  }

  setSensitivity(v: number): void {
    this.sensitivity = Math.max(0.0002, v);
  }

  setAdsSensitivityScale(v: number): void {
    this.adsScale = v;
  }

  /** Called by the weapon system so aiming slows the look speed. */
  setAiming(v: boolean): void {
    this.aiming = v;
  }

  requestPointerLock(): void {
    const el = this.target as HTMLElement & { requestPointerLock?: (o?: object) => Promise<void> | void };
    try {
      const r = el.requestPointerLock?.({ unadjustedMovement: true });
      // Some browsers reject unadjustedMovement; fall back to the plain call.
      if (r && typeof (r as Promise<void>).catch === 'function') {
        (r as Promise<void>).catch(() => el.requestPointerLock?.());
      }
    } catch {
      el.requestPointerLock?.();
    }
  }

  exitPointerLock(): void {
    document.exitPointerLock?.();
  }

  /**
   * Must be called once per frame, before systems update: promotes the pending
   * mouse deltas and clears the edge-triggered sets from the previous frame.
   */
  beginFrame(): void {
    const scale = this.sensitivity * (this.aiming ? this.adsScale : 1);
    this.lookDelta.x = this.pendingLook.x * scale;
    this.lookDelta.y = this.pendingLook.y * scale;
    this.pendingLook.x = 0;
    this.pendingLook.y = 0;

    this.wheelDelta = this.pendingWheel;
    this.pendingWheel = 0;

    let mx = 0;
    let my = 0;
    if (this.isDown('right')) mx += 1;
    if (this.isDown('left')) mx -= 1;
    if (this.isDown('forward')) my += 1;
    if (this.isDown('back')) my -= 1;
    const len = Math.hypot(mx, my);
    if (len > 1) {
      mx /= len;
      my /= len;
    }
    this.moveAxis.x = mx;
    this.moveAxis.y = my;
  }

  /** Called at the very end of the frame to clear edges. */
  endFrame(): void {
    this.pressed.clear();
    this.released.clear();
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    this.target.removeEventListener('mousedown', this.onMouseDown);
    window.removeEventListener('mouseup', this.onMouseUp);
    window.removeEventListener('mousemove', this.onMouseMove);
    this.target.removeEventListener('wheel', this.onWheel);
    this.target.removeEventListener('contextmenu', this.onContextMenu);
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    window.removeEventListener('blur', this.onBlur);
  }

  // -- handlers ------------------------------------------------------------

  private press(code: string): void {
    if (!this.down.has(code)) {
      this.down.add(code);
      this.pressed.add(code);
    }
  }

  private release(code: string): void {
    if (this.down.delete(code)) this.released.add(code);
  }

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (SWALLOW.has(e.code)) e.preventDefault();
    if (e.repeat) return;
    this.press(e.code);
  };

  private readonly onKeyUp = (e: KeyboardEvent): void => {
    this.release(e.code);
  };

  private readonly onMouseDown = (e: MouseEvent): void => {
    this.press(`Mouse${e.button}`);
  };

  private readonly onMouseUp = (e: MouseEvent): void => {
    this.release(`Mouse${e.button}`);
  };

  private readonly onMouseMove = (e: MouseEvent): void => {
    if (!this.pointerLocked) return;
    // Guard against the occasional huge spike browsers emit on lock acquisition.
    const dx = Math.abs(e.movementX) > 400 ? 0 : e.movementX;
    const dy = Math.abs(e.movementY) > 400 ? 0 : e.movementY;
    this.pendingLook.x += dx;
    this.pendingLook.y += dy;
  };

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.pendingWheel += Math.sign(e.deltaY);
  };

  private readonly onContextMenu = (e: Event): void => {
    e.preventDefault();
  };

  private readonly onPointerLockChange = (): void => {
    this.pointerLocked = document.pointerLockElement === this.target;
    if (!this.pointerLocked) this.clearAll();
  };

  private readonly onBlur = (): void => {
    this.clearAll();
  };

  private clearAll(): void {
    for (const code of this.down) this.released.add(code);
    this.down.clear();
    this.pendingLook.x = 0;
    this.pendingLook.y = 0;
  }
}
