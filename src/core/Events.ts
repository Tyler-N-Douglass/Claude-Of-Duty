import type { EventBus, GameEvents } from './Contracts';

type Handler = (payload: any) => void;

/**
 * Minimal typed pub/sub. Handlers added during a dispatch are not called until
 * the next emit, and removal during dispatch is safe (we iterate a snapshot).
 */
export class Emitter implements EventBus {
  private readonly map = new Map<string, Set<Handler>>();

  on<K extends keyof GameEvents>(type: K, fn: (payload: GameEvents[K]) => void): () => void {
    let set = this.map.get(type as string);
    if (!set) {
      set = new Set();
      this.map.set(type as string, set);
    }
    set.add(fn as Handler);
    return () => this.off(type, fn);
  }

  once<K extends keyof GameEvents>(type: K, fn: (payload: GameEvents[K]) => void): () => void {
    const wrapped = (payload: GameEvents[K]) => {
      this.off(type, wrapped);
      fn(payload);
    };
    return this.on(type, wrapped);
  }

  off<K extends keyof GameEvents>(type: K, fn: (payload: GameEvents[K]) => void): void {
    this.map.get(type as string)?.delete(fn as Handler);
  }

  emit<K extends keyof GameEvents>(type: K, payload: GameEvents[K]): void {
    const set = this.map.get(type as string);
    if (!set || set.size === 0) return;
    for (const fn of Array.from(set)) {
      try {
        fn(payload);
      } catch (err) {
        console.error(`[events] handler for "${String(type)}" threw`, err);
      }
    }
  }

  clear(): void {
    this.map.clear();
  }
}
