import type { DamageTarget, EntityRegistry } from './Contracts';

let nextId = 1;

/** Allocates a process-unique entity id. Id 0 is reserved for the local player. */
export function allocEntityId(): number {
  return nextId++;
}

export class Entities implements EntityRegistry {
  private readonly map = new Map<number, DamageTarget>();
  private list: DamageTarget[] = [];
  private dirty = false;

  register(target: DamageTarget): void {
    this.map.set(target.entityId, target);
    this.dirty = true;
  }

  unregister(entityId: number): void {
    if (this.map.delete(entityId)) this.dirty = true;
  }

  get(entityId: number): DamageTarget | undefined {
    return this.map.get(entityId);
  }

  get all(): readonly DamageTarget[] {
    if (this.dirty) {
      this.list = Array.from(this.map.values());
      this.dirty = false;
    }
    return this.list;
  }
}
