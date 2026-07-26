/**
 * Boot: builds the context, registers every system in dependency order, and
 * starts the loop. Registration order is the update order, so it matters:
 *
 *   physics -> player -> ai -> weapons -> ballistics -> fx -> audio -> ui -> render
 *
 * `render` must be last because it is the RenderSystem and presents the frame.
 */
import { Engine } from './core/Engine';
import { Input } from './core/Input';
import { Entities } from './core/Entities';
import { createQuality, detectPreset, applyPreset } from './core/Quality';

import { PhysicsSystem } from './physics/Physics';
import { SkySystem } from './render/Sky';
import { LightingSystem } from './render/Lighting';
import { RenderPipeline } from './render/RenderPipeline';
import { LevelSystem } from './world/Level';
import { PlayerSystem } from './player/Player';
import { WeaponSystem } from './weapons/Weapons';
import { BallisticsSystem } from './weapons/Ballistics';
import { EffectsSystem } from './fx/Effects';
import { AISystem } from './ai/AI';
import { AudioSystem } from './audio/Audio';
import { HUDSystem } from './ui/HUD';
import { MenuSystem } from './ui/Menu';

async function boot(): Promise<void> {
  const container = document.getElementById('app');
  if (!container) throw new Error('#app container missing');

  const input = new Input(container);
  const entities = new Entities();
  const physics = new PhysicsSystem();
  const quality = createQuality('high');

  const engine = new Engine({ container, input, physics, entities, quality });
  applyPreset(quality, detectPreset(engine.renderer));

  engine
    .add(physics)
    .add(new SkySystem())
    .add(new LightingSystem())
    .add(new LevelSystem())
    .add(new PlayerSystem())
    .add(new AISystem())
    .add(new WeaponSystem())
    .add(new BallisticsSystem())
    .add(new EffectsSystem())
    .add(new AudioSystem())
    .add(new HUDSystem())
    .add(new MenuSystem())
    .add(new RenderPipeline());

  await engine.init();
  engine.start();

  // Expose for the automated visual-QA harness and for debugging.
  (window as unknown as { GAME: unknown }).GAME = { engine, ctx: engine.ctx, input, quality };
  document.dispatchEvent(new CustomEvent('game-ready'));
}

boot().catch((err) => {
  console.error(err);
  const el = document.createElement('pre');
  el.style.cssText = 'color:#f66;font:14px monospace;padding:24px;white-space:pre-wrap';
  el.textContent = `Failed to start:\n${err?.stack ?? err}`;
  document.getElementById('app')?.appendChild(el);
});
