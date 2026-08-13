# Hyperframes Composition Brief: Claude of Duty (hype cut, narrated)

## Objective
Cut a 2010s-style console-shooter hype trailer for **Claude of Duty**, narrated
by an epic announcer, aimed at hooking a viewer who has never seen the game.
Every cut on the downbeat; one boast per scene.

## Output
- Composition directory: `brag-output-2026-08-13-174427/composition/`
- Rendered video: `brag-output-2026-08-13-174427/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 24.5 seconds

## Source Material
- Project root: `/home/user/Claude-Of-Duty`
- Primary files read: `src/world/Level.ts` (Dust Corridor, the three lanes),
  `src/weapons/WeaponSpecs.ts` (the six-weapon roster and per-weapon recoil
  patterns / damage falloff), `src/ai/AI.ts` (the enemy FSM: perception cone,
  line of sight, A* pathfinding, cover that breaks the player's line),
  `src/ui/Style.ts` (`PALETTE` and type), `src/ui/Menu.ts` (Task Force // Dust
  Corridor)
- Measured live off `renderer.info` in the running game: 1,039,810 triangles,
  202 textures, 152 geometries, 299 draw calls
- Product name: **Claude of Duty**
- Tagline / strongest claim: a real shooter — six weapons, an enemy squad, a
  three-lane map — with **zero image files**, in one browser tab
- Frames to use as scene backgrounds (stills captured from the running game by
  driving `PlayerSystem.teleport` in headless Chromium; live HUD in every one):
  `assets/frames/tower.jpg`, `square.jpg`, `east.jpg`, `arcade.jpg`,
  `street.jpg`

- Copy that must appear verbatim:
  - `DUST CORRIDOR`
  - `MK4 RANGER`, `VKS-9 WASP`, `M16-BR`, `KV-800 BALLISTA`, `M870 BREACHER`,
    `P226 SIDEARM`
  - `Task Force // Dust Corridor`
  - `CLAUDE OF DUTY`

## Creative Direction
- Tone preset: `chaotic` (for pacing and SFX density — 6 scenes, hard cuts)
- Creative direction: *incredibly braggadocios gameplay trailer from the 2010s
  meant to hook new players*, with an epic announcer.
- Interpretation: ALL-CAPS callouts that **snap** rather than fade, fast pushes
  on every frame, a heavy impact on every cut, dense but purposeful SFX. The
  narration owns the pace — never compress a line to fit a beat.
- Angle: see `brag-plan.md`. Every boast is literally true and traceable to a
  file in the repo; the braggadocio is the register, not an exaggeration.
- Hook: hard in on the bell-tower square, fast push, announcer: *"This is Dust
  Corridor."*
- Outro / punchline: title slam on the biggest downbeat, then
  **No download. No install. One tab.**
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Any claim not backed by the source
  - Letting an SFX cue collide with a narration line

## Visual Identity
- Background: `#0b0d0e`, ink `#05070a`
- Text: `#f3f1ea`; dim `rgba(243,241,234,.62)`
- Accent: `#e2a53d`
- Display font: the game's condensed stack, resolving to
  `"Liberation Sans","DejaVu Sans",Arial` here, weights 700, uppercase,
  `scaleX(.9)`, tracking `.03em`-`.42em`
- Data font: `"DejaVu Sans Mono","Liberation Mono",ui-monospace`, tabular nums
- Visual references from the project: 1px hairline rules, amber eyebrow of a
  62x1px rule plus wide-tracked caps, hard black shadow under every stroke

## Storyboard
Use the storyboard in `brag-plan.md` as the creative contract.

Scene summary (all cuts beat-locked at 120.19 BPM):
1. **This is Dust Corridor** — 0.00–3.52 — bell-tower frame, fast push,
   `TASK FORCE` / `DUST CORRIDOR`.
2. **Three lanes** — 3.52–7.52 — fountain frame; `THREE LANES.` /
   `ONE SQUARE.` / `NOWHERE TO HIDE.` snap in on beats, then hold together.
3. **Six weapons** — 7.52–11.52 — plaza frame; `SIX WEAPONS` then the roster as
   two columns of three, one name per beat with a click, full set held 1.0s.
4. **They hunt you** — 11.52–16.02 — arcade frame; `THEY HUNT YOU.` /
   `AND TAKE REAL COVER.` / mono `perception cone · line of sight ·
   A* pathfinding`.
5. **The numbers** — 16.02–20.02 — street frame; `1,039,810 TRIANGLES` /
   `202 TEXTURES` / `0 IMAGE FILES` on three consecutive strong cues.
6. **Title** — 20.02–24.50 — fountain frame; `CLAUDE OF DUTY` slams with a sheen
   sweep, eyebrow, then `NO DOWNLOAD. NO INSTALL. ONE TAB.`

## Audio
- Audio role: dense rhythmic layer under a narrated trailer.
- Audio arc: full energy from frame 0, ducking under each of six announcer
  lines and lifting in the gaps, biggest downbeat under the title slam, then a
  clean ring-out.
- Music: `happy-beats-business-moves-vol-1-by-ende-dot-app.mp3` (120.19 BPM).
- Music treatment: baseline `data-volume` 0.32; tween to 0.15 under each VO
  line and back up in each gap; back to 0.34 for the ring-out; fade to 0 across
  the final 0.6s.
- Music cue guidance: bundled preset at
  `assets/music/cues/happy-beats-business-moves-vol-1-by-ende-dot-app.music-cues.json`.
  Cuts at 3.52 / 7.52 / 11.52 / 16.02 / 20.02 (all beats; 16.02 and 20.02 are
  1.00-intensity strong cues). Weapon roster on the beat grid at 8.02 → 10.52
  in 0.5s steps. Stat lines on the consecutive strong cues 16.52 / 17.52 /
  18.52. Title furniture on 20.02 / 21.01 / 22.01. This is more locking than the
  usual 1-3 because the genre demands it and the window is uniformly strong;
  readability is protected by the narration timing instead.
- Audio-reactive treatment: subtle — amber horizon warmth in the gameplay frames
  breathes with music RMS and bass. No waveform, equalizer, or strobing.
- Voiceover: **enabled** (`--voice`). Kokoro `am_onyx` at speed 0.9, chosen by
  measuring median F0 across six candidate male voices (86.6 Hz vs 123-157 Hz).
  Six pre-generated WAVs in `assets/vo/vo-01.wav` … `vo-06.wav`, each on its own
  track, placed at 0.50 / 3.80 / 7.80 / 11.70 / 16.30 / 20.45. Scene durations
  were derived from these, not the reverse.
- Audio-coupled moments: every scene cut (heavy punch); each weapon name
  (click); each stat line (drop); the title slam (deep bell).
- SFX selection guidance: `impact/impactPunch_heavy_*` and
  `impact/impactPlate_heavy_000` on cuts, `interface/click_001` per weapon
  name, `interface/drop_001` per stat, `impact/impactBell_heavy_000` on the
  slam. Short cues only, placed in the transient before a narration line — never
  over one.
- Audio files: all copied into `composition/assets/`.

## Hyperframes Instructions
Load `hyperframes-core`, `hyperframes-animation`, `hyperframes-creative`,
`hyperframes-keyframes`, `hyperframes-cli`. /brag is its own workflow: do not
enter the `hyperframes` entry-point intent interview or its generic promo
workflow.

Requirements:
- Show real captured game frames in every scene.
- Keep all text readable; the narration sets the pace.
- Keep the video within 15-25 seconds (24.5s here, set by the VO).
- Voiceover on its own track per line; music ducks to 0.15 for the duration of
  each line and returns between them.
- Dense but non-colliding SFX, per the tone.
- Subtle audio-reactive warmth only.
- Run `hyperframes check` before render — brag's single gate.
