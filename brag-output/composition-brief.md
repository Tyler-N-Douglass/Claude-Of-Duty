# Hyperframes Composition Brief: Claude of Duty

## Objective
Create a short launch-style brag video for **Claude of Duty** — a browser FPS
built in TypeScript and three.js with no image assets at all. Play it dead
straight as a AAA game trailer; the spec sheet underneath is the joke.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 21 seconds

## Source Material
- Project root: `/home/user/Claude-Of-Duty`
- Primary files read: `index.html` (boot overlay markup + inline CSS),
  `src/ui/Style.ts` (the whole UI stylesheet and `PALETTE`), `src/ui/Menu.ts`
  (main-menu brand and Deployment brief copy), `src/weapons/WeaponSpecs.ts`
  (the weapon roster), `src/world/Level.ts` (map description),
  `qa/RUBRIC.md` and `qa/CHAMPION.md` (the visual-QA loop), `package.json`
- Product name: **Claude of Duty**
- Tagline / strongest claim: 38,926 lines of TypeScript, **zero image files** —
  every texture, weapon and building is generated at runtime — running in a
  browser tab
- Key UI or visual moment to recreate: the boot overlay (letterspaced title +
  260x2px amber progress bar + lowercase status label), and the main menu's
  Deployment brief rows rendered in the game's own condensed uppercase type
- Real captured frames to use as scene backgrounds (1280x720 stills from the
  running game, captured with the project's own Playwright QA harness):
  `assets/frames/*.jpg` — town square, sun-raked shadows, MK4 Ranger viewmodel,
  and the live HUD (compass, `GRID K10` minimap, `MK4 RANGER 30/210` ammo)

- Copy that must appear verbatim:
  - `CLAUDE OF DUTY`
  - `building level`
  - `Task Force // Dust Corridor`
  - `THEATRE` / `DUST CORRIDOR`
  - `CONTRACT` / `SEARCH & DESTROY`
  - `THREAT` / `VETERAN`
  - `r3b  vs  r3a` → `REGRESSED`
  - `r3c  vs  r3a` → `CHAMPION`

## Creative Direction
- Tone preset: `cinematic`
- Creative direction: a straight-faced AAA console trailer for something that is
  actually 37 TypeScript files and no images.
- Interpretation: big letterspaced caps, full-bleed frames, slow pushes instead
  of quick cuts, real holds on every read. Never wink at the joke — the humour
  is entirely in the deadpan delivery of true facts. Restraint over energy.
- Angle: The boot screen, the deployment brief, the compass and the MK4 Ranger's
  30/210 are all real and all shipping, so treat the thing as a game and cut a
  game trailer. Then let the spec sheet land — thirty-nine thousand lines, no art
  pipeline, one tab — and finish on the QA loop, where the project screenshotted
  itself, judged the frames blind against real Call of Duty stills, and lost to
  itself twice before it won.
- Hook: black, then the real boot bar filling in amber under letterspaced
  `CLAUDE OF DUTY`, with the actual `building level` status ticking underneath.
- Outro / punchline: the title slams over the darkened square and the last line
  lands dry — **No textures. No art team. One tab.**
- Avoid:
  - Generic SaaS language
  - Abstract filler visuals
  - Unrelated visual redesign
  - Anything that undercuts the straight face (comedy SFX, wink typography)

## Visual Identity
- Background: `#0b0d0e` (boot / page), deepest ink `#05070a`
- Text: `#f3f1ea`; dim `rgba(243,241,234,.62)`; faint `rgba(243,241,234,.30)`;
  boot title `#e8e4dc`
- Accent: `#c8a44a` (boot bar) and `#e2a53d` (in-game UI amber). Use `#c8a44a`
  in the boot scene and `#e2a53d` everywhere else.
- Danger (only on `REGRESSED`): `#e0413a`
- Display font: the game's own condensed stack —
  `"Roboto Condensed","Barlow Condensed","Archivo Narrow","Liberation Sans Narrow","Arial Narrow","Helvetica Neue",Inter,system-ui,sans-serif`,
  weights 700-800, uppercase, `transform: scaleX(.93)`, tracking `.05em`-`.4em`
- Body / data font: `ui-monospace, "SF Mono","Roboto Mono","DejaVu Sans Mono",Menlo,Consolas,monospace`,
  with `font-variant-numeric: tabular-nums`
- Visual references from the project: 1px hairline rules under section labels;
  an amber eyebrow of a 44x1px rule plus `.42em` tracked uppercase text; a hard
  black text shadow (`0 1px 2px rgba(0,0,0,.92), 0 0 10px rgba(0,0,0,.5)`) under
  every stroke so type survives a blown-out sky; a 2px amber left-edge marker on
  the active row; key/value rows separated by `1px solid rgba(243,241,234,.05)`

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. **Boot** — 3.0s — black; letterspaced `CLAUDE OF DUTY`; the amber 2px bar
   fills 0→100%; the status label ticks `starting` → `building level` → `ready`.
2. **Deployment** — 4.2s — real gameplay frame with a slow 1.04→1.00 push; the
   Deployment brief arrives row by row on the right third
   (`THEATRE / DUST CORRIDOR`, `CONTRACT / SEARCH & DESTROY`, `THREAT / VETERAN`),
   then holds.
3. **The spec sheet** — 4.8s — second gameplay frame, darkened; three lines land
   in sequence and hold together: `38,926 LINES OF TYPESCRIPT`,
   `ZERO IMAGE FILES` (with small amber `every texture generated at runtime`),
   `ONE BROWSER TAB.`
4. **Judged blind** — 4.6s — near-black over a ghosted frame;
   `JUDGED BLIND AGAINST THE REAL THING.` then two monospace ladder rows,
   `r3b vs r3a → REGRESSED` (red) and `r3c vs r3a → CHAMPION` (amber).
5. **Title** — 4.4s — full-bleed darkened square; `CLAUDE OF DUTY` slams
   1.06→1.00 with a single sheen sweep; amber eyebrow rule +
   `Task Force // Dust Corridor`; then the dry closing line
   `No textures. No art team. One tab.`

Reading-time floor already budgeted in the plan: each short line gets ≥0.8s
settled and each sentence ≥0.3s per word. Do not compress a read to fit a beat.

## Audio
- Audio role: cinematic support — a low steady bed that carries the trailer and
  gets out of the way of the reads.
- Audio arc: enters from silence under the boot bar, opens up as the square
  appears, sits at full weight through the spec sheet, pulls back for the
  blind-judgement rows, fades out under a single bell on the title slam.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady, clean,
  109.96 BPM)
- Music treatment: `data-start="0"`, volume 0.30-0.35, hold through the body,
  duck slightly under the outro so the final hit rings past the bed; fade out
  across the last ~1.5s.
- Music cue guidance: bundled preset at
  `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`
  (copied into the composition). Target strong cues: **8.74s** (spec sheet
  lands), **13.11s** (ladder scene opens), **17.47s** or **18.56s** (title slam)
  — lock 1-3 of these, not all. Beat grid available for the sequential rows:
  brief rows near 4.39 / 5.34 / 6.00; spec rows near 9.29 / 10.37 / 11.46
  (every other beat — the full grid at ~0.54s spacing is too fast for readable
  lines); ladder rows near 14.20 / 15.29.
- Audio-reactive treatment: subtle. Let the sky/horizon glow in the gameplay
  frames and the amber accent presence breathe with music RMS. No waveform or
  equalizer visuals, no strobing, no text scaling driven by audio.
- Audio-coupled moments:
  - Boot bar completing — one very soft cue as it fills; nothing on the label swaps
  - First Deployment row arriving — a single dry accent, not one per row
  - Spec-sheet first line landing — a soft impact, on or near the 8.74s cue
  - `CHAMPION` row — one dry accent
  - Title slam — deep bell, beat-locked
- SFX selection guidance: cinematic restraint, 3-4 cues total across 21 seconds.
  A deep bell family (`impact/impactBell_heavy_*`) for the title; soft impacts
  (`impact/impactSoft_medium_*`) for the reveal; something dry and small for the
  ladder accent. Nothing comedic, nothing glitchy, nothing on every row.
- SFX analysis guidance: `~/.claude/skills/brag/assets/sfx/sfx-analysis.md` —
  prefer low high-frequency-risk files; this edit is polished, not chaotic.
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and
  volume based on the implemented animation.
- Audio files: copy the chosen music and any Hyperframes-selected SFX into
  `brag-output/composition/assets/`.

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — `hyperframes-core`
(composition contract + `data-*` timing), `hyperframes-animation` (motion),
`hyperframes-creative` (design spec, beats, audio-reactive),
`hyperframes-keyframes` (seek-safe keyframes), and `hyperframes-cli`
(lint/check/render). /brag is its own workflow: do not enter the `hyperframes`
entry-point intent interview and do not route into its generic promo /
launch-video workflow. Prefer native Hyperframes conventions over anything in
`/brag`.

Requirements:
- Show at least one real UI, copy, or visual element from the source project —
  here, real captured game frames plus the game's own boot overlay and brief.
- Keep all text readable in the final render.
- Keep the video within 15-25 seconds (target 21s).
- Include the planned music/SFX layer.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet. Choose SFX after
  the visual animation exists.
- Treat music cue metadata as optional timing hints; ignore cues that hurt
  readability, scene pacing, or the product story.
- Major reveals may move toward nearby strong cues within about 0.15s; smaller
  entrances may align to nearby beat points within about 0.10s. Use only 1-3
  strong cue locks.
- Honor the planned music treatment (fade under the final line, let the bell
  ring past it).
- Consider the Hyperframes audio-reactive workflow for a subtle RMS-driven glow
  on the gameplay frames and the amber accent. No visualizer graphics.
- Use local assets for audio, frames, and any runtime dependency where possible.
- Run `hyperframes check` before render — it is brag's single gate.
