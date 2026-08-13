# Brag Plan: Claude of Duty (hype cut, narrated)

## What is this app?
A browser FPS — a real one, with six weapons, an enemy squad and a three-lane
Mediterranean map — built in 38,926 lines of TypeScript and three.js with zero
image files, playable from a single tab.

## The angle
The first cut was a deadpan spec-sheet film. This one is the opposite: a 2010s
console-shooter trailer, the kind cut to hook someone who has never heard of the
game. Deep announcer, hard cuts on the downbeat, ALL-CAPS callouts, one boast
per scene, escalating to a title slam and a "get in now" line.

Everything it brags about is real and pulled from the source: `Dust Corridor`
and its three lanes come from `src/world/Level.ts`; the six weapons and their
per-weapon recoil patterns from `src/weapons/WeaponSpecs.ts`; the enemy squad's
hunt-and-take-cover behaviour from the `AISystem` FSM in `src/ai/AI.ts` (which
picks cover that actually breaks the player's line, not merely the nearest);
and the 1,039,810 triangles / 202 textures / 0 image files were measured off
`renderer.info` in the running game.

The braggadocio is the register, not an exaggeration. Nothing in the script is
a claim the repo can't back.

## Hook (first 2-3 seconds)
No boot screen this time — a hype trailer opens in the world. Hard in on the
square with the bell tower, fast push, and the announcer's first four words over
it: *"This is Dust Corridor."* The map has a name and the voice says it like it
matters.

## Key moments (the middle)
- **Three lanes. One square. Nowhere to hide.** — the map sold as a place, with
  the three claims snapping in on the beat over the fountain.
- **Six weapons** — the full roster from `WeaponSpecs.ts` ticking in two columns
  on consecutive beats: MK4 RANGER, VKS-9 WASP, M16-BR, KV-800 BALLISTA,
  M870 BREACHER, P226 SIDEARM.
- **They hunt you** — the enemy squad, with the real mechanism in small type:
  perception cone, line of sight, A* pathfinding.
- **The numbers** — 1,039,810 TRIANGLES / 202 TEXTURES / 0 IMAGE FILES, each
  landing on a strong beat.

## Outro / punchline
Title slam on the biggest downbeat in the track, then the closer that converts:
**No download. No install. One tab.**

## User flow worth showing
Entry → key action → result, all real captures from the running game:
1. **Entry** — the square, weapon up, HUD live the moment the scene cuts in.
2. **Key action** — the arsenal and the enemy squad: what you actually do here.
3. **Result** — the frame itself, sold on its own numbers, then the title.

## Tone
- Preset: `chaotic` (used for pacing and density: 6 scenes, hard cuts, dense SFX)
- Creative direction: *incredibly braggadocios gameplay trailer from the 2010s
  meant to hook new players* — with an epic announcer.
- Interpretation: six scenes, every cut on a downbeat, one boast per scene,
  ALL-CAPS callouts that snap rather than fade, dense impact SFX. But the
  narration sets the pace, so holds are honest: no line is pulled before the
  announcer finishes it. Chaotic energy, not chaotic legibility.

## Format: landscape — 1920x1080
## Duration: 24.5 seconds

## Visual identity (from the project)
- Background: `#0b0d0e`, deepest ink `#05070a`
- Accent: `#e2a53d` (in-game UI amber)
- Text: `#f3f1ea`, dim `rgba(243,241,234,.62)`
- Display font: the game's condensed stack (falls back to Liberation Sans here,
  as it does in the game on this machine), heavy weights, `scaleX(.9)`, caps
- Data font: `ui-monospace` / DejaVu Sans Mono, tabular numerals
- Strongest visual element: the captured gameplay frames themselves — bell
  tower, fountain, arcade, plaza — with the live HUD in every one

## Share copy (draft)
Six weapons. Three lanes. An enemy squad that actually takes cover. Claude of
Duty runs in a browser tab — no download, no install, and not one image file in
the whole game.

## Audio direction
- Role: dense rhythmic layer under a narrated trailer — the bed drives the cuts,
  the announcer carries the meaning.
- Music: `happy-beats-business-moves-vol-1-by-ende-dot-app.mp3` (most energetic
  of the bundled tracks, 120.19 BPM).
- Music treatment: baseline 0.32, ducked to 0.15 under every narration line and
  lifted back in the gaps, one swell between line 1 and line 2, full level for
  the last ring-out, then out.
- Music cue guidance: bundled preset read. At 120 BPM the grid is a clean 0.50s
  and the 16-23s window is wall-to-wall 1.00-intensity strong beats, so this cut
  locks more than the usual 1-3: **every scene cut** sits on a beat (3.52, 7.52,
  11.52, 16.02, 20.02) and the three stat lines land on three consecutive strong
  cues (16.52 / 17.52 / 18.52). That is the point of the genre — a 2010s trailer
  edit *is* the beat grid. Readability is still protected by the narration
  timing, which never gets compressed to fit a beat.
- Audio-reactive treatment: subtle; amber horizon warmth in the gameplay frames
  breathes with music RMS and bass. No waveform or equalizer visuals.
- SFX posture: dense, per the tone — a heavy punch on each hard cut, a click per
  weapon name, a drop per stat line, one deep bell on the title slam.
- Audio-coupled moments: every scene cut; the six-name weapon sequence; the
  three stat reveals; the title slam.
- Restraint rule: nothing may collide with the announcer. SFX are short and land
  in the transient before a line starts, never mid-sentence.

## Voiceover script
Kokoro voice `am_onyx` at speed 0.9 — chosen by generating the closing line in
six candidate male voices and measuring median F0: am_onyx came in at 86.6 Hz
against 123-157 Hz for the rest, which is the announcer register.

| # | Line | Duration | Placed at |
|---|------|----------|-----------|
| 1 | This is Dust Corridor. | 2.18s | 0.50s |
| 2 | Three lanes. One square. Nowhere to hide. | 3.20s | 3.80s |
| 3 | Six weapons. Every one with its own recoil. | 3.41s | 7.80s |
| 4 | An enemy squad that hunts you and takes cover that actually works. | 4.31s | 11.70s |
| 5 | A million triangles. Not one image file. | 3.41s | 16.30s |
| 6 | Claude of Duty. No download. One tab. | 3.16s | 20.45s |

Scene durations were set from these timings, not the other way round.

## Storyboard

### Scene 1 — This is Dust Corridor — 3.52s
Hard in on the captured bell-tower frame, fast push 1.10 → 1.00. Amber eyebrow
`TASK FORCE` and a huge `DUST CORRIDOR` snap in low-left. Announcer line 1 over
it. No fade in — the trailer starts already moving.
Sequential/interaction: none — one headline snap.
Audio intent: the bed arrives with the picture; nothing soft.
Audio-coupled idea: none at t=0; let the music's own entry carry it.
Music: full energy from the first frame.
Transition mood: hard cut on the beat at 3.52 → Scene 2

### Scene 2 — Three lanes — 4.00s
The fountain frame. Three claims stack in on beats: `THREE LANES.`,
`ONE SQUARE.`, `NOWHERE TO HIDE.` Each snaps (no fade), tracking the announcer.
All three hold together for the last ~1.4s.
Sequential/interaction: yes — three lines arriving one per beat-pair (4.02,
5.02, 6.03), then held as a set.
Audio intent: the boast opens up.
Audio-coupled idea: heavy punch on the cut into the scene.
Music: full.
Transition mood: hard cut on the beat at 7.52 → Scene 3

### Scene 3 — Six weapons — 4.00s
The plaza frame, darkened on the left. `SIX WEAPONS` headline, then the roster
ticks in as two columns of three on consecutive beats — MK4 RANGER, VKS-9 WASP,
M16-BR, KV-800 BALLISTA, M870 BREACHER, P226 SIDEARM — with a click on each.
The full set holds for 1.0s.
Sequential/interaction: yes — six names, beat-grid at 8.02 / 8.52 / 9.02 / 9.52
/ 10.02 / 10.52, each with its own click. Individually they are short proper
nouns and the set holds afterward, which is how a fast list stays readable.
Audio intent: rhythmic, mechanical, satisfying.
Audio-coupled idea: one click per name, exactly on the name's own frame.
Music: full.
Transition mood: hard cut on the beat at 11.52 → Scene 4

### Scene 4 — They hunt you — 4.50s
The arcade frame under the archway. `THEY HUNT YOU.` lands, then
`AND TAKE REAL COVER.`, then a small monospace line naming the actual mechanism:
`perception cone · line of sight · A* pathfinding`. Announcer line 4 runs the
whole scene and the cut lands the instant it finishes.
Sequential/interaction: yes — two headline snaps (11.85, 14.02) and one mono
detail (15.02), each held to the cut.
Audio intent: menace under the boast.
Audio-coupled idea: heavy punch on the cut in; nothing during the line.
Music: full.
Transition mood: hard cut on the strong beat at 16.02 → Scene 5

### Scene 5 — The numbers — 4.00s
The long-shadow street frame, darkened. Three numbers land on three consecutive
1.00-intensity strong cues: `1,039,810 TRIANGLES` (16.52), `202 TEXTURES`
(17.52), `0 IMAGE FILES` (18.52), the last in amber. All three hold to the cut.
Sequential/interaction: yes — three stat lines, one per strong cue, each with a
soft drop.
Audio intent: the flex. Weight on each number.
Audio-coupled idea: a drop cue on each stat, landing with the number.
Music: full, into the biggest downbeat of the window.
Transition mood: hard cut on the strong cue at 20.02 → Scene 6

### Scene 6 — Title — 4.48s
Back to the fountain, darkened. `CLAUDE OF DUTY` slams 1.12 → 1.00 on the 20.02
strong cue with a single sheen sweep and a deep bell. Amber eyebrow rule and
`Task Force // Dust Corridor` at 21.01, then the closer
`NO DOWNLOAD. NO INSTALL. ONE TAB.` at 22.01. Announcer line 6 over it, and the
title holds alone for the last second as the music rings out.
Sequential/interaction: yes — slam, eyebrow, closer, each on a strong cue.
Audio intent: payoff, then a clean ring-out.
Audio-coupled idea: deep bell on the slam, locked to 20.02; nothing after.
Music: back to full for the ring-out, then out across the last 0.6s.
Transition mood: end.

**Music mood for this video:** chaotic
**Audio summary:** A 120 BPM bed runs wall-to-wall and every cut lands on its
grid, ducking under each of the six announcer lines and lifting in the gaps,
with heavy punches on the cuts, a click per weapon, a drop per stat, and one
deep bell on the title slam before the bed rings out.
