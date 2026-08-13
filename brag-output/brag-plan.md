# Brag Plan: Claude of Duty

## What is this app?
A browser FPS built in 38,926 lines of TypeScript and three.js with **zero image
files** — every texture, weapon, building and cloud in it is generated at runtime
by code — and it was tuned by shooting screenshots of itself and judging them
blind against real Call of Duty frames.

## The angle
Play it completely straight as a AAA game trailer, because it genuinely is a
game — the boot screen, the deployment brief, the compass, the killfeed, the
MK4 Ranger's 30/210 are all real, all shipping. The brag is not "look, a demo."
The brag is the spec sheet underneath it: thirty-nine thousand lines, no art
pipeline, one browser tab. And the punchline is the QA loop — an agent that
screenshotted its own game, compared it blind to Call of Duty, and lost to
itself twice before it won.

Specific to this project and no other: the trailer's stats are the repo's stats,
and its ladder rows are copied out of `qa/CHAMPION.md`.

## Hook (first 2-3 seconds)
The real boot screen. Black, then the amber 2px bar filling under letterspaced
**CLAUDE OF DUTY**, with the actual loading label ticking `building level`. It is
the first thing anyone who opens this project sees, and it reads as a game
booting — which earns the next twenty seconds before a single claim is made.

## Key moments (the middle)
- A real captured gameplay frame — Mediterranean town square, sun-raked shadows
  across the road, MK4 Ranger viewmodel with the optic on the camera axis, live
  HUD: compass, `GRID K10` minimap, `MK4 RANGER 30/210`. Not a mockup. A frame
  from the running game.
- The deployment brief from the main menu, arriving row by row:
  `THEATRE / DUST CORRIDOR`, `CONTRACT / SEARCH & DESTROY`, `THREAT / VETERAN`.
- The spec sheet over gameplay: **38,926 LINES OF TYPESCRIPT / ZERO IMAGE FILES /
  ONE BROWSER TAB.**
- The champion ladder out of `qa/CHAMPION.md` — `r3b vs r3a → REGRESSED`,
  `r3c vs r3a → CHAMPION` — under the line that it was judged blind against a
  real Call of Duty frame.

## Outro / punchline
The title slams full-screen over the darkened square, `Task Force // Dust
Corridor` under it, and the last line lands dry: **No textures. No art team.
One tab.**

## User flow worth showing
Entry → key action → result, and all three are real captures:
1. **Entry** — the boot overlay fills and the square resolves out of black.
2. **Key action** — the player is in the street, weapon up, HUD live, ammo
   counter reading 30/210.
3. **Result** — the QA harness's own verdict: the champion ladder that decided
   which build shipped.

## Tone
- Preset: `cinematic`
- Creative direction: a straight-faced AAA console trailer for something that is
  actually 37 TypeScript files and no images.
- Interpretation: big letterspaced caps, full-bleed frames, slow pushes rather
  than quick cuts, 4-5 scenes with real holds. The humour comes entirely from
  playing it seriously — never wink at it, never speed it up for effect. The
  spec sheet does the joke on its own.

## Format: landscape — 1920x1080
## Duration: 21 seconds

## Visual identity (from the project)
- Background: `#0b0d0e` (boot / page), deepest ink `#05070a`
- Accent: `#c8a44a` (boot bar) and `#e2a53d` (in-game UI amber) — use `#e2a53d`
  for UI-derived elements, `#c8a44a` for the boot moment
- Text: `#f3f1ea`, dimmed `rgba(243,241,234,.62)`, faint `rgba(243,241,234,.30)`
- Danger red (sparing, for `REGRESSED`): `#e0413a`
- Display font: the game's own condensed stack — Roboto Condensed / Barlow
  Condensed / Archivo Narrow / Arial Narrow, heavy weights, `scaleX(.93)`,
  wide tracking, uppercase
- Body / data font: `ui-monospace` — the game uses it for every number and key
- Strongest visual element: the captured gameplay frame itself, plus the game's
  hairline-rule-and-amber-eyebrow UI language (1px rules, `.4em` tracking, hard
  black shadow under every stroke)

## Share copy (draft)
Built a Call of Duty in a browser tab. 38,926 lines of TypeScript, zero image
files — every texture is generated at runtime — and it graded its own frames
blind against the real thing.

## Audio direction
- Role: cinematic support — a low steady bed that carries the trailer and gets
  out of the way of the reads.
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (steady, clean,
  109.96 BPM) at 0.30-0.35, fading under the final title.
- Music treatment: start at 0, hold level through the body, duck slightly on the
  outro slam so the last hit rings.
- Music cue guidance: bundled preset read from
  `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json`.
  Strong cues to target: **8.74s** (spec sheet lands), **13.11s** (ladder scene
  opens), **17.47s or 18.56s** (title slam). Beat grid for sequential rows:
  brief rows near 4.39 / 5.34 / 6.00; spec rows near 9.29 / 10.37 / 11.46;
  ladder rows near 14.20 / 15.29.
- Audio-reactive treatment: subtle — let the sky/horizon glow in the gameplay
  frames and the amber accent presence breathe with RMS. No waveform or
  equalizer graphics, no strobing.
- SFX posture: 2-3 big ones, cinematic restraint. A deep bell on the title slam,
  a soft impact on the spec-sheet reveal, one dry accent on `CHAMPION`.
- Audio-coupled moments: the boot bar filling; the brief rows arriving one by
  one; the spec rows landing; the title slam.
- Restraint rule: no SFX on every row, nothing comedic, nothing that undercuts
  the straight face. If the edit already feels busy, drop a cue rather than add.

## Storyboard

### Scene 1 — Boot — 3.0s
Pure `#0b0d0e`. Letterspaced **CLAUDE OF DUTY** at `.40em` tracking, thin weight,
`#e8e4dc` — the boot overlay's exact treatment. A 260x2px track under it fills
left-to-right in amber `#c8a44a`. Under that, the real loading label in 11px
uppercase grey: `starting` → `building level` → `ready`. Nothing else on screen.
Sequential/interaction: yes — the bar fills 0→100% over ~2.0s and the label
swaps twice underneath it, exactly as the game does it.
Audio intent: a single low bed starting from nothing; anticipation, no payoff yet.
Audio-coupled idea: none on the label swaps — let the bar fill silently under
the music. One very soft cue as the bar completes.
Music: steady low bed, entering from silence.
Transition mood: hard cut → Scene 2

### Scene 2 — Deployment — 4.2s
Hard cut to the real captured gameplay frame (`hero`): the town square, sun
raking across the road, MK4 Ranger viewmodel bottom-right, live HUD. Slow push
from 1.04 to 1.00 across the whole scene. On the right third, the menu's
Deployment brief arrives row by row in the game's own type: amber `.38em` eyebrow
**DEPLOYMENT** over a hairline rule, then `THEATRE — DUST CORRIDOR`,
`CONTRACT — SEARCH & DESTROY`, `THREAT — VETERAN`, each row a hairline-separated
key/value pair. Hold all three settled for at least 1.0s.
Sequential/interaction: yes — three brief rows arrive one by one, ~0.55s apart,
snapped to the beat grid, then the full set holds.
Audio intent: the bed opens up; the world arrives.
Audio-coupled idea: a soft dry accent on the first row only, not all three.
Music: bed continues, first real energy.
Transition mood: dramatic wipe → Scene 3

### Scene 3 — The spec sheet — 4.8s
A second real gameplay frame, darkened and slightly desaturated so type reads
over it. Three lines land in sequence, huge, condensed, uppercase, centred-left:
**38,926 LINES OF TYPESCRIPT** / **ZERO IMAGE FILES** / **ONE BROWSER TAB.**
Under the second line, small and amber: `every texture generated at runtime`.
Each line holds at least 0.9s settled; all three are on screen together at the
end of the scene.
Sequential/interaction: yes — three stat lines, beat-grid spaced ~1.05s apart
(every other beat, so each line clears the reading floor), then held as a set.
Audio intent: the claim scene. Weight under each line.
Audio-coupled idea: one soft impact as the first line lands; the reveal cue at
the strong 8.74s cue. Nothing on lines two and three.
Music: bed at full, strongest section.
Transition mood: slow crossfade with scale → Scene 4

### Scene 4 — Judged blind — 4.6s
Near-black, the gameplay frame reduced to a dim ghost behind. One line, stated
seriously: **JUDGED BLIND AGAINST THE REAL THING.** Under it, two monospace rows
in the game's data language, hairline-separated:
`r3b  vs  r3a` → `REGRESSED` in danger red `#e0413a`;
`r3c  vs  r3a` → `CHAMPION` in amber `#e2a53d`.
Both rows verbatim from `qa/CHAMPION.md`. The headline holds ~1.6s before the
first row arrives; each row holds ~1.0s settled.
Sequential/interaction: yes — two ladder rows arrive one after the other,
~1.1s apart, snapped to alternate beats so the text is readable.
Audio intent: dry, factual, a little ominous — the joke is the straight face.
Audio-coupled idea: one dry accent on `CHAMPION` only.
Music: bed pulls back slightly under the reads.
Transition mood: hard cut → Scene 5

### Scene 5 — Title — 4.4s
Full-bleed darkened square. **CLAUDE OF DUTY** slams in at scale 1.06 → 1.00
with the game's own sheen sweeping across the letterforms once. Under it, the
amber eyebrow rule and `Task Force // Dust Corridor`. Beat, then the last line
dry and small: **No textures. No art team. One tab.** Hold on the title for the
final second as the music fades.
Sequential/interaction: yes — title slam, then eyebrow, then the closing line.
Audio intent: payoff. The bell rings over a fading bed.
Audio-coupled idea: deep bell on the title slam, locked to the strong cue at
17.47s or 18.56s; nothing after it.
Music: fade out under the final line, let the bell ring past it.
Transition mood: end.

**Music mood for this video:** cinematic
**Audio summary:** A low steady bed enters from black under the boot bar, opens
up as the square appears, carries the spec sheet at full weight, pulls back for
the blind-judgement rows, and fades out under a single deep bell on the title
slam.
