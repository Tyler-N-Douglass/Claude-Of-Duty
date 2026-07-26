# Visual QA Rubric — "would this pass as a Call of Duty frame?"

This is the standard every critic agent judges against. It exists because
"looks good" is not actionable and "looks bad" is not fixable. Every failure
must name the frame, the region, the cause, and the file that owns it.

## The test

Look at the screenshot. Then recall a still frame from a recent Call of Duty
(MW2019 / MWII / MWIII / BO6) — a mid-day or golden-hour infantry engagement in
a built-up area, which is what this map is. Ask, in order:

1. **The two-second test.** If both frames were shown side by side with no
   labels for two seconds, could you tell which was the real game? If yes, you
   must be able to say *what gave it away*. That giveaway is the finding.
2. **The squint test.** Blur your reading of the image down to values and
   masses. Does the frame have a clear focal hierarchy — a bright zone, a dark
   zone, and a readable silhouette structure? Or is it a uniform mid-grey mush?
   Uniform mush is the single most common failure in engine screenshots.
3. **The crop test.** Take any 200×200 region of the frame. Does it still hold
   up? Real game frames survive cropping because detail exists at every scale.

## Hard fails — any one of these means the round is not done

- Any flat untextured surface, anywhere.
- Visible tiling: the same noise pattern recognisably repeating.
- Sharp unbeveled 90° edges on man-made objects catching a hard specular line.
- Shadows that swim, shimmer, or detach from their caster (peter-panning).
- Aliased edges — jaggies on any silhouette against the sky.
- The horizon meeting the ground with no atmospheric haze.
- Uniform lighting with no directional read; everything equally lit.
- Blown-out white or crushed-black regions with no detail.
- The viewmodel clipping into world geometry, or lit differently from the world.
- HUD elements that look like unstyled browser DOM.
- Z-fighting anywhere.
- A skybox that reads as a gradient rather than an atmosphere.

## Scored dimensions (1–10 each; below 8 is a finding)

**Lighting & exposure** — Directional read with a clear key. Contact shadows
that harden near the contact point. Bounce/fill that is colour-tinted by what it
bounced off, not grey. Exposure that puts skin/concrete in the upper-mid range
and keeps sky detail. Specular response that varies across a surface.

**Materials** — Every surface tells you what it is made of and what has happened
to it. Roughness varies spatially and correlates with the story: polished where
touched, rough where dust settles. Metal reads as metal (dark diffuse, bright
varied specular), not grey plastic. Albedo stays in 0.03–0.85.

**Geometry & silhouette** — Beveled edges everywhere. Depth in window reveals.
Asymmetry and irregularity. Nothing perfectly straight or perfectly repeated.
Instanced props visibly varied in scale and rotation.

**Atmosphere & depth** — Aerial perspective: distant geometry desaturates and
lifts toward the sky colour. Fog that is height-based, not a uniform tint. Light
shafts where the sun rakes through openings. A sense of air in the volume.

**Post & finish** — Anti-aliased edges. Bloom with a tight core and wide skirt,
not a uniform glow. Filmic tonemapping with an S-curve, not linear clipping.
Subtle grain, subtle vignette, subtle chromatic aberration at the edges. Slight
sharpening. Motion blur under motion.

**Composition & art direction** — The frame is *designed*: a focal point,
leading lines, value contrast placing the subject. Colour is a deliberate,
limited palette, not whatever the materials happened to be.

**Viewmodel** — The weapon is the best-crafted object on screen. Correct
positioning. Materials that respond to the world's light. Beveled everything.
An optic that reads as glass.

**HUD** — Restrained, legible over any background, tightly kerned, one accent
colour, thin strokes, correct optical alignment. Reads as shipped UI.

## How to report

For each finding:

    [dimension] [frame name] — <what is wrong, specifically, and where in frame>
    Cause: <the actual technical reason>
    Owner: <src/path/File.ts>
    Fix: <concrete change, not "improve the lighting">

Rank by how much the fix moves the two-second test. A single wrong exposure
value outranks ten missing props.

## What not to do

Do not praise. Do not soften. Do not say "this is impressive for a browser" —
the browser is not a handicap the player can see. Do not accept "good enough".
Do not report a finding you cannot point at in the image. If the frame genuinely
passes a dimension, say so in three words and move on.
