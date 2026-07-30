# The champion build, and why the loop needs one

`git tag champion` marks the best-looking build produced so far, decided by
blind A/B. It is NOT the latest commit and must not be assumed to be.

## Why this file exists

Three rounds in a row were built on top of the previous round rather than on
the best round, and two of them regressed. Stacking a fix on a regression
compounds it. Every round from here is judged against the champion, not
against its immediate predecessor, and only a build that beats the champion
blind becomes the new champion.

## Results so far

| comparison   | winner | margin |
|--------------|--------|--------|
| r2  vs r1    | r2     | 5/5    |
| r3a vs r2    | r3a    | 3/5    |
| r3b vs r3a   | r3a    | 5/5 — r3b regressed |
| r3c vs r3a   | r3c    | 4/5    |
| r3d vs r3c   | r3c    | 5/5 — r3d regressed |

Champion: **r3c** (commit 92ab3b2, "Restore the road's material identity").

## The failure pattern to watch for

Every regression so far has been an overcorrection that satisfied its gate
while introducing the opposite defect:

- r3b was told to kill veiling glare. It lowered brightness without adding
  structure, so glare fell 0.267 -> 0.004 while flat-bright area stayed at
  0.339. It dimmed a wash.
- r3d was told to restore detail in the distance. It did — local contrast rose
  everywhere, genuinely — but it also crushed blacks to 3% pure-black pixels
  and posterized the clouds into hard-edged inkblots.

A gate proves the named defect moved. It does not prove the frame improved.
The blind A/B is the arbiter; the metrics only decide whether a build is worth
showing to a judge.

## Gates

    p1       0.002 - 0.025   low but NOT clipped
    crush    < 0.010         pure-black pixels; p1 alone rewards crushing
    range    > 0.75
    midBand  < 0.55          uniform mid-grey mush
    glare    < 0.06          bright and featureless
    lcon     > 0.075         local contrast; distinguishes detail from dimming
    flat     < 0.10          bright featureless tiles
    R/B      1.00 - 1.20     warm light on materials, not a sepia filter
