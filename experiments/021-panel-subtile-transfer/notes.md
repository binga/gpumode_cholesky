# Experiment 021 — panel-inner subtiling transfer

**Status: ADOPTED — ranked winner `#882958`.** Exact baseline is ranked
experiment 020 / submission `#882927` (`baseline-exp020.py`). Exact
ranked/adopted SHA-256:
`3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`.

## Candidate

`candidate-all-split32.py` extends experiment 020's exact 64x64 panel-inner
specialization to `64x256`, `16x512`, `640x512`, and `60x1024`. The existing
ranked routes at `4x1024` and `8x2048` remain unchanged.

## Gates

- Python compilation, whitespace, source-policy, and baseline snapshot checks.
- Alternating-order paired B200 timing on all four changed shapes.
- Six official input families per changed shape.
- Keep only individually positive routes, then run the full 15-shape grid.
- Popcorn test 17/17 before at most one leaderboard submission.

## Initial transfer probe

All four changed shapes passed six families each (24/24):

| shape | baseline | all-transfer candidate | speedup |
|---|---:|---:|---:|
| `64x256` | 234.8us | 222.8us | **1.054x** |
| `16x512` | 429.1us | 410.6us | **1.045x** |
| `640x512` | 1742.6us | 1551.1us | **1.123x** |
| `60x1024` | 1530.2us | 1450.9us | **1.055x** |

The first full grid passed 15/15 and improved `1166.8us -> 1149.3us`
(`1.0152x`), but `60x1024` reversed to `0.977x`. Because that route is known
to be noisy, `candidate-final.py` leaves it on the exact ranked baseline and
retains only the other three stable transfers.

## Final full grid

The selected candidate passed 15/15 and improved `1141.9us -> 1123.8us`
(**1.0160x**). The retained routes reproduced against the exact baseline:

| shape | baseline | final candidate | speedup |
|---|---:|---:|---:|
| `64x256` | 238.6us | 227.8us | **1.047x** |
| `16x512` | 424.5us | 393.9us | **1.078x** |
| `640x512` | 1740.7us | 1542.8us | **1.128x** |

`60x1024` remained on the exact baseline and measured `1.001x`. The largest
off-target change was the noisy unchanged `4x1024` route at `0.992x`; all other
untouched paths were approximately flat.

## Popcorn and ranked result

Popcorn test `#882957` passed **17/17**. Exactly one leaderboard submission,
`#882958`, passed every public and secret stage at
**1096.0842452192236us public / 1109.6451814508845us secret**. This improves
`#882927` by **2.1540% public / 1.4930% secret**. The exact ranked source is
`submission.py` and is adopted at repository root.

Artifacts: `probe-all-four.json`, `fullgrid.json`, `fullgrid-final.json`,
`test-882957.json`, and `ranked-882958.json`.
