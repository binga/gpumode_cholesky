# Experiment 032 — panel-width schedules (lever L2): RESULT

Baseline: ranked `#883174`. Measured 2026-07-18 on B200 via the new
`schedprobe` paired same-process harness (`scripts/_gpu_runner.py`), drift <0.9%
on every row. Raw JSON in `../../results/exp032-schedprobe-*.json`.

## Verdict: panel width is a live axis at ONE shape (8×2048). Banked, not solo-shippable.

### Variant A — tail taper (`(128,)*(k)+(64,32,32)`): REFUTED on all shapes
| shape | baseline µs | cand µs | speedup |
|---|---|---|---|
| 256×128 | 137.4 | 148.7 | **0.925×** |
| 64×256 | 221.7 | 223.0 | 0.994× |
| 16×512 | 382.6 | 389.8 | 0.982× |
| 640×512 | 1537.3 | 1567.2 | **0.981×** (kill) |
| 4×1024 | 720.6 | 721.6 | 0.999× |
| 60×1024 | 1470.8 | 1468.8 | 1.001× (noise) |
| 8×2048 | 1799.1 | 1802.6 | 0.998× (kill) |

Enrolled geomean 0.9824×. Both kill-criterion shapes lost. Every added panel pays
the ~16µs serial-tile-loop launch floor (S27/S29) while its tapered trailing
corner processes almost no data. Taper is the wrong direction.

### Variant W — wide uniform NB=256 (`(256,)*k`): spills, wins only at 8×2048
| shape | baseline µs | cand µs | speedup | note |
|---|---|---|---|---|
| 64×256 | 224.4 | 232.5 | 0.965× | |
| 16×512 | 389.2 | 390.8 | 0.996× | |
| 640×512 (eager) | 1570.5 | 1876.8 | **0.837×** | spill |
| 4×1024 | 726.7 | 724.0 | 1.004× | flat |
| 60×1024 (eager) | 1800.5 | 6293.8 | **0.286×** | catastrophic spill |
| **8×2048 (graph)** | 1814.8 | 1759.6 | **1.031×** | **WIN** |

NB=256 doubles `_trailing_nb`'s `[TILE×NB]` tile → register spill. Catastrophic on
the two eager-mode shapes (no graph amortization). 8×2048 is the sole winner: most
panels (16→8, half the launches) and enough per-panel tensor-core compute to hide
the spill.

### Variants X/X2 — NB=512 on 8×2048: overshoot
`(512,512,512,512)` = 0.972×; `(512,512,256,256,256,256)` = 0.983×. Spill grows
faster than the launch saving past 256. **NB=256 is the sweet spot.**

## Banked change (root submission.py)
`_SPLIT32_NB_SCHEDULE = {(8,2048): (256,)*8}`. Off-target shapes keep an empty
schedule → byte-identical launches to #883174.

## Why NOT a solo LB submission
1.031× on one shape = `1.031^(1/15)` = **+0.20% geomean**, below the leaderboard's
~1-2% run-to-run noise. Two separately-launched full 15-grids (same-day, different
B200 sandboxes) differed **3.6%** in geomean on byte-identical off-target code —
`4096×32` alone swung 15% between sandboxes. Inter-sandbox variance swamps a 0.2%
signal; only paired same-process measurement is trustworthy at this scale
(`results/exp032-grid-finalist.json` vs `exp032-grid-baseline.json`). Plan: keep
8×2048 banked, stack a bigger lever, submit ONE combined ranked entry.

## Reusable harness added
- `schedprobe` mode + `_load_sched_baseline_module` in `scripts/_gpu_runner.py`;
  `baseline_sched.py` wired into `scripts/modal_verify.py::_build_image`.
- `--combined VARIANT` in `make_candidates.py`; `baseline-scaffold.py` snapshot.
- Follow-up (untried): non-power-of-two widths need a padded+masked load in
  `_trailing_nb`; and 640×512/60×1024 might tolerate wide panels if moved from
  eager to graph mode — separate experiments.
