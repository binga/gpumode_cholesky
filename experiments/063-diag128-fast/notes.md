# Experiment 063 — faster resident diagonal block, and wider enrollment

Baseline: ranked `#909269` (public 733.540us / secret 721.821us), source sha
`f408a020ea94…`, which is the repo root `submission.py` at commit `21c50f9`.

## Result 1 (SHIPPABLE) — enroll three more shapes on the existing kernel

`results/063-enroll-v1-pairedgrid.json`, full 15-shape paired grid against the
exact `#909269` incumbent:

| shape | ratio | note |
|---|---:|---|
| 16x512 | **1.1703x** | newly enrolled |
| 4x1024 | **1.1492x** | newly enrolled |
| 8x2048 | **1.1344x** | newly enrolled |
| 2x2048 | 1.0009 | already enrolled, flat as expected |
| 2x4096 | 1.0014 | already enrolled, flat as expected |
| other 10 | 0.9992-1.0014 | flat |

**geomean 1.0289**, CI95 [1.0281, 1.0296], `all_shapes_ok: true`,
`new_fallbacks: {}`, every expected counter present on every shape.

The existing `e62_diag128` block kernel is *already* worth 1.13-1.17x on the
three split32 mid shapes, because it collapses 7 launches per 128-block
(4x micro potrf + 4x panel apply + 3x panel inner) into one. No kernel change
was needed for this — only dispatch enrollment.

Per-shape probe walls (`results/063-probe-v5.json`, v0 = shipped kernel):

| shape | shipped | v0 | ratio |
|---|---:|---:|---:|
| 16x512 | 412.2 | 303.1 | 1.3601 |
| 4x1024 | 721.1 | 580.9 | 1.2413 |
| 8x2048 | 1616.4 | 1374.9 | 1.1756 |
| 2x2048 | 1383.7 | 1195.1 | 1.1578 |
| 2x4096 | 3221.4 | 2501.4 | 1.2879 |

(The subset probe overstates versus the full grid, exactly as exp 050 warned —
1.36x at 16x512 in the probe became 1.17x on the grid. Trust the grid.)

## Result 2 (REJECTED, numerically) — fused chain+inverse

Plan item 1 (fold the triangular inverse back into the pivot chain) **is
faster and is wrong as written**:

| variant | us/block | ns/row | block abs_err | inv_err | shapes |
|---|---:|---:|---:|---:|---|
| v0 shipped | 48.902 | 382.0 | 4e-07 | 6e-08 | ok |
| v1 fused | **41.710** | **325.9** | **0.049** | **0.573** | **NaN everywhere** |

Phase split confirms the mechanism works — `triinv` goes 14.918 -> **0.0us**
and `chain` only grows 15.220 -> 18.682 — so the fusion saves a net ~11.5us of
a 48.9us block (a real 15%). It is purely a correctness bug in the fused
Gauss-Jordan update, not a dead design. The round-1 version in
`experiments/062-midshape-2x/tail-v1.py` was numerically correct
(inverse error 2.4e-07), so diffing v1's inner loop against that is the way in.

**Do not ship v1 until `inv_err < 1e-6` and whole-shape `abs_err` is finite.**

## Status

Preserved from an interrupted run. Result 1 has passed the authoritative gate
and is ready for Popcorn `--mode test` + one ranked submission. Result 2 needs
the correctness bug fixed before it is worth anything.
