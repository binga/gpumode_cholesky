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

## Ranked outcome — `#909488` / `#909492` (adopted)

Source `experiments/063-diag128-fast/ship-v1.py`, sha
`57da792a8c5b126fdfe9ff9495a03f1cd16dbe0f4aeaaf00ae1f12a3b9f4b3c2`.
Enrolled: `(16,512):512, (4,1024):1024, (8,2048):1024, (2,2048):1024,
(2,4096):1024`. Popcorn test passed twice on the exact source
(`#909471`, `#909487`).

| submission | public | secret |
|---|---:|---:|
| `#909488` | **682.912us** | — |
| `#909492` | 685.210us | **686.145us** |

Two ranked submissions of the identical source went out concurrently — one
from the orchestrator and one from the shape worker. That breached the
one-ranked-at-a-time rule and must not recur; it was harmless only because the
source was byte-identical. It did yield one useful number: two independent
leaderboard runs of the same file differ by **0.34%** (682.912 vs 685.210),
a clean run-to-run variance figure for this board.

### Campaign delta

| | `#907267` (start) | `#909269` | `#909488/92` |
|---|---:|---:|---:|
| public | 745.765us | 733.540us | **~683.4us** |
| secret | 741.378us | 721.821us | **686.145us** |

Public **-8.4%**, secret **-7.4%**.

### The paired grid under-predicted

The grid measured geomean **1.0289** (predicting ~713us) and the board
delivered ~683us, i.e. **-6.9% rather than -2.9%**. Every prior comparison in
exps 061-062 matched within ~0.1%, so this is the first material
under-prediction. Likely cause: the newly enrolled shapes shed *launch count*
as well as device time, and the grid's paired A-B-B-A interleave partly hides
dispatch-side wins that the real harness exposes. Re-check before using the
grid to size the remaining 40%-per-shape work.

## Six-family correctness — `results/063-ship-v1-familygrid.json`

`passed: false` but **`checker_ok: true` on all 48 rows**. This is the exp-061
gotcha: `familygrid` reports failure whenever *any* fallback fires. The
fallbacks are confined to the `spectrum` and `lowrank` families, which the
incumbent already falls back on — visible on the **unchanged** `640x512` row,
which shows `_FUSED_CTA_FALLBACKS: 1` on both families without any exp-063
change. Residuals on the fallback rows are tiny (0.0006-0.005). Correctness of
the shipped candidate is intact.

## Fused chain+inverse — measured, still rejected

`results/063-probe-v6.json` adds error localisation.

| variant | us/block | ns/row | chain | triinv | inv_err |
|---|---:|---:|---:|---:|---:|
| v0 shipped | 47.95 | 374.6 | 14.74 | 15.51 | 6e-08 |
| v1 fused | 42.40 | 331.3 | 19.54 | **0.00** | 0.573 |
| v2 fused+panel | 42.90 | 335.2 | 22.71 | **0.00** | 0.085 |

**The mechanism works but the accounting disappoints.** Folding the inverse in
does exactly what the plan predicted on those two phases — chain+triinv goes
30.25us -> 19.54us, a 10.7us saving. But the *block* only improves 47.95 ->
42.40us (5.55us), because the remaining phases all slow down
(`stageP+Qt` 4.32->5.79, `trailing+inv` 5.81->7.46, `load` 2.45->3.19,
`store` 1.55->2.11) — consistent with higher register pressure costing
occupancy and ILP. **Net upside of plan item 1 is ~11%, not the ~30% assumed.**
The plan's section 2 should be re-scoped accordingly.

**The bug is localised.** v1 first-bad indices are
`(9,8),(9,9),(10,8),(10,9),(10,10),(11,8)...`; v2's are
`(4,4),(5,4),(5,5),(6,4),(6,5),(6,6)...`. Both start on a multiple of 4 and
form a lower-triangular band from there, which points squarely at the
**4-way partial-sum split** in the inverse update (`s0..s3` with
`if (p+k < i)` guards) including or excluding one term at the wrong pivot
index. v2 halved the error (0.573 -> 0.085) without fixing it. Fix the guard
arithmetic against the known-correct round-1 version in
`experiments/062-midshape-2x/tail-v1.py` before spending more GPU time.

---

## Round 3 — the defect was memory ordering, not algebra

`results/063-probe-v7.json`. Two free local checks came first, and between them
they excluded every algebraic explanation, including the partial-sum-guard
hypothesis above:

- `sim_fused_chain.py` / `sim_panel32.py` replay the lane-and-register
  bookkeeping of both kernels element for element in float64. Reconstruction
  error 7.2e-16 and 5.3e-15, and `sim_panel32` additionally asserts
  `Qi @ L11 = I` for every panel. **The algorithms are exact as written.**
- `sim_fp32.py` replays the update in float32 both ways — the round-2 "stage
  raw, scale after, one `d2` multiply" form and the shipped "scale first,
  symmetric rank-1" form. Both give **2.469e-07**. The reassociation is
  innocent, so a 1.8e-2 result cannot be arithmetic at all.

That leaves memory ordering, and the fix confirms it. Rounds 1 and 2 both
stage values through shared memory *inside a loop*, which gives the optimizer
far more scope to hoist or cache a shared load than exp-062's straight-line
staging ever did. Two changes:

1. drop `__restrict__` from the staging pointer — it licenses the optimizer to
   assume nothing else writes that memory, but the other lanes do; and
2. wrap every `__syncwarp()` / `__syncthreads()` in `E62_CBAR()`, a pure
   compiler barrier (`asm volatile("" ::: "memory")`) that emits no
   instructions. The barriers order the hardware; this stops the *compiler*
   from moving a shared load across them.

| | v1 fused | v2 panel+inv | v3 panel L-only |
|---|---:|---:|---:|
| inv_err before | 0.573 | 0.085 | — |
| inv_err after | **2.4e-07** | **2.4e-07** | **6e-08** |

v1's 2.4e-07 is exactly round 1 of exp 062's known-good figure, which confirms
the fused Gauss-Jordan was correct all along.

**Reusable lesson:** `__syncwarp()` and `__syncthreads()` are hardware
barriers, not compiler barriers. Shared-memory staging inside a loop needs an
explicit compiler barrier as well. The failure is silent and correct on the
first loop iteration, which is exactly why the wrong indices looked like a
guard bug at a group boundary.

## Round 3 results — variant 3 wins

| variant | us/block | ns/row | chain | triinv | regs | inv_err |
|---|---:|---:|---:|---:|---:|---:|
| v0 shipped | 49.096 | 383.6 | 15.26 | 14.90 | 162 | 6e-08 |
| v1 fused chain (1 warp) | 41.131 | 321.3 | 19.23 | — | 167 | 2.4e-07 |
| v2 panel + fused inverse | 39.971 | 312.3 | 20.40 | — | 115 | 2.4e-07 |
| **v3 panel L-only + shipped inverse** | **37.904** | **296.1** | **10.76** | 12.84 | 159 | **6e-08** |

Variant 3 factors the whole 128x32 column panel with all 256 threads, which
subsumes the pivot chain *and* the panel solve (`panel` falls to 8 cycles), and
leaves the 32x32 triangular inverse on the shipped single-warp routine. It is
the fastest block and the only variant whose `abs_err` is identical to the
shipped kernel's on every shape, because the inverse path is untouched.

Whole-shape, against the exact shipped dispatch (`custom_kernel`):

| shape | shipped | v0 (= ranked `#909488`) | v2 | **v3** | v3 over v0 |
|---|---:|---:|---:|---:|---:|
| 16x512 | 409.0 | 300.7 (1.360x) | 275.2 | **270.9 (1.510x)** | +11.0% |
| 4x1024 | 718.5 | 575.9 (1.248x) | 523.8 | **517.4 (1.389x)** | +11.3% |
| 8x2048 | 1618.2 | 1367.5 (1.183x) | 1271.6 | **1248.8 (1.296x)** | +9.5% |
| 2x2048 | 1373.5 | 1149.7 (1.195x) | **1049.0** | 1065.5 (1.289x) | +7.9% |
| 2x4096 | 3219.1 | 2495.3 (1.290x) | 2294.9 | **2275.6 (1.415x)** | +9.7% |

### Register pressure was a red herring

The round-2 read that fusion's shortfall was register pressure does not survive
the counts: v2 uses **115** registers against v0's 162 and is still slower per
block than v3 at 159. Nor did the other phases actually slow down — their
*cycle* counts are flat across all four variants (`load`
8079/8061/8078/8067, `trailing+inv` 14880/14774/14913/14688, `store`
4047/4149/4086/4130). The apparent slowdown was an artifact of the phase table
reporting `net * cycles / total_cycles`: when the chain's share collapses,
every other phase's *share* rises even though its cycle count is unchanged.

**What actually caps the block is that the 128 dependent pivots cost 10-20us
however they are parallelised.** v0's one-warp shuffle chain is 15.26us, v1's
one-warp fused chain 19.23us, v2's 256-thread panel 20.40us — each pivot is a
full round trip through staging, barrier and reload, and extra warps buy
throughput the phase cannot use. Variant 3 is fastest (10.76us) only because
its panel phase also absorbs the panel solve.

So the plan's ~120 ns/row is not reachable by parallelising the serial phase;
~296 ns/row is about where that road ends. The remaining lever is plan item 2
(**overlap**): with the chain at ~11us and the parallel phases at ~14us, a
look-ahead schedule running them concurrently would approach `max` rather than
`sum`, i.e. ~25us/block (~195 ns/row) against today's 37.9us.

## Ranked outcome — `#912756` (adopted)

Source `ship-v2.py`, sha `6f602db3b3e7ced63918fe484c9ce7873ad1619589c4079aa5d061264a8e72d6`.
All six Popcorn runs passed.

| | public | secret |
|---|---:|---:|
| `#909488` | 682.912us | 686.145us |
| **`#912756`** | **675.753us** | **674.448us** |

-1.05% public, -1.70% secret.

### Campaign total

| | `#907267` (start) | `#912756` |
|---|---:|---:|
| public | 745.765us | **675.753us** (-9.4%) |
| secret | 741.378us | **674.448us** (-9.0%) |

### The launch-count hypothesis gains a second data point

`#912756` was predicted to *track* its paired grid rather than beat it, because
variant 3 changes device time only and leaves launch count untouched. Grid said
1.0321; the board delivered 1.011 public / 1.017 secret — close, and nothing
like the 2.4x under-prediction seen on the enrollment change, which *did* cut
launch count. Two points now support the reading that **the paired grid
systematically undervalues launch-count reductions and tracks pure device-time
changes accurately.** Worth one controlled experiment before it is relied on.
