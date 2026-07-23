# Experiment 050 notes — fused 128x128 diagonal block

Goal (user): **2.00x aggregate geomean** against ranked `#890798`
(801.977us public / 847.836us secret), i.e. ~401us public.
Exact control snapshot: `baseline-890798.py`, SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.

## Fresh full-grid B200 constituent profile of the exact incumbent

Artifact: `../../results/inc-890798-shapediag.json` (local geomean 846.6us).

| shape | wall us | device us | dominant constituent |
|---|---:|---:|---|
| 4096x32 | 16.1 | 14.1 | `chol32_register_rank2` 14.1 |
| 1024x64 | 25.3 | 22.7 | `cholesky64_rank2` 22.7 |
| 256x128 | 62.2 | 60.6 | `cholesky128_block16` 60.6 |
| 64x256 | 99.2 | 97.1 | `cholesky256_wmma16` 97.1 |
| 16x512 | 402.4 | 365.9 | Triton micro potrf **210.0 (16 calls)** |
| 4x1024 | 706.7 | 670.7 | Triton micro potrf **419.2 (32 calls)** |
| 60x1024 | 1221.0 | 1006.4 | trailing 374.9, CUDA micro 303.1; **215us idle** |
| 640x512 | 1480.7 | 1128.9 | fused panel 361.5, trailing 195.0; **352us idle** |
| 2x2048 | 1353.9 | 1342.6 | **cuSOLVER `getrf_wo_pivot` 1223.5 (2 calls, 91%)** |
| 8x2048 | 1593.5 | 1566.2 | Triton micro potrf **841.9 (64 calls)** |
| 1x4096 | 1527.2 | 1528.7 | **cuSOLVER `getrf_wo_pivot` 1391.8 (1 call, 91%)** |
| 2x4096 | 3199.2 | 3187.8 | **cuSOLVER `getrf_wo_pivot` 2774.6 (2 calls, 87%)** |
| 1x8192 | 5794.0 | 5402.9 | cuSOLVER getrf 2526.0, cuSOLVER trsm 1360.5 |
| 1x16384 | 15241.0 | 14536.4 | cuSOLVER getrf 5054.0, cuSOLVER trsm 4496.8 |
| 1x32768 | 42431.5 | 41173.4 | cuSOLVER getrf 11185.8, cuSOLVER trsm 8997.2 |

Three facts drive this experiment:

1. **cuSOLVER still owns `2x2048`, `1x4096` and `2x4096` outright** (87-91% of
   device time in one `getrf_wo_pivot` kernel), and it factors each matrix
   *serially*: 611.8us per 2048 and 1387.3us per 4096 whether the batch is 1 or
   2. The split32 route is already 3x faster per matrix at 2048 (`8x2048` runs
   8 matrices in 1593.5us).
2. **The Triton 32x32 diagonal micro is the mid-shape bottleneck** — 57%, 62%
   and 54% of device time at `16x512`, `4x1024`, `8x2048`, at a flat
   13.1us/call that is independent of batch. Exp 044 already built a CUDA
   replacement measuring 10.26us/call, and full-grid 1.1910x / 1.2214x /
   1.1857x on exactly these three shapes, but could not ship it: a raw
   `<<<grid, block>>>` launch cannot enter a CUDA graph, and naming the current
   work queue is refused by popcorn's source policy.
3. Ratio to hardware floor is 81-143x on the mid shapes against 2.2x at
   `1x32768`. The geometric mean weights all 15 shapes equally, so the mid
   shapes are where the score is.

## The lever

`diag128_potrf`: one CUDA CTA (8 warps, 70.8 KB shared) factors an entire
128x128 diagonal block and publishes its four 32x32 triangular inverses. The
shipped schedule spends **seven launches** on that same block (4x micro potrf,
4x panel apply, 3x panel inner), six of which touch only the 66 KB block.

Inside the CTA the serial 32-pivot chains stay **warp-synchronous on warp 0**
(exp 044 measured ~134ns/pivot for one warp against ~324ns/pivot for an
eight-warp `__syncthreads` chain, which is why exp 044's own fused-block
probes — block128/BK=16 at 47.4us and block128/BK=32 at 67.8us — all lost).
The other seven warps join only for the two block-parallel phases, so the
whole block costs **16 `__syncthreads`, not 128**.

Because the kernel cannot be graph-captured, every enrolled shape also moves
`graph` -> `eager`. Exp 044 v10 showed that swap alone is a net loss (0.9610x
at `16x512`: the CUDA micro saved ~50us of device time but eager launch gaps
cost ~66us over 54 launches). The fusion is what pays for it — the fused
schedule emits **11 launches at `16x512` instead of 53**.

## Measured ladder

All paired same-process on B200 against the exact snapshot, outputs retained
through the unchanged official checker, backend proven by positive
`_DIAG128_HITS` with zero new fallbacks.

| V | change | shape | baseline | candidate | ratio |
|---|---|---|---:|---:|---:|
| 1 | `diag128_potrf` at two shapes | 16x512 | 408.3us | 375.9us | **1.0858x** |
| 1 | | 4x1024 | 714.0us | 694.1us | **1.0288x** |
| 1 | off-target control | 640x512 | 1287.4us | 1287.0us | 1.0011x |
| 1 | off-target control | 60x1024 | 1191.4us | 1189.7us | 1.0016x |
| 4 | V1 + vectorised broadcasts + 4 more shapes | 16x512 | 387.2us | 363.1us | 1.0664x |
| 4 | | 4x1024 | 687.6us | 668.2us | 1.0289x |
| 4 | | 2x2048 | 1359.1us | 1343.4us | 1.0118x |
| 4 | | 8x2048 | 1571.2us | 1614.2us | **0.9735x** |
| 4 | aggregate over the six probed shapes | — | — | — | **1.0133x** |

The kernel is correct everywhere it runs and *improves* the residual
(16x512 `2.59 -> 2.54`, 4x1024 `9.25 -> 8.10`, both against a 20 tolerance).

## Why the device win does not reach the wall clock

`variant-01-shapediag.json`, the candidate's own kernel breakdown:

| shape | wall | device | `diag128_potrf` | panel | trailing | idle |
|---|---:|---:|---:|---:|---:|---:|
| 16x512 | 351.7us | 306.8us | **216.5 (4 calls, 54.1 each)** | 48.9 | 26.4 | 44.9 |
| 4x1024 | 648.5us | 458.3us | **333.2 (8 calls)** | 60.4 | 50.3 | **190.2** |

At `4x1024` the fusion cut *device* time `670.7 -> 458.3us` (**-32%**) but wall
only `706.7 -> 648.5us` (-8%). Two hard blockers ate the rest.

### Blocker 1 — the eager-launch tax (~7.6us per launch)

Idle grew from 36us (graph replay) to 190us over ~25 eager launches. The
mechanism is `custom_kernel`'s closing `torch.isfinite(...).all().item()`: it
drains the GPU on every call, so the *next* call's Python-side Triton dispatch
is fully exposed instead of overlapped. The shipped eager `640x512` shows the
identical ratio (352us idle / ~46 launches). CUDA graph replay costs ~0.4us
per launch instead — but a `<<<grid, block>>>` launch cannot be captured, and
naming the current work queue is refused by popcorn's source policy. **Any
CUDA kernel therefore costs ~7.6us x launch_count**, which is why `8x2048`
(49 launches) regressed even though its device time fell.

### Blocker 2 — the six-minute compile budget

Popcorn test `#898552` (V1) and `#898531` (V4) both failed at **exactly 360s**
— the service timeout, not a numerical failure. Adding one kernel to the
existing extension is enough to break a cold build. Exp 044 hit the same wall
going from three to four `load_inline` extensions. Note also that the renamed
extension forces a cold build; the incumbent's 94s test run benefits from a
warm cache.

(Separately, popcorn's source scanner is a literal substring match: the first
V4 submission was rejected for the word "stream" appearing in one of my own
CUDA *comments*.)

## Verdict

**FRONTIER, NOT PROMOTABLE.** +1.33% over six probed shapes is under the
measured leaderboard noise floor (byte-identical resubmissions vary 0.42%
public / 2.6% secret, exp 033), and the candidate cannot pass the compile gate
as written. No ranked slot was spent. The repository keeps `#890798`.

## What the evidence says is required for 2x

The diagonal chain is irreducible at ~200ns/pivot for a lone warp (exp 044
floor, reconfirmed here at 54.1us per 128-pivot block, identical at batch 4 and
16). For the leaders' ~317us geomean to be ~7.5x off the hardware floor on
every shape, `4x1024` must run near 36us, i.e. **~35ns/pivot** — only possible
if the pivot chain never leaves registers and *all* panel/trailing work is
overlapped with it by other warps, in one launch.

That is the persistent/cooperative architecture, and it is the only design that
removes both blockers at once (2 launches, so no eager tax; one kernel, so no
extra compile unit). Exp 048 V2 already measured **1.167x** with a crude
version of it — bulk-synchronous barriers, TILE=32, and a scalar panel solve
that consumed 46% of the kernel. Modelling it with this experiment's fused
128-wide diagonal block, a vectorised panel and a rank-128 WMMA trailing
update gives ~508us at `4x1024` (1.41x) and ~250us at `16x512` (1.63x); adding
lookahead so the trailing update overlaps the next diagonal factorisation is
what closes the remaining distance to 2x.

Two reusable results are banked here for that work: `diag128_potrf` itself
(correct, residual-improving, 54.1us per 128x128 block) and the measured
per-launch tax that makes launch count, not device time, the mid-shape
currency.
