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

### Blocker 2 — the build cache, and how it was actually solved

Popcorn tests `#898552` (V1), `#898531` (V4) and `#898645` (V5) all failed at
**exactly 360s** — the service timeout, not a numerical failure. Three
hypotheses were tested rather than guessed at:

| probe | result | conclusion |
|---|---|---|
| V5: the expensive 32-pivot chain + inverse refactored into **one** shared `__device__ __noinline__` function instead of two inlined copies | still 360s | not code duplication |
| `probe-v5-nodispatch.py`: `diag128_potrf` **compiled but never launched** | still 360s | not a runtime hang — it is build time |
| Modal timing of `load_inline` itself | base **42.3s**, with `diag128` **41.1s** | the new kernel costs *nothing* to compile |
| exact ranked `submission.py` (`#898670`) | **passed in 91s** | the runner is healthy |
| `probe-rename-only.py`: ranked source, **only the extension name changed** (`#898675`) | **360s** | ← root cause |

**The official runner keeps a build cache keyed by extension name.** The
incumbent's 91s test is a cache hit on names first compiled by earlier
submissions; any new name forces a cold build, and a cold build of this
submission's **four** `load_inline` calls — four compiles of the very expensive
`torch/extension.h` pybind glue — does not fit in 360s. This, not "a fourth
extension", is what exp 044 actually hit.

**Fix: one extension for every CUDA kernel** (`_CUDA_ALL_SOURCE`, V6). The
four sources concatenate cleanly once `_CUDA64_SOURCE`'s `N` is renamed `N64`
(the only symbol collision). The pybind glue is then compiled once instead of
four times. V6 — a brand-new extension name, cold build, *plus* the new
`diag128_potrf` kernel — passes Popcorn **17/17 in 36 seconds** (`#898689`),
against the incumbent's 91s warm-cache run.

This is the most reusable result in the experiment: it is what makes any future
CUDA work shippable at all, and it removes the constraint that shaped exps 042,
043 and 044.

(Separately, popcorn's source scanner is a literal substring match: the first
V4 submission was rejected for the word "stream" appearing in one of my own
CUDA *comments*.)

## V6 — the full-grid gate reverses the subset result

With the compile blocker gone, V6 (merged extension + `diag128_potrf` at
`16x512`, `4x1024`, `2x2048`) went through the real promotion gates.

**Six-family correctness: clean.** 36/36 cases pass the official checker, worst
residual 9.59 of 20, zero errors. The `lowrank`/`spectrum` rows that show an
inactive backend fall back exactly as the unchanged `640x512` and `8x2048`
shapes already do.

**Full 15-shape paired grid: `geomean 0.9865`, CI [0.9853, 0.9878] — a 1.35%
regression.** The twelve unchanged shapes are all flat (0.9996–1.0052,
including `64x256`, which absorbs the `-O2 -> -O3` change from the merge). The
entire loss is on the three enrolled shapes, and every one of them **reversed**
against its own subset probe:

| shape | subset probe | full grid | candidate us (subset -> grid) |
|---|---:|---:|---|
| `16x512` | 1.0858x | **0.9794x** | 375.9 -> 412.4 |
| `4x1024` | 1.0288x | **0.9252x** | 694.1 -> 763.9 |
| `2x2048` | 1.0118x | **0.8920x** | 1343.4 -> 1526.6 |

The baselines barely moved (408.3->403.7, 714.0->706.7, 1359.1->1361.7); it is
the *candidate* that got 10-14% slower once the other twelve shapes shared its
process. That is the eager path degrading — it allocates per call and is bound
by Python-side dispatch, so it is sensitive to allocator and interpreter state
that CUDA-graph replay is immune to. The measured ~7.6us/launch tax is
therefore a *floor*, not the real cost in the scoring environment.

**Lesson worth more than the kernel: a subset paired probe systematically
overstates an eager-mode candidate.** Only the full 15-shape paired grid is
trustworthy for any change that leaves graph replay. The first four
measurements in this experiment were all subset probes and all pointed the
wrong way.

## Verdict

**REJECTED on the full-grid gate (0.9865). Nothing ranked; the repository
keeps `#890798`.**

Two results are banked and reusable:

1. **The single merged extension.** Performance-neutral across all twelve
   unchanged shapes and it turns a >360s cold build into a 36s one
   (`#898689`, 17/17). Without it no new CUDA kernel can ship at all.
2. **`diag128_potrf` itself** — correct, residual-improving, 54.1us per 128x128
   block, batch-independent.

What is *disproved* is the delivery vehicle: enrolling shapes in a CUDA kernel
forces eager mode, and eager mode loses more inside the real 15-shape process
than the fusion wins.

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
