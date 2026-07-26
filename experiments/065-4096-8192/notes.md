# Experiment 065 — 1x4096 and 1x8192: four levers, four rejections

**Baseline:** ranked `#913511` (exp 064 candidate-v1), sha256
`8e4603e5…0baa`, public 672.383us / secret 655.423us.

**Verdict: ALL CANDIDATES REJECTED.** Nothing ranked, nothing adopted. The
value of this experiment is the measured model of where `1x4096`'s time goes
and the elimination of every config-level lever on both shapes.

## Why these two shapes were picked

Per-shape speedup since the original cuSOLVER submission `#876988` (official
leaderboard per-shape times; their geomean reproduces its 2080us score exactly):

| shape | `#876988` | now | speedup |
|---|---:|---:|---:|
| `1x4096` | 1535us | 1446us | **1.06x** |
| `1x8192` | 6400us | 5452us | **1.17x** |
| (median of the other 13) | | | 3.2x |

These are the two least-improved shapes on the board. Profile
(`results/inc-890798-shapediag.json`) explains `1x4096` completely:

```
batch=1 n=4096   wall 1527.2us   6 kernel launches   0.2% idle
   91.0%   1391.8us   x1   getrf_wo_pivot          <- ONE call
    4.9%     75.3us   x1   elementwise
    3.8%     57.3us   x1   triu_tril
```

One kernel, no GEMM of any kind, 338 ns/row. cuSOLVER does not block `n=4096`
at all — the whole factorization is a single serial per-row dependency chain.
`1x8192` is different: it is already on this repo's `_left_looking_large`
(`_LARGE_CFG[8192]`, nb=2048, tf32) and splits 46.8% `getrf_wo_pivot` /
25.2% `trsm_right` / ~16% tf32 cutlass GEMMs.

> Note: the comment at `submission.py:3686` ("8192 … stays on cuSOLVER") is
> **stale**. The `batch == 1 and n in _LARGE_CFG` gate above it catches 8192
> first. `1x4096` is the only single-matrix shape still on stock cuSOLVER.

## Candidates and results

All four are one-line diffs against the exact ranked incumbent. All were
measured with the paired same-process B200 harness (`pairedgrid`), which
interleaves A-B-B-A and discards the first repeat.

| # | change | shape | ratio | CI95 | verdict |
|---|---|---|---:|---|---|
| v1 | `_LARGE_CFG[8192]` `rec_inv=True` | 8192 | **0.9460** | [0.9459, 0.9465] | REJECT — 5.4% slower |
| v3 | `_LARGE_CFG[8192]` `panel_mode="mxfp8"` | 8192 | 1.0023 | [1.0020, 1.0028] | REJECT — numerics |
| v4 | enroll `(1, 4096)` in `_EXP062_SHAPES` | 4096 | **0.7232** | [0.7223, 0.7239] | REJECT — 38% slower |
| v2 | v1 + v3 combined | — | — | — | not run; v1 is a loss |

### v1 — recursive triangular inverse at 8192 (`rec_inv=True`)

5755.5us -> 6084.2us. The exact flag that won **1.4177x** at `1x16384` in
exp 057 *loses* 5.4% at `1x8192`. Mechanism: `_tri_inv_recursive(2048)` costs
the same per panel at both n, but 8192 has only 4 panels with trailing heights
6144/4096/2048, versus 16384's 8 panels up to 14336 rows. There are not enough
trailing rows to amortize the inverse against the `solve_triangular` it
replaces. **`rec_inv` is an n>=16384 lever, not a large-n lever.**

### v3 — MXFP8 block-scaled panels at 8192

5768.7us -> 5754.0us, a real but negligible **+0.23%** — and the scaled
reconstruction residual goes **0.19 -> 13.7** against a tolerance of 20.
That is 72x the error for a quarter of a percent, leaving only 1.45x of
margin on a shape whose secret-split conditioning we do not control. Rejected
on risk, not on speed. (`_MXFP8_HITS: 2`, `new_fallbacks: {}` — the path did
engage.) The panel GEMMs are only ~16% of this shape and are already on tf32
tensor cores, so the quantization overhead eats most of the demotion gain.

### v4 — enrolling `1x4096` in the exp-062/063 blocked path

The headline result:

| | batch=1 | batch=2 |
|---|---:|---:|
| cuSOLVER (`#913511` baseline) | **1528.7us** | 2280.6us |
| exp-062 blocked path | 2113.8us | 2280.9us |
| ratio | **0.7232** | 0.9998 |

`(2, 4096)` is already enrolled and is unaffected (0.9998), so this is a clean
isolation. Reading the two columns together:

- exp-062 path, batch=1: 2113.8us = **516 ns/row** in situ
- exp-062 path, marginal cost of the 2nd matrix: 2280.9 - 2113.8 = **167us**
- cuSOLVER, batch=1: 1528.7us = **373 ns/row**, fused into one kernel

The blocked path's entire advantage is that it "pays the pivot chain once for
the whole batch" (its own docstring). At batch=1 there is nothing to amortize,
and it loses to cuSOLVER by 1.38x because it adds explicit `baddbmm` trailing
updates and per-block panel `bmm`s that cuSOLVER keeps inside one launch.

**Standing lesson: for a single matrix, cuSOLVER's unblocked fused potrf is
the fastest implementation available in this repo. Enrolling any `batch == 1`
shape into a blocked path is a regression until the block kernel's in-situ
ns/row beats 373.**

## What this rules out, and what is left

Ruled out by measurement this round:

1. Reduced precision at `1x4096` — there is no GEMM in the shape to demote.
   Its profile has zero cutlass/nvjet kernels. This is the cleanest instance
   of the documented fp16 null results.
2. Triton at `1x4096` — the bottleneck is a loop-carried dependency inside a
   single kernel; exp 029 measured ~16us/launch for serial Triton loops and
   killed persistent kernels. Triton's wins here have all been bandwidth-shaped
   (exp-034 quantizer, exp-061 block mover); this shape has no such work.
3. Both `_LARGE_CFG[8192]` config flips (v1, v3).
4. Blocked enrollment of `1x4096` (v4).

Superseded without needing a run: **exp 063's "fix the fused chain+inverse
correctness bug" (plan item 1)**. That fusion measured 325.9 ns/row and v2
"panel + fused inverse" measured 312.3 ns/row — both *slower* than the v3
kernel that actually shipped in `#912756` at **296 ns/row**. Fixing v1's
`inv_err 0.573` bug would be repairing a design that v3 already beats.

**The one remaining lever is absolute ns/row**, and it is exp 063's own stated
`next_lever` (echoed by exp 064's `next_action`): look-ahead / named-barrier
overlap inside the 128x32 diagonal block kernel. Its measured budget is
chain ~11us vs parallel phases ~14us per 37.9us block; running them
concurrently approaches max instead of sum, ~25us/block = **~195 ns/row**.

Projected value if that lands:

- five already-enrolled shapes (`16x512`, `4x1024`, `8x2048`, `2x2048`,
  `2x4096`) gain on top of their current 1.09-1.11x
- `1x4096` becomes reachable for the first time: 4096 x 195ns = 799us of chain
  against cuSOLVER's 1528.7us, i.e. the first blocked design with real margin
  at batch=1

This is a warp-specialization rewrite of the kernel, not a config change, and
it should be scoped as its own experiment.

## Artifacts

- `baseline-913511.py` — exact ranked incumbent
- `candidate-v1.py` (rec_inv), `candidate-v3.py` (mxfp8), `candidate-v4.py`
  (enroll 1x4096); `candidate-v2.py` (v1+v3) built but not run
- `results/exp065-v1-probe8192.json`
- `results/exp065-v3-probe8192.json`
- `results/exp065-v4-probe4096.json`

## Cost

Three B200 `pairedgrid` sandbox runs, subset-filtered (`--shapes 8192`,
`--shapes 4096`). No full 15-shape grid was run: the program's gate ladder
only spends that after a credible subset win, and no candidate produced one.
No Popcorn test or ranked quota consumed.

---

## Addendum — where the Triton and precision levers *do* pay

The rejections above are specific to `1x4096`'s serial chain, not to the levers
themselves. A fresh 13-shape profile of the exact incumbent `#913511`
(`results/exp065-inc913511-shapediag.json`) locates both.

| shape | wall | lnch | idle (graphs) | GEMM-ish (precision) | glue (Triton) | chain |
|---|---:|---:|---:|---:|---:|---:|
| 4096x32 | 16.6 | 1 | 14.5% | — | — | — |
| 1024x64 | 25.4 | 1 | 10.2% | — | — | — |
| 256x128 | 62.2 | 1 | 2.9% | — | — | — |
| 64x256 | 99.3 | 1 | 1.8% | — | — | — |
| 16x512 | 337.7 | 21 | **25.5%** | 9.5% | 12.5% | 52.6% |
| **640x512** | 1334.3 | 53 | 15.4% | **69.5%** | 2.3% | 12.9% |
| 4x1024 | 607.8 | 37 | **22.9%** | 11.1% | 8.2% | 57.8% |
| **60x1024** | 1282.2 | 76 | **20.5%** | **43.2%** | 1.9% | 34.5% |
| 2x2048 | 1137.1 | 69 | 17.6% | 13.5% | 7.4% | 61.5% |
| 8x2048 | 1302.9 | 69 | 11.5% | 17.7% | **16.1%** | 54.8% |
| 1x4096 | 1531.6 | 6 | -0.1% | **0.0%** | 8.9% | **91.2%** |
| 2x4096 | 2319.0 | 133 | 10.6% | 18.3% | 10.4% | 60.7% |
| 1x8192 | 5802.7 | 242 | 6.8% | 15.0% | 10.9% | 67.2% |

Buckets: `idle` = wall - device (exposed eager launch gap); `GEMM-ish` =
cutlass/nvjet + the `_trailing_nb` / `_panel_fused128` / `_panel_inner32` /
`_panel_apply32` Triton kernels; `glue` = elementwise, `triu_tril`, memcpy,
memset, fill, reduce; `chain` = `getrf_wo_pivot`, `trsm_right`, `e62_diag128`,
`micro_potrf32_rank4`, `_diag_block_step`.

### Precision — two shapes carry it

**`640x512` is the precision shape: 69.5% of wall (926.8us) is GEMM-shaped,
chain is only 12.9%.** batch=640 is throughput-bound with maximum parallelism,
the opposite regime from `1x4096`. Breakdown: `_panel_fused128` 359.1us,
`_trailing_nb` 193.8us, `_panel_inner32_subtile64` 155.7us, `_panel_apply32`
105.4us, cutlass tf32 112.8us. **Caveat: n=512 sits below the documented
fp16-safety line (n>=1024)**, so it needs Higham-capped scaling, not a naive
demotion.

**`60x1024` is the risk-adjusted pick: 43.2% GEMM-ish (553.3us — `_trailing_nb`
375.2us + `_panel_fused128` 178.1us) and n=1024 clears the fp16 safety
threshold.**

Note both shapes' hot kernels are *already Triton*. The change there is the
operand dtype / `tl.dot` precision inside existing kernels, not a conversion —
the two levers converge on the same code.

Then `2x4096` (18.3%), `8x2048` (17.7%), `2x2048` (13.5%). `1x8192` (15.0%) is
already rejected above on numerics, and `1x4096` is 0.0% — there is nothing
there.

Geomean if GEMM-ish halves: `640x512`+`60x1024` alone **643.0us (-4.4%)**;
adding `8x2048`+`2x4096`+`2x2048` **632.0us (-6.0%)**.

### Triton conversion — the glue, and it is modest

`8x2048` 16.1% (209.2us: `triu_tril` 78.9, elementwise 66.5, Memcpy DtoD 45.1),
`1x8192` 10.9% (634.4us, elementwise alone 509.8), `2x4096` 10.4% (240.2us).
Fusing all three: **654.7us (-2.6%)**.

Worth recording: even `1x4096` has 135.9us of glue (8.9%) outside its single
cuSOLVER call — `triu_tril` 57.8us + elementwise 75.1us. A fused epilogue is a
genuine ~1.10x on that shape (**-0.6% geomean**). So Triton is not void at
`1x4096`; it is void on the 91.2% chain and available on the 8.9% glue.

### The biggest lever is neither: exposed launch gap

The exp-062/063 blocked path is eager, and the gap shows up as idle:

```
16x512    86.0us / 21 launches = 4.10 us/launch   (25.5% of wall)
4x1024   138.9us / 37          = 3.75            (22.9%)
60x1024  262.8us / 76          = 3.46            (20.5%)
640x512  205.5us / 53          = 3.88            (15.4%)
2x2048   199.9us / 69          = 2.90            (17.6%)
2x4096   245.9us / 133         = 1.85            (10.6%)
```

Recovering it on those six shapes is **618.4us (-8.0%)**, the largest of the
three levers — and `submission.py` already carries `_graph_cholesky_16x512`,
`_graph_cholesky_256x128` and `_graph_cholesky_1024x64`. The exp-062 gate sits
*above* the `16x512` graph helper in `custom_kernel`, so that shape traded a
captured graph for an eager 21-launch path and now shows the worst idle
fraction on the board. Graph-capturing the exp-062 path is the follow-up.

The three buckets overlap and are not additive (recovering idle changes what
fusing glue is worth).

---

## Round 2 — CUDA graphs on 60x1024: REJECTED, and the graph branch is dead code

**Target:** the 262.8us (20.5% of wall) of exposed launch gap on `60x1024`.
`_SPLIT32_SHAPES` already carries a per-shape `"graph"` / `"eager"` mode, so
this looked like a one-line flip.

### It is not a one-line flip: every `"graph"` entry is unreachable

| entry | mode | intercepted by |
|---|---|---|
| `(256, 128)` | graph | `cholesky128_block16` CUDA kernel (1 launch) |
| `(64, 256)` | graph | `cholesky256_wmma16` CUDA kernel (1 launch) |
| `(16, 512)` | graph | `_EXP062_SHAPES` gate |
| `(4, 1024)` | graph | `_EXP062_SHAPES` gate |
| `(8, 2048)` | graph | `_EXP062_SHAPES` gate |
| **`(640, 512)`** | **eager** | — reaches `_split32_factor` |
| **`(60, 1024)`** | **eager** | — reaches `_split32_factor` |

Dispatch order in `custom_kernel` is cuda128/cuda256 -> exp062 -> split32, so
the only two shapes that reach `_split32_factor` are the two marked `"eager"`.
**No `"graph"` entry has a live caller, and the graph branch of
`_split32_factor` is therefore dead code.** The 1-launch profiles for
`256x128` / `64x256` confirm the interception directly.

### Both attempts fail the same way

| # | change | `60x1024` ratio | CI95 | `4x1024` control |
|---|---|---:|---|---:|
| v5 | `(60,1024)` -> `"graph"` | **0.2910** | [0.2905, 0.2917] | 1.0019 |
| v6 | v5 + graph branch captures the eager two-buffer dataflow | **0.2968** | [0.2967, 0.2972] | 0.9996 |

1214.3us -> 4092.1us, a 3.4x regression. The mechanism is identical in both:

- `errors: {}` in `familygrid` — **no exception is raised**, so capture
  succeeds and it is the *replay* that is wrong.
- the dispatch's `torch.isfinite(l.diagonal(...)).all()` guard rejects the
  replayed factor, `_FUSED_CTA_FALLBACKS` goes 0 -> 1, and the call falls
  through to a much slower route.
- the candidate's residual drops 9.33 -> 0.00254, i.e. it landed on exact-FP32
  cuSOLVER rather than the tf32 split32 path — consistent with the fallback.
- `_MICRO32_HITS: 96` = 3 x 32 (two warmup passes + the capture pass). Note
  replay does **not** increment Python-side counters, so these counts say
  nothing about whether replay ran; they only confirm warmup and capture did.

v6 tested the one structural difference between the dead branch and the live
path: the branch factors **in place** (`_split32_launch(work, ...)`) while every
live call uses the two-buffer first-touch form (`_split32_launch(out, ...,
src=data)`). Capturing the eager dataflow instead changed nothing —
0.2910 -> 0.2968 is the same failure. **That hypothesis is disproved; the root
cause of the non-finite replay is not identified.**

### Assessment

The prize is smaller than the idle number suggests. `60x1024` is 251.7 MB, so
graph mode must add a copy-in and a clone-out (~126us of extra traffic at
~8 TB/s) against 262.8us of recovered idle — a net ~137us, roughly 1.11x on the
shape, ~1.0% geomean. `640x512` (205.5us idle) rides the same code path and
would roughly double that.

Finding the actual defect needs a numerical diff of graph-replay output against
eager output, which means a new `_gpu_runner.py` probe mode and further paid
runs. Two bounded attempts were spent; stopping here rather than guessing
further. What the next attempt should do first is establish whether *any* CUDA
graph path still replays correctly in this codebase — the `_graph_cholesky_*`
helpers for `256x128` / `1024x64` / `16x512` are equally shadowed by later
dispatch entries and equally unexercised, so graph capture may be broadly
rotted rather than specifically broken at `60x1024`.

Artifacts: `candidate-v5.py`, `candidate-v6.py`,
`results/exp065-v5-probe1024.json`, `results/exp065-v5-family1024.json`,
`results/exp065-v6-probe1024.json`.
