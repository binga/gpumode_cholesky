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
