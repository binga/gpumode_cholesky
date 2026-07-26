## Session 52 — 2026-07-26 — Experiment 064: the two biggest shapes

**Goal (user):** research how to improve latency on the two largest shapes,
implement, and submit to the leaderboard.

**Baseline.** Ranked `#912756` (public 675.753us / secret 674.448us), source sha
`6f602db3b3e7ced63918fe484c9ce7873ad1619589c4079aa5d061264a8e72d6`, commit
`e267fd3`.

### What the research said, and why most of it was already spent

Three external sources agreed on one architecture: FP16 on the large
off-diagonal blocks, FP32 on the diagonal, per-block rescaling
(arXiv 2601.08082, 5.07x over cuSOLVER FP32 at n >= 8192); FP8 blockwise GEMM
underperforming FP16 on B200 (arXiv 2512.02189, flashinfer #2146); and fusing
the panel factorization into one kernel to kill launch overhead (ICL/Dongarra).

Experiment 061 had already put both large shapes' trailing and panel updates on
FP16 and MXFP8. The measured MXFP8 block-column product runs at **3,466
TFLOP/s** and the FP16 panel apply at ~1,766 TFLOP/s, so the GEMM side of the
literature's recipe has no headroom left here. The probe confirmed this
directly: **swapping MXFP8 for FP16 on `1x32768` costs 15%** (24,434 ->
28,770us). The Blackwell FP8-vs-FP16 finding is about *blockwise-scaled* GEMM
shapes and does not transfer to exp 061's single-quantization formulation.

### The diagonal is a wall, and this is now measured three ways

`results/exp064-inc-shapediag.json` charges cuSOLVER's `getrf_wo_pivot`
**59.6% of `1x16384`** (5,037.9us) and **46.9% of `1x32768`** (11,154.0us).

| implementation | ns/row |
|---|---:|
| cuSOLVER `getrf_wo_pivot`, m=2048 | 308 |
| cuSOLVER `getrf_wo_pivot`, m=4096 | 340 |
| exp-063 v3 resident block kernel | 296 |
| register-resident pivot chain, isolated | 63 |

The repo's best custom block kernel is only 4% better per row than the vendor,
and a 2048 diagonal built from sixteen of its 128-blocks (16 x 37.9 = 606us
against 630us) loses once the inter-block panel and trailing GEMMs are added.
Experiment 061's `probe-01-diag` had already measured every op-level blocked
alternative at m=2048 as slower than cuSOLVER, and experiment 063 concluded
"~296 ns/row is about where that road ends". **The diagonal needs plan item 2
(named-barrier overlap of the serial chain with the parallel phases, projected
~195 ns/row) and nothing less.** That is a separate CUDA project.

So this experiment took the 40-53% that is *not* the diagonal.

### Adopted — two pure-Python driver changes

1. **`1x16384`: every strided block move through the exp-061 Triton mover.**
   The shipped path still ran four strided operations per block step on
   PyTorch's generic `OffsetCalculator` elementwise kernel — **1,216us over 38
   launches, 14% of the shape**, at ~2 TB/s against ~7 TB/s achievable. This is
   the exact defect exp 061 diagnosed and fixed on 32768 and never ported back.
   The subtract and the fp16 down-cast fold into the gather that was already
   reading the data, and the fp16 panel operand goes into a reused buffer
   instead of a fresh `.to(float16)` copy of the whole panel each step.
2. **`1x32768`: the trsm-free base-32 leaf inverse.**
   `_blocked_tri_inv_32768` bottomed out at 256-wide `solve_triangular` leaves,
   which the profile charges **850us of `batch_trsm_left_kernel` over 28
   launches**. The 16384 path already used a base-32 Triton leaf; reusing it
   replaces those 28 launches with 7 and leaves the recursive tree's GEMMs
   untouched.

No CUDA source changed — all five CUDA blobs are byte-identical to the ranked
source and the `load_inline` count is unchanged — so the cold-build budget that
exps 044/050 fought over is provably untouched.

### Rejected with evidence

- **Zeroing only the strict upper block triangle** is *slower* than a bulk
  `zeros_like`, on both shapes at every `nb`. The bulk fill is one vectorized
  `FillFunc` at HBM write bandwidth; the block-triangle version issues strided
  `zero_()` calls that land back on the generic elementwise path. Writing 44% of
  the bytes badly costs more than writing 100% of them well.
- **FP16 instead of MXFP8** on the 32768 block-column update: 0.849x.
- **`nb` sweeps**: 2048 is already optimal for 16384 (1024 -> 0.958x, 4096 ->
  0.923x) and 4096 for 32768 (2048 -> 0.781x).

### Gates

| gate | result |
|---|---|
| free (compile, 10/10 property, whitespace, source policy) | pass; cuSOLVER/stream/queue token counts byte-identical |
| full 15-shape paired grid | **geomean 1.0073** CI95 [1.0068, 1.0079], excludes 1.0, `all_shapes_ok` |
| counter diff | new counters only (`_EXP061_MOVE_HITS` 38, `_EXP064_TRSMFREE_HITS` 7); `new_fallbacks` empty |
| cold build | all 5 CUDA blobs byte-identical, `load_inline` count unchanged |

| shape | control us | candidate us | speedup |
|---|---:|---:|---:|
| `1x16384` | 8,781.0 | 8,374.4 | **1.0485x** |
| `1x32768` | 24,529.9 | 22,971.2 | **1.0679x** |
| other 13 | — | — | 0.9978-1.0009 (flat) |

Residual: `1x16384` unchanged at 0.211; `1x32768` 6.44 -> 6.45 against a
tolerance of 20.
