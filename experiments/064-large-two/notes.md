# Experiment 064 — the two biggest shapes, `1x16384` and `1x32768`

**Goal (user):** research how to improve latency for the large shapes,
implement, and submit to the leaderboard.

**Baseline.** Ranked `#912756` (public 675.753us / secret 674.448us), source
sha `6f602db3b3e7ced63918fe484c9ce7873ad1619589c4079aa5d061264a8e72d6`, which
is the repo root `submission.py` at commit `e267fd3`.

## Web research

Three findings, all of which point at the same architecture:

- **arXiv 2601.08082, "Hierarchical Precision and Recursion for Accelerating
  Symmetric Linear Solves on MXUs"** — recursive tree-POTRF/TRSM/SYRK with FP16
  on the large off-diagonal blocks and FP32/FP64 on the diagonal, per-block
  rescaling `alpha = max(1, ||B||_inf / Rmax)`. Reports 5.07x over cuSOLVER FP32
  on H200, advantageous only at `n >= 8192`.
- **arXiv 2512.02189 (Blackwell microbenchmarks)** and **flashinfer #2146** —
  FP8 blockwise GEMM on B200 is *slower* than FP16 in most configurations and
  Hopper still beats Blackwell on FP8 GEMM. Independent confirmation of this
  repo's own measurement (fp16 1262.7 TFLOP/s vs tf32 736.9).
- **ICL/Dongarra batched Cholesky (icl-utk-987-2017)** — merge the panel
  factorization into a single kernel to remove small-kernel launch overhead;
  right-looking exposes the most parallelism for the trailing update.

**The literature's headline lever is already spent in this repository.**
Experiment 061 put the trailing and panel updates of both large shapes on FP16
and MXFP8 tensor cores. The measured MXFP8 block-column product on `1x32768`
runs at **3,466 TFLOP/s** and the FP16 panel apply at ~1,766 TFLOP/s, so the
GEMM side has no headroom left.

## Where the time actually goes (`results/exp064-inc-shapediag.json`)

| | `1x16384` | `1x32768` |
|---|---:|---:|
| wall | 8,748.4us | 24,384.2us |
| `getrf_wo_pivot` (cuSOLVER diagonal) | **5,037.9 (59.6%)** | **11,154.0 (46.9%)** |
| MXFP8 block-column GEMM | — | 3,330.9 (14.0%) |
| FP16 GEMM | 1,301.8 (15.4%) | 2,179.1 (9.2%) |
| generic strided elementwise | 1,216.3 (14.4%) | 1,484.1 (6.2%) |
| MXFP8 quantizer | — | 1,112.5 (4.7%) |
| exp-061 Triton block move | — | 1,035.6 (4.4%) |
| cuBLAS `batch_trsm` | — | 850.5 (3.6%) |
| `zeros_like` fill | 145.5 | 647.1 (2.7%) |
| `triu_tril` (from `.L`) | 122.6 | 456.1 (1.9%) |

## The diagonal is a measured wall, not an unexplored lever

The diagonal `potrf` is the single largest cost in both shapes, and it cannot
be moved with the tools this repository already has:

| implementation | ns/row | source |
|---|---:|---|
| cuSOLVER `getrf_wo_pivot`, m=2048 | 308 | this profile (5037.9/8/2048) |
| cuSOLVER `getrf_wo_pivot`, m=4096 | 340 | this profile (11154.0/8/4096) |
| exp-063 v3 resident block kernel | **296** | `063-diag128-fast/notes.md` |
| exp-063 v0 (ranked `#909488`) | 384 | same |
| register-resident pivot chain, isolated | 63 | exp-062 round 2 |

Substituting the repo's best custom block kernel for the vendor is a ~4% win on
paper (16 x 37.9us = 606us against 630us for a 2048 block) and a loss once the
inter-block panel and trailing GEMMs are added back. Experiment 061's
`probe-01-diag` had already measured every op-level blocked alternative at
m=2048 and found all of them slower than cuSOLVER (best `blocked_leaf1024_tf32`
1089us against 676us).

Experiment 063 closed this question directly: *"the plan's ~120 ns/row is not
reachable by parallelising the serial phase; ~296 ns/row is about where that
road ends."* The remaining lever is plan item 2 — overlapping the serial chain
with the parallel phases via named barriers, projected at ~195 ns/row — which
is a separate CUDA project and primarily a mid-shape lever.

**So this experiment targets the 40-53% that is not the diagonal.**

## Candidate changes

1. **Port exp-061's strided Triton move to the `1x16384` driver.** The shipped
   16384 path still routes four strided operations per block step through
   PyTorch's generic `OffsetCalculator` elementwise kernel — 1,216us over 38
   launches, 14% of the shape. This is exactly the defect exp 061 diagnosed and
   fixed on 32768 and never ported back.
2. **Emit the fp16 panel operand directly** instead of allocating a fresh
   `.to(torch.float16)` temporary of the whole panel on every step.
3. **Give `1x32768` the trsm-free base-32 inverse** that 16384 already uses,
   retiring 850us of `batch_trsm_left_kernel`.
4. **Zero only the strict upper block triangle.** Both drivers write every
   block of the lower block triangle in full, so `zeros_like` writes the whole
   matrix when ~44% is enough (647us of pure HBM write at 32768).
5. **`nb` sweep and fp16-vs-MXFP8 column mode**, priced rather than assumed.

## Probe results — `results/exp064-largephase-v1.json`

Phase breakdown of the shipped drivers (CUDA events between phases):

| phase | `1x16384` | `1x32768` |
|---|---:|---:|
| diagonal | 5,594.4us (64.7%) | 12,514.3us (52.4%) |
| update | 1,346.8 (15.6%) | 4,800.5 (20.1%) |
| inverse | 781.4 (9.0%) | 3,323.0 (13.9%) |
| panel | 918.7 (10.6%) | 2,296.2 (9.6%) |
| block moves | — | 962.7 (4.0%) |

Variants. **The control row goes through `custom_kernel` and therefore carries
the end-of-call `isfinite` sync; every other row calls the driver directly, so
the honest comparison is against the same-shape reimplementation of the shipped
logic (`nb2048` / `nb4096_mxfp8`), not against `control_shipped`.**

`1x16384`:

| variant | us | vs shipped logic | tol_frac |
|---|---:|---:|---:|
| control_shipped (via `custom_kernel`) | 8,822.7 | — | 0.0106 |
| nb2048 (= shipped logic) | 8,810.1 | 1.000 | 0.0106 |
| **v2_move_nb2048_zeros** | **8,317.2** | **1.059x** | 0.0106 |
| v2_move_nb2048_upper | 8,441.9 | 1.044x | 0.0106 |
| nb1024 | 9,200.3 | 0.958x | 0.0113 |
| nb4096 | 9,542.4 | 0.923x | 0.0091 |

`1x32768`:

| variant | us | vs shipped logic | tol_frac |
|---|---:|---:|---:|
| control_shipped (via `custom_kernel`) | 25,279.1 | — | 0.3220 |
| nb4096_mxfp8 (= shipped logic) | 24,434.3 | 1.000 | 0.3220 |
| **v2_nb4096_trsmfree_zeros** | **22,872.4** | **1.068x** | 0.3226 |
| v2_nb4096_trsmfree_upper | 23,298.7 | 1.049x | 0.3226 |
| v2_nb4096_trsm_zeros | 24,442.3 | 1.000 | 0.3220 |
| nb4096_fp16 | 28,770.2 | 0.849x | 0.0053 |
| nb2048_mxfp8 | 31,273.7 | 0.781x | 0.3436 |
| nb2048_fp16 | 34,168.7 | 0.715x | 0.0057 |

### Two hypotheses killed

- **Zeroing only the strict upper block triangle is slower than a bulk
  `zeros_like`**, on both shapes and at every `nb`. The bulk fill is a single
  vectorized `FillFunc` running at HBM write bandwidth; the block-triangle
  version issues strided `zero_()` calls that land back on the generic
  elementwise path. Writing 44% of the bytes badly costs more than writing 100%
  of them well.
- **Replacing MXFP8 with FP16 on the `1x32768` block-column update loses 15%.**
  The external literature (Blackwell microbenchmarks, flashinfer #2146) reports
  FP8 GEMM underperforming FP16 on B200, but that does not transfer here: this
  path's MXFP8 product is measured at 3,466 TFLOP/s against FP16's 1,262. The
  generic finding is about *blockwise-scaled* FP8 GEMM shapes, and exp 061's
  single-quantization block-column formulation is already past it. FP16 does
  buy a much better residual (0.0053 against 0.3220) but no speed.

`nb` is also confirmed already optimal on both shapes: 2048 for 16384 and 4096
for 32768, in both directions.

## Adopted changes

1. `_exp061_factor_1x16384` — every strided block move through the exp-061
   Triton mover; fp16 panel operand emitted into a reused buffer.
2. `_exp061_factor_1x32768` — trsm-free base-32 leaf inverse with a
   caller-owned buffer, guarded by `_EXP064_TRSM_FREE` with a
   `_blocked_tri_inv_32768` fallback and hit/fallback counters.

Both are pure-Python driver changes. No CUDA source changed, so the extension
count and the cold-build budget are untouched.

