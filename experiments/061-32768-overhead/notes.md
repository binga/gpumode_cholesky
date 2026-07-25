# Experiment 061 — 1x32768 overhead

Baseline: ranked #906955, commit `c965e08`,
`submission.py` SHA-256 `003fa60d1b59f6cd31656aaa069e3f02b0693d7af4f536585335e6f81d7ab175`
(copied verbatim to `baseline-906955.py`).

## Component profile (B200, `baseline-shapediag.json`)

wall 31245.5us, device 30766.2us, idle 479.3us (1.5%), 422 launches.

| share | us | calls | us/call | kernel | what it is |
|---:|---:|---:|---:|---|---|
| 36.3% | 11167.4 | 8 | 1395.9 | `getrf_wo_pivot` | `cholesky_ex` on the 4096x4096 diagonal blocks |
| 16.3% | 5002.9 | 107 | 46.8 | `elementwise_kernel<128,2>` | **generic strided** block moves (clones + factor stores) |
| 11.1% | 3419.6 | 4 | 854.9 | cutlass tf32 gemm | diagonal SYRK `addmm_` (large half) |
| 7.1% | 2195.2 | 5 | 439.0 | `nvjet ... Avec32UE8M0` | MXFP8 panel product |
| 7.1% | 2177.2 | 6 | 362.9 | `nvjet_sm100_hss` | FP16 solve-apply |
| 4.3% | 1313.4 | 3 | 437.8 | cutlass tf32 gemm | diagonal SYRK `addmm_` (small half) |
| 3.3% | 1021.2 | 12 | 85.1 | `_mx_quant_e4m3_blocked` | per-step requantization |
| 2.8% | 855.1 | 28 | 30.5 | `batch_trsm_left` | 256-wide leaves of `_blocked_tri_inv_32768` |
| 2.1% | 647.8 | 16 | 40.5 | FillFunctor | `zeros_like` (4.29 GB) + inverse zeroing + `eye` |
| 1.9% | 586.7 | 6 | 97.8 | CUDAFunctor | `panel.sub_(product)` |
| 1.7% | 509.2 | 14 | 36.4 | float16 cast | `.to(torch.float16)` for the solve |
| 1.5% | 458.5 | 8 | 57.3 | `triu_tril` | inside `cholesky_ex(...).L` |

Key reading: the 5003us in `elementwise_kernel<128,2>` is PyTorch's *generic*
(OffsetCalculator, non-vectorized) elementwise path. Every block move in the
loop has a strided operand — a 4096-wide window of a 32768-wide row, or the
column-major factor `cholesky_ex` returns — so none of them vectorize. Measured
throughput over those 107 launches is ~2.0 TB/s against ~7 TB/s achievable.

The diagonal (11167us) is cuSOLVER and pinned at ~0.33-0.38us/row; the sibling
1x16384 worker measured twelve blocked replacements and every one lost. Not
attacked here.

## Variants

### V1 — `candidate-v1-triton-move.py` — FRONTIER, 1.1204x

Mechanism: one Triton `_exp061_block_move_kernel` with explicit 2D strides for
both operands replaces every strided block move, and folds work into each pass:

* the diagonal clone and panel clone become strided->contiguous gathers;
* the panel gather also subtracts the MXFP8 product and emits FP16 directly,
  absorbing the old `panel.sub_(...)` and the old `.to(torch.float16)`;
* the two factor stores become contiguous->strided scatters, and the
  solve-apply skips its scatter entirely — `torch.mm(..., out_dtype=float32,
  out=factor[j:, k:j])` is accepted and cuBLAS writes through `ldc = 32768`
  (`_EXP061_MM_OUT_HITS: 7` proves it took that path);
* workspaces (`diagonal`, `panel_half`) are allocated once per call.

Arithmetic is unchanged: same TF32 diagonal SYRK, same MXFP8 panel product,
same recursive blocked inverse, same FP16 solve-apply, same order.

Selected by `_LARGE_CFG[32768]["path"] == "exp061_v1"`; `_left_looking_large`,
`_tri_inv_recursive`, `_mxfp8_panel_update` and `_factor_1x32768_blocked_inverse`
are all untouched and still reachable.

Paired (`variant-01-paired.json`): baseline 31575.9us -> candidate 28183.7us,
ratio 1.12036, CI95 [1.11982, 1.12161], MAD 0.01%, A-vs-A 0.03%,
`new_fallbacks: {}`, counters `_EXP061_V1_HITS 1 / _EXP061_MOVE_HITS 23 /
_EXP061_MX_PRODUCT_HITS 6 / _EXP061_MM_OUT_HITS 7 / _EXP061_STEP_HITS 8`.
Dense residual 5.28 vs baseline 5.28 (budget 20, margin 3.8x — unchanged).

Family (`variant-01-familygrid.json`): checker passes on all six families with
residuals 5.29 / 0.000544 / 6.1e-05 / 0.000494 / 4.26e-05 / 0.00633, matching
the incumbent's `combined-v1-familygrid.json` figure-for-figure. The fallbacks
on spectrum / lowrank / rowscale are the incumbent's inherited safety cases for
this shape, not a regression.

#### Invalid first attempt (repaired, not counted as a variant)

The initial V1 guarded on `stride(1) == 1` and raised, because
`torch.linalg.cholesky_ex(...).L` is returned **column-major**
(`cloneBatchedColumnMajor`), so its trailing stride is 4096, not 1. The whole
path fell through to `_left_looking_cholesky_32768` and measured 0.631x with
`new_fallbacks {_LARGE_FP8_FALLBACKS: [0, 1]}`. Fix: the move kernel takes both
strides for both operands and uses square tiles for a transposing move.
