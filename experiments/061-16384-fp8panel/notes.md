# Experiment 061 - batch=1 n=16384

## Result

**VALIDATED_FRONTIER: 1.15555x paired** (10159.4us -> 8792.0us), CI95 [1.15484, 1.15733],
zero accuracy cost (residual 0.211/20 on both incumbent and candidate).

## How the target was found

The brief's primary hypothesis was that this shape's panel GEMMs should move to MXFP8
like the sibling 1x32768 path. The kernel profile (`experiments/057-1x16384/variant-04-shapediag.json`)
says otherwise: at 9608us device time, `getrf_wo_pivot` -- cuSOLVER's potrf on the eight
2048x2048 diagonal blocks -- is 5010us over 8 calls = **52.1%**, while all TF32 GEMMs
together are ~25%. The GEMMs were never the bottleneck.

Note also that the shipped 16384 path is `_factor_1x16384_trsm_free`, not
`_left_looking_large`; the `_LARGE_CFG[16384]` entry is dead for this shape.

## probe-01: the diagonal is a wall (and why)

Measured cuSOLVER `cholesky_ex` cost against block size:

| m | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| us | 61.6 | 103.4 | 186.7 | 348.5 | 676.4 | 1537.9 |
| us/row | 0.481 | 0.404 | 0.365 | 0.340 | **0.330** | 0.376 |

This is **near-linear, not cubic**: the kernel is serial-latency-bound at ~0.33us per row,
i.e. ~20us per 64-wide panel step regardless of trailing size. Two consequences:

1. Total diagonal cost over the factorization is `(n/nb) * c*nb = c*n`, **independent of nb**.
   nb=2048 is already the optimum (5411us) and the whole 128..4096 range spans only
   5411-7878us. **nb tuning is dead.**
2. Splitting the block only pays back the small constant term while adding launch-bound
   steps. All twelve PyTorch-op blocked replacements at m=2048 lost to a single cuSOLVER
   call: best `blocked_leaf1024_tf32` 1089us (1.6x slower), `blocked_leaf256_invapply`
   2536us (3.8x slower). Even with **free** leaves the leaf-128 inner loop still costs
   ~730us vs cuSOLVER's 676us. **Op-level blocking of the diagonal is dead.**

TF32 inside the diagonal is strictly dominated -- slower than fp32 *and* 500x worse
residual (tol_frac 0.037 vs 7e-5).

## probe-02: what is actually addressable

Whole-shape prototypes, all with **identical residual 0.211**:

| prototype | us | speedup |
|---|---:|---:|
| P0 control | 10096.9 | 1.000x |
| P1 hygiene | 9317.9 | 1.084x |
| P2 + fp16 shadow | 8806.4 | 1.147x |
| P3 + fp16 apply | 8700.4 | 1.161x |

Big-GEMM microbenchmark (14336x2048x14336): TF32 736.9 TFLOP/s vs FP16 **1262.7 TFLOP/s**
(1.71x). FP16 and TF32 carry the same 11-bit effective mantissa, so the shadow buys
throughput at no precision cost -- only the exponent range narrows, and the shipped
`isfinite` guard already routes overflow to the exact fallback chain. This is why the
residual does not move at all.

## V1 mechanisms (all in `_exp061_factor_1x16384`)

1. One reused `(n-nb, nb)` block-column scratch instead of a fresh `.contiguous()` per step.
2. `_exp061_leaf_inverse`: exp-057's trsm-free inverse writing into a caller-owned buffer
   zeroed **once** instead of per block column (every non-zero region is fully overwritten
   each call), with the tree's `neg` folded into `baddbmm(alpha=-1)`.
3. `torch.mm(..., out=factor[j:, k:j])` -- the apply writes the factor directly instead of
   materializing a product and copying it in (~1.4GB of elementwise traffic removed).
4. No block-column copy on step k=0, which has no update to accumulate.
5. A persistent FP16 shadow of the factor driving the left-looking block-column GEMM and
   the inverse apply, with FP32 accumulation.

The cuSOLVER diagonal potrf is deliberately **unchanged** -- probe-01 shows every
available replacement is slower.

## Correctness

Official checker PASS on all six families. Residuals vs the budget of 20 are identical
to the incumbent on every family:

| family | incumbent | candidate | margin |
|---|---:|---:|---:|
| dense | 0.213 | 0.212 | 94x |
| spectrum | 0.107 | 0.107 | 187x |
| diagonal | 0.000122 | 0.000122 | 163934x |
| lowrank | 0.000822 | 0.000822 | 24331x |
| rowscale | 0.124 | 0.124 | 161x |
| tridiagonal | 0.0135 | 0.0135 | 1481x |

spectrum and lowrank take safety fallbacks. These are the **inherited safety cases** of
ranked incumbent #906955 (`experiments/060-two-large-followup/combined-v1-family-comparison.json`,
`inherited_safety_cases`), so candidate behaviour matches the incumbent exactly. No new
fallback family.

## Integration surface

Added: `_exp061_leaf_inverse`, `_exp061_factor_1x16384`, counters `_EXP061_16384_HITS` /
`_EXP061_16384_INVERSE_CALLS`. Modified: two lines in `custom_kernel` (the `n == 16384`
dispatch and its counter). **Untouched:** `_left_looking_large`, `_LARGE_CFG`,
`_tri_inv_recursive`, `_mxfp8_panel_update`, `_mx_quant_e4m3_blocked`,
`_factor_1x32768_blocked_inverse`, `_trsm_free_inverse_16384`,
`_exp057_tri_inv_leaf32_kernel`. `load_inline` count unchanged at 4.

1x16384 and 1x32768 dispatch through different private factor functions, so this diff does
not overlap the sibling 1x32768 worker's region at all.

## Honest ceiling for this shape

After V1 the shape is ~8792us of which the cuSOLVER diagonal is still ~5010us = **57%**.
The remaining GEMM time is ~1500us at 1262 TFLOP/s and the residual copy traffic is
~1200us, so even eliminating *all* non-diagonal work entirely would only reach ~5500us
(1.85x). Realistically the non-diagonal levers left are worth ~3-8% combined.

Reaching the campaign's 6885us stretch target requires beating cuSOLVER's ~0.33us/row on
the diagonal, which needs the entire nb x nb block factored in **one** kernel launch
(a 2048x2048 fp32 block is 16MB and cannot be shared-memory resident, so this means a
cooperative-grid or spin-barrier persistent kernel). That is the one lever probe-01 did
not close, and it is the same design the repo already failed at twice (exp029 persistent
kernels, exp050 fused 128x128 diagonal at 422ns/row -- worse than cuSOLVER's 330ns/row).
It was judged out of reach within this experiment's budget and risk envelope, and is
reported as an open lever rather than attempted and left half-finished.

## probe-03: both remaining non-diagonal levers are closed

Extended prototype ladder (same harness, same input, `--submission` = the committed v1):

| prototype | us | vs P0 | vs P3 | residual/20 | margin |
|---|---:|---:|---:|---:|---:|
| P0 control | 10081.6 | 1.000x | 0.860x | 0.211 | 95.0x |
| P1 hygiene | 9298.8 | 1.084x | 0.933x | 0.211 | 95.0x |
| P2 + fp16 shadow | 8788.1 | 1.147x | 0.987x | 0.211 | 95.0x |
| **P3 + fp16 apply (= shipped v1)** | **8674.6** | **1.162x** | **1.000x** | **0.211** | **95.0x** |
| P4 MXFP8 left-looking | 8340.9 | 1.209x | 1.040x | **12.484** | **1.6x** |
| P5 no-memset | 8799.7 | 1.146x | 0.986x | 0.211 | 95.0x |

**P4 (MXFP8) REJECTED on accuracy, not speed.** It is genuinely 4.0% faster than the
banked design, but it spends the accuracy budget to buy that: residual goes 0.211 -> 12.484
of 20.0, collapsing the margin from 95x to **1.6x**. The shipped `isfinite` guard only
catches NaN/Inf, not accuracy loss, so at 1.6x margin a harder secret-set input fails the
official checker outright with **no graceful fallback**. For comparison the 1x32768 MXFP8
path -- already the most aggressive thing in the ranked source -- ships at 5.28/20 (3.8x).
Trading a 95x margin for 1.6x to gain 4% on one of fifteen shapes is not a good trade.

Note this also confirms a persistent FP8 shadow is impossible with the shipped quantizer:
in `_mx_quant_e4m3_blocked_kernel` the scale tile index is
`(pid_m // 4) * (columns // 128) + pid_k`, so the layout depends on the **total** K. A
per-block-column quantization cannot be concatenated into the K=k layout the GEMM needs,
which is why P4 has to re-quantize the whole growing left operand every step (~1.76GB of
traffic) and why its speed gain is only 4% despite FP8's raw throughput advantage.

**P5 REJECTED on performance.** Leaving the factor uninitialized and zeroing only the
strict block-upper triangle -- the sole region the loop never writes -- is *slower* than
one big memset (8799.7us vs 8674.6us). The 7 strided slice-zero launches cost more than
the 600MB of fill they remove.

## Final state

Six prototypes and one gated candidate measured. The committed v1 is the best safe point.
Remaining time on this shape is 57% cuSOLVER diagonal potrf, which probe-01 closed for
every mechanism reachable without a single-launch custom diagonal kernel; that in turn
needs a grid-wide barrier (a 2048x2048 fp32 block is 16MB and cannot be shared-memory
resident), i.e. cooperative launch from a new CUDA extension -- excluded by the campaign's
constraints -- or a hand-rolled Triton spin barrier, which risks hanging the ranked runner
for no guaranteed gain. Reported as an open lever rather than attempted.
