# Experiment 046 — block-inverse GEMM design for 640x512 / 60x1024 / 8x2048

Control: ranked `#890089`. **Verdict: PARTIAL.** Ranked `#890659` =
**806.037us public** (+0.52% over 810.246). The block-inverse architecture is
**REJECTED with a quantitative floor**; what shipped is a trailing-only cuBLAS
swap on two of the three shapes. No shape reached 2x.

## The measurement that decided it

Exp 045 rejected a cuBLAS Schur rewrite because the *inner* update is K=32,
N<=96 and reached only 26 TFLOP/s. The block-inverse design deletes that update
entirely: factor the nb x nb diagonal block, build `L11^-1`, then do the whole
block column as one K=nb GEMM. Before writing any kernel, a probe measured the
GEMM shapes such a design produces (`variant-02-gemmprobe.json`):

| shape | nb | panel TF/s | trailing TF/s | design GEMM | shipped Triton |
|---|---|---|---|---|---|
| 640x512 | 64 | 60.2 | 65.7 | 1183.3us | 1104.9us |
| 640x512 | 128 | 144.3 | 131.3 | 621.1us | 1104.9us |
| 640x512 | **256** | **257.4** | **249.9** | **336.2us** | 1104.9us |
| 60x1024 | 256 | 220.8 | 255.8 | 274.3us | 772.4us |
| 8x2048 | 256 | 72.0 | 235.5 | 472.6us | 694.3us |

A batched triangular solve was measured as an exact alternative to the explicit
inverse and is hopeless: **1489-4484us**. The inverse is the only viable panel
route.

At nb=256 the GEMM budget looks like a 768.7us saving at `640x512`. It is an
illusion, and the flop accounting says why:

- level-0 panel + trailing: 8.59e10 flops -> 336us measured = **256 TFLOP/s**
- the two 256x256 diagonal blocks it leaves behind: 4.29e10 flops, which are
  *skinny at every sub-level* and run at the ~30 TFLOP/s the shipped kernels
  achieve -> **1432us**

Design total ~2028us against a shipped 1394.6us: **0.69x**. The general rule
this campaign has now confirmed five times over: *the flops you hand to cuBLAS
are the ones already running acceptably; the flops left behind are the skinny
ones that dominate the clock.*

## What shipped

Trailing-only cuBLAS: replace `_trailing_nb` with an in-place `baddbmm_` on a
strided view (ldc = n), keeping the Triton kernel for the first-touch block
because cuBLAS cannot read `src` and write `work` in one pass and
`baddbmm(src, ..., out=work)` materialises the accumulator (180us at 640x512).

| shape | ratio |
|---|---|
| 640x512 | **1.0328x** |
| 8x2048 | **1.0400x** |
| 60x1024 | 0.9320x -> **excluded** |

`60x1024` regressed with an unstable 0.63% MAD and 0.9% order spread (ratios
0.89-1.02): at batch 60 the strided in-place accumulate does not hold the
throughput the isolated GEMM probe predicted.

Full 15-shape paired grid **1.004902x CI [1.004323, 1.005481]**, 15/15 pass,
every other shape inside the 0.55% A-vs-A noise floor, residuals byte-identical
on all 15. Popcorn test 17/17.

## Why 2x is unreachable on these three shapes

`640x512` needs 697us. Its 2.86e10 useful flops would take 112us at the 256
TFLOP/s cuBLAS reaches on the fattest available GEMM, but a Cholesky cannot be
all fat GEMMs: the diagonal and panel work is inherently skinny, carries ~30%
of the flops, and runs an order of magnitude slower. Raising panel efficiency
by dropping tf32x3 to tf32 is barred by accuracy -- exp 044 v11/v12 isolated
the n=512 residual blow-up (2.59 -> 17.7 / 20) to the panels specifically.
`8x2048`'s dominant cost is its Triton diagonal micro (52.1%), still blocked
from the CUDA replacement by CUDA-graph capture (exp 044).

Closing the remaining gap needs a fused CUTLASS-style Cholesky kernel that
keeps the panel resident in shared memory across sub-blocks, which is outside
what `load_inline` plus PyTorch ops can express here.
