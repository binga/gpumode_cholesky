# Experiment 045 — cuBLAS Schur updates for the throughput-bound mid shapes

Targets: `640x512`, `60x1024`, `8x2048`. Control: ranked `#890037`
(SHA-256 `bc4536c700c95ba34f268d5a7aa6cc200ba9c403b0000ecc67abb15ec262fcb6`),
the concurrent session's 64x256 winner, which did **not** contain experiment
044.

**Verdict: PARTIAL.** Ranked `#890089` = **810.246us public**, improving
`#890037` (825.466) by **1.84%**. The win is experiment 044 carried onto the
new baseline (`640x512` 1.0987x, `60x1024` 1.1140x); the cuBLAS Schur
architecture itself is **REJECTED**. No shape reached 2x.

## Constituent diagnosis (`shapediag` on `#889994`)

| shape | wall | device | idle | dominant |
|---|---|---|---|---|
| 640x512 | 1389.6us | 1306.4 | 83.1 (6.0%) | `_panel_inner32_subtile64` 432.4 (33.1%), `_trailing_nb` 354.0 (27.1%), `_panel_apply32` 318.5 (24.4%) |
| 60x1024 | 1264.6us | 1101.9 | 162.7 (12.9%) | `_trailing_nb` 380.1 (34.5%), micro 306.4 (27.8%), `_panel_inner32` 241.8 (21.9%) |
| 8x2048 | 1828.4us | 1668.2 | 160.2 (8.8%) | `_micro_potrf_gj32` 869.3 (52.1%) |

At `640x512` the three Triton GEMM kernels are **84.6% of device time** and
achieve only **47-53 TFLOP/s**. Hardware floors: 2.86e10 flops (tf32 ~48us at
peak), 1342MB traffic (~168us). The shape is 8.3x above its memory floor, so
the arithmetic looked like the lever.

## Why the cuBLAS rewrite failed

**v1 — full torch-level blocked factorization: 0.5285x.** Profiled:

| constituent | cost | cause |
|---|---|---|
| `magma_sgemmEx` x12 | 609.9us | inner update ran fp32 (allow_tf32 unset) -> SIMT |
| `triu_tril_kernel` | 392.4us | a final `tril_()` -- redundant, the mirrored zero-fills already cover the strict upper |
| cutlass tf32 trailing x3 | 304us | the good path |
| DtoD memcpy | 216.3us | `clone()`; the Triton eager route uses first-touch |
| strided copies/zeros x30 | 398us | torch-level slicing |

**v4/v5 — surgical: keep every Triton kernel, swap only the two Schur updates
to in-place `baddbmm_` on strided views.** 0.5658x fp32, **0.8972x** with
tf32 enabled. Still net negative. The per-kernel profile explains it:

- **Trailing update: cuBLAS wins decisively.** M=N=384, K=128, batch 640 ->
  2.41e10 flops in 84.5us = **285 TFLOP/s**, against Triton's 53 TFLOP/s.
- **Inner update: cuBLAS loses.** M=480, N=width<=96, K=32 -> 1.89e9 flops in
  72.5us = **26 TFLOP/s**. The tile is too skinny to fill a tensor-core
  fragment; Triton's hand-written subtile kernel beats it (432us vs ~513us).
- **First touch costs two materialising copies.** `torch.baddbmm(src, ...,
  out=work)` with a distinct input copies input into out before the GEMM:
  2 x 179.9us at this shape. The Triton kernels get first-touch free because
  they read `src` and write `work` inside one kernel.

Net over the three effects: -312us against a -137us micro gain.

## What the ceiling looks like

Best composition of measured parts at `640x512`: micro 172 + Triton panel
apply 325 + Triton inner 432 + cuBLAS trailing ~215 + misc 110 = **~1253us**,
i.e. **1.22x** -- not 2x. The blocker is the panel work (apply + inner =
757us), which must stay **tf32x3**: exp 044 v11/v12 isolated the residual
blow-up at n=512 to the *panels* specifically (2.59 -> 17.7 / 20 when they
drop to plain tf32, while the trailing precision change was numerically
inert). Three tensor-core passes at K=32/N<=96 is the floor for that work.

The open path to 2x is a **128-wide block inverse**: factor the 128x128
diagonal block and build `L11^-1` in one CTA (exp 044 measured 57.5us/block,
batch-independent), then do the whole block column as a single K=128 bmm.
That raises the panel GEMM from K=32 to K=128 and deletes the inner update
entirely -- estimated ~863us (1.77x) at `640x512`, still short of 2x.

`8x2048` was not changed: its `_trailing_nb` already runs at ~380 TFLOP/s
(M=N=1792, K=256), so cuBLAS has nothing to add, and its dominant cost is the
Triton diagonal micro (52.1%), which is blocked from the CUDA replacement by
CUDA-graph capture exactly as in exp 044.

## Shipped

`ranked-890089.py` = `#890037` + experiment 044's micro, folded into the
experiment-042 extension so the submission keeps four nvcc invocations.
Full 15-shape paired grid **1.013544x CI [1.012948, 1.014140]**, 15/15 pass,
`640x512` 1.0987x, `60x1024` 1.1140x, every other shape inside the 0.35%
A-vs-A noise floor, residuals byte-identical. Popcorn test 17/17 in 102s.
