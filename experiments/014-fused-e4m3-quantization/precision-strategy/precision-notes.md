# Precision strategy for `batch=1, n=32768`

Scope: static/read-only-first investigation of the ranked `#878893` path. No
B200 run was made here.

## Shipped path and numerical headroom

The path uses `nb=4096` and seven left-looking panel updates. Each update:

1. takes a global `amax` of each FP32 operand;
2. scales and casts both operands to `torch.float8_e4m3fn`;
3. calls `torch._scaled_mm(..., out_dtype=torch.float32,
   use_fast_accum=True)` with scalar FP32 decode scales; and
4. subtracts the materialized FP32 product from the panel.

Diagonal updates remain TF32/FP32 `addmm_`; panel solve/output remain FP32.
Measured dense residual is `4.52 / 20` (22.6% of tolerance), while paired mean
is `52.139 ms` and full-grid mean is `51.909 ms`.

## PyTorch 2.12 / CUDA 13 feasibility

Verified against the official PyTorch `v2.12.0` schemas and CUDA source:

- Existing `torch._scaled_mm` is callable exactly as shipped. Its schema also
  has an `out=` overload, but it has no `alpha`, `beta`, or matrix-C argument;
  therefore it cannot fuse `panel -= product`.
- `torch.nn.functional.scaled_mm` and low-level `torch._scaled_mm_v2` expose
  scaling recipes. On B200 the native MXFP8 recipe is
  `ScalingType::BlockWise1x32` (numeric value `3`) with
  `SwizzleType::SWIZZLE_32_4_4` (numeric value `1`). Inputs are FP8; decode
  scales are `torch.float8_e8m0fnu`; scale buffers must use NVIDIA's tiled
  128-by-4 swizzle. Required scale elements for `[outer,K]` are
  `round_up(outer,128) * round_up(ceil(K/32),4)`.
- The v2.12 MXFP8 dispatcher explicitly calls the backend with
  `use_fast_accum=False`, regardless of the public argument. Thus MXFP8 cannot
  be used as "current GEMM plus fast accumulation and better scales."
- CUDA 13 cuBLASLt supports Blackwell 1x32/UE8M0 block scaling. The older FP32
  scale recipes (`BlockWise1x128`, `BlockWise128x128`) are explicitly restricted
  by PyTorch v2.12 to SM90 and reject B200; they are not candidates here.
- E4M3/E5M2 mixed GEMMs are accepted; E5M2-by-E5M2 is explicitly rejected.
  With exact per-tensor `amax` scaling, E5M2 provides no useful range advantage
  and loses one mantissa bit, with no throughput reason to expect a win.
- `torch.float8_e8m0fnu`, the v2 scaled-MM entry point, and both enums are in
  the 2.12 wheel. There is no high-level PyTorch quantize-and-swizzle operation
  in the submission. MXFP8 therefore needs a custom Triton/CUDA quantizer (or
  plain PyTorch plus an explicit swizzle, likely too launch-heavy).
- A direct CUDA extension can call CUDA 13 `cublasLtMatmul` and set tensorwide
  scale modes, `CUBLASLT_MATMUL_DESC_FAST_ACCUM`, `alpha=-1`, and `beta=1` with
  the panel as C/D. This is the realistic route to a fused FP8 Schur update.

## Three bounded serious variants (in order)

### P1: fused E4M3 quantization, unchanged GEMM/numerics

Jointly quantize the row-major LHS and the row-major source of the RHS, then
transpose the quantized RHS. Use either a full-graph `torch.compile` helper or,
preferably, two small Triton kernels: one fused `abs+amax` reduction for both
operands and one fused scale+cast kernel for both. Preserve scalar FP32 decode
scales, E4M3/E4M3, FP32 output, and `use_fast_accum=True`.

This removes eager FP32 temporaries and pointwise launches without spending any
of the correctness margin. It is the safest first probe, though GEMM dominates,
so a modest frontier rather than 2x is expected. Verify that the intended
compiled kernels ran; reject compile/fallback timing.

### P2: fused tensorwide E4M3 GEMM-subtract

Keep P1 quantization, but replace `_scaled_mm` plus `panel.sub_` with a minimal
CUDA/C++ cuBLASLt wrapper:

```text
D = alpha * dequant(Qlhs) @ dequant(Qrhs) + beta * C
alpha=-1, beta=1, C=D=panel, compute=CUBLAS_COMPUTE_32F,
A/B scale mode=SCALAR_32F, FAST_ACCUM=1
```

This removes the large FP32 product allocation/write/read and the subtraction
launch. Ensure the selected cuBLASLt algorithm permits C/D aliasing; if not,
use a persistent scratch output and classify it separately. This is the best
precision-preserving native-Blackwell latency variant, but it carries extension
compile/source-policy and cuBLASLt layout/heuristic risk.

### P3: MXFP8 only for diagonal products, current E4M3 panels

Use a custom quantize+swizzle kernel on `previous_row`, producing E4M3 values
and UE8M0 scales per 32 K-elements. Call `torch._scaled_mm_v2` (or the public
functional wrapper) for `previous_row @ previous_row.T`, FP32 output, then
subtract from the 4096-square diagonal tile. Leave the already-fast panel GEMMs
on tensorwide E4M3 with fast accumulation.

This targets the remaining TF32 product work while using MXFP8's accuracy where
positive-definiteness is most sensitive. The custom kernel should emit the
128-by-4 tiled scale layout directly; materializing unswizzled scales and then
permuting them is not a serious latency implementation. Probe tensorwide E4M3
for this diagonal first as a cheap upper-bound; if it passes all six families,
MXFP8 is unnecessary. MXFP8's forced non-fast accumulation and quantization
cost make converting the already-shipped panel updates a poor standalone bet.

## Rejections / risks

- Do not spend a run on E4M3/E5M2 mixes absent observed overflow/clipping; global
  `amax` already prevents it and E5M2 reduces precision.
- Do not add iterative refinement now. A Cholesky-factor correction needs a
  residual plus triangular/matrix work of order `n^3`; the current factor
  already has 4.4x tolerance headroom. Refinement is justified only if P3 makes
  more of the factorization FP8 and fails narrowly, and must then be component
  profiled before a full 32768 run.
- MXFP8 scale generation is not free and v2.12 disables fast accumulation for
  it. Treat it as an enabler for replacing TF32 diagonal work, not as a presumed
  faster replacement for shipped E4M3 panels.
- All changed arithmetic must cover dense, spectrum, low-rank, row-scaled,
  diagonal, and tridiagonal inputs. Keep the official tolerance and require
  finite lower-triangular output with positive diagonal.
