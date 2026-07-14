# Experiment 003 — CUDA warp/block-per-matrix Cholesky for n=64 and n=128

**Status: REJECTED** — no custom CUDA design beat cuSOLVER at n=64/128, so nothing
was submitted. Root `submission.py` remains experiment 002 (Triton n=32 + cuSOLVER).

## Hypothesis
cuSOLVER's batched `potrf` leaves headroom at n=64 (110μs ranked) and n=128
(152μs ranked); a custom one-CTA/one-warp-per-matrix Cholesky (register-/shared-
blocked columns, right-looking) could reach ~0.5× and push the geomean toward the
board leader (~1924μs, est. ~1810μs if both shapes halved).

## What changed
- `submission.py`: kept the Triton n=32 kernel; added a CUDA kernel for n∈{64,128}
  compiled on the runner via `torch.utils.cpp_extension.load_inline`, cached at
  module scope, wrapped in try/except → cuSOLVER fallback so a compile failure can
  never break correctness. Dispatch: n=32→Triton, n∈{64,128}→CUDA, else cuSOLVER.
- `scripts/modal_verify.py`: switched the Modal image from plain pip-torch
  (no nvcc) to **`nvidia/cuda:13.0.0-devel-ubuntu24.04` + `pip install torch numpy
  ninja`** so `load_inline` can compile on the B200 sandbox. Two required fixes:
  1. `.entrypoint([])` to clear the nvidia image's default entrypoint.
  2. **`ninja`** — without it `load_inline` raised `verify_ninja_availability()`
     and silently fell back to cuSOLVER (caught early via a `custom_cuda_loaded`
     diagnostic + captured `_CUDA_LOAD_ERROR`; the n=64/128 residuals were
     byte-identical to cuSOLVER, which is what exposed the silent fallback).
- `scripts/_gpu_runner.py`: prints `custom_cuda_loaded=<bool>` and any
  `_CUDA_LOAD_ERROR` so a failed compile is never mistaken for a passing kernel.

## Designs tried (both correct, both too slow)
1. **Block-per-matrix, N threads, matrix in shared memory, `__syncthreads`.**
   Thread t owns column t; right-looking (scale col k, rank-1 update trailing
   lower triangle). 3 block barriers per column step ⇒ 3N `__syncthreads`.
2. **Warp-per-matrix, 32 lanes, matrix in shared memory, `__syncwarp`.** Same
   algorithm, cheap intra-warp ordering instead of block barriers, strided rows.

## Results (Modal B200, L2-clear method) — correctness + speed
Correctness: **Modal verify 19/19** across all families with the CUDA kernel
active (`custom_cuda_loaded=True`); n=64 dense residual 0.021, n=128 dense 0.013,
n=128 spectrum 0.011 — all ~1000× inside the tolerance of 20.

Speed (per-shape mean μs), lower is better:

| shape | cuSOLVER | Triton (exp 002 style) | CUDA block (128-thr) | CUDA warp (32-lane) |
|---|---|---|---|---|
| 1024×64  | **135.7** | 152 | 205 | 214 |
| 256×128  | **201.5** | 429 | 413 | 693 |

**cuSOLVER wins every variant.** The block version is sync-bound (3N
`__syncthreads` with limited occupancy at 64KB shared for n=128 → ~3 blocks/SM);
the warp version has too little per-matrix parallelism (32 lanes) and a badly
load-imbalanced rank-1 update inner loop (row i does O(i) work), so n=128 blows
up to ~693μs.

## Verdict
**Rejected.** A naive right-looking batched Cholesky cannot beat cuSOLVER's tuned
`potrf` at n=64/128 on B200. Adopting this would *regress* the geomean, so — per
the guardrail — no ranked submission was made. Current best stays **#877091**
(exp 002, ~2062μs). Ranked quota used this session: **0** (still 2 of 3 remaining).

## Modal spend
~5 Modal B200 runs this experiment (1 heavy image build ~110s incl. torch/CUDA
download, then verify/benchmark ~40–95s each) ≈ ~8–10 min B200 wall ≈ **~$1–2**.

## What would actually be needed to win n=64/128 (future work)
cuSOLVER is near-optimal here; beating it needs a genuinely sophisticated kernel,
not a naive one:
- **Blocked / recursive right-looking** (e.g. 16- or 32-wide panels) so the trailing
  update is a batched GEMM-like op that saturates the SMs, with the panel factorized
  in registers — amortizes sync and fixes load balance.
- **Tensor-core (bf16/tf32) trailing updates** with FP32 accumulation; the checker's
  tolerance (20·n·eps·‖A‖₁) has ample headroom (residuals are ~1000× inside), so a
  mixed-precision Schur update is likely admissible — verify the FP32 gate per family.
- **Multiple matrices per block** to raise occupancy and hide the pivot-step latency.
- Consider n=128 as several 64-blocks (2×2 block Cholesky) to cap shared-mem usage
  and lift occupancy above 3 blocks/SM.
This is a multi-hour kernel-engineering effort with uncertain payoff; deferred.
