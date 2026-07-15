# Lessons from the QR Problem — What Reduced Latency at Each Size

Distilled from the `qrproblem` project (`../../kaggle/qrproblem`): a batched
Householder QR kernel on a B200, scored on the geomean of 12 shapes and
optimized across 43 hand-written versions + 68 evo autoresearch experiments.
Source of record: that project's `README.md` and `experiment_plan.md`.

## Context

- Shapes are written as `(batch, n)`. The kernel factorizes `batch` square-ish
  matrices of dimension `n`.
- Baseline was `torch.geqrf`. Two shapes dominate the geomean — `(640,512)` and
  `(60,1024)` — so most engineering effort went there.
- **The universal finding across every size: the serial Householder panel
  factorization is the bottleneck, not the trailing GEMM.**

## What worked, per size

### `(640,512)` — 640 matrices, the #1 target (1074 ms → 14.6 ms, ~73×)

Nearly all wins compounded here:

- **Blocked WY Householder** (V3): 1074 → 617 ms. First structural win over `geqrf`.
- **Full-double panel in shared memory** replacing cuSOLVER's panel (V13):
  617 → **65.6 ms** (9.4×). Biggest single hand-tuned jump.
- **Prealloc workspaces + CUDA V-builder + CUDA T-builder** (V20b/V21/V22):
  65.9 → 62.2 ms, eliminating ~496 ATen `bmm` calls.
- **`at::matmul` with no `contiguous()` copy + `baddbmm` fused trailing**
  (V24/V27): 62 → 57.9 ms.
- **Vectorized Triton FP32 panel** replacing the serial double-precision panel
  (V36, the single biggest evo win): 62 → **31.5 ms (−48%)**. Proved the
  double-precision panel was unnecessary at batch=640.
- **`nb` 32→16 sub-blocking** to relieve register spill and shift work onto the
  cuBLAS WY GEMM (V37): 31.5 → 24.8 ms.
- **Apply-only TF32 tensor-core trailing** (exploiting loose correctness gates)
  (V38): → 23.4 ms.
- **Strided trailing W1 (skip explicit contiguous copy, let cuBLAS handle the
  strided view)** (V42): → 18.6 ms.
- **Builder-fusion + apply-GEMM fusion** (exp_0051): → **14.6 ms**.

### `(60,1024)` — the #2 target (239 ms → 17.3 ms, ~14×)

- **Batch-parallel C++ blocked WY** replacing 60 sequential `at::geqrf` calls
  (V31): 239 → **54.2 ms (4.4×)**.
- **Vectorized Triton FP32 panel** (V36): 54 → **30.7 ms (−40%)**.
- **`nb`=16 register relief** (V37): 30.7 → 19.5 ms.
- **Full-TF32 trailing (W1/W2/apply all TF32)** (V41): 18.4 → **17.3 ms**.

### `(40,176)` and `(40,352)` — medium shapes

Never dominant, but improved steadily:

- **cuSOLVER batched / C++ blocked loop** removing Python dispatch overhead
  (V5/V9): `(40,176)` 22 → 6.8 ms.
- **FP32 panel + CUDA V/T builders + prealloc** (V18/V20b/V21): `(40,176)`
  → ~4.25 ms, `(40,352)` → ~10.5 ms.
- **Medium-path trailing fusion** (strided `at::matmul` for W1 + in-place
  `baddbmm_`) (exp_0056): held ~10.9 ms with less memory traffic. Now
  register-tile/launch-bound, so returns flattened.

### `(20,32)` — smallest shape

- **Fused single CUDA kernel for n≤32** (V11): 324 → 131 µs (2.6×).
- **Shared-memory CuTe DSL QR32 kernel** (V16/V17): → ~84 µs. Effectively free
  by the end (~0.15 µs class in later rows); negligible to the geomean.

### `(8,2048)` and `(2,4096)` — large, low-batch shapes (the remaining wall)

These stayed **panel-bound on sequential cuSOLVER `geqrf`**, and almost every
promising fix was **non-submittable** on Popcorn (its anti-cheat disqualifies
any work on a non-default CUDA stream):

- **What worked but was banned:** grid-sync cooperative panel (V39, −47% on
  `(8,2048)`), thread-block clusters (−31%), software global barriers (−42%),
  CholeskyQR2 + Householder reconstruction (V40, −53%).
- **What actually shipped:** for `(8,2048)`, a **single-pass FP16 tensor-core
  blocked-Householder trailing** (V38) trimmed 76.8 → 71.8 ms; and a
  **submittable CholeskyQR2 with GEMM-based triangular solve** (exp_0054, no
  cuSOLVER `potrf`/TRSM so it stays on the current stream) cut `(8,2048)` to
  ~38 ms. `(2,4096)` barely moved (~52 → 49 ms) — at batch=2 it's
  latency/combine-bound, not sync-bound.

## Cross-cutting techniques that generalized

1. **Move the block loop off Python / out of the panel** — C++/CUDA loop,
   prealloc workspaces, CUDA V/T builders.
2. **Attack the panel, not the GEMM** — vectorized Triton FP32 panel + `nb`=16
   register relief delivered the largest wins.
3. **Exploit the loose correctness gates** — the checker only gates factor
   residual and orthogonality (with 4–10× / 280–610× margins), so FP16/TF32
   tensor-core math is "free" on the apply/trailing steps. FP16 was the sweet
   spot (BF16 failed the factor gate; TF32 tied).
4. **Test submittability on the real target early** — Modal can't detect
   Popcorn's stream anti-cheat, so several validated wins (cooperative launch,
   clusters, batched `torch.linalg`) died only on the real target.

## Best result

**V43** (`solutions/v43_cholqr2048_medium_fusion_merge/`, also root
`submission.py`) — ≈**10,901 µs** geomean, 26/26 correct on Modal. Merges the
CholeskyQR2 `(8,2048)` win with medium-path trailing fusion on top of the
`(640,512)`/`(60,1024)` builder-fusion spine — ~27% reduction from the V31
hand-tuned baseline, ~42% on evo's internal metric.

Caveat: V43 was **never resubmitted to live Popcorn** (deadline passed
2026-06-30); the last number confirmed on the real leaderboard was exp_0051 at
~11,003 µs. V43's 10,901 µs is the `bench_grader.py --twelve` grader-methodology
proxy.
