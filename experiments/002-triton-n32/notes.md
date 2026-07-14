# Experiment 002 — Triton batched Cholesky for n=32

**Hypothesis:** the small-`n`/high-batch shapes are launch/overhead-bound; a custom
batched Triton kernel (one program per matrix) can beat cuSOLVER's per-launch overhead.

**Change:** `custom_kernel` dispatches `n == 32` to a Triton kernel (`num_warps=1`, so
per-column reductions become in-warp shuffles instead of shared-mem syncs); all other
shapes stay on `torch.linalg.cholesky_ex`. Guarded by try/except → cuSOLVER fallback.

**Results (B200):**
- **4096×32: 113μs → 63.7μs (−44%).** All 14 other shapes unchanged (cuSOLVER).
- Correctness: popcorn test 17/17; Modal verify 19/19 across all families (worst n=32
  scaled residual 0.082, tolerance 20 — well inside spec).
- Ranked run geomean ≈ **2062μs** vs baseline 2080μs. Same-environment (Modal, L2-clear)
  the n=32 change moves the geomean ~3.9%; the small absolute delta is because unrelated
  cuSOLVER shapes drifted on environment noise this session.

**Ranked submission:** `#877091` (`done`, 17/17). Test: `#877088`.

**Verdict:** **adopted — current best** (root `submission.py`).

**Modal spend:** part of ~$1–2 across the session.

**Next:** Triton loses at n=64/128 (register spill, sync cost). Beating cuSOLVER there
needs a warp-per-matrix CUDA kernel (nvcc) → experiment 003.
