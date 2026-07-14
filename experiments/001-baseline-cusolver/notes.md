# Experiment 001 — cuSOLVER baseline

**Hypothesis:** the stock `torch.linalg.cholesky_ex` (cuSOLVER) is correct and a
reasonable starting point on B200.

**Change:** none beyond the one-liner `custom_kernel`.

**Results (B200):**
- Correctness: popcorn test 17/17; Modal verify 13/13 across all families.
- Ranked geomean ≈ **2080μs** (per-shape means in `../../results/baseline-benchmark.json`).
- Board reference at submission time: leaders ~1924μs (xuan9938), ~2041μs (msaroufim).

**Ranked submission:** `#876988` (`done`).

**Verdict:** baseline. Superseded by 002. Kept as the reference point.
