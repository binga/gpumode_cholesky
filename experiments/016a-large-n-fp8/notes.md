# Experiment 016a — large single-matrix overhaul (1×8192, 1×16384, 1×32768)

**FRONTIER — ready for parent integration (no ranked submission from here).**
Baseline: exact ranked `#881981` (root `submission.py`, SHA-256
`0400f06ad250b3bef240eb5cebae10ca9c045cc227924e06185be966e7310bb5`).

## Final candidate (`candidate-final.py`)

One generalized left-looking path `_left_looking_large` dispatched for
batch==1, n ∈ {8192, 16384, 32768}, falling through to the shipped dispatch
on any failure (counters `_LARGE_FP8_HITS/_FALLBACKS/_ERROR`):

- **1×8192** (was pure cuSOLVER): left-looking TF32, nb=2048 → **1.138×**
  paired (6572→5773μs). FP8 panels lost to TF32 here (1.070×): the frontier
  is too narrow to amortize quantization.
- **1×16384**: shipped TF32 structure + **recursive block triangular
  inversion** (TF32 combine GEMMs, 512 base) replacing the panel TRSM →
  **1.055×** (15928→15095μs).
- **1×32768**: shipped FP8-panel structure + recursive inversion replacing
  the per-panel `solve_triangular(diag.T, identity)` → **1.028×**
  (47568→46266μs).

Paired-final residuals: 0.19/20, 0.213/20, 4.51/20 (dense). Fast-path
counters exact; expected safety fallbacks only on spectrum/lowrank.
Single-module gates: `modal_verify.py verify` **57/57**, benchmark **15/15**
at geomean **1323.6μs** (8192: 5847μs, 16384: 15202μs, 32768: 46146μs; all
other shapes match exp-015 within noise — no graph interaction).

## Rejected on measurement (rounds r1–r2, 6 variants)

| variant | result |
|---|---|
| v2: FP8 panels at 8192 | 1.070× < v1's 1.118× TF32 — rejected |
| v4: 32768 full stack (FP8 shadow of L + fixed scale from max diag + FP8 diagonal update) | **0.996×** — contiguous fp8 operand copies + shadow cast ate the amax/requantize savings |
| v5: 16384 FP8-shadow stack | **0.972×** — rejected |
| v6: 8192 nb=1024 + rec-inv | 0.976× — rejected (nb=2048 stands) |

Ladder items 3+4+5 (FP8 diag update, fixed scale, FP8 shadow) are closed
with direct negative evidence as implemented; a Triton-fused shadow update
that avoids the `.contiguous()` copies might reopen them.

## Cost

3 Modal probe runs + 1 combined final/verify/benchmark chain ≈ **$4–5**.
No popcorn usage.

## Expected integration effect

1.138×/1.055×/1.028× on three of 15 geomean terms → ×(1.2126)^(1/15) ≈
**1.3% geomean** (~1263 → ~1247μs ranked, all else equal).

Artifacts: `probe-r1.json`, `probe-r2.json`, `probe-final.json`,
`verify-final-single.json`, `benchmark-final-single.json`,
`baseline-exp015.py`, candidates v1–v6 + `candidate-final.py`, adapted
multi-candidate `modal_probe.py`/`probe_runner.py`.
