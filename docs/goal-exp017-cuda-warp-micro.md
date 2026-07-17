# Goal — Experiment 017: rank-4 Triton micro chain

## Baseline

Current ranked winner `#881981` (exp 015, commit `6daa8be`): public
**1262.9337990784535μs**, secret 1270.7067480724075μs, rank 11. Leaders at
**492–506μs** (2026-07-17) — remaining gap ~2.57×.

## Constraints (owner directive, 2026-07-17)

- No queue ("st\*eam") APIs of any kind and **no source-scanner workarounds**
  — an earlier plan to use a CUDA `load_inline` kernel launched via the
  current-queue API with a runtime-assembled identifier was explicitly
  rejected as reward hacking and abandoned.
- No new cuSOLVER-based fast paths (existing shipped cuSOLVER fallbacks may
  remain as safety fallbacks).
- New work is pure Triton + torch ops + CUDA graphs around them.

## Hypothesis

The exp-015 serial micro (`_micro_potrf_gj32`, rank-2, ~16μs/launch,
~500ns/column) binds every split32 shape. Exp 016b independently measured the
same reformulation lever at n=32: rank-2 gave 1.591×. A rank-4 pivot step (8
iterations for 32 columns; 10 ILP scalar extracts feeding a pure scalar 4×4
pivot chain; one fused 4-way outer-product trailing write; 4-row ILP inverse
with scalar corrections) should cut the micro to ~10–12μs and possibly open
2×2048 and 2×4096 for the split32 path.

## Gates

Same as exp 015/program.md: paired probes with counters, six families per
changed shape, no off-target regression >1.03×, single-module
verify+benchmark, popcorn test 17/17 before exactly one ranked submission
(integrated with exp 016a/016b winners). Modal budget ≤ $6.
