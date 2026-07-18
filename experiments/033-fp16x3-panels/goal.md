# Experiment 033 — panel precision (lever L4)

Baseline: L2-banked source (`baseline-l4.py`: ranked #883174 + 8x2048 NB=256).
Source: `docs/qr-transfer-proposal.md` lever L4.

## Hypothesis
The split32 panel dots ship at tf32x3 (3 tensor-core passes) on all 7 shapes,
chosen when "the n-scaled tolerance is tight at small n." The reconstruction gate
`20*n*eps*|A|` grows with n, and dense residuals run 100-1000x inside it, so a
cheaper panel dot should be safe at large n. Two candidates from the proposal:
1. fp16x3 (three-fp16-MMA emulated fp32): same 10-bit mantissa as tf32, ~2x the
   B200 tensor-core rate -> three fp16 products under three tf32 products.
2. plain tf32 (1-pass): the native cheap rung, no data-type change.

## Success threshold
Aggregate paired improvement across enrolled shapes with NO per-shape regression
and NO family-correctness failure (57-spec verify + 17/17 test). Judge on paired
same-process speedup, not the noise-dominated 15-shape geomean (see notes.md and
proposal §1/§3).

## Correctness constraint
Every enrolled shape must pass all six input families with comfortable headroom
(target max residual < ~10/20, i.e. >=2x margin) because the leaderboard secret
seeds differ from the probe seeds and the split32 path has only an isfinite
fallback, not a residual fallback -- a finite-but-inaccurate result would fail
the ranked correctness gate outright.

## Result
fp16x3 rejected (register spill in already-tight panel kernels). tf32 panels
shipped on the three large-n shapes (4x1024, 60x1024, 8x2048); the smaller shapes
lack accuracy headroom and keep tf32x3. See notes.md.
