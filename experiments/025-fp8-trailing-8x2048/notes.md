# Experiment 025 — FP8 trailing update at 8x2048

**Status: REJECTED — incorrect/fallback-only.** Exact baseline is ranked
submission `#882958`.

The first architecture fuses per-output-tile dynamic scaling and E4M3 casts
into `_trailing_nb`, retaining FP32 accumulation and output. Only `8x2048`
enables the new compile-time branch.

The native FP8 specialization compiled, but the paired run measured
`1854.6us -> 3612.7us` (**0.513x**) because all 18 timed calls took the ranked
safety fallback after attempting the FP8 route. One retained dense output also
failed reconstruction (`relative_residual=0.023`). The family sweep's 6/6
headline is not promotion evidence: dense, spectrum, lowrank, and rowscale each
fell back; only diagonal and tridiagonal completed on the candidate fast path.

Per the no-fallback evidence rule, the timing is invalid and the architecture
is rejected. No full grid, Popcorn test, or leaderboard submission was run.
