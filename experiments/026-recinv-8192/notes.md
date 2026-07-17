# Experiment 026 — recursive inversion at 1x8192

**Status: REJECTED — slower at the winning block size.** Exact baseline is
ranked submission `#882958`.

The candidate changes only `rec_inv=False -> True` for the existing
left-looking TF32 `1x8192, nb=2048` route.

The paired B200 probe passed all six families but measured `5843.8us ->
6126.0us` (**0.954x**). Recursive GEMM inversion does not amortize at 8192,
even with the winning block size held fixed. No full grid, Popcorn test, or
leaderboard submission was run.
