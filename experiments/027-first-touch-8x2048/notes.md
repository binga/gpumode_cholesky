# Experiment 027 — first-touch eager at 8x2048

**Status: REJECTED — graph replay is essential.** Exact baseline is ranked
submission `#882958`.

The candidate changes only the `8x2048` execution mode from graph to eager;
factorization kernels, precision, and fallback behavior are unchanged.

The paired B200 probe passed all six families but regressed `1906.7us ->
5678.1us` (**0.336x**). Removing copy-in/clone-out cannot compensate for
replaying 190+ launch instances eagerly. Because `4x1024` has less buffer
traffic to save, the same transfer has still lower expected ROI and was not
run. No full grid, Popcorn test, or leaderboard submission was used.
