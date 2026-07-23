# Experiment 054 — large left-looking panel width

Optimize the frozen ranked incumbent's `batch=1,n=16384` route by changing
only the left-looking panel width (`nb`). The B200 profile attributes 65.4% of
device time to diagonal POTRF plus triangular inversion/solve, whose call count
and matrix size are controlled by this parameter.

## Bounds

- Baseline: Popcorn `#890798`, commit `f90ef909`, exact source SHA-256
  `fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
- First variants: `nb=1024` and `nb=4096`; test `512` or `8192` only if the
  first pair identifies a monotone direction worth resolving.
- Paired B200 dense measurement first. A candidate must be correct, active,
  fallback-free, and have a CI excluding 1.0 to advance.
- Six-family, full-grid, clean-build, and Popcorn gates apply only to a
  promotion-sized robust winner.

No benchmark replay, memoization, pointer reuse, or checker bypass is in scope.
