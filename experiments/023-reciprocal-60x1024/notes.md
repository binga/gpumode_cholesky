# Experiment 023 — reciprocal solve at 60x1024

**Status: REJECTED — inconsistent/no net paired evidence.** Exact baseline is
ranked submission `#882958`.

The candidate changes only `RECIPROCAL_SOLVE` for `60x1024`; its TF32 trailing
precision, eager schedule, panel kernels, checker, and fallback behavior remain
unchanged.

Two independent paired probes passed all six families each but disagreed:

| run | baseline | candidate | speedup |
|---|---:|---:|---:|
| r1 | 1983.4us | 1969.7us | **1.007x** |
| r2 | 1626.5us | 1637.0us | **0.994x** |

The rewrite is below the route's run-to-run noise and does not establish a
stable improvement. Per the promotion gate, no full grid, Popcorn test, or
leaderboard submission was run. Root remains exact `#882958`.
