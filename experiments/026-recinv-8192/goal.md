# Goal — Experiment 026: recursive inversion at 1x8192

Baseline: exact current winner `#882958`.

Cleanly isolate recursive GEMM block triangular inversion at the winning
`nb=2048` configuration for `1x8192`. The earlier negative experiment coupled
recursive inversion with `nb=1024`, so it did not determine this transfer.

Keep TF32 panel/diagonal products, left-looking structure, checker, and safety
fallbacks unchanged. Require paired improvement, all six families, a positive
full grid, Popcorn 17/17, and at most one leaderboard submission.
