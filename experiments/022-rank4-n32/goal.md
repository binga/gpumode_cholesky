# Goal — Experiment 022: standalone rank-4 n=32 kernel

Baseline: exact ranked experiment 021, submission `#882958`, public
`1096.0842452192236us`, secret `1109.6451814508845us`.

Transfer experiment 017's rank-4 pivot chain to the standalone `4096x32`
kernel. Replace the ranked rank-2 kernel's sixteen serial two-column steps
with eight four-column steps while retaining one warp per matrix, FP32
arithmetic, lower-triangular output, and the official checker.

Run free gates, paired same-process B200 timing, all six input families at
`4096x32`, and the full 15-shape grid. Proceed to Popcorn test and one
leaderboard submission only if the aggregate improves.
