# Goal — Experiment 025: FP8 trailing update at 8x2048

Baseline: exact current winner `#882958`.

Transfer FP8 tensor-core arithmetic to the split32 trailing Schur update at
`8x2048`. Compute tile-local maxima and E4M3 scales inside the existing Triton
CTA, cast operands in registers, execute a native FP8 dot with FP32
accumulation, and decode the result before the ranked subtraction. This avoids
extra quantization launches and global temporaries.

Reject on compile failure, correctness failure, unexpected fallback, or paired
regression. A positive candidate must pass all six families, the full grid,
Popcorn 17/17, and at most one leaderboard submission.
