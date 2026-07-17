# Goal — Experiment 023: reciprocal solve at 60x1024

Baseline: exact current winner `#882958`, public `1096.0842452192236us`,
secret `1109.6451814508845us`.

Decouple experiment 019's reciprocal inverse-row solve from the FP16 trailing
flag. Keep `60x1024` on its ranked TF32 trailing update while replacing four
late full divides per rank-4 step with multiplies by the already-computed
reciprocal square roots.

Measure paired B200 latency twice because this route has shown material noise.
Require all six families, a positive full 15-shape grid, Popcorn test 17/17,
and at most one leaderboard submission.
