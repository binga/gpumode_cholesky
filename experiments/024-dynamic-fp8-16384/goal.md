# Goal — Experiment 024: dynamic FP8 panels at 1x16384

Baseline: exact current winner `#882958`, public `1096.0842452192236us`,
secret `1109.6451814508845us`.

Transfer the successful dynamic fused-amax E4M3 panel products from `1x32768`
to `1x16384`. Keep the 2048 block size, TF32 diagonal updates, recursive
triangular inversion, official checker, and safety fallbacks unchanged.

The fixed-scale FP8-shadow design previously lost at 16384; this experiment
tests the distinct dynamic quantization path that won at 32768. Require paired
B200 improvement, all six families with expected fallback evidence, a positive
15-shape grid, Popcorn 17/17, and at most one leaderboard submission.
