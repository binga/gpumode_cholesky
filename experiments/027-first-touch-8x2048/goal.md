# Goal — Experiment 027: first-touch eager at 8x2048

Baseline: exact current winner `#882958`.

Transfer first-touch eager execution from `640x512`/`60x1024` to `8x2048`.
Keep every kernel and precision specialization unchanged, but read the live
input on the first launches and write a fresh output, eliminating graph
copy-in and clone-out while giving up graph replay.

Require paired improvement, six families, a positive full grid, Popcorn
17/17, and at most one leaderboard submission.
