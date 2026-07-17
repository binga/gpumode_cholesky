# Goal — Experiment 021: transfer panel-inner subtiling

Baseline: exact ranked experiment 020, submission `#882927`, public
`1120.2139424233us`, secret `1126.4634299045994us`.

Transfer experiment 020's verified 64x64 `_panel_inner32_subtile64` kernel from
`4x1024` and `8x2048` to the four remaining split32 shapes: `64x256`,
`16x512`, `640x512`, and `60x1024`.

The candidate changes only the compile-time dispatch set. The kernel,
arithmetic precision, factorization schedule, official checker, and fallback
behavior remain unchanged. Measure all four shapes against the frozen baseline
in one B200 process, cover all six input families for every changed shape, and
retain only individually positive routes. A finalist must pass the full
15-shape grid with no material off-target regression before Popcorn test 17/17
and exactly one leaderboard submission.

Classification follows `program.md`: positive but sub-2x routes are
`FRONTIER`; slower, incorrect, fallback-only, or noisy routes are `REJECTED`.
