# Experiment 055 — exact 32768 decomposition

Profile the frozen ranked incumbent at `batch=1,n=32768` on B200, normalize
host wall, device time, idle, launch counts, kernel families, and dominant
constituents, then select one bounded candidate from the evidence.

The current route is left-looking with `nb=4096`, MXFP8 panel updates, TF32
diagonal updates, and recursive triangular inversion. Historical measurements
predate the current MXFP8 and integration stack and are not sufficient for a
new promotion decision.

Baseline: Popcorn `#890798`, commit `f90ef909`, source SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
