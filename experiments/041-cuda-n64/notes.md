# Experiment 041 result — `1024x64` reaches 2x and ranks

Status: **WINNER / ADOPTED.** Exact ranked source `ranked-888803.py`, SHA-256
`aa7a5badc577ba365f468773f9516b18b1f470809de077934d8e88c2f2317b42`.

## Constituents and architecture

The frozen graph path was 119.8us wall / 119.4us device over 17 operations:
73.56us vendor factor/SYRK/TRSM, 33.08us output elementwise plus triangular
cleanup, 9.53us copies, and 2.26us info setup. The revision-4 candidate-side
baseline was 124.540us, setting a 62.270us 2x threshold.

V1 replaces the whole graph with one cuSOLVER-free launch. Each warp owns one
64x64 matrix; every lane holds two rows in registers. A padded 64x65 shared
tile coalesces input/output, two shared pivot columns exchange only the live
cross-lane state, and rank-2 processing fuses two trailing updates.

## Promotion evidence

- Isolated paired target: **122.324 -> 53.896us = 2.2696x**, CI
  [2.2646, 2.2747], `_CUDA64_HITS=1`, residual 0.025/20.
- Integrated three-shape audit: target **122.284 -> 53.540us = 2.2836x**;
  `4096x32` and `256x128` had no regression. Overall audit remains rejected
  only because the third contract shape has not yet reached 2x.
- Six-family gate: 6/6 active custom paths, no fallback, worst residual
  0.0376/20.
- Full paired grid: 15/15 correct, target **2.2130x**, all other shapes at
  parity, geomean **1.05441x**, CI [1.05363, 1.05520].
- No cuSOLVER and no auxiliary/concurrent CUDA queue API in the new path.
- Popcorn test `#888798`: 17/17.
- Ranked `#888803`: **928.0782200444549us public / 921.3029017430186us
  secret**, improving `#888636` by **6.496% / 8.176%**.

The first serious architecture crossed 2x. Post-target rank-4 and staging
refinements remain eligible for another serial leaderboard submission only if
paired evidence shows an additional real gain.
