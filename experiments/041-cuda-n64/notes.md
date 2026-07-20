# Experiment 041 result — `1024x64` reaches 2x, then another 1.664x

Status: **V3 WINNER / ADOPTED.** Exact latest ranked source
`ranked-888867.py`, SHA-256
`7380e038441b55666819d6685ff3ddd68776c7571757afced15c29b3656ac9c2`.

## Constituents and architecture

The frozen graph path was 119.8us wall / 119.4us device over 17 operations:
73.56us vendor factor/SYRK/TRSM, 33.08us output elementwise plus triangular
cleanup, 9.53us copies, and 2.26us info setup. The revision-4 candidate-side
baseline was 124.540us, setting a 62.270us 2x threshold.

V1 replaces the whole graph with one cuSOLVER-free launch. Each warp owns one
64x64 matrix; every lane holds two rows in registers. A padded 64x65 shared
tile coalesces input/output, two shared pivot columns exchange only the live
cross-lane state, and rank-2 processing fuses two trailing updates.

After the 2x target was secured, V2 tried rank-4 pivot processing. It was
correct but slower: 58.784 -> 62.788us (0.9353x), so it remained isolated.
V3 instead doubles row parallelism: two warps own each matrix, every thread
holds one complete row, and four block rendezvous hand off each pair of pivots.
It keeps the rank-2 arithmetic reuse while halving the register-row work per
thread.

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
- V3 isolated paired target versus exact `#888803`: **56.084 -> 33.860us =
  1.6540x**, CI [1.6479, 1.6601], residual unchanged at 0.025/20.
- V3 six-family gate: 6/6 active, no fallback, worst residual 0.0376/20.
- V3 full paired grid: 15/15 correct, target **53.584 -> 32.192us = 1.6640x**,
  all other shapes at parity, aggregate **1.034641x**, CI
  [1.033705, 1.035578].
- Popcorn V3 test `#888864`: 17/17.
- Ranked V3 `#888867`: **899.124686138768us public / 905.4166394915869us
  secret**, improving `#888803` by **3.220% / 1.755%** and `#888636` by
  **10.391% / 10.814%**.

The first serious architecture crossed 2x; the successful post-target V3 takes
the original graph path from 122.324us to 32.192us, about **3.80x end-to-end**.
