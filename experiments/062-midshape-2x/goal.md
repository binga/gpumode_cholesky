# Experiment 062 — 2x2048 and 2x4096

**User goal:** pick 2 medium shapes, improve the leaderboard geomean by >= 10%.

## Why these two shapes

Fresh B200 profile of the exact ranked incumbent `#907267`
(`results/exp062-inc-shapediag.json`):

| shape | wall us | dominant constituent |
|---|---:|---|
| 2x2048 | 1358.4 | vendor `getrf_wo_pivot` **1195.0us / 2 calls (91.2%)** |
| 2x4096 | 3204.0 | vendor `getrf_wo_pivot` **2779.9us / 2 calls (87.0%)** |

Both shapes run the vendor factorization **once per matrix, serially** — 597.5us
per 2048 and 1390.0us per 4096, independent of batch. That kernel is
dependent-pivot-latency bound at ~0.33us/row, so the batch dimension buys
nothing today.

## Required speedup

The score is an equal-weight geomean over 15 shapes, so k shapes must each
improve by `(1/(1-N))^(15/k)`. For N=10%, k=2: **2.204x per shape**, i.e.
2x2048 <= 616.3us and 2x4096 <= 1453.7us.

## Mechanism

Right-looking blocked factorization, panel width 128, trailing update deferred
to `nb_outer`:

1. `e62_diag128` — one resident CTA per matrix factors a whole 128x128
   diagonal block and publishes `inv(L11)`. The 32-pivot chains are
   register-resident inside one warp (broadcast-only cross-lane traffic), so
   the chain never round-trips through global memory.
2. Panel below becomes a batched GEMM against the explicit inverse.
3. Trailing update is a batched TF32 rank-k GEMM.

Because both matrices are factored by two co-resident CTAs, the serial pivot
chain is paid **once for the batch** rather than once per matrix.
