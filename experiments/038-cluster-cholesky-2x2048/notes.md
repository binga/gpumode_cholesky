# Experiment 038 result — `2x2048` boundedly exhausted

The ranked path is 1366--1374us and spends 1236.5us (90.5% of wall time) in one
vendor factorization kernel. Wall-minus-device gaps are only 11us, so launch
latency and data movement are not the primary terms. Nsight Compute was also
attempted on Modal, but initialization failed with `LibraryNotLoaded`; this
report makes no counter-based saturation claim.

## New hardware-cluster variants

| variant | architecture | candidate us | paired speedup vs ranked |
|---|---|---:|---:|
| 1 | entire factorization persistent in two-CTA clusters, rank-16 WMMA | 21744.9 | 0.063x |
| 2 | cluster factors 128-wide superpanel, custom inverse, full-grid trailing | 3248.2 | 0.423x |
| 3 | cluster factors 128-wide superpanel, cuBLAS TRSM, full-grid trailing | **2303.9** | **0.595x** |
| 4 | same TRSM design with a 64-wide superpanel | 2345.4 | 0.584x |

Every candidate was correct on the dense gate and proved the new path active
with `_CLUSTER_2048_HITS=1`; no fallback timing was accepted. Variant 2's
profile attributes 2541.9us (90.6% of device time) to the 16 cluster diagonal
calls, not to panel/trailing math. Replacing the custom inverse with cuBLAS TRSM
saves about 0.94ms, but still leaves the best architecture 1.68x slower than
the ranked vendor path. Width 64 is slightly worse than width 128.

## Bounded exhaustion evidence

Together with the repository's earlier serious attempts, six distinct correct
architectures now bound this target:

1. Experiment 015 split32/two-level route: 0.651x.
2. Experiment 028 persistent grid with global spin barriers: best 0.495x.
3. Whole-factorization hardware cluster (variant 1): 0.063x.
4. Cluster-128 plus custom inverse (variant 2): 0.423x.
5. Cluster-128 plus cuBLAS TRSM (variant 3): 0.595x.
6. Cluster-64 plus cuBLAS TRSM (variant 4): 0.584x.

Verdict: **EXHAUSTED under the six-variant campaign bound.** None beat the
ranked route, so the six-family, full-grid, Popcorn test, and leaderboard gates
were correctly not spent. Root `submission.py` remains byte-identical to
ranked submission `#888352`.
