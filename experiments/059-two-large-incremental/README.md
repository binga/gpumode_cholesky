# Experiment 059 — first two-large-shape frontier

Frozen baseline: ranked `#890798`, commit
`f358e879b1287ca50d29115ad9a403c6bd10a69d`, source SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.

## V1 result

The preserved experiment-052 breadth-first blocked triangular inverse was
restricted to the two selected shapes and remeasured as an exact source:

| Shape | Control | Candidate | Speedup |
|---|---:|---:|---:|
| `1×16384` | 15,209.6 µs | 14,333.8 µs | `1.0611×` |
| `1×32768` | 42,331.5 µs | 32,733.0 µs | `1.2932×` |

The two-shape paired geomean speedup was `1.1714×`, CI95
`[1.1710,1.1719]`, with both dense outputs passing the official checker and no
new fallback. If the other thirteen shapes were at parity, the implied full
grid would be about `1.0213×`, or `2.09%` lower latency.

All twelve six-family outputs passed the official checker. Spectrum/low-rank
at `16384` and spectrum/low-rank/row-scale at `32768` entered the existing
large-shape safety chain; exact-baseline control evidence is therefore required
before promotion.

## V2 result

V2 qualified each blocked inverse by `max(abs(I - L @ inv(L))) <= 0.05` and
used a stable triangular panel solve otherwise. The same final family safety
chains still activated, so V2 did not repair the robustness boundary and was
rejected without a dense timing run.

The campaign continues with the stronger independent shape frontiers in
experiments 057 and 058. No Popcorn job was spent from this source.

## V3/V4 combined frontier

Experiment 057 V2 (scalar-leaf trsm-free inverse) and experiment 058 V1
(batched 256-wide inverse leaves) were combined without overlapping dispatch:

| Shape | Control | Candidate | Speedup |
|---|---:|---:|---:|
| `1×16384` | 15,223.3 µs | 10,738.4 µs | `1.4177×` |
| `1×32768` | 42,771.2 µs | 33,164.1 µs | `1.2897×` |

The exact V4 source also merges the unchanged CUDA32/64/128 extension sources,
the compile repair previously proven by experiment 055. Its fresh
`TORCH_EXTENSIONS_DIR` import took `65.27 s` (`71.51 s` runner total), below
the `288 s` promotion budget.

Exact V4 full-grid paired result: `1.039915×`, CI95
`[1.039570,1.040259]`, all 15 shapes correct. This is `3.838%` lower aggregate
latency. The six-family exact source passed the official checker 12/12; its
five safety-path families exactly match the incumbent controls, and both dense
leaderboard paths reached the intended optimized counters with no fallback.

Exact candidate SHA-256:
`f8d67dce5a7a0dd68fc96e24613444970aa8c637b168bcb252cab01f2db89e5a`.

## Ranked checkpoint

Popcorn test `#904530` passed 17/17. The exact same source was ranked as
`#904546`:

| Split | `#890798` | `#904546` | Improvement |
|---|---:|---:|---:|
| public | 801.9772 µs | 764.8768 µs | **4.6261%** |
| secret | 847.8362 µs | 785.8614 µs | **7.3098%** |

This is an adopted partial win. It becomes the next incumbent while the same
two-shape campaign continues toward the requested cumulative 10% geomean
reduction.
