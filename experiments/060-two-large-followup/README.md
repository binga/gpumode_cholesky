# Experiment 060 — two-large-shape follow-up

Frozen incumbent: ranked `#904546`, commit
`6c754f9b4ef59f2d90161dcc901e1bc1be58f0e9`, exact source SHA-256
`f8d67dce5a7a0dd68fc96e24613444970aa8c637b168bcb252cab01f2db89e5a`.

This checkpoint will combine only independently revalidated follow-up changes
for `1×16384` and `1×32768`. No candidate is promotable until it passes the
official checker on the target pair and all six input families, the full
15-shape paired grid, and the cold-build budget.

## V1 combined frontier

The two independent changes apply cleanly to disjoint shape paths:

- `1×16384`: Triton base-32 triangular-inverse leaves replace scalar PyTorch
  leaves inside experiment 057's recursive inverse.
- `1×32768`: FP16 inputs with FP32 accumulation/output accelerate the seven
  inverse panel applies from experiment 058.

Paired target evidence against exact ranked `#904546`:

| Shape | Control | Candidate | Speedup |
|---|---:|---:|---:|
| `1×16384` | 10,676.1 µs | 10,128.8 µs | **1.0541×** |
| `1×32768` | 32,979.2 µs | 31,454.0 µs | **1.0488×** |

Target geomean speedup is `1.051456×`, CI95 `[1.050341, 1.052572]`.
The full 15-shape paired grid is **1.006785×**, CI95
`[1.006202, 1.007368]`, with all shapes correct and off-target shapes at
parity.

The six-family run passed the official checker 12/12. All fallback dictionaries
exactly match the ranked incumbent: spectrum/low-rank at `16384` and
spectrum/low-rank/row-scale at `32768` retain the inherited safety paths; the
other seven cases remain on their optimized paths. A clean extension-cache
import passed in 28.66s (30.15s runner total).

Exact promotion source SHA-256:
`003fa60d1b59f6cd31656aaa069e3f02b0693d7af4f536585335e6f81d7ab175`.
