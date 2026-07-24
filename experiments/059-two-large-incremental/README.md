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
