# Experiment 053 — recursive-inverse base and split diagonal

Status: **exhausted; no promotion**.

This experiment targeted the measured `1x16384` bottleneck in the frozen ranked
incumbent (`#890798`, source SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`).
B200 decomposition attributed 35.1% of device time to the eight 2048 POTRF
calls and 30.3% to triangular solves/inversion.

## Results

| Variant | Change | Paired B200 result | Disposition |
| --- | --- | ---: | --- |
| V1 | recursive inverse base 512 → 1024 | `1.027036x`, CI `[1.023880, 1.030201]` | dense-only diagnostic |
| V2 | recursive inverse base 512 → 2048 | `1.035934x`, CI `[1.034954, 1.036915]` | rejected: spectrum and low-rank fell back |
| V3 | replace each 2048 POTRF with batch-1 split32 | `0.770312x`, CI `[0.769973, 0.770757]` | rejected: 22.97% slower |

V2 retained identical dense-family residual (`0.213`) and made seven direct
triangular solves, but the six-family no-fallback gate passed only dense,
diagonal, row-scale, and tridiagonal. Spectrum and low-rank invoked the shipped
fallback, so the optimized backend did not handle every required family.

V3 reached all eight intended diagonal blocks and passed correctness, but batch
1 underutilization made the custom split32 path substantially slower than
cuSOLVER POTRF. It was not combined with V2.

No exact standalone candidate, full-grid benchmark, clean-build gate, or
Popcorn submission was warranted. The next experiment should tune the
left-looking panel width, which jointly changes POTRF count/size, inverse/TRSM
work, and panel-update balance.
