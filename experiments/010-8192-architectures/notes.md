# Experiment 010 — exact `1 x 8192` architectural search

Status: **REJECTED — BOUNDED_EXHAUSTION**

No candidate reached the strict paired 2.00x gate. Root `submission.py` remains
the byte-identical experiment-009 / ranked `#878273` source. No Popcorn test or
leaderboard submission was launched.

## Baseline and measurement contract

- Baseline commit: `4b4d557` (`#878273`, public geomean `1500.704 us`).
- Exact baseline source SHA-256:
  `39261a153a0df9826b0e6c8aa1b3f948179f6445da8be2a74a8ab5040ab7adf8`.
  It matches `experiments/009-combined-shape-frontiers/submission.py` byte for
  byte and remained unchanged throughout this experiment.
- Recorded retained-output latency: `6435.588 us`; recorded target
  `<=3217.794 us`.
- Fresh paired controls were `6395.663`, `6397.835`, and `6384.363 us`, giving
  strict process-local targets of `3197.831`, `3198.918`, and `3192.182 us`.
- Every timing rotated across four `1 x 8192 x 8192` inputs, retained all
  outputs, cleared L2 between samples, and checked retained input/output pairs.
- Compiled probes reported `compiled_active=true`; no fallback timing was
  accepted.

## Architectural results

Block sizes within V3 are calibration points, not separate architectures.

| ID | architecture | best mean us | speedup vs paired | dense correctness / margin | verdict |
|---|---|---:|---:|---|---|
| V1 | direct legacy `cusolverDnSpotrf` + reusable workspace | 12891.725 | 0.496x | pass, >25k× | reject: 2.02× slower |
| V2 | direct 64-bit expert `cusolverDnXpotrf` | 12892.701 | 0.496x | pass, >25k× | reject: same slow kernel; Xpotrf has only default algorithm |
| V3 | host-fused one-level cuSOLVER diagonal + TRSM + lower-only TF32 SYRK | 12192.842 (`nb=4096`) | 0.525x | pass, ~196× | reject: best compiled lower-only path is 1.91× slower |
| V4 | graph-captured compiled V3 pipeline + owned output | 254.893 apparent | invalid | **fail**, tolerance fraction 118–121 | reject once: library work did not replay correctly; no retry |
| V5 | Triton custom diagonal/panel + triangular TF32 update tiles | 15258.520 | 0.419x | pass, ~39× | reject: launch/panel dominated |
| V6 | Triton triangular BF16 update proxy, FP32 diagonal/panel | 14883.872 | 0.430x | pass, ~13× | reject: already 4.65× above target before any correction |
| V7 | two-level batch-one cuSOLVER diagonal pivots + lower-only TF32 SYRK | 42275.865 | 0.151x | pass, ~196× | reject: batch-one API is pathological at large n |

The V7 component control, direct full-size `cusolverDnSpotrfBatched`, was also
correct but took `95331.338 us`. This disproves the hypothesis that PyTorch's
fast `6.4 ms` route is simply a public batch-one potrf call.

V6 was stopped before iterative correction: its uncorrected factor already
used only 7.6% of tolerance but took `14.884 ms`. Any exact residual, triangular
solves, and correction update add work, so the corrected path cannot approach
the `3.20 ms` promotion bound. CUDA 13's standard lower-only SYRK surface also
does not expose FP8/MXFP8 inputs; implementing that rung would require a new
tcgen05/CUTLASS factor-panel core, not a guarded extension of this slow path.

## Shipped device profile

`shipped-device-profile.json` measured:

- clone only: `88.965 us` mean;
- shipped 3-D call: `6405.019 us`;
- direct PyTorch 2-D call: `6415.950 us`;
- direct PyTorch 3-D call: `6401.129 us`.

One profiled shipped call spent `5787.212 us` in
`aten::linalg_cholesky_ex`. Four internal `getrf_wo_pivot` kernels accounted
for `3811.643 us`; three SYRK/HERK kernels accounted for `1965.648 us`; copy
and lower-triangular cleanup accounted for roughly `521 us`. Therefore a
`<=3.20 ms` result needs a genuinely new persistent/cluster factor-panel core
plus faster updates. Graph wrapping, lower-only vendor SYRK, or a correction
stage cannot deliver the required reduction.

## Gates and artifacts

- Local property gate: `10/10` passed with torch 2.13 CPU.
- Python syntax: passed.
- Root snapshot: byte-identical to exp 009.
- Candidate source scan: clean for the forbidden queue literal/API family.
- `git diff --check`: passed.
- Raw evidence: `stage1-direct-and-syrk.json`,
  `stage2-graph-triton-precision.json`, `stage3-batched-components.json`, and
  `shipped-device-profile.json`.
- Candidate source SHA-256:
  `a4f9c9701733f0ad8b0573767597bf21f62ce3d5222d9c3a83c4143e24e07bef`.
- Promotion-only gates (six-family sweep, full 15-shape comparison, Popcorn
  17/17) were not run because no candidate passed the first 2.00x timing gate.

Approximate paid work: four bounded B200 jobs (three paired probes and one
profile), about 3–4 minutes of sandbox/GPU wall time including compilation.

## Next credible axis

A future attempt must replace the dominant `getrf_wo_pivot` panel/factor work:
a cooperative persistent or thread-block-cluster Cholesky kernel using SM100
tcgen05/TMA, with panel factorization and triangular update coordinated in one
device-resident pipeline. It should first demonstrate a sub-`2.0 ms` factor and
panel phase, leaving about `1.2 ms` for lower-only Schur updates and output
cleanup. A block-size sweep or another library wrapper is not credible.
