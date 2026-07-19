# Goal — Experiment 038: cluster Cholesky for `2x2048`

## Frozen baseline and target

- Ranked source: `#888352`, commit `f84e1de`, public geomean 1052.594us,
  secret geomean 1140.758us.
- Exact paired baseline snapshot: `audit/baseline-submission.py` and
  `experiments/038-cluster-cholesky-2x2048/baseline.py`.
- Current `2x2048`: 1366us wall / 1355us device. Target: **at most 683us**
  paired mean, with the candidate backend proven active and no fallback timing.
- Off-target full-grid regression limit: 3%; aggregate geomean must improve.

## Constituent diagnosis

`shapediag` on the exact ranked path attributes 1236.5us (90.5% of wall) to a
single `getrf_wo_pivot_params_` factorization kernel, about 90us to output and
housekeeping kernels, and only 11us (0.8%) to wall-minus-device gaps. Analytic
floors are 8.7us of compulsory FP32 traffic and 71.6us of FP32 arithmetic, so
the dominant term is neither launch nor data movement. Nsight Compute was
attempted on Modal and failed before collection with the platform/driver error
`LibraryNotLoaded`; no counter-based saturation claim is made.

## Falsifiable architecture

Replace the vendor factorization for this shape with a cuSOLVER-free,
single-launch, two-CTA-per-matrix CUDA cluster kernel. Each cluster keeps the
blocked factorization on two co-scheduled SMs, uses hardware `cluster.sync()`
instead of Experiment 028's global-memory spin barriers, and uses warp-level
TF32 MMA for the trailing rank-16 Schur updates with FP32 accumulation. A
separate lower-triangle output kernel is allowed; concurrent or auxiliary CUDA
streams are not.

Prediction: hardware cluster barriers remove the ~2.75ms spin-barrier floor of
Experiment 028, while tensor-core trailing updates make a 683us end-to-end path
arithmetically plausible. Kill the mechanism if an active, correct two-CTA
cluster cannot beat the ranked path on `2x2048`; do not transfer it to the two
4096 shapes in that case.

## Bounded ladder

1. Compile/launch gate: rank-16, two-CTA cluster, WMMA TF32 trailing update.
2. Tile mapping and warp-count tuning without changing the algorithm.
3. Rank-32 panels to halve cluster barriers if register/shared-memory evidence
   permits.
4. FP16x3 or TF32x2 trailing precision only if the rank-16 path is throughput-
   limited and the six-family margin permits it.
5. Four-CTA cluster only after the two-CTA mechanism beats the baseline.
6. Stop after six genuinely distinct measured variants.

## Gates

- Free: syntax/import structure, source-policy scan, JSON parse, `git diff
  --check`, and audit-contract validation.
- B200: paired rotating-input timing against the frozen source; dense plus
  spectrum, diagonal, lowrank, rowscale, and tridiagonal; finite positive
  diagonal; official residual threshold unchanged.
- Promotion: full 15-shape paired grid, Popcorn test 17/17, then one ranked
  submission at a time. After the shape reaches 2x, every further verified
  aggregate gain remains eligible for another serial submission.

Modal cost guardrail: about $15 for this shape, with cheap target-only probes
before any full grid.
