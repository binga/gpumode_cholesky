# Goal — Experiment 028: persistent dual-matrix factorization

Baseline: exact current winner `#882958`, public
`1096.0842452192236us`, secret `1109.6451814508845us`, repository commit
`1fc6ac258a80b2c8e2a086823c20edca63b31ab3`, and source SHA-256
`3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`.

Targets: ranked shapes `2x2048` and `2x4096`. The shipped route executes two
single-matrix cuSOLVER factorizations sequentially. Build a cuSOLVER-free,
single-factorization-launch Triton path that exposes both independent matrices
to the GPU together and removes the host-serialized diagonal/panel/trailing
launch chain.

## Hypothesis and primary variant

A fixed resident grid can process both matrices with a device-side phased
scheduler. One worker per matrix factors each 32-column diagonal microblock;
resident workers cooperatively apply the panel and rank-128 Schur updates; GPU
atomics provide phase barriers. With both dependency chains resident, work from
one matrix can occupy the GPU while the other advances its diagonal chain.

Variant 1 uses 16 persistent workers per matrix, rank-4 FP32 diagonal microsteps,
TF32x3 panels, TF32 trailing products, 128-column outer panels, and 128x128
trailing tiles. It must report backend hits and zero unexpected fallbacks.

## Gates

- `WINNER`: both targets pass all six official input families and paired mean
  latency is at least `2.00x` faster than the exact baseline on each target.
- `FRONTIER`: both targets are correct and aggregate target geomean improves,
  but one or both speedups are below `2.00x`.
- `REJECTED`: incorrect, slower in aggregate, compiler/runtime failure, or any
  timed fallback.
- Preserve the official reconstruction tolerance, finite lower-triangular
  factors, and positive diagonals. Do not weaken the checker.
- No cuSOLVER call may occur on the candidate fast path. No non-default queue,
  queue API, source-scan workaround, or new vendor fast path is allowed.
- Run local syntax, policy, snapshot, JSON, and whitespace gates before B200.
- Start with the two dense targets plus retained-output correctness. Expand to
  dense, spectrum, low-rank, row-scaled, diagonal, and tridiagonal families only
  after the kernel compiles and a dense target is competitive.
- Run the full 15-shape grid only for a stable aggregate target improvement.
  Popcorn test 17/17 and at most one ranked submission are allowed only after a
  positive full grid with no material off-target regression.

## Bounded fallback ladder and cost

Measure at most six materially distinct variants: worker count/tile residency;
phase-overlapped dual-matrix scheduling; 64-column versus 128-column outer
panels; narrower trailing tiles; FP16/MX precision with correction; and a
cluster/DSM implementation if the toolchain positively supports it. Do not
spend variants on the already-rejected host-scheduled split32 transfer.

Use at most three B200 jobs for initial compile/correctness/paired diagnosis and
approximately `$10` total Modal spend before a checkpoint. Preserve every valid
partial frontier and stop early on decisive compiler, residency, or numerical
evidence.
