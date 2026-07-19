# Goal — Experiment 039: CUDA `4096x32` Cholesky

## Contract revision 2

Experiment 038 bounded the original first target, `2x2048`, after six distinct
correct architectures all lost. The three-shape contract is therefore revised
explicitly—not silently—to `4096x32`, `1x4096`, and `2x4096`. The latter two
retain the original vendor-kernel targets; `4096x32` replaces the exhausted
shape and is attempted first. A fresh null baseline is stored separately as
`audit/baseline-rev2.json`.

## Frozen baseline and target

- Ranked source: `#888352`, commit `fc69597` (factorization source unchanged
  from `f84e1de`).
- `4096x32`: revision-2 null baseline 43.18us paired wall (the earlier
  single-module profile was 40.0us wall / 38.1us device) in one
  `_chol32_rank2_kernel` launch.
- Required 2x target: **at most 21.59us wall** and a paired ratio of at least
  2.00x against the interleaved ranked source.
- Off-target regression limit: 3%; official correctness threshold unchanged.

## Constituent diagnosis and hypothesis

The current shape is neither host-launch nor math bound: wall-minus-device is
about 2us (5%), one Triton kernel accounts for all 38.1us of device time, and
compulsory 16MiB input plus 16MiB output traffic is about 4.4us at the B200 HBM
roof. The gap is the kernel's predicated full-tile state transformation and
serial pivot chain.

Replace only this exact dispatch with a default-queue CUDA kernel: one warp per
matrix, one column per lane in registers, and a 32-float shared pivot column.
This removes Triton's full 32x32 `tl.where` rebuild while retaining thousands
of independent matrices for occupancy. No auxiliary/concurrent queues, CUDA
graphs, or cuSOLVER call are introduced by the new path.

## Bounded ladder

1. Register-column, shared-pivot, one-warp right-looking kernel using `sqrtf`.
2. Same dataflow with reciprocal square root if family margin permits.
3. Two warps per matrix with split trailing rows if one warp is throughput-bound.
4. Four warps / 128 threads with cooperative trailing-element updates.
5. Register-row formulation with shared pivot column.
6. Stop after six genuinely distinct correct measured variants.

Promotion requires an active-backend counter, six families, paired target at or
below 21.59us, full-grid non-regression, Popcorn test 17/17, and then one ranked
submission. Further verified gains after 2x remain eligible serially.
