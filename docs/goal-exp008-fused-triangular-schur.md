# Goal — exp 008: fuse and triangularize the TF32 Schur update

**Owner:** dedicated optimization thread. **Supervisor:** parent session (babysits).

## Objective

Produce a correct ranked submission with a geometric-mean latency strictly below
the current best, `#878015` (about **1559 us**), by removing avoidable work from
the blocked-TF32 Cholesky path used for `1x16384` and `1x32768`. Continue through
measurement, correctness validation, full-grid validation, and one justified
leaderboard submission; do not stop at a promising local microbenchmark.

The current implementation already proves that TF32 trailing updates are the
right algorithmic direction, but each step currently evaluates:

```python
a[j:, j:] -= l21 @ l21.transpose(-1, -2)
```

This materializes a full product, launches a separate subtraction, and computes
the upper triangular half even though later Cholesky steps consume only the lower
triangle. The trailing update is the dominant O(n^3) work, so this is the most
direct untried continuation of the shipped winner.

## Current baseline and evidence

- Ranked best: `#878015`, geomean about **1559 us**, 17/17.
- Ranked `1x16384`: about **19,400 us** with blocked TF32, versus 34,200 us
  cuSOLVER.
- Ranked `1x32768`: about **77,200 us** with blocked TF32, versus 221,000 us
  cuSOLVER.
- Modal full-grid reference for experiment 006: 19,982 us and 78,357 us.
- BF16x9 was conclusively rejected in experiment 007; do not repeat it.
- Plain FP16/BF16 trailing updates lost to TF32; do not repeat them.

Geomean leverage: a 10% reduction on `32768` alone lowers the total score by
roughly 0.7%; a 20% improvement on both changed shapes lowers it by roughly 3%.
Small but robust wins are valid, but account for benchmark noise before spending
a ranked submission.

## Experiment ladder

Work through these stages in order, keeping only measured improvements.

### Stage A — fused in-place TF32 update

Replace the temporary-product-plus-subtraction sequence with a fused update such
as an in-place `addmm`/`addmm_` formulation using `beta=1` and `alpha=-1`, while
preserving TF32 and FP32 accumulation. Confirm that the strided trailing view does
not cause a hidden contiguous copy or a slower backend.

Also inspect adjacent full-matrix traffic that can be removed safely:

- avoid redundant copies of the panel result where an output buffer is supported;
- replace the full-factor `isfinite` scan with a cheaper failure signal only if
  every previously failing family is still detected;
- fold upper-triangle zeroing into result construction if it avoids a separate
  full-matrix pass.

Do not bundle speculative cleanups before measuring the fused Schur update alone.

### Stage B — lower-triangular TF32 SYRK-style update

If Stage A is insufficient, implement a custom CUDA, Triton, or CuTe/CUTLASS
update that computes only lower-triangular output tiles of
`A22 -= L21 @ L21.T`. It must:

- use Blackwell tensor cores with FP32 accumulation;
- skip upper-triangular output tiles entirely;
- handle diagonal tiles correctly without double-updating;
- support the large dimensions and block sizes used by experiment 006;
- remain on the default CUDA queue and contain no forbidden queue-management
  APIs or banned source text in the submitted file;
- preserve the existing numerical fallback for difficult inputs.

Compare the custom triangular kernel against the fused dense GEMM, not merely
against the original expression. A theoretically smaller FLOP count is not a win
unless end-to-end Cholesky latency improves on B200.

### Stage C — hierarchical blocking if the diagonal path becomes limiting

If the triangular update wins but the overall gain stalls, profile by operation
and try a two-level scheme: coarse outer blocks to reduce step count, with the
diagonal block recursively factored using the proven smaller blocked-TF32 method.
Retune block sizes for `16384` and `32768`; do not assume the existing
2048/4096 choices remain optimal after the update changes.

### Stage D — bounded pivot if the primary path is exhausted

If Stages A-C produce no end-to-end win after measured tuning, pivot once to a
batched version of the same fused TF32 blocked algorithm for `4x1024` and
`8x2048`. The prior diagnostic ceilings were about 0.63 ms and 3.48 ms versus
current ranked times around 1.30 ms and 5.05 ms. Capture concurrency inside
batched matrix operations or a custom kernel; non-default CUDA queues are not
submittable. Do not revisit loop/chunk scheduling already rejected in experiments
004-005.

## Required loop

1. Create `experiments/008-fused-triangular-schur/` and preserve every serious
   candidate/result there.
2. Run the free local property check after edits.
3. Iterate on targeted Modal B200 benchmarks for `16384,32768`; use `16384` while
   developing and add `32768` only for credible candidates.
4. Validate all input families at every changed size, including the ill-conditioned
   cases that trigger the experiment-006 fallback.
5. Once a candidate is clearly faster, run the full 15-shape Modal benchmark and
   reject any regression outside its dispatch region.
6. Submit to Popcorn test mode and require 17/17.
7. Make at most one leaderboard submission, only after the full-grid result
   projects a geomean below 1559 us with a margin large enough to survive noise.
8. Confirm the ranked result from the submissions list. If it is better, adopt it
   to root `submission.py` and update the README, journal, optimization tracker,
   experiment log, benchmark artifact, and notes.
9. Commit the complete experiment. Do not push.

## Guardrails

- Correctness is non-negotiable across dense, diagonal, spectrum, low-rank,
  row-scaled, and tridiagonal families.
- Do not change the reference checker or timing semantics.
- Do not regress any unchanged ranked shape.
- Do not repeat BF16x9, plain FP16/BF16, naive n=64/128 kernels, high-batch
  cuSOLVER loop/chunk scheduling, or non-default CUDA queue approaches.
- Keep Modal spending disciplined: targeted shapes first, full grid only for a
  credible finalist, and terminate stalled sandboxes promptly.
- Preserve a safe fallback for numerical failures. A dense-shape speedup that
  breaks difficult families is not acceptable.
- Avoid blind leaderboard submissions. Mid-size cuSOLVER timings have shown
  substantial run-to-run drift; judge candidates with paired same-run data.

## Definition of done

The task is complete only when one of these is true:

1. **Success:** a ranked submission is confirmed below about 1559 us, passes
   17/17, is adopted and documented, and the experiment is committed; or
2. **Exhausted primary plus pivot:** the fused/triangular/hierarchical large path
   and the single bounded batched pivot have measured rejection evidence, notes,
   artifacts, and a committed rejected experiment.

The desired outcome is success. Continue iterating autonomously while safe,
measured, untried options remain; report to the supervising thread whenever a
candidate wins, fails correctness, encounters infrastructure trouble, or is ready
for the ranked submission.

## Report back

Report the best Modal full-grid geomean, paired per-shape deltas versus `#878015`,
which stage won, correctness coverage and fallback behavior, ranked submission ID
and score, experiment path and commit, approximate Modal spend, and any remaining
untried idea.
