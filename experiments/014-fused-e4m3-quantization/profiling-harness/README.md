# Experiment 014 paired profiling harness

This directory defines the reproducible B200 evidence contract for improving
the ranked `batch=1, n=32768` shape. It deliberately does not alter the shared
runner or submission while candidate architectures are still being explored.

## Exact control

The baseline is the ranked experiment-012 source at commit `141d015`, submission
`878893`. Its SHA-256 is
`112ee017f96f4dafb95a173cf51bb59190c2ded1e7702a64992e6795985759dd`.
Both the local driver and GPU runner refuse to execute timing if the snapshot at
`experiments/012-large-left-looking-frontiers/submission.py` does not have that
digest. Candidate and baseline are imported as distinct modules in one process.

## Evidence sequence

1. Run `target`. Two ranked-family dense inputs rotate through six paired
   rounds. Baseline-first and candidate-first order alternates, L2 is cleared
   before every invocation, and CUDA events measure each invocation separately.
   The latest timed output for every backend/input pair remains referenced until
   the official checker validates it. Raw samples, mean, median, best, worst,
   standard deviation, and per-round speedups are emitted.
2. The target run separately captures a `torch.profiler` operator profile. This
   is component evidence, not latency evidence. It exposes clone/copy, factor,
   triangular solve, GEMM/update, quantization/scaling, and clear-upper costs
   according to the candidate's actual operators. Candidates may add
   `torch.profiler.record_function` labels around finer phases without changing
   the harness.
3. Run `families` only after a useful target result. Dense, spectrum, low-rank,
   row-scaled, diagonal, and tridiagonal inputs all run at the changed
   `1x32768` dispatch. Every output must be finite, lower triangular, have a
   positive diagonal, and pass the unmodified official reconstruction checker.
   `tolerance_fraction` records how much of the official factor-20 allowance was
   consumed. The ranked path is expected to take exactly one safety fallback on
   spectrum, low-rank, and row-scaled inputs; both modules must first record the
   intended 32768/fused-quantization hits, then increment the fallback counter
   exactly once. Dense, diagonal, and tridiagonal require zero fallback. Family
   runs are explicitly not timing evidence.
4. Run `full-grid` only for a promotion candidate. All 15 ranked shapes are
   paired in one process. Every candidate and baseline output is validated,
   every off-target candidate mean must be no more than 1.03 times its paired
   baseline, and candidate geometric mean must improve. Each module receives at
   least four per-shape warmups before arithmetic-mean timing so lazy graph/JIT
   setup is excluded symmetrically. Large result envelopes use chunked stdout
   transport to stay below Modal's per-line limit; the local driver verifies the
   exact serialized byte count before parsing. The final warmup output reference
   is released before timing so the first reversed-order call does not measure a
   one-time allocator expansion; every timed backend/input output remains live
   until official validation.

Example commands, to be run only when remote profiling is authorized:

```text
uv run python experiments/014-fused-e4m3-quantization/profiling-harness/modal_profile_013.py target --candidate PATH --json experiments/014-fused-e4m3-quantization/VARIANT-paired.json
uv run python experiments/014-fused-e4m3-quantization/profiling-harness/modal_profile_013.py families --candidate PATH --json experiments/014-fused-e4m3-quantization/VARIANT-families.json
uv run python experiments/014-fused-e4m3-quantization/profiling-harness/modal_profile_013.py full-grid --candidate PATH --json experiments/014-fused-e4m3-quantization/VARIANT-full-grid.json
```

## Backend proof contract

Timing is rejected unless the configured hit counter increases, the fallback
counter delta is zero, the runtime error field is `None`, required readiness is
truthy, module import succeeded, and the profiler observed at least one declared
backend operator. The default names match the exact experiment-012 backend:
`_LEFT_32768_HITS`, `_LEFT_LARGE_FALLBACKS`, `_LEFT_32768_ERROR`, and
`_HAVE_TRITON`.

Each materially different candidate must expose equivalent counters/status and
update only the candidate half of `backend_contract` plus the candidate
`required_candidate_operator_any_of` list before profiling. A candidate using a
CUDA extension should additionally expose its load/compile readiness in
`truthy_after`; a candidate using a different expert primitive should name that
primitive in the profiler operator list. Missing status is a failure, never
interpreted as zero.

Expected safety fallbacks must be given their own correctness-only specs and
contract. They must not be included in target timing. The dense target requires
zero fallback; the family contract records the ranked path's three intentional
ill-conditioned safety fallbacks explicitly.

## Promotion decision

- `WINNER`: all target validity/backend gates pass and paired mean speedup is at
  least `2.00x`.
- `FRONTIER`: all target gates pass and paired mean improves, but by less than
  `2.00x`.
- `REJECTED`: incorrect, slower, missing backend proof, runtime error, or any
  fallback-only timing.

No result is promotion-ready until target classification is `WINNER`, all six
families pass, the paired full-grid geometric mean improves with no material
off-target regression, and the exact candidate later passes Popcorn `17/17`.
Ranking, journal/README integration, commit, push, and remote verification remain
the later `program.md` gates; this harness does not perform them.
