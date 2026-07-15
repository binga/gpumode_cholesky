# Goal — experiment 010 Blackwell `1x16384`

Own the exact ranked shape `batch=1, n=16384` on B200, starting from ranked
winner `#878273` at commit `4b4d557`. The shipped path is experiment 008's
right-looking blocked Cholesky with `nb=2048`: FP32 diagonal factorization and
panel solve plus a fused full-matrix TF32 `addmm_` trailing update. Its current
rank-faithful Modal latency is `18591.1 us`; the strict same-process promotion
threshold is therefore `<= 9295.6 us` (at least `2.00x`).

## Constraints

- Preserve reconstruction correctness across dense, spectrum, diagonal,
  lowrank, rowscale, and tridiagonal families.
- Do not change dispatch behavior for the other fourteen ranked shapes.
- Retain rotating inputs and every output during paired timing, matching the
  Popcorn benchmark ownership contract.
- Reject a compiled/custom candidate if its intended backend does not load or
  silently falls back.
- Keep `submission.py` free of forbidden non-default queue APIs and their source
  text. Run the static source scan before every paid test.
- Do not launch leaderboard mode. A passing candidate stops at
  `READY_FOR_RANKED` until the supervising task authorizes ranking.

## Bounded architectural ladder

These are architectural experiments, not a block-size sweep:

1. Lower-triangle-only TF32 Schur updates using a tiled Triton tensor-core
   kernel, eliminating the unused upper-half GEMM work.
2. Lower-triangle-only cuBLAS SYRK control through the active PyTorch handle,
   to distinguish triangular library support from the custom kernel result.
3. FP8 E4M3 Schur updates with explicit scaling and guarded fallback/recovery.
4. Lower-triangle-only FP8 E4M3 updates, composing reduced precision with
   triangular work elimination.
5. Hierarchical blocked factorization: recursively factor the diagonal block
   so diagonal work also uses tensor-core Schur updates.
6. Graph replay composed with the fastest arithmetic kernel, with owned output
   storage and rotating-input refresh.

If needed after a credible near miss, a seventh candidate may test a persistent
grouped/cluster-oriented scheduling formulation. CUTLASS sm100/tcgen05/TMA is
preferred where a self-contained, source-policy-clean build is available; an
unavailable backend is recorded as infrastructure evidence and never presented
as a timed candidate.

## Promotion gates

1. Free: local 10/10 property test, Python compilation, JSON parsing,
   `git diff --check`, root-baseline snapshot check, and forbidden-source scan.
2. Paired B200: baseline and candidate in the same process, rotating among the
   Popcorn-faithful number of inputs, retaining and validating every output.
3. Candidate mean `<= 50%` of paired current-path mean, with candidate best and
   backend-engagement evidence recorded.
4. All six target families pass with correctness margins recorded.
5. Full 15-shape Modal verification passes with no regression outside the exact
   target dispatch.
6. Popcorn test mode passes 17/17.
7. Report `READY_FOR_RANKED` and wait. Only an authorized ranked winner may be
   integrated into the latest ranked baseline, documented, committed, pushed,
   and monitored to completion.

If six serious measured variants do not reach `2.00x`, preserve all source and
raw JSON, document bounded exhaustion, leave root `submission.py` byte-identical
to commit `4b4d557`, and commit/push the rejected experiment.
