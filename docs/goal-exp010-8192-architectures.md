# Goal — experiment 010: exact `1 x 8192` architecture search

## Baseline and hard target

- Start point: ranked winner `#878273`, commit `4b4d557`, public geomean
  `1500.7037765896727 us`.
- Owned ranked shape: `batch=1, n=8192`, currently dispatched to
  `torch.linalg.cholesky_ex` / cuSOLVER.
- Retained-output Modal evidence: `6435.588 us` mean and `6427.520 us` best in
  `experiments/009-combined-shape-frontiers/full-grid-owned-outputs.json`.
- Strict paired promotion threshold: candidate mean at most 50% of the current
  path's same-process mean. Against the recorded mean this is `3217.794 us`;
  the actual decision uses the newly paired control from the candidate process.

## Correctness and policy constraints

- Return owned, lower-triangular FP32 output with positive diagonal.
- Pass the vendored reconstruction checker for dense, spectrum, diagonal,
  lowrank, rowscale, and tridiagonal families at the exact shape.
- Retain all rotated outputs until after correctness checking.
- Reject any compiled path that fails to load or silently falls back.
- The eventual submission source must not contain non-default CUDA queue APIs
  or their literal source text.
- Root `submission.py` remains byte-identical to experiment 009 unless a
  candidate clears the 2.00x paired gate, full 15-shape no-regression gate, and
  Popcorn test 17/17.
- Do not launch leaderboard mode. Stop at `READY_FOR_RANKED` and wait for the
  supervising task.

## Bounded architectural ladder

The ladder contains at least six materially different implementations. A block
size-only change is calibration and does not count as another architecture.

1. Direct legacy `cusolverDnSpotrf` with explicit reusable workspace.
2. Direct 64-bit expert `cusolverDnXpotrf`; document that NVIDIA exposes only
   its default potrf algorithm, so no hidden algorithm mode can be swept.
3. Host-fused one-level blocked factorization: cuSOLVER diagonal, cuBLAS TRSM
   panel, true lower-only TF32 SYRK update.
4. CUDA-graph replay of the host-fused lower-only blocked pipeline with owned
   output.
5. Triton lower-only blocked factorization: fused diagonal/panel kernels and
   triangular TF32 update tiles.
6. Reduced-precision lower-only Schur update (FP8/MXFP8 when the installed
   CUDA/PyTorch surface supports it; otherwise BF16/FP16 proxy) plus an exact or
   iterative correction stage.
7. If an earlier rung is close but below 2x, add a two-level diagonal/panel
   factorization or a persistent cooperative fused panel/update kernel.

## Measurement and promotion

- Run syntax, local properties, JSON parsing, snapshot comparison, source scan,
  and `git diff --check` before billed work.
- On B200, generate Popcorn-family inputs once, rotate across a memory-bounded
  set, clear L2 before each timing sample, retain outputs, and measure baseline
  and candidates in the same process.
- Record raw JSON for every serious measured architecture, including compile
  activation, mean/best latency, speedup, checker message, tolerance fraction,
  and rejection reason.
- Promote only if the exact-shape mean is at most half the paired control, all
  six families pass, the full 15-shape Modal comparison has no regression, and
  Popcorn test mode passes 17/17.

## Terminal outcomes

- `READY_FOR_RANKED`: all gates above passed; report artifacts and wait.
- `BOUNDED_EXHAUSTION`: at least six serious architectures were measured but no
  candidate reached 2.00x. Preserve all evidence, leave root best unchanged,
  document, commit, and push the rejected experiment.
