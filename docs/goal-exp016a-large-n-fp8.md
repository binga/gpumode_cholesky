# Goal — Experiment 016a: large single-matrix overhaul (1×8192, 1×16384, 1×32768)

## Baseline

Ranked `#881981` (exp 015), public 1262.9337990784535μs, secret
1270.7067480724075μs. Exact source = root `submission.py`, SHA-256
`0400f06ad250b3bef240eb5cebae10ca9c045cc227924e06185be966e7310bb5`.
Modal single-module means: 1×8192 = 6410μs (pure cuSOLVER),
1×16384 = 15992μs (left-looking TF32), 1×32768 = 47525μs (left-looking
FP8 panels). Exp-014 component profile at 32768 (48ms): TRSM/inverse 12.1ms,
diag cuSOLVER 11.2ms, TF32 diag update 8.1ms, addmm 4.8ms, copies 4.6ms,
panel mm 4.4ms, FP8 GEMM 2.5ms, amax 2.4ms.

## Ladder (bounded, ≤6 serious variants, ~$8 Modal)

1. 1×8192 left-looking path (TF32 and FP8-panel variants, nb 1024/2048).
2. Recursive block triangular inversion (TF32 combines, 512 base) replacing
   `solve_triangular` panel solves at 16384/32768.
3. FP8 diagonal-block update + panel mm at 32768 (FP32 accumulation);
   abort if dense scaled residual > ~12/20.
4. Fixed global FP8 scale from max(diag(A)) (|L_ij| ≤ sqrt(max A_ii)),
   guarded by a diag-ratio dispatch check; dynamic-amax path retained.
5. FP8 shadow of L: quantize each panel once when finalized.

## Gates

Official tolerance unchanged (20·n·eps·‖A‖₁). Paired same-process probes per
shape vs the exact ranked baseline; families full at 8192/16384,
dense/lowrank/tridiagonal at 32768 (S6 policy); backend counters prove the
fast path (fallback timings invalid); final single-module
`modal_verify.py verify` + `benchmark`. No popcorn submission from this
experiment (parent owns integration/ranking). No "stream" substring.
