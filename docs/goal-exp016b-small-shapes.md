# Goal — Experiment 016b: small shapes (4096×32, 1024×64, 256×128)

## Baseline

Ranked `#881981` (exp 015), public 1262.9337990784535μs, secret
1270.7067480724075μs. Exact source SHA-256
`0400f06ad250b3bef240eb5cebae10ca9c045cc227924e06185be966e7310bb5`.
Modal single-module means: 4096×32 ≈ 63–87μs (Triton one-warp kernel, not
graphed), 1024×64 = 119μs (exp-015 cuSOLVER graph), 256×128 = 164μs (manual
cuSOLVER graph). Memory floors ~5–10μs — all three are launch/alloc-bound.

## Ladder (≤5 Modal runs, ~$4)

1. 4096×32: CUDA-graph the Triton n=32 kernel (static buffers, shared pool,
   owned clone out). Removes per-call `contiguous`+`empty_like`+launch.
2. 256×128: route to the exp-015 split32 two-level path (4 micros, no outer
   trailing; tf32x3; expect ~half the serial chain of the 164μs graph).
3. 1024×64: split32 config (2 micros) vs the 119μs graph path.
4. If split32 loses at these n, keep graph paths and report.

Known hazard: `_clear_upper_tiles` has no n-bound (TILE=128 > n=64 would
write out of bounds) — pass TILE=min(128, n).

## Gates

py_compile, no "stream" substring, paired probes vs exact baseline with
families and counters, and — authoritative for any graph change —
single-module `scripts/modal_verify.py verify` (57/57) + `benchmark`.
The dual-module probe can false-alarm on graph paths (residual-1.42
artifact, exp 015).

No popcorn submissions from this experiment; the parent session owns
integration and ranking.
