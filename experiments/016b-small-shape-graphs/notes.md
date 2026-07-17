# Experiment 016b — small shapes (4096×32, 1024×64, 256×128)

**Outcome: one winner (4096×32 rank-2 kernel, 1.591× paired), two shapes
closed as already-optimal.** No popcorn submission from this experiment;
integration owned by the parent session.

Baseline: exact `#881981` root source, SHA-256
`0400f06ad250b3bef240eb5cebae10ca9c045cc227924e06185be966e7310bb5`.

## v1 (probe-v1.json) — all three REJECTED

| shape | baseline | candidate | speedup | verdict |
|---|---:|---:|---:|---|
| 4096×32 graphed Triton kernel | 64.1μs | 75.8μs | 0.845× | rejected — shape is kernel-bound; graph's copy-in + clone-out (2×16MB) costs more than the launches it saves |
| 1024×64 split32 two-level | 119.7μs | 151.9μs | 0.788× | rejected — serial micro chain can't beat graphed batched cuSOLVER at n=64 |
| 256×128 split32 two-level | 156.5μs | 173.1μs | 0.904× | rejected — same at n=128 |

All 18 family checks passed (the losing paths were correct, just slower).

## v2 (probe-v2.json) — WINNER

`_chol32_rank2_kernel`: rank-2 reformulation of the one-warp n=32 kernel —
two columns per serial step (16 iterations instead of 32), single fused
trailing write per step, upper zeroed on store. No graphs, no dispatch
changes beyond routing n==32 to the new kernel.

- Paired same-process vs exact `#881981`: **62.8 → 39.5μs = 1.591×**.
- Families 6/6 at 4096×32; scaled residuals 0.016–0.084 of 20 (≈240×
  margin), same order as the rank-1 kernel.
- Single-module gates (authoritative): verify **57/57**
  (verify-v2-single.json), benchmark **15/15**, geomean 1314.7μs
  (benchmark-v2-single.json).

## Deliverable

`experiments/016b-small-shape-graphs/candidate-small-v2.py` — exact
`#881981` plus only the rank-2 n=32 kernel. py_compile clean, zero
"stream" occurrences.

Cost: 4 Modal B200 runs ≈ $2–3.

## Insights

- 4096×32 was **not** launch-bound as assumed: the kernel itself was ~60μs
  (serial 32-step chain), 6× above the ~10μs memory floor. Rank-2 halves
  the chain; a rank-4 step might yield another ~1.2–1.3×.
- Graph replay is only worth it when per-call CPU/launch overhead exceeds
  2× matrix-size copy traffic; at 16MB inputs that crossover is ~10μs.
- n=64/128: graphed batched cuSOLVER stands. The split32 micro chain
  (~16μs per 32 columns) is the wrong tool below n=256.
