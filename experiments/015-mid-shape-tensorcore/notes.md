# Experiment 015 — mid-shape batched tensor-core factorization

**ADOPTED.** Ranked `#881981`: public **1262.9337990784535μs**, secret
**1270.7067480724075μs** (previous best `#880770`: 1447.259/1443.226μs →
**−12.74% / −11.95%**). Popcorn test `#881978` passed. Rank 12 → 11.
Exact ranked source SHA-256:
`0400f06ad250b3bef240eb5cebae10ca9c045cc227924e06185be966e7310bb5`.

## Hypothesis (from the 2026-07-17 leader-gap analysis)

The geomean weights all 15 shapes equally; nine mid shapes on stock cuSOLVER
sat 19–260× above B200 hardware floors. The S5 "640×512 saturated" verdict
only compared cuSOLVER dispatch variants, never custom tensor-core math.

## What shipped

1. **Two-level blocked tensor-core factorization** (new Triton kernels) for
   64×256, 16×512, 640×512, 4×1024, 60×1024, 8×2048:
   - `_micro_potrf_gj32`: 1-warp-per-matrix rank-2 diagonal potrf with the
     triangular inverse built in the same 16-step serial loop.
   - `_panel_apply32`: panel columns via `tl.dot(P, Dinv^T)` (tf32x3).
   - `_panel_inner32`: narrow rank-32 update of remaining panel columns.
   - `_trailing_nb`: one rank-128 Schur update per outer panel (tf32 at
     n≥1024, tf32x3 at n≤512), cutting trailing RMW traffic 4× vs rank-32.
   - Whole launch sequence replayed as a per-shape CUDA graph (shared memory
     pool), `copy_` in / owned `clone` out.
2. **Graph-replayed exact cuSOLVER path for 1024×64** (candidate B fork).
3. **256×128 converted from `make_graphed_callables` to the manual
   static-buffer capture pattern** (same vendor kernel).

## Paired B200 evidence (final source vs exact #880770, same process)

| shape | baseline | candidate | speedup |
|---|---:|---:|---:|
| 64×256 | 358.9μs | 265.8μs | 1.350× |
| 16×512 | 581.4μs | 496.7μs | 1.171× |
| 640×512 | 3932.8μs | 2293.9μs | **1.714×** |
| 4×1024 | 1312.9μs | 923.6μs | 1.421× |
| 60×1024 | 3203.9μs | 1611.1μs | **1.989×** |
| 8×2048 | 3496.4μs | 2199.2μs | 1.590× |
| 1024×64 | 129.4μs | 119.1μs | 1.086× |

Full-grid paired aggregate: **1.1859×** (1522.3 → 1283.6μs Modal geomean),
all untouched shapes 0.999–1.001×. Single-module gates (popcorn-like process):
verify **57/57**, benchmark 15/15 at geomean 1325.7μs. Timed dense runs:
fast-path counters exact, zero fallbacks, zero errors. Residuals: tf32x3
shapes ≈FP32 (0.005–0.017 scaled); tf32 shapes 2.0–7.28 of 20 allowed
(2.7–10× margin). Lowrank family takes exactly one expected safety fallback
per tf32 shape; all six families pass on every changed shape.

## Rejected on measurement (bounded ladder, rounds r1–r6)

- Fused one-CTA whole-matrix potrf (r1): serial 256-thread reductions; slower
  everywhere and over-tolerance at n=256 (all-TF32 inverse panel).
- IEEE `tl.dot` panels (r2): correct but slow lowering; superseded by tf32x3.
- Rank-32 single-level trailing (r3): trailing RMW re-traffic n/32 passes.
- TILE=256 trailing (r6): exceeds register/smem budget, compile fails.
- 2×2048 on the new path: 0.651× (batch=2 cannot amortize serial diag) —
  stays on the shipped per-matrix loop.
- Candidate B superpanels for 1×4096/2×4096: 0.18–0.97× across 3 variants
  (cuSOLVER's single-matrix potrf there is compute-busy, not launch-bound).
- 1024×64 fused Triton (0.673×), consistent with exp-002/003.

## Defects found and fixed

- Graph-entry use-after-free: the `dinv` workspace wasn't kept alive with the
  captured graph → deterministic-looking async illegal memory access.
- Paired-probe artifact: `make_graphed_callables` in the *baseline* module
  corrupts replays when another module captured a graph earlier in the same
  process (residual exactly 1.42 at 256×128). Single-module runs are clean;
  the candidate also converts 256×128 to manual capture and shares one graph
  memory pool across all captures.

## Cost

~14 Modal B200 sandbox runs ≈ $6–9 total (incl. candidate B fork's 4 runs).
Popcorn: one test (`#881978`) + one ranked (`#881981`).

## Next ideas

- Micro kernel is still ~16μs/launch (serial floor ~n×500ns): rank-4 pivot
  blocks, or manual CUDA graph node dependencies to overlap diag with
  trailing (capture API records linear order only).
- 2×2048/1×4096/2×4096 remain open; need a fundamentally cheaper serial path.
- Leaders moved to 492–506μs (2026-07-17): another ~2.5× needed; likely a
  single persistent kernel per shape and FP8/BF16 trailing at mid n.
