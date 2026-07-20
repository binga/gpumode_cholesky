# Experiment 047 — fused resident-panel kernel

Control: ranked `#890659` (SHA-256
`59558b501fb32d403667fd85a338ece7bb196f352a93685f7934bab8526d5e52`),
806.037us public. Targets `640x512`, `60x1024`, `8x2048`; threshold 2.00x.

**Verdict: PARTIAL / ADOPTED.** Ranked `#890798` = **801.977us public /
847.836us secret**, improving `#890659` by **0.504% public**. Exact ranked
SHA-256 `fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
**No shape reached 2x.** `640x512` 1.0985x, `60x1024` 1.0924x, `8x2048`
rejected at 0.9070x.

## What shipped

`_panel_fused128` — one CTA owns a `TILE_R x 128` tile of the block column
below the diagonal block, loads it once as four `(TILE_R, 32)` register
tensors, runs all four 32-wide sub-steps against the four diagonal inverses,
and stores once. Per sub-step `s` it does one solve `L_s = t_s @ dinv_s^T`
and one update `t_u -= L_s @ D[u-block, s-block]^T` for each `u > s`: ten
`M=TILE_R, N=32, K=32` dots, only 1.24x the exact triangular-solve flop count.
The per-sub-step `_panel_apply32` / `_panel_inner32_subtile64` launches are
restricted to the diagonal block (`remaining = panel_end - k - 32`), so every
row below the block is now touched exactly once per 128-wide block instead of
seven times.

`_diag_block_step` — merges the restricted apply and inner into one
CTA-per-matrix launch. The inner update inside the block is `L @ L^T` of the
tile the apply just produced, so it needs no second global read. Enrolled at
`60x1024` only (see below).

`dinv` becomes `(slots, batch, 32, 32)` for enrolled shapes: the fused panel
consumes all four 32x32 inverses of a block *after* the block is finished, so
they can no longer share one buffer. Slot `s` is a contiguous slice, so the
Triton and CUDA micro kernels are unchanged.

Enrollment, `(TILE_R, num_warps, merge_diag_step)`:

    (640, 512): (128, 8, False)
    (60, 1024): (128, 8, True)

## The premise was half right

The goal document's premise was a **traffic** bound: `_panel_inner32_subtile64`
moves 275 MB per call in 36.0us = 7.6 TB/s = HBM peak, so the only lever is to
move less data, and one load plus one store of each block column would take
72us at nb=128 against the shipped 751us.

The traffic argument is correct and the fused kernel does remove the traffic —
but the resulting kernel is **not bandwidth-bound**, so the 72us projection
never applied. Measured achieved bandwidth of `_panel_fused128` at its best
configuration (`variant-02-midprobe.json`):

| config | us (3 panels) | TB/s total | TB/s tile-only |
|---|---|---|---|
| TILE_R=64  warps=4 | 502.7 | 2.13 | 1.00 |
| TILE_R=64  warps=8 | 493.4 | 2.17 | 1.02 |
| **TILE_R=128 warps=8** | **375.4** | **2.43** | **1.34** |
| TILE_R=128 warps=4 | 445.9 | 2.05 | 1.13 |
| TILE_R=256 warps=8 | 535.2 | 1.61 | 0.94 |
| shipped apply+inner, same 3 panels | 731.1 | — | — |

2.43 TB/s is far below the 5 TB/s gate the goal document set, but **not**
because of register spills. The kernel is arithmetic-limited: 1.007e10 useful
FLOP in 358us in situ = **28 TFLOP/s**, against the ~52 TFLOP/s the shipped
`_panel_inner32_subtile64` reaches on the same tf32x3 precision. The dots are
`N=32, K=32` and form a seven-deep dependency chain, and tf32x3 triples every
one of them. Halving the traffic bought 1.95x on the panel component, not the
10-15x the traffic ratio suggested.

Two micro-optimisations measured inside the same probe:

- **`tl.trans` elimination** (loading the 32x32 operands with swapped index
  expressions instead of transposing after load): **null**, 375.4 -> 375.1us.
  Triton already folds the transpose into the MMA operand layout.
- **Mirrored upper-triangle zero-fill**: 375.4 -> 331.4us without it, i.e.
  44us. Kept — the eager first-touch path has no separate clear pass, and a
  `_clear_upper_tiles` pass over 335 MB would cost at least as much.

## Where the panel saving went

`shapediag` on the candidate at `640x512` (`variant-04-shapediag-cand.json`),
wall 1356.0us / device 1125.9us / idle 230.1us / 53 launches:

| kernel | us | calls | us/call |
|---|---|---|---|
| `_panel_fused128` | 358.45 | 3 | 119.5 |
| `_trailing_nb` (first-touch panel) | 194.45 | 1 | 194.5 |
| `micro_potrf32_rank4` | 169.41 | 16 | 10.6 |
| `_panel_inner32_subtile64` (restricted) | 155.51 | 12 | 13.0 |
| `_panel_apply32` (restricted) | 104.86 | 12 | 8.7 |
| cuBLAS trailing (2 tiles) | 113.3 | 2 | — |

The shipped panel pair costs ~751us; the fused kernel costs 358us. But the 24
restricted diagonal-block launches cost **260us for almost no data** — `nrows`
is at most 96, so they are pure fixed cost at ~10.8us per launch. That is why
the first drop-in measured only 1.0392x at `640x512` despite a 356us kernel
saving. This reproduces the campaign's standing per-launch floor (exp 029:
~16us per serial Triton launch; exp 044: 13.55us for `_micro_potrf_gj32`
independent of batch).

## Measured variants

Paired same-process vs the exact ranked `#890659`:

| variant | change | 640x512 | 60x1024 | 8x2048 |
|---|---|---|---|---|
| v2 | fused panel only | **1.0392x** | — | — |
| v3 | + `_diag_block_step` on both | **0.9973x** | **1.2044x** | — |
| v4 | fused panel only, 3 shapes | **1.0566x** | **0.9147x** | **0.9070x** |
| **v5** | per-shape merge flag | **1.0985x** | **1.0924x** | not enrolled |

Three causal readings:

- **`_diag_block_step` is batch-dependent.** At batch 640 it is a net loss
  (1.0566 -> 0.9973): one CTA per matrix holding two live `128x128` register
  tiles costs more than the twelve launches it removes. At batch 60 it is the
  whole win (0.9147 -> 1.2044): there is no occupancy to lose, and the
  eight-panel schedule at n=1024 emits twice as many of these launches.
- **`60x1024` without the merge regresses.** At batch 60 the fused panel's
  grid is only `(ceil(nrows/128), 60)` CTAs, so removing traffic it was never
  bound by does not pay for the restricted launches it adds.
- **`8x2048` is rejected structurally, not marginally.** Its shipped schedule
  is NB=256 (exp 032, 1.031x, the one shape where fewer panels beat the
  `_trailing_nb` spill). The fused panel solves one 128-wide block column
  against the diagonal block above it; a 256-wide panel would need an extra
  rank-128 Schur update between its two halves, which is nb=128 again. Forcing
  uniform 128 doubles the panel count and the trailing launches
  (`_BMM_TRAILING_HITS` 21 -> 45) and measures **0.9070x**.

The v2 -> v4 spread at `640x512` (1.0392 vs 1.0566 for a functionally
identical path) and the v4 -> v5 spread (1.0566 vs 1.0985) are cross-run
drift, larger than the 0.09% within-run A-vs-A floor. The full-grid number is
the one to trust.

## Gates

- Free: `ast.parse`, `verify_local.py` 10/10, `git diff --check`,
  `grep -c -i stream` = 0.
- Six families on both changed shapes (`variant-07`, `variant-08`): all 24
  rows `checker_ok`, and **every residual is identical to the baseline's to
  three significant figures** — `640x512` dense 2.59, rowscale 0.247,
  tridiagonal 0.127; `60x1024` dense 9.59, rowscale 7.86, spectrum 7.82. Same
  `_FUSED_CTA_FALLBACKS` pattern as the baseline on the lowrank/spectrum rows
  (pre-existing, not introduced here). Panels stay tf32x3 at n=512.
- Full 15-shape paired grid (`variant-09-fullgrid.json`):
  **1.012106x CI95 [1.011423, 1.012789]**, 15/15 pass, residuals byte-identical
  on all 15, every off-target shape within 0.9985-1.0010 against a 0.60%
  worst A-vs-A spread.

| shape | baseline | candidate | ratio |
|---|---|---|---|
| 640x512 | 1350.5us | 1229.0us | **1.0985x** |
| 60x1024 | 1253.3us | 1147.5us | **1.0924x** |
| other 13 | — | — | 0.9985-1.0010 (parity) |

- Popcorn test `#890791`: 17/17 in 95s. Compile budget untouched — both new
  kernels are Triton, so the submission still performs three nvcc invocations.
- Ranked `#890798`: public **801.977us**, secret **847.836us**.

## Why 2x remains out of reach on these shapes

The 2x targets are 676 / 627 / 784us; the shipped result is 1229 / 1147 /
1571us. After this experiment `640x512`'s device time is 1125.9us with 230.1us
of eager-launch idle on top. The three remaining blocks are:

1. **358us of fused panel at 28 TFLOP/s.** Every reformulation that improves
   the dot shape increases the MAC count: the `K=128, N=128` single-dot form
   using an explicit 128x128 block inverse needs 1.6x the MACs, and tf32x3 —
   which exp 044 v11/v12 showed is mandatory at n=512 (residual 2.59 -> 17.7 /
   20 under plain tf32) — caps a well-shaped dot near 52 TFLOP/s. The
   available headroom is ~1.9x on this constituent, not 5x.
2. **260us of restricted diagonal-block launches carrying no data.** Merging
   them is a loss at batch 640; the alternative, one CTA per matrix over the
   whole 128-wide block, was measured in exp 044 at 47-58us per block and is
   batch-independent, i.e. ~4.3 waves at batch 640.
3. **169us of `micro_potrf32_rank4`**, the n sequential square roots, already
   at the 134ns/pivot floor exp 044 established.

Nothing in this experiment changes exp 046's conclusion, and it adds a
quantitative one: the panel kernels were at HBM peak, and removing the
redundant traffic *did* work, but it converts a bandwidth-bound kernel into an
arithmetic-bound one at less than half the useful throughput of the kernel it
replaced. The remaining gap needs a fused CUTLASS-style Cholesky that keeps
the panel resident across sub-blocks *and* issues fat MMAs, which `load_inline`
plus PyTorch ops cannot express here without breaking the six-minute compile
budget or the CUDA-graph capture rule.

## Artifacts

- `goal.md` — copy of `docs/goal-exp047-fused-panel.md`.
- `baseline-890659.py` — exact frozen control.
- `candidate-v1.py` / `candidate-v2.py` — first fused drop-in and the
  TRANSLOAD/MIRROR probe build (both carry `mid_probe()`).
- `candidate-v3.py` / `candidate-v4.py` — merged diagonal step, and the
  no-merge three-shape build.
- `candidate-v5.py` = `ranked-890798.py` — **ranked `#890798`**.
- `variant-01/02-midprobe.json` — kernel sweep and achieved bandwidth.
- `variant-03/05/06-paired*.json` — per-shape paired probes.
- `variant-04-shapediag-cand.json` — candidate constituent profile.
- `variant-07/08-families-*.json` — six families, candidate and baseline.
- `variant-09-fullgrid.json` — the 15-shape promotion grid.
