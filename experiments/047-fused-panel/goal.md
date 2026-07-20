# Goal — Experiment 047: fused resident-panel kernel

Control: ranked `#890659` (SHA-256
`59558b501fb32d403667fd85a338ece7bb196f352a93685f7934bab8526d5e52`).
Targets `640x512`, `60x1024`, `8x2048`. Threshold: 2.00x per shape.

## The finding that motivates it

The panel kernels are **bandwidth-bound, not compute-bound**. Measured from the
`#889994` shapediag at `640x512`:

| kernel | traffic/call | time/call | achieved BW | achieved FLOPs |
|---|---|---|---|---|
| `_panel_apply32` | 79 MB | 21.2us | **3.7 TB/s** | 89 TFLOP/s |
| `_panel_inner32_subtile64` | 275 MB | 36.0us | **7.6 TB/s** | 157 TFLOP/s |

B200 HBM peak is ~8 TB/s. `_panel_inner32` is already **at peak**. It cannot be
made faster as written -- the only way to speed it up is to *move less data*.

It moves 275 MB per call because the block-column tile is re-read from global
memory on every one of the seven launches that make up one 128-wide block
(4 x `_micro_potrf_gj32` interleaved with 4 x apply + 3 x inner). Total panel
traffic across the factorization is ~3.5 GB; the information-theoretic minimum
is one load and one store of each block column:

| nb | fused traffic | at 7 TB/s | shipped apply+inner |
|---|---|---|---|
| 128 | 503 MB | **72us** | 751us |
| 256 | 336 MB | **48us** | 751us |

This is a 10-15x reduction on 56% of the shape's device time, and it is the
first lever in this campaign justified by a *traffic* bound rather than a
throughput estimate. Every previous attempt (exps 045, 046) tried to raise
FLOP/s on work that was already memory-limited, which is why all of them lost.

## Architecture

Split the block column at `j` into the diagonal block and the rows below.

1. **Diagonal block `A[j:j+nb, j:j+nb]`** — keep the existing distributed
   launches (`micro32` + small inner updates) but restrict every `remaining`
   argument to `min(n, j+nb) - k - 32` so they touch only `nb` rows. This is a
   small region and its traffic is negligible. Do **not** move it into a single
   CTA per matrix: at batch 640 that costs 4.3 waves x the full serial chain
   (~660us), while the distributed micro does all 512 pivots in 173us.

2. **Rows below, `A[j+nb:, j:j+nb]`** — ONE fused kernel, grid = (row-tiles,
   batch). Each CTA loads its `TILE_R x nb` tile into registers/shared **once**,
   runs all `nb/32` sub-steps against the diagonal inverses already published in
   `dinv`, then stores **once**. No cross-CTA synchronisation is needed: the
   diagonal is fully factored by step 1 before this kernel launches, and row
   tiles are independent of each other.

3. **Trailing** — the shipped cuBLAS `baddbmm_` (exp 046), which already
   reaches 250 TFLOP/s at nb=256.

Panels must stay **tf32x3**: exp 044 v11/v12 isolated the n=512 residual cliff
(2.59 -> 17.7 / 20) to the panels specifically.

## Projected budget at 640x512 (shipped 1351us, 2x target 676us)

| constituent | nb=128 | nb=256 |
|---|---|---|
| micro32 (distributed, unchanged) | 173 | 173 |
| diagonal-restricted inner | ~50 | ~50 |
| fused resident panel | 72 | 48 |
| trailing (cuBLAS) | 215 | 86 |
| copies, finite gate, idle | ~110 | ~110 |
| **total** | **~620us (2.18x)** | **~467us (2.89x)** |

nb=256 needs a `TILE_R x 256` tile resident; at TILE_R=32 that is 32 KB, well
inside the budget, so the wider block is viable and preferred.

## Risks

- Register/shared pressure in the fused kernel forcing spills, which would
  reinstate the traffic it exists to remove. Check the achieved bandwidth of
  the new kernel directly (target >= 5 TB/s) before trusting any speedup.
- `8x2048` is graph-mode; the fused kernel must be a Triton kernel (or a CUDA
  kernel launched without naming a work queue, which is not capturable -- see
  exp 044). Prefer Triton for that shape.
- `60x1024` regressed under the exp-046 trailing swap at batch 60; re-measure
  it independently rather than assuming the 640x512 result transfers.

## Gates

Free checks, paired probe on `640x512` first, achieved-bandwidth check on the
new kernel, six families on every changed shape, full 15-shape grid, Popcorn
17/17, then one ranked submission.
