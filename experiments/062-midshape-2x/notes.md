# Experiment 062 notes — 2x2048 and 2x4096

Exact control snapshot: `baseline-907267.py`, SHA-256
`06799fb095b9fbccb476e7da2c0567a3d36ba57ccb09ccf278b49149db8814c2`
(ranked `#907267`, public 745.765us / secret 741.378us, commit `1d657f7`).

## Incumbent shape budget (fresh B200, `results/exp062-inc-shapediag.json`)

| shape | wall us | device us | idle | launches | dominant constituent |
|---|---:|---:|---:|---:|---|
| 16x512 | 402.3 | 376.4 | 6.4% | 54 | `_micro_potrf_gj32` 217.5 (16 calls) |
| 640x512 | 1333.0 | 1135.1 | 14.8% | 53 | `_panel_fused128` 361.0 |
| 4x1024 | 729.6 | 695.0 | 4.7% | 102 | `_micro_potrf_gj32` 436.1 (32 calls) |
| 60x1024 | 1228.8 | 1047.3 | 14.8% | 76 | `_trailing_nb` 387.1 |
| **2x2048** | **1358.4** | 1310.5 | 3.5% | 13 | **vendor `getrf_wo_pivot` 1195.0 (2 calls, 91.2%)** |
| 8x2048 | 1608.3 | 1660.3 | — | 197 | `_micro_potrf_gj32` 912.5 (64 calls) |
| 1x4096 | 1525.1 | 1522.3 | 0.2% | 6 | vendor `getrf_wo_pivot` 1385.0 (1 call, 91.0%) |
| **2x4096** | **3204.0** | 3194.8 | 0.3% | 13 | **vendor `getrf_wo_pivot` 2779.9 (2 calls, 87.0%)** |

The two enrolled shapes are the only mid shapes whose entire cost is one vendor
kernel run **once per matrix, serially** — 597.5us per 2048 and 1390.0us per
4096. That kernel is dependent-pivot-latency bound at ~0.33us/row, so its cost
is `c*n` per matrix and the batch dimension buys nothing today. A batched
blocked factorization pays the pivot chain **once for the whole batch**.

## Required speedup

Equal-weight geomean over 15 shapes: k shapes must each improve by
`(1/(1-N))^(15/k)`. For N=10%, k=2 that is **2.204x per shape**:

| shape | control us | target us |
|---|---:|---:|
| 2x2048 | 1358.4 | **<= 616.3** |
| 2x4096 | 3204.0 | **<= 1453.7** |

## Architecture

Right-looking blocked factorization, panel width 128, trailing update deferred
to `nb_outer`:

1. `e62_diag128` — one resident CTA per matrix factors a whole 128x128
   diagonal block in shared memory and publishes `inv(L11)`, so the panel
   below is a plain batched GEMM instead of a triangular solve.
2. Panel: `L21 = A21 @ inv(L11)^T`, one batched GEMM.
3. Trailing: `A22 -= L21 @ L21^T`, one batched TF32 rank-k GEMM.

## Round 1 — `results/exp062-probe-v1.json`

**The algorithm is right and the kernel implementation is wrong.**

Numerics are excellent and rule out any accuracy objection:

| check | value |
|---|---|
| 128x128 block reconstruction error | **0.0** (scale 1.093) |
| `inv(L11) @ L11 - I` | **2.4e-07** |
| whole-shape reconstruction, 2x2048 | 7e-05 (scale 1.12) |
| whole-shape reconstruction, 2x4096 | 6e-05 (scale 1.10) |

Speed is not:

| measurement | value | budget |
|---|---:|---:|
| `e62_diag128` per 128x128 block | **153.77us** | <= 21us |
| implied ns per row | **1201.3** | <= 164 |

At 32 blocks for n=4096 the diagonal kernel alone is 4920us, which is the whole
runtime. Prototype walls against the **shipped** control (`_loop_cholesky`,
not the batched vendor call — see the harness note below):

| shape | shipped | prototype (nbo=1024) | ratio |
|---|---:|---:|---:|
| 2x2048 | 1358.4 | 2711.2 | 0.50x |
| 2x4096 | 3204.0 | 5703.8 | 0.56x |

Subtracting the kernel gives the **non-diagonal** cost of the blocked driver:
**251us at 2x2048 and 784us at 2x4096**. Both already fit inside the targets
with room for a 15-20us/block diagonal kernel, so the entire experiment reduces
to the pivot-chain implementation.

### Harness note that cost a run

`torch.linalg.cholesky_ex` on a *batched* tensor takes the batched vendor path,
which is far slower than the incumbent's own route: it measured 4543us at
2x2048 and 14875us at 2x4096 against the shipped 1358.4us and 3204.0us. The
incumbent deliberately dispatches `2 <= batch <= 4, n >= 1024` to
`_loop_cholesky`. **Any control for these shapes must call `_loop_cholesky`,
never `cholesky_ex` on the batch.**

### Prime suspect

Round 1 fused L and `inv(L)` into ONE fully unrolled 32x32 Gauss-Jordan
carrying `float a[32]` *and* `float m[32]` live across 1024 unrolled bodies
with two shuffles each. If `ptxas` spills those arrays to local memory, every
pivot round-trips through DRAM, which is the right order of magnitude for a 7x
miss.

## Round 2 — repairs under measurement

Compiled with `-Xptxas -v` so register and spill counts are printed.

- **v2** register chain (`a[32]` only, 32 shuffles/pivot) plus a separate
  *column-parallel* triangular inverse: lane j solves `L x = e_j`, so every
  `L[i][p]` read is a shared broadcast and every `x_p` is lane-local — zero
  cross-lane traffic for the inverse.
- **v3** the same split, but the chain itself stays in shared memory, so there
  is no large register array to spill at all.
- an isolated `e62_chainbench` that reports **ns per pivot** for each design
  with no block setup, panel or trailing work in the way.

## Round 3 — `results/exp062-probe-v3.json`

4x4 register tiling over a staged transpose, `float4`-legal row stride, and
`clock64` phase accounting. **153.77 -> 62.221us per block.** Phase table:

| phase | us | pct |
|---|---:|---:|
| triinv | 18.52 | 29.8 |
| chain | 15.07 | 24.2 |
| load | 12.35 | 19.9 |
| trailing+inv | 5.83 | 9.4 |
| stageP+Qt | 4.53 | 7.3 |
| store/panel/commit | 5.93 | 9.5 |

The GEMM phases collapsed from ~80us to 13.9us, confirming the round-2
diagnosis. Whole-shape 2x2048 1.022x, 2x4096 1.116x.

## Round 4 — `results/exp062-probe-v4.json`

Two-level blocked triangular inverse (16 dependent steps instead of 32) and
`float4` global load/store. **62.221 -> 50.290us per block**, 162 registers,
zero spills.

| phase | round 3 | round 4 |
|---|---:|---:|
| load | 12.35 | **3.52** |
| store | 2.44 | **1.65** |
| triinv | 18.52 | **15.09** |
| chain | 15.07 | 15.68 |

The inverse did *not* fall as far as the halved chain length predicted — its
cost is dominated by the two 16x16x16 coupling products in a single warp, not
by the substitution chain. Whole-shape: 2x2048 **1.208x**, 2x4096 **1.308x**.

## The full-grid gate earns its keep

First ship build measured **geomean 0.8403 — a 16% regression** — while both
enrolled shapes were exactly on target (2x2048 **1.1517x**, 2x4096
**1.2680x**, `_EXP062_HITS: 1` on both).

Cause: the ship merge declared `e62_diag128_launch` with **four** parameters
while rounds 3-4 had added a fifth (the profiling buffer). The signature
mismatch broke the whole combined `load_inline`, so `_CUDA32`, `_CUDA64`,
`_CUDA128` and `micro32` all failed to load and five unrelated shapes fell
back to slow routes: 4096x32 0.458x, 1024x64 0.275x, 256x128 0.478x,
640x512 0.912x, 60x1024 0.919x.

**A 57/57 correctness pass cannot catch this** — every fallback is numerically
correct, only slower. The paired grid's per-shape `candidate_counters` are the
only signal that the fast paths vanished. Always diff baseline vs candidate
counters before trusting any merged-extension build.

## Root cause of the merge failure

`NameError: name '_EXP062_SOURCE' is not defined`.

The combined `load_inline` runs at **import time**, ~3600 lines before the
appended tail that defined the CUDA source string. The surrounding
`except Exception` swallowed the NameError, set `_CUDA128 = None`, and every
dispatch guarded by `_CUDA32/_CUDA64/_CUDA128 is not None` silently took a
slower route. The ship build now hoists the source declaration above the
`load_inline`.

**Reusable lesson:** when merging a new kernel into the shipped combined
extension, the CUDA source string must be defined *before* the `load_inline`
call, not appended at the end of the module. The failure mode is invisible to
correctness gates and only shows up as missing `candidate_counters` in the
paired grid.

## Shipped result

Two ship layouts, both passing the full 15-shape paired grid against the exact
ranked incumbent `#907267`:

| build | layout | geomean | 2x2048 | 2x4096 | 13 others |
|---|---|---:|---:|---:|---|
| `ship-v5.py` | new kernel in its own extension | **1.0256** | 1.1482x | 1.2754x | 0.998-1.001 |
| `ship-v6.py` | merged into the combined extension | **1.0261** | 1.1541x | 1.2690x | 0.9998-1.0020 |

`all_shapes_ok: true`, every expected counter present on every shape, no new
fallbacks, official checker passing throughout.

Popcorn `--mode test` on `ship-v5` (`#909265`): **passed**, 68 seconds — the
three-extension cold build fits the 360s budget comfortably, so exp-050's
compile blocker does not bite here.

## Honest accounting against the 10% goal

+10% geomean from exactly 2 of 15 equally weighted shapes requires
`(1/0.9)^7.5 = 2.204x` **per shape**. Delivered 1.15x and 1.27x, i.e. **+2.6%**.

The gap is entirely the diagonal kernel: it would have to reach **<=16.6us per
128x128 block** and it is at 48-50us. The remaining cost is dominated by two
single-warp serial phases:

| phase | us | share |
|---|---:|---:|
| chain | 14.9 | 31% |
| triinv | 14.6 | 30% |
| everything else | 18.6 | 39% |

The pivot chain is already at **63.3 ns/pivot isolated** (2x better than
exp-050's 134 and 5x better than the vendor's ~330 ns/row), so the chain is
near its floor for a one-warp design. Closing the rest needs either the
triangular inverse folded into the chain (round-1's Gauss-Jordan, which was
numerically correct) or the block's serial phases overlapped with its parallel
phases via named barriers -- plus an fp16 trailing update to cut the ~330us /
~850us non-diagonal driver cost.
