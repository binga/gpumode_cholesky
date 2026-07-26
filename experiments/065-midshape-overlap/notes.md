# Experiment 065 — named-barrier overlap in the 128x128 diagonal block

**Goal (user):** improve latency on a mid shape. North star 50% (2.00x).

**Baseline.** Ranked `#913511` (exp 064), source sha
`8e4603e56432b86be263d74743dd4d52940d043682cfca515a71e69c10a26baa`, commit
`8d1704e`.

**Lever.** Exp 064's own next-lever list, item 1 — plan item 2, "named-barrier
overlap in the diagonal block kernel", the only lever it identified with real
headroom on the 47-60% of these shapes that the diagonal costs.

## The north star was not reachable, and was known not to be before any spend

Stated up front rather than discovered late. Exp 063 established the wall:
**"the 128 dependent pivots cost 10-20us however they are parallelised"** — v0's
one-warp shuffle chain 15.26us, v1's fused chain 19.23us, v2's 256-thread panel
20.40us. Every parallel architecture for these shapes is already measured
negative (`docs/experiment-matrix.md`: persistent/cooperative/cluster/DSM is
5-for-5, exps 028/038/040/048/049, best 0.697x).

The arithmetic on the remaining lever: if overlap takes the block from 37.9us to
the predicted ~25us (1.52x) and the diagonal is ~55% of a shape, the shape gains
`1 / (0.45 + 0.55/1.52)` = **1.23x**. Not 2x. `program.md` treats 2.00x as an
aspirational research target, not a promotion threshold, so the work proceeded
under a stated expectation of a `PROMOTABLE FRONTIER`, not a `WINNER`.

## Mechanism

Variant 3 (shipped) leaves seven of eight warps idle for the whole ~13-16us that
`e62_tri_inv32` spends on warp 0. Two phases in the same iteration read only the
panel output and never touch `Qi`:

| | reads | writes |
|---|---|---|
| `e62_tri_inv32` | `S[kk:kk+32, kk:kk+32]` (`const __restrict__`) | `Qi`, `Tp` |
| phase 4 stage P | `S[lwid:, kk:kk+32]` | `P` |
| phase 6 trailing | `P` | `S[lwid:, lwid:]` |

All three footprints are disjoint, so warp 0 can build the inverse while warps
1-7 run staging and the trailing update, turning `sum` into `max`.

Warps 1-7 need a barrier between staging and the trailing update that **warp 0
must not join** — `__syncthreads()` there would hang on warp 0, which is inside
the inverse. That is the named barrier:

```
#define E62_OVT ((E62_NW - 1) * 32)          // 224 threads
#define E62_BAR(id, cnt) \
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt) : "memory")
```

`bar.sync` id 1 with an explicit participant count is the hardware primitive for
a partial-block barrier. Phases 4 and 6 are then skipped in their original
positions via `if (VAR != 4)`.

**No arithmetic changed.** Only which warp does already-existing work, and when.
That is why `abs_err` and `inv_err` come out bit-identical to the control.

## Kernel result — `results/exp065-probe-v1.json`

| phase | v3 (control) | v4 (overlap) | cycles |
|---|---:|---:|---|
| load | 2.636 | 2.467 | 6191 -> 6172 |
| chain | 13.147 | 12.413 | 30876 -> 31060 |
| triinv | 15.762 | 16.122 | 37019 -> 40341 |
| stageP+Qt | 4.661 | **3.169** | 10948 -> 7929 |
| commit | 1.453 | 1.502 | 3413 -> 3759 |
| trailing+inv | 6.282 | **2.432** | 14755 -> 6085 |
| store | 1.724 | 1.636 | 4048 -> 4093 |
| **block** | **45.669us** | **39.742us** | **356.8 -> 310.5 ns/row** |

**1.149x on the block.** The counters confirm the mechanism exactly: staging and
the trailing update leave the serial timeline (their cycles drop by 3019 and
8670), while `triinv` grows 2% from shared-memory contention with the seven warps
now working behind it. That contention is the price of the overlap and it is
small.

Correctness identical to the control: `abs_err 4e-07`, `inv_err 6e-08`.

## Full 15-shape paired grid — `results/exp065-v1-pairedgrid.json`

**geomean 1.0122**, CI95 [1.0112, 1.0133], excludes 1.0, `all_shapes_ok: true`,
`passed: true`.

| shape | baseline us | candidate us | ratio |
|---|---:|---:|---:|
| 2x2048 | 1080.7 | 1035.4 | **1.0449** |
| 2x4096 | 2297.8 | 2207.3 | **1.0408** |
| 4x1024 | 554.6 | 534.5 | **1.0376** |
| 16x512 | 309.9 | 301.6 | **1.0287** |
| 8x2048 | 1292.7 | 1257.1 | **1.0277** |
| 4096x32 | 20.92 | 20.74 | 1.0098 |
| 640x512 | 1300.3 | 1297.6 | 1.0021 |
| 1x32768 | 22853.7 | 22850.0 | 1.0001 |
| 1x16384 | 8387.2 | 8383.4 | 1.0001 |
| 1x8192 | 5781.2 | 5781.5 | 1.0000 |
| 64x256 | 113.66 | 113.62 | 1.0000 |
| 1x4096 | 1537.5 | 1538.0 | 0.9994 |
| 256x128 | 72.92 | 73.00 | 0.9992 |
| 60x1024 | 1198.9 | 1198.8 | 0.9991 |
| 1024x64 | 33.90 | 34.04 | 0.9959 |

Every shape's `candidate_counters` equals its `baseline_counters` and
`new_fallbacks` is empty throughout — positive proof the intended backend ran on
all fifteen, not a fallback.

`1024x64` at 0.9959 is the only shape whose CI sits below 1.0. It does not use
this kernel (`_CUDA64_HITS`), the deviation is 0.41% against a measured A-vs-A
spread of 1.50%, and the grid CI excludes 1.0 positively. Recorded, not material.

## Six-family correctness — baseline-attributed

`results/exp065-v1-familygrid.json` vs `results/exp065-base-familygrid.json`,
48 rows over n = 512/1024/2048/4096 x dense/spectrum/diagonal/lowrank/rowscale/
tridiagonal:

- **48/48 `checker_ok: true` on the candidate, 48/48 on the baseline.**
- **Zero differences** in `ok`, `checker_ok`, `active_backend`, or `fallbacks`
  between baseline and candidate on any row.

`passed: false` on both is the harness reporting that safety fallbacks engage on
spectrum/lowrank and that `1x4096` runs no custom backend at all. That is the
baseline's existing design, reproduced exactly. Attributing it cost one extra
Modal run and is the only way the parity claim is honest.

## Gates

| gate | artifact | result |
|---|---|---|
| py_compile / whitespace | — | ok, `git diff --check` clean |
| source policy | — | `cusolver` 2, `cuSOLVER` 18, `stream` 0, `queue` 5 — identical to baseline |
| `load_inline` count | — | 7 -> 7 |
| kernel probe | `exp065-probe-v1.json` | 1.149x block, err identical to control |
| full 15-shape paired grid | `exp065-v1-pairedgrid.json` | **1.0122** CI95 excludes 1.0, 15/15 |
| counter diff | same | no new fallbacks on any shape |
| six-family, baseline-attributed | `exp065-{v1,base}-familygrid.json` | 48/48 checker_ok, **0 diffs** |
| cold build | Popcorn `#914336` | succeeded, 7s |
| Popcorn `--mode test` | `#914336` | **passed** |
| Popcorn ranked | `#914341` | in flight |

## Ranked outcome — `#914341` (REJECTED, not adopted)

| | `#913511` (baseline) | `#914341` | delta |
|---|---:|---:|---|
| public | 672.383us | **646.868us** | **-3.79%** |
| secret | 655.423us | **692.860us** | **+5.71%** |

All six Popcorn runs passed. Public improved by more than three times what the
paired grid predicted (1.0122 = 1.21%); **the secret split regressed 5.71%**.
Promotion requires both, so the candidate is not adopted and root
`submission.py` stays on `#913511`.

**No ranked retry.** `program.md` permits one only after a *concrete defect* is
found and fixed. There is no defect to fix here: the arithmetic is unchanged by
construction, `abs_err`/`inv_err` are bit-identical to the control, the
six-family gate is byte-identical to the baseline on 48/48 rows, and the paired
grid measured every one of the fifteen shapes faster-or-flat with no fallbacks.
Retrying an unchanged performance rejection is explicitly forbidden.

### This is the third instance of a documented public/secret divergence

It is not new, and the pattern has both signs:

| experiment | paired grid | public | secret |
|---|---|---|---|
| exp 022 | 1.0052 | regressed | improved |
| exp 035 (`#888352`) | +0.61% | -2.69% | +4.50% |
| **exp 065 (`#914341`)** | **+1.22%** | **-3.79%** | **+5.71%** |

Three points now say the same thing: **on this leaderboard a paired-grid
geomean in the 0.5-1.5% band does not predict the sign of the secret split.**
The device-time evidence here is unusually strong — CI95 [1.0112, 1.0133]
excluding 1.0, 15/15 shapes ok, identical counters, zero new fallbacks — and it
still did not survive. The secret split evaluates inputs this repository cannot
see, and a change that is a clear device-time win on the fifteen public shapes
can land the other way there.

**Operational consequence for `program.md`:** a sub-1.5% paired grid is not
sufficient evidence to spend a ranked submission. Either raise the paired-grid
bar for spending a ranked slot, or accept that submissions in that band are
coin-flips on the secret split. That is a policy decision for the owner, not
one to make silently.

## Classification

`FRONTIER` — correct and faster in device time on all fifteen shapes, clears
every local and Modal promotion gate, **rejected at the leaderboard** on the
secret split. Preserved in `ship-v1.py` for a future rebase; the kernel change
itself is sound and independently reusable.

## Next lever, with the number that justifies it

`triinv` is now **40.6% of the block** (16.122us) running on **one warp** while
the other seven have only ~5.6us of work to hide behind it — they idle ~10.5us.
The overlap has converted the problem from "seven idle warps" into "one warp is
the critical path". The next move is therefore to attack the triangular inverse
itself, not to schedule around it:

1. **Parallelise `e62_tri_inv32` across two or four warps.** It already splits
   into two independent 16x16 halves (`base = (lane<16) ? 0 : 16`), so the
   structure is there. Exp 063 rejected *fusing* the inverse into the chain
   (v1/v2, both slower), which is a different move and does not rule this out.
2. **Cross-iteration look-ahead.** Overlap the deferred part of the trailing
   update with the *next* iteration's panel chain, prioritising only the 32
   columns the next panel needs. This is the schedule exp 063 costed at ~25us/
   block; the intra-iteration overlap shipped here captures roughly half of it.

Ceiling check before either is attempted: if `triinv` went to zero the block
would be ~23.6us (1.61x over v3), which propagates to roughly 1.05-1.08x on the
five affected shapes and ~1.02x on the grid. Worth one bounded attempt, not six.
