# Levers — what to try next

**Loop stage [O2] pick targets, [I1] pick a lever.** This is the single place a
shape worker looks before choosing what to build. It has two axes on the same
question:

- **Part 1** is per **shape**: which levers are shipped, rejected, or untried on
  each of the fifteen ranked shapes.
- **Part 2** is per **lever**: the structural ladder ported from the QR project,
  with the measured ROI of the steps this repository has not yet harvested.

For what has already been *measured*, see `docs/experiments.md`. For the loop
itself, see `program.md`. For hard-won rules, see `docs/lessons.md`.

Cells in Part 1: **✓** shipped / current winning path · **✗** tried and rejected,
or not applicable · **TBD** plausible, not yet conclusively tried.

> The "current adopted source" line below is historical (it names `#890798`).
> The live incumbent is in `docs/STATUS.md` — that is the only place it is
> maintained.

---

## Part 1 — Per-shape tracker (living: update on progress/regress)

Rows = the 15 ranked B200 shapes. Columns = latency-reduction levers. Cells:

- **✓** — shipped / current winning path for that shape (see the referenced session).
- **TBD** — plausible lever, not yet conclusively tried (a path worth exploring).
- **✗** — tried and rejected, or not applicable / no expected benefit for this shape.

Current adopted source: **`#890798` = 801.977μs public / 847.836μs secret
(Session 43, exp 047)**. It improves `#890659` by 0.504% public on a paired
grid of 1.012106x. Exact SHA-256:
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
Previously: **`#890659` = 806.037μs public / 806.396μs secret (Session 42,
exp 046)**.
Previously: **`#890089` = 810.246μs public (Session 41, exp 045
= exp 044 carried onto the 64x256 winner `#890037`)**. Previously: **`#889994`
= 852.746μs public / 847.396μs secret (Session 40)**. Previously: **`#888996` = 916.5768129471865μs public /
863.8500740634134μs secret** (Session 39). `#888867` remains the best public
score at 899.124686138768μs; `#888996` is adopted because its same-process full
grid improves 1.04787x and its secret score improves 4.812%. Experiments 039,
041, and 042 replace all three campaign shapes with cuSOLVER-free CUDA kernels;
every stage-specific control exceeds 2x.
`nb` = block size.

| Shape (b×n) | Batched cuSOLVER | Per-matrix loop | Triton kernel | Custom CUDA (tcgen05/CUTLASS) | Blocked / tiled | TF32 trailing | BF16x9 FP32-emu | FP8 / MXFP8 + iter-refine | CUDA Graphs |
|---|---|---|---|---|---|---|---|---|---|
| 4096×32  | ✗ | ✗ | ✗ superseded (S16b/S22) | **✓ rank-2 warp CUDA** (S36, 2.269×) | ✗ | ✗ | ✗ | ✗ | ✗ graph copy cost (S16b) |
| 1024×64  | ✗ (S15) | ✗ | ✗ (S2/S15 0.67×; S28 split32 route 0.998×) | **✓ two-warp rank-2 CUDA** (S38, ~3.80× end-to-end) | ✗ | ✗ | TBD | TBD | ✗ superseded (S15) |
| 256×128  | ✗ | ✗ | ✗ split32 superseded (S28) | **✓ eight-warp blocked-16 CUDA** (S39, 2.216× stage control) | **✓ FP32 rank-16** (S39) | ✗ (tf32x3) | TBD | TBD | ✗ superseded (S39) |
| 64×256   | ✗ (S15) | ✗ | ✗ superseded (S21) | **✓ packed-tile CUDA/WMMA** (S41, 2.018×) | **✓ FP32 rank-16** (S41) | **✓ TF32 WMMA + FP32 retry** (S41) | ✗ | ✗ | ✗ superseded (S41) |
| 16×512   | ✗ | TBD | **✓** (S21, panel-inner 64×64); fused resident panel locally gated but B200-pending (S45/exp049) | ✗ full-resident cluster, one-CTA persistent, atomic CTA groups, and rank-128 superpanels (S45; best 0.697×) | **✓** (S21) | ✗ (tf32x3) | TBD | TBD | ✓ (S9→S15 in-path) |
| 640×512  | ✗ (S5/S15) | ✗ (S5) | ✓ panel-inner (S21) + **✓ CUDA rank-4 diagonal micro** (S40, 1.098×) + **✓ fused resident panel** (S43, 1.098×) | ✗ e62_diag128 enroll (exp066, **0.606× / 1.651× slower** — occupancy: batch 640 = ~4.3 waves; do NOT re-enroll) | **✓** (S21) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 4×1024   | ✗ | ✗ (S15) | **✓** (S20, panel-inner 64×64) | ✗ CUDA micro not graph-capturable (S40b); cooperative + cluster/DSM persistent paths rejected (S44/exp048, best dense 1.167× and family-invalid) | **✓** (S20) | **✓** TF32 (S15); ✗ persistent FP16 trailing (S44/exp048, 0.883×) | TBD | TBD | ✓ (in-path S15) |
| 60×1024  | ✗ (S15) | ✗ (S4) | ✓ (S15, 1.99×) + fused resident panel + merged diag step (S43, 1.092×, superseded by e62 pending) | **✓ CUDA rank-4 diagonal micro** (S40, 1.106×); **◐ e62_diag128 enroll** (exp067, paired **1.2426×** vs the S43 route; ranked #922201 public+secret passed, adoption PENDING secret score) | **✓** (S15) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 2×2048   | ✗ | **✓** (S4) | ✗ (S15, 0.65×) | ✗ (S35, cluster 0.063–0.595×) | ✗ (S15/S35) | TBD | TBD | TBD | TBD |
| 8×2048   | ✗ (S9) | ✗ (S5) | **✓** (S20, panel-inner 64×64, 1.055× vs S19); ✗ fused resident panel (S43, 0.907× — needs uniform NB=128, loses exp 032's NB=256) | ✗ | **✓** (S20) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 1×4096   | **✓** | — | ✗ (S15 cand-B 0.18–0.97×) | ✗ cooperative six-path bound (S37, best 0.376×) | ✗ (S15) | TBD | TBD | TBD | ✗ (S15, 0.97×) |
| 2×4096   | ✗ | **✓** (S4) | ✗ (S15 cand-B) | ✗ | ✗ (S15) | TBD | TBD | TBD | TBD |
| 1×8192   | **✓** | — | ✗ | ✗ | ✗ 1.07× (S6) | ✗ 1.07× (S6) | ✗ 0.95× (S7) | TBD | ✗ |
| 1×16384  | ✗ | — | ✗ | ✗ | **✓ left-looking** (S10) | **✓ active-panel** (S10) | ✗ 1.15× (S7) | TBD | ✗ |
| 1×32768  | ✗ | — | ✗ (S13/S14 no-cusolver potrf) | ✗ (S13/S14) | **✓ left-looking** (S10) | **✓ diagonal** (S10) | ✗ (S7, extrap.) | **✓ native FP8 panel + fused quantization** (S10/S14, 1.084×) | ✗ (S13 two-level) |

Notes: **CUDA streams** win several launch-bound shapes but are **banned** by
popcorn's static source scan (S4/S6) — not a column. FP16/BF16 (plain, not
BF16x9) were tried in the blocked path and **lost to TF32** on B200 (S6), so
they're folded into the TF32 result rather than tracked separately.

**Panel precision / width (S30).** `8×2048` now factors with an NB=256 uniform
panel schedule (halves the panel count; the one shape where fewer launches beat
the `_trailing_nb` spill — L2/exp 032). `4×1024`, `60×1024`, `8×2048` now use
plain **tf32 (1-pass) panels** instead of tf32x3 (L4/exp 033); safe only at large
n because the `20·n·eps·‖A‖` gate grows with n (smaller shapes lack margin —
256×128 dense *fails* under tf32 panels). **fp16x3 emulated-fp32 panels rejected**
(register spill in the tight panel kernels). Panel precision/width is not yet a
tracker column; recorded here.

**Per-call fixed overhead** (copy-in/clone-out ~9μs + finite-check chain
~12–15μs) is a top-3 cost on every sub-400μs shape (S28) but is not yet a
column, because no variant has been *measured* — S29's cheap finite check was
refuted on a free gate before any GPU spend. Standing constraint for this
lever: the finite check may be made cheaper but **never weaker**. Shrinking it
to the last diagonal entry is invalid — `finite/Inf == 0`, so an overflowed
pivot is absorbed into a zero column and never reaches `L[n-1][n-1]` (S29). The
open variant is an in-kernel flag written at *pivot* time, which is both cheaper
and strictly stronger than the shipped full-diagonal reduction.

### Transfer opportunities — build and leaderboard-test queue

Build each opportunity in priority order against the exact current ranked
source. Run free checks, paired same-process B200 timing, all six families for
every changed shape, and the full 15-shape grid. Only an aggregate improvement
may proceed to Popcorn test 17/17 and then exactly one leaderboard submission;
leave losing shapes on their current ranked route.

| Priority | Technique with positive evidence | Proven shapes | Untried transfer targets | Expected opportunity |
|---:|---|---|---|---|
| 1 | 64×64 panel-inner subtiling | `4×1024` **1.089×**, `8×2048` **1.055×** (S20); `64×256` **1.047×**, `16×512` **1.078×**, `640×512` **1.128×** (S21) | None | **COMPLETED.** S21 ranked at `#882958`; `60×1024` was measured and excluded after conflicting `1.055×` isolated / `0.977×` grid evidence. |
| 2 | Rank-4 pivot processing | Six split32 shapes **1.05–1.26×** (S17); standalone `4096×32` **1.077×** Modal (S22) | None | **COMPLETED, NOT ADOPTED.** Ranked `#882969` regressed public 1.510% while improving secret 1.440%; current source stays rank-2. |
| 3 | Reciprocal multiply replacing divides | `4×1024` **1.007×**; `8×2048` **1.005×** (S19) | None | **COMPLETED, REJECTED at `60×1024`.** Two correct probes measured `1.007×` then `0.994×`; below route noise, so no LB run. |
| 4 | Dynamic E4M3 panel products with fused quantization | `1×32768` **1.373×**, then another **1.084×** from fused amax/quantization (S12/S14) | None | **COMPLETED, REJECTED at `1×16384`.** S24 passed 6/6 but measured `0.997×`; ~1.17ms quantization overhead erased the gain. |
| 5 | FP8 or MXFP8 trailing Schur updates | FP8 wins at `1×32768`; FP16 trailing wins on five split32 shapes | MXFP8/refined variants remain | **FP8 ARCHITECTURE REJECTED.** S25 native tile-local E4M3 at `8×2048` was incorrect/fallback-only (0.513× invalid timing); no LB run. |
| 6 | Recursive triangular inversion | `1×16384` **1.055×**; `1×32768` **1.028×** (S16a/S17) | None | **COMPLETED, REJECTED at `1×8192`.** Clean `nb=2048` isolation passed 6/6 but measured `0.954×` (S26). |
| 7 | First-touch eager execution | Strongest at `640×512`; also shipped at `60×1024` (S17) | None | **COMPLETED, REJECTED at `8×2048`.** S27 passed 6/6 but measured `0.336×`; `4×1024` has less copy traffic and was closed by the stronger negative proxy. |
| 8 | Graph-captured per-matrix loop | Graph replay wins at `1024×64`, `256×128`, and blocked mid shapes | Ineligible | **NOT BUILT:** would create a new cuSOLVER-based fast path for `2×2048`/`2×4096`, prohibited by the standing owner boundary. |

Do not reopen already measured losses as transfers: rank-4 split32 at
`2×2048`/`2×4096` (**0.764×/0.784×**), split32 at `1024×64`/`256×128`
(**0.788×/0.904×**), graphed `4096×32` (**0.845×**), FP8 panels at
`1×8192` versus its faster TF32 path, or fixed-scale FP8-shadow stacks at
16384/32768 (≤1.0×).

### Blackwell B200-specific candidate solutions (the "what else")

These are the levers that are *specific to* / most leveraged by the B200
(sm_100, 5th-gen tensor cores, `tcgen05.mma`, block-scaled MX formats). Ordered
roughly by expected ROI given the loose accuracy gate (`‖A−LLᵀ‖₁ ≤ 20·n·eps·‖A‖₁`,
which *grows with n* → the huge shapes have the most numerical headroom).

1. **BF16x9 FP32 emulation on tensor cores** — ✗ **REJECTED (S7).** Engages on the
   B200 via `CUBLAS_EMULATE_SINGLE_PRECISION=1` + `CUBLAS_FP32_EMULATED_BF16X9_MATH=1`
   (the BF16X9 var alone does nothing; the PyTorch `fp32_precision` knob only exposes
   ieee/tf32, no BF16x9). Confirmed engaged (standalone 8192 FP32 matmul 16.7→6.3ms,
   2.6×) and ≈FP32-accurate (margins 65k–139k× vs TF32's ~100–210×; and *robust* where
   TF32 NaNs on lowrank). **But slower than the shipped paths:** 8192 0.95× vs cuSOLVER,
   16384 bf16x9 1.15× vs TF32's 1.60×. Reason: BF16x9 ≈ 6–9 BF16 products per FP32
   GEMM, so ~3× slower than a single-product TF32 GEMM — TF32 tensor cores are the real
   bar, not native FP32. Speed order TF32 > BF16x9 > native FP32. See exp 007.
2. **FP8 / MXFP8 trailing update + mixed-precision iterative refinement** —
   ◐ **PARTIALLY SHIPPED (S10/S14).** Native E4M3 FP8 panel products with FP32
   accumulation in the left-looking `1×32768` path passed all six families and
   improved the exp-009 path by **1.373×**. Experiment 014 fused both operands'
   tiled amax reductions and E4M3 scale/cast passes, improving that shipped path
   another **1.084×** and producing ranked `#880770`. The dense scaled residual
   remains 4.52/20 (22.6% of tolerance). MXFP8, iterative refinement, and an FP8
   path for `1×16384` remain genuinely untested.
3. **CUTLASS 3.x Blackwell fused kernel (`tcgen05.mma`, TMA, 2-SM MMA)** — a
   warp-specialized collective kernel that fuses panel + trailing SYRK, using the
   Tensor Memory Accelerator for async bulk copies and CTA-pair (2-SM) MMA. Beats
   the PyTorch-op blocked path by avoiding per-step launch + global-memory
   materialization. See CUTLASS example 78 (`blackwell_emulated_bf16x9_gemm`).
   High effort. *Target: large-n; possibly a real n=64/128 blocked kernel.*
4. **CUDA Graphs (legal — not streams)** — capture the many small launches in the
   blocked path and the per-matrix loop into a graph to amortize launch overhead.
   Cheap, shippable, Blackwell-agnostic but complementary. *Target: launch-bound
   small-n shapes, the Python-loop blocked path, 1×32768 (2-level).*
5. **Latest cuSOLVER + expert `cusolverDnXpotrf[Batched]` API** — ensure the
   toolkit ships Blackwell-tuned `potrf`; try algorithm selection / the 64-bit
   expert API vs `cholesky_ex`. Low effort, may quietly improve mid-batch shapes.
6. **Two-level blocked scheme for 1×32768** — recurse the diagonal `potrf` so the
   FP32 diagonal factorization also becomes tensor-core work. Diminishing returns
   (S6) but 32768 is ~76% of the clock, so even a few % moves the geomean most.
7. **Thread-block clusters / distributed shared memory (sm_90+/sm_100)** — a
   cluster-wide-shared-memory panel kernel could finally crack the mid-n shapes
   (n=256–1024) currently stuck on saturated cuSOLVER. Speculative.

---

---

# Part 2 — The QR transfer ladder

## Why this transfers

QR and Cholesky on this hardware share the load-bearing property: **the serial
panel/diagonal factorization is the bottleneck, not the trailing GEMM.** That is
recorded independently on both sides — `lessons_qrproblem.md` states it as the
universal finding across every size, and the memory note
`diagonal-potrf-is-the-large-n-bottleneck` measures `getrf_wo_pivot` at 52%/36%
of the two large Cholesky shapes against 7% for MXFP8 trailing work.

So the QR ladder is not an analogy. It is a list of levers already proven
against this bottleneck on this GPU under this scoring rule.

## Ladder status

| # | QR structural change | QR geomean | Cholesky equivalent | Status |
|---|---|---|---|---|
| 1 | `torch.geqrf` everywhere | >108.8k us | cuSOLVER `potrf` everywhere | shipped (baseline) |
| 2 | Blocked WY on one shape | 108.8k us | Left-looking blocked on 1x16384 | shipped (S10) |
| 3 | Blocked route on all shapes | 10.2k us | Blocked route + per-shape dispatch | shipped (S15/S20/S21) |
| 4 | Triton panels + grouped updates | 4.3k us | Triton panel-inner 64x64 | shipped (S20/S21) |
| 5 | Cholesky-ORHR for n4096 | 4.0k us | Recursive / blocked inverse | partial — `rec_inv` is n>=16384-only (exp 065) |
| 6 | CUDA graph replay | 3.4k us | CUDA Graphs | **rejected here** — graph copy cost (S16b), not capturable (S40b) |
| 7 | Fused V/T assembly: no slice copies, cats, temporaries | 2.75k us | Per-call copy-in/clone-out + finite-check chain | **UNMEASURED** — see below |
| 8 | split16 panels + tail-Gram | 2.5k us | Variable NB near the trailing edge | UNTRIED |
| 9 | Fixed-shape kernel specialization | 2.0k us | Per-shape custom CUDA | shipped for n<=256 only; mid/large shapes still run generic runtime-`n` code |
| 10 | Composed superpanels, direct-H returns | 1.80k us | Superpanel composition, direct-L return | rejected — rank-128 superpanels 0.697x (S45) |

Steps 1-5 are fully harvested here. Steps 6, 7, 9 are not, and in QR those three
carried 4.3k -> 2.0k us: **more than half the total gain**.

## Lever 7 is the largest unmeasured item on the board

`journal.md` records the cost and then stops:

> Per-call fixed overhead (copy-in/clone-out ~9us + finite-check chain ~12-15us)
> is a top-3 cost on every sub-400us shape (S28) but is not yet a column,
> because no variant has been measured.

Sensitivity against the exp059 full grid (`experiments/059-two-large-incremental/combined-v3-fullgrid.json`,
`baseline_us`), removing a flat per-call constant from every shape:

```
 4096x32      21.9us   <-- total shape cost is BELOW the recorded overhead
 1024x64      35.3us
  256x128     73.8us
   64x256    115.1us
   16x512    405.1us
  640x512   1355.5us
    4x1024   713.8us
   60x1024  1282.0us
    2x2048  1358.3us
    8x2048  1598.4us
    1x4096  1536.0us
    2x4096  3210.3us
    1x8192  5804.5us
    1x16384 15058.8us
    1x32768 42331.5us
 geomean     873.2us

 remove  9us/call ->  811.2us   1.077x   -7.1%
 remove 15us/call ->  754.5us   1.157x  -13.6%
 remove 21us/call ->  668.7us   1.306x  -23.4%
```

Because the score is a geomean, an additive constant is worth more than any
multiplicative win on 1x32768 — the shape that has absorbed most recent effort.

Three caveats, so the band is read as a band:

- The ~21us figure is from S28. Exp 061/062 have since attacked overhead on the
  large shapes, so the surviving constant today is likely nearer the 9us end.
- Full removal is not reachable. The finite check is a correctness gate. S29's
  standing constraint holds: **cheaper is allowed, weaker is not** — shrinking it
  to the last diagonal entry is invalid because `finite/Inf == 0`, so an
  overflowed pivot is absorbed into a zero column and never reaches `L[n-1][n-1]`.
- The local grid geomean (873.2us) is not the leaderboard geomean (733.5us
  public, ranked #909269). Use it for lever ranking, not for score prediction.

Even the floor of the band is a 7% geomean win on a lever with zero measured
variants against it.

### First probe

Cheap enough that it is not a research program:

1. Free gates only: fuse the finite check into the tail of the final
   factorization kernel instead of running it as a separate launch chain, and
   remove the clone-out wherever the output buffer can be written in place.
2. One `pairedgrid` run restricted to the four smallest shapes
   (4096x32, 1024x64, 256x128, 64x256), where the constant is the majority of
   the cost and the signal is largest.
3. Only if that clears, expand to the full grid.

## Lever 9 for mid shapes

Fixed-shape specialization shipped for `n<=256` (S36/S38/S39/S41, all custom
CUDA) but the mid and large shapes still run generic code with runtime `n`. QR
took 2.5k -> 2.0k us from hardcoding dimensions and fusing reductions once the
shape was known at compile time. The mid shapes are exactly where
`mid-shape-cusolver-headroom` says the geomean is still 19-260x above hardware
floors.

## Lever 6 deserves one re-test, not a reopen

CUDA Graphs were rejected here on graph copy cost (S16b) and on the CUDA micro
not being capturable (S40b). Both rejections predate the current overhead
structure. If lever 7 removes the copy-in/clone-out, the graph copy cost that
sank S16b is a different number. Re-test only after lever 7 lands, not before.

## Loop hygiene (from the autoresearch post)

Recorded here as process backlog, not adopted into `program.md`:

- **Beam search.** The post's central anti-stall device is keeping 3-5 candidate
  idea families alive rather than hill-climbing one incumbent, so structural
  changes that start slower get time to mature. `program.md` is single-incumbent
  by construction (step 11 rebases onto the latest winner; step 9 classifies
  anything slower as `REJECTED`). Exp 048's V2 won 1.167x and was discarded —
  the exact failure mode beam search exists to prevent. Changing this touches the
  promotion rules, so it needs an explicit decision.
- **Cleanup cycles.** `submission.py` is ~141KB and `journal.md` ~143KB. The post
  calls out periodic archiving/refactoring; the submission size also feeds
  directly into the cold-build budget (`program.md` step 13).
- **Cycle time.** 1x32768 is 42.3ms of the 74.9ms full-grid run (~57%); the two
  largest shapes together are ~77%. Gating them behind a 13-shape pass would cut
  integration latency substantially without weakening the final gate, since a
  candidate that regresses a small shape never needs the expensive shapes run.
