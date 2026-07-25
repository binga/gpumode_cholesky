# Plan — 40% latency reduction on every shape

**Goal parameter.** `-40%` wall clock on **each** of the 15 ranked shapes
(1.667x per shape). Because the score is an equal-weight geomean, hitting it
everywhere is also a 40% score improvement: public **733.540us -> ~440us**.

Unlike a "N% from k shapes" goal, this one does *not* amplify:
`(1/(1-N))^(15/k)` with k=15 is just 1.667x. Every shape carries its own
weight, so the work is broad rather than deep.

**Baseline.** Ranked `#909269` (public 733.540us / secret 721.821us), source
sha `f408a020ea94…`, which is the repository root `submission.py`. Per-shape
walls below are the paired-grid baseline of `#907267` except `2x2048` and
`2x4096`, which carry the `#909269` values.

---

## 1. The board

Floor = `max(2·b·n²·4 B / 7 TB/s, b·n³/3 / 600 TFLOP/s tf32)`.

| shape | current us | −40% target | floor us | x floor | dominant cost |
|---|---:|---:|---:|---:|---|
| 4096x32 | 21.6 | 13.0 | 4.8 | 4.5 | 7.5us wall−device gap |
| 1024x64 | 34.1 | 20.5 | 4.8 | 7.1 | 11.4us wall−device gap |
| 256x128 | 72.9 | 43.7 | 4.8 | 15 | `cholesky128_block16` 60.6 |
| 64x256 | 115.2 | 69.1 | 4.8 | 24 | `cholesky256_wmma16` 97.1 |
| 16x512 | 408.6 | 245.2 | 4.8 | 85 | micro potrf 217.5 (54%) |
| 640x512 | 1307.1 | 784.3 | 191.7 | 6.8 | fused panel 361 + 198 idle |
| 4x1024 | 716.8 | 430.1 | 4.8 | 150 | micro potrf 436 (63%) |
| 60x1024 | 1217.4 | 730.4 | 71.9 | 17 | trailing 387 + 182 idle |
| 2x2048 | 1195.0 | 717.0 | 9.6 | 125 | exp062 diagonal blocks 805 |
| 8x2048 | 1614.7 | 968.8 | 38.3 | 42 | micro potrf 913 (55%) |
| 1x4096 | 1541.7 | 925.0 | 38.2 | 40 | vendor getrf 1385 (91%) |
| 2x4096 | 2541.0 | 1524.6 | 76.4 | 33 | exp062 diagonal blocks 1609 |
| 1x8192 | 5821.7 | 3493.0 | 305.5 | 19 | vendor diagonal ~2700 |
| 1x16384 | 8847.7 | 5308.6 | 2444 | 3.6 | vendor diagonal ~5400 |
| 1x32768 | 24317 | 14590 | **19550** | **1.24** | trailing GEMM |

**Read the last two rows carefully.** `1x32768` is only 1.24x off its *TF32
arithmetic* floor, and the −40% target of 14,590us is **below** that floor. No
scheduling change can reach it; it requires moving the bulk of the trailing
update to fp16/FP8 (fp16 floor ~9,300us at the measured 1262.7 TFLOP/s).

---

## 2. The lever that carries most of the board

**Finish the resident diagonal-block kernel** (`e62_diag128`, shipped in
`#909269`). The pivot chain is 54-63% of every mid shape and 46-57% of every
large shape.

Measured this session:

| quantity | value |
|---|---:|
| register-resident pivot chain, isolated | **63.3 ns/pivot** |
| exp-050 fused block (previous repo best) | 134 ns/pivot |
| vendor `getrf_wo_pivot` | ~330 ns/row |
| **current whole 128x128 block** | **375 ns/row (48-50us)** |

So ~80% of the block is non-chain overhead, and the round-4 `clock64` phase
table says exactly where:

| phase | us | share |
|---|---:|---:|
| chain | 14.9 | 31% |
| triinv | 14.6 | 30% |
| trailing+inv | 5.8 | 12% |
| stageP+Qt | 4.5 | 9% |
| load | 3.5 | 7% |
| panel | 2.4 | 5% |
| store | 1.7 | 3% |
| commit | 1.1 | 2% |

Three changes, in order:

1. **Fold the triangular inverse back into the chain.** Round 1's Gauss-Jordan
   computed L and `inv(L)` in one pass and was numerically correct (inverse
   error 2.4e-07); it was split out only while chasing a different bottleneck.
   Removes most of the 14.6us triinv phase at a partial cost to the chain.
   Target chain+inverse combined: **~18-20us** against today's 29.5us.
2. **Overlap the serial and parallel phases with named barriers**
   (`bar.sync` with distinct barrier IDs, or `cooperative_groups` tiled
   partitions). Warps 1-7 update *only* the next 32x32 pivot tile, then warp 0
   starts the next chain while they finish the rest of the trailing update.
   Converts `sum(chain, phases)` into `max(chain, phases)`.
3. **Larger resident block.** Folding the inverse into the chain frees the
   128x132 `M` buffer (~87 KB), which buys a 192x192 or 256x256 resident block
   and amortises load/store and launch count over more rows.

**Target: 375 -> ~120 ns/row (≈20us per 128 block).** That single number is
what turns exp 062's +2.6% into 40% on eight shapes at once.

---

## 3. Cross-cutting levers

- **L1 — kill the end-of-call sync.** `custom_kernel` closes with
  `torch.isfinite(...).all().item()`, which drains the GPU and exposes the next
  call's dispatch (exp 050 measured ~7.6us per exposed eager launch). Worth
  7.5us on `4096x32`, 11.4us on `1024x64`, and ~190us each on `640x512` and
  `60x1024`. Fix: have the factorization kernel write the finiteness flag into
  a device buffer and defer the read, or drop the check where the kernel is
  exact FP32 and cannot produce NaN.
- **L2 — fp16 trailing updates at n >= 1024.** exp 061 measured fp16 GEMM at
  **1262.7 TFLOP/s vs tf32 736.9** (1.71x) with residual identical to five
  significant figures. Applies to `_trailing_nb`, the exp062 driver's
  `baddbmm_` calls, and the large-n panel applies.
- **L3 — `float4` global I/O everywhere.** Measured 12.35 -> 3.52us on the
  block kernel's load phase alone.
- **L4 — occupancy.** Keep shared-memory footprint small enough that many CTAs
  co-reside; several small-shape kernels are latency-exposed at 1 CTA/SM.

---

## 4. Per-shape work items

### 4096x32, 1024x64 — overhead-bound
Wall−device gaps are **35% and 33% of total time**, so L1 is nearly the whole
target. Then `float4` I/O and enough CTAs/SM for a single wave. Chain is only
2.0us / 4.1us; the 4.8us memory floor dominates.

### 256x128 — easiest large win
A 128x128 FP32 matrix is 64 KB: **one CTA holds an entire matrix resident**.
256 CTAs over 148 SMs = 2 waves x (128 pivots x 63.3 ns) ≈ 16us + 4.8us I/O
≈ **21us against 60.6us today** (~3x).

### 64x256 — two-level resident
256² = 256 KB exceeds shared, so 2 resident 128-blocks plus in-CTA panel and
trailing. Chain 256 x 63.3 ns = 16.2us. Only 64 CTAs (43% SM occupancy) —
consider splitting each matrix across 2 CTAs. Estimate ~40us vs 97.1us.

### 16x512, 4x1024, 8x2048 — retire the split32 chain
These run 16/32/64 separate `_micro_potrf_gj32` calls at 13.6-14.3us *plus*
separate `_panel_apply32` and `_panel_inner32` launches. The fused 128-block
kernel collapses 7 launches per block into 1. At 20us/block:
16x512 ≈ 240us (−41%), 4x1024 ≈ 420us (−41%), 8x2048 ≈ 950us (−41%).
**Section 2 alone hits target on all three.**

### 640x512, 60x1024 — idle + trailing
198us and 182us of pure idle (14.8%), plus `_trailing_nb` at 195/387us.
L1 + L2 together ≈ 40% for both.

### 2x2048, 2x4096 — finish the kernel
At 20us/block: 2x2048 = 16x20 + 330 = **650us (−46%)**;
2x4096 = 32x20 + 850 = **1490us (−41%)**. Add L2 for margin.

### 1x4096 — port onto the exp062 path
Still 91% vendor. batch=1 means no chain amortisation, so it needs both
section 2 and L2: 32x20 + ~450 ≈ 1090us, then fp16 -> ~950us (−38%).
Borderline; the deepest lever of the mid group.

### 1x8192, 1x16384 — recursive resident diagonal
A 2048-wide diagonal block cannot be resident (16 MB), but it can be factored
*with* the 128-resident kernel plus GEMMs between: 2048 x 0.12us = 246us per
block against the vendor's 676us. At 16384 that is 8 blocks -> ~2000us vs
5400us, taking 8848 -> **~5450us (−38%)**. 8192 is 19x off floor and has the
most slack of the three.

### 1x32768 — needs a new quantizer, not a new schedule
Only 1.24x off the TF32 floor; the target is below it. exp 061 already landed
MXFP8 on part of the trailing update and found a *persistent* FP8 shadow
impossible with the shipped quantizer, because the scale-tile index
`(pid_m // 4) * (columns // 128) + pid_k` depends on total K, so per-block
column scales cannot be concatenated into the K-major layout the GEMM needs.
**Deliverable here is a K-independent block-scaled quantizer.** Highest effort,
lowest certainty — schedule last.

---

## 5. Sequencing

| # | item | shapes moved | confidence |
|---|---|---|---|
| 1 | fused chain+inverse in the block kernel | 8 | high |
| 2 | named-barrier phase overlap | 8 | medium |
| 3 | L1 end-of-call sync | 4 | high |
| 4 | L2 fp16 trailing | 6 | high |
| 5 | resident whole-matrix 256x128 / 64x256 | 2 | high |
| 6 | port 1x4096 / 1x8192 / 1x16384 | 3 | medium |
| 7 | 1x32768 K-independent quantizer | 1 | low |

Realistic read: items 1-6 plausibly reach −40% on ten to twelve shapes.
`1x4096` and `1x32768` are the two expected to fall short, and `1x32768` is
provably unreachable without deeper low precision.

---

## 6. Gates (non-negotiable, from hard-won experience)

1. **Full 15-shape `pairedgrid` is the only trustworthy number.** exp 050
   proved a subset probe systematically overstates an eager candidate; exp 062
   shipped a build that was correct on 57/57 and still measured **geomean
   0.8403** because a merged extension failed to load.
2. **Diff `baseline_counters` against `candidate_counters` on every shape.**
   Missing counters mean a fast path silently fell back. Correctness gates
   cannot see this — fallbacks are correct, only slower.
3. **When merging a kernel into the combined `load_inline`, the CUDA source
   string must be defined BEFORE the `load_inline` call**, not appended in a
   tail. Otherwise `NameError` -> `except` -> `_CUDA128 = None` -> every
   pre-existing CUDA fast path disappears.
4. **Controls for `2 <= batch <= 4, n >= 1024` must call `_loop_cholesky`**,
   never `torch.linalg.cholesky_ex` on the batch (the batched vendor path is
   3-5x slower and will fake a win).
5. Six-family correctness on every enrolled shape, then Popcorn `--mode test`
   17/17 on the exact source, then one ranked submission at a time.
6. No stream/queue APIs and no cuSOLVER in new fast paths (standing owner
   directive); popcorn's scanner is a literal substring match, so the token
   must not appear even in comments.
