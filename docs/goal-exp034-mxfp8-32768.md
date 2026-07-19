# Goal: Experiment 034 — MXFP8 block-scaled trailing products for 1×32768

## Objective

Replace the per-tensor-scaled FP8 pipeline in the `1×32768` left-looking path
with **MXFP8 block-scaled products** (E4M3 values + per-32-element E8M0 scales)
that lower to Blackwell `tcgen05` block-scaled MMA on sm_100. Stretch goal:
extend MXFP8 to the `1×16384` path (currently tf32 panel products).

Target per program.md: ≥2.00x paired speedup on the changed region = WINNER;
any verified paired improvement = FRONTIER (1×32768 is ~76% of total wall
clock, so even a few % moves the geomean more than any other shape).

## Baseline (exact)

- Ranked source: `submission.py` at commit `cda77c7` (finalist `#884868`,
  geomean 1081.737us).
- The 1×32768 path (`_left_looking_cholesky_32768` / `_left_looking_large`
  with `_LARGE_CFG[32768] = nb=4096, panel_mode="fp8"`) computes each panel
  product via `_fp8_product_32768` (submission.py:1426):
  1. `_dual_tiled_amax_e4m3_32768` — tiled partial-amax reduction over both
     operands (one full read pass),
  2. host-side `.amax()` finish + per-tensor scale computation
     (device-wide reduce + sync),
  3. `_dual_scale_cast_e4m3_32768` — scale & cast both operands to E4M3
     (second full pass),
  4. `torch._scaled_mm` with per-tensor scales.
- Shipped accuracy: dense scaled residual **4.52 / 20** at n=32768
  (budget `20·n·eps·‖A‖₁` ≈ 7.8e-2 relative).

## Hypothesis

On sm_100 the block-scaled MMA applies per-32-element E8M0 scales inside the
instruction. Expected wins:

1. **Kernel elimination** — scales become local to each 32-element block, so
   quantization is a single fused pass (cast + local amax→E8M0), removing the
   global amax reduction, the host round-trip, and one full memory pass per
   product.
2. **Accuracy** — per-block scaling beats per-tensor scaling, which should
   *lower* the 4.52/20 residual and may make FP8 viable at n=16384
   (budget ≈ 3.9e-2) where per-tensor FP8 was not shipped.
3. **Possibly faster MMA** — same FP8 throughput class, but no separate
   scale multiply epilogue.

## Implementation ladder (bounded: max 6 serious variants)

- **V1 (primary)** — Triton MXFP8 GEMM: `tl.dot_scaled(lhs, lhs_scale,
  "e4m3", rhs, rhs_scale, "e4m3")` with fp32 accumulation, plus a fused
  one-pass quantize kernel per operand emitting E4M3 values and per-32 E8M0
  scales in the layout `tl.dot_scaled` requires. Swap into
  `_fp8_product_32768` call sites behind a new `panel_mode="mxfp8"`.
- **V2 (fallback API)** — if the Triton path software-emulates: native
  `torch._scaled_mm` with block-wise MX scaling if the image's torch build
  supports it (check `ScalingType`/mx recipes; record torch + Triton
  versions either way).
- **V3** — extend the validated winner of V1/V2 to `1×16384`
  (`_LARGE_CFG[16384]`: replace tf32 `panel_mode`), gated on residual.
- **V4–V6** — tile-shape/num-warps sweep, scale-layout variants
  (contiguous vs swizzled), quantize-fused-into-GEMM-prologue.

## Hard requirements

1. **Backend proof (program.md step 7)** — dump TTGIR/PTX for the V1 kernel
   and confirm block-scaled tcgen05 MMA instructions are emitted. If Triton
   dequantizes and falls back to fp16/tf32 dots, the variant is
   invalid-as-MXFP8 regardless of timing; record it as such and move to V2.
   Also require `_LEFT_32768_ERROR is None` and hit counters showing the new
   path executed (add an `_MXFP8_HITS` counter mirroring the existing ones).
2. **Free gates before any GPU spend** — `scripts/verify_local.py`, syntax
   /compile checks, `git diff --check`, source-policy scan.
3. **Policy boundaries** — no cuSOLVER in new code paths, no CUDA
   stream/queue APIs, no obfuscated identifiers to evade the popcorn source
   scanner. Custom Triton kernels + torch ops only. Existing fallbacks stay.
4. **Correctness** — all six input families (dense, spectrum, lowrank,
   rowscale, diagonal, tridiagonal) at every changed dispatch shape via the
   vendored checker; scaled residual target ≤ 8/20 (safety margin for secret
   seeds), finite outputs, positive diagonal, no fallback engagement.
5. **Timing** — paired same-process Modal B200 probe vs the exact ranked
   source (scripts/modal_verify.py conventions; standing Modal authorization
   in program.md applies). Rotate representative inputs; report mean/best.
6. **Integration** — only if paired 1×32768 improves: run the full 15-shape
   grid, require every shape to pass, no material off-target regression,
   geomean must improve.
7. **Stop line** — commit the experiment locally with artifacts. Do NOT push
   to GitHub and do NOT submit to the popcorn leaderboard; both remain
   explicit user actions per program.md.

## Artifacts

`experiments/034-mxfp8-32768/` containing: `baseline.py` (exact ranked
source), `candidate-*.py` per variant, results JSON under `results/`,
PTX/TTGIR backend evidence, and a journal.md session entry classifying every
measured variant (WINNER / FRONTIER / REJECTED / EXHAUSTED) with paired
numbers and residuals.

## Risks

- Triton on the image may not support `tl.dot_scaled` MX formats on sm_100
  (needs a recent Triton); V2 covers this. If neither engages hardware
  block-scaled MMA, classify EXHAUSTED-blocked and record the exact version
  evidence — do not ship an emulated "MXFP8" that is really fp16 dots.
- Scale-layout mismatch silently degrades to garbage → the six-family
  checker gate catches it; validate before timing.
- n=16384 residual may exceed 8/20 on rowscale/lowrank; V3 is optional and
  reverts to tf32 if so.
