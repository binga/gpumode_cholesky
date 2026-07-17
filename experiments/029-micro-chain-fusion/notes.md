# Experiment 029 — micro-chain cost reduction for the split32 shapes

Baseline: exact ranked winner `#882958`
(`3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`,
public 1096.084us). Probe targets: `16x512`, `4x1024` (paired, six families,
per-kernel profile, compiled-artifact capture; runner cloned from exp 020).

## Diagnosis the experiment acted on

The split32 chain is graph-replayed, so end-to-end ~= sum of kernel
self-times. `_micro_potrf_gj32` (13.7us/launch x n/32 launches, one-warp
serial rank-4 chain) is 53-58% of the low-batch shapes and does not amortize
with batch. Roughly half its per-iteration work is the interleaved
Gauss-Jordan inverse maintenance.

## Results (paired, B200, 2026-07-18)

| variant | 16x512 | 4x1024 | verdict |
|---|---:|---:|---|
| v1 noinv micro + substitution apply | 0.840x | 0.820x | **REJECTED** |
| v2 left-looking fusion (as filed) | fallback (arange pow-2 CompilationError) | fallback | invalid |
| v2fix (PRIOR_PAD masked dot) | 0.964x | 0.961x | **REJECTED** |
| v3 elimination inverse | 0.879x | 0.875x | **REJECTED** |
| v4 `tl.rsqrt` in the micro pivot chain | **1.028x** | **1.029x** | **WINNER (small)** |

All variants passed 6/6 families on both shapes; v2's first run timed the
cuSOLVER fallback (604 fallbacks) and is not a measurement of the design.

## What the profiles taught (the real finding)

- **Any 32-step serial tile loop costs ~16us/launch in Triton regardless of
  per-step arithmetic.** v1's substitution apply went 4.0 -> 16.4us/launch
  (123.9 -> 508.0us at 4x1024) even though each step is 3 cheap tile ops;
  v3's separated elimination inverse made the micro 13.7 -> 16.8us/launch.
  Step latency, not arithmetic, is the floor — which is exactly why the
  shipped rank-4 blocking (8 steps) + interleaved GJ inverse won exps 015/017.
- **The inverse really is ~45% of the micro** (v1's noinv micro measured
  7.5us/launch), but every alternative way to obtain the panel solve
  (substitution in the apply, separated elimination) re-pays that cost at a
  worse rate. The GJ interleave is the cheapest known home for it.
- **v2fix's structure worked as designed** (panel_inner eliminated; fused
  apply 210 -> 138us at 4x1024) but the pending-correction dot inside the
  one-warp micro costs ~+3us/launch (437 -> 534us), more than the fusion
  saves. Correction work is on the serial chain wherever it lives; a one-warp
  tl.dot is a bad place for it.
- v4: PTX-level change only (`rsqrt.approx` replaces `sqrt.approx.f32` +
  `div.full` per pivot); micro 13.7 -> 12.8us/launch, end-to-end +2.8% on
  both probed shapes, residuals unchanged (families all pass).

## Disposition

v4 promoted; remaining four split32 shapes probed
(`probe-v4-remaining.json`), then combined with the exp-030 routing candidate
for the full grid and a single ranked submission. v1/v2/v3 preserved here as
negative evidence; do not reopen 32-step serial structures for this chain.
