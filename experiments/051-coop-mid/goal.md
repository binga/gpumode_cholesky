# Experiment 051 — cooperative tile-32 factorization for the tiny-batch mid shapes

## Goal

User goal: **improve the overall leaderboard geometric mean by 2x** against the
ranked incumbent `#890798` (public 801.977us), i.e. reach ~401us, submitting
each verified improvement as it appears.

## Why this experiment

A fresh full-grid shape diagnosis of the incumbent (`results/inc-890798-shapediag.json`)
puts every shape's wall time next to its hardware floor:

| shape | wall us | x floor | dominant cost |
|---|---:|---:|---|
| 4x1024 | 706.7 | 147x | 32 serial `_micro_potrf_gj32` (419us) |
| 2x2048 | 1353.9 | 141x | serial cuSOLVER `getrf_wo_pivot` |
| 16x512 | 402.4 | 84x | 16 serial `_micro_potrf_gj32` (210us) |
| 2x4096 | 3199.2 | 42x | serial cuSOLVER |
| 8x2048 | 1593.5 | 42x | 64 serial `_micro_potrf_gj32` (842us) |
| 1x4096 | 1527.2 | 40x | serial cuSOLVER |
| 1x32768 | 42431.5 | **2.2x** | already near SOL |

The loss is dependent-step latency, not FLOPs and not launch count. Getting the
whole grid to the ~200ns-per-dependent-pivot floor that experiment 044 measured
for a standalone 32x32 micro is worth roughly 3x on the geometric mean, which
is the size of the gap to the leaders (viridale 317.5us, zhongmingee 320.8us).

## Hypothesis

Experiment 048 V2 — a 128-CTA cooperative tile-32 kernel — already measured
**1.167x** on 4x1024 and was rejected for two fixable reasons, not because the
architecture failed:

1. It produced NaN on the `lowrank` family, with no finiteness gate wired.
2. Its own phase timings (`experiments/048-4x1024-2x/variant-02-phase.json`)
   show **41.5% of runtime in a scalar panel solve** whose dependent chain is
   528 serial FMAs per 32x32 tile, executed by warp 0 only, reloading the
   diagonal factor from global memory once per panel tile.

A single-launch cooperative kernel is also the only mid-shape architecture that
does not pay experiment 050's ~7.6us-per-launch eager tax, because it replaces
54-198 launches with one.

## Variant ladder

| V | change | state |
|---|---|---|
| v1 | 048 V2 + finiteness gate + right-looking register panel solve (chain 528 -> 32 FMAs, 4 warps, diagonal factored redundantly per CTA so one grid barrier per block step disappears) + occupancy-derived CTA count | measuring |
| v2 | shared-staged trailing update (V2 spends 29.3% there at ~7 TFLOP/s against a 51us bandwidth floor) | designed |
| v3 | rank-k diagonal micro / larger NB to shorten the pivot chain | designed |

## Gates

Free: python compile, source-policy scan, exact-baseline sha256 check in the
builder. Remote: paired same-process B200 (`pairedgrid`), six-family correctness
(`familygrid`), full 15-shape paired grid, cold-build proof, popcorn test 17/17,
then exactly one ranked submission.

## Constraints carried forward

- Single merged `load_inline` extension (experiment 050 L5): the popcorn build
  cache is keyed by extension name and a cold four-extension build exceeds the
  360s service timeout.
- No popcorn source-scanner evasion, no queue/stream APIs, no new cuSOLVER-based
  fast paths (owner directive). Existing cuSOLVER fallbacks stay as fallbacks.
