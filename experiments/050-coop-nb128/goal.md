# Goal exp 050 — collapse the dependent-launch chain on the mid shapes

## Baseline

Exact ranked source `#890798` = **801.977us public / 847.836us secret**,
SHA-256 `fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
Local paired-grid geomean 836.6us (`experiments/047-fused-panel/variant-09-fullgrid.json`).

Board (2026-07-21, B200): Ravi Theja 112.611us `sc2cap_hffull16_1.py`,
yanchi_72526 178.772, viridale 215.401, zhongmingee 375.420,
Sebastian Kimberk 496.527, xuan9938 506.539 ... **binga 801.977 (rank 17/66)**.
Deadline 2026-07-30.

## Target

Owner goal: **2x overall geomean** (836.6 -> ~418us local, ~401us ranked),
submitting to the leaderboard whenever a verified gain appears.

## Diagnosis (this session, fresh B200 measurements)

`shapediag` on all six mid shapes (`experiments/050-coop-nb128/exp050-shapediag.json`):

| shape | wall | device | idle | launches | dominant constituent |
|---|---:|---:|---:|---:|---|
| 16x512 | 386.5 | 362.2 | 24.3 (6.3%) | 54 | `_micro_potrf_gj32` 206.9us/16 = **12.93us/call**, 57.1% |
| 640x512 | 1324.2 | 1119.0 | 205.2 (15.5%) | 53 | `_panel_fused128` 357.2/3; `_trailing_nb` 193.5/1; `micro_potrf32_rank4` 168.0/16 @10.50 |
| 4x1024 | 696.1 | 654.9 | 41.2 (5.9%) | 101 | `_micro_potrf_gj32` 407.0/32 = **12.72us/call**, 62.1% |
| 60x1024 | 1210.8 | 1016.4 | 194.4 (16.1%) | 76 | `_trailing_nb` 378.8/7; `micro_potrf32_rank4` 308.0/32 @9.62; `_panel_fused128` 175.7/7 |
| 2x2048 | 1357.7 | 1341.9 | 15.8 (1.2%) | 13 | cuSOLVER `getrf_wo_pivot` 1223.6/2 = **611.8us/call**, 91.2% (4.7 TFLOP/s) |
| 8x2048 | 1562.9 | 1554.7 | 8.2 (0.5%) | 198 | `_micro_potrf_gj32` 830.6/64 = **12.98us/call**, 53.4% |

`microprobe` (`experiments/050-coop-nb128/exp050-microprobe.json`), 32 dependent
launches at batch=4, the exact structure the `4x1024` path runs:

| num_warps | regs | spills | chain32 | per call |
|---:|---:|---:|---:|---:|
| **1** | 236 | 0 | **482.1us** | **15.065us** |
| 2 | 137 | 0 | 657.5us | 20.546us |
| 4 | 102 | 0 | 723.0us | 22.593us |
| 8 | 80 | 2 | 1051.2us | 32.849us |

**Conclusion.** `_micro_potrf_gj32` costs ~12.9us/call *independent of batch*
(12.933 @b=16, 12.718 @b=4, 12.977 @b=8) and independent of work
(exp 036: ~12.7us fixed + 0.10us/step). More warps make it strictly worse and
there are no spills. Therefore the cost is **dependent-kernel-launch
turnaround**, not serial-step latency, not occupancy, not register pressure.
We pay ~10-15us of turnaround 32-64 times per mid shape for ~3us of real work.

Hardware floors (max of 2*b*n^2*4 / 7 TB/s, and b*n^3/3 / 600 TFLOP/s tf32):
4x1024 **143x** over floor, 2x2048 **142x**, 16x512 **81x**, 2x4096 42x,
8x2048 41x, 1x4096 40x -- versus 1x32768 only **2.2x**. Geomean of the 15
floors is 42.4us, so the leader's 112.6us is 2.65x off SOL. Because the
leaderboard geomean equal-weights all 15 shapes, a 2x on 1x32768 is worth
836.6 -> 798.8us (-4.5%), while taking the six tiny-batch mid shapes to
near-floor is worth 836.6 -> 215.4us.

## Hypothesis

Collapsing a mid shape's factorization into **one cooperative launch with
barrier count = n/NB at NB>=128** removes the dominant term. Barriers replace
turnarounds, and at NB=128 there are 8 of them for n=1024 instead of 101
launches.

## Why prior attempts failed (and what is actually untried)

- exp 048 V1 (graph, 102->52 launches): device 664->394us but graph dependency
  idle rose 10.5->326us, wall 719.9us. Launch *count* alone is not enough.
- exp 048 V2 (128-CTA cooperative tile-32): **1.167x, 616.8us** -- the only
  architecture that ever beat the incumbent on a mid shape. Rejected for
  low-rank NaN/Inf, a numerical bug. Internals: 197.1us diagonal, **286.2us
  scalar panel**, 201.9us TF32 trailing WMMA. Too many barriers (tile-32) and
  an unvectorized panel.
- exp 048 V4 (four atomic rank-128 superpanels): 0.186x, **scalar panel
  1552.9us**. Large NB but again a scalar panel.
- exp 049 V1/V2/V3 (cluster/DSM, one persistent CTA, atomic CTA groups):
  0.418x / 0.299x / 0.697x. All died on the *trailing* update starving the GPU
  ("only 16 CTAs use the GPU"), not on the diagonal.

**Untried cell: large NB (>=128) AND a tensor-core panel AND a full-grid
trailing update, in one cooperative launch.** Every failure above is missing
exactly one of those three.

## Success threshold

Per-shape paired speedup >= 2.00x on `4x1024` against exact `#890798`, with no
regression outside the dispatch region. Promotion to a ranked submission needs
only a reproducible aggregate improvement on the full 15-shape paired grid.

## Correctness constraints

Gate is `||A - LL^T||_1 <= 20*n*eps*||A||_1` with **fp32** eps, growing with n.
Measured margins: 362x @n=32, 800x @n=64, 1575x @n=128, widening. All six
families (dense, diagonal, spectrum, lowrank, rowscaled, tridiagonal) must pass
for every changed shape -- exp 048 V2's low-rank NaN is the specific trap.
fp16 inputs cost ~8192x in input rounding, so full-fp16 is only affordable at
n>=1024; do not attempt it at n<=256.

## Guardrails

- No stream-based and no new cuSOLVER-based fast paths (standing owner
  directive); new paths must be custom-kernel based.
- No input-keyed caching, shape anchoring, or any evaluator-scanner evasion.
- At most one ranked Popcorn submission in flight, after an exact-source 17/17
  test pass.

## Fallback ladder

1. Cooperative NB=128 single-launch kernel for `4x1024`.
2. If barrier cost dominates: raise NB to 256 (4 barriers) and sub-block the
   diagonal factorization inside the kernel.
3. If the cooperative grid starves at batch=4: one CTA-cluster per matrix with
   the panel resident in distributed shared memory.
4. If none reach 2x: bank any >=1.05x aggregate and ship it.
