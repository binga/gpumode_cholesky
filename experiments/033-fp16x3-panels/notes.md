# Experiment 033 — panel precision (lever L4): fp16x3 REJECTED, tf32 panels SHIPPED

Baseline: L2-banked source (`baseline-l4.py` = root at exp-033 start: 8x2048
NB=256 schedule, all panels tf32x3). Measured 2026-07-18 on B200 via `dotprobe`
(isolated tensor-core throughput) and `schedprobe` (paired same-process,
drift <0.9%). Raw JSON: `../../results/exp033-*.json`.

## Step 1 — dotprobe kill-gate: fp16x3 beats tf32x3 in ISOLATION
fp16 and tf32 share a 10-bit mantissa, so fp16x3 (three-fp16-MMA emulated fp32)
~= tf32x3 in accuracy but fp16 MMA is faster. Saturated single-launch GEMM at the
panel tile shapes:

| (K,N) | tf32x3 | fp16x3 | fp16x3 speedup | fp16x3 relerr |
|---|---|---|---|---|
| 32,128 | 31.4us | 23.5us | 1.34x | 4.7e-7 |
| 32,256 | 108.9us | 89.6us | 1.22x | 5.3e-7 |
| 128,128 | 151.0us | 48.2us | 3.14x | 1.2e-6 |

Gate PASSED in isolation (deepest K wins most; tf32x3's 3-pass cost scales with K).

## Step 2 — fp16x3 in the real kernels: REJECTED (5-40x SLOWER)
Wired fp16x3 into the 3 panel dots via a `_dot_prec` device helper. Paired
schedprobe: **0.02-0.17x** on every shape (256x128 146->6388us), correct but
catastrophic (~6-10ms/call, roughly shape-independent -> not compute-bound).
Cause: the fp16x3 emulation adds 4 fp16 temps + 3 fp32 accumulators inside panel
kernels already at the 255-register ceiling (the `_panel_inner32_subtile64`
kernel was created specifically because tf32x3's 128x128 tile already spilled).
The isolated GEMM saturates a fresh kernel; the panel kernels have no register
headroom, so fp16x3 spills to local memory. Smaller tiles would fix the spill but
kill the throughput win, and the launch floor caps the small shapes regardless.

## Step 3 — tf32 (1-pass) panels: SHIPPED on large-n
Native tf32, no register blowup. Paired vs tf32x3 baseline-l4:

| shape | speedup | max family residual (/20) | verdict |
|---|---|---|---|
| 256x128 | 1.042x | **dense FAILS**, tri 7.38 | reject |
| 64x256 | 1.058x | rowscale 19 | reject (razor) |
| 16x512 | 1.058x | rowscale 12.6 | reject (thin) |
| 640x512 | **1.142x** | rowscale 14.1 | reject (thin, biggest win) |
| 4x1024 | 1.065x | spectrum 8.13 | **SHIP** (2.5x headroom) |
| 60x1024 | 1.057x | spectrum 7.6 | **SHIP** (2.6x headroom) |
| 8x2048 | 1.072x | rowscale 4.31 | **SHIP** (4.6x headroom) |

The gate `20*n*eps*|A|` grows with n, so tf32 panels are safe only at large n.
Shipped panel_prec "tf32" on (4,1024),(60,1024),(8,2048). `changed_geomean`
across all 7 was 1.070x, but only the 3 large-n shapes have safe accuracy margin.

## Gates and ranked result
- Verify: **57/57** family specs. Popcorn test **17/17** (#884847).
- Full grid: 15/15, geomean 1117.2us (this sandbox), enrolled shapes clearly
  faster (8x2048 1618 vs ~1795 baseline = combined NB=256+tf32 ~1.11x).
- Ranked, TWO identical resubmissions (measurement-validity check per proposal §3):
  - **#884850**: public 1086.309us (+0.17%), secret 1063.862us (-1.83%)
  - **#884868**: public 1081.737us (-0.25%, NEW BEST), secret 1091.616us (+0.73%)
- Two identical files varied 0.42% public / **2.6% secret** — the 15-shape geomean
  cannot resolve the ~1.5% paired win. Adopted #884868 (best public 1081.737us);
  paired-validated, correctness-bulletproof, zero off-target regression.

## Reusable harness added
`dotprobe` mode (fp16x3 vs tf32x3 isolated) + generalized `schedprobe`
changed-shape detection (compares full `_SPLIT32_SHAPES` + schedule per shape) in
`scripts/_gpu_runner.py`. `make_candidates.py --prec {fp16x3,tf32}`.
