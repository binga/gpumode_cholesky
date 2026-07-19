# Campaign: 2x per shape, iterated

Goal (2026-07-20): improve each shape's latency by 2x, one shape at a time,
submitting verified gains to the leaderboard.

## Fact 1 — every shape is worth the same

The score is a geometric mean over 15 equally-weighted shapes. Halving *any*
single shape multiplies the geomean by `0.5^(1/15)` = **0.9548 → 4.52%
better**, whether it is the 46ms `1x32768` or the 50us `4096x32`.

Consequence: **order the campaign by tractability, not by absolute latency.**
Chasing the biggest number on the grid is a trap.

## Fact 2 — headroom above hardware floor (baseline grid 2026-07-20)

Floor = max(HBM traffic at ~7.7TB/s, tf32 roofline at ~450 TFLOP/s).
`n^3/3` FLOPs, `2*b*n^2*4` bytes.

| shape | measured us | floor us | x above floor |
|---|---|---|---|
| 4x1024 | 692.4 | 4.4 | **158.9x** |
| 2x2048 | 1373.2 | 12.7 | **107.9x** |
| 16x512 | 390.5 | 4.4 | **89.6x** |
| 64x256 | 229.1 | 4.4 | 52.6x |
| 256x128 | 147.2 | 4.4 | 33.8x |
| 8x2048 | 1636.5 | 50.9 | 32.1x |
| 2x4096 | 3214.1 | 101.8 | 31.6x |
| 1x4096 | 1537.3 | 50.9 | 30.2x |
| 1024x64 | 125.2 | 4.4 | 28.7x |
| 60x1024 | 1389.4 | 65.4 | 21.3x |
| 1x8192 | 5750.9 | 407.2 | 14.1x |
| 4096x32 | 49.6 | 4.4 | 11.4x |
| 640x512 | 1534.6 | 174.3 | 8.8x |
| 1x16384 | 15116.0 | 3257.8 | 4.6x |
| 1x32768 | 46320.1 | 26062.5 | **1.8x** |

## Fact 3 — 1x32768 cannot be doubled

At 1.8x above its own tf32 roofline, a 2x on `1x32768` would require running
faster than the hardware can multiply. Exp 034's MXFP8 V2 (1.090x) is close
to what remains extractable there; FP8 raises the roofline but the path is
already FP8 on its dominant products. **Declare this shape frontier-complete
at 2x and stop spending on it.** `1x16384` at 4.6x is the last large shape
where 2x is even arithmetically possible.

## Fact 4 — the mid-shape floor is launches, not FLOPs

The 20-160x headroom shapes are not compute-bound; they are bound by ~32
serial panel steps at ~16us of launch overhead each
(exp 029). `16x512` at 390us over ~24 launch-equivalents is the signature.
So on these shapes, 2x means **halving the launch count**, not the math.
Levers: deeper CUDA-graph capture of the serial chain, fusing adjacent chain
kernels, larger NB (fewer steps), or a one-CTA-per-matrix in-kernel
factorization for n<=256 where a matrix fits in one SM's smem.

## Attack order

1. **4x1024** (158.9x headroom, 692us) — worst ratio on the board.
2. **2x2048** (107.9x) — same launch-bound family, adjacent code path.
3. **16x512** (89.6x)
4. **64x256** (52.6x), **256x128** (33.8x)
5. **1x4096 / 2x4096 / 8x2048** (~30x) — enough headroom that both math and
   launch levers are live.
6. **1x16384** (4.6x) — hard but possible; MXFP8 V3 blocked at 10.1/20
   residual, revisit with the mantissa-clip fix.

Blocked on: the paired grid harness (exp 035). Nothing below ~2% is
measurable until the null calibration passes, and most single-shape 2x wins
are worth only 4.5% geomean each — well inside current grid noise.
