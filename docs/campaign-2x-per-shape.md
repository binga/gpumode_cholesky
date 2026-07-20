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

## Fact 4 — CORRECTED 2026-07-20 (exp 036): the floor is one kernel, not launches

**The launch-bound thesis below was WRONG for `4x1024` and is retained only
as the refuted hypothesis.** Measured: `4x1024` is wall 715us / device 682us
— **idle only 33.6us (4.7%)**. The CUDA graph already removed launch
overhead, so exp 029's ~16us launch floor does not apply to this shape.

The real floor is a single Triton kernel, **`_micro_potrf_gj32`**
(`submission.py:640`): 424us over 32 calls = **62.2% of `4x1024`**. It costs
~13.5us/call at *every* batch from 4 to 640 (13.81 / 13.55 / 13.24 / 13.26 /
13.65 / 13.27) — a kernel that does not speed up with 160x more parallel work
is **latency-bound, not throughput-bound**. Cause is exposed single-warp
instruction latency: 4 CTAs x 1 warp on 148 SMs, no occupancy to hide it.
A `num_warps` sweep refuted register pressure (1 warp: 236 regs, 0 spills,
14.39us; more warps monotonically worse, up to 32.85us at 8).

Step-count is near-exhausted too: 32-step ~16us vs today's 8-step 13.5us
implies **~12.7us fixed + 0.10us/step**, so rank-8 returns <5%.

**`4x1024` cannot reach 2x**: deleting the micro kernel entirely still leaves
259us = 2.67x. Classified EXHAUSTED-diagnosed.

### The target is a kernel, not a shape

`_micro_potrf_gj32` dominates six shapes (62.2 / 57.3 / 54.8 / 51.6 / 47.3 /
35.5%) and is 21.4% of `640x512` — **2395us of device time grid-wide**. One
rewrite moves seven shapes. Triton rebuilds the full 32x32 tile through a
5-deep nested `tl.where` cascade every iteration, plus a second for the
inverse (~10k predicated ops per iteration, one warp, zero latency hiding).
A CUDA rewrite could update only the shrinking trailing sub-block and schedule
`__shfl_xor_sync` reductions explicitly. Experiment 037 tested that premise:
the shipped 14.379us kernel sits above 10.083--10.456us empty/load-store/
synthetic floors, leaving only **1.38x** measurable headroom. Thus a rewrite
may help incrementally but cannot supply the required 2x. Exp 017 had already
cut this kernel 16.5 -> 13.9us via a rank-4 pivot micro.

### Untouched: the cuSOLVER trio

`2x2048`, `1x4096`, `2x4096` have never had custom work — **87-91% of their
device time is one cuSOLVER kernel** (`getrf_wo_pivot_params_`), 618us/matrix
at n=2048 and 1393us/matrix at n=4096. The split32 chain does n=2048 at
206us/matrix, **3.0x faster**. But the micro floor follows it: n=4096 would
need 128 micro calls = 1702us of a 3209us budget before any real math, which
is why S16 measured those routes at 0.764x/0.784x. **Fix the micro kernel
first, then these three shapes open up.**

## Fact 4 (REFUTED, kept for the record) — "the mid-shape floor is launches"

The 20-160x headroom shapes are not compute-bound; they are bound by ~32
serial panel steps at ~16us of launch overhead each
(exp 029). `16x512` at 390us over ~24 launch-equivalents is the signature.
So on these shapes, 2x means **halving the launch count**, not the math.
Levers: deeper CUDA-graph capture of the serial chain, fusing adjacent chain
kernels, larger NB (fewer steps), or a one-CTA-per-matrix in-kernel
factorization for n<=256 where a matrix fits in one SM's smem.

## Attack order (revised 2026-07-20 after exp 036)

1. ~~**4x1024**~~ — EXHAUSTED-diagnosed. 2x unreachable; see Fact 4.
2. ~~**`_micro_potrf_gj32` CUDA rewrite**~~ — EXHAUSTED-diagnosed by exp 037.
   Synthetic one-warp floors leave only 1.38x headroom.
3. ~~**2x2048**~~ — EXHAUSTED under six distinct correct architectures. Exp
   038's best hardware-cluster/TRSM path was 0.595x versus ranked.
4. ~~**4096x32**~~ — **2x ACHIEVED** by exp 039. Register-row/shared-pivot
   rank-2 CUDA: 43.29 -> 19.09us = 2.269x; ranked as `#888636`.
5. ~~**1x4096**~~ — EXHAUSTED under six correct cooperative architectures.
   Device-clock V1 profile: 837us diagonal + 1017us panel + 2142us trailing;
   best candidate 4066us versus 1531us ranked. Tile 64, tensor-core panel
   inverse, occupancy saturation, left-looking, and rank-128 superpanels all
   lost. Do not transfer this mechanism to `2x4096`.
6. ~~**1024x64**~~ — **2x ACHIEVED** by exp 041. V1's one-warp, two-register-
   row design replaced the 17-operation vendor graph at 122.32 -> 53.90us =
   2.270x. Post-target V3 moved to two warps, one row per thread, and a
   four-rendezvous rank-2 handoff: 53.584 -> 32.192us = another 1.664x, or
   about 3.80x end-to-end. Latest ranked source is `#888867`.
7. **256x128** — current third contract target; enough matrices
   to saturate B200 with one whole-matrix CTA each.
8. **16x512 / 64x256** — re-diagnose per shape before assuming a
   lever; the `4x1024` lesson is that the campaign's headroom table says
   *whether* a shape is slow, never *why*.
9. **1x16384** (4.6x) — MXFP8 V3 blocked at 10.1/20 residual; revisit with
   the mantissa-clip fix.
10. **1x32768** — frontier-complete at 1.8x above roofline. Do not spend here.

**Method note.** Diagnose before building. Exp 036 spent 3 profiling runs to
kill a 6-variant plan whose premise was wrong, and produced the campaign's
best finding by doing so.

Measurement note: the paired grid harness from exp 035 is the campaign's
acceptance signal. Once exp 041 replaced the `1024x64` graph route, the V3
same-process comparison was stable: 1.664x with 0.28% order spread.
