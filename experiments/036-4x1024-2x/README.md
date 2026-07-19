# Experiment 036 — 4x1024 2x attempt: DIAGNOSED, no variant shipped

## Verdict

`4x1024` cannot reach 2x (or 1.15x) through any available lever. The shape is
**not launch-bound** as the campaign doc assumed, and the cost that dominates
it is at a hard single-warp instruction-latency floor already attacked by two
prior experiments. No candidate was built; three diagnostic runs were spent
establishing why, and they redirect the campaign.

## Diagnosis 1 — the launch-bound thesis is wrong

`shapediag` (new probe mode, profiles kernel breakdown vs wall clock):

```
batch=4 n=1024: wall=715.2us device=681.6us idle=33.6us (4.7%) launches=102
  62.2%  424.22us  32 calls  13.257us/call  _micro_potrf_gj32
  14.7%  100.34us  31 calls   3.237us/call  _panel_apply32
  10.0%   68.35us  24 calls   2.848us/call  _panel_inner32_subtile64
   9.4%   63.8us    7 calls   9.117us/call  _trailing_nb
```

Only **4.7% idle**. The CUDA graph already eliminated launch overhead, so
exp 029's "~16us per launch" launch floor does not apply to this shape.
95% of the wall clock is real kernel execution, and 62% of it is one kernel.

## Diagnosis 2 — `_micro_potrf_gj32` is batch-independent

Cost per call across the whole grid, at every batch size:

| shape | calls | us/call |
|---|---|---|
| 256x128 | 4 | 13.81 |
| 64x256 | 8 | 13.55 |
| 16x512 | 16 | 13.24 |
| 4x1024 | 32 | 13.26 |
| 60x1024 | 32 | 13.65 |
| 8x2048 | 64 | 13.27 |
| 640x512 | 16 | 19.33 |

**13.5us/call from batch=4 to batch=640.** A kernel whose cost does not move
with 160x more parallel work is latency-bound, not throughput-bound. Total
device time across the grid: **2395us**.

## Diagnosis 3 — it is not register pressure, and num_warps=1 is optimal

`microprobe` (new mode) compiles the shipped kernel at several widths and
times a realistic 32-launch dependent chain at batch=4:

| num_warps | regs | spills | us/call |
|---|---|---|---|
| **1** | 236 | 0 | **14.39** |
| 2 | 137 | 0 | 20.53 |
| 4 | 102 | 0 | 22.58 |
| 8 | 80 | 2 | 32.85 |

Zero spilling at the shipped setting, and more warps is monotonically worse —
cross-warp reductions cost far more than the register relief returns. The
"num_warps=1 is the unlock" finding from `4096x32` (S3) holds here too, for a
different reason: at batch=4 there are 4 CTAs of 1 warp on 148 SMs, so there
is **no occupancy to hide instruction latency**. ~26,000 cycles of fully
exposed single-warp dependent chain is exactly what 14us buys.

This also closes the larger-diagonal-block lever by arithmetic: a 64x64 block
needs 128 regs/thread for `a` alone at num_warps=1 (spill guaranteed at the
255 ceiling), and num_warps>=4 pays the penalty measured above.

## Why no variant was built

Combined with prior work, every axis on this kernel is now measured:

- rank-4 pivot interleave — shipped (exp 017)
- `tl.rsqrt` on the pivot chain — shipped, +2.8% (exp 029 v4)
- inverse-free micro + substitution apply — 0.82-0.84x (exp 029 v1)
- left-looking fusion — 0.96x (exp 029 v2)
- separated elimination inverse — 0.87x (exp 029 v3)
- persistent single-launch kernel — 0.40-0.49x (exp 028)
- **num_warps 2/4/8 — 1.43x/1.57x/2.28x SLOWER (this experiment)**
- **larger diagonal block — closed by register arithmetic (this experiment)**

Step count is also near-exhausted: exp 029 records a 32-step loop at ~16us
against today's 8-step rank-4 at 13.5us, i.e. cutting steps 4x bought 15%.
The implied model is ~12.7us fixed + ~0.10us/step, so rank-8 (4 steps) would
return under 5%. Even eliminating the kernel *entirely* leaves 4x1024 at
259us — 2.67x — so nothing short of a near-total rewrite reaches the target.

## Campaign consequence — the real finding

`_micro_potrf_gj32` dominates **six** shapes and is material in a seventh:

| shape | micro share |
|---|---|
| 4x1024 | 62.2% |
| 16x512 | 57.3% |
| 64x256 | 54.8% |
| 8x2048 | 51.6% |
| 256x128 | 47.3% |
| 60x1024 | 35.5% |
| 640x512 | 21.4% |

Because its cost is batch-independent and scales as `n/32` calls, it is the
single highest-leverage target on the board — and it also blocks the obvious
route for the three cuSOLVER-bound shapes below.

**Second finding: `2x2048`, `1x4096`, `2x4096` have never had custom work.**
They spend 87-91% of device time in one cuSOLVER kernel
(`getrf_wo_pivot_params_`), at 618us/matrix (n=2048) and 1393us/matrix
(n=4096). The custom split32 chain runs n=2048 at 206us/matrix — 3.0x
faster per matrix. But routing them onto split32 does not reach 2x either,
because the micro floor follows: n=4096 needs 128 micro calls = 1702us of
the 3209us budget before any real math. This is consistent with S16 having
measured those routes at 0.764x/0.784x.

## Recommended next lever (untried, lifts 7 shapes at once)

Rewrite `_micro_potrf_gj32` as a hand-written CUDA kernel (the `_CUDA_MOD`
load path already exists and is currently unused). The specific inefficiency
Triton cannot avoid: each of the 8 iterations rebuilds the full 32x32 tile
through a 5-deep nested `tl.where` cascade plus a second cascade for the
inverse — roughly 10k predicated ops per iteration over the tile, on a single
warp with zero latency hiding. A CUDA version can update only the shrinking
trailing sub-block in registers and schedule `__shfl_xor_sync` reductions
explicitly. This is the one axis with plausible multi-x upside, and at 2395us
of grid-wide device time it is worth a dedicated experiment.

## Artifacts

- `exp036-diag.json` — n=1024 kernel breakdown
- `exp036-diag-all.json` — all 15 shapes
- `exp036-microprobe.json` — num_warps sweep with register/spill stats

Tooling added: `shapediag` and `microprobe` modes in `scripts/_gpu_runner.py`.
