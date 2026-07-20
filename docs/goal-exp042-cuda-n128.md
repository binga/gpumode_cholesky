# Goal — Experiment 042: cuSOLVER-free CUDA `256x128`

## Frozen control and threshold

The control is exact ranked submission `#888867`, commit `7a77cc3`, SHA-256
`7380e038441b55666819d6685ff3ddd68776c7571757afced15c29b3656ac9c2`.
Revision-5 byte-identical Modal B200 calibration measured the normalized
`256x128` control at **154.824us**, so the user-directed 2x threshold is
**77.412us**. Its worst A/A deviation was 0.98% and worst order spread 0.61%; a
50% reduction is far outside measurement noise.

## Constituent question

The current path is a CUDA-graph replay of a 32-wide Triton factorization plus
copy and finiteness gates. The last complete profile measured about 150.0us
wall / 116.7us device over 18 operations: 55.24us in four serial
`_micro_potrf_gj32` launches, 19.37us in three panel-inner tensor-core updates,
13.81us in three panel applications, 9.10us in device copies, 6.51us in the
finite reduction, 3.31us in its device-to-host result, and 9.34us in remaining
elementwise kernels. Wall-minus-profiled-device was 33.3us. A fresh profile
against `#888867` must confirm the budget before candidate work.

The old chain cannot reach 2x by optimizing only one constituent: even deleting
all four diagonal micros leaves roughly 100us wall. The first candidate must
therefore replace the launch chain, copy-out, and finite-check path together.

## Primary architecture

Use one cuSOLVER-free CUDA CTA per 128x128 matrix, 128 threads (four warps), and
one register-resident row per thread. Padded shared-memory staging coalesces the
input and required lower-triangular output. Shared pivot columns hand off live
cross-row state; rank-2 pivot processing amortizes four block rendezvous over
two columns. This is the successful exp-041 mechanism scaled to four warps and
256 independent matrices. It replaces all 18 current operations with one
launch and has enough CTAs to cover the 148-SM B200 in two waves.

The target path may use neither cuSOLVER nor any auxiliary/concurrent CUDA queue
API. It must return the required lower-triangular representation directly.

## Bounded ladder

1. Rank-2, four-warp whole-matrix CUDA with padded staging.
2. Rank-4 pivot handoff if synchronization, rather than per-thread arithmetic,
   dominates V1.
3. Split row ownership across eight warps if V1 is instruction-latency-bound
   and register/occupancy evidence supports it.
4. Two-level 64+64 CUDA factorization with tensor-core panel/trailing work if a
   scalar whole-matrix CTA cannot cross 77.412us.
5. Fuse the existing split32 diagonal/panel sequence into a persistent
   whole-matrix CUDA CTA as the final non-vendor architecture.

Stop after six distinct correct architectures if none crosses 2x; record
device-clock phase constituents for the best scalar and tensor-core designs so
the exhaustion decision is causal rather than empirical only.

## Promotion gates

First require syntax/static checks and an active-backend dense paired result.
Then require all six input families with zero new fallbacks, an integrated
three-shape audit against revision 5, full 15-shape paired parity, Popcorn test
17/17, and one serial ranked submission. After 2x is achieved, continue only
with refinements that show a statistically real same-process gain, and submit
each fully gated gain as directed by the campaign goal.

## Result — 2x achieved and ranked

The fresh `#888867` profile confirmed 143.3us wall / 115.2us device over 18
operations: 55.13us diagonal micros, 32.81us panel math, 8.90us copies,
9.13us finite/host gate, 9.18us other elementwise work, and 28.1us
wall-minus-device. V1's register-row rank-2 kernel was only 1.081x; internal
timing attributed 89.25us of its 98.94us device span to scalar factor work,
while a forced 256-barrier floor was only 2.88us. V2/V3 element-parallel shared
updates measured 0.988x/0.983x, proving neither launches, data movement,
synchronization, nor code size was the remaining lever.

V4 changed the algorithm to FP32 blocked-16 Cholesky inside one eight-warp CTA:
factor a 16x16 diagonal block, solve the 16-wide panel rows in registers, then
apply a coarse rank-16 trailing dot. It crossed 2x at 150.940 -> 71.828us.
V5 added `#pragma unroll 1` to the eight-block outer loop, preserving the win
while cutting import time enough for Popcorn's six-minute test limit. Its exact
full grid measured **140.932 -> 69.852us = 2.0191x**, aggregate **1.047866x**
CI [1.047341,1.048391], all 15 shapes correct and all others at parity.

V5 passed all six families on the active backend with no fallback and worst
residual 0.0176/20. Popcorn test `#888995` passed 17/17; ranked `#888996`
scored 916.577us public / 863.850us secret. Exact ranked SHA-256 is
`5bbb8e8de3eedfadfb70fdcfaa902723fab03c2998bc18e9cb80512e82cce80c`.
The earlier V4 test `#888971` ended exactly at six minutes with no failed case
output and `passed=false`; this was compile timeout, fixed by V5.

Instrumented V5 device span was 59.55us: 3.30us staging, 21.28us diagonal,
6.05us panel, 23.62us trailing, 1.12us output, and about 4.18us phase-boundary
overhead. Post-target dual accumulation was noise (1.0018x, CI crosses 1),
BK=32 regressed 21.8%, and 16 warps regressed 5.2%; none was promoted.
