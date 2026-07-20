# Goal — Experiment 043: cuSOLVER-free CUDA `64x256`

## Frozen control and threshold

The control is exact ranked submission `#888996`, commit `ec33b31`, SHA-256
`5bbb8e8de3eedfadfb70fdcfaa902723fab03c2998bc18e9cb80512e82cce80c`.
Revision-4 will establish a paired same-process Modal B200 baseline before any
candidate edit. The user-directed acceptance threshold is a 50% latency
reduction for dense `64x256`; the full 15-shape grid remains the regression
gate.

## Constituent diagnosis

The ranked path measured 219.0us wall and 195.0us profiled device time over 30
operations. Its constituents were 105.96us in eight serial 32-wide diagonal
factorizations, 55.14us in panel apply/inner tensor-core work, 8.21us in the
trailing update, 8.80us in two device copies, about 15.31us in the finite gate
and other output bookkeeping, and 24.0us wall-minus-device gaps. Analytic work
floors classify the dominant diagonal and panel kernels as exposed
latency/serialization, not bandwidth or arithmetic-throughput limited.

The 2x threshold is approximately 109.5us from this diagnostic run. Deleting
only the diagonal micros would leave about 113us, so the next architecture must
replace the serialized launch chain, copies, and host-visible finite gate
together.

## Primary architecture

Use one cuSOLVER-free CUDA CTA per 256x256 matrix. Stage only the lower triangle
in packed shared memory (32,896 FP32 values, 131,584 bytes), which fits where a
full padded tile would not. Factor 16-wide diagonal blocks, solve panel rows in
registers, and apply rank-16 trailing dots inside the same launch. Use 512 or
1024 threads so 64 CTAs expose enough per-matrix row/element parallelism on the
B200.

The target path may use neither cuSOLVER nor any auxiliary/concurrent CUDA
queue API. It must return the required lower-triangular representation directly.

## Bounded ladder

1. Packed-lower shared-memory blocked-16 factorization, 512 threads.
2. Increase to 1024 threads if trailing work is the dominant internal phase.
3. Change block width to 32 if diagonal synchronization dominates and compile
   resources remain safe.
4. Replace packed scalar staging with a lower-only padded/tiled layout if index
   arithmetic dominates.
5. Use a two-level 128+128 decomposition with tensor-core panel/trailing work if
   the scalar one-CTA design cannot cross 2x.

Stop after six distinct correct architectures if none crosses 2x. Record device
clock phase constituents for the best design so an exhaustion decision is
causal rather than empirical only.

## Promotion gates

Require static checks and dense paired correctness first. Then require all six
input families with zero new fallbacks, the revision-4 machine audit, full
15-shape paired parity, Popcorn test 17/17, and a serial ranked submission. After
2x, continue only for statistically real same-process gains and submit each
fully gated improvement.

## Outcome

V35 satisfied the shape goal on the exact full grid at 225.192 -> 111.608us
(2.0177x), passed all six numerical families and Popcorn 17/17, and was adopted
as ranked submission `#890037`. Public/secret geometric means are
825.4657219594694us / 824.9085045342571us, improving frozen `#888996` by
9.940% / 4.508%. See `notes.md` for the architecture ladder and compile-time
failure analysis.

A later non-overlapping merge with exp 044 improved the local paired aggregate
another 1.0130x, but official test `#890068` hit the exact six-minute compile
limit. It was not ranked, and the root source was restored to exact `#890037`.
