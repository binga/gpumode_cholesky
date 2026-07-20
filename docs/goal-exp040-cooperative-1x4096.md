# Goal — Experiment 040: cuSOLVER-free cooperative `1x4096`

## Frozen baseline and target

- Current ranked source: `#888636`, commit `2b757f9`, public 992.551us,
  secret 1003.332us.
- Fresh byte-identical revision-3 audit baseline: 19.296us / **1528.456us** /
  3200.064us for `4096x32` / `1x4096` / `2x4096`.
- Prior exact-path profile: `1x4096` is 1528.4us wall / 1529.6us device.
  Required paired 2x target: at most **764.228us**.
- Off-target regression limit: 3%; aggregate grid must improve.

## Constituent diagnosis

One `getrf_wo_pivot_params_` factorization kernel costs 1393.0us, **91.1%** of
device time. Output staging costs 74.6us, lower-triangle cleanup 57.4us, and
the remaining setup kernels 4.6us. Wall-minus-device is effectively zero.
Analytic floors are about 17.4us of compulsory FP32 input/output traffic,
50.9us of TF32 arithmetic, and roughly 382us at a representative FP32 CUDA-
core roof. The only term large enough to cut is factorization itself.

Experiment 015 already rejected launch-only explanations: whole-call graph
capture was 0.969x, a Triton superpanel path was 0.181x, and a graph path with
vendor diagonal blocks was 0.578x. The new experiment therefore uses neither
cuSOLVER nor auxiliary/concurrent queues.

## Falsifiable architecture

Use a cooperative full-GPU CUDA kernel with one resident CTA per SM and
hardware grid synchronization. The matrix stays in global memory; a 64x64
right-looking factorization advances through 64 outer tiles. One CTA factors
the diagonal tile, the active panel tiles solve in parallel, and all resident
CTAs distribute lower-triangular trailing tiles. Tensor-core TF32 trailing
updates with FP32 accumulation target the dominant cubic work. A separate
upper-clear pass is allowed.

The mechanism is viable only if the measured hardware grid-barrier budget plus
diagonal/panel floor leaves credible room below 767us. A barrier-only probe is
the first paid gate; do not build the full kernel if synchronization alone
consumes the target.

## Bounded ladder

1. Cooperative barrier/occupancy floor for 64 steps and three phases per step.
2. Tile-64 cooperative FP32 correctness kernel.
3. WMMA TF32 trailing updates with FP32 accumulation.
4. Rank-2 diagonal work and fused panel phases.
5. Tile 32/128 comparison if the measured component budget identifies a clear
   barrier-versus-diagonal tradeoff.
6. Stop after six genuinely distinct correct measured variants.

Promotion requires an active cooperative-backend counter, all six input
families, paired latency at or below the fresh 2x threshold, the full paired
grid, Popcorn test 17/17, and then one ranked submission. No fallback timing is
accepted.

## Result — boundedly exhausted

The barrier gate passed, but six distinct correct cooperative architectures
all lost to the ranked vendor path. Best was V4 at 4066.43us versus 1530.73us
(0.376x); the required threshold was 764.228us. Device-clock instrumentation
split V1's 4019.65us accounted time into 837.12us diagonal, 1016.96us panel,
2142.08us trailing update, and 23.49us cleanup. Tile 64, tensor-core inverse
application, extra residency, left-looking products, and rank-128 superpanels
were each measured and rejected. Full evidence and the six-result table are in
`../experiments/040-cooperative-1x4096/notes.md`. No root-source change and no
Popcorn submission were warranted.
