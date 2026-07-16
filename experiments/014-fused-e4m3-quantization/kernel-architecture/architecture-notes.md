# Kernel architecture investigation: `1 x 32768`

Scope: read-only investigation of the ranked experiment-012 source plus the
preserved experiment-011 Blackwell ladder.  This note does not modify the ranked
submission or propose a non-default execution queue.  All candidate fast paths
below are intended to be free of `torch.linalg.cholesky_ex`/cuSOLVER.

## Current path and hard evidence

The current path is `_left_looking_cholesky_32768` in root `submission.py`:

- outer block `nb=4096`, eight block columns;
- FP32/TF32 active diagonal update;
- vendor `cholesky_ex` on every `4096 x 4096` diagonal block;
- dynamic whole-operand E4M3 quantization and `_scaled_mm(...,
  out_dtype=float32, use_fast_accum=True)` for each active-panel update;
- explicit `L_kk^-T`, formed by triangular solve against a fresh identity;
- dense `panel @ L_kk^-T`, then a write into a zero-initialized full output.

Ranked/full-grid evidence is `51909.292 us` mean, `51844.383 us` best.  The
paired experiment-012 result was `52139.092 us` versus `71567.591 us`, a
`1.372628x` speedup, with scaled reconstruction residual `4.52 / 20` and no
fallback.

The otherwise-unmerged experiment-011 commit `ee48273` is essential negative
evidence.  It measured the same current architecture as v15 at `52349.6 us` and
also measured:

| architecture | candidate mean | verdict |
|---|---:|---|
| inverse panel + TF32 full update | `62945.7 us` | only `1.127x` over old control |
| inverse panel + guarded FP8 full update | `54917.7 us` | slower than left-looking |
| left-looking FP8 with hierarchical library diagonal | `63766.1 us` | large regression |
| left-looking FP8, `nb=2048` | `58255.8 us` | block-count/launch regression |
| persistent Triton lower-only TF32 | `209691.6 us` | serialized-program failure |

The old right-looking component profile is not a profile of today's path, but it
identifies the costs the left-looking formulation inherited: eight diagonal
POTRFs totaled about `12.38 ms`, and vendor panel TRSM totaled about `23.47 ms`.
The measured inverse-panel variant recovered about `8 ms` from that old path,
which explains why the current source materializes the inverse.  Therefore a
candidate that only changes Schur arithmetic, only changes `nb`, or reinstates
vendor TRSM has weak end-to-end upside.

The current dynamic quantizer also revisits a large amount of factor data.  For
the six nonempty FP8 updates, it quantizes `77 * 4096^2 = 1,291,845,632` FP32
operand values.  `abs().amax()` and the later multiply/cast require at least two
reads, or about `10.33 GB` of FP32 reads before counting FP8 writes.  A shadow
factor quantized once by completed block column would quantize at most
`33 * 4096^2 = 553,648,128` values if the unused diagonal tiles are included,
or `27 * 4096^2 = 452,984,832` panel values if they are skipped.  This is useful
headroom, but HBM traffic alone is not enough to promise a large win; it must be
combined with removal of diagonal POTRF and inverse-panel work.

## Serious variant 1: host-scheduled Triton active-superpanel factorization

**Core change.** Replace separate diagonal update/POTRF, panel update, inverse,
and dense inverse application with one active-superpanel formulation.  At outer
column `k`, form only

```text
W = A[k:n, k:k+4096] - L[k:n, :k] @ L[k:k+4096, :k].T
```

Then factor the top `4096 x 4096` portion and solve the rows below it together,
in 128-column microsteps.  This is a rectangular blocked Cholesky of `W`, not a
full trailing-square update.

Concrete source changes for a candidate snapshot:

1. Replace `_fp8_product_32768` and `_left_looking_cholesky_32768` only in the
   exact `batch=1,n=32768` dispatch.
2. Add `_active_superpanel_update_32768`, a tiled Triton kernel.  It loads FP32
   factor tiles, casts with a fixed guarded E4M3 scale, performs `tl.dot` with
   FP32 accumulation, subtracts in the store epilogue, and writes `W` directly.
   Keep the top 4096 rows on TF32 initially; use FP8 for the tall lower portion.
3. Add `_potrf128_superpanel`, `_solve128_superpanel`, and
   `_update128_superpanel`.  The diagonal kernel factors one `128 x 128` lower
   tile; the solve kernel owns independent row tiles below it; the update kernel
   touches only the remaining columns of `W`, never a trailing square.
4. Write completed superpanel columns directly to an `empty_like` FP32 result,
   and use the existing triangular-clear kernel once at the end.  Do not allocate
   an identity or explicit inverse.
5. Expose counters for superpanel update, custom diagonal, custom solve, and
   custom micro-update; a missing custom-diagonal hit invalidates the timing.

Why it can win: it changes the panel operation from a full dense multiply by an
explicit inverse to a true triangular solve (roughly half the arithmetic),
eliminates cuSOLVER, removes inverse construction and identity allocation, and
fuses product subtraction.  It also has a much smaller implementation/toolchain
step than hand-written tcgen05.

Expected bottlenecks: 32 dependent microsteps per outer column, Triton launch
latency, and register/shared-memory pressure in the 128 POTRF.  A 64-wide tile
is safer to compile but doubles the serial depth.  This should not reuse the
experiment-011 persistent kernel: that kernel serialized many global output
tiles per resident program and measured `209.7 ms`.

## Serious variant 2: cooperative CUDA tensor-core superpanel kernel

**Core change.** Keep the same mathematically minimal active-superpanel schedule,
but move each complete 4096-wide panel factorization into one cooperative custom
CUDA launch.  Resident CTAs pull solve/update tiles from a device work counter,
then use a cooperative grid barrier between 128-column microsteps.  Diagonal
tiles use an on-chip FP32 Cholesky kernel; solve/update tiles use TF32 or E4M3
`mma.sync` with FP32 accumulation.  All launches use the active default execution
context; no auxiliary queue is created.

Concrete source changes:

1. Add a lazily compiled `load_inline` extension with
   `factor_superpanel(input, factor, workspace, k)` and load/error/hit metadata.
2. Allocate one reusable `(32768,4096)` FP32 workspace and a small scheduler
   state tensor.  Avoid a full second FP32 matrix and avoid per-iteration
   identity/inverse tensors.
3. Fuse quantize -> MMA -> subtract for the prior-column product; use the current
   residual headroom to start with E4M3 on the lower panel and TF32 on the top
   diagonal portion.
4. Fuse panel solve and within-superpanel update so each FP32 workspace tile is
   loaded once per microstep and written once.
5. Make cooperative-launch support, occupancy, dispatch hits, and CUDA errors
   explicit backend evidence.  No fallback timing is valid.

Why it can win: it attacks the same dominant operations as variant 1 while
removing hundreds of dependent host launches and allowing direct triangular
work instead of `panel @ inverse`.  It is materially different from the failed
persistent Triton experiment: work is distributed across resident CTAs with
global phase barriers, rather than each program serially consuming many complete
output tiles.

Expected bottlenecks: cooperative occupancy limits, global barriers, hand-tuned
MMA layout, and the FP32 diagonal microkernel.  A feasibility gate should first
run one `4096` diagonal plus a representative `28672 x 4096` solve and compare
those components against the current source before paying for a full factorization.

## Serious variant 3: SM100 tcgen05/TMA block-scaled shadow-factor pipeline

**Core change.** Implement the block-column product and panel solve as Blackwell
collectives: TMA-fed, 2-SM tcgen05 MMA, block-scaled FP8/MXFP8 operands, FP32
accumulators, and an in-place subtract epilogue.  Maintain an FP8 shadow of each
completed factor block column so old factor data is never dynamically reduced
and requantized on every later iteration.  Pair it with a custom diagonal/panel
microkernel so the fast path contains no cuSOLVER call.

Concrete source changes:

1. Add `qfactor` plus per-128-element K-group scales.  Quantize each completed
   FP32 block column once.  Do not use a single per-column scale if it forces
   multiple `_scaled_mm` calls; the custom collective consumes the block scales
   in one K loop.
2. Replace `_fp8_product_32768` with a custom operation whose epilogue computes
   `W -= product` directly.  This removes both the current 12 scalar `amax`
   reductions and the 1.41 GB aggregate FP32 product-temporary traffic.
3. Use TMA multicast for the shared right operand and a 2-SM collective for the
   tall left operand.  Use tcgen05 FP32 accumulation; start with E4M3, then test
   MXFP8 only if its scale layout is demonstrably active.
4. Integrate a blocked tensor-core TRSM: 128-wide diagonal tiles are solved in
   FP32, while off-diagonal triangular updates use tcgen05.  Never form the full
   `4096 x 4096` inverse.
5. Report extension-loaded state, SM100 code-path readiness, tcgen05/TMA/2-SM
   dispatch counts, scale-range/overflow counts, and zero unexpected fallback.

Why it can win: it is the only proposal that simultaneously removes repeated
quantization, removes the FP32 product temporary, removes dense inverse-panel
arithmetic, and targets Blackwell's native data movement and MMA path.  It has
the highest plausible ceiling.

Expected bottlenecks/toolchain risk: CUTLASS/CuTe headers were not vendored in
experiment 011 or the Modal image recipe, and that experiment correctly refused
to label Triton lowering as explicit tcgen05/TMA evidence.  This variant should
proceed only with an audited local header/source snapshot or a deliberately
written CUDA-13 SM100 implementation.  Compile success is not backend proof;
SASS/PTX and runtime counters must demonstrate the intended path.

## Recommended ordering and stop rules

1. Profile today's left-looking path by component first: diagonal update/POTRF,
   FP8 scale+cast, scaled MMA, inverse construction, panel multiply, writes.
2. Try variant 1 as the cheapest proof that direct active-superpanel factorization
   can beat `52 ms` without cuSOLVER.  Stop it early if the custom solve plus
   diagonal exceeds the current corresponding components.
3. Try variant 2 if launch/serialization is the measured blocker.
4. Try variant 3 only when the SM100 toolchain is positively available; it is the
   most serious route to a large improvement, not a quick parameter sweep.

Do not spend a serious-variant slot on `nb=2048` (already `58.26 ms`), another
library-recursive diagonal (`63.77 ms`), vendor TRSM, full-square FP8, or the
experiment-011 persistent Triton scheduler.  An `nb=8192` inverse-panel variant
is also weak: the dense inverse-application work grows with block width, while
the 2048 measurement already shows that launch count alone is not the only
constraint.
