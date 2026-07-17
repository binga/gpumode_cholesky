# Goal — Experiment 029: micro-chain cost reduction for the split32 shapes

Baseline: exact current ranked winner `#882958`, public `1096.0842452192236us`,
secret `1109.6451814508845us`, commit `1fc6ac258a80b2c8e2a086823c20edca63b31ab3`,
source SHA-256 `3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`
(copied here as `baseline-exp021.py`).

## Diagnosis this experiment acts on

The split32 chain is CUDA-graph replayed (or eager first-touch), so host launch
overhead is already amortized and end-to-end time is approximately the sum of
kernel self-times (exp-019 profile: 790us of kernels vs 827us end-to-end at
4x1024). The dominant term is `_micro_potrf_gj32`: 13.7us/launch x n/32
launches = 437us of 827us (53%) at 4x1024, ~880us of ~2400 at 8x2048, and it is
latency-bound (one warp, serial rank-4 chain), so batch does not amortize it.
Roughly half of the kernel's per-iteration work is the interleaved Gauss-Jordan
inverse maintenance: four full-tile multiply+cross-lane-reduce chains (`g0..g3`)
plus four row extractions and a third full-tile select-merge per rank-4 step.

## Hypothesis and variants (bounded ladder, at most 6)

1. `candidate-v1-noinv-solve.py` — remove the inverse from the micro entirely;
   the panel apply becomes a 32-step in-register forward substitution against
   the factored 32x32 diagonal block (the `_panel_solve_8x2048` idiom). Micro
   should drop toward ~7us/launch at the cost of ~2us/launch in the apply.
2. `candidate-v2-fused-apply-inner.py` — keep the micro; fuse `_panel_apply32`
   and `_panel_inner32(_subtile64)` into one kernel per micro step (each CTA
   recomputes the narrow `lj = Q @ Dinv^T` redundantly, removing one launch and
   one panel global-memory round trip per step) for the five subtile shapes.
3. `candidate-v3-elim-inverse.py` — keep the split kernel structure; replace the
   interleaved Gauss-Jordan with factor-first then a 32-step column-elimination
   inverse (broadcast FMA, no cross-lane multiply-reduce on the critical path).
4. `candidate-v4-rsqrt.py` — replace the four `1/sqrt` (sqrt.approx + div.full)
   chains per iteration with `tl.rsqrt` in the micro. Isolated compiler-level
   probe.
5. v5 — combination of the measured winners, all six split32 shapes.
6. Reserve.

## Gates

- `WINNER`: paired aggregate >= 1.10x on the probe targets (16x512, 4x1024),
  all six families pass on every changed shape, no unexpected fallback.
- `FRONTIER`: correct and any stable positive paired aggregate.
- `REJECTED`: incorrect, <= 1.0x, or any timed fallback.
- Official checker untouched; finite lower-triangular output preserved.
- No cuSOLVER on the changed fast path, no queue APIs, no scanner workarounds.
- Full 15-shape grid only after a stable target win; popcorn test 17/17 then at
  most one ranked submission.

## Cost guardrails

Initial probes on 16x512 + 4x1024 only (cheap shapes), four candidates in
parallel sandboxes, ~$5. Combination + six-shape probe + full grid ~$5 more
before checkpoint. Stop early on decisive negative evidence.
