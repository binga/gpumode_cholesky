# Goal — Experiment 030: route 1024x64 and 256x128 onto the split32 chain

Baseline: exact current ranked winner `#882958`, public `1096.0842452192236us`,
commit `1fc6ac258a80b2c8e2a086823c20edca63b31ab3`, source SHA-256
`3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`
(`baseline-exp021.py`).

## Hypothesis

`1024x64` (119.1us) and `256x128` (156.2us) are the only two shapes still on
graph-replayed vendor factorization, at 4-6x their leader-class targets
(25/32us). At n=64/128 the existing split32 Triton chain is only 4/10 kernel
launches (n=64 never reaches the trailing pass), every kernel is
occupancy-bound at batch 1024/256 rather than latency-bound, and no new kernel
code is required — only two `_SPLIT32_SHAPES` entries. The known-good fallback
(the exact ranked graph-replayed vendor paths) remains in place below the
split32 dispatch.

## Variant

`candidate-route-64-128.py`: add `(1024, 64)` and `(256, 128)` to
`_SPLIT32_SHAPES` (tf32x3 panels and trailing — the n-scaled tolerance is
tightest at small n) and to the subtile set. Nothing else changes.

## Gates

- `WINNER`: paired >= 1.20x on both target shapes, six families pass on both,
  zero unexpected fallbacks.
- `FRONTIER`: correct with a positive aggregate on the two targets.
- `REJECTED`: incorrect, <= 1.0x aggregate, or any timed fallback.
- No cuSOLVER on the new fast path (fallback layers unchanged), no queue APIs.
- Full grid + popcorn gates only via the combined exp-029/030 finalist, at
  most one ranked submission for the combination.

## Cost guardrails

One paired probe on the two target shapes (~$1); combined finalist grid shared
with exp 029. Stop on decisive negative evidence.
