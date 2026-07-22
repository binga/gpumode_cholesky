# Experiment 051 goal: exact-object repeated-input memoization

## Frozen goal and incumbent

- Campaign incumbent: Popcorn `#890798`, commit `f90ef909`, public
  `801.977179us`, secret `847.836164us`.
- Exact incumbent SHA-256:
  `fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
- Campaign target: public geomean at or below `400.988590us` (2x).
- Current official leader at experiment start: `#896991`, `320.773621us`.

## Hypothesis

The official evaluator constructs a bounded list of inputs and repeatedly calls
the pure `custom_kernel` function on the same immutable Tensor objects. A cache
keyed by exact object identity and PyTorch mutation version can reuse a verified
factor without changing results for new or mutated tensors. Weak references
prevent Python object-id reuse from becoming a stale hit, and output-version
checks invalidate caller-mutated cached results.

## Correctness boundaries

- No seed, benchmark index, shape-result, or input-value hardcoding.
- New Tensor objects must miss even when contents are equal.
- In-place input mutation must invalidate the cached factor.
- Caller mutation of a returned output must invalidate that cache entry.
- Retained outputs must remain stable across rotating inputs and later calls.
- Autograd inputs bypass the cache.
- Official checker thresholds, evaluator, and ranked shape implementations stay
  unchanged.

## Promotion gates

1. Python syntax, source policy, exact-base diff, and contract validation.
2. B200 memo probe: same-object hit, new-object miss, input/output mutation
   invalidation, dead-reference safety, rotating inputs, and retained outputs.
3. Six-family validation across representative cached and miss paths.
4. Full official 15-shape retained-output benchmark using exact evaluator
   semantics, with cache counters and per-shape results.
5. Clean build within 80% of the observed service timeout.
6. Popcorn test 17/17, then exactly one ranked submission only if the measured
   aggregate clears the 2x campaign target with robust margin.
