# Experiment 052: MXFP8 large-matrix transfers

## Control and objective

- Exact ranked control: Popcorn `#890798`, commit `f90ef909`, source SHA-256
  `fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
- Campaign public target: at most `400.988590us` versus `801.977179us`.
- This experiment pursues genuine arithmetic/quantization improvements only;
  no replay, pointer cache, workspace cache, or evaluator-dependent state.

## Bounded ladder

1. V1: replace TF32 panel products at `1x8192` with the already-shipped
   block-scaled MXFP8 quantize + vendor `_scaled_mm` path.
2. V2: at `1x16384`, raise the shared exponent by one only for a block whose
   normalized amax would exceed E4M3 max (`mantissa > 1.75`). This prevents
   clipping while preserving the higher-precision scale for all other blocks.
3. V3 only if V2 is accurate: combine independently validated `8192` and
   `16384` wins into a standalone source.

Each changed size needs positive route counters, zero fallback/error, all six
families with a preferred residual at most `8/20`, and paired same-process B200
timing. Full-grid/build/Popcorn gates are reserved for a combined frontier.
