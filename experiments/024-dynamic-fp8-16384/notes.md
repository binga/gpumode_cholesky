# Experiment 024 — dynamic FP8 panels at 1x16384

**Status: REJECTED — slower than TF32.** Exact baseline is ranked submission
`#882958`.

The candidate changes only the `1x16384` panel product from TF32 `addmm_` to
the existing dynamic tiled-amax, fused E4M3 quantization, FP32-accumulating
scaled-matmul path already ranked at `1x32768`.

The paired B200 probe passed all six families but measured `15825.5us ->
15874.2us` (**0.997x**). Profiling attributed approximately `633us` to six
dual-amax launches and `541us` to six fused scale/cast launches; at 16384 that
quantization overhead erases the FP8 compute saving. The successful 32768 path
therefore does not transfer downward. No full grid, Popcorn test, or leaderboard
submission was run; root remains exact `#882958`.
