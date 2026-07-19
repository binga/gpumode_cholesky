# Experiment 037 result — CUDA-rewrite premise refuted

Modal B200 compilation and timing (`asmprobe.json`) show:

| probe | latency per call |
|---|---:|
| shipped Triton micro | 14.379us |
| synthetic arithmetic floor | 10.456us |
| load/store-only floor | 10.083us |
| empty one-warp kernel | 10.409us |

The shipped kernel uses 236 registers with zero spills. Its PTX contains 474
`selp`, 148 shuffle, 32 barrier, and four `rsqrt` instructions, but removing
essentially all arithmetic leaves a roughly 10.1--10.5us device/dispatch floor.
The measured maximum rewrite headroom is therefore only **1.38x**, short of the
2x target before accounting for useful factorization work.

Verdict: **REJECTED / premise refuted.** A handwritten CUDA version can reduce
instruction count, but cannot halve this kernel on the measured launch path.
No candidate was integrated and no Popcorn slot was spent.
