# Speeding up kernels on Blackwell B200 GPUs — last30days

> Research captured 2026-07-29 via `/last30days` (v3.18.3). Date range: 2026-06-28 to 2026-07-28.
> Sources: GitHub, Hacker News, Reddit, YouTube (X unavailable this run — treat real-time X pulse as missing, not quiet).

## What I learned

**Blackwell kernel speedups are a dataflow rewrite, not a Hopper retune.** Practitioner writeups (Hazy Research / ThunderKittens, Colfax CUTLASS tutorials, Modular's matmul series, Paul Chan's B200 GEMM walkthrough) keep converging on the same stack: fifth-gen tensor cores via `tcgen05.mma`, Tensor Memory (TMEM) as the accumulator home, and CTA pairs so two SMs cooperate on one larger MMA (`cta_group::2`). Tensor cores are roughly 2–2.5x Hopper, so under-feeding them with small tiles wastes most of the chip — Hamza's TK deep-dive cites the shape trap where `M=64` on hardware that wants ~128 can leave you at roughly a quarter of peak.

**The libraries people actually ship against are CUTLASS and ThunderKittens.** Live GitHub this window: NVIDIA/cutlass at ~10.2K stars with CUTLASS 4.6.1 dated July 2026 (CuTe DSL fixes), and HazyResearch/ThunderKittens at ~3.6K stars as "tile primitives for speedy kernels," with open questions still asking which compute capabilities are supported. TK's Blackwell story is BF16/FP8 GEMM near cuBLAS and attention near cuDNN on B200 once TMEM + CTA pairs are in the pipeline. On the DIY side, regionaltantrums' cuda-oxide stream frames SoL GEMM bluntly: "a fast one looks like a single scary 800-line kernel, but it is really about eight ideas stacked on top of each other."

**Clusters, occupancy, and warp specialization are where the last 15–20% hides.** NVIDIA's Blackwell Tuning Guide keeps portable cluster size at 8 and lets B200 opt into 16 via `cudaFuncAttributeNonPortableClusterSizeAllowed`, with `cudaOccupancyMaxActiveClusters` as the launch check. Colfax and Modular both emphasize 2-SM UMMA plus warp-specialized TMA vs MMA pipelines coordinated with mbarriers; Modular reports ~85% of SOTA after that pass, and Paul Chan's iterative BF16 path claims ~106% of cuBLAS on 8192³ when cluster sync and TMEM epilogues are clean. NVIDIA's own nvMatmulHeuristics + CUTLASS auto-tune story on B200 is that a short heuristic candidate list can beat exhaustive search on wall clock and sometimes beat dynamic-cluster precompiled baselines.

**Precision and architecture splits matter more than people want.** Microbenchmark guidance (arXiv / IPDPS slides) pushes TMEM for multi-stage accumulators, ~64×64 tiles on TMEM-heavy paths, the decompression engine (DE) for memory-bound weight streams, and per-tensor precision (FP6 often strong on attention; FP8 still common on weight-heavy GEMM). DeepGEMM remains the SM100-flavored FP4/FP8 MoE path; consumer/SM120 Blackwell does not get the same `tcgen05`/TMEM playbook. That matches the loudest on-topic Reddit signal this window: the r/LocalLLaMA FlashAttention-3/4 post notes vLLM/SGLang fall back to FA-2 on consumer cards, rebuilds to FA-2 parity (~206µs on RTX 5090 for their config), and concludes "FA-3/4 optimizations are either not applicable or not helpful on consumer cards." Separately, an r/LocalLLaMA workstation post is less about MMA math and more about MoE runtime gymnastics — Ornith-397B at Q4 on one RTX PRO 6000 Blackwell 96GB with experts spilled to CPU RAM at ~2,354 tok/s prefill and ~20–24 tok/s decode.

**Social chatter this month was thin and skewed.** GitHub + long-form eng blogs carried the technique signal. HN surfaced Fireworks' MiniMax M3 sparse-attention-on-Blackwell note and a few Blackwell inference Show HNs, but also noise (B200 cloud scarcity after "Inkling"). YouTube was mostly rack tours and chip-war explainers, not kernel recipes.

## Key patterns from the research

1. **Rewrite for tcgen05 + TMEM + CTA pairs** — Hopper `wgmma` mental models underfeed Blackwell; tile for full-height MMA (often M≈128 / 2SM M≈256), keep accumulators in TMEM. (per Hazy Research, Colfax, Hamza)
2. **Prefer CUTLASS 4.6 / ThunderKittens over greenfield PTX** unless you are doing a learning kernel; both now expose Blackwell-shaped primitives. (per GitHub live stars + READMEs)
3. **Pipeline with warp specialization** — overlap TMA loads and MMA; use cluster-aware barriers, not just `__syncthreads()`. (per Modular, Paul Chan)
4. **Tune clusters with occupancy math** — portable 8, B200 opt-in 16; heuristics (nvMatmulHeuristics) beat blind exhaustive search. (per NVIDIA Tuning Guide + NVIDIA Developer Blog)
5. **Do not assume FA-3/4 or DeepGEMM tricks on every "Blackwell"** — datacenter SM100 ≠ consumer SM120; FA-3/4 often stay datacenter-only. (per r/LocalLLaMA, Blackwell GPU Wiki)
6. **Pick precision per tensor and use DE when memory-bound** — FP4/FP6/FP8 are workload-dependent, not free speedups. (per arXiv Blackwell microbenchmarks)

## Engine footer

```
✅ All agents reported back!
├─ 🟠 Reddit: 4 threads │ 51 upvotes │ 15 comments
├─ 🔴 YouTube: 10 videos │ 8,314,064 views │ 8/10 with transcripts
├─ 🟡 HN: 5 storys │ 26 points │ 4 comments
├─ 🐙 GitHub: 2 items │ 13,720 stars │ 707 comments
├─ 🗣️ Top voices: r/LocalLLaMA, r/Compilers
└─ 📎 Raw results saved to ~/Library/CloudStorage/OneDrive-NetAppInc/Documents/Last30Days/speeding-up-kernels-on-blackwell-b200-gpus-raw-v3.md
```

## Sources (web supplements)

- **Hazy Research** (hazyresearch.stanford.edu) — ThunderKittens on Blackwell: tcgen05, Tensor Memory, and CTA pairs; BF16/FP8 GEMM near cuBLAS on B200 and attention near cuDNN.
- **Hamza Elshafie** (hamzaelshafie.bearblog.dev) — TK anatomy for Blackwell: under-tiling (e.g. M=64 on hardware that wants ~128) leaves most tensor-core throughput unused; TMEM as dedicated accumulator space.
- **Colfax Research** (research.colfax-intl.com) — CUTLASS tutorial for Blackwell GEMM with thread-block clusters and 2-SM UMMA (`tcgen05.mma` + `cta_group::2`, multicast arrives).
- **Modular** (modular.github.io) — Matmul on Blackwell series: 2xSM MMA + warp-specialized TMA/MMA pipelining reaches ~85% of SOTA.
- **Paul Chan** (paulwillchan.com) — Iterative BF16 matmul on B200 reaching ~106% of cuBLAS on 8192³ via cluster sync, 2-CTA MMA, and TMEM epilogue discipline.
- **NVIDIA Docs** (docs.nvidia.com) — Blackwell Tuning Guide: portable cluster size 8, B200 opt-in cluster size 16 via `cudaFuncAttributeNonPortableClusterSizeAllowed`; use `cudaOccupancyMaxActiveClusters`.
- **NVIDIA Developer Blog** (developer.nvidia.com) — nvMatmulHeuristics + CUTLASS auto-tuning on B200 can beat exhaustive search wall-clock and sometimes beat dynamic-cluster baselines.
- **Blackwell GPU Wiki** (0xsero.github.io) — DeepGEMM as SM100-focused FP4/FP8 MoE GEMM using tcgen05 + TMEM; consumer SM120 needs CUTLASS/FlashInfer alternatives.
- **NVIDIA Developer Forums** (forums.developer.nvidia.com) — FP4 on DGX Spark/SM120 does not get B200-style tcgen05/TMEM paths; dequant-to-FP8/FP16 and smaller SMEM change the playbook.
- **arXiv / IPDPS slides** (arxiv.org) — Microbenchmarking Blackwell: TMEM for accumulator staging, ~64×64 tiles for TMEM paths, DE for memory-bound regions, FP6 vs FP8 tradeoffs.
- **Fireworks AI** (fireworks.ai) — HN-linked writeup on optimizing MiniMax M3 sparse attention kernels on NVIDIA Blackwell.
