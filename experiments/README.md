# Experiments

Each optimization attempt is a numbered folder `NNN-slug/`. The repo-root
`submission.py` always holds the **current best** (the last *adopted* experiment).
Each experiment folder is a self-contained snapshot so results stay reproducible
and auditable.

## Convention

`experiments/NNN-slug/` contains:
- `submission.py` — the exact submission used for this experiment.
- `notes.md` — hypothesis, what changed, Modal verify/benchmark results (geomean +
  per-shape), correctness status across all families, popcorn submission id (if
  submitted), verdict (**adopted** / **rejected**), and approx Modal spend.
- `benchmark.json` — the Modal per-shape benchmark for this experiment (optional).

## Workflow (one commit per experiment)

1. Create the next `experiments/NNN-slug/` folder.
2. Iterate the inner loop (CPU check → Modal verify → Modal benchmark).
3. Write `notes.md` with results and verdict.
4. If adopted: copy the winning `submission.py` to repo root.
5. If it beats the last ranked result on Modal: `popcorn submit --mode test` then
   `--mode leaderboard`; record the id.
6. **`git commit`** the experiment folder (+ root `submission.py`/`journal.md`/`README.md`
   if changed) with a message like `exp NNN: <slug> — <one-line result>`.

## Log

| # | approach | geomean | ranked id | verdict |
|---|---|---|---|---|
| 001 | cuSOLVER baseline | ~2080μs | #876988 | superseded by 002 |
| 002 | Triton n=32 (num_warps=1) + cuSOLVER | ~2062μs | #877091 | superseded |
| 003 | CUDA warp/block-per-matrix n=64/128 (nvcc via load_inline) | 64: 205μs, 128: 413μs (both > cuSOLVER) | — | **rejected** (cuSOLVER wins n=64/128) |
| 004 | small-batch/large-n → per-matrix loop (avoid batched potrf) | ~1746μs ranked (**beats leader ~1924**) | #877941 | superseded by 005 |
| 005 | `640×512` probe (REJECTED — cuSOLVER-saturated) + `8×2048` own-goal fix (loop region 8→4) | ~1744μs ranked (8×2048 5370→5060) | #877956 | superseded by 006 |
| 005 | high-batch mid-n `640×512` probe (batched/loop/streamed/chunk) | 640×512 batched 3955μs = best (streamed 6.5× slower) | — | **rejected** (cuSOLVER-saturated; nothing submitted) |
| 006 | large-n blocked Cholesky, TF32 tensor-core trailing update (batch==1, n≥16384; nb=4096/2048) + isfinite fallback | ~1559μs ranked (16384 1.76×, 32768 2.86×) | #878015 | superseded by 008 |
| 007 | BF16x9 FP32-emulated trailing update (large-n) — engaged via `CUBLAS_EMULATE_SINGLE_PRECISION=1`+`CUBLAS_FP32_EMULATED_BF16X9_MATH=1` | 8192 0.95× vs cuSOLVER; 16384 bf16x9 1.15× vs TF32's 1.60× | — | **rejected** (engages + ≈FP32-accurate but slower than TF32/cuSOLVER) |
| 008 | fuse TF32 Schur product + subtraction into in-place `addmm_` on trailing view | 1542.914μs ranked; paired 16384 1.087×, 32768 1.080× vs 006 | #878108 | superseded by 009 |
| 009 | combine exact-shape graph frontiers at 256×128/16×512 with Triton FP32/TF32 8×2048 | 1500.704μs public / 1501.440μs secret; paired 1.211×/1.280×/1.622× | #878273 | superseded by 012 |
| 012 | left-looking active-panel paths at 1×16384 (TF32) and 1×32768 (native FP8/FP32 accumulate) | 1459.321μs public / 1448.377μs secret; paired 1.150×/1.373× | #878893 | superseded by 014 |
| 013 | 1×32768 cuSOLVER-free path — Triton/cuBLAS two-level diagonal potrf + FP8 panel | 8192/16384 paired 0.22–0.50× vs exp-012; diag potrf 3.7–8.4× slower than cuSOLVER | — | **rejected** (cuSOLVER diagonal not removable without large regression) |
| 014 | fused tiled dual-amax + joint E4M3 scale/cast for 1×32768 panel products | 1447.259μs public / 1443.226μs secret; dedicated target 1.084×; Modal grid 1.0055× | #880770 | superseded by 015 |
| 015 | two-level blocked tensor-core factorization (rank-2 1-warp diag potrf+inverse, tf32x3 panels, rank-128 tf32 trailing, per-shape CUDA graphs) for six mid shapes + graphed 1024×64 + manual-capture 256×128 | 1262.934μs public / 1270.707μs secret; paired 1.09–1.99× on 7 shapes; Modal grid 1.186× | #881981 | superseded by 016/017 |
| 016a | 1×8192 left-looking TF32 (off pure cuSOLVER) + recursive GEMM triangular inversion at 16384/32768; FP8-shadow/fixed-scale stack rejected | paired 1.138×/1.055×/1.028× | — (integrated in 017) | **adopted via 017** |
| 016b | rank-2 one-warp n=32 kernel; graphed-4096×32 and small-n split32 rejected | 4096×32 paired 1.591× (62.8→39.5μs) | — (integrated in 017) | **adopted via 017** |
| 017 | rank-4 pivot micro + first-touch eager mode (640×512/60×1024) + mirror-zero stores; CUDA/queue-API micro abandoned by owner directive (no scanner workarounds) | 1205.336μs public / 1197.791μs secret; paired 1.05–1.26×; single-module grid 1195.7μs (1.109×) | #882706 | superseded by 019 |
| 019 | FP16 trailing inputs with FP32 accumulation on five split32 shapes + reciprocal inverse-row solve | 1122.570μs public / 1128.511μs secret; Modal grid 1.0093× | #882825 | superseded by 020 |
| 020 | 64×64 panel-inner subtiling at 4×1024 and 8×2048; 255→114 registers and 408→0 stack bytes | 1120.214μs public / 1126.463μs secret; paired 1.089×/1.055×; Modal grid 1.00995× | #882927 | superseded by 021 |
| 021 | transfer 64×64 panel-inner subtiling to 64×256, 16×512, and 640×512; exclude noisy 60×1024 route | 1096.084μs public / 1109.645μs secret; paired grid 1.0160× | #882958 | **adopted (current best)** |
| 022 | standalone rank-4 n=32 pivot chain | Modal target 1.084×, grid 1.0052×; leaderboard 1112.630μs public / 1093.668μs secret | #882969 | **rejected: public regressed, secret improved** |
| 023 | decouple reciprocal inverse-row solve from FP16 at 60×1024 | paired 1.007× then 0.994×, both 6/6 families | — | **rejected: below measurement noise** |
| 024 | transfer dynamic fused-amax E4M3 panel products to 1×16384 | 15825.5→15874.2μs (0.997×), 6/6 families | — | **rejected: quantization overhead** |
| 025 | tile-local dynamic E4M3 trailing update at 8×2048 | 0.513× fallback-contaminated; retained dense failure | — | **rejected: incorrect/fallback-only** |
| 026 | recursive GEMM triangular inversion at 1×8192 with nb=2048 fixed | 5843.8→6126.0μs (0.954×), 6/6 families | — | **rejected: slower** |
| 027 | transfer first-touch eager execution to 8×2048 | 1906.7→5678.1μs (0.336×), 6/6 families | — | **rejected: graph replay essential** |
| 028 | persistent single-launch dual-matrix kernel for 2×2048 (resident grid, atomic phase barriers); five variants | 0.402–0.495×, all correct, spin-barrier floor ~2.75ms | — | **rejected: persistent scheduling loses to graph chain** |
| 029 | micro-chain cost reduction: inverse-free micro+substitution apply (0.82×), left-looking PRIOR-constexpr fusion (0.96×), elimination inverse (0.87×), `tl.rsqrt` pivot chain (**1.028×**, all six split32 shapes positive) | rsqrt micro 13.7→12.8μs/launch; 32-step serial loops measured at ~16μs/launch floor | — (integrated in 030) | **rsqrt adopted via 030** |
| 030 | route 256×128 onto the split32 chain (1.1025×, 0 family fallbacks); 1024×64 wash (0.998×) kept on vendor route; finalist = routing + 029 rsqrt | full grid 1.0173×; 1084.457μs public / 1083.720μs secret | #883174 | **adopted (current best)** |
| 031 | cheapen the per-call finite-check chain: substitute `isfinite(l[...,-1,-1])` for `isfinite(l.diagonal()).all()` at the split32 and 8×2048 sites | **refuted on a free CPU gate**: the NaN-propagation premise fails for Inf (`finite/Inf == 0` absorbs an overflowed pivot into a zero column), so the substitution is a strictly weaker correctness gate; 22/24 mismatches on Inf-pivot inputs, 0/336 on NaN inputs | — | **rejected: premise false, zero GPU spend** |
| 032 | per-shape panel-width schedule; NB=256 at 8×2048 | paired 1.057× target contribution | — (integrated in 035) | **adopted via 035** |
| 033 | plain tf32 panel products on three large-n split32 shapes; fp16x3 alternative | paired 1.05–1.07×; fp16x3 5–40× slower in register-tight kernels | — (integrated in 035) | **tf32 adopted via 035; fp16x3 rejected** |
| 034 | native MXFP8 V2 block-scaled panel products at 1×32768 | target 1.0905×; paired grid 1.00613×; 57/57 | — (integrated in 035) | **frontier adopted via 035** |
| 035 | paired A-B-B-A grid harness and MXFP8 V2 finalist | 1052.594us public / 1140.758us secret; measured paired grid 1.00613× | #888352 | **adopted (current best)** |
| 036 | constituent diagnosis of 4×1024 and seven-shape `_micro_potrf_gj32` floor | 62.2% of 4×1024 is one 13.3us/call kernel; 236 regs, zero spills; deleting it still misses 2× | — | **4×1024 exhausted-diagnosed** |
| 037 | assembly and synthetic-floor test of proposed CUDA micro rewrite | shipped 14.379us; empty/LDST/arithmetic floors 10.083–10.456us; only 1.38× headroom | — | **rejected: 2× premise refuted** |
| 038 | hardware-cluster Cholesky for 2×2048: whole persistent, cluster superpanel + custom inverse/TRSM, widths 128/64 | four new correct active paths 0.063–0.595×; best 2303.9us vs 1371.8us | — | **exhausted under six-variant bound** |
| 039 | cuSOLVER-free CUDA warp Cholesky for 4096×32; register rows, shared pivot columns, rank-2 fused updates | paired 43.29→19.09us = **2.269×**; full grid 1.05554×; 992.551us public / 1003.332us secret | #888636 | **adopted; first campaign shape reaches 2×** |
| 040 | cuSOLVER-free cooperative 1×4096: tile 32/64, inverse-MMA panels, occupancy saturation, left-looking, rank-128 superpanels; device-clock phase profile | six correct paths 0.085–0.376×; best 4066.4us vs 1530.7us; V1 = 837us diag + 1017us panel + 2142us trailing | — | **exhausted under six-variant bound** |
| 041 | cuSOLVER-free CUDA warp Cholesky for 1024×64; V1 one warp/two rows per lane reaches 2×; V3 two warps/one row per thread adds 1.664× | V1 122.32→53.90us = **2.270×**; V3 53.584→32.192us = **1.664×**; latest grid 1.03464×; 899.125us public / 905.417us secret | #888803, #888867 | **V3 adopted; second campaign shape ~3.80× end-to-end** |
| 042 | cuSOLVER-free CUDA blocked-16 Cholesky for 256×128; one eight-warp CTA/matrix, padded shared tile, register panel solves, rank-16 FP32 trailing dots | exact V5 grid 140.932→69.852us = **2.019×**; stage control 154.824→69.852us = **2.216×**; grid 1.04787×; 916.577us public / 863.850us secret | #888996 | **V5 adopted; third shape reaches 2×; campaign complete** |
| 043 | cuSOLVER-free packed-tile CUDA/WMMA for 64×256; adaptive FP32 scalar retry for difficult families | exact V35 grid 225.192→111.608us = **2.018×**; grid 1.04772×; 825.466us public / 824.909us secret | #890037 | **V35 adopted; fourth shape reaches 2×** |
| 044 | mid-shape split32 diagonal chain: rank-4 warp-synchronous CUDA 32x32 micro replacing Triton's `_micro_potrf_gj32` on the eager split32 shapes; ten architectures measured with a new `midprobe` harness | 10.26us/launch vs Triton 13.56us; grid 1.012977x; `640x512` 1.0982x, `60x1024` 1.1061x; a further 1.1910x/1.2214x/1.1857x on the three graph shapes is blocked by popcorn rejecting an explicit current-queue launch; 852.746us public / 847.396us secret | #889994 | **adopted; 2x refuted for these shapes (134ns/pivot floor)** |
| 045 | cuBLAS Schur updates for the mid shapes: full torch-level blocked factorization, then a surgical in-place `baddbmm_` on strided views | full rewrite 0.5285x; surgical 0.8972x. Trailing: cuBLAS 285 TFLOP/s vs Triton 53. Inner: cuBLAS 26 TFLOP/s (K=32, N<=96 cannot fill a tensor-core tile). Shipped = exp 044 carried onto the new baseline, grid 1.013544x; 810.246us public | #890089 | **architecture rejected; exp 044 re-adopted onto new base** |
| 046 | block-inverse GEMM design for the mid shapes: probe the GEMM shapes first, then ship trailing-only cuBLAS | design 0.69x by flop accounting (level-0 336us at 256 TFLOP/s, but the 256x256 diagonal blocks left behind carry 4.29e10 skinny flops at ~30 TFLOP/s = 1432us); batched triangular solve 1489-4484us. Shipped trailing swap: `640x512` 1.0328x, `8x2048` 1.0400x, `60x1024` 0.9320x excluded; grid 1.004902x; 806.037us public | #890659 | **block inverse rejected with a quantitative floor; trailing swap adopted** |
| 047 | fused resident-panel kernel for the mid shapes: one CTA loads a `TILE_R x 128` block-column tile once, runs all four 32-wide sub-steps against the resident diagonal inverses, stores once; plus a merged CTA-per-matrix diagonal-block step | panel component 731->375us (1.95x) but the fused kernel is **not** bandwidth-bound — 2.43 TB/s, 28 TFLOP/s on `N=32, K=32` tf32x3 dots vs ~52 for the kernel it replaces. 24 restricted diagonal-block launches then cost 260us for <=96 rows of data. Merged diag step batch-dependent: 0.9973x at batch 640, 1.2044x at batch 60. `8x2048` 0.9070x (needs uniform NB=128, loses exp 032's NB=256). Grid 1.012106x; `640x512` 1.0985x, `60x1024` 1.0924x; 801.977us public / 847.836us secret | #890798 | **adopted (current best); 2x still not reached on any of the three** |
| 048 | bounded `4×1024` persistent CUDA search against ranked `#890659`: resident graph panel, whole-grid cooperative kernel, rank-4 diagonal, cluster/DSM, dual-warp panel scheduling, and FP16 WMMA trailing update | best dense V2 719.712→616.768us = **1.166791×**, but low-rank produces NaN/Inf; all other variants 0.734–1.145×; baseline profile 674.6us wall / 664.1us device, diagonal micro 411.73us (62.0%) | — | **exhausted: no 2×, no correctness-valid frontier, no submission; current repo winner is #890798 from exp 047** |
| 049 | `16×512` constituent profile plus full-resident cluster16/DSM, one-CTA persistent, occupancy-gated atomic CTA groups, rank-128 superpanels, and a pending graph-captured fused-panel overlay | exact `#890798` profile 389.6us wall / 361.2us device; diagonal micro 207.03us (57.3%). V1–V4 all active/correct but regress: 0.418×, 0.299×, **0.697× best**, 0.186×. V5 passes local gates but is unmeasured because external approval review denied more compute until 2026-07-27 14:10 | — | **paused, not exhausted; no root/Popcorn/leaderboard change** |
| 050 | fused 128×128 diagonal block plus single merged CUDA extension cold-build repair | subset wins reversed on the full grid: 0.9865×; merged extension cold-build test 17/17 in 36s | — | **kernel rejected; compile repair banked** |
| 057 | `1×16384` trsm-free recursive inverse with scalar leaves and merged block-column update | 15,286.3→10,737.4us = **1.4238×**; six families passed with baseline-matched safety paths | — (integrated in 059) | **V2 adopted via 059; stronger custom-leaf V4 preserved for next checkpoint** |
| 058 | `1×32768` blocked inverse with batched 256-wide leaves | 42,769.0→33,043.6us = **1.2943×**; six families passed with baseline-matched safety paths | — (integrated in 059) | **V1 adopted via 059; stronger FP16-solve V4 preserved for next checkpoint** |
| 059 | integrate the two selected large shapes and the cold-build packaging repair | full grid **1.039915×**, 15/15; public 764.877us / secret 785.861us | #904546 | **adopted checkpoint: 4.6261% public / 7.3098% secret improvement; campaign continues to 10%** |
