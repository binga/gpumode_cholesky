# Experiments — what has been tried, and what it moved

**Loop stage [I1] check what is spent, [I8]/[O8] record an outcome.** One row per
experiment. **Add a row here when an experiment closes**, adopted or rejected —
this file plus `experiments/NNN-*/notes.md` are the record; `journal.md` is the
narrative archive.

Two tables: the **strategy matrix** (which levers each experiment used, and the
column totals that say where effort has gone) and the **prose log** (what each
experiment actually did, with its ranked id and verdict).

For what to try next, see `docs/levers.md`. For the current incumbent, see
`docs/STATUS.md`.

---

## Strategy columns

| Key | Strategy |
|---|---|
| `Rt` | Routing / dispatch — which path a shape takes |
| `Bk` | Blocked, left-looking, or recursive factorization |
| `Tr` | Triton kernel |
| `CU` | Custom CUDA / WMMA / tcgen05 |
| `LP` | Low precision — TF32, FP16, FP8/MXFP8, BF16x9 |
| `Fu` | Fusion — fewer launches, fewer temporaries |
| `Gr` | CUDA Graphs |
| `Pe` | Persistent / cooperative / cluster / DSM |
| `Ov` | Per-call overhead — copy-in, clone-out, finite check |
| `Hn` | Harness, measurement, or packaging work |

Latency column shows the headline measured number: paired speedup for a shape
experiment, ranked public geomean for an integration.

## Matrix

| # | Rt | Bk | Tr | CU | LP | Fu | Gr | Pe | Ov | Hn | Latency impact | Verdict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|---|
| 001 | | | | | | | | | | | ~2080us ranked | baseline |
| 002 | ✓ | | ✓ | | | | | | | | ~2062us ranked | superseded |
| 003 | | | | ✓ | | | | | | | n=64 205us, n=128 413us — both worse | rejected |
| 004 | ✓ | | | | | | | | | | ~1746us ranked | superseded |
| 005 | ✓ | | | | | | | | | | ~1744us ranked; 8x2048 5370→5060us | superseded |
| 006 | | ✓ | | | ✓ | | | | | | ~1559us; 16384 1.76x, 32768 2.86x | superseded |
| 007 | | | | | ✓ | | | | | | 8192 0.95x; 16384 1.15x vs TF32's 1.60x | **rejected** |
| 008 | | | | | ✓ | ✓ | | | | | 1542.9us; paired 1.087x / 1.080x | superseded |
| 009 | ✓ | | ✓ | | | | ✓ | | | | 1500.7us; paired 1.211x/1.280x/1.622x | superseded |
| 012 | | ✓ | | | ✓ | | | | | | 1459.3us; paired 1.150x/1.373x | superseded |
| 013 | | ✓ | ✓ | | ✓ | | | | | | 0.22–0.50x; diag potrf 3.7–8.4x slower | **rejected** |
| 014 | | | | | ✓ | ✓ | | | | | 1447.3us; target 1.084x, grid 1.0055x | superseded |
| 015 | | ✓ | ✓ | | ✓ | | ✓ | | | | 1262.9us; paired 1.09–1.99x on 7 shapes | superseded |
| 016a | | ✓ | | | ✓ | | | | | | paired 1.138x / 1.055x / 1.028x | adopted via 017 |
| 016b | | | | ✓ | | | ✓ | | | | 4096x32 62.8→39.5us = 1.591x | adopted via 017 |
| 017 | | | | ✓ | | ✓ | | | ✓ | | 1205.3us; paired 1.05–1.26x | superseded |
| 019 | | | | | ✓ | | | | | | 1122.6us; grid 1.0093x | superseded |
| 020 | | ✓ | ✓ | | | | | | | | 1120.2us; paired 1.089x / 1.055x | superseded |
| 021 | ✓ | | ✓ | | | | | | | | 1096.1us; paired grid 1.0160x | superseded |
| 022 | | | | ✓ | | | | | | | target 1.084x but public regressed | **rejected** |
| 023 | | | | | ✓ | | | | | | 1.007x then 0.994x | **rejected** (noise) |
| 024 | | | | | ✓ | ✓ | | | | | 15825.5→15874.2us = 0.997x | **rejected** |
| 025 | | | | | ✓ | | | | | | 0.513x, fallback-contaminated | **rejected** |
| 026 | | ✓ | | | | | | | | | 5843.8→6126.0us = 0.954x | **rejected** |
| 027 | | | | | | | ✓ | | ✓ | | 1906.7→5678.1us = 0.336x | **rejected** |
| 028 | | | | ✓ | | | | ✓ | | | 0.402–0.495x; spin-barrier floor ~2.75ms | **rejected** |
| 029 | | | ✓ | | | ✓ | | | | | rsqrt 1.028x; 13.7→12.8us/launch | partial adopt |
| 030 | ✓ | | | | | ✓ | | | | | 1084.5us; grid 1.0173x | superseded |
| 031 | | | | | | | | | ✓ | | **refuted on a free CPU gate, zero GPU spend** | **rejected** |
| 032 | | ✓ | | | | | | | | | paired 1.057x target contribution | adopted via 035 |
| 033 | | | | | ✓ | | | | | | tf32 1.05–1.07x; fp16x3 5–40x slower | split verdict |
| 034 | | | | | ✓ | | | | | | target 1.0905x; paired grid 1.00613x | adopted via 035 |
| 035 | | | | | ✓ | | | | | ✓ | 1052.6us public / 1140.8us secret | superseded |
| 036 | | | | | | | | | | ✓ | diagnosis: 62.2% of 4x1024 is one 13.3us kernel | diagnosis |
| 037 | | | | ✓ | | | | | | ✓ | shipped 14.379us vs 10.08–10.46us floor | **2x premise refuted** |
| 038 | | | | ✓ | | | | ✓ | | | 0.063–0.595x; best 2303.9 vs 1371.8us | **exhausted** |
| 039 | | | | ✓ | | | | | | | 43.29→19.09us = **2.269x**; 992.6us | **adopted** |
| 040 | | | | ✓ | | | | ✓ | | | 0.085–0.376x; best 4066.4 vs 1530.7us | **exhausted** |
| 041 | | | | ✓ | | | | | | | 122.3→53.9us **2.270x**, then 1.664x; 899.1us | **adopted** |
| 042 | | ✓ | | ✓ | | | | | | | 140.9→69.9us = **2.019x**; 916.6us | **adopted** |
| 043 | | | | ✓ | ✓ | | | | | | 225.2→111.6us = **2.018x**; 825.5us | **adopted** |
| 044 | | | | ✓ | | | | | | ✓ | 10.26 vs 13.56us/launch; 852.7us | adopted; 2x refuted |
| 045 | ✓ | ✓ | | | | | | | | | full rewrite 0.5285x; surgical 0.8972x | **rejected** |
| 046 | ✓ | ✓ | | | | | | | | | design 0.69x by flop accounting; 806.0us | mixed |
| 047 | | | | ✓ | | ✓ | | | | | panel 731→375us (1.95x) but grid only 1.0121x; 802.0us | **adopted** |
| 048 | | | | ✓ | ✓ | | | ✓ | | | best V2 **1.167x** but low-rank NaN/Inf | **exhausted, discarded** |
| 049 | | | | ✓ | | | | ✓ | | | 0.418x / 0.299x / 0.697x / 0.186x | paused |
| 050 | | | | ✓ | | ✓ | | | | ✓ | subset wins reversed on full grid: 0.9865x | kernel rejected |
| 057 | | ✓ | | | | ✓ | | | | | 15286.3→10737.4us = **1.4238x** | adopted via 059 |
| 058 | | ✓ | | | | | | | | | 42769.0→33043.6us = **1.2943x** | adopted via 059 |
| 059 | ✓ | | | | | | | | | ✓ | grid **1.0399x**; 764.9us public / 785.9us secret | **adopted** |
| 060 | | ✓ | ✓ | | ✓ | | | | | | target paired 1.0515x; grid 1.0068x | **adopted** |
| 061 | | ✓ | | | ✓ | | | | ✓ | | 1.156x / 1.280x on the two large shapes; 745.8us | **adopted** |
| 062 | ✓ | | | ✓ | | | | | | | 733.5us public / 721.8us secret (-1.64%) | **adopted** |
| 063 | | | | ✓ | ✓ | | | | | | 675.8us public / 674.4us secret (-9.4% / -9.0%) | **adopted** |
| 064 | | ✓ | | | ✓ | | | | | ✓ | full grid **1.0073x** CI95 excludes 1.0 | **adopted** |
| 065 | | | | ✓ | | ✓ | | | | | block 45.669→39.742us = **1.149x**; grid **1.0122x** CI95 excludes 1.0; public **−3.79%** but secret **+5.71%** | **rejected at LB** |
| 066 | ✓ | | | ✓ | | | | | | | e62_diag128 enroll 640x512: 1242→2051us = **0.606x (1.651x SLOWER)**; grid 725.21→750.02 | **rejected: occupancy (batch 640 = ~4.3 waves)** |
| 067 | ✓ | | | ✓ | | | | | | | e62_diag128 enroll 60x1024: paired **1.2426–1.3101x** (1186/1253→956us), 14 other shapes flat, 0 new fallbacks, more accurate (resid 3.31 vs 9.33); ranked #922201 public+secret passed; re-confirm grid **1.0180x** CI95 excludes 1.0; Popcorn test 17/17 | **adopted (current best #922201); pure-latency win carries to secret, distinct from #914341 precision class** |
| 068 | | | | ✓ | ✓ | | | | | ✓ | tcgen05 GEMM via **ThunderKittens** (level_06 class, 293 TFLOPs @4096³ = TK's published number); **0.19–0.28× cuBLAS bf16, 0.36–0.56× cuBLAS tf32** on 4 large-n/trailing shapes; correct (rel_err 0.2–0.3%) | **rejected: GEMM primitive not the bottleneck. TK's production ceiling (~1540 TF) only ties cuBLAS bf16 (1330–1588 measured), which is reachable via torch.matmul and still loses to shipped TF32/MXFP8 trailing. Closes lever #3** |
| 069 | | | | ✓ | | | | | ✓ | | replace `_exp062_factor`'s full-matrix `tril_()` (145us / 16.8% of `60×1024`, ~2.4× its bandwidth floor) with a **write-only `e62_zero_upper` CUDA kernel**; byte-identical L; grid **1.0136×** CI95[1.0131,1.0140] excludes 1.0; `60×1024` **1.0979×**, `8×2048` **1.0478×**, `2×4096` **1.0282×**, `2×2048`/`4×1024`/`16×512` 1.014–1.015×; 0 new fallbacks; ranked #926130 public+secret passed; test #926123 17/17 | **adopted (current best #926130); first clean `Ov` measurement — value-independent, carries to secret (exp-067 class)** |
| 071 | | | | ✓ | | | | | ✓ | | warp4 strict-upper mask: four shapes faster, led by `60×1024` **1.0341×**; six-e62 geomean **1.00676×** | frontier; rejected via 074 |
| 072 | | | | ✓ | | ✓ | | | ✓ | | 640×512 one-CTA launch fusion **0.748–0.851×** across three correct variants | **exhausted** |
| 073 | | | | | | | | | | ✓ | reusable N1 3× bitwise determinism + N2 guaranteed-SPD cond 1e6/1e8/1e10 stressgrid | harness complete |
| 074 | | | | ✓ | | | | | ✓ | ✓ | grid **1.00375×**; public 630.403→651.017us, secret 670.301→626.486us | **rejected at LB #926737** |

## Column totals — where effort has gone

| | Rt | Bk | Tr | CU | LP | Fu | Gr | Pe | Ov | Hn |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| experiments | 13 | 17 | 8 | 28 | 22 | 11 | 4 | 5 | **8** | 12 |

## What the matrix says

1. **`CU` and `LP` are saturated.** 20 and 21 experiments respectively. Exp 064's
   web research independently confirms the GEMM side is spent: MXFP8 block-column
   product measured at 3,466 TFLOP/s, FP16 panel apply at ~1,766 TFLOP/s.
   Another precision experiment is very unlikely to pay.

2. **`Ov` has one shipped win and one banked frontier (exps 069/071).** For a long time the lever with the
   largest modelled payoff (7–23% geomean, `levers.md` lever 7) had zero clean
   direct measurements — exp 031 refuted on a free gate, exp 027 regressed 0.336x,
   exps 017/061 only touched overhead as a side effect. Exp 069 landed the first
   clean one: replacing `_exp062_factor`'s inefficient full-matrix `tril_()` with
   a write-only strict-upper CUDA mask lifted the six e62 shapes (grid 1.0136×,
   `60×1024` 1.098×) with byte-identical output. The remaining `Ov` on those
   shapes is the `data.clone()` copy-in (82us on `60×1024`, an efficient memcpy).
   Exp 071's warp4 mask improved four shapes locally but was rejected through
   exp 074 when the public split regressed. Exp 072 tried `640×512`'s ~144us of
   inter-launch idle directly; serializing the work into one CTA lost 15–25%.

3. **`Pe` is 5-for-5 negative.** Persistent, cooperative, cluster, and DSM paths
   have never produced a shippable win (028, 038, 040, 048, 049; best 0.697x
   except exp 048's correctness-invalid 1.167x). Treat as closed unless something
   structural changes.

4. **The 2x wins are all one strategy on one shape class.** Exps 039/041/042/043
   each hit ~2.0–2.3x, all `CU`, all on `n<=256`. That is QR ladder step 9
   (fixed-shape specialization) applied only to small shapes. It has never been
   applied to the mid shapes, where `mid-shape-cusolver-headroom` says 19–260x
   above hardware floors remains.

5. **Diminishing returns are visible in the ranked column.** 2080 → 1084us took
   30 experiments; 1084 → 675.8us took 25 more. Recent adoptions are 1.007–1.04x
   grid wins. The board is asking for a new lever, not another increment on the
   large shapes.

7. **The resident diagonal-block kernel (`e62_diag128`) is occupancy-gated, and
   the gate is bidirectional.** Exps 066/067 are a matched pair on the same
   one-line lever (enroll a shape onto `_EXP062_SHAPES`): `60×1024` (batch 60)
   gained **1.24×**, `640×512` (batch 640) lost **1.65×**. It is a resident
   8-warp CTA-per-matrix kernel that only wins when the batch fits ~one wave of
   the ~148 SMs. Before enrolling any shape, check `batch ≤ ~148`; above that,
   keep the split32 + cuBLAS-trailing route. This also means the two shapes are
   individually exhausted for this lever — do not re-enroll either.

6. **Device-time wins do not automatically predict either leaderboard split.** Exp 065
   is the cleanest instance on the board: 1.149x on the kernel, 1.0122 on the
   full grid with CI95 excluding 1.0, identical counters, zero fallbacks,
   correctness bit-identical to the control — and the secret split still
   regressed 5.71% while public improved 3.79%. Exp 074 produced the inverse:
   public regressed 3.27% while secret improved 6.54%, despite bit-identical
   arithmetic and a 1.00375× paired grid. Four experiments now show split
   inversion (022, 035, 065, 074). Read `program.md`'s secret-split section
   before spending a ranked slot.

---

# Prose log


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
| 065 | named-barrier overlap in the 128×128 diagonal block: warp 0 builds `inv(L11)` while warps 1–7 run staging + the trailing update behind a `bar.sync` id 1 over 224 threads (exp 064 plan item 2). No arithmetic change — only which warp does existing work | block 45.669→39.742us = **1.149×** (356.8→310.5 ns/row); full grid **1.0122×** CI95 [1.0112,1.0133] 15/15; six-family baseline-attributed 48/48 `checker_ok`, 0 diffs; public 646.868us (**−3.79%**) but secret 692.860us (**+5.71%**) | #914341 | **rejected: secret split regressed; root stays on #913511** |
| 066 | enroll `640×512` (batch 640) onto the `e62_diag128` fused-block path (one-line `_EXP062_SHAPES` add, nb_outer=512) — the lever that won +1.13–1.17× on the low-batch split32 siblings 16×512/4×1024/8×2048 (exp 063) | paired 1242.16→2051.14us = **0.6056× (1.651× slower)**; full grid 725.21→750.02us; only the target moved (14 others within ±0.7%, no off-target regression); 60x1024 dense checker_ok, not fallback-contaminated | — | **rejected: `e62_diag128` is occupancy-gated. It is a resident 8-warp CTA-per-matrix kernel that only wins at ~1 wave (≤ ~148 matrices); batch 640 is ~4.3 waves of a per-CTA-latency-bound kernel** |
| 067 | enroll `60×1024` (batch 60) onto the `e62_diag128` fused-block path (one-line `_EXP062_SHAPES` add, nb_outer=1024) — the occupancy hypothesis that fell out of exp 066: batch 60 is on the winning side of the ~148 one-wave threshold, and 60×1024 was NOT previously enrolled (it ran the older exp043/047 fused-resident-panel + exp040 rank-4 diagonal-micro route) | same-process paired **1.2426×** CI95 [1.2411,1.2443] (1185.9→955.5us, −230us), all 14 other shapes within ±0.34% (0 new fallbacks; more accurate: dense residual 3.31 vs 9.33). evo frozen-baseline full-grid score misread it as a regression (725→751) purely from ~3.6% day-over-day B200 clock drift — the paired grid is the authoritative gate | #922201 (test #922196) | **LB submitted; public+secret runs both PASSED. Adoption DEFERRED — popcorn CLI does not expose the official public/secret geomean (Score `-`); per exp 065 precedent adoption needs the secret score. Root stays on #913511** |
| 068 | tcgen05 Blackwell GEMM built on **ThunderKittens 2.0** (educational_b200/level_06 class: tcgen05 + TMA, FP32 TMEM accumulate, generalized to M×N×K), compiled via `load_inline` on cuda13-devel + torch 2.13.0+cu130 at `sm_100a`; benchmarked vs `torch.matmul` bf16/tf32 on 4 large-n/trailing shapes | TK **293 TFLOPs @4096³** (= TK's published level_06 number), **0.19–0.28× cuBLAS bf16, 0.36–0.56× cuBLAS tf32**; correct (rel_err 0.2–0.3%). TK production ceiling (~1540) only ties cuBLAS bf16 (1330–1588 measured) which loses to shipped TF32/MXFP8 trailing | — | **rejected: the GEMM primitive is not the bottleneck (serial diagonal factorization is); a faster trailing GEMM cannot move the geomean. Toolchain recipe banked. Closes lever #3** |
| 069 | `Ov` lever (QR-ladder lever 7). Fresh incumbent shapediag showed `_exp062_factor`'s final `work.tril_()` costing 145us (16.8%) on `60×1024` — a full-matrix read+rewrite at ~2.4× its bandwidth floor. Replaced with a **write-only `e62_zero_upper` CUDA kernel** (one block per matrix-row; threads stride the strict-upper columns writing 0, never touching the lower triangle). Kept `data.clone()` copy-in so the factorization reads/arithmetic are unchanged → L byte-identical. Shared across the six e62 shapes | same-process paired full grid **1.0136×** CI95[1.0131,1.0140] excludes 1.0; `60×1024` **1.0979×** (936.1→852.6us), `8×2048` **1.0478×**, `2×4096` **1.0282×**, `2×2048` **1.0153×**, `4×1024` **1.0139×**, `16×512` **1.0062×**; nine other shapes flat (≤0.23% off-target, inside 0.57% A-vs-A); identical residuals + counters, 0 new fallbacks; six-family checker_ok on 512/1024/2048/4096; test #926123 17/17 | #926130 | **adopted (current best); first clean `Ov` win — value-independent byte-identical latency reorchestration, carries to secret (exp-067 class, not exp-065 precision risk)** |
| 070 | **public-LB stack.** Owner goal: improve the *public* board rank before the competition closed. Profiling (`results/070-largephase.json`, `070-nocusolver.json`) confirmed the large-shape diagonal `potrf` (52–65%) is a cuSOLVER wall with no custom kernel left in the source to build on → no tractable large-shape win. Pivoted to combining exp 065's **named-barrier overlap** (VAR=4 in the e62 128×128 diagonal block, our live LB entry `#914341`'s public win) with exp 067 + exp 069, which were already in root. Disjoint code paths → composes. Built by patching the 067+069 diff onto exp 065's `ship-v1.py` | same-process paired full grid vs `#926130` **1.0171×** CI95[1.0167,1.0176]; six e62 shapes `2×2048` **1.0507×**, `2×4096` **1.0474×**, `4×1024` **1.0455×**, `16×512`/`8×2048` **1.0393×**, `60×1024` **1.0369×**; nine others flat (≤0.06%, large shapes untouched); 0 new fallbacks, identical counters/accuracy; six-family 48/48 checker_ok; test #926455 17/17. **Official public 646.868→630.403us, board rank #32→#31** | #926462 | **adopted as ranked winner under owner's public-optimization rule; NOT secret-safe (inherits exp 065's +5.71% secret). `#926130` kept as secret-safe fallback** |
| 071 | bounded N=3 search over strict-upper mask launch geometry. Warp4 (four row-owned warps per 128-thread CTA) won the kernel probe; the v4 hybrid retained the incumbent row kernel for batch-2 n≥2048 | N1 3× bitwise and N2 18/18 per variant; family 48/48; four whole-shape CIs above parity: `16×512` 1.00577×, `4×1024` 1.00095×, `60×1024` 1.03406×, `8×2048` 1.00226×; six-e62 geomean 1.00676× | — (integrated in 074) | **promotable frontier; final LB rejection via 074** |
| 072 | `640×512` launch-overhead fusion: three race-free one-CTA-per-matrix variants after rejecting an initial multi-CTA data race on audit | V1/V2/V3 all N1/N2-correct but 0.8495× / 0.8509× / 0.7478×; ncu unavailable (`LibraryNotLoaded`) | — | **shape exhausted: serialization dominates launch saving** |
| 073 | reusable `stressgrid` harness for program2 N1/N2: three retained same-input outputs plus guaranteed-SPD tiny-diagonal, near-singular banded, and mixed-dynamic cases at cond exponents 6/8/10 | B200 smoke 3/3 bitwise + 9/9 adversarial; consumed by exps 071/072 without changing checker or timing paths | — | **harness complete** |
| 074 | integrated exp071 hybrid against exact `#926462`; full 15-shape paired, clean cold build, Popcorn test, then one ranked attempt | grid **1.003750×** CI95[1.002814,1.004687], four changed-shape CIs above parity, 15/15 and 0 new fallbacks; clean 57/57; test #926716 17/17. Ranked public **630.403→651.017us** (3.27% slower), secret **670.301→626.486us** (6.54% faster) | #926737 | **rejected under optimize-public rule; exact #926462 root restored** |
