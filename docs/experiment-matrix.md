# Experiment × strategy matrix

Every numbered experiment, the strategies it used, and the latency it moved.
Purpose: verify what has actually been tried and steer the next inner-loop pick.

Maintained alongside `experiments/README.md` (which holds the prose log) and the
`journal.md` Optimization Tracker (which is per *shape*, not per *experiment*).
**Add a row here whenever an experiment closes**, adopted or rejected.

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

## Column totals — where effort has gone

| | Rt | Bk | Tr | CU | LP | Fu | Gr | Pe | Ov | Hn |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| experiments | 11 | 17 | 8 | 20 | 21 | 9 | 4 | 5 | **4** | 9 |

## What the matrix says

1. **`CU` and `LP` are saturated.** 20 and 21 experiments respectively. Exp 064's
   web research independently confirms the GEMM side is spent: MXFP8 block-column
   product measured at 3,466 TFLOP/s, FP16 panel apply at ~1,766 TFLOP/s.
   Another precision experiment is very unlikely to pay.

2. **`Ov` has 4 entries and not one clean measurement.** Exp 031 was refuted on a
   free gate before any GPU spend (correct call — the premise was false). Exp 027
   *regressed* 0.336x. Exps 017 and 061 touched overhead only as a side effect of
   another change. So the lever with the largest modelled payoff — 7–23% geomean,
   see `lever-ladder.md` — has zero direct measurements. This is the clearest
   steer on the board.

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
