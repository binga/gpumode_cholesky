# Experiment 010 — Blackwell `1x16384`

Status: **REJECTED — bounded architectural exhaustion; nothing submitted**

Baseline: ranked `#878273`, commit `4b4d557`, exact source in
`baseline-exp009.py`. The target path is inherited unchanged from exp008.

Strict paired threshold: current-path mean `18591.1 us`; candidate must measure
`<= 9295.6 us` in the same B200 process and pass all promotion gates.

## Component profile

The warmed device-event profile explains why Schur substitutions could not
reach the hard target:

| shipped component | mean us | share |
|---|---:|---:|
| FP32 panel TRSM | 7,235.2 | 38.8% |
| serial FP32 diagonal POTRF | 5,454.8 | 29.3% |
| fused full TF32 Schur updates | 3,255.6 | 17.5% |
| finite check | 976.6 | 5.2% |
| triangularization | 582.0 | 3.1% |
| panel/diagonal stores | 746.2 | 4.0% |
| input clone | 389.6 | 2.1% |
| **total** | **18,640.0** | **100%** |

Raw per-step events are in `shipped-component-profile.json`. One cold
initialization pass is excluded from the reported means.

## Candidate ledger

| ID | Architecture | engaged | baseline -> candidate us | speedup | tolerance fraction / margin | Verdict |
|---|---|---:|---:|---:|---:|---|
| v1 | lower-only tiled TF32 Schur | Triton | 18,724.6 -> 40,445.0 | 0.463x | 0.011646 / 85.9x | reject: generic kernel far below cuBLAS collective |
| v2 | cuBLAS TF32 SYRK lower-only control | extension loaded | 18,519.0 -> 48,929.2 | 0.378x | 0.0000269 / 37,129x | reject: SYRK did not use a competitive TF32 path |
| v3/v4 | scaled FP8 E4M3 Schur, full/lower formulations | native FP8 | 18,493.4 -> 21,790.2; 18,496.2 -> 21,412.3 | 0.849x; 0.864x | 0.551379 / 1.81x | reject: cast/scaling plus fixed panel work dominate |
| v5 | hierarchical diagonal + lower-only TF32 | Triton | 18,485.0 -> 48,100.4 | 0.384x | 0.011585 / 86.3x | reject: diagonal hierarchy inherits slow generic update |
| v6 | graph replay around v1 | graph captured | 18,480.1 -> 40,543.7 | 0.456x | 0.011646 / 85.9x | control only: replay cannot rescue slower arithmetic |
| v7 | hierarchical tensor-core diagonal + panel solve | native TF32 ops | 18,436.2 -> 24,277.6 | 0.759x | 0.004826 / 207.2x | reject: small solves/materialization overwhelm GEMM savings |
| v8 | compact triangular inverse + TF32 panel GEMM | native TF32 ops | 18,534.4 -> 17,275.7 | **1.073x** | 0.011342 / 88.2x | reject: real win, far short of 2x |
| v9 | left-looking active diagonal/panel updates | native TF32 ops | 18,512.6 -> **15,882.0** | **1.166x** | 0.004823 / 207.3x | best, but reject: far short of 2x |

Architecture counting follows the supervisor checkpoint: v2 is a library
control, v3/v4 count as one FP8 axis, and v6 does not count because it wraps
arithmetic already slower than shipped. The six serious measured axes are:

1. lower-triangle-only custom TF32;
2. FP8/MX-style scaled Schur arithmetic;
3. hierarchical diagonal factorization;
4. hierarchical diagonal plus panel solve;
5. inverse-panel tensor-core GEMM;
6. left-looking factorization.

All candidates directly executed their intended backend; exceptions were
fail-closed and no timing row silently fell back. Every dense output and every
retained output passed the reference checker. Because no candidate crossed the
`2.00x` dense paired gate, the expensive six-family, full-grid, and Popcorn test
promotion gates were correctly not launched.

## Verdict

**REJECTED.** The best serious candidate is v9 at `1.166x`, still `6,586.4 us`
above the `9,295.6 us` threshold. Root `submission.py` remains byte-identical to
ranked `#878273` and to `baseline-exp009.py` (SHA-256
`39261a153a0df9826b0e6c8aa1b3f948179f6445da8be2a74a8ab5040ab7adf8`). No
Popcorn test or leaderboard submission was used.

Raw artifacts: `paired-v1.json`, `paired-v2-v4.json`, `paired-v5-v6.json`,
`paired-v7.json`, `paired-v8-v9.json`, and `shipped-component-profile.json`.

Modal usage was bounded to seven paired/profile B200 sandboxes (including the
corrected warmed profile), roughly 4–6 minutes of aggregate GPU wall time. Ranked
quota used: zero. The self-contained environment did not expose CUTLASS headers,
so no unshippable dynamic dependency was presented as a CUTLASS/tcgen05 result;
that high-effort fused collective remains distinct from the six measured axes.
