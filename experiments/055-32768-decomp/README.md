# Experiment 055 — 32768 decomposition and inverse-base frontier

Status: **promotable frontier; Popcorn gates pending**.

## Profile

The exact `#890798` route at `1x32768` measured `42,543.3us` wall and
`41,349.7us` device time with only 2.8% idle. Its dominant constituents were:

- eight POTRF calls: `11,176.8us` (27.0%);
- 112 recursive-inverse TRSM calls: `9,000.2us` (21.8%);
- elementwise kernels: `5,194.6us` (12.6%);
- MXFP8 GEMM plus quantization: about 8.5%.

This made recursive inversion, not MXFP8 panel math, the selected axis.

## Variants and exact candidate

At `1x32768`, base 1024 measured `1.022515x`; base 2048 measured
`1.020832x`. The exact source therefore changes only the default recursive
inverse base from 512 to 1024. It also activates the earlier measured win at
`1x16384`.

Exact-source full grid: **`1.003722x`**, CI95
`[1.003149, 1.004295]`, 15/15 correct. The two affected rows were:

| Shape | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `1x16384` | 15115.1us | 14651.3us | `1.031913x` |
| `1x32768` | 42535.1us | 41650.3us | `1.021166x` |

All other shapes remain at parity. Exact candidate SHA-256:
`7593b3a3f5e79749d2cbc37f4093fdce44197a440755a0141e43f2909e46a93b`.

Both changed shapes passed all six official checker families. Spectrum and
low-rank at 16384, plus spectrum/low-rank/row-scale at 32768, take the same
pre-existing safety fallback as the exact incumbent; the candidate introduces
no new fallback. Cold import/extension build completed in `133.49s`, below the
`288s` promotion threshold, with all three CUDA extensions ready.

## Official compile defect and repair

Popcorn test `#897729` ended `passed=false` at exactly six minutes. The source
therefore was not ranked and the unchanged test is not eligible for retry.
The concrete repair combines the unchanged CUDA32, CUDA64, CUDA128, and
micro32 sources into one `-O3` translation unit, removing two fixed compiler
startups. This is the previously validated Exp051 packaging change only: no
memoization, pointer reuse, workspace cache, or last-call replay is present.
The compile-repaired exact source SHA-256 is
`ae8af22c9cae7e072136fa29a66a0d4f821619ed30d1c21e390a4802442c922a`;
its affected promotion gates must pass before the one permitted test retry.

Those gates passed: timed import is `27.73s`, B200 verification is `57/57`,
and the repaired exact-source grid is `1.003235x`, CI95
`[1.002521, 1.003948]`. The affected large rows remain `1.029606x` and
`1.020598x`; all 15 rows are correct.

Official retry `#897759` passed **17/17** in about 52 seconds. This supersedes
the compile-time failure `#897729` and qualifies the exact repaired source for
one ranked submission.

## Ranked verdict

Ranked `#897763` completed at **905.104689us public / 810.868840us secret**.
Relative to `#890798`, secret improved by about 4.36%, but public regressed by
about 12.86%. The program requires both splits to improve, so the candidate is
**not adopted** and the incumbent remains exact `#890798` / `f90ef90` /
`fd3072b…`. No duplicate ranked retry is permitted without a new candidate and
fresh gates.
