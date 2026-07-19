# Experiment 039 result — `4096x32` reaches 2x and ranks

Status: **WINNER / ADOPTED.** Exact ranked source:
`ranked-888636.py`, SHA-256
`e6672b39a324a4d6247d803fdf4bf62422b7afb66d1aac09063a55e5990770d1`.

## Constituent diagnosis

The frozen ranked path was 43.18us in the revision-2 null calibration (40.0us
in the earlier single-module profile): one `_chol32_rank2_kernel` consumed
38.1us device time, wall-minus-device was only 2us, and compulsory 16MiB input
plus 16MiB output traffic was about 4.4us. The target was therefore the
predicated full-tile Triton state transformation, not host launch latency or
matrix arithmetic.

## Architecture ladder

| candidate | architecture | candidate us | paired result |
|---|---|---:|---:|
| V1 | register columns, one lane serializes the pivot column | 75.37 | 0.650x |
| V2 | one shared-memory row per lane | 37.51 | 1.156x |
| V3 | four-warps cooperative trailing update | 91.77 | 0.522x |
| V4 | two-sync shared rows with shuffle-broadcast rsqrt | 38.32 | 1.242x run ratio; slower absolute than V2 |
| V5 | register rows with shuffle-only exchange | 54.32 | 0.864x |
| V6 | register rows plus one shared pivot column | 22.55 | 1.915x |
| V6d | V6 with rank-2 fused trailing updates | **20.49** | **2.282x** |

The integrated standalone source reproduced at **43.29 -> 19.09us = 2.269x**.
Its active `_CUDA32_HITS=1` counter proves the new backend ran. The audit
measurement reports 43.18 -> 19.188us (**55.6% lower**) with zero regression
on the two untouched 4096 targets. The overall three-shape audit remains
`rejected` by design until the other two shapes also reach 2x.

## Promotion gates

- Changed region: 7/7 checks, covering dense, spectrum, diagonal, low-rank,
  row-scaled, and tridiagonal; worst recorded residual 0.0782/20.
- Full paired grid: 15/15 correct; target 2.244x; all other shapes at parity;
  aggregate geomean **1.05554x**, CI [1.05481, 1.05628].
- No auxiliary/concurrent queue API and no cuSOLVER call in the new fast path.
- Popcorn test `#888631`: 17/17.
- Ranked `#888636`: **992.551us public / 1003.332us secret**. Versus `#888352`,
  this is **5.704% public / 12.047% secret better**.

The winning mechanism is one warp per matrix, one register-resident row per
lane, padded shared-memory input/output staging, two shared pivot columns, and
rank-2 pivot processing that fuses two trailing rank-1 updates. The root source
retains the shipped Triton implementation only as a compile-failure fallback;
all timed evidence proves the CUDA path active.
