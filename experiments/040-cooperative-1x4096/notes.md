# Experiment 040 result — `1x4096` boundedly exhausted

Status: **REJECTED / EXHAUSTED.** The exact ranked source remains `#888636`;
root `submission.py` was not changed and no Popcorn submission was made.

## Frozen target and vendor constituents

The revision-3 byte-identical baseline measured **1528.456us**, so the 2x
threshold was **764.228us**. The ranked path is almost entirely one vendor
factorization: 1393.0us (91.1%), plus 74.6us output staging, 57.4us lower
cleanup, and about 4.6us setup. Wall-minus-device time is effectively zero.

The cooperative-barrier gate passed on B200. For 148 resident CTAs, 192
rendezvous cost 231.586us and 256 cost 307.611us; a 296-CTA grid was also
legal. Synchronization alone therefore did not rule out the architecture.

## Correct architecture ladder

Every timed candidate used the active `_COOP4096_HITS=1` backend, passed the
dense checker, left `2x4096` at parity, used no cuSOLVER in the new path, and
used no auxiliary/concurrent CUDA queue API.

| variant | architecture | baseline us | candidate us | speed ratio |
|---|---|---:|---:|---:|
| V1 | tile-32 cooperative right-looking, FP32 panel solve, TF32 trailing | 1530.57 | 4112.25 | 0.372x |
| V2 | tile-64, 256 threads, eight resident MMA warps | 1537.97 | 6944.30 | 0.221x |
| V3 | tile-32 diagonal inverse plus tensor-core panel application | 1533.91 | 4804.90 | 0.319x |
| V4 | occupancy-query / residency-saturated tile-32 grid | 1530.73 | **4066.43** | **0.376x** |
| V5 | left-looking tiles, eliminating the explicit trailing phase | 1526.19 | 18040.50 | 0.085x |
| V6 | four-tile superpanels with consolidated rank-128 updates | 1529.79 | 4838.74 | 0.316x |

V2 first exposed the 512-thread cooperative residency limit; the measured
candidate is the corrected 256-thread form. All six final variants were
correct. Residuals were 1.01--2.46 against an allowed 20.

## In-kernel constituent profile

The instrumented V1 used B200 `%globaltimer` timestamps inside the single
cooperative launch. Seven samples were stable and the medians account for
4019.648us of the roughly 4.1ms wall time:

| constituent | median us | share |
|---|---:|---:|
| diagonal factor phases | 837.120 | 20.8% |
| panel solve phases | 1016.960 | 25.3% |
| trailing tensor-core updates | 2142.080 | 53.3% |
| strict-upper cleanup | 23.488 | 0.6% |

This profile explains the ladder. A wider tile lengthened the serial panel
dependency and spilled state; tensor-core inverse application did not repay
inverse construction; extra residency did not change throughput; left-looking
products collapsed parallelism as their depth grew; and rank-128 consolidation
saved output traffic but lost efficiency on longer MMA chains.

## Verdict

The best cuSOLVER-free result is **2.657x slower** than the ranked path and
**5.321x slower** than the required threshold. The six-architecture bound is
exhausted with direct component evidence, so transferring this mechanism to
`2x4096` would repeat a measured failure. The campaign replaces the two
remaining picks with the more tractable `1024x64` and `256x128` shapes, while
retaining `4096x32` as the first achieved 2x win.
