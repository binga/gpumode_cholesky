# Experiment 049 notes — `16x512`

Status: **PAUSED — external approval-review usage limit; V5 unmeasured**.

Exact current control snapshot: `baseline-890798.py`, SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
Experiment 047's paired full grid measured `389.408us`; strict 2x target:
`194.704us`.

The intended initial control, ranked `#890659` (SHA-256
`59558b501fb32d403667fd85a338ece7bb196f352a93685f7934bab8526d5e52`),
became stale when `#890798` ranked. A B200 `shapediag` job had already finished
when the stop directive arrived. Its `395.3us` wall / `371.3us` device result is
**STALE and diagnostic-only**; the raw JSON and stale source snapshot were
removed. No candidate was built or launched. No Modal job remains active.

## Fresh B200 constituent profile

Artifact: `baseline-shapediag.json`. Exact source SHA-256 was rechecked before
the run. The `batch=640, n=512` row is off-target and not used below.

| constituent | latency | calls | share of device |
|---|---:|---:|---:|
| wall | 389.6us | — | — |
| device kernels/copies | 361.2us | 53 | 92.7% of wall |
| wall minus device (launch/dependency idle) | 28.5us | — | 7.3% of wall |
| Triton diagonal micro | 207.03us | 16 | 57.3% |
| panel apply | 60.17us | 15 | 16.7% |
| panel inner update | 44.94us | 12 | 12.4% |
| trailing matmul | 26.75us | 3 | 7.4% |
| input D2D copy | 7.55us | 1 | 2.1% |
| finite/cleanup/reduction/D2H gates | 14.76us | 6 | 4.1% |

The strict `194.704us` target is below the diagonal micro alone. Keeping all
other measured work unchanged would consume `182.67us` (`154.17us` non-micro
device work plus `28.5us` idle) and leave only `12.03us` for all 16 diagonal
steps. Therefore a drop-in micro replacement cannot reach 2x; the launch chain,
panel work, and diagonal dependency span must be collapsed together.

Hardware floors are not the limiter. The complete batch performs about
`0.716 GFLOP` of Cholesky arithmetic, approximately `1.59us` at the
repository's `450 TFLOP/s` TF32 model. The `16x512x512` input is `16.78 MB`;
one whole-matrix read plus write is `33.55 MB`, approximately `4.36us` at
`7.7 TB/s`. `scripts/latency_budget.py` likewise classifies micro, apply,
inner, and trailing kernels as residual/dependency limited: their modeled HBM
floors are `0.3/2.0/1.6/3.3us`, and modeled math floors are
`~0/0.3/0.2/1.8us`, versus `207.03/60.17/44.94/26.75us` measured.

## First architecture proposal (not yet built)

Use one nonportable **16-CTA Blackwell cluster per matrix**, giving 16
independent clusters for batch 16. Each CTA owns a `32x512` row tile in 64 KiB
shared memory, so the cluster's DSM collectively holds the full matrix. Run 16
blocked Cholesky steps in one persistent kernel: the pivot CTA uses the proven
register rank-4/`rsqrt` 32x32 factorization, active row CTAs solve their panel
tiles from the DSM diagonal, and then update their owned lower tiles from DSM.
Use accuracy-preserving TF32x3 (three TF32 MMA terms with FP32 accumulation),
not the already-rejected plain-TF32 path. The wrapper contributes one D2D clone
and one kernel launch; no stream or auxiliary queue is named.

This differs materially from Experiment 048's cluster candidate, which kept
only the 32x32 diagonal in DSM and repeatedly moved panel/trailing tiles through
global memory. Here row tiles remain resident across all 16 steps. Record
`%globaltimer` deltas for diagonal+cluster barriers, panel solve, TF32x3 trailing
update, initial load/final store, and cleanup. Positive hit/ready/error/fallback
counters must prove cluster execution before any timing is accepted.

## Measured architecture ladder

Every result must use paired same-process measurement against the exact
snapshot, retain outputs through the unchanged checker, and prove the intended
backend with positive counters and zero errors/fallbacks.

| V | architecture | baseline | candidate | ratio | verdict |
|---|---|---:|---:|---:|---|
| 1a | full-resident cluster16/DSM + TF32x3 | 403.368us | 404.888us | 0.99631x | **INVALID: compile/load fallback** |
| 1b | V1 after mechanical TF32 rounding fix | 398.464us | 953.344us | 0.41787x | **REJECTED: 2.39x slower** |
| 2 | one persistent 512-thread CTA/matrix | 387.800us | 1296.976us | 0.29898x | **REJECTED: 3.34x slower** |
| 3 | occupancy-gated per-matrix atomic CTA groups | 399.576us | 572.896us | 0.69735x | **REJECTED: 1.43x slower** |
| 4 | four atomic rank-128 superpanels | 386.776us | 2080.584us | 0.18587x | **REJECTED: 5.38x slower** |
| 5 | graph-captured Triton resident-panel fusion | — | — | — | **UNMEASURED: B200 gate denied** |

`variant-01-paired.json` is not candidate timing. The extension was unavailable,
`_COOP512_FALLBACKS` incremented once, and there were no positive
`_COOP512_HITS`, readiness, or phase counters. Consequently both timed sides
executed the exact ranked path; the unchanged official checker reported the
same `2.59` residual. The paired harness did not serialize `_COOP512_ERROR`, so
the precise compile/load exception is missing from this artifact. No phase
split exists and V1 is rejected fail-closed until a diagnostic build exposes
the exception.

Diagnostic source `candidate-v1-error.py` reran import with verbose nvcc and
failed before timing at CUDA line 153:
`error: identifier "__float_to_tf32" is undefined`. This is a mechanical
intrinsic-availability issue, not cluster16 or dynamic-shared infeasibility.
V1 replaces it with explicit round-to-nearest-even TF32 mantissa truncation
(`13` discarded FP32 mantissa bits) and proceeds to the approved paired retry.

The valid retry (`variant-01b-paired.json`) proves the intended backend:
`_COOP512_HITS=1`, `_COOP512_READY_HITS=1`, zero new fallbacks, no module
error, and unchanged off-target `640x512` (`1.00112x`, CI crosses one). The
official dense checker passes at residual `9.45/20` versus baseline `2.59/20`.
The target ratio is stably bad: `0.41787x`, 95% bootstrap CI
`[0.41730, 0.41904]`, MAD `0.051%`, same-process A-vs-A spread `0.47%`.

One-shot `%globaltimer` medians across the 16 matrix clusters:

| V1 phase | latency | share of 796.064us accounted |
|---|---:|---:|
| DSM initial load + barrier | 9.936us | 1.2% |
| diagonal + cluster barriers | 90.288us | 11.3% |
| panel solve + publication | 201.024us | 25.3% |
| TF32x3 trailing + publication | 504.000us | 63.3% |
| final lower store/cleanup | 2.688us | 0.3% |

Candidate wall exceeds the accounted median by about `157us`; cluster16
scheduling puts 16 clusters (256 CTAs) through more than one hardware residency
wave. The dominant actionable gap is not HBM load/store but inter-CTA DSM MMA,
cluster-barrier, and panel serialization. Full-matrix cluster residency is
closed for this shape.

Proposed V2 is materially different: **one persistent 512-thread CTA per
matrix**, 16 CTAs total, with only the current `512x32` panel and its TF32
residual resident in 128 KiB shared memory. Warp 0 factors each diagonal;
all threads solve one panel row each; 16 warps distribute the lower trailing
16x16 MMA quarters. This removes cluster/DSM communication and cluster-wide
barriers while preserving one launch, TF32x3/FP32 accumulation, and FP32
pivots. Global trailing traffic returns, but its modeled multi-pass HBM cost is
far below V1's measured `504us` trailing phase.

V2 (`variant-02-paired.json`) also proves its intended backend with
`_CTA512_HITS=1`, `_CTA512_READY_HITS=1`, zero new fallbacks/errors, and an
unchanged off-target ratio of `0.99986x`. Dense correctness again passes at
`9.45/20`, but latency regresses to `1296.976us` with a tight ratio CI of
`[0.29892, 0.29915]` and only `0.08%` A-vs-A spread.

| V2 phase | latency | share of 1281.936us accounted |
|---|---:|---:|
| diagonal global loads | 1.680us | 0.1% |
| FP32 diagonal + CTA barriers | 99.248us | 7.7% |
| scalar panel solve + CTA barriers | 307.472us | 24.0% |
| TF32x3 trailing update + CTA barriers | 857.264us | 66.9% |
| upper cleanup/store | 12.992us | 1.0% |

Removing DSM did not offset using only 16 of B200's SMs. The same panel and
trailing phases became `1.53x` and `1.70x` slower than V1; under-parallelized
single-CTA residency is closed. A plausible materially distinct V3 is a
software-synchronized 16-CTA group per matrix using global panels/workspace:
all 256 small CTAs can be proven simultaneously resident with the CUDA
occupancy API, avoiding both cluster16 multi-wave placement and whole-grid
barriers. Per-matrix atomic phase barriers would fail before launch unless
resident capacity covers all 256 CTAs.

V3 (`variant-03-paired.json`) passes the hard occupancy gate with capacity for
`740` resident CTAs versus `256` required. Active hit/readiness counters are
positive, new fallbacks/errors are zero, dense residual is again `9.45/20`,
and `640x512` remains parity (`0.99952x`, CI crosses one). It is the best new
architecture so far but still regresses to `572.896us`: ratio `0.69735x`, CI
`[0.69643, 0.69832]`, A-vs-A spread `0.06%`.

The median per-matrix critical span is `541.520us`. Per-phase maxima are
load `1.536us`, diagonal `99.296us`, panel `191.456us`, trailing `106.560us`,
cleanup `1.376us`, and atomic-barrier wait `408.896us`. These maxima overlap
and are drawn from different CTAs, so they are not additive; the result still
unambiguously identifies 49 per-matrix barriers as the dominant gap.

Proposed V4 changes the factorization hierarchy rather than tuning V3: four
**rank-128 superpanels** instead of sixteen rank-32 panels. CTA 0 factors a
128x128 FP32 diagonal tile locally in about 66 KiB shared memory, active CTAs
solve 32x128 panels, and TF32x3 trailing MMA uses K=128. This reduces software
barriers from 49 to 13 and fattens tensor-core work. The exact compiled-resource
occupancy gate remains mandatory because every CTA inherits the diagonal
CTA's shared-memory allocation.

Fail-fast bound before V4 construction: Experiment 044 measured its block-128,
BK16 diagonal factor at `47.36us/block`; four blocks are `189.4us` including
four approximately `3.5us` launch floors, or still about `175.4us` with those
floors removed. Adding the current path's roughly `22.3us` copy/gate work gives
`197.7us` before V4 panel, trailing, or barrier work—already above the strict
`194.704us` target. V4 is therefore a barrier/fat-MMA frontier probe, not a
credible 2x path unless its integrated diagonal factor materially beats that
measured bound.

V4 (`variant-04-paired.json`) passes its exact occupancy gate at `444` resident
CTAs versus `256` required, has positive active/readiness counters, zero
fallback/error, and leaves `640x512` unchanged (`0.99886x`). Dense residual is
`8.17/20`: officially valid but already above the preferred `8/20` ship
margin. Latency is `2080.584us`, ratio `0.18587x` with CI
`[0.18574, 0.18603]` and `0.09%` A-vs-A spread.

The critical matrix span is `1986.064us`. Phase maxima are global diagonal
loads `45.328us`, FP32 diagonal `238.464us`, scalar panel `1552.880us`,
TF32x3 trailing `90.912us`, cleanup `1.344us`, and barrier wait
`1893.392us`. Maxima overlap across CTAs. V4's integrated diagonal is
`283.8us`, worse than the pre-build `175.4us` launch-free bound, and the
32x128 scalar triangular solve becomes the decisive straggler. Fewer barriers
cannot help when each barrier waits for a millisecond panel; the rank-128
hierarchy is closed.

Proposed V5 returns to the exact ranked graph but enrolls only `16x512` in the
already-shipped Triton `_panel_fused128` path. It is materially distinct from
the four custom persistent designs: graph-captured resident **panel fusion**
loads each below-diagonal `TILE_R x 128` block-column tile once and performs
the four TF32x3 substeps in registers, preserving the ranked diagonal micro,
graph replay, and accuracy. Experiment 047 validated this mechanism and its
families at other shapes, but did not enroll or measure `16x512`.

V5 is materialized as the minimal 33-line `candidate-v5.py` overlay. Root
independently completed only the safe local gates: `py_compile` PASS, exact
baseline hash PASS, `git diff --check` PASS, and prohibited-source policy scan
PASS. No B200 correctness, route-counter, latency, phase, or family evidence
exists. The attempted local command in this task was rejected before execution
with: `Automatic approval review failed: You've hit your usage limit... try
again at Jul 27th, 2026 2:10 PM.` No retry, escalation, Modal job, V6, Popcorn
gate, root-source edit, or leaderboard action followed. Therefore V5 remains
**UNMEASURED**, Experiment 049 is **PAUSED rather than exhausted**, and exact
ranked `#890798` remains authoritative.

Final packaging gates parsed every Python and JSON artifact, rechecked the
three exact incumbent hashes, passed `git diff --check`, and found no conflict
markers. The optional local property command could not run because this host
environment lacks `torch` (`ModuleNotFoundError`); no result is claimed from
that omitted gate. The stronger official checker did run on every measured V1–V4
paired row as recorded above.
