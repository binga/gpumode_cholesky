# Experiment 048 notes — `4x1024`

Status: **EXHAUSTED — no 2x winner and no six-family-clean frontier.**
Exact control snapshot:
`baseline-890659.py` (`59558b…d5e52`).

## Fresh B200 constituent profile

Artifact: `baseline-shapediag.json`.

| constituent | latency | calls | share of device |
|---|---:|---:|---:|
| wall | 674.6us | — | — |
| device kernels/copies | 664.1us | 102 | 98.4% of wall |
| wall minus device | 10.5us | — | 1.6% of wall |
| Triton diagonal micro | 411.73us | 32 | 62.0% |
| panel apply | 98.44us | 31 | 14.8% |
| panel inner update | 67.53us | 24 | 10.2% |
| trailing rank-128 update | 63.64us | 7 | 9.6% |
| D2D input/output copies | 8.90us | 2 | 1.3% |
| finite gate/reduction/D2H | about 12.9us | 5 | about 2.0% |

`scripts/latency_budget.py` classifies every modeled compute kernel as
residual/dependency limited: micro HBM/math floors 0.1/~0us, apply 2.1/0.3us,
inner 1.6/0.2us, trailing 7.6/4.2us.  Whole-operation useful Cholesky work is
1.432 GFLOP, about 3.18us at the repository's 450-TFLOP/s TF32 model; one
read plus one write of the 4x1024 matrices is 33.55MB, about 4.36us at
7.7TB/s.  Neither math nor HBM is the observed floor.

With 263us of measured non-micro work, the unchanged design gives the micro
only about 74us at the 337us 2x target.  A successful architecture must shorten
the serial dependency span, not merely make GEMM throughput higher.

## Measured ladder

All ratios below are paired same-process against `baseline-890659.py`.  Dense
checker residuals use the unchanged official checker.  Backend counters were
positive and no timed candidate introduced a fallback or runtime error.  Dense
correctness is only the first gate; the V2 changed-family result below is the
authoritative promotion verdict.

| V | architecture | baseline | candidate | ratio | verdict |
|---|---|---:|---:|---:|---|
| 1 | graph-capturable resident rank-128 panel | 677.368us | 687.228us | 0.985594x | REJECTED |
| 2 | 128-CTA cooperative tile-32 right-looking kernel | 719.712us | **616.768us** | **1.166791x** | REJECTED: lowrank NaN |
| 3 | V2 with reciprocal-multiply panel solve | 706.588us | 617.016us | 1.145283x | REJECTED/null vs V2 |
| 4 | V3 with rank-4 diagonal pivot groups | 706.128us | 768.696us | 0.918603x | REJECTED |
| 5 | four independent cluster16 kernels + DSM diagonal | 707.340us | 962.912us | 0.734453x | REJECTED |
| 6 | V5 with dual-warp concurrent panel ownership | 715.052us | 843.400us | 0.848006x | REJECTED |
| 7 | best V2 core with explicit FP16 trailing WMMA | 686.696us | 777.860us | 0.882847x | REJECTED |

### V1 diagnosis

`variant-01-shapediag.json` shows why halving launches did not help.  Device
work fell from 664.1us/102 launches to 393.8us/52 launches, and the resident
panel itself took only 29.96us.  But the graph dependency span exposed 326.1us
of idle time instead of 10.5us, so wall regressed to 719.9us.  The checker
residual remained exactly 9.25.

### V2 diagnosis

`variant-02-phase.json` records seven stable `%globaltimer` samples inside the
single cooperative kernel:

| phase | median | share of accounted 689.280us |
|---|---:|---:|
| diagonal + barriers | 197.120us | 28.6% |
| scalar panel solves + barriers | 286.240us | 41.5% |
| TF32 trailing WMMA + barriers | 201.856us | 29.3% |
| strict-upper cleanup | 4.064us | 0.6% |

V3 proved explicit reciprocal syntax is null; nvcc already schedules the
division equivalently.  V4 proved the standalone rank-4 micro mechanism does
not transfer: its register/shared pressure slows the persistent kernel.
V5 proved smaller cluster scope loses more trailing/panel parallelism than it
saves in barrier scope, despite a valid cluster backend and DSM broadcast.
V6 recovered 119.5us by solving both cluster-owned panel tiles concurrently,
but remained 226.6us behind V2, closing the cluster mechanism.  V7 directly
transferred the ranked path's six-family-safe FP16 trailing precision; its
residual improved to 1.49/20, but explicit FP32-to-half shared staging cost
more than the two fewer WMMA steps saved.

The serious architecture bound comprises the resident graph panel, whole-call
cooperative grid, rank-4 persistent diagonal, hardware cluster/DSM, dual-warp
cluster panel scheduling, and FP16 persistent trailing update.  The reciprocal
probe is a null instruction-level control, not a separate architecture.

No candidate reached the <=50% paired threshold.  V2 is 14.5--16.7% faster on
dense input than the exact ranked path depending on the same-process baseline
state, but its absolute 616.768us latency is still 1.83x above the
representative 337us 2x budget.  More importantly,
`variant-02-familygrid.json` fails closed: active 4x1024 passes dense (4.09),
spectrum (5.57), diagonal (0.00146), rowscale (1.13), and tridiagonal (0.105),
but the lowrank family produces NaN/Inf.  Thus V2 is diagnostic dense-only
evidence, not a valid frontier.  The batch60 rows all pass the checker but are
marked inactive by the target-specific overlay counter, as expected for
unchanged dispatch.

Therefore no full 15-shape, Popcorn, or leaderboard gate is authorized from
this task; root retains the exact ranked source.

Raw evidence: `variant-01-paired.json`, `variant-01-shapediag.json`,
`variant-02-paired.json`, `variant-02-phase.json`, `variant-03-paired.json`,
`variant-04-paired.json`, and `variant-05-paired.json`.
Additional evidence: `variant-06-paired.json`, `variant-07-paired.json`, and
`variant-02-familygrid.json`.
