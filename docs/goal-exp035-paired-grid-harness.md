# Goal: Experiment 035 — paired same-process 15-shape grid harness

## Problem

Every per-shape probe in this repo is paired (baseline and candidate measured
in one process, interleaved). The **full-grid** run is not: it times one
submission per sandbox, so candidate-vs-baseline comparison crosses process
and machine boundaries.

Measured consequence (exp 034): the byte-identical `60x1024` path produced
1565.6 / 1389.4 / 1989.7us across three grid runs — a **43% spread on code
that did not change**. The three grid geomeans (baseline-stale 1166.1,
baseline-today 1118.9, candidate 1183.5) were therefore uninformative, and a
verified 1.090x on `1x32768` (~+0.6% geomean) could not be promoted.

Leaderboard resubmission does not solve this: identical files vary 0.42%
public / 2.6% secret (exp 033). Both instruments are blind below ~2%.

## Objective

A `pairedgrid` mode that resolves **per-shape ratios to <=0.5%** and reports a
geomean-of-ratios with a confidence interval, in one sandbox, for all 15
shapes. Success = re-running it on a byte-identical pair yields every shape's
ratio within 1.000 +/- 0.005 and a CI that contains 1.0.

## Design

**Dual-module loading.** Load both sources in one process under distinct
module names via `importlib.util.spec_from_file_location` ("sub_base",
"sub_cand"), as the existing probes already do for `/root/baseline_*.py`.
Each module keeps its own globals, counters, Triton JIT cache entries, and
CUDA graph pools.

**Interleaving.** Per shape, per seed: `A B B A` blocks (order-reversal
cancels linear drift within a pair), R repeats, all on the same input tensor
allocated once. Never `AAAA...BBBB`.

**Warmup discipline.** Both modules fully warmed (JIT compile, autotune,
graph capture) on the target shape before any timed block, since first-call
cost differs by orders of magnitude and would otherwise land entirely on
whichever module runs first.

**Statistic.** Per shape, the paired ratio per repeat `t_base/t_cand`, then
median across repeats + bootstrap 95% CI. Aggregate = geomean of per-shape
medians, CI by bootstrap over shapes. Report **ratios**, never a ratio of
independently-measured geomeans.

**Timing.** `torch.cuda.synchronize()` + CUDA events per block; discard the
first repeat of each shape; report median and MAD, not mean (the 1989.7us
outlier above is a tail event, not a shift).

**Correctness plumbing (fixes the exp-034 gate bug).** Run the vendored
checker on both modules' outputs; record per-module fallback counters and
scaled residuals as *data*, not as a pass/fail gate. The current
`fallbacks == 0` gate fires on `_left_looking_large`'s finiteness handoff at
32768, which is pre-existing behavior present in baseline and candidate
alike — it made exp 034 report `passed: false` on a clean result. Gate on
checker pass + no *new* fallback relative to baseline.

## Deliverables

1. `run_pairedgrid(...)` in `scripts/_gpu_runner.py`; `pairedgrid` in
   `modal_verify.py` mode choices; `--candidate` arg to upload a second
   source alongside `--submission`.
2. **Null calibration run**: baseline vs a byte-identical copy of itself.
   This is the acceptance test — it establishes the harness's own noise
   floor, and no candidate verdict is trustworthy until it passes.
3. **Re-measure exp-034 V2** (`experiments/034-mxfp8-32768/candidate-v2.py`)
   against `cda77c7` on the calibrated harness. Promote to `submission.py`
   only if the paired geomean CI excludes 1.0 and no shape regresses beyond
   the null-run noise floor.

## Guardrails

- Free gates before GPU spend. Modal covered by program.md standing auth.
- Do not weaken the official checker; residual margins unchanged.
- No cuSOLVER in new paths, no stream APIs, no scanner evasion.
- Cost: the null run plus the V2 run are two sandbox sessions; shape-filter
  during development to keep spend near prior sessions (~$1-3).
- STOP LINE: local commit only. No push, no popcorn submission.

## Risks

- **Cross-module interference** (shared Triton cache dir, cuBLAS workspace,
  memory fragmentation at n=32768 where one fp32 matrix is 4GB). The null
  calibration run detects this: if A-vs-identical-A does not center on 1.000,
  the harness is measuring itself and must be fixed before use.
- **Order effects surviving A-B-B-A** — check by comparing first-half vs
  second-half ratios in the null run.
- If the null run cannot reach +/-0.5%, report that honestly as the
  achievable floor rather than tightening the statistic to fit; a harness
  that cannot resolve 0.6% means V2 stays unranked on its own merits.
