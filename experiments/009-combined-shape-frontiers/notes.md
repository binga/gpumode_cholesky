# Experiment 009 — combined shape frontiers

Status: **ADOPTED — ranked #878273**

## Hypothesis

Three independently measured exact-shape wins are composable because their
dispatch regions do not overlap:

| shape | candidate | paired B200 evidence before integration |
|---|---|---|
| `256x128` | vendor batched factorization via `make_graphed_callables` | `199.394 -> 169.425 us` (`1.1769x`), exact |
| `16x512` | static-buffer graph replay of vendor batched factorization | `764.992 -> 592.531 us` (`1.2911x`), exact |
| `8x2048` | Triton FP32 block factorization + lower TF32 Schur tiles | `5573.056 -> 3441.702 us` (`1.6193x`), tolerance fraction `0.102196` |

The static-buffer `16x512` graph is intentionally used instead of the faster
single-pointer frontier because the official harness rotates among several
input allocations. The static refresh remains valid without recapture.

## Final integrated evidence

- Local property gate: **10/10**; `py_compile`, `git diff --check`, snapshot
  comparison, and forbidden queue-source scan all clean.
- Rank-faithful paired Modal B200 probe retains outputs from 15 rotating inputs:

| shape | exp 008 | exp 009 | speedup | correctness |
|---|---:|---:|---:|---|
| `256x128` | 197.311 us | 162.913 us | **1.2111x** | exact |
| `16x512` | 772.362 us | 603.532 us | **1.2797x** | exact |
| `8x2048` | 5723.026 us | 3527.579 us | **1.6224x** | residual 2.04 / 20 |

- Target-family Modal verification: **25/25**. All six families pass at each
  integrated shape. The Triton path falls back to exact cuSOLVER when its
  diagonal contains a non-finite value; this was exercised by low-rank input.
- Final full 15-shape Modal geomean: **1652.1986356907646 us**, versus exp 008's
  **1738.120579869936 us**. No off-target shape regressed materially.
- Popcorn test `#878272`: **17/17**.

## Ranked workflow and graph-output fix

The first ranked attempt, `#878263`, failed benchmark validation. The cause was
reusable graph output storage: Popcorn retains outputs for several rotated
inputs and checks them afterward, so later graph replays overwrote earlier list
entries. Both graph paths now clone their reusable result before returning it.
The paired probe was strengthened to retain and check every rotated output, and
all local, family, full-grid, and Popcorn test gates were repeated before retry.

Ranked `#878273` completed successfully:

- public: **0.0015007037765896727 s = 1500.7037765896727 us**;
- secret: **0.0015014402012082579 s = 1501.4402012082579 us**;
- improvement over `#878108`: **2.736% public**, **2.827% secret**.

## Verdict

**ADOPTED.** Root `submission.py` is byte-identical to this experiment's final
snapshot. The raw Modal artifacts and ranked summary are preserved here.
