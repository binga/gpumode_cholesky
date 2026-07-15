# Experiment 012 — large left-looking frontiers

Status: **ADOPTED — ranked winner `#878893`**

Baseline: ranked `#878273`, commit `4b4d557`, public `1500.704 us`, secret
`1501.440 us`.

## Integrated candidates

| shape | source frontier | prior paired evidence |
|---|---|---:|
| `1x16384` | left-looking TF32 active diagonal/panel updates | `18512.6 -> 15882.0 us` (`1.166x`) |
| `1x32768` | left-looking native FP8 panel updates, FP32 accumulation | `72535.4 -> 52349.6 us` (`1.386x`) |

`1x8192` remains on the ranked cuSOLVER path because its bounded search found no
valid improvement.

## Paired same-process B200 evidence

The probe rotates between two inputs, retains outputs until validation, confirms
the intended backend counters, and compares against the exact exp-009 source in
the same process.

| shape | exp 009 mean / best | candidate mean / best | speedup | residual |
|---|---:|---:|---:|---:|
| `1x16384` | 18495.512 / 18477.217 us | 16082.949 / 16008.816 us | **1.150007x** | 0.0965 / 20 |
| `1x32768` | 71567.591 / 71281.715 us | 52139.092 / 51606.770 us | **1.372628x** | 4.52 / 20 |

Backend evidence: 16 native 16384 hits, 12 native 32768 hits, zero large-path
fallbacks, and no FP8 error.

## Correctness and no-regression gates

- Local CPU property check: **10/10**.
- Python compilation, whitespace, source-policy, snapshot, and JSON checks: pass.
- Changed-region Modal sweep: **12/12** (dense, spectrum, lowrank, rowscale,
  diagonal, and tridiagonal at both sizes).
- Full 15-shape retained-output Modal benchmark: every shape passed; geomean
  **1574.881992 us**, versus exp 009's **1652.198636 us**. The 13 untouched
  dispatch regions remain on the exact exp-009 implementation.
- Popcorn test `#878891`: **17/17**.

## Ranked result

Exactly one ranked submission was launched. `#878893` succeeded at
**1459.321342997556 us public** and **1448.3768036226527 us secret**, improving
`#878273` by **2.7575% public** and **3.5342% secret**.

## Verdict

**ADOPTED.** The two positive large-shape frontiers were integrated; the
unimproved `1x8192` search was deliberately excluded. No evo workflow was used.
The repository owner's explicit Modal upload authorization is recorded in the
root README; uploads were limited to the candidate, baseline, runner, and
vendored reference harness.
