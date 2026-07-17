# Experiment 020 — panel-inner 64x64 subtiling

**Status: ADOPTED — ranked winner `#882927`.** Exact baseline is ranked
experiment 019 / submission `#882825` (`baseline-exp019.py`). The exact
ranked/adopted candidate SHA-256 is
`535813d6dcdb7589d43800dc49b2fc54de86a9a2aa4712112def52ec7ce80438`.

## Hypothesis

The shipped `_panel_inner32` produces a 128x128 output tile. Static compiler
evidence from experiment 019 reports 255 registers, a 408-byte stack, and 122
`LDL`/`STL` instructions for that specialization. Its measured self time is
156.6us (18.9% of end-to-end) on 4x1024 and 326.2us (13.6%) on 8x2048.

Variant 1 splits both output axes to 64x64 only for those two shapes, reducing
the accumulator surface by 4x while retaining the same rank-32 update and
arithmetic precision. It increases CTA count inside the existing CUDA Graph.

Falsifiable target: reduce panel-inner self time by at least 10% and improve
paired end-to-end mean on both shapes without changing correctness or causing
an unexpected fallback. Resource evidence should show fewer registers and/or
less stack/local-memory traffic than the exact exp-019 specialization.

## Guardrails

- Official reconstruction checker unchanged.
- Six input families on both target shapes.
- Alternating-order, same-process paired timing against the frozen source.
- No root submission change unless the candidate passes target, family, and
  later full-grid gates.
- Other dispatch shapes stay on the existing `_panel_inner32` kernel.

## Results

The instrumented probe passed 12/12 target-family checks and produced stable
paired gains:

| shape | baseline | candidate | speedup |
|---|---:|---:|---:|
| 4x1024 | 821.93us | 755.93us | 1.08731x |
| 8x2048 | 1954.47us | 1853.94us | 1.05422x |

The clean candidate repeated the result with another 12/12 family pass:

| shape | baseline | candidate | speedup |
|---|---:|---:|---:|
| 4x1024 | 820.73us | 753.89us | 1.08866x |
| 8x2048 | 1968.10us | 1865.20us | 1.05517x |

Panel-inner profiler time fell from the exp-019 measurements of 156.59us and
326.17us to 85.36us (**-45.5%**) and 216.12us (**-33.7%**) respectively. The
kernel launch count stayed at 24/48 per operation; the additional work is in
the grid dimensions inside each launch.

`cuobjdump --dump-resource-usage` on all four 64x64 specializations reports
**114 registers, zero stack, zero local allocation, and 1024 bytes shared**.
The corresponding shipped panel-inner specialization used **255 registers and
a 408-byte stack**. This confirms the intended compiler-resource movement.

## Full grid

The first 15-shape run passed but was retained only as noisy evidence:
1172.89us -> 1160.49us (**1.01068x**). The unchanged 60x1024 route had
7.6-16.6% CV, so it was not used as the primary aggregate claim.

The retry passed 15/15 at 1168.91us -> 1157.40us (**1.00995x**). Target gains
were 1.08958x and 1.05434x. The largest off-target mean regression was 0.254%
at 256x128. The unchanged 4096x32 baseline retained a repeatable first-sample
outlier (3.07% CV), so a formal contract with a 1% CV ceiling would still fail
closed even though its candidate samples were stable and the target evidence
was independently reproducible.

Artifacts: `subtile64-probe.json`, `artifacts-subtile64.zip`,
`subtile64-clean-repeat.json`, `subtile64-fullgrid.json`, and
`subtile64-fullgrid-r2.json`.

## Nsight Compute coverage

The B200 image exposes `ncu`, but the counter library cannot initialize inside
the Modal sandbox: `LibraryNotLoaded: Check that a compatible driver library
is loaded`. The packaging defect from the first attempt (missing vendored
`task`) was fixed; the second attempt reached NCU and failed at profiler
initialization. No NCU report was produced. Runtime counter coverage therefore
remains unavailable, while paired latency, torch-profiler kernel time, and
static cubin resources all agree on the improvement.

## Classification

`ADOPTED`: correct, stable on both changed shapes, resource movement confirmed,
positive on the full grid, and improved both completed leaderboard scores. The
repository still has no approved `kernel-audit.json`; the full-grid retry
exceeds the proposed CV ceiling on one unchanged workload, and NCU counter
collection failed at the provider boundary, so those audit coverage limits
remain recorded rather than being retroactively weakened.

## Popcorn and ranked result

The exact clean candidate was copied to the literal filename `submission.py`.
Popcorn test `#882926` passed **17/17**. Exactly one leaderboard job was then
launched: `#882927` passed public and secret test, benchmark, and leaderboard
stages at **1120.2139424233us public / 1126.4634299045994us secret**. This
improves ranked baseline `#882825` by **0.2099% public / 0.1815% secret**.

Artifacts: `test-882926.json`, `ranked-882927.json`, and the exact ranked source
`submission.py`. That source is adopted at repository root.
