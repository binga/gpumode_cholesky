# Experiment 073 — program2 determinism and adversarial stress harness

## Purpose

Land the two verification primitives required by `program2.md` without changing
the official checker or any benchmark input/timing path:

- N1: call the candidate fast path three times on independent clones of the
  same input and require bitwise-identical retained outputs.
- N2: run exact ranked shapes at condition exponents 6, 8, and 10 over tiny
  diagonal, guaranteed-SPD near-singular banded, and mixed-dynamic-range SPD
  matrices, judged by the unchanged official reconstruction checker.

The new `stressgrid` mode is evidence-only. `pairedgrid` remains the latency
oracle and `familygrid` remains the standard six-family correctness gate.

## Smoke evidence

`results/073-stressgrid-smoke-v2.json` on B200 passed:

- determinism: `4096x32`, 3/3 outputs bitwise equal, active CUDA backend;
- adversarial: 9/9 checker passes for the three structures at 1e6/1e8/1e10.

The first smoke used the old low-rank generator at cond 1e8/1e10. In fp32 that
input was not reliably positive definite and the unchanged incumbent produced
NaN/Inf. It was replaced—not waived—with a guaranteed-SPD 2x2-block generator;
the corrected v2 smoke passed every row.

## Integrity

No changes to `reference/`, correctness thresholds, `kernel-audit.json`,
`pairedgrid`, benchmark workloads, or candidate source.
