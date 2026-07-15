# Experiment 008 — fused in-place TF32 Schur update

**Status: ADOPTED — ranked `#878108`, new current best.**

Public geomean: **1542.9137409531085μs**. Secret geomean:
**1545.1284990962687μs**. This beats experiment 006 / ranked `#878015`
(approximately 1559μs). Popcorn test `#878107` and ranked `#878108` both passed
17/17.

## Hypothesis and isolated change

Experiment 006 evaluated `A22 -= L21 @ L21.T` at every blocked step. That
materialized a dense product and launched a second subtraction. Stage A changed
only this expression to an in-place fused update:

```python
A22.addmm_(L21, L21.transpose(-1, -2), beta=1.0, alpha=-1.0)
```

The target remains the same strided trailing view; TF32 and FP32 accumulation,
blocking, panel factorization/solve, final `tril`, finite check, and numerical
fallback are unchanged.

## Paired Modal B200 probe

Both variants ran in the same process with identical data, blocking, warmup,
L2 clearing, checker, and result construction.

| shape | block | separate TF32 | fused `addmm_` | speedup | residual/tolerance |
|---|---:|---:|---:|---:|---:|
| 1×16384 | 2048 | 18924.8μs | **17411.5μs** | **1.087×** | identical 0.004796 |
| 1×32768 | 4096 | 73700.7μs | **68246.1μs** | **1.080×** | identical 0.002397 |

Artifacts: `stage-a-16384.json`, `stage-a-32768.json`.

## Correctness

- CPU property check: **10/10**.
- Modal B200 at every changed size: **12/12**, covering dense, spectrum,
  lowrank, rowscale, diagonal, and tridiagonal at both 16384 and 32768.
- Dense margins were 208.5× (16384) and 417.1× (32768) inside tolerance.
- Spectrum and lowrank still trigger the established finite-value fallback and
  return exact FP32 cuSOLVER factors; stable families retain the fused fast path.
- Popcorn test `#878107`: **17/17**.
- Popcorn ranked `#878108`: **17/17**.

## Full 15-shape Modal benchmark

Geomean: **1738.120579869936μs**. The absolute total is affected by ordinary
drift on untouched cuSOLVER shapes; only two dispatch shapes changed in source:

| shape | exp006 Modal | exp008 Modal | delta |
|---|---:|---:|---:|
| 1×16384 | 19981.6μs | **18531.4μs** | −7.3% |
| 1×32768 | 78357.1μs | **73463.5μs** | −6.2% |

The paired probes above are the promotion evidence because they isolate the
algorithmic delta from session drift. The resulting ranked score improved as
projected.

## Ranked result

- Test: `#878107`, 17/17.
- Leaderboard: `#878108`, 17/17.
- Public: **0.0015429137409531085 s = 1542.9137409531085μs**.
- Secret: **0.0015451284990962687 s = 1545.1284990962687μs**.
- Exactly one leaderboard submission was launched.

## Ladder verdict

**Stage A won and was adopted.** Stages B (custom lower-triangular tensor-core
update), C (hierarchical blocking), and D (bounded batched pivot) were not needed
because Stage A already produced a confirmed ranked improvement. BF16x9,
FP16/BF16, queue-based scheduling, and prior loop/chunk approaches were not
repeated. Adjacent finite-scan/final-triangle cleanups were deliberately not
bundled into the measured Stage-A change.

## Modal spend

Four B200 jobs (two paired probes, changed-size family verification, one full
grid) used roughly 2–3 minutes of GPU wall time, approximately <$0.5–1 depending
on Modal billing. No sandbox stalled.

## Verdict

**ADOPTED.** Root `submission.py` carries the fused update. The exact submitted
source is preserved as this directory's `submission.py`.
