# Goal — Experiment 015: mid-shape batched tensor-core factorization

## Baseline

- Current ranked winner: `#880770` (exp 014, commit `48fa14a`), public geomean
  **1447.2589334363144μs**, secret **1443.2264907145392μs**. Exact source =
  root `submission.py`, SHA-256
  `78b2282d436243393897e61a5e4b8206d52c3950ec6f4495cbc71da895abd1fc`.
- Leaderboard 2026-07-17: rank 12. Leaders: seanyang 701.6μs, Voldemort4321
  732.2μs, Olek 758.4μs, xuan9938 760.7μs. Gap to #1 is **2.06×**.

## Hypothesis

The score is an equal-weight geomean over 15 shapes. Nine mid shapes still run
stock cuSOLVER (or launch-heavy multi-kernel Triton) and sit **19–260× above
B200 hardware floors** (memory ~6.5TB/s effective, TF32 ~0.9PFLOPS achievable):

| shape | exp014 Modal mean | floor | headroom |
|---|---:|---:|---:|
| 64×256 | 369.0μs | ~5μs | 71× |
| 16×512 | 624.0μs | ~5μs | 121× |
| 640×512 | 3942.8μs | ~207μs | 19× |
| 4×1024 | 1344.7μs | ~5μs | 260× |
| 60×1024 | 3226.8μs | ~77μs | 42× |
| 2×2048 | 1373.7μs | ~10μs | 133× |
| 8×2048 | 3525.4μs | ~41μs | 85× |
| 1×4096 | 1539.9μs | ~26μs | 61× |
| 2×4096 | 3215.6μs | ~51μs | 63× |

The S5 "640×512 saturated" verdict compared only cuSOLVER dispatch variants,
never custom tensor-core math; it is explicitly reopened. Each of these shapes
is worth a ~1.10–1.18× geomean factor at target, versus ~1.08× for a further
3× on 1×32768 — this region dominates the leader gap.

## Candidates (bounded ladder)

1. **A — fused one-CTA-per-matrix blocked Triton potrf**: one CTA factorizes
   one matrix; BK-block right-looking; serial in-register diagonal factor +
   in-register triangular inverse (BK serial steps each); panel = `tl.dot(P,
   Dinv^T)`; trailing = `tl.dot` TF32 with FP32 accumulate; out-of-place
   (reads A, writes L, zeroes upper during stores — no clone, no clear pass);
   `tl.debug_barrier()` between phases. Targets: 64×256, 16×512, 640×512,
   4×1024, 60×1024, 2×2048, 8×2048.
2. **B — graph-captured cuSOLVER-free superpanel** for 1×4096, 2×4096 (and
   graph capture for 1024×64): NB-superpanel factored by kernel A's one-CTA
   path, blocked triangular inverse, full-GPU TF32 `addmm_` trailing, entire
   loop captured/replayed as a CUDA graph. (Complements S13: that negative
   result replaced the diag potrf inside a TRSM-heavy structure at 32768; here
   the trailing runs on full-GPU GEMMs and TRSM is eliminated.)
3. Fallback variants if A/B underperform: BK/num_warps sweep, left-looking
   variant, 3xTF32 split or IEEE `tl.dot` if residuals fail at small n.

Per program.md: no cuSOLVER in new fast paths, no streams. Up to six distinct
serious variants before declaring exhaustion.

## Success thresholds and gates

- Default target per program.md: **≥2.00× paired speedup** per target shape vs
  the exact `#880770` path; correct FRONTIER results below 2.00× are preserved
  and eligible for integration if the aggregate geomean improves.
- Correctness: official checker semantics (lower-triangular, positive diagonal,
  finite, reconstruction ≤ 20·n·eps·‖A‖₁). Families dense/spectrum/diagonal/
  lowrank/rowscale/tridiagonal for every changed shape; safety fallback to
  exact cuSOLVER on non-finite diagonal is allowed and must be proven to fire
  only where intended.
- Timing evidence: paired same-process Modal B200 runs, rotating inputs as the
  official harness does (256MiB target), retained outputs, backend counters
  proving the fast path executed. Fallback timings are invalid.
- No off-target regression >1.03× on any unchanged shape in the full grid.

## Guardrails

- Modal budget: ≤ ~$15 this experiment; prefer ≤60s probe jobs on target
  shapes before any full grid.
- Popcorn: test mode 17/17 required first; then **exactly one** ranked
  submission, monitored to completion. No concurrent ranked runs.
- All artifacts under `experiments/015-mid-shape-tensorcore/`; journal +
  Optimization Tracker updated whether adopted or rejected; commit and push to
  GitHub is the terminal gate.
