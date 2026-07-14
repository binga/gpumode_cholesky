# Optimization goal — beat the cholesky baseline

**Owner:** autonomous optimization thread. **Supervisor:** main session (babysits).

## Objective
Improve our ranked **geometric-mean** runtime on the GPU MODE `cholesky`
leaderboard (776, B200) below the current cuSOLVER baseline (**~2080μs**, ranked
submission `#876988`), and ideally below the board leaders (~1924μs). Do it by
replacing the cuSOLVER call in `custom_kernel` with custom kernels for the
shapes that have the most headroom — **without ever regressing correctness**.

## Why these shapes (from the baseline benchmark)
Because ranking is a *geometric* mean, a speedup on any one shape helps as much
as the same-factor speedup on any other. Target by headroom, not absolute μs:

- **High ROI (do first):** `4096×32` (113μs), `1024×64` (110μs), `256×128` (152μs).
  Overhead-bound — a 32×32 factorization is trivial, so the time is launch/dispatch
  overhead across many tiny matrices. Custom batched kernel (one program per matrix)
  is the known win; see `reference`-style `triton_cholesky32.py`.
- **Medium ROI:** `640×512`, `8×2048`, `2×4096` (occupancy/batch-parallel tuning).
- **Low ROI (leave on cuSOLVER):** `n ≥ 8192`, esp. `32768²` (compute-bound).

Keep `custom_kernel` as a `(batch, n)` dispatcher: custom kernels for tuned
buckets, `torch.linalg.cholesky_ex` fallback everywhere else.

## The loop
**Inner (cheap, iterate here):**
1. Edit `submission.py`.
2. `uv run --with torch --with numpy python scripts/verify_local.py` (CPU, free).
3. `uv run --with modal python scripts/modal_verify.py verify` (B200 correctness, ~40s).
4. `uv run --with modal python scripts/modal_verify.py benchmark` (B200 per-shape μs);
   compare geomean to `results/baseline-benchmark.json`.
5. Repeat until a target shape is convincingly faster **and** still passes.

**Outer (scored, rare):** only when Modal benchmark shows a real geomean win →
`popcorn submit --mode test --no-tui submission.py` (expect 17/17) →
`popcorn submit --mode leaderboard --no-tui submission.py` → record in `journal.md`.

## Guardrails (hard rules)
- **Correctness is non-negotiable.** Every candidate must pass `check_implementation`
  across all test families (dense/diagonal/spectrum/lowrank/rowscale/tridiagonal) on
  Modal B200 before any leaderboard submission. A faster-but-wrong kernel is a failure.
- **Cost discipline.** Modal B200 is billed per second. Prefer benchmarking only the
  shapes under active work (add a shape filter to `_gpu_runner.py` if helpful); run the
  full 15-shape grid (incl. the 221ms 32768² case) only immediately before a ranked submit.
- **Quota discipline.** At most **3 ranked `--mode leaderboard` submissions** this run.
  Never submit unless Modal already showed a geomean strictly better than the last
  submitted result. No blind submissions to "see what happens".
- **Timing fidelity.** Modal timing omits popcorn's L2-cache clear, so treat Modal numbers
  as a *relative* signal and the ranked result as *absolute* truth. Tightening
  `_gpu_runner.py` to clear L2 + adaptive iters (mirroring popcorn) is allowed and encouraged.
- **Numerical headroom, carefully.** The checker tolerates `20·n·eps·‖A‖₁`; trading a little
  accuracy for speed (e.g., TF32 intermediates) is fine **only if** the FP32 reconstruction
  gate still passes on Modal across all families.
- **No scope creep.** Do not change the evaluation harness, `reference/`, the plan's
  product intent, or git config. Do not commit unless the supervisor asks.

## Definition of done
- A ranked submission whose geomean is **strictly better than ~2080μs**, confirmed via
  `popcorn submissions list --leaderboard cholesky`, with a new `journal.md` entry
  (submission id, geomean, per-shape deltas, what worked).
- OR, if no correct improvement is found within the quota/cost budget: a `journal.md`
  entry documenting what was tried, the measured results, and why it didn't beat baseline
  — enough for the next session to continue.

## Report back (to supervisor)
End with: best geomean achieved, ranked submission id(s), which shapes improved and by
how much, correctness status, remaining ideas, and approx Modal spend if known.
