# Goal — exp 004: speed up small-batch / large-n (starting with 2×4096)

**Owner:** autonomous optimization thread. **Supervisor:** main session (babysits).

## Objective
Cut the runtime of the **`2×4096`** benchmark shape (currently ~13,130μs on the
Modal harness) — and any sibling small-batch/large-n shapes that share the same
pathology — enough to improve the ranked geomean below the current best
`#877091` (~2062μs), ideally below the board leader (~1924μs). Fixing 2×4096
alone is worth ~9% on the geomean (see below), which by itself could beat the leader.

## Core hypothesis (test this FIRST, before building anything)
`2×4096` is anomalously slow and this looks like a **dispatch artifact**, not real compute:
- `1×4096` (batch=1) = ~1,537μs. Two serialized should be ~3,000μs, but `2×4096`
  measures ~13,130μs — ~4.3× worse than linear.
- Likely cause: `torch.linalg.cholesky[_ex]` routes **batched** inputs (batch ≥ 2) to
  `cusolverDnSpotrfBatched`, which is tuned for **many small** matrices, while `batch=1`
  uses the blocked single-matrix `potrf` optimized for **large** n. So few-but-large
  matrices hit the wrong path.

**First experiment step:** on Modal B200, benchmark `2×4096` three ways and compare:
1. Baseline: `torch.linalg.cholesky_ex(data).L` (current).
2. Per-matrix loop: `torch.stack([torch.linalg.cholesky_ex(data[i]).L for i in range(batch)])`.
3. Streamed per-matrix: same, but each slice on its own `torch.cuda.Stream` with a final sync.
Confirm whether (2)/(3) beat (1). If yes, the hypothesis holds; if no, report the real cause.

## If the hypothesis holds
- Find the `(batch, n)` region where per-matrix (or streamed) beats the batched call —
  candidates: `2×4096`, `2×2048`, `4×1024`, maybe `8×2048`, `1×4096` (already batch=1).
- Add a dispatch branch in `custom_kernel`: for large-n / small-batch, use the winning
  per-matrix or streamed path; keep the batched cuSOLVER call where it's already best
  (high-batch small-n), and keep the adopted Triton n=32 kernel.
- The threshold must be data-driven from Modal measurements, not guessed. Make sure NO
  shape regresses vs the current best.

## The loop (unchanged)
Inner: edit → `verify_local.py` (CPU) → `modal_verify.py verify` (B200 correctness) →
`modal_verify.py benchmark` (B200 per-shape; use the shape filter to iterate cheaply).
Outer (only on a confirmed Modal geomean win): `popcorn submit --mode test` →
`--mode leaderboard` → `popcorn submissions list` → record.

## Process requirements
- Experiment folder `experiments/004-<slug>/` with `submission.py`, `notes.md`
  (hypothesis, 3-way benchmark numbers, chosen threshold, per-shape deltas, correctness
  across all families, ranked id if submitted, verdict, Modal spend), `benchmark.json`.
- If adopted, copy winning `submission.py` to repo root; update `experiments/README.md`
  log + `journal.md`.
- **One git commit for the experiment** (adopted or rejected). This shell has NO heredoc /
  `$(...)`; use repeated `-m` flags: `git commit -m "exp 004: ... — <result>" -m "<details>"`.
  Do not push; do not touch git config.

## Guardrails
- Correctness across ALL families (dense/diagonal/spectrum/lowrank/rowscale/tridiagonal)
  on Modal B200 before any ranked submission. Per-matrix/streamed paths must be numerically
  identical to cuSOLVER (they call the same routine), so this should be clean — verify anyway.
- Cost: benchmark only affected shapes while iterating; full 15-shape grid only right before
  a ranked submit. Run modal/popcorn shell commands with the "all" permission.
- Quota: 2 of 3 ranked submissions remain. Use AT MOST 1 this experiment. Only submit if
  Modal shows geomean strictly better than #877091. No blind submissions.
- If per-matrix/streamed does NOT beat batched, STOP and document the real finding as a
  rejected experiment — that's a valid, useful outcome.

## Leverage math (why this shape matters)
Geomean over 15 shapes: improving one shape by factor f multiplies the geomean by
f^(1/15). Taking `2×4096` from 13,130μs → ~3,000μs (≈4.4×) ⇒ geomean ×(1/4.4)^(1/15) ≈ ×0.906,
i.e. ~9.4% — larger than the ~6.7% gap to the leader.

## Definition of done
- Ranked submission beating `#877091` (ideally the leader), confirmed via
  `popcorn submissions list`, with experiment folder + journal + git commit; OR
- Committed rejected-experiment folder + notes documenting why the dispatch change
  didn't help, with the 3-way measured numbers.

## Report back
Best geomean, whether leader beaten, ranked id(s), per-shape deltas (esp. 2×4096),
correctness status, experiment folder + commit hash, remaining ideas, approx Modal spend.
