# Goal — exp 006: speed up the large single-matrix shapes via tensor-core blocked Cholesky

**Owner:** autonomous optimization thread. **Supervisor:** main session (babysits).

## Objective
Cut the runtime of the **large single-matrix shapes** — `1×8192` (~6400μs ranked),
`1×16384` (~34,200μs), `1×32768` (~221,000μs) — enough to improve the ranked
geomean below the current best `#877956` (~1744μs). These three shapes were
dismissed in every prior session as "compute-bound, cuSOLVER near speed-of-light,
leave it" — but that was **never actually tested against lower-precision
tensor-core math**. They dominate the wall-clock (32768² alone is ~76% of total
runtime) and have the largest *achievable* speedup ratios left on the board, so
they carry the most geomean leverage of anything remaining (math below).

Secondary (only if the primary kernel generalizes cheaply): `8×2048` (~5060μs),
the biggest non-huge shape, still batched/panel-bound on cuSOLVER.

Everything else is already at its frontier and **must not regress**: `n=32`
(Triton, exp002), small-batch large-`n` `2×4096 / 2×2048 / 4×1024` (loop, exp004),
`n=64/128` (cuSOLVER wins, exp003), `60×1024` and `640×512` (cuSOLVER
saturated/optimal, exp004/exp005). Do NOT touch those dispatch paths.

## The transferable lesson (from `lessons_qrproblem.md`)
The #1 cross-cutting win in the QR project was **"exploit the loose correctness
gates — TF32/FP16 tensor-core math is free on the trailing/apply steps"** (FP16
was the sweet spot there; BF16 failed the factor gate; TF32 tied). The cholesky
checker gates only the reconstruction residual `‖A − LLᵀ‖ ≤ 20·n·eps·‖A‖₁`, and
prior sessions measured residuals **100–1000× inside tolerance**. Crucially the
tolerance *grows with n*, so the huge shapes have the most numerical headroom to
spend. cuSOLVER's `Spotrf` runs the whole factorization in FP32; the O(n³)
trailing update is the bulk of the FLOPs and is exactly where B200 tensor cores
(TF32 ≈ several× FP32 throughput; FP16 more) can win.

## Core hypothesis (test this FIRST, before building a real kernel)
A **right-looking blocked Cholesky** where:
- the **diagonal block** `L11 = chol(A11)` stays **FP32** (small, cheap, keeps
  stability),
- the **panel solve** `L21 = A21 · L11⁻ᵀ` and especially the **trailing Schur
  update** `A22 ← A22 − L21·L21ᵀ` (the O(n³) cost) run on **tensor cores in TF32
  (or FP16) with FP32 accumulation**,

can beat cuSOLVER's all-FP32 `potrf` on `n ≥ 8192` while still passing the
reconstruction gate across all families. This needs **no custom CUDA** to
prototype — it can be expressed with `torch` ops on the default stream:
`torch.linalg.cholesky_ex` on the diagonal block, `torch.linalg.solve_triangular`
or a matmul-based solve for the panel, and `@`/`baddbmm` for the trailing update
with `torch.backends.cuda.matmul.allow_tf32=True` (or explicit `.half()` casts,
FP32 accumulate). A blocked Python loop is fine here: at block size `nb=2048`,
`n=32768` is only 16 outer steps, so launch overhead is negligible vs the GEMM.

## First experiment step — a characterization probe (do this before writing the dispatcher)
Add a large-`n` precision probe to `scripts/_gpu_runner.py` (a new probe mode or
extend `run_probe`). For `1×8192` and `1×16384` (hold `1×32768` out until the
approach works — it is ~221ms/iter and expensive), compare **as callables on the
default stream**, each checked with `check_implementation`:
1. `batched` — `torch.linalg.cholesky_ex(A).L` (current baseline / control).
2. `blocked_tf32` — blocked Cholesky, trailing GEMM in TF32 (`allow_tf32=True`),
   FP32 diagonal block. Sweep `nb ∈ {512, 1024, 2048}`.
3. `blocked_fp16` — same, trailing GEMM in FP16 (cast operands, FP32 accumulate).
4. (optional) `blocked_bf16` — for reference; expect it to lose the residual gate.

Report, per shape and precision: mean μs, speedup vs `batched`, and the
**worst-case reconstruction residual / tolerance ratio** so we can see the
numerical margin. Decision rule:
- If `blocked_tf32` (or `_fp16`) is convincingly faster than `batched` **and**
  passes the gate with margin → hypothesis holds; proceed to build the shippable
  path and sweep `nb`.
- If neither precision beats `batched`, or they only pass the gate by a thread →
  document and **reject** (a valid outcome, like exp003/exp005). cuSOLVER FP32 is
  then genuinely near-SOL for these shapes.
- Pick the **coarsest precision that keeps a comfortable residual margin** (aim
  for ≥5–10× inside tolerance, not 1.1×) — the huge shapes are condition-sensitive.

## If the hypothesis holds
- Implement the blocked Cholesky as a helper in `submission.py` (plain `torch`,
  **default stream only**), and add a `custom_kernel` dispatch branch keyed on
  `(batch, n)` for the winning region, e.g. `batch == 1 and n >= 8192` (data-driven
  — only include the sizes where you *measured* a win; do not extrapolate).
- Re-tune `nb` per size if needed (larger `nb` → fewer steps but bigger diagonal
  FP32 potrf; smaller `nb` → more tensor-core-friendly trailing but more steps).
- Consider a **two-level** scheme only if a single block level leaves the diagonal
  potrf as the bottleneck (recurse the diagonal block, or just call cuSOLVER on it).
- **Secondary `8×2048`:** only if the same kernel, applied per-matrix or batched,
  beats the current `8×2048` batched cuSOLVER (~5060μs) on Modal. Otherwise leave it.

## Hard constraint — no non-default streams in `submission.py`
popcorn statically scans the source and **disqualifies** any use of non-default
CUDA streams (it flagged even the literal word "stream" in a comment, HTTP 500).
So `torch.cuda.Stream`, `torch.cuda.stream(...)`, stream kwargs, etc. are
**forbidden in `submission.py`**. The TF32/FP16 approach needs none of this — it is
pure default-stream matmuls. (Streamed timing may only ever live in
`scripts/_gpu_runner.py` as a probe, never submitted.)

## The loop
**Inner (cheap, iterate here):**
1. Edit `submission.py` (and `_gpu_runner.py` for probes/benchmarks only).
2. `uv run --with torch --with numpy python scripts/verify_local.py` (CPU, free).
3. `uv run --with modal python scripts/modal_verify.py verify` (B200 correctness —
   add large-`n` specs to `TEST_SPECS` across families before submitting).
4. `uv run --with modal python scripts/modal_verify.py benchmark --shapes 8192,16384`
   (fast: only the shapes under work). Compare per-shape to `#877956`.
5. Repeat until `8192/16384` are convincingly faster **and** correct, then add
   `32768` for the final confirmation run.

**Outer (scored, rare):** only when a **full 15-shape** Modal benchmark
(`benchmark` with no `--shapes`, incl. the ~221ms 32768² case) shows a geomean
strictly better than ~1744μs → `popcorn submit --mode test --no-tui submission.py`
(expect 17/17) → `popcorn submit --mode leaderboard --no-tui submission.py` →
confirm with `popcorn submissions list --leaderboard cholesky` → record in `journal.md`.

## Process requirements
- Experiment folder `experiments/006-large-n-tensorcore/` with `submission.py`,
  `notes.md` (hypothesis, probe table with precision × nb × residual-margin,
  chosen approach + threshold, per-shape deltas, correctness across ALL families,
  ranked id if submitted, verdict, Modal spend), `benchmark.json`.
- If adopted, copy the winning `submission.py` to repo root; update
  `experiments/README.md` log + `journal.md` (newest entry on top) + the README
  status/timings table.
- **One git commit for the experiment** (adopted or rejected). This shell has NO
  heredoc / `$(...)`; use repeated `-m` flags:
  `git commit -m "exp 006: ... — <result>" -m "<details>"`. Do not push; do not
  touch git config.
- Run modal/popcorn shell commands with the "all" (or full_network) permission.

## Guardrails (hard rules)
- **Correctness is non-negotiable.** Every candidate must pass
  `check_implementation` across ALL families (dense/diagonal/spectrum/lowrank/
  rowscale/tridiagonal) at the relevant large `n` on Modal B200 before any ranked
  submission. Add large-`n` (`8192`, `16384`; at least one `32768`) specs across
  families to `TEST_SPECS`. A precision that only passes by a thin margin is a
  reject, not a ship.
- **No non-default streams in `submission.py`** — instant DQ.
- **No regressions.** No shape may get slower than `#877956`. The dispatch branch
  must be tightly scoped (only the measured-win sizes); everything else stays on
  its current path.
- **Cost discipline.** Iterate on `8192`/`16384` only; add `32768` (~221ms/iter,
  expensive) only for the final pre-submit full-grid confirmation. Reuse the
  cached Modal image.
- **Quota discipline.** Ranked `--mode leaderboard` submissions are scarce and each
  ranked run has ~±20% cuSOLVER run-to-run drift on untouched shapes. Submit AT
  MOST once, and only when a full 15-shape Modal benchmark shows geomean strictly
  < ~1744μs with the large shapes clearly improved. No blind submits. If in doubt,
  report the Modal numbers to the supervisor and stop before the ranked submit.
- **Numerical headroom, carefully.** Tolerance is `20·n·eps·‖A‖₁`. TF32/FP16 in the
  trailing update is allowed ONLY if the FP32 reconstruction gate still passes with
  margin across all families at the target `n`. Prefer TF32 unless FP16 also passes
  comfortably.
- **No scope creep.** Do not change `reference/`, the eval harness semantics, or git
  config. Do not commit until the experiment is complete (adopted or rejected).

## Leverage math (why the large shapes)
Geomean over 15 shapes: improving one shape by factor `f` multiplies the geomean by
`f^(1/15)`.
- All three (`8192/16384/32768`) at **2×**: ×(1/2)^(3/15) ≈ **×0.871 → ~1519μs (−13%)**.
- All three at **3×** (plausible with FP16 on the huge ones): ×(1/3)^(3/15) ≈
  **×0.803 → ~1400μs (−20%)**.
- Just `16384`+`32768` at 2×: ×(1/2)^(2/15) ≈ **×0.910 → ~1585μs (−9%)**.
- Even `32768` alone at 2×: ×(1/2)^(1/15) ≈ ×0.955 → ~1665μs (−4.5%).
This is the single largest lever left; it dwarfs the `8×2048` cleanup (~0.05%).

## Definition of done
- A ranked submission with geomean strictly < ~1744μs, confirmed via
  `popcorn submissions list --leaderboard cholesky`, with experiment folder +
  journal entry + README update + git commit; OR
- A committed **rejected**-experiment folder + notes: the probe table showing that
  TF32/FP16 blocked Cholesky cannot beat cuSOLVER FP32 at these sizes (too slow, or
  fails/barely-passes the residual gate) — a valid, useful outcome that closes the
  large-`n` shapes.

## Report back (to supervisor)
Best full-grid Modal geomean; whether ranked was submitted (+id); per-shape deltas
(esp. `8192/16384/32768`); the probe table (precision × nb × speedup × residual
margin); which precision/nb won or why none did; correctness status across
families; experiment folder + commit hash; remaining ideas; approx Modal spend.
