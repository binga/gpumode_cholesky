# Goal — exp 005: speed up high-batch mid-`n` (primary `640×512`)

**Owner:** autonomous optimization thread. **Supervisor:** main session (babysits).

## Objective
Cut the runtime of the **`640×512`** benchmark shape (currently ~3800μs ranked,
`#877941`) — the biggest *attackable* contributor left on the board — enough to
improve the ranked geomean below the current best `#877941` (~1746μs). A 2.5×
win on `640×512` is worth ~6% on the geomean (leverage math below). Secondarily,
resolve the known **`8×2048` own-goal regression** (5370 vs 5010 on batched).

Everything else is already at its frontier: `n=32` (Triton), the small-batch
large-`n` shapes `2×4096 / 2×2048 / 4×1024` (loop, exp004), `n=64/128` (exp003
proved cuSOLVER wins), `60×1024` (exp004 probe proved cuSOLVER batched is best),
and the huge single matrices `n>=8192` (compute-bound, near speed-of-light).
Do NOT touch those paths except to avoid regressing them.

## Core hypothesis (test this FIRST, before building anything)
`640×512` is slow because `torch.linalg.cholesky[_ex]` routes batch>=2 to
`cusolverDnSpotrfBatched`, which is tuned for **many-tiny** matrices and
**under-utilizes the B200** for "hundreds of medium (512²) matrices" — it neither
saturates the SMs the way `4096×32` does, nor uses the fast single-matrix blocked
`potrf`. Evidence from exp004: `8×2048` batched=5612μs vs **streamed=3477μs**
(~35% concurrency left on the table).

**First experiment step — a characterization probe (cheap):** add
`{"batch": 640, "n": 512, "cond": 2, "seed": 510512}` (and keep `8×2048`) to
`PROBE_SPECS` in `scripts/_gpu_runner.py`, then run
`uv run --with modal python scripts/modal_verify.py probe --shapes 512,2048`.
Compare **batched vs loop vs streamed** for `640×512`:
- If **streamed >> batched** → concurrency headroom is real; the bottleneck is
  under-occupancy, and the task is to capture it **without non-default streams**
  (see constraint below).
- If **streamed ≈ batched** → cuSOLVER already saturates the GPU at this shape;
  it is compute/memory-bound → likely a dead end. Document and **reject** (a valid
  outcome, like exp003). Do not burn quota chasing a saturated shape.
- `loop` (640 sequential potrf) will almost certainly be terrible here — that's
  expected; it's a control.

`streamed` is a **diagnostic only** to reveal headroom. It must NEVER ship.

## Hard constraint — no non-default streams in the submission
popcorn statically scans the source and **disqualifies** any use of non-default
CUDA streams (it even flagged the literal word "stream" in a comment last session,
HTTP 500). So `torch.cuda.Stream`, `torch.cuda.stream(...)`, stream kwargs, etc.
are **forbidden in `submission.py`**. Streamed timing is allowed *only* inside
`scripts/_gpu_runner.py` (never submitted) as a headroom probe.

## If the hypothesis holds (streamed shows real headroom for `640×512`)
Ways to capture medium-matrix concurrency without streams, in rough order of ROI:
1. **CUDA graph capture** of the per-matrix or batched call (single default
   stream, replayed) — removes per-launch overhead; test if it overlaps.
2. **Chunked batched calls** — split 640 into a few `cholesky_ex` calls on
   sub-batches to hit a better-occupancy code path; data-driven sweep of chunk size.
3. **A custom batched blocked kernel** (Triton or CUDA `load_inline` — the Modal
   image already has nvcc, exp003 unlocked it): panel factorization + batched-GEMM
   trailing update, multiple matrices per block for occupancy. High effort; exp003
   showed a *naive* right-looking kernel loses at n=128, so only pursue a **blocked
   / tensor-core (tf32 with fp32 accumulate)** design — the tolerance has ~1000×
   headroom. This is the leaders' known trick for mid-`n`.
Pick the cheapest approach that beats batched on Modal; escalate only if needed.

Add a `custom_kernel` dispatch branch keyed on `(batch, n)` for the winning region
(e.g. `n==512 and batch>=~64`). **Must not regress `16×512`** (600μs, batch=16,
stays on cuSOLVER) or any other shape.

## `8×2048` own-goal (secondary, cheap)
The exp004 loop region `2<=batch<=8, n>=1024` regressed `8×2048` (5010→5370).
Cheapest fix: restrict the loop region to `2<=batch<=4` so `8×2048` returns to
batched cuSOLVER (5010). Bundle this into the same submission if you spend the
ranked slot. Verify it doesn't disturb `2×2048/2×4096/4×1024`.

## The loop
**Inner (cheap, iterate here):**
1. Edit `submission.py` (and `_gpu_runner.py` for probes/benchmarks only).
2. `uv run --with torch --with numpy python scripts/verify_local.py` (CPU, free).
3. `uv run --with modal python scripts/modal_verify.py verify` (B200 correctness).
4. `uv run --with modal python scripts/modal_verify.py benchmark --shapes 512,2048`
   (fast, only the shapes under work); compare to `#877941` per-shape.
5. Repeat until `640×512` is convincingly faster **and** correct, with no regressions.

**Outer (scored, rare):** only when a **full 15-shape** Modal benchmark
(`benchmark` with no `--shapes`) shows a geomean strictly better than ~1746μs →
`popcorn submit --mode test --no-tui submission.py` (expect 17/17) →
`popcorn submit --mode leaderboard --no-tui submission.py` → record in `journal.md`.

## Process requirements
- Experiment folder `experiments/005-highbatch-mid-n/` with `submission.py`,
  `notes.md` (hypothesis, probe numbers, chosen approach + threshold, per-shape
  deltas, correctness across all families, ranked id if submitted, verdict,
  Modal spend), `benchmark.json`.
- If adopted, copy winning `submission.py` to repo root; update
  `experiments/README.md` log + `journal.md` (newest entry on top).
- **One git commit for the experiment** (adopted or rejected). This shell has NO
  heredoc / `$(...)`; use repeated `-m` flags:
  `git commit -m "exp 005: ... — <result>" -m "<details>"`. Do not push; do not
  touch git config.
- Run modal/popcorn shell commands with the "all" permission (network + full).

## Guardrails (hard rules)
- **Correctness is non-negotiable.** Every candidate must pass
  `check_implementation` across ALL families (dense/diagonal/spectrum/lowrank/
  rowscale/tridiagonal) on Modal B200 before any ranked submission.
- **No non-default streams in `submission.py`** (see constraint above) — instant DQ.
- **No regressions.** No shape may get slower than `#877941`. Guard `16×512`
  specifically when adding an `n==512` branch.
- **Cost discipline.** Benchmark only `--shapes 512,2048` while iterating; run the
  full 15-shape grid (incl. the 221ms 32768² case) only immediately before a
  ranked submit.
- **Quota discipline.** Only **1 ranked `--mode leaderboard` submission remains**
  this run (2 of 3 already used: `#877091`, `#877941`). Use it AT MOST once, and
  only if a full Modal benchmark shows geomean strictly < ~1746μs. No blind submits.
- **Numerical headroom, carefully.** Tolerance is `20·n·eps·‖A‖₁`; tf32/bf16
  intermediates are fine ONLY if the FP32 reconstruction gate still passes on Modal
  across all families.
- **No scope creep.** Do not change `reference/`, the eval harness semantics, the
  plan's intent, or git config. Do not commit unless the experiment is complete.

## Leverage math (why `640×512`)
Geomean over 15 shapes: improving one shape by factor `f` multiplies the geomean
by `f^(1/15)`. `640×512` 3800→1520μs (2.5×) ⇒ ×(1/2.5)^(1/15) ≈ ×0.941, i.e.
**~6%** — comfortably widens our lead over the ex-leader (~1924μs). Even 3800→2500
(1.5×) is ~2.7%. `8×2048` 5370→5010 is only ~0.05% (a cleanup, not the prize).

## Definition of done
- A ranked submission with geomean strictly < ~1746μs, confirmed via
  `popcorn submissions list --leaderboard cholesky`, with experiment folder +
  journal entry + git commit; OR
- A committed **rejected**-experiment folder + notes: the probe numbers showing
  `640×512` is already cuSOLVER-saturated (streamed ≈ batched) or that no
  non-stream approach beat batched — a valid, useful outcome that closes the shape.

## Report back (to supervisor)
Best geomean, whether ranked was submitted (+id), per-shape deltas (esp.
`640×512`, `8×2048`), the probe table (batched/loop/streamed), which approach won
or why none did, correctness status, experiment folder + commit hash, remaining
ideas, and approx Modal spend.
