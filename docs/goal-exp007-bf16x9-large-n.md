# Goal — exp 007: BF16x9 FP32-emulated trailing update for the large single-matrix shapes

**Owner:** autonomous optimization thread. **Supervisor:** main session (babysits).

## Objective
Push the ranked geomean strictly below the current best `#878015` (~1559μs) by
speeding up the **three large single-matrix shapes** — `1×8192` (~6390μs ranked,
still on cuSOLVER), `1×16384` (~19,400μs, blocked-TF32), `1×32768` (~77,200μs,
blocked-TF32) — using **BF16x9 FP32 emulation** on the O(n³) trailing Schur update
(and, if it helps, the panel solve / diagonal potrf).

These three shapes are the ones the Optimization Tracker in `journal.md` marks
**"TBD (top)"** in the *BF16x9 FP32-emu* column — the single highest-ROI unexplored
lever left on the board (candidate solution #1: "highest ROI, low effort"). They
dominate the wall clock (32768² alone is ~76% of total runtime) and, because the
correctness gate `‖A−LLᵀ‖₁ ≤ 20·n·eps·‖A‖₁` **grows with n**, they carry the most
numerical headroom and the largest achievable speedup ratios of anything remaining.

## Why BF16x9 (not just more TF32)
Session 6 shipped a blocked Cholesky with a **TF32** tensor-core trailing update
(16384 1.76×, 32768 2.86×), but on `1×8192` TF32 was only **1.07×** and
precision-marginal, so 8192 stayed on cuSOLVER. BF16x9 is a *different* lever:

- cuBLAS 12.9+ / CUDA 13 emulates a **true FP32 GEMM** as 9 BF16 products
  (`a = a1 + a2·2⁻⁸ + a3·2⁻¹⁶`, likewise `b`; `a·b+c` = 9 BF16 MACs). On Blackwell
  BF16 tensor cores this is **~3–4× faster than native FP32** at large M=N=K, at
  **~FP32 accuracy** (it correctly handles denormals/NaNs; accuracy ≥ native FP32).
- So BF16x9 is *higher* accuracy than TF32 **and** potentially faster. It could
  (a) finally give `1×8192` a real win (extend the blocked path down to 8192), and
  (b) beat the TF32 trailing update on 16384/32768 — or let the FP32 panel solve /
  diagonal potrf also run on tensor cores.

## The transferable lesson (from `lessons_qrproblem.md` / Sessions 1–6)
The recurring win here is "exploit the loose, n-scaled correctness gate — do the
heavy O(n³) trailing math on tensor cores." Prior sessions measured reconstruction
residuals **100–400× inside tolerance** on the ranked dense shapes. BF16x9 spends
that headroom more conservatively than TF32 (better accuracy) while chasing more
speed. cuSOLVER's `Spotrf` runs the whole factorization in native FP32; the trailing
update is the bulk of the FLOPs and is exactly where BF16x9 tensor cores can win.

## Concrete API leads (you must confirm the exact one that works on the Modal B200)
The Modal image is `nvidia/cuda:13.0.0-devel` with torch ~2.13+cu130 on a real
B200 (SM100a) — BF16x9 is *supported* on this hardware+toolkit. Options, cheapest
first; **measure which actually engages BF16x9** (see "How to confirm it engaged"):

1. **Env var (no code beyond one `os.environ`)**:
   `CUBLAS_FP32_EMULATED_BF16X9_MATH=1`. This makes cuBLAS use compute type
   `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` for FP32 GEMMs. **Must be set before the
   cuBLAS handle is created** — i.e. set `os.environ[...]="1"` at the very top of
   `submission.py`, before any matmul runs (ideally before `import torch`).
2. **PyTorch backend knobs**: `torch.backends.cuda.matmul.fp32_precision` (newer
   fine-grained FP32 precision framework) and/or
   `torch.backends.cuda.preferred_blas_library("cublaslt")`. Check what values the
   installed torch exposes (`print([...])`) — the emulated compute type may be
   reachable through this rather than the env var.
3. **Manual 3-way BF16 split in pure torch** (fallback, always works, no special
   API): decompose `L21 = h1 + h2 + h3` where `h1 = L21.bfloat16()`,
   `h2 = (L21 - h1.float()).bfloat16()`, `h3 = (L21 - h1.float() - h2.float()).bfloat16()`,
   then accumulate the 9 (actually 6 unique, by symmetry) BF16 matmuls in FP32.
   Slower than the fused cuBLAS path but proves the numerics and gives a floor.

**Hard constraint (unchanged):** popcorn statically scans the source and
disqualifies any use of **non-default CUDA streams** — it even flagged the literal
word "stream" in a comment (HTTP 500). BF16x9 needs none of this; keep
`submission.py` pure default-queue matmuls and avoid the literal word s-t-r-e-a-m
anywhere in the file.

## First experiment step — a precision probe (before touching the dispatcher)
Extend the existing `precprobe` harness in `scripts/_gpu_runner.py` (it already
sweeps `blocked_tf32_*` / `blocked_fp16_*` / `blocked_bf16_*` variants and reports
`speedup_vs_batched` + `tol_frac` + `margin_x`). Add BF16x9 variants:

- `blocked_bf16x9_nb{1024,2048,4096}` — blocked Cholesky, FP32 diagonal potrf +
  FP32 panel solve, **trailing GEMM `L21 @ L21ᵀ` via BF16x9-emulated FP32**.
- (optional) a variant that *also* runs the panel `solve_triangular`/potrf under
  emulation, to see if pushing more of the factorization onto tensor cores helps.

Run on `1×8192` and `1×16384` first (hold `1×32768` out — it's ~221ms/iter and
expensive; add it only for the final confirmation). Report per shape × variant:
mean μs, **speedup vs `batched` cuSOLVER**, and **worst-case reconstruction
residual / tolerance ratio** (margin). Also probe on the ill-conditioned families
(spectrum cond=5, lowrank cond=4) at 8192/16384, since those are what could trip a
low-precision trailing update.

**How to confirm BF16x9 actually engaged** (do not trust a speedup number blindly):
compare the BF16x9-variant time to both `batched` and `blocked_tf32`. If it's not
distinctly different from a plain-FP32 blocked run, the emulation likely did **not**
turn on (env var set too late, wrong toolkit path, etc.). Sanity-check by timing a
standalone large FP32 `A @ B` with vs without the env var/knob — BF16x9 should be
several× faster on the B200.

**Decision rule:**
- BF16x9 blocked is convincingly faster than the *current shipped path* for a shape
  (cuSOLVER for 8192; blocked-TF32 for 16384/32768) **and** passes the gate with a
  comfortable margin (aim ≥5–10× inside tolerance, across all families) → adopt for
  that shape.
- If it does not beat the current path, or only passes the gate by a thread →
  document and **reject** that shape (a valid outcome, like exp003/exp005). Ship only
  the shapes that actually won.

## Known pitfall — don't let emulation contaminate the accuracy measurement
The checker reconstructs `L @ Lᵀ` with a matmul. If you enable BF16x9 **globally**
(env var), the checker's / `_recon_ratio`'s reconstruction GEMM also becomes
emulated, which could *flatter* the residual. Make sure your reported margin
reflects a **genuine FP32 reconstruction**: either scope the emulation to only the
factorization's trailing update, or compute the residual with emulation disabled
(and rely on the real `reference.check_implementation`, whose PASS/FAIL is the true
gate — verify that runs clean on Modal for every family). If BF16x9 is truly
≈FP32-accurate this won't matter, but confirm it rather than assume.

## If the hypothesis holds
- Implement the winning path in `submission.py` (plain `torch`, **default queue
  only**). Prefer reusing/extending `_blocked_cholesky_tf32` with a
  precision/emulation switch rather than duplicating it.
- Widen the dispatch branch to include only the **measured-win** sizes. If 8192
  now wins, its branch changes from cuSOLVER to blocked-BF16x9; if 16384/32768 win
  vs TF32, switch their trailing update to BF16x9. Do **not** extrapolate to sizes
  you didn't measure.
- Re-tune `nb` per size under BF16x9 (its cost profile differs from TF32).
- Keep the `torch.isfinite(L).all()` → cuSOLVER **fallback** for ill-conditioned
  inputs (exp 006's safety net). It's ~memory-bound and negligible vs the O(n³) cost.

## The loop
**Inner (cheap, iterate here):**
1. Edit `submission.py` (and `_gpu_runner.py` for probe/benchmark variants only).
2. `uv run --with torch --with numpy python scripts/verify_local.py` (CPU, free).
3. `uv run --with modal python scripts/modal_verify.py precprobe --shapes 8192,16384`
   (the BF16x9 vs TF32 vs batched comparison + residual margins).
4. `uv run --with modal python scripts/modal_verify.py verify` (B200 correctness —
   the large-`n` specs across families are already in `TEST_SPECS`; add BF16x9-
   specific ones if you change the dispatch region).
5. `uv run --with modal python scripts/modal_verify.py benchmark --shapes 8192,16384`
   — per-shape mean vs `#878015`. Repeat until 8192/16384 are convincingly faster
   **and** correct, then add `32768`.

**Outer (scored, rare):** only when a **full 15-shape** Modal benchmark
(`benchmark` with no `--shapes`, incl. the ~221ms 32768² case) shows a geomean
strictly better than ~1559μs with no regressions →
`popcorn submit --mode test --no-tui submission.py` (expect 17/17) →
`popcorn submit --mode leaderboard --no-tui submission.py` → confirm with
`popcorn submissions list --leaderboard cholesky` → record in `journal.md`.

## Process requirements
- Experiment folder `experiments/007-bf16x9-large-n/` with `submission.py`,
  `notes.md` (hypothesis; probe table = precision × nb × speedup × residual-margin,
  incl. how you confirmed BF16x9 engaged; chosen approach + per-shape thresholds;
  per-shape deltas vs `#878015`; correctness across ALL families; ranked id if
  submitted; verdict; Modal spend), and `benchmark.json`.
- If adopted: copy the winning `submission.py` to repo root; update the
  Optimization Tracker cells (`1×8192/16384/32768` BF16x9 column), the
  `experiments/README.md` log, `journal.md` (newest session entry on top), and the
  README status/timings table.
- **One git commit for the experiment** (adopted or rejected). This shell has NO
  heredoc / `$(...)`; use repeated `-m` flags:
  `git commit -m "exp 007: ... — <result>" -m "<details>"`. Do not push; do not
  touch git config.
- Run modal/popcorn shell commands with the `all` (or `full_network`) permission.

## Guardrails (hard rules)
- **Correctness is non-negotiable.** Every candidate must pass
  `check_implementation` across ALL families (dense/diagonal/spectrum/lowrank/
  rowscale/tridiagonal) at the relevant large `n` on Modal B200 before any ranked
  submission. A precision that only passes by a thin margin is a reject, not a ship.
- **No non-default CUDA queues in `submission.py`** — instant DQ. Avoid the literal
  banned word anywhere in the file.
- **No regressions.** No shape may get slower than `#878015`. The dispatch must be
  tightly scoped to measured-win sizes; every other shape stays on its exact
  current path (n=32 Triton, small-batch/large-n loop, cuSOLVER elsewhere).
- **Cost discipline.** Iterate on 8192/16384 only; add 32768 (~221ms/iter) only for
  the final pre-submit full-grid confirmation. Reuse the cached Modal image.
- **Quota discipline.** Ranked `--mode leaderboard` runs are scarce and carry ~±20%
  cuSOLVER run-to-run drift on untouched shapes. Submit AT MOST once, and only when
  a full 15-shape Modal benchmark shows geomean strictly < ~1559μs with the large
  shapes clearly improved. No blind submits. If in doubt, report the Modal numbers
  to the supervisor and stop before the ranked submit.
- **No scope creep.** Do not change `reference/`, the eval harness semantics, or git
  config. Do not commit until the experiment is complete (adopted or rejected).

## Leverage math (why these three)
Geomean over 15 shapes: improving one shape by factor `f` multiplies the geomean by
`f^(1/15)`. Relative to the current `#878015` per-shape means (8192 ≈ 6390μs,
16384 ≈ 19,400μs, 32768 ≈ 77,200μs):
- If BF16x9 gives 8192 a 2× (6390→~3200) **and** improves 16384/32768 to 3× vs
  cuSOLVER (i.e. beats the current TF32 further): geomean drops well below 1559μs.
- Even 8192 alone going 1.0×→2× is ×0.5^(1/15) ≈ ×0.955 → ~1490μs.
- 16384+32768 each getting another 1.3× on top of TF32 → another ~×0.965 → compounding.
This remains the single largest lever left; small-batch/mid-n cleanups are ~0.05%.

## Definition of done
- A ranked submission with geomean strictly < ~1559μs, confirmed via
  `popcorn submissions list --leaderboard cholesky`, with experiment folder +
  journal entry + README + tracker update + git commit; **OR**
- A committed **rejected**-experiment folder + notes with the probe table showing
  BF16x9 cannot beat the current shipped paths at these sizes (didn't engage / too
  slow / fails or barely-passes the gate) — a valid outcome that closes the BF16x9
  column for large-n.

## Report back (to supervisor)
Best full-grid Modal geomean; whether ranked was submitted (+id); per-shape deltas
(esp. 8192/16384/32768) vs `#878015`; the probe table (precision × nb × speedup ×
residual margin) and how you confirmed BF16x9 engaged; which API path worked
(env var vs torch knob vs manual split); correctness across families; experiment
folder + commit hash; remaining ideas; approx Modal spend.
