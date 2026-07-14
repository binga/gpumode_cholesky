# Journal — GPU MODE `cholesky` leaderboard

Running log of work, results, and findings. Newest entries at the top.

---

## 2026-07-15 — Session 2: first custom kernel (Triton n=32) → ranked #877091

### Goal
Beat the cuSOLVER baseline (ranked `#876988`, geomean ≈ 2080μs) by replacing
cuSOLVER with custom kernels on the highest-ROI (launch/overhead-bound) shapes:
`4096×32`, `1024×64`, `256×128`.

### What was built
- **Triton batched Cholesky kernel** (`submission.py`): one program (CTA) per
  matrix, whole `N×N` matrix held in a single tile spread across the block's
  threads. Right-looking factorization — at step k: `inv = 1/sqrt(A[k,k])`,
  scale column k, rank-1 update of the trailing submatrix — then zero the strict
  upper triangle. `N` is a `constexpr` so the k-loop is fully unrolled and the
  kernel is specialized per size (Triton caches the compile at module scope).
- **Dispatcher**: `custom_kernel` routes `n==32` (CUDA, fp32) to the Triton
  kernel; everything else stays on `torch.linalg.cholesky_ex` (cuSOLVER).
- **Harness upgrades** (`scripts/_gpu_runner.py`, `scripts/modal_verify.py`):
  - `--shapes` filter (e.g. `--shapes 32,64,128`) to benchmark only active shapes
    in the inner loop and save B200 cost.
  - **L2-cache clear** (256 MB buffer zeroed between timed iters) + **adaptive
    iters** (50 for n≤256, down to 8 for the huge matrices) to better mirror
    popcorn's official timing (which clears L2 via `clear_l2_cache`).
  - Extra `n=32` verify specs across all families (spectrum/diagonal/lowrank/
    rowscale/tridiagonal + high batch) to harden the correctness gate.

### The decisive experiment (Modal B200, L2-clear method — apples-to-apples)
| shape | cuSOLVER | Triton (num_warps=1) | verdict |
|---|---|---|---|
| 4096×32  | 137.8μs | **84→76μs** | **Triton −39%** |
| 1024×64  | 135.7μs | 152μs (best cfg) | cuSOLVER wins |
| 256×128  | 201.5μs | 429μs | cuSOLVER wins |

**Key insight — `num_warps=1` is the unlock for n=32.** With one warp per matrix,
Triton's per-column reductions (`tl.sum`) compile to cheap in-warp shuffles
instead of shared-memory syncs. That beats cuSOLVER's batched-launch overhead.
For n≥64 a single warp spills registers (n=64→128 regs/thread; n=128 catastrophic
at ~5ms), and multi-warp configs re-introduce sync cost, so both lose to cuSOLVER.
→ **Triton only pays off at n=32** with the current tile-per-matrix design.

### Results — ranked submission `#877091` (17/17 pass, B200)
Custom kernel correct on **all** families at n=32 (worst scaled reconstruction
residual 0.082, tolerance is 20 — huge margin). Modal verify: 19/19.

#### Ranked per-shape (popcorn), baseline `#876988` → this run `#877091`
| shape | #876988 | #877091 | Δ |
|---|---|---|---|
| **4096×32** | **113μs** | **63.7μs** | **−44%** ← the win |
| 1024×64 | 110 | 110 | — |
| 256×128 | 152 | 152 | — |
| 64×256 | 276 | 276 | — |
| 16×512 | 597 | 600 | — |
| 640×512 | 3810 | 3800 | — |
| 4×1024 | 1280 | 1395 | +9% (cuSOLVER drift) |
| 60×1024 | 2900 | 2900 | — |
| 2×2048 | 3220 | 3840 | +19% (cuSOLVER drift) |
| 8×2048 | 4910 | 5010 | +2% (drift) |
| 1×4096 | 1540 | 1534 | — |
| 2×4096 | 11400 | 13400 | +18% (cuSOLVER drift) |
| 1×8192 | 6400 | 6410 | — |
| 1×16384 | 34200 | 34200 | — |
| 1×32768 | 221000 | 221000 | — |

**Geomean of this ranked run ≈ 2062μs** (computed from the per-shape means; the
`popcorn submissions list` Score column shows `-`). That is below the recorded
baseline of ~2080μs, so the definition of done is met — but only marginally in
*absolute* terms, because several **cuSOLVER** shapes (identical code) ran
notably slower this session (`2×2048`, `2×4096`, `4×1024`). That is pure
run-to-run environment drift, not a regression. Same-environment (Modal,
L2-clear, everything but n=32 held fixed) the win is **~3.9%**: n=32 alone moves
the geomean-monotone score from an equivalent pure-cuSOLVER ~2388μs to 2296μs.

### Findings & insights
- **Confirmed the launch/overhead-bound hypothesis for n=32.** 113→63.7μs from a
  single fused Triton launch vs cuSOLVER's batched dispatch across 4096 tiny
  matrices. The floor is ~memory-bound (~5μs for 32 MB R/W); 63.7μs is still
  mostly fixed overhead, so there may be a little more with a multi-matrix-per-
  program design, but returns are small.
- **Triton's tile-per-matrix model caps out at n=32 here.** The right-looking
  loop needs per-step column extraction (a reduction). One warp keeps that as
  shuffles (fast) but limits registers; more warps add sync cost. n=64/128 need
  a **warp-per-matrix CUDA kernel** (register-blocked columns + `__shfl`), which
  needs nvcc — not available in our pip-torch Modal image (would require a CUDA
  *devel* base image to test). Deferred: higher effort + risk.
- **cuSOLVER shapes drift run-to-run** on the board (~±20% on some mid shapes),
  so absolute geomean deltas < a few % are noisy. Trust per-shape same-seed
  deltas (n=32: 113→63.7 is rock-solid) over the raw geomean number.
- **Accuracy is a non-issue** for this simple FP32 right-looking factorization —
  residuals are 100–1000× inside tolerance across all families.

### Cost
~9 Modal B200 sandbox runs this session (verify/benchmark, ~40–65s each) ≈ ~10 min
B200 wall ≈ **~$1–2** Modal spend. popcorn test+leaderboard run on GPU MODE infra
(not billed to our Modal). Ranked submissions used this session: **1 of 3**.

### Next steps (to chase the board leader ~1924μs)
1. **Warp-per-matrix CUDA kernel for n∈{64,128}** via `load_inline` (nvcc is on
   popcorn's runner per the brief). Design: block-per-matrix, `n` threads, thread
   `j` owns column `j`; right-looking with a shared-mem pivot-column broadcast;
   ~2n syncs, O(n³/n) work/thread. To iterate on Modal, switch the image to an
   `nvidia/cuda:*-devel` base so `load_inline` can compile there. Wrap in
   try/except → fall back to cuSOLVER so a compile failure never breaks ranking.
   Potential: if 64/128 also reach ~0.5× cuSOLVER, geomean → ~1810μs (beats leader).
2. **Multi-matrix-per-program Triton for n=32** to shave the remaining launch
   overhead (63.7 → maybe ~50μs). Small ROI but cheap and low-risk.
3. Leave `n≥256` on cuSOLVER (compute-bound; cuSOLVER already near SOL).

---

## 2026-07-15 — Session 1: setup → first ranked submission

### Goal
Participate in the GPU MODE [`cholesky` leaderboard (776)](https://www.gpumode.com/leaderboard/776?tab=rankings)
— batched dense Cholesky factorization on **B200**, ranked by geometric mean of
runtime across 15 benchmark shapes. Ambition for this session: **land a correct
ranked submission first**, defer deep optimization.

### Environment
- Dev machine: macOS, **no local NVIDIA GPU**.
- `popcorn` CLI installed; authenticated via **GitHub** this session.
- `modal` used on-demand via `uv run --with modal` (`~/.modal.toml` already present).

### What was built
- `submission.py` — cuSOLVER baseline (`torch.linalg.cholesky_ex(...).L`) with
  `#!POPCORN leaderboard cholesky` / `#!POPCORN gpu B200` directives and a
  shape-dispatcher structure for future custom kernels.
- `reference/` — vendored read-only harness (`task.py`, `reference.py`, `eval.py`,
  `utils.py`); the checker here is the real spec.
- Three-tier verification:
  - `scripts/verify_local.py` — free CPU property check.
  - `scripts/modal_verify.py` + `scripts/_gpu_runner.py` — real **B200** via a Modal sandbox.
- Plan: `docs/plans/2026-07-15-001-feat-cholesky-leaderboard-submission-plan.md`.

### Results

| Check | Result |
|---|---|
| CPU property check (`verify_local.py`) | **10/10 pass** |
| Modal B200 verify (`modal_verify.py verify`) | **13/13 pass** on `NVIDIA B200`, torch 2.12.0+cu130 |
| popcorn `--mode test` | **17/17 pass** on B200 |
| popcorn `--mode leaderboard` (`#876988`) | **done, 17/17 pass**, ranked geomean ≈ **2080μs** |

Reference points on the board at submission time: xuan9938 ~1924μs, msaroufim ~2041μs.
The raw cuSOLVER baseline (~2080μs) is already competitive — roughly ~2% behind 2nd.

#### Ranked per-shape times (popcorn, B200)
| shape | mean | | shape | mean |
|---|---|---|---|---|
| 4096×32 | 113 µs | | 60×1024 | 2.90 ms |
| 1024×64 | 110 µs | | 2×2048 | 3.22 ms |
| 256×128 | 152 µs | | 8×2048 | 4.91 ms |
| 64×256 | 276 µs | | 1×4096 | 1.54 ms |
| 16×512 | 597 µs | | 2×4096 | 11.4 ms |
| 640×512 | 3.81 ms | | 1×8192 | 6.40 ms |
| 4×1024 | 1.28 ms | | 1×16384 | 34.2 ms |
|  |  | | 1×32768 | 221 ms |

Raw logs: `results/leaderboard-*.txt`, `results/test-*.txt`.
Summaries (committed): `results/ranked-submission-876988.json`, `results/baseline-benchmark.json`.

### Findings & insights
- **The baseline is already strong.** Plain `torch.linalg.cholesky_ex` (cuSOLVER) on
  B200 lands within ~2% of 2nd place. The competition is tight at the top; wins are marginal.
- **Only soft spots are small-`n` / high-batch shapes** (`4096×32`=113μs, `1024×64`=110μs,
  `256×128`=152μs). These are **launch/overhead-bound**, not compute-bound — a 32×32
  factorization is trivial, so ~110μs is almost pure per-call + dispatch overhead across
  thousands of tiny matrices. This is exactly where custom batched kernels win, and matches
  the leaders' known trick (cf. the repo's `triton_cholesky32.py`, one program per matrix).
- **Large single matrices are compute-bound** (`32768²`=221ms, `16384²`=34ms). cuSOLVER is
  already near speed-of-light here; low ROI — leave on cuSOLVER.
- **Property-based checker is forgiving on accuracy** — scaled reconstruction residuals were
  ~0.0006–0.024 (tolerance is `20·n·eps·‖A‖₁`). There's headroom to trade a little accuracy
  for speed (e.g., TF32 in intermediate steps) *if* it doesn't break the FP32 reconstruction gate.
- **Modal verification paid off as a pre-flight.** Both the Modal B200 verify and the popcorn
  test reported identical residuals — Modal caught nothing broken here, but it means future
  kernel work can be validated on the exact hardware without burning ranked quota.

### Gotchas
- `modal.Sandbox.exec()` timed out connecting to Modal's newer per-task command-router
  (blocked egress here). Fix: run the command as the sandbox **entrypoint** and stream
  `sandbox.stdout` — the documented pattern, works over the standard control channel.
- Default torch wheel (2.12.0+cu130) already ships Blackwell/sm_100 kernels — no cu128 pin needed.
- `popcorn register` is OAuth (github/discord); must be completed in a browser.

### Next steps (deferred optimization)
1. Custom batched kernel for `n ∈ {32, 64, 128}` (Triton or CUDA `load_inline`), starting from
   the `triton_cholesky32.py` pattern; dispatch on `(batch, n)` in `custom_kernel`.
2. Re-benchmark on Modal (`modal_verify.py benchmark`) before each ranked submission.
3. Tune high-batch mid-size shapes (`640×512`, `8×2048`, `2×4096`) for occupancy.
4. Leave `n ≥ 8192` on cuSOLVER.
