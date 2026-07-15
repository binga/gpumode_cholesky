# Journal — GPU MODE `cholesky` leaderboard

Running log of work, results, and findings. Newest entries at the top.

---

## 2026-07-15 — Session 5: `640×512` probe → REJECTED (cuSOLVER-saturated)

### Goal

Attack the biggest attackable board shape, **`640×512`** (~3800μs ranked), to push
the geomean below the current best `#877941` (~1746μs) — or prove it's already
cuSOLVER-saturated and cleanly reject. Secondary: the `8×2048` own-goal (5010→5370).

### Result — REJECTED, nothing submitted

The characterization probe closes the shape. `640×512` **cuSOLVER batched is optimal**
— every decomposition is dramatically slower:

| shape | batched | loop | streamed | chunk64 | chunk128 | best |
|---|---|---|---|---|---|---|
| **640×512** | **3954.9** | 104887.1 | 25729.9 | 10494.9 | 7007.6 | **batched** |
| 2×2048 (control) | 4627.3 | **1384.2** | 1467.1 | 4674.2 | 4669.7 | loop |
| 8×2048 (control) | 5738.1 | 5427.9 | **3478.1** | 5890.7 | 5883.7 | streamed |

`streamed` (max concurrency) is **6.5× SLOWER** than batched for `640×512` — the exact
opposite of the exp-004 headroom signal. There is **no under-occupancy to capture**:
cuSOLVER's batched `potrf` already saturates the B200 for hundreds of medium matrices.
`chunk64/128` (the shippable default-stream alternative) are 1.8×–2.7× slower, ruling
out chunked batched calls **with data**. Controls reproduce exp-004 (loop/streamed win
for few-large), confirming the harness. CUDA graph capture is pointless (single batched
launch, nothing to amortize); a custom blocked kernel would have to beat a *saturated*
vendor routine — not worth it after exp-003 showed naive kernels lose at n≤128.

### `8×2048` own-goal + ranked-slot decision (flagged to supervisor)

Fix is trivial (trim loop region `2<=batch<=8` → `2<=batch<=4`, sending 8×2048 back to
batched: 5010 < 5370 on popcorn). Prepared as `experiments/005-.../submission.py`,
correct by construction, but **NOT submitted**. Since the prize is dead, the slot would
only buy a ~0.05% cleanup, against ~±20% cuSOLVER run-to-run drift risk on other shapes.
**Recommendation: keep the last ranked slot; don't burn it on 0.05%.** Root
`submission.py` stays exactly `#877941`.

### Quota / cost

Ranked used: **2 of 3** (unchanged; nothing submitted). Modal spend ≈ **~$0.2–0.4**
(1 probe run, image cached). `verify_local.py` 10/10 (repo intact).

### Insight

Two distinct high-batch failure modes now mapped: **few-large** (batch≤4, n≥1024) →
batched cuSOLVER under-occupies → loop wins (exp-004); **many-medium** (batch=640,
n=512) → batched cuSOLVER *saturates* → batched wins (exp-005). The dividing line is
total work/occupancy, not just "batched is bad." `640×512` is at its frontier.

---

## 2026-07-15 — Session 4: small-batch/large-n loop → ranked #877941, BEATS THE LEADER

### Result

**New best `#877941`, ranked geomean ≈ 1746μs** — beats prior best `#877091`
(~~2062μs) by ~15% **and the board leader (~~1924μs) by ~9%.** 17/17 tests pass.
Experiment `004`, **adopted** to root `submission.py`.

### What & why

`torch.linalg.cholesky_ex` sends batch≥2 to `cusolverDnSpotrfBatched` (tuned for
many-small matrices) — terrible for few-but-large. Confirmed on B200 with a 3-way
probe (batched vs per-matrix loop vs streamed):


| shape   | batched  | loop     | streamed |
| ------- | -------- | -------- | -------- |
| 2×4096  | 12580    | **3222** | 3391     |
| 2×2048  | 3900     | 1382     | **1132** |
| 8×2048  | 5612     | 5427     | **3477** |
| 4×1024  | 1646     | 1353     | **634**  |
| 60×1024 | **3233** | 19707    | 5782     |
| 1×4096  | **1546** | 1627     | 2447     |


Streamed was fastest but **popcorn forbids non-default streams** (static source
scan → HTTP 500 "work on another stream ... disqualification"; it even flagged the
literal word "stream" in a comment). So shipped the **loop**: dispatch
`2<=batch<=8 and n>=1024 → per-matrix loop`, keep Triton n=32 + batched cuSOLVER
elsewhere.

### Ranked per-shape (`#877091` → `#877941`)

- **2×4096: 13400 → 3200μs (4.19×)** — the big one.
- **2×2048: 3840 → 1357μs (2.83×)**.
- 4×1024: 1395 → 1297μs (1.08×).
- 8×2048: 5010 → 5370μs (**1.07× WORSE** — loop regresses here on popcorn even
though it tied/won on Modal; Modal↔popcorn fidelity gap on a marginal shape).
- others unchanged.

### Correctness

popcorn test 17/17; Modal verify 26/26 across all families (added in-region cases:
2×1024 spectrum/diagonal, 4×1024 rowscale/tridiagonal, 8×2048, 2×4096 dense/lowrank).
Loop calls the same `potrf` per slice → numerically identical to cuSOLVER.

### Quota / cost

Ranked used: **2 of 3** overall (session 2 `#877091` + this `#877941`). Test id
`#877940`. Modal spend this session ≈ **~$0.5–1**.

### Next steps

1. **Cheap fix for the 8×2048 regression:** restrict the region to `2<=batch<=4`
  (leave 8×2048 on batched, 5010 < 5370). Est. ~~1746→~~1738μs. Costs the last ranked
   submission to confirm; deferred (leader already beaten). Root keeps the exact
   `#877941` code (region `2<=batch<=8`) so it matches a confirmed ranked result.
2. Revisit whether a Triton/CUDA single-large-matrix kernel could shave 8192/16384/
  32768 (compute-bound, low ROI) — unlikely.

---

## 2026-07-15 — Session 3: CUDA n=64/128 attempt → REJECTED (cuSOLVER wins)

### Goal

Beat the board leader (~1924μs) via a **warp/block-per-matrix CUDA kernel for
n=64 and n=128** (experiment `003`), keeping Triton n=32 + cuSOLVER elsewhere.

### Infra unlocked (kept — enables all future CUDA experiments)

- Switched the Modal image to `**nvidia/cuda:13.0.0-devel-ubuntu24.04`** (has
`nvcc`) + `pip install torch numpy ninja` + `.entrypoint([])`. This lets
`torch.utils.cpp_extension.load_inline` compile CUDA on the B200 sandbox
(the plain pip-torch image has no nvcc). torch resolved to 2.13.0+cu130.
- **Gotcha:** without `ninja`, `load_inline` fails `verify_ninja_availability()`
and the try/except silently falls back to cuSOLVER. Caught it because the
n=64/128 residuals were byte-identical to cuSOLVER. Added a
`custom_cuda_loaded=<bool>` + `_CUDA_LOAD_ERROR` diagnostic to `_gpu_runner.py`
/ `submission.py` so a failed compile is never mistaken for a working kernel.

### Result — REJECTED, nothing submitted

CUDA kernel is **correct** (Modal verify 19/19, all families, `custom_cuda_loaded=True`,
residuals ~1000× inside tolerance) but **slower than cuSOLVER** at both shapes:


| shape   | cuSOLVER    | Triton | CUDA block (128-thr, `__syncthreads`) | CUDA warp (32-lane, `__syncwarp`) |
| ------- | ----------- | ------ | ------------------------------------- | --------------------------------- |
| 1024×64 | **135.7μs** | 152    | 205                                   | 214                               |
| 256×128 | **201.5μs** | 429    | 413                                   | 693                               |


Block-per-matrix is sync-bound (3N `__syncthreads`, ~3 blocks/SM at 64KB shared
for n=128); warp-per-matrix has too little per-matrix parallelism + a load-
imbalanced rank-1 update (n=128 → 693μs). Adopting either would **regress** the
geomean, so per the guardrail no ranked submission was made.

- **Ranked quota used this session: 0** (still **2 of 3** remaining overall).
- **Current best unchanged: `#877091`** (exp 002, Triton n=32, ~2062μs).
- Modal spend this session ≈ **~$1–2** (one heavy image build + ~4 short runs).

### Insight

cuSOLVER's batched `potrf` is near-optimal at n=64/128 on B200; a *naive* right-
looking kernel can't beat it. Winning would need a **blocked/recursive** kernel
(panel factorization + batched-GEMM trailing update), likely **tensor-core
(tf32/bf16) Schur updates with FP32 accumulation** (the tolerance has ~1000×
headroom), and multiple matrices per block for occupancy. That's a multi-hour
kernel effort with uncertain payoff — deferred. Details in
`experiments/003-cuda-n64-n128/notes.md`.

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


| shape   | cuSOLVER | Triton (num_warps=1) | verdict         |
| ------- | -------- | -------------------- | --------------- |
| 4096×32 | 137.8μs  | **84→76μs**          | **Triton −39%** |
| 1024×64 | 135.7μs  | 152μs (best cfg)     | cuSOLVER wins   |
| 256×128 | 201.5μs  | 429μs                | cuSOLVER wins   |


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


| shape       | #876988   | #877091    | Δ                     |
| ----------- | --------- | ---------- | --------------------- |
| **4096×32** | **113μs** | **63.7μs** | **−44%** ← the win    |
| 1024×64     | 110       | 110        | —                     |
| 256×128     | 152       | 152        | —                     |
| 64×256      | 276       | 276        | —                     |
| 16×512      | 597       | 600        | —                     |
| 640×512     | 3810      | 3800       | —                     |
| 4×1024      | 1280      | 1395       | +9% (cuSOLVER drift)  |
| 60×1024     | 2900      | 2900       | —                     |
| 2×2048      | 3220      | 3840       | +19% (cuSOLVER drift) |
| 8×2048      | 4910      | 5010       | +2% (drift)           |
| 1×4096      | 1540      | 1534       | —                     |
| 2×4096      | 11400     | 13400      | +18% (cuSOLVER drift) |
| 1×8192      | 6400      | 6410       | —                     |
| 1×16384     | 34200     | 34200      | —                     |
| 1×32768     | 221000    | 221000     | —                     |


**Geomean of this ranked run ≈ 2062μs** (computed from the per-shape means; the
`popcorn submissions list` Score column shows `-`). That is below the recorded
baseline of ~~2080μs, so the definition of done is met — but only marginally in
*absolute* terms, because several **cuSOLVER** shapes (identical code) ran
notably slower this session (`2×2048`, `2×4096`, `4×1024`). That is pure
run-to-run environment drift, not a regression. Same-environment (Modal,
L2-clear, everything but n=32 held fixed) the win is **~~3.9%**: n=32 alone moves
the geomean-monotone score from an equivalent pure-cuSOLVER ~2388μs to 2296μs.

### Findings & insights

- **Confirmed the launch/overhead-bound hypothesis for n=32.** 113→63.7μs from a
single fused Triton launch vs cuSOLVER's batched dispatch across 4096 tiny
matrices. The floor is ~~memory-bound (~~5μs for 32 MB R/W); 63.7μs is still
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

~~9 Modal B200 sandbox runs this session (verify/benchmark, ~40–65s each) ≈ ~10 min
B200 wall ≈ **~~$1–2** Modal spend. popcorn test+leaderboard run on GPU MODE infra
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

Participate in the GPU MODE `[cholesky` leaderboard (776)](https://www.gpumode.com/leaderboard/776?tab=rankings)
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


| Check                                        | Result                                              |
| -------------------------------------------- | --------------------------------------------------- |
| CPU property check (`verify_local.py`)       | **10/10 pass**                                      |
| Modal B200 verify (`modal_verify.py verify`) | **13/13 pass** on `NVIDIA B200`, torch 2.12.0+cu130 |
| popcorn `--mode test`                        | **17/17 pass** on B200                              |
| popcorn `--mode leaderboard` (`#876988`)     | **done, 17/17 pass**, ranked geomean ≈ **2080μs**   |


Reference points on the board at submission time: xuan9938 ~~1924μs, msaroufim ~2041μs.
The raw cuSOLVER baseline (~~2080μs) is already competitive — roughly ~2% behind 2nd.

#### Ranked per-shape times (popcorn, B200)


| shape   | mean    |     | shape   | mean    |
| ------- | ------- | --- | ------- | ------- |
| 4096×32 | 113 µs  |     | 60×1024 | 2.90 ms |
| 1024×64 | 110 µs  |     | 2×2048  | 3.22 ms |
| 256×128 | 152 µs  |     | 8×2048  | 4.91 ms |
| 64×256  | 276 µs  |     | 1×4096  | 1.54 ms |
| 16×512  | 597 µs  |     | 2×4096  | 11.4 ms |
| 640×512 | 3.81 ms |     | 1×8192  | 6.40 ms |
| 4×1024  | 1.28 ms |     | 1×16384 | 34.2 ms |
|         |         |     | 1×32768 | 221 ms  |


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

