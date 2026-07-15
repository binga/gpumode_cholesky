# Experiment 013 — 1×32768 without cuSOLVER (Triton / Blackwell tensor cores)

Status: **REJECTED — no cuSOLVER-free variant beats the exp-012 ranked path; every
one regresses substantially. No ranked submission, no popcorn spend, root
`submission.py` unchanged (byte-for-byte `#878893`).**

Baseline: ranked `#878893` (exp 012), public geomean `1459.321342997556 us`.
Target shape: `batch=1, n=32768` (~76% of the ranked clock). Paired reference on
this harness: exp-012 `1×32768` path ≈ **52139 us** (from exp-012 notes).

## Goal & hard constraint

Make `1×32768` **faster** than exp-012 **with no cuSOLVER anywhere on the
timed/adopted fast path** — no `torch.linalg.cholesky`/`cholesky_ex` for the
diagonal block or any inner step. The exp-012 path is left-looking (nb=4096, 8
diagonal blocks), factors each 4096×4096 diagonal block with
`torch.linalg.cholesky_ex` (**cuSOLVER**), does the panel product in native FP8,
and solves the panel via an explicit triangular inverse + a TF32 matmul.

The **only** cuSOLVER call on that path is the diagonal `potrf`. Removing it means
supplying a cuSOLVER-free `potrf` for the eight 4096×4096 diagonal blocks.

## What was tried

Candidate file: `submission.py` (this folder). It keeps the exp-012 left-looking
structure and only changes the `1×32768` region:

1. **Diagonal potrf, cuSOLVER-free** (approach-ladder step 1, two-level blocked):
   - `triton` — right-looking blocked potrf reusing the validated exp-009 Triton
     kernels (`_diag_factor`/`_panel_solve`/`_lower_schur`), bk∈{64,128,256}, TF32
     trailing (`_triton_blocked_potrf`).
   - `cublas32_fp32` / `cublas32_tf32` — right-looking blocked potrf with a Triton
     n=32 diagonal base, cuBLAS `solve_triangular` panel, and a cuBLAS matmul
     trailing update in FP32 or TF32 (`_blocked_cublas_potrf`).
2. **FP8 panel solve** (approach-ladder step 2 spirit): push the large TF32
   `panel @ inverse_transpose` product onto native FP8 tensor cores (`_PANEL_SOLVE_FP8`).

A paired same-process B200 probe (`nocusolverprobe` in `scripts/_gpu_runner.py`)
was added. It loads `baseline-exp012.py` as the baseline (added to the Modal
`IMAGE` in `scripts/modal_verify.py`), micro-benchmarks the diagonal `potrf` on a
single 4096 block, and compares the candidate vs exp-012 `_left_looking_cholesky_32768`
at the cheap n=8192/16384 proxies (both generic in n), reporting reconstruction
margins and a backend counter proving the cuSOLVER-free path ran with zero
fallbacks.

## Causal evidence (B200, paired, same process)

### Diagonal potrf micro-benchmark — single 4096×4096 dense SPD block

| method | mean | vs cuSOLVER | recon margin |
|---|---:|---:|---:|
| **cuSOLVER `cholesky_ex`** (the term to replace) | **1579 us** | 1.00× | 59688× |
| Triton blocked bk=64 (fastest free) | 5794 us | **3.67× slower** | 19.7× (compounds to NaN) |
| Triton blocked bk=128 | 16109 us | 10.2× slower | 19.7× |
| Triton blocked bk=256 | — | OOM (shared memory) | — |
| cuBLAS blocked bk=32, TF32 trailing | 13577 us | 8.6× slower | 53× |
| cuBLAS blocked bk=32, FP32 trailing (accurate) | 13261 us | 8.4× slower | 17616× |

### Full left-looking path — candidate vs exp-012 cuSOLVER path (paired)

| config | n | candidate | exp-012 baseline | speedup | candidate correctness |
|---|---:|---:|---:|---:|---|
| fast (Triton bk64 diag, FP8 panel) | 8192  | 14182 us | 5811 us  | **0.410×** | **FAIL — NaN/Inf** |
| fast (Triton bk64 diag, FP8 panel) | 16384 | 33212 us | 16536 us | **0.498×** | **FAIL — NaN/Inf** |
| accurate (cuBLAS32 FP32 diag, TF32 panel) | 8192  | 26537 us | 5809 us  | **0.219×** | PASS (margin 71.9×) |
| accurate (cuBLAS32 FP32 diag, TF32 panel) | 16384 | 54648 us | 16532 us | **0.303×** | PASS (margin 3.4×) |

Backend counter (proves this is a real cuSOLVER-free measurement, not a fallback):
`nocusolver_32768_hits=48`, `nocusolver_potrf_calls=171`, `left_32768_error=null`,
`fallbacks=0`. The fast path genuinely executed the cuSOLVER-free code.

## Why it fails (root cause)

cuSOLVER's `potrf` is a single, highly-optimized fused kernel: ~1.6 ms for a 4096
block, essentially speed-of-light, and 59688× inside tolerance. Every
cuSOLVER-free scheme available from PyTorch/Triton is **Python-orchestrated** and
either single-CTA-bound (the Triton unblocked base kernel factors a tile with one
CTA over a sequential, unrolled column loop) or launch-bound (blocked cuBLAS with
~64–128 sequential tiny steps per 4096 block). All land 3.7–10× slower.

The eight diagonal factorizations are ~24% of the 52 ms baseline (~12.6 ms with
cuSOLVER). Replacing them cuSOLVER-free adds **+33 ms** (fastest, but NaN) to
**+92 ms** (accurate). The only FP8 lever that could pay it back is converting the
TF32 `panel @ inverse_transpose` / diagonal-trailing products (~a few ms of TF32
work) to FP8 — worth only ~2–5 ms, and pushing them to FP8 on top of the already
FP8 panel product drives the reconstruction past tolerance (NaN in the fast
config). There is no operating point where cuSOLVER-free is both correct and
faster: **fast ⇒ NaN**, **accurate ⇒ 3–5× slower**.

Even a *hypothetical perfect* cuSOLVER-free `potrf` exactly matching cuSOLVER
(1.6 ms) would leave the rest of the path unchanged and yield ≈0% speedup — the
required ≥1.10× must come from FP8, which is a few ms at most and costs accuracy.
The only theoretical route to a win is a single-launch fused CUTLASS/`tcgen05`
`potrf` that matches NVIDIA's tuned routine (journal candidate #3): multi-hour,
high-risk, and unlikely to beat cuSOLVER — not justified for a term that, even if
perfected, does not move the geomean.

## Verdict

**REJECTED.** Gate 1 (paired `1×32768` ≥ 1.10× vs `#878893`, passing every
family) fails decisively on the cheap 8192/16384 proxies and the 4096 diagonal
micro-benchmark. Per the bounded fallback ladder, the experiment stops here:

- No expensive n=32768 probe was run (the cheap proxies + micro-benchmark already
  prove the regression; a 32768 probe would only confirm it at ~50–90 ms/iter).
- No family sweep or full 15-shape grid (nothing to promote).
- No popcorn test and **no ranked submission** (ranked slot preserved).
- Root `submission.py` remains byte-for-byte the exp-012 winner `#878893`.

This closes the **Triton / Custom-CUDA "no-cuSOLVER" column for 1×32768** with
negative evidence: cuSOLVER's diagonal `potrf` is not removable from this path
without a large net regression using PyTorch/Triton primitives.

## Cost

Modal B200: 3 `nocusolverprobe` runs (~65–77 s each incl. image build) ≈ **~3–4
min B200 wall ≈ ~$2–3**. No cuSOLVER-free 32768 probe, no full grid, no popcorn
quota used. Ranked submissions: **0**.

## Artifacts

- `submission.py` — the cuSOLVER-free candidate (rejected), with tunables
  `_DIAG_METHOD` / `_DIAG_POTRF_BK` / `_PANEL_SOLVE_FP8` and the diagnostics
  `_NOCUSOLVER_32768_HITS` / `_NOCUSOLVER_POTRF_CALLS`.
- `baseline-exp012.py` — exact exp-012 ranked source used as the paired baseline.
- `potrf-microbench.json` — 4096-block diagonal potrf comparison.
- `paired-32768.json` — full paired probe (both configs at 8192/16384) + micro-bench
  + backend counter. (Named per the deliverable convention; the timed proxies are
  8192/16384, not 32768 — the expensive 32768 probe was intentionally not run.)
- `paired-dev.json` — first dev iteration (Triton bk sweep).
- Harness additions kept in `scripts/`: `nocusolverprobe` mode + exp-012 baseline
  wired into the Modal image; `--submission` flag on `modal_verify.py`.

## Possible future directions (not pursued)

- A single-launch fused Blackwell `potrf` (CUTLASS 3.x / `tcgen05.mma`, TMA,
  2-SM MMA) is the only credible way to make a cuSOLVER-free diagonal competitive.
  High effort, high risk, and even success only unlocks the small FP8 lever — poor
  ROI versus other shapes.
