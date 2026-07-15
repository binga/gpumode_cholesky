# GPU MODE `cholesky` submission

Batched dense Cholesky factorization for the GPU MODE
[`cholesky` leaderboard (776)](https://www.gpumode.com/leaderboard/776?tab=rankings), target GPU **B200**.

Input `A`: `batch x n x n` float32 CUDA tensor, SPD up to FP32 roundoff.
Output `L`: lower-triangular float32 with positive diagonal, `A = L @ L.T`.
Ranking: geometric mean of runtime across 15 benchmark shapes.

## Layout

- `submission.py` — the entry point (`custom_kernel` + `#!POPCORN` directives).
- `reference/` — vendored, read-only harness from `gpu-mode/reference-kernels`
  (`task.py`, `reference.py`, `eval.py`, `utils.py`). The checker here is the spec.
- `scripts/verify_local.py` — zero-cost CPU property check (no GPU / no cost).
- `scripts/modal_verify.py` — real **B200** verification/benchmark via a Modal sandbox.
- `scripts/_gpu_runner.py` — runs inside the Modal sandbox (do not run locally).
- `results/` — captured outputs (`baseline-benchmark.json` committed).

## Verification tiers

This machine has no local NVIDIA GPU, so verification is layered:

1. **CPU property check (free):**
   ```bash
   python scripts/verify_local.py
   ```
2. **Real B200 via Modal (billed per second):** requires `modal` installed + authed.
   ```bash
   uv run --with modal python scripts/modal_verify.py            # correctness
   uv run --with modal python scripts/modal_verify.py benchmark --json results/baseline-benchmark.json
   ```

## Submit (via popcorn CLI)

Directives are embedded in `submission.py`, so no flags needed:

```bash
popcorn register                                   # one-time auth
popcorn submit --mode test --no-tui submission.py  # remote correctness on B200
popcorn submit --mode leaderboard --no-tui submission.py  # ranked
popcorn submissions                                # view your entries
```

## Status

- Baseline: `torch.linalg.cholesky_ex` (cuSOLVER). Correct across all input families.
- CPU property check: **10/10 pass**.
- Real B200 verify (Modal sandbox): **13/13 pass** (torch 2.12+cu130 on `NVIDIA B200`). The default torch wheel already ships Blackwell/sm_100 kernels — no cu128 pin needed.
- **Ranked submission `#876988`** (cuSOLVER baseline): `done`, 17/17 on B200, geomean ≈ **2080μs**.
- **Ranked submission `#877091`** (custom Triton kernel for `n=32`): `done`, 17/17 on B200. The `4096×32` shape dropped **113μs → 63.7μs (−44%)**; all other shapes stay on cuSOLVER. Geomean ≈ **2062μs**.
- **Ranked submission `#877941`** (exp 004 — small-batch/large-n per-matrix loop): `done`, 17/17 on B200. Avoids the slow batched `cusolverDnSpotrfBatched` path for few-but-large matrices: **2×4096 13400μs→3200μs (4.19×)**, 2×2048 3840→1357 (2.83×), 4×1024 1395→1297. Ranked geomean ≈ **1746μs — beats the board leader (~1924μs) by ~9%** and the prior best by ~15%. (Known minor own-goal: 8×2048 5010→5370.) See `journal.md` Session 4 and `experiments/004-small-batch-large-n/`.
- **Ranked submission `#877956`** (exp 005): `done`, 17/17 on B200. Fixes the exp-004 `8×2048` own-goal by trimming the loop region to `2<=batch<=4` so `8×2048` returns to batched cuSOLVER: **5370→5060μs (−5.8%)**; all other shapes unchanged. Ranked geomean ≈ **1744μs**. exp 005's primary target, `640×512`, was probed and **rejected** (cuSOLVER-batched-saturated — max-concurrency queues 6.5× slower than `batched`; no default-queue path beats it). See `journal.md` Session 5 and `experiments/005-highbatch-mid-n/`.
- **Ranked submission `#878015`** (exp 006 — current best): `done`, 17/17 on B200. Blocked right-looking Cholesky for large single matrices (`batch==1, n>=16384`): FP32 diagonal potrf + FP32 panel solve, **O(n³) trailing Schur update on TF32 tensor cores** (FP32 accumulate), with an `isfinite` fallback to cuSOLVER for ill-conditioned inputs. **1×16384 34200→19400μs (1.76×)**, **1×32768 221000→77200μs (2.86×)**; `1×8192` (only ~1.07×) stays on cuSOLVER; all other shapes unchanged (no regressions). Ranked geomean ≈ **1559μs (−10.6% vs `#877956`)**. TF32 beat FP16/BF16 in the probe; nb=4096 for n≥32768 else 2048. See `journal.md` Session 6 and `experiments/006-large-n-tensorcore/`.

### Baseline B200 timings (Modal harness, `results/baseline-benchmark.json`)

cuSOLVER baseline, geomean of per-shape means = **2402.9μs** across 15 shapes.
Note: our harness (warmup 3, 10 iters, no L2-cache clear) differs from popcorn's
official method, so absolute numbers are not directly comparable to the
leaderboard — use them for *relative* per-shape targeting.

| shape | mean μs | | shape | mean μs |
|---|---|---|---|---|
| 4096×32 | 141 | | 60×1024 | 3214 |
| 1024×64 | 155 | | 2×2048 | 3848 |
| 256×128 | 202 | | 8×2048 | 5559 |
| 64×256 | 368 | | 1×4096 | 1542 |
| 16×512 | 766 | | 2×4096 | 12473 |
| 640×512 | 3941 | | 1×8192 | 6416 |
| 4×1024 | 1634 | | 1×16384 | 34243 |
|  |  | | 1×32768 | 220811 |

**Optimization targets (deferred work), by ROI for the geomean:**
- **Highest ROI — small-`n` / high-batch** (`n ∈ {32,64,128}`, 141–202μs): these are launch/overhead-bound, not compute-bound (a 32×32 factorization is trivial). Custom batched kernels (cf. `triton_cholesky32.py`) can cut these to tens of μs — this is the leaders' trick.
- **Medium ROI — high-batch mid-size** (`640×512`, `8×2048`, `2×4096`): batch-parallelism/occupancy tuning.
- **DONE (exp 006) — large single matrices** (`n ≥ 16384`, esp. `32768²`): a blocked Cholesky with a TF32 tensor-core trailing update beats cuSOLVER's all-FP32 `potrf` (16384 1.76×, 32768 2.86× ranked). `1×8192` was only ~1.07× so it stays on cuSOLVER. The loose reconstruction gate (`20·n·eps·‖A‖₁`, tolerance grows with n) leaves ample headroom (residuals >100× inside tolerance).
