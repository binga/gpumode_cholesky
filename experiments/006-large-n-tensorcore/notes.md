# exp 006 — large-n blocked Cholesky with a TF32 tensor-core trailing update

## Hypothesis

The three large single-matrix shapes (`1×8192`, `1×16384`, `1×32768`) were
dismissed in prior sessions as "cuSOLVER near speed-of-light, leave it" — but that
was never tested against lower-precision tensor-core math. A **right-looking
BLOCKED Cholesky** where the diagonal block potrf and the panel triangular solve
stay FP32 (stability) but the O(n³) trailing Schur update `A22 -= L21·L21ᵀ` runs on
B200 tensor cores (TF32 / FP16, FP32 accumulate) should beat cuSOLVER's all-FP32
`potrf` while still passing the reconstruction gate `‖A − LLᵀ‖₁ ≤ 20·n·eps·‖A‖₁`
(whose tolerance grows with n). Pure `torch`, default queue only.

## Characterization probe (Modal B200, `precprobe` in `_gpu_runner.py`)

`tol_frac` = residual / allowed (so `<1` passes); `margin` = 1/tol_frac (× inside
tolerance). Control = `batched` (`torch.linalg.cholesky_ex`).

### n = 8192
| variant            | mean μs | speedup | tol_frac | margin |
|--------------------|--------:|--------:|---------:|-------:|
| batched            |  6429.6 |  1.00×  | 3.5e-5   | 28821× |
| blocked_tf32 nb512 |  7796.8 |  0.82×  | 9.5e-3   |   105× |
| blocked_tf32 nb1024|  6811.6 |  0.94×  | 9.6e-3   |   104× |
| blocked_tf32 nb2048|  5994.5 |  1.07×  | 9.5e-3   |   106× |
| blocked_fp16 nb512 |  8628.0 |  0.75×  | 1.1e-2   |    89× |
| blocked_fp16 nb1024|  7495.5 |  0.86×  | 1.1e-2   |    95× |
| blocked_fp16 nb2048|  6858.6 |  0.94×  | 9.3e-3   |   108× |
| blocked_bf16 nb1024|  7495.3 |  0.86×  | 8.4e-2   |    12× |

### n = 16384
| variant            | mean μs | speedup | tol_frac | margin |
|--------------------|--------:|--------:|---------:|-------:|
| batched            | 34213.3 |  1.00×  | 2.7e-5   | 37346× |
| blocked_tf32 nb512 | 27233.9 |  1.26×  | 4.8e-3   |   208× |
| blocked_tf32 nb1024| 21909.6 |  1.56×  | 4.8e-3   |   207× |
| **blocked_tf32 nb2048** | **18958.8** | **1.80×** | 4.8e-3 | 209× |
| blocked_fp16 nb512 | 33278.3 |  1.03×  | 5.7e-3   |   174× |
| blocked_fp16 nb1024| 25738.7 |  1.33×  | 5.6e-3   |   179× |
| blocked_fp16 nb2048| 23186.5 |  1.48×  | 5.3e-3   |   189× |
| blocked_bf16 nb1024| 25736.9 |  1.33×  | 4.5e-2   |    22× |

### n = 32768
| variant            | mean μs | speedup | tol_frac | margin |
|--------------------|--------:|--------:|---------:|-------:|
| batched            |220956.7 |  1.00×  | 1.9e-5   | 52057× |
| blocked_tf32 nb1024|101018.2 |  2.19×  | 2.4e-3   |   416× |
| blocked_tf32 nb2048| 83580.9 |  2.64×  | 2.4e-3   |   417× |
| **blocked_tf32 nb4096** | **75145.0** | **2.94×** | 2.4e-3 | 417× |
| blocked_fp16 nb2048| 99472.8 |  2.22×  | 2.8e-3   |   359× |

## Which precision / nb won

- **TF32 beats FP16 everywhere** on B200: FP16 adds operand-cast + fp16-output
  truncation overhead, and B200's TF32 tensor cores are already several× FP32
  throughput. BF16 has a thin residual margin (12–22×) and is slower than TF32 —
  rejected.
- **Bigger nb wins as n grows** (fewer Python steps + larger, more efficient
  trailing GEMMs; the FP32 diagonal potrf stays a small fraction). Best measured:
  **nb=2048 @ 16384 (1.80×)**, **nb=4096 @ 32768 (2.94×)**.
- **8192 is only ~1.07×** (marginal; the diagonal FP32 potrf is a larger fraction
  at this size) → excluded from the dispatch (regression risk > gain). Stays on
  cuSOLVER.
- Residual margins are comfortable (>100× inside tolerance) for the shipped TF32
  configs — far above the ≥5–10× target.

## Shipped approach

`custom_kernel` dispatch adds: `batch == 1 and n >= 16384 → _blocked_cholesky_tf32`
with `nb = 4096 if n >= 32768 else 2048`. Diagonal potrf (`cholesky_ex`) + panel
`solve_triangular` stay FP32; trailing update runs with `allow_tf32=True`.

**Numerical safety net:** TF32 error can drive a late diagonal block indefinite on
ill-conditioned inputs (`spectrum` cond=5, `lowrank` cond=4), producing NaN/Inf. A
post-factorization `torch.isfinite(L).all()` check falls back to exact FP32
cuSOLVER in that case. The ranked shapes are well-conditioned dense (residual ~10%
of tolerance — never trips the net); the fallback only fires on pathological
families, guaranteeing correctness across every family at negligible cost
(isfinite is memory-bound, <1ms at 32768 vs the ~75ms factorization).

## Correctness across ALL families

- **Modal B200 verify: 37/37 pass** across the small/mid grid (28/28) + large-n
  grid (9/9). Large-n families: 16384 {dense, spectrum, lowrank, rowscale,
  diagonal, tridiagonal} + 32768 {dense, lowrank, tridiagonal}. The `spectrum`
  and `lowrank` large-n cases correctly route through the isfinite fallback
  (residuals match cuSOLVER); dense/rowscale/diagonal/tridiagonal use the fast
  TF32 path.
- **popcorn `--mode test`: 17/17 pass** on GPU MODE B200 (the official test grid
  maxes at n=2048, so the blocked path is not exercised there; the fallback and
  the dense large-n benchmark shapes are validated on Modal).

## Full 15-shape Modal benchmark (`benchmark.json`) — geomean 1730.2 μs

Only the two large shapes changed vs the current best (`#877956`, exp 005); the
other 13 use identical code.

| shape   | baseline (Modal) | exp006 (Modal) | speedup |
|---------|-----------------:|---------------:|--------:|
| 1×16384 |          34213.3 |        19981.6 |  1.71×  |
| 1×32768 |         220956.7 |        78357.1 |  2.82×  |
| 1×8192  |           6416.0 |         6407.2 |  1.00×  (cuSOLVER, unchanged) |

Per-shape ratios on the two changed shapes imply the exp005-equivalent Modal
geomean is ~1922 μs, so exp006 is **~10% better on the Modal metric** (1922 →
1730). The huge single matrices have near-exact Modal↔popcorn fidelity in prior
sessions (8192/16384/32768 matched to <0.5%), so the **projected ranked geomean is
≈ 1570 μs** (1744 × 0.900), clearly below the 1744 μs current best.

## Ranked submission `#878015` (17/17, B200) — geomean ≈ 1559 μs

Supervisor authorized the ranked slot ("3 of 3" was a self-imposed per-run budget,
not a platform cap). popcorn `--mode leaderboard`: **17/17 tests pass**, ranked
benchmark below. Ranked geomean of the 15 per-shape means ≈ **1559 μs** (from
~1744 μs at `#877956`, **−10.6%**) — matches the ~1570 μs Modal projection.

Modal↔popcorn fidelity on the two changed shapes is excellent (within ~2%):

| shape   | prior ranked | `#878015` ranked | speedup | Modal (this exp) |
|---------|-------------:|-----------------:|--------:|-----------------:|
| 1×16384 |     34200 μs |        19400 μs  |  1.76×  |         19982 μs |
| 1×32768 |    221000 μs |        77200 μs  |  2.86×  |         78357 μs |
| 1×8192  |      6400 μs |         6390 μs  |  1.00×  (cuSOLVER) |     6407 μs |

No untouched shape regressed (this was a low-drift ranked run): n=32 61.8 μs,
640×512 3.78 ms, 60×1024 2.89 ms, 8×2048 5.05 ms, 2×4096 3.20 ms, 2×2048 1355 μs,
4×1024 1303 μs — all at/under `#877956`.

## Verdict

**ADOPTED — current best, ranked `#878015` (~1559 μs), beats prior best `#877956`
(~1744 μs) by ~10.6%.** Root `submission.py` carries the exp006 code.

## Modal spend

≈ **$2.5–3** this session. The bulk (~$2) was a single `verify` run that hung after
GPU init and burned the full 1200 s sandbox timeout (transient Modal issue — the
same grid re-ran cleanly in ~45 s afterward, split via a new `--shapes`/progress
filter added to `run_verify`). Probes (3 runs) + verifies + the full benchmark were
~30–60 s each. popcorn test runs on GPU MODE infra (not billed to our Modal).
