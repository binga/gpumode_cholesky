# Experiment 004 — small-batch / large-n: per-matrix loop instead of batched cuSOLVER

**Status: ADOPTED — new current best. Beats the board leader.**
Ranked `#877941`, geomean ≈ **1746μs** (beats prior best `#877091` ~2062μs by ~15%
and the board leader ~1924μs by ~9%).

## Hypothesis
`torch.linalg.cholesky[_ex]` routes batch≥2 to `cusolverDnSpotrfBatched` (tuned for
many-small matrices), which is pathologically slow for few-but-large matrices, while
batch=1 uses the fast blocked single-matrix `potrf`. Confirmed: `2×4096` was ~13,130μs
vs `1×4096` ~1,537μs (≈8.5× for 2 matrices).

## 3-way probe (Modal B200, L2-clear) — hypothesis CONFIRMED
| shape | batched | loop | streamed | best vs batched |
|---|---|---|---|---|
| 2×4096 | 12580 | **3222** | 3391 | loop 3.90× |
| 2×2048 | 3900 | 1382 | **1132** | streamed 3.44× |
| 8×2048 | 5612 | 5427 | **3477** | streamed 1.61× |
| 4×1024 | 1646 | 1353 | **634**  | streamed 2.59× |
| 60×1024| **3233**| 19707| 5782 | batched (keep) |
| 1×4096 | **1546**| 1627 | 2447 | batched (keep) |

Streamed (each matrix on its own `cuda.Stream`) was fastest, but **popcorn forbids
work on non-default streams** — `--mode test` returns HTTP 500 *"Your code contains
work on another stream ... may result in disqualification."* (It's a static source
scan: it even rejected a stream-free loop that merely mentioned the word "stream" in
a comment. Scrubbing the keyword + all stream usage fixed it.) So the shippable path
is the plain sequential **loop**, which still beats batched everywhere in-region.

## Change
Dispatch branch in `custom_kernel`: **`2 <= batch <= 8 and n >= 1024` → per-matrix
loop** (`torch.stack([cholesky_ex(data[i]).L ...])`), pure default-stream ops. Keep
Triton n=32 (exp 002); keep batched cuSOLVER everywhere else (batch=1 large-n and
high-batch, e.g. 60×1024, where batched is best). Threshold is data-driven: the four
in-region grid shapes (4×1024, 2×2048, 8×2048, 2×4096) all beat batched in the probe;
batch=1 and batch≥60 excluded because batched wins there.

## Correctness (all clean)
- popcorn `--mode test`: **17/17** on B200 (incl. 2×1024 dense/lowrank via loop).
- Modal verify: **26/26** across all families, incl. added in-region cases:
  2×1024 spectrum/diagonal, 4×1024 rowscale/tridiagonal, 8×2048 dense, 2×4096
  dense/lowrank. Residuals ~1e-3 (tolerance 20). The loop calls the same `potrf`
  per slice, so it is numerically identical to per-matrix cuSOLVER.

## Ranked per-shape (`#877091` → `#877941`)
| shape | #877091 | #877941 | Δ |
|---|---|---|---|
| **2×4096** | 13400μs | **3200μs** | **4.19×** |
| **2×2048** | 3840μs  | **1357μs** | **2.83×** |
| **4×1024** | 1395μs  | **1297μs** | 1.08× |
| 8×2048 | 5010μs | 5370μs | **1.07× WORSE** (see below) |
| 4096×32 | 63.7μs | 62.0μs | 1.03× |
| all other 10 shapes | — | unchanged | — |

Ranked geomean **≈1746μs** (computed from per-shape means; `submissions list` Score
shows `-`). Net huge win despite the one regression.

## Known regression + cheap fix (next experiment)
`8×2048`: the loop (5370μs) is **slower than batched (5010μs)** on popcorn — the
opposite of the Modal probe (loop 5427 < batched 5612). This is the Modal-vs-popcorn
fidelity gap on a shape where loop was already only ~1.03–1.05× ahead on Modal. The
fix is trivial: **restrict the region to `2 <= batch <= 4`** (or `batch <= 4`), leaving
8×2048 on batched. Estimated additional gain ~0.5% geomean (~1746→~1738μs). Not done
here to preserve ranked quota (already beat the leader; exp-004 brief capped at 1
ranked submit, which is used). This experiment's `submission.py` is kept EXACTLY as
submitted for `#877941` (region `2<=batch<=8`) so root == a confirmed ranked result.

## Ranked submissions
- Test (ok): `#877940`. Leaderboard (ranked): **`#877941`** (`done`, 17/17).
- Two earlier `--mode test` attempts were rejected by the stream static-scan (the
  streamed variant, then a loop variant whose comment still said "stream"); neither
  consumed leaderboard quota.
- Quota: **2 of 3** ranked submissions now used overall (session 2 `#877091` + this).

## Modal spend
~5 Modal B200 runs (1 probe, verify, 2 full-grid benchmarks) ≈ ~4 min B200 wall ≈
**~$0.5–1** (image already cached from exp 003).

## Verdict
**ADOPTED.** Root `submission.py` = this file. New current best, beats the leader.
