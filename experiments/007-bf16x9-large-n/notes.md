# exp 007 — BF16x9 FP32-emulated trailing update for large single-matrix shapes

**Verdict: REJECTED.** Nothing submitted. Root `submission.py` stays `#878015`
(~1559μs). This closes the *BF16x9 FP32-emu* column for the large-n shapes.

## Hypothesis

cuBLAS 12.9+/CUDA 13 can emulate a true FP32 GEMM as 9 BF16 products on Blackwell
BF16 tensor cores (~3–4× native FP32, ≥FP32 accuracy). If that engages through
PyTorch on the B200, replacing the O(n³) trailing Schur update `A22 -= L21·L21ᵀ`
of the blocked Cholesky with BF16x9-emulated FP32 might (a) finally give `1×8192`
a real win (it stayed on cuSOLVER because TF32 was only 1.07× and precision-
marginal), and (b) beat the shipped TF32 trailing update on `1×16384`/`1×32768`.

## Which API path worked (and how engagement was confirmed)

**Engagement is via two cuBLAS env vars set before `torch` import**, on PyTorch's
default BLAS path:

```
CUBLAS_EMULATE_SINGLE_PRECISION=1     # the MASTER switch — this is what engages it
CUBLAS_FP32_EMULATED_BF16X9_MATH=1    # pins the algorithm to BF16x9
```

- `CUBLAS_FP32_EMULATED_BF16X9_MATH=1` **alone did NOT engage** (measured — see
  below). It only takes effect together with `CUBLAS_EMULATE_SINGLE_PRECISION=1`.
- `torch.backends.cuda.matmul.fp32_precision` exposes only `ieee`/`tf32` (reported
  `none` on this build) — **no BF16x9 value**, so the PyTorch knob cannot reach it.
- `torch.backends.cuda.preferred_blas_library("cublaslt")` / `TORCH_BLAS_PREFER_CUBLASLT=1`
  is **not needed and is harmful**: forcing cuBLASLt was *slower* (16384 FP32
  matmul 56.6ms default → 75.6ms cuBLASLt). The default heuristic already picks the
  emulated path and is fastest. So the shipped-style config touches only the two
  env vars and leaves BLAS selection on default.

**Engagement confirmation — standalone FP32 `A@B` (8192², tf32 disabled):**

| config | time | engaged? |
|---|---|---|
| emu OFF (native FP32) | 16712 μs | — (baseline) |
| `CUBLAS_FP32_EMULATED_BF16X9_MATH=1` only | 16729 μs | **NO** (identical) |
| + `CUBLAS_EMULATE_SINGLE_PRECISION=1` (default BLAS) | **6333 μs** | **YES (2.64×)** |

At the blocked-Cholesky level, the *same* `blocked_fp32` variant sped up between
the emu-off and emu-on passes, independently confirming engagement:
`8192 nb4096` 8676 → 6733 μs (1.29×); `16384 nb4096` 50782 → 29990 μs (1.69×).

## Probe table (Modal B200, precprobe; dense unless noted)

`margin_x` = tolerance/residual (higher = more accurate; gate passes if >1).
`blocked_fp32` runs are **BF16x9-emulated** (emu ON). `bf16x9split` = manual
3-way BF16 split with native FP32 GEMMs (genuine-accuracy proxy, emu ON).

### n = 8192  (current path: cuSOLVER `batched` ≈ 6410 μs)

| variant | mean μs | speedup vs batched | margin_x | note |
|---|---|---|---|---|
| batched (cuSOLVER) | 6410 | 1.00× | 38,245 | current ship |
| blocked_tf32 nb2048 | 5978–6233 | 1.03–1.07× | ~105 | **FAILs lowrank (NaN)** |
| blocked_fp32 (bf16x9) nb1024 | 8557 | 0.75× | 91,323 | |
| blocked_fp32 (bf16x9) nb2048 | 7528–7911 | 0.82–0.85× | 91,032 | |
| **blocked_fp32 (bf16x9) nb4096** | **6733** | **0.95×** | 65,874 | best bf16x9 → still slower |
| bf16x9split nb2048 | 18,025–18,398 | 0.35× | 65,758 | |

### n = 16384  (current path: blocked-**TF32** ≈ 19,400 μs ranked; here TF32 ≈ 21,533 μs)

| variant | mean μs | speedup vs batched | margin_x | note |
|---|---|---|---|---|
| batched (cuSOLVER) | 34,449 | 1.00× | 67,886 | |
| **blocked_tf32 nb2048** | **21,533** | **1.60×** | ~208 | **current ship** (FAILs lowrank NaN → isfinite fallback) |
| blocked_fp32 (bf16x9) nb1024 | 35,101 | 0.98× | 134,105 | |
| blocked_fp32 (bf16x9) nb2048 | 31,454 | 1.10× | 138,963 | |
| blocked_fp32 (bf16x9) nb4096 | 29,990 | 1.15× | 139,099 | best bf16x9 → far behind TF32 |
| bf16x9split nb2048 | 126,417 | 0.27× | 126,095 | |

### n = 32768 — NOT measured (deliberately)

The current ship is blocked-TF32 at ~77,200 μs (2.86× vs cuSOLVER). BF16x9 was
only 1.15× vs cuSOLVER at 16384 (vs TF32's 1.60×); the gap widens with n, and the
throughput argument below is decisive. Confirming a loss would cost ~220 ms/iter ×
many variants for no decision value, so 32768 was held out per the cost guardrail.

## Correctness across families

Every BF16x9 (`blocked_fp32` emu-on) variant **PASSES `check_implementation`** on
all probed families — dense, **spectrum (cond=5)**, and **lowrank (cond=4)** at
8192 and 16384 — with margins 31,000–139,000× inside tolerance (≈ FP32-equivalent,
~600–900× more accurate than TF32's ~90–210×). The `bf16x9split` genuine-accuracy
proxy tracks the emulated residual closely (e.g. 8192 dense 78,736× vs 92,982×),
so the global-emulation "known pitfall" (checker reconstructs with a matmul) does
**not** flatter the result — BF16x9 reconstruction ≈ genuine FP32.

Notably BF16x9 is **more robust than TF32**: TF32 produced NaN/Inf on `lowrank`
at both 8192 and 16384 (the reason exp 006 needed the isfinite→cuSOLVER fallback),
whereas BF16x9 stayed finite and accurate. This robustness is real but irrelevant
to the ranked score (ranked inputs are well-conditioned dense; the fallback is
near-free), and does not offset the speed deficit.

## Why it loses (the throughput math)

BF16x9 emulates one FP32 product with ~6–9 BF16 tensor-core products. TF32 uses
**one** tensor-core product. On the B200, BF16 throughput is only ~2× TF32, so
BF16x9 ≈ (2×TF32)/6 ≈ **~3× slower than TF32** per FP32-equivalent GEMM. Hence
the ordering **TF32 > BF16x9 > native FP32** in speed (and BF16x9 > TF32 in
accuracy). NVIDIA's "3–4× faster than native FP32" is real, but native FP32
(no tensor cores) is not the bar here — **TF32 tensor cores already are**, and
BF16x9 can't beat them. Separately, at 8192 the FP32 diagonal `potrf` + FP32 panel
TRSM are a large fixed fraction (the emulation only speeds the trailing GEMM), so
even 2.6× on the GEMM can't beat cuSOLVER's single fused call.

## Per-shape decision vs `#878015`

| shape | current path (#878015) | best BF16x9 | Δ | decision |
|---|---|---|---|---|
| 1×8192 | cuSOLVER ~6390 μs | 6733 μs (nb4096) | **+5% slower** | keep cuSOLVER |
| 1×16384 | blocked-TF32 ~19,400 μs | 29,990 μs (nb4096) | **+55% slower** | keep TF32 |
| 1×32768 | blocked-TF32 ~77,200 μs | not measured (would lose) | — | keep TF32 |

No shape measurably wins → **REJECT**, no ranked submit (per the guardrail:
"Ship only the shapes that actually won"; a rejected experiment is a valid outcome).

## Harness changes (kept — useful infra)

- `scripts/_gpu_runner.py`: sets the two cuBLAS emulation env vars before `torch`
  import when invoked with an `emu` token; `_blocked_cholesky` gained `fp32`
  (native/emulated) and `bf16x9` (manual-split) trailing modes; `precprobe` adds
  `blocked_fp32_nb{1024,2048,4096}` + `blocked_bf16x9split_nb2048` variants, an
  engagement micro-bench, and ill-conditioned family specs; new `emuprobe` mode
  sweeps BLAS backends to locate the engaging config.
- `scripts/modal_verify.py`: `--emu` flag + `emuprobe` mode.

## Modal spend

≈ **$3–5**. Five B200 runs: precprobe emu-off (~52s), precprobe emu-on with the
wrong (non-engaging) single-var config (~55s), a first `emuprobe` that hung after
GPU init and was killed (~9.7 min wall — the bulk of the cost, same transient
noted in Session 6), the `emuprobe` retry that found the engaging config (~40s),
and the final precprobe emu-on with the correct config (~44s). popcorn quota: 0
used.

## Remaining ideas (for the BF16x9 / large-n column)

- **FP8 / MXFP8 trailing update + iterative refinement** (tracker candidate #2):
  FP8 tensor cores are ~2× BF16 on Blackwell; a lossy FP8 trailing update plus
  1–2 SPD iterative-refinement steps could beat TF32 on 32768 where the loose
  n-scaled gate has the most headroom. Higher effort/risk; the real remaining
  lever if any.
- **CUTLASS 3.x Blackwell fused kernel** (tcgen05.mma/TMA) to fuse panel+trailing
  and avoid per-step launch + global-memory materialization. High effort.
- BF16x9's accuracy/robustness edge is genuine but not monetizable on the ranked
  (dense) grid; not worth pursuing for score.
