# Experiments

Each optimization attempt is a numbered folder `NNN-slug/`. The repo-root
`submission.py` always holds the **current best** (the last *adopted* experiment).
Each experiment folder is a self-contained snapshot so results stay reproducible
and auditable.

## Convention

`experiments/NNN-slug/` contains:
- `submission.py` — the exact submission used for this experiment.
- `notes.md` — hypothesis, what changed, Modal verify/benchmark results (geomean +
  per-shape), correctness status across all families, popcorn submission id (if
  submitted), verdict (**adopted** / **rejected**), and approx Modal spend.
- `benchmark.json` — the Modal per-shape benchmark for this experiment (optional).

## Workflow (one commit per experiment)

1. Create the next `experiments/NNN-slug/` folder.
2. Iterate the inner loop (CPU check → Modal verify → Modal benchmark).
3. Write `notes.md` with results and verdict.
4. If adopted: copy the winning `submission.py` to repo root.
5. If it beats the last ranked result on Modal: `popcorn submit --mode test` then
   `--mode leaderboard`; record the id.
6. **`git commit`** the experiment folder (+ root `submission.py`/`journal.md`/`README.md`
   if changed) with a message like `exp NNN: <slug> — <one-line result>`.

## Log

| # | approach | geomean | ranked id | verdict |
|---|---|---|---|---|
| 001 | cuSOLVER baseline | ~2080μs | #876988 | superseded by 002 |
| 002 | Triton n=32 (num_warps=1) + cuSOLVER | ~2062μs | #877091 | superseded |
| 003 | CUDA warp/block-per-matrix n=64/128 (nvcc via load_inline) | 64: 205μs, 128: 413μs (both > cuSOLVER) | — | **rejected** (cuSOLVER wins n=64/128) |
| 004 | small-batch/large-n → per-matrix loop (avoid batched potrf) | ~1746μs ranked (**beats leader ~1924**) | #877941 | superseded by 005 |
| 005 | `640×512` probe (REJECTED — cuSOLVER-saturated) + `8×2048` own-goal fix (loop region 8→4) | ~1744μs ranked (8×2048 5370→5060) | #877956 | superseded by 006 |
| 005 | high-batch mid-n `640×512` probe (batched/loop/streamed/chunk) | 640×512 batched 3955μs = best (streamed 6.5× slower) | — | **rejected** (cuSOLVER-saturated; nothing submitted) |
| 006 | large-n blocked Cholesky, TF32 tensor-core trailing update (batch==1, n≥16384; nb=4096/2048) + isfinite fallback | ~1559μs ranked (16384 1.76×, 32768 2.86×) | #878015 | superseded by 008 |
| 007 | BF16x9 FP32-emulated trailing update (large-n) — engaged via `CUBLAS_EMULATE_SINGLE_PRECISION=1`+`CUBLAS_FP32_EMULATED_BF16X9_MATH=1` | 8192 0.95× vs cuSOLVER; 16384 bf16x9 1.15× vs TF32's 1.60× | — | **rejected** (engages + ≈FP32-accurate but slower than TF32/cuSOLVER) |
| 008 | fuse TF32 Schur product + subtraction into in-place `addmm_` on trailing view | 1542.914μs ranked; paired 16384 1.087×, 32768 1.080× vs 006 | #878108 | superseded by 009 |
| 009 | combine exact-shape graph frontiers at 256×128/16×512 with Triton FP32/TF32 8×2048 | 1500.704μs public / 1501.440μs secret; paired 1.211×/1.280×/1.622× | #878273 | superseded by 012 |
| 012 | left-looking active-panel paths at 1×16384 (TF32) and 1×32768 (native FP8/FP32 accumulate) | 1459.321μs public / 1448.377μs secret; paired 1.150×/1.373× | #878893 | superseded by 014 |
| 013 | 1×32768 cuSOLVER-free path — Triton/cuBLAS two-level diagonal potrf + FP8 panel | 8192/16384 paired 0.22–0.50× vs exp-012; diag potrf 3.7–8.4× slower than cuSOLVER | — | **rejected** (cuSOLVER diagonal not removable without large regression) |
| 014 | fused tiled dual-amax + joint E4M3 scale/cast for 1×32768 panel products | 1447.259μs public / 1443.226μs secret; dedicated target 1.084×; Modal grid 1.0055× | #880770 | superseded by 015 |
| 015 | two-level blocked tensor-core factorization (rank-2 1-warp diag potrf+inverse, tf32x3 panels, rank-128 tf32 trailing, per-shape CUDA graphs) for six mid shapes + graphed 1024×64 + manual-capture 256×128 | 1262.934μs public / 1270.707μs secret; paired 1.09–1.99× on 7 shapes; Modal grid 1.186× | #881981 | superseded by 016/017 |
| 016a | 1×8192 left-looking TF32 (off pure cuSOLVER) + recursive GEMM triangular inversion at 16384/32768; FP8-shadow/fixed-scale stack rejected | paired 1.138×/1.055×/1.028× | — (integrated in 017) | **adopted via 017** |
| 016b | rank-2 one-warp n=32 kernel; graphed-4096×32 and small-n split32 rejected | 4096×32 paired 1.591× (62.8→39.5μs) | — (integrated in 017) | **adopted via 017** |
| 017 | rank-4 pivot micro + first-touch eager mode (640×512/60×1024) + mirror-zero stores; CUDA/queue-API micro abandoned by owner directive (no scanner workarounds) | 1205.336μs public / 1197.791μs secret; paired 1.05–1.26×; single-module grid 1195.7μs (1.109×) | #882706 | superseded by 019 |
| 019 | FP16 trailing inputs with FP32 accumulation on five split32 shapes + reciprocal inverse-row solve | 1122.570μs public / 1128.511μs secret; Modal grid 1.0093× | #882825 | superseded by 020 |
| 020 | 64×64 panel-inner subtiling at 4×1024 and 8×2048; 255→114 registers and 408→0 stack bytes | 1120.214μs public / 1126.463μs secret; paired 1.089×/1.055×; Modal grid 1.00995× | #882927 | **adopted (current best)** |
