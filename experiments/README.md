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
| 012 | left-looking active-panel paths at 1×16384 (TF32) and 1×32768 (native FP8/FP32 accumulate) | 1459.321μs public / 1448.377μs secret; paired 1.150×/1.373× | #878893 | **adopted (current best)** |
