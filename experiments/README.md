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
| 002 | Triton n=32 (num_warps=1) + cuSOLVER | ~2062μs | #877091 | **adopted (current best)** |
| 003 | CUDA warp/block-per-matrix n=64/128 (nvcc via load_inline) | 64: 205μs, 128: 413μs (both > cuSOLVER) | — | **rejected** (cuSOLVER wins n=64/128) |
| 004 | small-batch/large-n → per-matrix loop (avoid batched potrf) | ~1746μs ranked (**beats leader ~1924**) | #877941 | superseded by 005 |
| 005 | `640×512` probe (REJECTED — cuSOLVER-saturated) + `8×2048` own-goal fix (loop region 8→4) | ~1744μs ranked (8×2048 5370→5060) | #877956 | superseded by 006 |
| 005 | high-batch mid-n `640×512` probe (batched/loop/streamed/chunk) | 640×512 batched 3955μs = best (streamed 6.5× slower) | — | **rejected** (cuSOLVER-saturated; nothing submitted) |
| 006 | large-n blocked Cholesky, TF32 tensor-core trailing update (batch==1, n≥16384; nb=4096/2048) + isfinite fallback | ~1559μs ranked (16384 1.76×, 32768 2.86×) | #878015 | **adopted (current best)** |
