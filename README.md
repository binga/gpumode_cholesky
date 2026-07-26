# GPU MODE `cholesky` submission

Batched dense Cholesky factorization for the GPU MODE
[`cholesky` leaderboard (776)](https://www.gpumode.com/leaderboard/776?tab=rankings), target GPU **B200**.

Input `A`: `batch x n x n` float32 CUDA tensor, SPD up to FP32 roundoff.
Output `L`: lower-triangular float32 with positive diagonal, `A = L @ L.T`.
Ranking: geometric mean of runtime across 15 benchmark shapes.

## Layout

- `submission.py` — the entry point (`custom_kernel` + `#!POPCORN` directives).
- `program.md` — the `set_goal`-triggerable optimization and leaderboard workflow.
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

## Modal source-upload authorization

The repository owner explicitly authorizes this workflow to upload the files
needed for verification to Modal, including `submission.py`, the vendored
`reference/` harness, `scripts/_gpu_runner.py`, and experiment candidate files.
This permission covers B200 correctness checks and benchmarks run by
`scripts/modal_verify.py`. Credentials and unrelated workspace files remain out
of scope and must never be embedded in an image or committed.

## Submit (via popcorn CLI)

Directives are embedded in `submission.py`, so no flags needed:

```bash
popcorn register                                   # one-time auth
popcorn submit --mode test --no-tui submission.py  # remote correctness on B200
popcorn submit --mode leaderboard --no-tui submission.py  # ranked
popcorn submissions                                # view your entries
```

## Status

The current ranked winner, its exact source hash, and recent history live in
**`docs/STATUS.md`** — one maintained place, so it cannot drift.

| I need to know… | Read |
|---|---|
| what is ranked right now | `docs/STATUS.md` |
| how the optimization loop runs | `program.md` |
| what to try next | `docs/levers.md` |
| what has been tried, and what it moved | `docs/experiments.md` |
| what we learned the hard way | `docs/lessons.md` |
| the narrative of a past experiment | `journal.md`, `experiments/NNN-*/notes.md` |

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
- **DONE (exp 006 + 008 + 012 + 014) — large single matrices** (`n ≥ 16384`, esp.
  `32768²`): exp 012's left-looking formulation updates only the active
  diagonal/panel, reaching another 1.150× at 16384; native Blackwell FP8 panel
  products reach 1.373× at 32768, and exp 014 removes another 8.4% from their
  dynamic quantization front end while passing every input family. `1×8192`
  stays on cuSOLVER.
