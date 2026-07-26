# Lessons — what we learned the hard way

Cross-cutting knowledge that is not tied to one experiment. `program.md` carries
the *rules*; this file carries the *reasons*, plus the transferable technique
notes.

Three sources feed it: this campaign's own failures, the `qrproblem` project on
the same B200 target under the same geomean scoring rule, and
<https://sankalp.bearblog.dev/autoresearch/>.

---

# Part 1 — This campaign

## Cheap gates that are still under-used

- **Compute the Amdahl ceiling before opening a shape.** It costs nothing and it
  classifies the outcome in advance. Exp 065: the diagonal is ~55% of a mid
  shape and the best predicted block speedup was 1.52x, so
  `1 / (0.45 + 0.55/1.52)` = 1.23x was the ceiling — correctly ruling out the
  2.00x research target before any GPU spend, and correctly predicting a
  frontier rather than a winner. Record the ceiling in the goal.
- **Reuse the candidate's own probe hook rather than writing a harness.**
  `mid_probe()` plus a variant-templated kernel gave, in **one** Modal run:
  per-block timing, the phase-cycle breakdown, `abs_err`/`inv_err` against an
  in-source control, and whole-shape speedups on five shapes. Restrict the
  variant tuple to (control, candidate) — probing the whole ladder every time
  multiplies the bill for rows the decision does not turn on.
- **The previous experiment's "next levers" section is load-bearing.** Exp 065
  took one variant instead of six because exp 064 had already named the lever
  and costed it. Keep those sections quantitative; they are the cheapest input
  the next experiment gets.

## Blackwell kernel technique notes

- **`bar.sync <id>, <count>` gives a partial-block barrier.** When a subset of
  warps must synchronise while other warps are elsewhere in the kernel,
  `__syncthreads()` deadlocks — it is barrier 0 over every thread in the block.
  A named barrier with an explicit participant count (a multiple of the warp
  size) is the primitive. Exp 065 used id 1 over 224 threads so warps 1-7 could
  synchronise between staging and the trailing update while warp 0 sat inside
  the triangular inverse.
- **Overlap is available wherever one warp holds a serial phase and the others
  idle** — but only after proving the memory footprints are disjoint. Exp 065's
  three regions (`S[kk:kk+32,kk:kk+32]` read-only, `S[lwid:,kk:kk+32]` -> `P`,
  `P` -> `S[lwid:,lwid:]`) provably do not alias, which is why the arithmetic
  came out bit-identical to the control. Establish that first; the win is
  scheduling, and a scheduling change that alters results is a bug.
- **Expect the overlapped serial phase to get slightly slower.** Exp 065's
  `triinv` grew 2% (37019 -> 40341 cycles) from shared-memory contention with
  the seven warps now working behind it. That is the price of the overlap, not
  a defect, and it is small next to the phases that leave the timeline.

## Operational failure modes

Each of these cost a session or a paid gate.

1. **The ranked incumbent is whatever `popcorn submissions list` says**, not what
   `main` or `README.md` says. The true winner has repeatedly lived on an
   unmerged branch while `main` lagged several winners behind. Verify by
   `shasum -a 256` before spending, and again immediately before any paid gate —
   `origin/main` moved twice mid-session during exp 064.
2. **Never edit a Modal-mounted file while a job is building.** `submission.py`,
   the candidate source, `reference/`, and `scripts/_gpu_runner.py` are copied
   into the image; touching any of them mid-build fails the job with
   `ExecutionError: ... was modified during build process`. Exp 064 lost an
   ~18-minute family grid this way. Edit notes and `state.json` while jobs are in
   flight; nothing else.
3. **Compare like with like in probes.** A row that calls a driver directly is
   not comparable to a control that goes through `custom_kernel` — the wrapper
   carries dispatch plus the end-of-call `isfinite(...).all().item()` sync,
   measured at 845us on `1x32768`. Always include a reimplementation of the
   *shipped* logic as the probe's own control.
4. **Read `familygrid` per-row `checker_ok`, not the top-level `passed` flag**,
   and attribute every fallback against a baseline run rather than from memory.
   Exp 065's grid reported `passed: false` on both baseline and candidate; the
   candidate was in fact byte-identical across all 48 rows. Also: `spectrum` is
   not generable at `n>=16384` (it needs an `n x n` QR) and must be excluded via
   `--families`.
5. **`__syncwarp()` and `__syncthreads()` are hardware barriers, not compiler
   barriers.** Shared-memory staging inside a loop needs an explicit compiler
   barrier as well (`asm volatile("" ::: "memory")`). The failure is silent and
   correct on the first loop iteration, which is exactly why exp 063's wrong
   indices looked like a guard bug at a group boundary for two rounds.
6. **Phase-share tables mislead when one phase collapses.** Exp 063 round 2 read
   a fusion shortfall as register pressure; the *cycle* counts were flat and only
   the shares had moved. Read cycles, not percentages.

---

# Part 2 — Transferred from the QR project

## Context

- Shapes are written as `(batch, n)`. The kernel factorizes `batch` square-ish
  matrices of dimension `n`.
- Baseline was `torch.geqrf`. Two shapes dominate the geomean — `(640,512)` and
  `(60,1024)` — so most engineering effort went there.
- **The universal finding across every size: the serial Householder panel
  factorization is the bottleneck, not the trailing GEMM.**

## What worked, per size

### `(640,512)` — 640 matrices, the #1 target (1074 ms → 14.6 ms, ~73×)

Nearly all wins compounded here:

- **Blocked WY Householder** (V3): 1074 → 617 ms. First structural win over `geqrf`.
- **Full-double panel in shared memory** replacing cuSOLVER's panel (V13):
  617 → **65.6 ms** (9.4×). Biggest single hand-tuned jump.
- **Prealloc workspaces + CUDA V-builder + CUDA T-builder** (V20b/V21/V22):
  65.9 → 62.2 ms, eliminating ~496 ATen `bmm` calls.
- **`at::matmul` with no `contiguous()` copy + `baddbmm` fused trailing**
  (V24/V27): 62 → 57.9 ms.
- **Vectorized Triton FP32 panel** replacing the serial double-precision panel
  (V36, the single biggest evo win): 62 → **31.5 ms (−48%)**. Proved the
  double-precision panel was unnecessary at batch=640.
- **`nb` 32→16 sub-blocking** to relieve register spill and shift work onto the
  cuBLAS WY GEMM (V37): 31.5 → 24.8 ms.
- **Apply-only TF32 tensor-core trailing** (exploiting loose correctness gates)
  (V38): → 23.4 ms.
- **Strided trailing W1 (skip explicit contiguous copy, let cuBLAS handle the
  strided view)** (V42): → 18.6 ms.
- **Builder-fusion + apply-GEMM fusion** (exp_0051): → **14.6 ms**.

### `(60,1024)` — the #2 target (239 ms → 17.3 ms, ~14×)

- **Batch-parallel C++ blocked WY** replacing 60 sequential `at::geqrf` calls
  (V31): 239 → **54.2 ms (4.4×)**.
- **Vectorized Triton FP32 panel** (V36): 54 → **30.7 ms (−40%)**.
- **`nb`=16 register relief** (V37): 30.7 → 19.5 ms.
- **Full-TF32 trailing (W1/W2/apply all TF32)** (V41): 18.4 → **17.3 ms**.

### `(40,176)` and `(40,352)` — medium shapes

Never dominant, but improved steadily:

- **cuSOLVER batched / C++ blocked loop** removing Python dispatch overhead
  (V5/V9): `(40,176)` 22 → 6.8 ms.
- **FP32 panel + CUDA V/T builders + prealloc** (V18/V20b/V21): `(40,176)`
  → ~4.25 ms, `(40,352)` → ~10.5 ms.
- **Medium-path trailing fusion** (strided `at::matmul` for W1 + in-place
  `baddbmm_`) (exp_0056): held ~10.9 ms with less memory traffic. Now
  register-tile/launch-bound, so returns flattened.

### `(20,32)` — smallest shape

- **Fused single CUDA kernel for n≤32** (V11): 324 → 131 µs (2.6×).
- **Shared-memory CuTe DSL QR32 kernel** (V16/V17): → ~84 µs. Effectively free
  by the end (~0.15 µs class in later rows); negligible to the geomean.

### `(8,2048)` and `(2,4096)` — large, low-batch shapes (the remaining wall)

These stayed **panel-bound on sequential cuSOLVER `geqrf`**, and almost every
promising fix was **non-submittable** on Popcorn (its anti-cheat disqualifies
any work on a non-default CUDA stream):

- **What worked but was banned:** grid-sync cooperative panel (V39, −47% on
  `(8,2048)`), thread-block clusters (−31%), software global barriers (−42%),
  CholeskyQR2 + Householder reconstruction (V40, −53%).
- **What actually shipped:** for `(8,2048)`, a **single-pass FP16 tensor-core
  blocked-Householder trailing** (V38) trimmed 76.8 → 71.8 ms; and a
  **submittable CholeskyQR2 with GEMM-based triangular solve** (exp_0054, no
  cuSOLVER `potrf`/TRSM so it stays on the current stream) cut `(8,2048)` to
  ~38 ms. `(2,4096)` barely moved (~52 → 49 ms) — at batch=2 it's
  latency/combine-bound, not sync-bound.

## Cross-cutting techniques that generalized

1. **Move the block loop off Python / out of the panel** — C++/CUDA loop,
   prealloc workspaces, CUDA V/T builders.
2. **Attack the panel, not the GEMM** — vectorized Triton FP32 panel + `nb`=16
   register relief delivered the largest wins.
3. **Exploit the loose correctness gates** — the checker only gates factor
   residual and orthogonality (with 4–10× / 280–610× margins), so FP16/TF32
   tensor-core math is "free" on the apply/trailing steps. FP16 was the sweet
   spot (BF16 failed the factor gate; TF32 tied).
4. **Test submittability on the real target early** — Modal can't detect
   Popcorn's stream anti-cheat, so several validated wins (cooperative launch,
   clusters, batched `torch.linalg`) died only on the real target.

## Best result

**V43** (`solutions/v43_cholqr2048_medium_fusion_merge/`, also root
`submission.py`) — ≈**10,901 µs** geomean, 26/26 correct on Modal. Merges the
CholeskyQR2 `(8,2048)` win with medium-path trailing fusion on top of the
`(640,512)`/`(60,1024)` builder-fusion spine — ~27% reduction from the V31
hand-tuned baseline, ~42% on evo's internal metric.

Caveat: V43 was **never resubmitted to live Popcorn** (deadline passed
2026-06-30); the last number confirmed on the real leaderboard was exp_0051 at
~11,003 µs. V43's 10,901 µs is the `bench_grader.py --twelve` grader-methodology
proxy.
