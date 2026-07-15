# Goal — experiment 013: 1×32768 without cuSOLVER (Triton / Blackwell tensor cores)

## Ranked baseline

Current ranked winner **`#878893`** (`experiment 012`, commit `141d015`):
public geomean **1459.321342997556 µs**, secret **1448.3768036226527 µs**.
Root `submission.py` is this exact source.

## Target shape

**`batch=1, n=32768`** — the single slowest shape in wall-time (~52 ms/iter) and
**~76 % of the ranked clock**, so it is the largest lever on the geometric mean.
A speedup here moves the geomean more than any other shape.

### What the shape does today

`custom_kernel` routes `batch==1, n==32768` to `_left_looking_cholesky_32768`
(exp 012):

- left-looking, block size `nb = 4096` (8 diagonal blocks);
- each **diagonal block is factored with `torch.linalg.cholesky_ex` (cuSOLVER)**;
- the panel product uses native Blackwell FP8 (`torch._scaled_mm`, E4M3,
  per-tensor scaling) with FP32 accumulation;
- the panel triangular solve uses `torch.linalg.solve_triangular` + a matmul.

Paired same-process evidence at adoption: **71567.591 → 52139.092 µs (1.373×)**
vs the exp-009 path; reconstruction residual **4.52 / 20** tolerance.

## Hypothesis

The remaining cost is dominated by (a) the O(n³) trailing/panel tensor-core
products and (b) the eight FP32 `cholesky_ex` diagonal factorizations. We can go
faster **without any cuSOLVER call** by:

1. **Eliminating the cuSOLVER diagonal potrf.** Factor each `nb×nb` diagonal
   block with a **two-level / recursive blocked scheme** whose inner diagonal
   sub-blocks are factored by a **Triton** kernel (or small unblocked torch ops)
   and whose inner trailing updates run on Blackwell tensor cores (TF32 or FP8).
   This is tracker candidate #6 (two-level blocked for 32768) and closes the
   Triton/Custom-CUDA columns for this shape.
2. **Pushing more of the O(n³) work onto Blackwell 5th-gen tensor cores** at
   lower precision — e.g. **MXFP8 block-scaled** products (candidate #2) for the
   panel/trailing updates, which are more accurate per-tile than the current
   per-tensor FP8 scaling and may allow a larger FP8 fraction, and/or a fused
   Triton panel+trailing kernel to cut launch overhead and global-memory traffic.
3. Optionally **1–2 steps of SPD iterative refinement** to recover accuracy if a
   more aggressive low-precision factorization drifts toward the tolerance edge.

## Hard constraints

- **No cuSOLVER anywhere on the 1×32768 path.** Do not call
  `torch.linalg.cholesky` / `cholesky_ex` on this shape — not for the diagonal
  block, not as an inner step. (A cuSOLVER-only *safety fallback* for
  non-finite output is acceptable **only** as a correctness net that must never
  fire on the ranked dense input; the timed/adopted fast path must be
  cuSOLVER-free. Prefer no fallback if a Triton unblocked factorization is
  robust enough.)
- Prefer **Triton** and native Blackwell tensor-core formats (TF32, FP8 E4M3,
  MXFP8 block-scaled via `torch._scaled_mm`). CUTLASS/`tcgen05` custom CUDA is
  allowed if it compiles on the Modal `nvidia/cuda:13.0.0-devel` image.
- **Default CUDA stream only.** popcorn's static source scan rejects the literal
  word "stream" and any non-default queue → keep the source clean of it.
- Change **only** the `1×32768` dispatch region. The other 14 shapes must remain
  byte-for-byte on the current `#878893` implementation.
- Output contract: return an **owned** lower-triangular FP32 tensor with the
  strict upper triangle zeroed and positive diagonal; do not return a reused
  static buffer (see the exp-009 retained-output failure `#878263`).

## Correctness gate (the spec)

`check_implementation` requires `‖A − LLᵀ‖₁ ≤ 20·n·eps·‖A‖₁` (reconstruction
computed with TF32 disabled). At n=32768 the tolerance is large, but the current
path already spends most of it (margin ~4.4×), so watch the residual and prefer
approaches that keep or improve the margin. Validate across **dense, spectrum,
lowrank, rowscale, diagonal, tridiagonal** families at n=32768.

## Success threshold

1. A **paired same-process B200 probe** (rotating ≥2 inputs, all outputs retained
   through validation) shows the candidate `1×32768` path **faster than the
   current `#878893` path** — target **≥1.10×**, and it must pass every family.
2. The candidate confirms via a backend counter that the **fast cuSOLVER-free
   path actually ran with zero fallbacks** and no FP8/scaled-mm runtime error.
3. The **full 15-shape Modal grid** passes with the other 13/14 regions unchanged
   and a **lower geomean than exp 012** (`1574.881992 µs` on that harness).
4. Popcorn **test mode 17/17**.
5. **Exactly one** ranked submission, launched only after gates 1–4 pass, and
   adopted only if the completed public/secret score **improves on 1459.321 µs**.

## Bounded fallback ladder

Stop at the first approach that clears the paired gate; do not stack unrelated
changes before the first causal measurement.

1. **Two-level blocked, Triton diagonal, TF32 trailing** (drop-in cuSOLVER
   removal): recurse the `nb=4096` diagonal potrf into `mb`-sized sub-blocks;
   factor the small `mb×mb` diagonal with a Triton kernel; inner panel solve +
   trailing update on TF32 tensor cores. Keep the existing FP8 outer panel.
2. **+ MXFP8 block-scaled** outer trailing/panel products (replace per-tensor
   FP8 scaling) for a larger low-precision fraction at equal/better accuracy.
3. **+ Fused Triton panel+trailing** to cut per-step launches and global-memory
   materialization at nb granularity.
4. **+ 1–2 SPD iterative-refinement steps** if accuracy is the binding
   constraint after (2)/(3).
5. If none beats `#878893` paired, **reject** with the negative evidence
   recorded; do not spend a ranked slot.

## Cost / submission guardrails

- Free local checks (property test, `python -m py_compile`, `git diff --check`,
  JSON parse) before any B200 spend.
- Develop against the smallest useful proxy first (`precprobe 8192` / `16384`),
  then confirm on the expensive `32768` probe. The `32768` grid iter is ~50 ms;
  keep iter counts low (the harness already uses 3–4).
- **At most one** ranked submission this experiment; no duplicate submissions.
- Modal uploads are limited to `submission.py`, the vendored `reference/`
  harness, `scripts/_gpu_runner.py`, and the experiment candidate/baseline files,
  per the owner authorization recorded in the root README. No evo workflow.

## Workflow / definition of done

Follow the "Required end-to-end experiment workflow" in `journal.md`. The
experiment is complete only when: a winner is copied to root `submission.py` (or
explicitly rejected), artifacts + notes + both READMEs + the Optimization Tracker
+ a dated journal entry are updated, one descriptive commit is made on `main`,
and the commit is **pushed to `origin/main`** and verified present on the remote.
