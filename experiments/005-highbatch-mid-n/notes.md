# Experiment 005 — high-batch mid-`n` (`640×512`): probe → REJECTED

**Status: REJECTED (primary target). Nothing ranked-submitted.**
`640×512` is proven **cuSOLVER-batched-saturated** on B200 — no non-stream approach
can beat it. The current best `#877941` (~1746μs) stays the root submission. A ready
`8×2048` own-goal-fix candidate is included but NOT submitted (marginal; supervisor
decision on the last ranked slot).

## Hypothesis (from the goal)
`640×512` is slow because `torch.linalg.cholesky_ex` routes batch≥2 to
`cusolverDnSpotrfBatched` (tuned for many-tiny matrices), which was suspected to
**under-utilize** the B200 for "hundreds of medium (512²) matrices" — leaving
concurrency headroom capturable without streams. Test: does a concurrent/streamed
decomposition beat the single batched call (as it did for the exp-004 few-large
shapes)?

## Characterization probe (Modal B200, L2-clear, μs) — hypothesis REFUTED for 640×512
| shape | batched | loop | streamed | chunk64 | chunk128 | best |
|---|---|---|---|---|---|---|
| **640×512** | **3954.9** | 104887.1 | 25729.9 | 10494.9 | 7007.6 | **batched** |
| 2×2048 (control) | 4627.3 | **1384.2** | 1467.1 | 4674.2 | 4669.7 | loop |
| 8×2048 (control) | 5738.1 | 5427.9 | **3478.1** | 5890.7 | 5883.7 | streamed |

- `streamed` (max concurrency, each matrix on its own stream) is **6.5× SLOWER**
  than batched for `640×512` (25730 vs 3955). This is the exact **opposite** of the
  exp-004 headroom signal (where streamed beat batched for few-large shapes). It
  proves there is **no under-occupancy to capture**: cuSOLVER's batched `potrf`
  already saturates the B200 for hundreds of medium matrices.
- `loop` (640 sequential potrf) is **26× slower** (104887μs) — as expected, a
  control; per-matrix serialization is catastrophic at this batch.
- **Chunked batched** calls (`chunk64`, `chunk128`) — the shippable, default-stream
  alternative — are **1.8×–2.7× slower** (7008–10495μs). Splitting 640 matrices into
  sub-batches loses cuSOLVER's batching efficiency; it does *not* hit a
  better-occupancy code path. This directly rules out the goal's "chunked batched
  calls" idea with data.
- The `2×2048`/`8×2048` controls reproduce the known exp-004 numbers (loop/streamed
  win for few-large), confirming the probe harness is measuring correctly — the
  `640×512` result is a real property of the shape, not a harness artifact.

## Verdict on the approaches (cheapest-first, all closed)
1. **CUDA graph capture** — pointless. `640×512` batched is a *single*
   `cusolverDnSpotrfBatched` launch; there is no per-launch overhead to amortize and
   graph replay cannot raise intra-kernel occupancy. Capturing the loop instead
   would at best approach `streamed` (25730μs), still 6.5× worse than batched.
2. **Chunked batched calls** — measured, LOSES (1.8×–2.7× slower). Rejected by data.
3. **Custom blocked / tensor-core kernel** — not pursued. exp-003 already showed a
   naive kernel loses even at n=128 against cuSOLVER; here the vendor batched routine
   is *saturating* (streamed can't beat it), so a custom kernel would have to
   out-perform a fully-occupied, well-tuned vendor batched GEMM/POTRF pipeline at
   n=512. Multi-hour effort with essentially no expected payoff given the saturation
   evidence. Escalation not justified.

**Conclusion:** `640×512` is at its frontier on cuSOLVER batched. This closes the
shape (a valid outcome, like exp-003). The `640×512` dispatch stays on batched
cuSOLVER — no change needed in `submission.py` for the primary target.

## `8×2048` own-goal (secondary)
The exp-004 loop region `2<=batch<=8, n>=1024` regressed `8×2048` on popcorn
(5010 batched → 5370 loop). The fix is to trim the region to `2<=batch<=4` so
`8×2048` returns to batched cuSOLVER. This candidate `submission.py` carries that
change (and nothing else vs `#877941`).

- **Correct by construction:** it routes `8×2048` to the default batched cuSOLVER
  path already validated across all families in exp-004 (Modal 26/26, popcorn 17/17)
  and removes no other branch. `verify_local.py` on root: **10/10** (repo intact).
- **Modal benchmark deliberately NOT run for the fix:** on Modal, `8×2048` batched
  (5738) is *slower* than loop (5427) — the Modal↔popcorn fidelity gap noted in
  exp-004. The fix helps on **popcorn** (5010 < 5370), which a Modal benchmark cannot
  show; running it would only mislead. The decision rests on the confirmed popcorn
  per-shape numbers, not a fresh Modal number.
- **Impact:** ~0.05% geomean (~1746 → ~1738μs). A cleanup, not a prize.

## Ranked-slot recommendation (supervisor decision — flagged)
Since the primary prize (`640×512`) is dead, the only thing the last ranked slot
would buy is the ~0.05% `8×2048` cleanup. Against that tiny, deterministic gain,
a fresh ranked run risks **run-to-run cuSOLVER drift** (±~20% on mid shapes, seen
across exp-002/003/004) regressing *other* shapes. **Recommendation: do NOT spend
the last ranked slot on this.** Keep `#877941` as the confirmed best. The candidate
is ready to ship if the supervisor decides the cleanup is worth the slot.

## Decisions
- **`640×512`: REJECTED** — cuSOLVER-batched-saturated, no non-stream path beats it.
- **`8×2048` fix: prepared, NOT submitted** — marginal; supervisor call on quota.
- **Root `submission.py` unchanged** — remains exactly `#877941`.
- **Ranked quota unchanged: 2 of 3 used** (1 remains).

## Modal spend
1 probe run (3 shapes × 5 approaches, ~59s B200 wall) ≈ **~$0.2–0.4** (image cached).
No verify/benchmark runs needed (640×512 decisively dead; 8×2048 fix correct by
construction and not submitted).

## Remaining ideas (low priority)
- The board leader is already beaten by ~9%; every 15-shape shape is at/near its
  frontier. Only speculative remaining lever is a blocked tensor-core kernel for
  mid-`n`, but the saturation evidence here makes it a poor bet.
- If a future ranked slot opens with no drift risk, ship the `8×2048` batch<=4 fix.
