# Experiment 071 — the two largest shapes: `1×16384` and `1×32768`

**Verdict: `[I7] SATURATED` — no legitimate (shippable, default-stream)
custom diagonal `potrf` can beat cuSOLVER at batch=1. Classified at `[I1]` by
the Amdahl/roofline gate plus the complete exp 061/063/064/070 empirical
record. `$0` GPU spent. Root unchanged; `#926462` remains ranked.**

**Goal (owner, under `program2.md`).** Improve measured kernel latency for the
two largest ranked shapes. Owner scope: *build + verify a custom diagonal potrf
on the same-process paired grid; NO ranked submission* (so the secret split is
not at stake and the promotion rule is moot).

## `[O1]` incumbent verified
Local `HEAD` = `origin/main` = `1ec14b7`; root `submission.py` SHA-256
`582cde16…b869ff723`; ranked slot `#926462` (exp 070) confirmed on the
`cholesky` board. No drift.

## `[O2]` where the time goes (results/070-largephase.json)
Both dispatched drivers (`_exp061_factor_1x16384`, `_exp061_factor_1x32768`)
are already deeply harvested. The residual cost is the **serial diagonal
`cholesky_ex`** on the nb blocks:

| shape | wall | diagonal potrf | cuSOLVER rate | other phases (all tensor-cored) |
|---|---:|---:|---|---|
| `1×16384` | 8,414us | **5,569us (64.6%)** | ~340 ns/row | update 1353 / inverse 782 / panel 917 (FP16) |
| `1×32768` | 23,075us | **12,501us (52.2%)** | ~381 ns/row | update 4837 (MXFP8) / inverse 3315 (trsm-free) / panel 2319 (FP16) |

cuSOLVER's 4096³ `potrf` = **1566us = ~381 ns/row, latency-bound**
(`results/070-nocusolver.json`) → ~14.6 TFLOP/s achieved, i.e. it is at the
**serial-dependency floor, not the compute floor**.

## `[I1]` Amdahl ceiling for the one remaining lever
A from-scratch overlapped blocked FP32 potrf at an *optimistic* 296 ns/row (the
best this repo ever reached — exp 063, and only on the 128-block e62 kernel):

- `1×16384`: `1/(0.354 + 0.646·296/340)` ≈ **1.09×**
- `1×32768`: `1/(0.478 + 0.522·296/381)` ≈ **1.14×**

A FRONTIER at best, and 296 ns/row on 2048/4096 blocks was never demonstrated.

## Why a *legitimate* custom potrf cannot win at batch=1 (the airtight case)
The diagonal factorization is serial by construction. To beat cuSOLVER you must
parallelize the serial chain across the ~148 SMs. Every mechanism that does so
is either measured-slower or popcorn-banned:

1. **Op-level blocking** (nb block via smaller cuSOLVER leaves + cuBLAS
   trailing) — **measured slower**: exp 064 `blocked_leaf1024_tf32` = 1089us vs
   cuSOLVER 676us. The per-call `potrf` floor doesn't amortize (the "340 vs 381
   ns/row" gap is fixed-floor amortization, so splitting *loses*), and the extra
   launches swamp the diagonal. exp 061 had already measured every op-level
   blocked alternative at m=2048 as slower than cuSOLVER.
2. **Single-CTA fused megakernel** — one CTA = one SM ≈ 100× SM-starved vs
   cuSOLVER's whole-GPU trailing; and a 2048/4096 FP32 block (16–64 MB) is not
   shared-memory resident (228 KB/SM).
3. **Multi-CTA fused megakernel** — needs cross-SM sync (cooperative
   `grid.sync()` or software global barriers). Both are **popcorn-banned**
   (lessons.md Part 2; QR: cooperative panel −47%, software barriers −42%, all
   "worked but banned"). Such a kernel would win on Modal (which can't detect the
   anti-cheat) but is an **unshippable hollow number** — lessons.md failure #4.

The **exp-065 named-barrier overlap only ever helped MID shapes** because those
batch many 128-blocks across the *batch* dimension and fill the SMs. At batch=1
that parallelism does not exist — this is the fundamental wall.

## Nothing cheaper remains either
- Non-diagonal phases are harvested: FP16 (16384) / MXFP8 3466 TFLOP/s (32768)
  updates, trsm-free recursive inverse, fused Triton strided moves (exp 061/064,
  shipped).
- Per-call overhead is already tiny here: **diagonal-only** `isfinite` check, and
  **no copy-in clone** (the driver reads `data[0]` directly and writes a fresh
  `zeros_like` factor).
- nb is confirmed optimal in both directions (exp 064 sweep).

## `[N7]` fast_p
0 variants measured. The lever was classified SATURATED/FRONTIER at `[I1]`
before generation (lessons.md: "compute the Amdahl ceiling before opening a
shape — it classifies the outcome in advance"), and the only untried mechanism
(cooperative megakernel) is banned. `fast_0 = fast_1 = fast_targ = n/a`.

## Terminal state
`SATURATED` (closes like `SHAPE EXHAUSTED`): preserve the incumbent, release the
lease, do not keep spending against a resource ceiling already reached. The only
path past this wall is a from-scratch cooperative/grid-sync potrf, which is
**unshippable** by popcorn — do not reopen without a change to the anti-cheat
rules.
