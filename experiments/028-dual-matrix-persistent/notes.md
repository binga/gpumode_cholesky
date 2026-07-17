# Experiment 028 — persistent dual-matrix factorization

**Status: REJECTED — decisive negative evidence at the first target shape.**
Baseline: exact ranked winner `#882958`
(`3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`,
public 1096.084us).

## What was built

A cuSOLVER-free, single-launch persistent Triton path for `2x2048` (and
prospectively `2x4096`): a fixed resident grid with a device-side phased
scheduler, one worker per matrix for the 32-column diagonal microblocks,
cooperative panel and rank-128 Schur updates, and GPU atomics as phase
barriers. Five variants were measured on `2x2048` via the `dualprobe` harness
mode (added to `scripts/_gpu_runner.py` / `scripts/modal_verify.py`).

## Results (paired same-process, 2x2048, dense)

| variant | candidate mean | baseline mean | speedup |
|---|---:|---:|---:|
| v1 16 workers/matrix | 3183.8us | 1362.8us | **0.428x** |
| v2 64 workers | 2839.6us | 1361.6us | **0.479x** |
| v3 64-col micro | 3381.8us | 1359.8us | **0.402x** |
| v4 fused panel | 2800.0us | 1359.1us | **0.485x** |
| v5 FP16 trailing | 2754.9us | 1362.4us | **0.495x** |

All variants were *correct* (residuals 0.72–2.0 of 20, hits 105/105, zero
fallbacks) but 2.0–2.5x **slower** than the shipped per-matrix cuSOLVER loop.
The bounded variant ladder (worker count, micro width, panel fusion,
precision) moved the needle only 0.40x -> 0.49x — the ceiling of this
architecture is far below 1.0x, so the sixth variant was not spent.

## Why it loses

The spin-barrier phased scheduler serializes every phase across the resident
grid at kernel-wide latency (~microseconds per phase transition through L2
atomics), and Triton cannot warp-specialize inside a program: the serial
32-column diagonal chain occupies whole phases during which the cooperative
workers idle. The measured floor (~2.75ms) is consistent with
(number of phases) x (barrier latency) dominating, exactly the failure mode
that makes the graph-replayed multi-kernel chain (sum of kernel self-times,
no host overhead) the better structure on this workload.

## Disposition

- `2x2048`/`2x4096` remain on the ranked per-matrix loop.
- Conclusion recorded for the tracker: **persistent single-launch scheduling
  in Triton is rejected for the mid shapes**; future work on the split32
  chain should shorten the graph-replayed kernel sum instead (-> exp 029).
- No leaderboard submission was spent.
