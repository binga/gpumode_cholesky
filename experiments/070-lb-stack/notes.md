# Experiment 070 — public-LB stack (065 overlap + 067 + 069)

**Verdict: ADOPTED as ranked winner `#926462` (test `#926455` 17/17). Official
public 646.868→630.403us, board rank #32→#31.** NOT secret-safe.

## Why this experiment exists
Owner asked to improve our *leaderboard* score. The live gpumode board ranks by
**public** geomean and showed `binga` at **#32 / 646.868us** — which is **exp 065
(`#914341`)**, the named-barrier overlap we had rejected for a secret regression.
Our adopted lineage (`#913511→#922201→#926130`) is on a slower public base
(`#926130` ≈ 651us public, ~5us worse than exp 065), so it never set our rank.

## Large-shape lever was profiled and closed first
Owner's first choice was to attack the large-shape diagonal `potrf`.
- `results/070-largephase.json`: diagonal = **64.6% of 16384**, **52.2% of 32768**;
  every exp-064 structural variant (nb widths, v2 strided move, trsm-free,
  upper-fill) ties or loses the shipped driver.
- `results/070-nocusolver.json`: cuSOLVER 4096³ potrf = **1566us** (~381 ns/row,
  latency-bound); the custom potrf kernels (`_triton_blocked_potrf`,
  `_blocked_cublas_potrf`) are **gone from the source** (cleanup).
- Conclusion: a large-shape win needs a competitive blocked FP32 potrf written
  from scratch — secret-risky (exp-065 class), ~1–1.5% ceiling. Not tractable in
  the window. Owner then chose the cheap proven public stack.

## The change
exp 065 overlap (VAR=4 in the e62 128×128 diagonal block) + exp 067 (`60×1024`
e62 enroll) + exp 069 (write-only upper mask) touch **disjoint** code → compose.
Built: `cp experiments/065-midshape-overlap/ship-v1.py candidate.py; patch
--fuzz=3` with `diff(baseline-913511 → root submission.py)` (= 067+069). Verified
ast OK, 5 exp065 markers, `(60,1024):1024`, 6 exp069 markers, 0 banned constructs.

## Gates
- familygrid `results/070-family.json`: 48/48 `checker_ok` on 512/1024/2048/4096;
  VAR=4 active; only pre-existing spectrum/lowrank/`1×4096` fallbacks; no new.
- paired full grid `results/070-fullgrid.json` vs `#926130`: **1.0171×**
  CI95[1.0167,1.0176], `all_shapes_ok`, 0 new fallbacks, identical counters.

| shape | root us | cand us | ratio |
|---|---:|---:|---:|
| 2×2048 | 1055.3 | 1004.5 | 1.0507 |
| 2×4096 | 2241.8 | 2139.8 | 1.0474 |
| 4×1024 |  539.6 |  516.6 | 1.0455 |
| 16×512 |  297.6 |  286.0 | 1.0393 |
| 8×2048 | 1228.9 | 1182.9 | 1.0393 |
| 60×1024|  853.9 |  823.2 | 1.0369 |
| 9 others (incl 8192/16384/32768) | — | — | flat ≤0.06% |

- Popcorn test `#926455` 17/17 (cold build, torch 2.12.0+cu130).
- Ranked `#926462`: both public+secret splits passed.

## Promotion rule
**optimize-public-accept-secret** (owner-selected 2026-07-29). exp 065's overlap
regressed secret (+5.71% on `#914341`), so this winner is **not secret-safe**;
adopted because the board ranks by public and the competition closed the same day.
`#926130` (SHA `e187bfa9…282c5fae`) is the secret-safe fallback.

## Follow-up
The only path past ~#30 is breaking the cuSOLVER diagonal `potrf` wall on the
large shapes with a from-scratch overlapped blocked potrf — high effort, secret
risk. Nothing cheaper remains on the board.
