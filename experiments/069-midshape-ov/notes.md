# Experiment 069 — e62 write-only upper mask (Ov lever)

**Verdict: ADOPTED — ranked `#926130` (test `#926123` 17/17). Current best.**

## Goal
Run `program2.md`; improve kernel latency for ≥2 leaderboard shapes.

## Baseline
Incumbent `#922201` (exp 067), commit `fc949a2`, SHA-256
`f108cbba…a62a429`.

## Lever (Ov / QR-ladder lever 7)
The shared `_exp062_factor` ended with `work.tril_()` — a full-matrix
read+rewrite that a fresh incumbent `shapediag` measured at **145us = 16.8% of
`60×1024`**, ~2.4× its own bandwidth floor (torch reads and rewrites all n²
elements just to zero the strict upper triangle).

Replaced it with a **write-only `e62_zero_upper` CUDA kernel** (added to the e62
`load_inline` extension): one CUDA block per (matrix, row); threads stride the
strict-upper columns of that row writing `0.0f`, never reading or touching the
lower triangle. Traffic ~n²/2 writes vs torch's ~2n² read+write.

`work = data.clone()` copy-in is kept, so every read and all arithmetic in the
factorization are unchanged → the L factor is **byte-identical** to the
incumbent (both zero exactly the strict upper). Value-independent: the speedup is
independent of matrix values, so it carries to the secret split (exp-067 class,
distinct from the exp-065 precision-secret risk).

## Change (3 hunks in `candidate.py`, all inside the e62 block)
1. `_EXP062_SOURCE`: add `e62_zero_upper_kernel` + `e62_zero_upper_launch`.
2. `_load_exp062`: declare `e62_zero_upper_launch` in `cpp_sources` and `functions`.
3. `_exp062_factor`: replace `return work.tril_()` with a guarded call to
   `_EXP062.e62_zero_upper_launch(work)` (falls back to `tril_()` if a stale
   extension cache lacks the symbol).

## Gates
- Free: ast parse OK, clean diff, default-stream launch only (no banned construct).
- Determinism (N1): by construction (deterministic write-only mask, byte-identical).
- Correctness/adversarial (N2) `results/069-family.json`: all rows
  `checker_ok=true` on 512/1024/2048/4096; e62 active; only pre-existing
  spectrum/lowrank fallbacks (change runs after the `isfinite(l.diagonal())`
  decision → cannot add a fallback).
- Paired e62 `results/069-paired.json`; full grid `results/069-fullgrid.json`:
  geomean **1.0136** CI95 [1.0131, 1.0140] excludes 1.0.

## Results (same-process paired grid, control → candidate)
| shape | control us | candidate us | speedup |
|---|---:|---:|---:|
| `60×1024` | 936.1 | 852.6 | **1.0979×** |
| `8×2048`  | 1274.3 | 1216.2 | **1.0478×** |
| `2×4096`  | 2281.3 | 2217.7 | **1.0282×** |
| `2×2048`  | 1059.6 | 1044.1 | **1.0153×** |
| `4×1024`  | 539.4 | 532.3 | **1.0139×** |
| `16×512`  | 296.7 | 294.6 | **1.0062×** |
| other 9   | — | — | 0.998–1.001× (flat, ≤0.23% off-target) |

`fast_p`: fast_0 = 8/8 correct, fast_1 = 6/6 e62 shapes faster, fast_targ = 0.
0 new fallbacks on every shape; identical per-shape residuals + counters.

## Ranked
- Test `#926123`: 17/17.
- Ranked `#926130`: both public+secret splits PASSED. Public benchmark showed
  `60×1024` ≈ 811us. CLI does not expose the official geomean.

## Adoption
Root advanced to SHA `e187bfa9…282c5fae`. STATUS / experiments / levers / journal
updated. Adopted on paired-grid + byte-identity + passing ranked runs per the
exp-067 precedent and owner authorization.

## Follow-ups (unharvested Ov on these shapes)
- `data.clone()` copy-in (82us efficient memcpy on `60×1024`).
- `640×512` ~144us inter-launch idle across 53 launches (CUDA-graph / fusion; not
  an e62 shape, so untouched here).
