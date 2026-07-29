# Experiment 076 — e62 panel bounce-copy fusion (Fu/Ov, QR-ladder lever 7)

## Frozen control

Ranked `#926462` (exp 070), commit `81e9451`, `submission.py` SHA-256
`582cde1648b8b3e9d77a36173dd59cd36588123ae28800ca00e5342b869ff723`. Verified
against `popcorn submissions list` after re-auth (`#926737` exp-074 attempt is
`done`+rejected; `#926462` remains the ranked winner).

## Goal and promotion rule

Owner goal: improve measured kernel latency for **two mid shapes**. Targets
`60×1024` and `8×2048` (the two e62 shapes with the most historical headroom).
Promotion rule = **optimize public, accept secret** (`docs/STATUS.md` owner rule
for the live `#926462`). Ranked slot gated on measured grid magnitude vs LB noise.

## [I1] Lever and Amdahl ceiling

`results/069-inc-shapediag-mid.json` shows the e62 path spends, on `60×1024`
(~852us), ~318us in the already-overlap-optimized diagonal block plus two
value-independent overheads: the `data.clone()` copy-in (82us) and the panel
bounce copy `src.copy_(dst)` (the strided `elementwise_kernel` ×7 = **73.6us /
8.5%**, running at ~2.6× its bandwidth floor).

- **Copy-in clone — REFUTED on a free gate ($0).** `reference/eval.py` line 180
  runs the timed loop as `custom_kernel(data)` *without* cloning and reuses
  `data_list` across repeats (with `recheck`). Factoring in place would corrupt
  the inputs across timed iterations, so the internal `data.clone()` is
  load-bearing and cannot be removed. (exp-031 class: killed before GPU spend.)
- **Panel bounce copy — the target.** `torch.bmm(src, dinv^T, out=dst)` writes
  the scaled panel to the contiguous scratch `pan`, then `src.copy_(dst)` writes
  it back to the strided work panel. The copy is pure redundant traffic (the
  cuBLAS TF32 bmm already produced the values). Replacing it with a coalesced
  write-back kernel keeps `L` **byte-identical**.

Amdahl ceiling (a faster copy, not full elimination): `60×1024` ~1.056×,
`8×2048` ~1.02× → a **FRONTIER**, not a 2× winner. Sized effort accordingly.

## [I2] N4 variants (one lever, three copy-back mechanisms)

All keep the exact cuBLAS TF32 `bmm`; only `src.copy_(dst)` changes. Each has a
unique `load_inline` name (cache isolation) and a `getattr`-guarded fall back to
`src.copy_(dst)` so correctness never depends on the recompile.

- **V1** — float4, one 32-thread block per (matrix, panel row), grid=(rows,batch).
- **V2** — float4 **flat grid-stride**, block=256 sized to fill the SMs.
- **V3** — scalar float, block=128 (vectorization control).

## [I3] Free gates — PASS

`ast` syntax OK ×3; `git diff --check` clean; source-policy scan clean (every
launch is a 2-arg `<<<grid,block>>>` on the default stream; no cooperative /
grid-sync / stream construct). float4 alignment proven: every panel offset is a
multiple of 4 floats because `n` and `col0=jj` are multiples of 128.

## [I3.5]/[I4] N1 determinism + N2 adversarial — PASS (V1 and V2)

`results/076-*-stress.json`. On all four e62 shapes (`4×1024, 60×1024, 2×2048,
8×2048`): 3× same-input outputs **bitwise-equal** (N1), residuals byte-identical
to the incumbent, backend active, 0 fallbacks; and all 36 adversarial rows
(tiny_diagonal / near_singular_banded / mixed_dynamic at cond 1e6/1e8/1e10)
`checker_ok`, 0 fallbacks (N2). No race surface (one thread per output element,
disjoint writes).

## [I5] Paired same-process latency vs the exact `#926462` source

7 repeats, 2 rotating inputs, one process. `results/076-*-paired.json`.

| Shape | Control us | V1 | V2 (flat) | V3 |
|---|---:|---:|---:|---:|
| `4×1024` | 546.3 | 1.0000 | 1.0027 | 1.0046 |
| `60×1024` | 838.7 | 0.9489 | **1.0310×** | 0.9486 |
| `2×2048` | ~1035 | 0.9947 | 1.0039 | 1.0003 |
| `8×2048` | 1200.4 | 0.9860 | **1.0120×** | 0.9836 |
| **e62 geomean** | | 0.9822 | **1.0123×** | 0.9840 |

- **V2 wins**: e62 four-shape geomean **1.0123×**, CI95 [1.0104, 1.0142]
  excludes 1.0. Both requested targets improve with CI above parity:
  `60×1024` **1.0310×** (838.7→813.4us, −25us), `8×2048` **1.0120×**
  (1200.4→1186.9us, −13.5us). Byte-identical (residuals 3.31/3.22/1.65/1.74
  match control), 0 new fallbacks, identical counters.
- **V1 and V3 lose** (0.982–0.984×): one block per panel row makes ~53,760 tiny
  blocks on `60×1024`, worse occupancy than torch's tuned strided copy. The win
  requires the flat grid-stride that fills the SMs — that is the mechanism.

## [I7] Classification

**V2 = PROMOTABLE FRONTIER.** Correct, byte-identical, N1/N2-clean, faster on
both target shapes and on the e62 aggregate, below the 2× research target
(Amdahl-ruled). V1/V3 REJECTED (slower). Because the change is a
value-independent byte-identical latency reorchestration (exp-067/069 class, not
the exp-065 precision class), the paired win is expected to carry to the secret
split.

## [N7] fast_p for the search

`fast_0 = 3/3` (all variants correct + byte-identical) · `fast_1 = 1/3` (V2
faster on the e62 aggregate) · `fast_targ (2.0×) = 0/3` (Amdahl ruled out).

## Disposition (pending owner decision at the ranked gate)

The requested objective — measured latency improvement on two mid shapes — is
**met and fully gated**. Whether to spend a ranked slot is a coin-flip on the
*public* board: this is a ~1% grid class, and exp 074 showed a byte-identical
~0.4% full-grid change swung public +3.27% (LB run-to-run noise). Options: bank
the frontier (root stays on `#926462`), or run [O4] full grid → [O5] cold build
→ [O6] test 17/17 → [O7] one ranked attempt under the accept-secret rule.
