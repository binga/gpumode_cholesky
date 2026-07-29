# Experiment 077 — two largest e62 mid shapes: `2×4096` and `8×2048`

**Program:** program2.md. **Goal (owner):** improve measured kernel latency for 2 mid
shapes on the leaderboard. **Targets:** `2×4096` (2123.6us) and `8×2048` (1182.5us) —
the two largest e62 mid shapes by wall time, highest geomean leverage.

## [O1] Incumbent verified
Root `submission.py` SHA-256 `582cde16…b869ff723` = ranked `#926462` (exp 070).
`popcorn submissions list` confirms `#926462` is the live winner; `#927042` (exp 076)
is `done` but did not displace `#926462` on public (banked frontier, root correctly
restored). No drift.

## Promotion rule
Optimize public, accept secret (`docs/STATUS.md` owner rule for live `#926462`).
Ranked slot gated on measured grid magnitude vs LB noise: exp 074 (1.00375×) and
exp 076 (1.0036×) both showed sub-1.5% paired grids are public coin-flips. So this
experiment must clear ~1.5%+ on the paired grid to justify a ranked slot, OR stack
two byte-identical Ov frontiers whose combined grid clears noise.

## [O2] Fresh shapediag on the incumbent (results/077-inc-shapediag.json)

| Shape | wall us | idle % | e62_diag128 | bounce-copy | clone | zero_upper |
|---|---:|---:|---:|---:|---:|---:|
| `2×4096` | 2123.6 | 10.2 | 1310.1 (68.7%) | 99.0 (5.2%) | 44.4 (2.3%) | 13.5 |
| `8×2048` | 1182.5 | 11.8 | 665.8 (63.8%) | 66.2 (6.3%) | 45.4 (4.4%) | 18.0 |

- `e62_diag128<4>` is dominant (64-69%) but already overlap-optimized (exp 065,
  variant 4) and at ~319 ns/row — the serial-dependency floor. NOT the lever.
- **Panel bounce-copy** (`src.copy_(dst)` after `torch.bmm(src, dinv^T, out=dst)`):
  31 calls on 2×4096 / 15 on 8×2048, 99us / 66us. This is the **exp 076 V2 lever,
  banked but not adopted** — re-capturable, and a bigger fraction here than on the
  smaller e62 shapes.
- **Copy-in clone** (`data.clone()` Memcpy DtoD): 44/45us at ~38% of HBM peak.
  Load-bearing for the benchmark loop (exp 076 refuted removing it), but a tuned
  float4 copy can beat torch's memcpy → byte-identical faster copy.

## [I1] Lever and Amdahl ceiling
Stack two byte-identical Ov levers on the two target shapes:
1. **Panel bounce-copy → float4 flat grid-stride write-back** (exp 076 V2, proven).
2. **Copy-in clone → float4 flat grid-stride memcpy** (faster than torch's, byte-identical).

Amdahl ceiling (both removable, byte-identical):
- `2×4096`: (99 + ~14 of 44 clone) / 2123 → ~1.06× ceiling
- `8×2048`: (66 + ~15 of 45 clone) / 1182 → ~1.07× ceiling
Grid dilution (6 e62 shapes move, 9 flat): a ~1.06×/1.07× on the two largest e62
shapes plus ~1.03× on the four smaller e62 shapes (bounce copy only) → grid
~1.5-2.0% if both levers land. FRONTIER class; ranked slot only if grid clears noise.

## [I2] N variants
- **V1** = exp 076 V2 (bounce-copy only) — the proven banked frontier, re-measured here
  for isolation.
- **V2** = V1 + float4 fast-clone replacing `data.clone()` (the stacked candidate).

## [I3.5]/[I4] N1 + N2 — PASS (V2)
6/6 e62 shapes bitwise-identical to the incumbent (baseline_bitwise_equal=True), 3x
deterministic, active backend, 0 fallbacks. 18/18 adversarial rows (high_kappa
1e6/1e8 + mixed_dynamic 1e10) checker_ok, finite, positive diagonal, 0 fallbacks.
See `results/077-v2-stress.json`.

## [I5] paired grid vs exact `#926462` (results/077-v2-paired-e62.json)
| Shape | ratio | CI95 | base→cand |
|---|---|---|---|
| `2×4096` | 1.0084× | [1.0070,1.0087] | 2158.0→2140.2us |
| `8×2048` | 1.0104× | [1.0091,1.0110] | 1204.1→1191.7us |
| `60×1024` | 1.0254× | [1.0231,1.0267] | 836.1→815.2us |
| `16×512` | 1.0061× | [1.0003,1.0107] | 299.8→298.3us |
| `4×1024` | 1.0035× | [0.9997,1.0067] | 538.6→537.1us |
| `2×2048` | 0.9970× | (noise, CI incl 1.0) | flat |

6-e62 geomean ≈ 1.0084×; full 15-shape grid projection ≈ 1.0034× (9 flat shapes
dilute). Both requested targets improve with CI excluding 1.0.

## [I6]/[I7] Classification — the clone lever is NEUTRAL
V2 ≈ V1 (exp 076) within per-shape noise on every e62 shape: 60×1024 1.0254×
(vs exp 076 1.0300×), 8×2048 1.0104× (vs 1.0118×), 2×4096 1.0084× (vs 1.0097×),
and 2×2048 0.9970× (vs 1.0039× — the fast-clone's launch overhead on the
small 8.4M-element matrix nets slightly negative). The clone already runs as a
Memcpy DtoD at ~76% of the copy floor (2×4096 134MB/44us = 3.05 TB/s; copy floor
~33.5us); a float4 SM kernel does not beat the CUDA copy engine. **The clone
is not a real second lever — stacking it does not clear LB noise.**

**V2 = PROMOTABLE FRONTIER** (byte-identical, N1/N2 clean, both targets faster with
CI excluding 1.0, below 2× Amdahl target). Full-grid magnitude ~1.0034× = sub-1.5%
= public coin-flip. exp 074 (1.00375×) lost public −3.27%; exp 076 (1.0036×,
`#927042`) did not displace `#926462`. V2's net performance is unchanged from
exp 076 → re-ranking is an unchanged-performance retry, barred by the
non-negotiable no-retry rule.

## [N7] fast_p
`fast_0 = 1/1` (V2 correct + byte-identical) · `fast_1 = 1/1` (e62 aggregate
faster) · `fast_targ (2.0×) = 0/1` (Amdahl ruled out).

## Disposition — BANK, do not rank
- The measured latency improvement on the two requested mid shapes is **real and
  documented** (`2×4096` 1.0084×, `8×2048` 1.0104×, both CI excluding 1.0).
- It is the exp 076 bounce-copy frontier, re-confirmed on the two largest e62
  shapes. The clone lever is neutral. No bigger lever exists: the diagonal block
  (`e62_diag128<4>`, 64-69%) is at the ~319 ns/row serial floor (exp 065
  overlap shipped); the trailing GEMMs are already TF32 (cuBLAS sm100, with a
  ~5% sm80 fallback slice not controllable without replacing cuBLAS); the clone
  is at ~76% of its floor; idle (10-12%) is the irreducible serial diag-block
  dependency chain.
- **V2 is NOT adopted.** Root stays on `#926462`. V2 is preserved as a
  byte-identical re-usable frontier that stacks with any future above-noise
  change (same as exp 076's disposition).
- No ranked submission: the frontier is sub-threshold and already attempted
  (`#927042`); re-ranking an unchanged-performance candidate violates the
  program's non-negotiable no-retry rule.

## Terminal state
`PROMOTABLE FRONTIER (banked)`. Root unchanged. Lease released.

