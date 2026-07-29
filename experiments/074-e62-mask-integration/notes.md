# Experiment 074 — integrate e62 mask-v2 frontier

## Goal and promotion rule

Integrate exp071's correct hybrid output-mask geometry onto ranked `#926462`,
improve at least two whole shapes, and improve the public leaderboard geomean.
Promotion rule remains the owner-selected **optimize public, accept secret** rule
recorded in `docs/STATUS.md` for the live incumbent.

## Exact source identities

- Ranked control: commit `1ec14b7`, SHA-256 `582cde16...b869ff723`, Popcorn
  `#926462`, public `630.4029288615614us`.
- Integration candidate: byte-identical to
  `experiments/071-e62-mask-v2/candidate_v4_hybrid.py`, SHA-256
  `26dfc03f86fc04fbf998b7605998c62ea77a34876847682599e4d7ce5f8f8f36`.

The candidate uses four row-owned warps per mask CTA on `16x512`, `4x1024`,
`60x1024`, and `8x2048`; batch-2 `2x2048`/`2x4096` retain the incumbent row
kernel. Arithmetic and the lower triangle are unchanged; all variants produced
an output bitwise equal to the incumbent.

## Pre-integration evidence

See `experiments/071-e62-mask-v2/`:

- N1/N2: 3x bitwise determinism on all six shapes and 18/18 adversarial rows;
- verify 24/24 and family 48/48 checker passes with zero new fallbacks;
- paired eight-row geomean `1.00509x`, CI95 `[1.00445,1.00572]`;
- four whole-shape CIs above parity, led by `60x1024` at `1.03406x`.

Full-grid, cold-build, Popcorn, adoption, and ranked evidence are recorded here.

## Full 15-shape paired gate

`results/074-fullgrid.json`: **PASS**, all 15 rows checker-ok, identical route
counters, zero new fallbacks. Geomean `1.003750x`, CI95
`[1.002814,1.004687]` excludes parity.

| Shape | Control us | Candidate us | Speedup |
|---|---:|---:|---:|
| `16x512` | 296.296 | 292.712 | 1.01544x |
| `4x1024` | 527.440 | 524.736 | 1.00429x |
| `60x1024` | 832.988 | 807.820 | 1.03129x |
| `8x2048` | 1197.732 | 1193.968 | 1.00398x |

All four per-shape CI95 intervals are above 1.0. Every off-target row is flat
within its paired interval; the largest A-vs-A spread is 1.10% on `4x1024`.

## Clean cold build

`results/074-cold-verify.json`: exact candidate in a new Modal sandbox, **57/57
passed**. Image assembly was 1.22s; the CUDA extension compiled from an empty
sandbox cache (warp4 mask 15 registers, incumbent row mask 19 registers, zero
spills/barriers). End-to-end build plus validation was about 91s, comfortably
under 80% of the observed six-minute Popcorn service boundary (288s).

## Popcorn test

Exact `submission.py` test `#926716`: **17/17 passed**, B200, terminal
`succeeded` in 30s. See `results/074-popcorn-test.json`.

## Ranked result and verdict

The single ranked attempt `#926737` reached terminal `succeeded` after 185s;
test, benchmark, and leaderboard phases passed on both public and secret splits.
The CLI now exposes both component geomeans:

| Split | `#926462` control | `#926737` candidate | Change |
|---|---:|---:|---:|
| Public | 630.403us | 651.017us | **3.27% slower** |
| Secret | 670.301us | 626.486us | **6.54% faster** |

See `results/074-popcorn-ranked.json`. This is the inverse of exp065: a fully
qualified, bit-identical latency reorchestration improved secret substantially
but regressed the public board. Under the named **optimize public, accept
secret** rule, the public regression is disqualifying. The candidate is
therefore **REJECTED AT LEADERBOARD**, and root `submission.py` is restored to
the exact `#926462` SHA-256 `582cde16...b869ff723`.

The local kernel outcome remains valid: four shapes improved with per-shape
CI95 above parity, satisfying the requested multi-shape latency objective. The
leaderboard objective did not close because the public split moved against the
same-process evidence; no second ranked attempt was launched.
