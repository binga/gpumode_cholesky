# Experiment 071 — e62 output-mask launch geometry v2

## Goal and frozen baseline

Run the bounded `program2.md` inner loop on the shared `e62_zero_upper` path,
improving at least two shapes if possible. Priority shapes are `60×1024` and
`8×2048`; all six e62 shapes are in scope. Frozen control is ranked `#926462`,
commit `1ec14b72694dceed3d91cbf5ecff303c4ca5be80`, submission SHA-256
`582cde1648b8b3e9d77a36173dd59cd36588123ae28800ca00e5342b869ff723`.

## Amdahl ceiling

Exp069 measured the old torch mask at 145us and the whole-path transition
`936.1→852.6us` on `60×1024`, implying roughly 61.5us remains in the shipped
mask. Even deleting that cost entirely caps this sub-lever near 1.08×. Exp069's
whole-path mask gain on `8×2048` was only 1.0478×. Therefore 2.00× is impossible
for this bounded sub-lever; the valid terminal outcome is a reproducible
multi-shape frontier or a bounded rejection.

## N=3 variants

- `candidate_v1_warp8.py`: eight coalesced warp-owned rows per 256-thread CTA.
- `candidate_v2_warp4.py`: four coalesced warp-owned rows per 128-thread CTA.
- `candidate_v3_tile64.py`: 64×64 output tiles, with strict-upper predicates.

Every launch omits the fourth execution-configuration argument and therefore
uses the default CUDA stream. None changes factorization arithmetic or reads,
mutates, or overwrites lower-triangular output values.

## Gates

Free gates passed. Modal N1/N2, normal verify/family correctness, and paired
same-process latency are pending; raw JSON will be stored under `results/`.

### N1/N2 result

All three variants passed on all six exact e62 shapes. N1 retained three
same-input outputs per shape and found bitwise equality across repeats and to
the incumbent. N2 passed all 18 exact shape×condition rows per variant at
condition exponents 6/8/10; exponent 10 interleaved the largest and smallest
diagonals for a mixed-dynamic near-singular case. The e62 backend was active,
there were zero fallbacks, and the worst official tolerance fraction was only
0.01465%.

### Same-extension mask probe

`results/071-maskprobe.json` timed all four mechanisms in one extension. V2 won
the six-shape kernel geomean by a narrow margin: 1.24978× vs v1 1.24871× and v3
1.04989×. It cut the mask `56.288→29.696us` on `60×1024` (1.895×) and
`21.472→19.344us` on `8×2048` (1.110×), and also won on `16×512` and
`4×1024`; it tied/slightly lost at `2×2048` and lost `17.376→19.456us` at
`2×4096`. Select v2 for whole-shape gates, bank v1 as a tied frontier, reject
v3.

## Refinement and finalist

V2's probe was a wash at `2×2048` and 0.893× at `2×4096`, so v4 keeps the
incumbent one-block-per-row kernel when `batch==2 && n>=2048` and uses warp4 on
the four winning shapes. V4 repeated the full N1/N2 gate successfully: every
output remained bitwise equal to the incumbent, all 18 adversarial rows passed,
the e62 backend was active, and there were zero fallbacks.

Normal verify passed 24/24. The candidate and exact incumbent family grids each
had 48/48 `checker_ok`; a mechanical ordered-row comparison found zero
differences in checker status/message, active backend, counters, fallbacks, or
errors. The 14 top-level fallback rows are therefore all pre-existing.

## Same-process whole-shape result

| Shape | Control us | V4 us | Speedup | CI95 |
|---|---:|---:|---:|---:|
| `16×512` | 284.552 | 282.320 | **1.00577×** | [1.00326, 1.01305] |
| `4×1024` | 509.584 | 509.092 | **1.00095×** | [1.00035, 1.00140] |
| `60×1024` | 821.596 | 794.412 | **1.03406×** | [1.03331, 1.03437] |
| `2×2048` | 995.296 | 996.756 | 0.99872× | [0.99761, 0.99908] |
| `8×2048` | 1182.280 | 1179.864 | **1.00226×** | [1.00132, 1.00257] |
| `2×4096` | 2124.656 | 2125.944 | 0.99927× | [0.99903, 0.99958] |

The exact six-e62-shape geomean is **1.00676×**. The eight-row filtered grid
(including unchanged `640×512` and `1×4096`) is **1.00509×**, CI95
[1.00445, 1.00572]. Every paired row passed with zero new fallbacks. Four e62
shapes improve with CIs above parity, satisfying the ≥2-shape goal. The two
batch-2 siblings execute the preserved shipped row-kernel source and geometry;
their 0.128%/0.073% cross-module deltas are unchanged-path context noise below
the paired run's 0.33% maximum A-vs-A floor, not mask-mechanism regressions. The
candidate remains aggregate-positive and is returned as a `PROMOTABLE
FRONTIER`, not integrated here.

`fast_p`: `fast_0=4/4` correct including the refinement; `fast_1=1/1`
whole-path paired finalist faster on the exact e62 aggregate (the initial
same-extension batch was 3/3 aggregate kernel-faster); `fast_targ=0/1` reaches
2.00×, which the recorded Amdahl ceiling rules out.

The store-only mask reaches about 4.23 TB/s on `60×1024` (~52.9% of the 8 TB/s
B200 HBM ceiling) and 3.47 TB/s on `8×2048` (~43.3%), so it is not roofline
saturated. No extra profiler run was needed to classify a write-only kernel
whose same-extension device timing already isolates the mechanism.

## Verdict

**PROMOTABLE FRONTIER: return `candidate_v4_hybrid.py` to integration.** The parent owns
the full 15-shape, cold-build, Popcorn, leaderboard, adoption, and root-doc
gates. This worker does not touch root `submission.py` or submit a ranked job.
