# Experiment 043 result — `64x256` cuSOLVER-free CUDA reaches 2x

Status: **V35 WINNER / ADOPTED.** Exact ranked source `ranked-890037.py`,
SHA-256 `bc4536c700c95ba34f268d5a7aa6cc200ba9c403b0000ecc67abb15ec262fcb6`.

## Frozen control and constituent diagnosis

The frozen control is ranked submission `#888996`, source SHA-256
`5bbb8e8de3eedfadfb70fdcfaa902723fab03c2998bc18e9cb80512e82cce80c`.
Its `64x256` path measured 219.0us wall / 195.0us device over 30 operations:
105.96us in eight serial diagonal factorizations, 55.14us in panel inner/apply
work, 8.21us in the trailing update, 8.80us in two device copies, about 15.31us
in finite/output bookkeeping, and 24.0us wall-minus-device gaps. The dominant
cost was serialized dependency and launch latency, not a FLOP or DRAM floor.
Nsight Compute was attempted but unavailable with `LibraryNotLoaded`; no
counter claim was inferred from that failed probe.

## Architecture ladder

One CUDA CTA now owns each 256x256 matrix. It stages packed lower 16x16 tiles
in shared memory, factors FP32 diagonal blocks, solves panels in registers, and
uses warp-level TF32 WMMA for the dense rank-16 trailing updates. The first
fast pass detects small Schur pivots. Difficult inputs are restaged and retried
with scalar FP32 trailing updates; this accurate path is slower but rare and
removes the compile-heavy TF32x3 WMMA fallback.

The important measured steps were:

- V1 scalar trailing: 200.90us device; trailing alone was 130.58us.
- V7 first WMMA layout: 178.18us device.
- V9 scheduling/staging reduction: 135.58us device.
- V11 best instrumented precursor: 109.70us device, split into 12.19us staging,
  50.72us diagonal, 11.04us panel, 25.57us trailing, and 9.70us output.
- V13 dense finalist: 232.14 -> 117.55us = 1.977x.
- V18 fused output traversal: 224.11 -> 110.32us = 2.032x, but spectrum and
  low-rank families failed.
- V22 manual round-to-nearest TF32x3 retry passed all families but cost 174us.
- V26 adaptive fast/accurate selection restored dense latency near 118us.
- V28/V29 inlined the fast path and kept the accurate retry out of line. The
  grid reached 2.039x, but Popcorn tests `#889891`, `#889923`, and `#889940`
  hit the exact six-minute compile limit.
- V33 produced a useful public leaderboard probe (`#890008`, 823.02us) but its
  cold secret validation also hit six minutes.
- V35 replaced only the rare TF32x3 fallback with scalar FP32 and removed its
  scratch tiles. Popcorn compile/test time fell to 85.3 seconds while the dense
  tensor-core path became slightly faster from the smaller shared allocation.

## Final gates

- Six-family V35 gate: 6/6 active, no fallback/error metadata. Scaled residuals:
  dense 17.9, spectrum 0.00858, diagonal 0.00391, low-rank 0.00828,
  row-scaled 0.00739, and tridiagonal 0.551 (official limit 20).
- Exact V35 full grid: 15/15 correct. Target **225.192 -> 111.608us =
  2.0177x**, CI [2.0137, 2.0216]. All other shapes stayed at parity; aggregate
  **1.047717x**, CI [1.046994, 1.048440].
- The final quick audit's same-process ratio was **2.0477x**, CI
  [2.0432, 2.0522]. Its frozen-absolute verdict was conservatively rejected
  because that run's 114.688us exceeded the 112.652us line; a prior exact-math
  audit measured 112.200us and accepted, and the exact final grid crossed 2x.
- Popcorn test `#890035`: **17/17**.
- Ranked `#890037`: **825.4657219594694us public / 824.9085045342571us
  secret**. Versus `#888996`, latency improved **9.940% public / 4.508%
  secret**. All public and secret test, benchmark, and leaderboard stages passed.

The root `submission.py`, `candidate-v35.py`, and `ranked-890037.py` are
byte-identical. V33 and failed/test artifacts are retained because they explain
the compile-budget defect and justify the single ranked retry after a concrete
fix.

## Post-rank integration decision

After exp 044 landed concurrently, its non-overlapping `640x512` and `60x1024`
diagonal micro was rebased together with V35. The exact combined source passed
the 15-shape paired grid against `#890037` at **1.013042x**, CI
**[1.012580, 1.013503]**: `640x512` improved 1.1003x, `60x1024` improved
1.1079x, and `64x256` remained at parity with the ranked 2x kernel. All family
checker results were correct; the only inactive rows were exp 044's already
documented safety fallbacks on difficult 512/1024 inputs.

The official combined test `#890068` nevertheless failed at exactly the
six-minute compile limit. It was not ranked. `submission.py` was therefore
restored to exact successful ranked source `#890037`, and the goal closes on
that verified candidate. `combined-v35-four-extension.py`, the paired/family
artifacts, and `popcorn-test-combined-890068.json` preserve the rejected
integration evidence.
