# Experiment 022 — standalone rank-4 n=32 kernel

**Status: REJECTED AFTER MIXED LEADERBOARD RESULT.** Exact baseline is ranked
experiment 021 / submission `#882958` (`baseline-exp021.py`). Candidate
SHA-256: `8de4b8efe3d6a2dd89369e74db7a24d3f96cd6864044fabdc167cbb56a9bab15`.

`candidate-rank4.py` transfers the proven rank-4 scalar pivot chain from the
split32 microfactorization to the standalone `4096x32` path. It changes only
the n=32 Triton kernel; all other dispatches are byte-identical to the ranked
baseline.

## Evidence

The paired target probe passed all six families and improved `4096x32`
`39.7us -> 36.6us` (**1.084x**); the candidate kernel profiled at `34.3us`.
The full 15-shape grid passed and improved `1128.5us -> 1122.7us`
(**1.0052x**), with the target reproducing at **1.077x** and no material
off-target regression.

Popcorn test `#882968` passed **17/17**. Exactly one leaderboard submission,
`#882969`, passed every stage but produced a mixed result:
**1112.6302190816483us public / 1093.6676344172347us secret**. Versus
`#882958`, public regressed **1.5096%** while secret improved **1.4399%**.
Because the public/current score did not improve, the candidate is not adopted
and root `submission.py` remains exact `#882958`. No duplicate retry was made.

Artifacts: `probe-rank4.json`, `fullgrid.json`, `test-882968.json`, and
`ranked-882969.json`.
