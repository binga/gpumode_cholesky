# Experiment 053: reduce the `1x16384` factor/solve stack

## Evidence and target

Exact ranked `#890798` at `1x16384` measures `15075.2us` wall and `14403.6us`
device. The dominant constituents are:

- eight FP32 2048-block POTRF kernels: `5051.1us` (`35.1%` device),
- 56 triangular-solve kernels from recursive inverses: `4364.9us` (`30.3%`),
- all tensor-core GEMMs: less than `18%` combined.

The current inverse recursion bottoms out at 512, producing four base solves
and three internal combine nodes per 2048 diagonal block. The first bounded
axis is base size: 1024, then 2048, then 256 only if needed. This changes no
precision or mathematical formula.

If inverse tuning wins, the second axis is replacing each 2048 diagonal POTRF
with the existing family-validated split32 factorization. Each axis is measured
separately before combination. A promotable result requires positive route
counters, no fallback/error, all six families, exact-source full paired grid,
clean build, Popcorn 17/17, and both public and secret improvement.
