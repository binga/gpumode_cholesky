# Experiment 052 outcome

The shipped block-scaled MXFP8 path was transferred from `1x32768` to the two
smaller large-matrix shapes. At `1x8192` it was only `1.001591x` and raised the
dense residual to `13.7/20`, so V1 was rejected.

At `1x16384`, conditionally raising the E8M0 shared exponent only when a
block's normalized amax would exceed E4M3 max reduced the former `10.1/20`
residual to `9.03/20` at `1.031050x`. One final TF32 panel reached `8.85/20` at
`1.026446x`. Moving the second half of the panel sequence to TF32 finally met
the `8/20` margin: exact standalone V4 measured `15150.160 -> 14950.640us`,
`1.013583x`, with dense residual `7.85/20`.

V4 is not promotable. Spectrum, low-rank, and row-scaled families made the
active factor non-finite and invoked the incumbent fallback. Only dense,
diagonal, and tridiagonal stayed on the new backend. No full grid, build gate,
Popcorn submission, or root-source change followed.

The incumbent `1x16384` profile explains the low return: `5051.1us` (35.1%) is
eight FP32 2048-block POTRF calls and `4364.9us` (30.3%) is 56 triangular-solve
kernels. TF32 GEMMs are the third constituent. The next experiment should
target recursive-inverse base size and then diagonal factorization, not panel
precision.
