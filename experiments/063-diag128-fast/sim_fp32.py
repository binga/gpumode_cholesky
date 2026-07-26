"""Float32 replay of the round-2 panel update, to separate arithmetic from race.

The GPU build of `e62_panel32` returns 1.8e-2 where the float64 structural
simulation returns 5.3e-15. Either the CUDA translation has a memory-ordering
defect, or the float32 arithmetic of the *raw* formulation

    A[i][j] -= (raw_i * d2) * raw_j,      d2 = d*d,  d = rsqrt(A[k][k])

is far worse than the scaled formulation the shipped kernel uses

    A[i][j] -= L[i][k] * L[j][k],         L = raw * d

This replays both in float32 against a float64 reference. If neither
reproduces ~1e-2, the defect is a race and no amount of algebra will find it.
"""

import numpy as np

f32 = np.float32


def chol_raw_d2(A):
    """The round-2 formulation: stage raw, scale after, single d2 multiply."""
    S = A.astype(f32).copy()
    n = S.shape[0]
    for k in range(n):
        akk = S[k, k]
        d = f32(1.0) / np.sqrt(akk, dtype=f32)
        d2 = f32(d * d)
        col = S[k:, k].copy()                      # raw pivot column
        rr = (col * d2).astype(f32)
        S[k:, k:] = (S[k:, k:] - np.outer(rr, col)).astype(f32)
        S[k:, k] = (col * d).astype(f32)           # protected column write-back
    return np.tril(S)


def chol_scaled(A):
    """The shipped formulation: scale first, then a symmetric rank-1 update."""
    S = A.astype(f32).copy()
    n = S.shape[0]
    for k in range(n):
        d = f32(1.0) / np.sqrt(S[k, k], dtype=f32)
        col = (S[k:, k] * d).astype(f32)
        S[k:, k:] = (S[k:, k:] - np.outer(col, col)).astype(f32)
        S[k:, k] = col
    return np.tril(S)


def main():
    rng = np.random.default_rng(3)
    for n in (128,):
        for trial in range(4):
            B = rng.standard_normal((n, n))
            A = B @ B.T / n + np.eye(n)
            A = A / np.abs(A).max() * 1.09          # match the probe's scale
            ref = np.linalg.cholesky(A.astype(np.float64))
            e_raw = np.abs(chol_raw_d2(A) - ref).max()
            e_sca = np.abs(chol_scaled(A) - ref).max()
            print(f"n={n} trial={trial}  raw/d2 err={e_raw:.3e}   "
                  f"scaled err={e_sca:.3e}")


if __name__ == "__main__":
    main()
