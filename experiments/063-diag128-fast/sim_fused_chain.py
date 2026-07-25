"""Bit-for-bit structural simulation of `e62_chain32_fused`.

Reproduces the exact register/lane bookkeeping of the CUDA kernel (4x8 tiles,
32 lanes, shared staging of the pivot column and of the pivot row of the
inverse) so the algebra and the column-protection trick are validated before
any GPU time is spent.
"""

import numpy as np


def fused_chain32(A):
    n = 32
    # r[lane][u][v] = A[4*ri+u][8*cj+v],  lane = ri*4 + cj
    r = np.zeros((32, 4, 8), dtype=np.float64)
    m = np.zeros((32, 4, 8), dtype=np.float64)
    for lane in range(32):
        ri, cj = lane >> 2, lane & 3
        i0, j0 = ri << 2, cj << 3
        for u in range(4):
            for v in range(8):
                r[lane, u, v] = A[i0 + u, j0 + v]
                m[lane, u, v] = 1.0 if (i0 + u) == (j0 + v) else 0.0

    Lk = np.zeros(32)
    Mk = np.zeros(32)

    for kb in range(4):
        for kv in range(8):
            k = (kb << 3) + kv
            ku = kv & 3
            kr = (kb << 1) + (kv >> 2)
            src = (kr << 2) + kb
            akk = r[src, ku, kv]
            d = 1.0 / np.sqrt(akk)

            for lane in range(32):
                ri, cj = lane >> 2, lane & 3
                i0, j0 = ri << 2, cj << 3
                if cj == kb:                       # owns column k
                    for u in range(4):
                        Lk[i0 + u] = r[lane, u, kv] * d if (i0 + u) >= k else 0.0
                if ri == kr:                       # owns row k of the inverse
                    for v in range(8):
                        m[lane, ku, v] *= d
                        Mk[j0 + v] = m[lane, ku, v]

            for lane in range(32):
                ri, cj = lane >> 2, lane & 3
                i0, j0 = ri << 2, cj << 3
                rowv = [Lk[i0 + u] for u in range(4)]
                colv = [Lk[j0 + v] for v in range(8)]
                mrow = [Mk[j0 + v] for v in range(8)]
                if cj == kb:
                    colv[kv] = 0.0
                rowm = [0.0 if (i0 + u) == k else rowv[u] for u in range(4)]
                for u in range(4):
                    for v in range(8):
                        r[lane, u, v] -= rowv[u] * colv[v]
                        m[lane, u, v] -= rowm[u] * mrow[v]
                if cj == kb:
                    for u in range(4):
                        r[lane, u, kv] = rowv[u]

    L = np.zeros((n, n))
    M = np.zeros((n, n))
    for lane in range(32):
        ri, cj = lane >> 2, lane & 3
        i0, j0 = ri << 2, cj << 3
        for u in range(4):
            for v in range(8):
                L[i0 + u, j0 + v] = r[lane, u, v]
                M[i0 + u, j0 + v] = m[lane, u, v]
    return L, M


def main():
    rng = np.random.default_rng(7)
    worst_l = worst_i = 0.0
    for trial in range(20):
        B = rng.standard_normal((32, 32))
        A = B @ B.T + 32 * np.eye(32)
        L, M = fused_chain32(A)
        ref = np.linalg.cholesky(A)
        worst_l = max(worst_l, np.abs(L @ L.T - A).max() / np.abs(A).max())
        worst_i = max(worst_i, np.abs(M @ L - np.eye(32)).max())
        assert np.allclose(np.triu(L, 1), 0), "L upper triangle not zeroed"
        assert np.allclose(np.triu(M, 1), 0), "inv upper triangle not zeroed"
        assert np.abs(L - ref).max() < 1e-9, "L differs from reference"
    print(f"max relative reconstruction error : {worst_l:.3e}")
    print(f"max |inv(L) @ L - I|              : {worst_i:.3e}")
    print("OK")


if __name__ == "__main__":
    main()
