"""Structural simulation of `e62_panel32` (experiment 063 variant 2).

All 256 threads factor the 128x32 column panel together and carry the 32x32
inverse of the diagonal block alongside it, so the block kernel's three
single-warp phases (chain, triangular inverse, panel solve) collapse into one
fully parallel phase. This reproduces the exact thread/register bookkeeping.
"""

import numpy as np

TB = 128
NT = 256


def panel32(S, kk, M_in=None):
    """Factor S[:, kk:kk+32] in place and return the 32x32 inverse."""
    t = np.zeros((NT, 4, 4))
    mt = np.zeros((NT, 4, 4))
    ib = kk >> 2
    for tid in range(NT):
        tr, tc = tid >> 3, tid & 7
        r0, c0 = tr << 2, kk + (tc << 2)
        for u in range(4):
            for v in range(4):
                t[tid, u, v] = S[r0 + u, c0 + v]
        if ib <= tr < ib + 8:
            mrow0 = r0 - kk
            for u in range(4):
                for v in range(4):
                    mt[tid, u, v] = 1.0 if (mrow0 + u) == ((tc << 2) + v) else 0.0

    Lc = np.zeros((2, TB))
    Mr = np.zeros((2, 32))

    for kq in range(8):
        for kv in range(4):
            kl = (kq << 2) + kv
            k = kk + kl
            lc, mr_ = Lc[kl & 1], Mr[kl & 1]

            for tid in range(NT):                       # staging (raw)
                tr, tc = tid >> 3, tid & 7
                r0 = tr << 2
                if tc == kq:
                    for u in range(4):
                        lc[r0 + u] = t[tid, u, kv] if (r0 + u) >= k else 0.0
                if tr == ib + kq:
                    for v in range(4):
                        mr_[(tc << 2) + v] = mt[tid, kv, v]

            # __syncthreads()
            akk = lc[k]
            d = 1.0 / np.sqrt(akk)
            d2 = d * d

            for tid in range(NT):
                tr, tc = tid >> 3, tid & 7
                r0, c0 = tr << 2, kk + (tc << 2)
                rowv = [lc[r0 + u] for u in range(4)]
                colv = [lc[c0 + v] for v in range(4)]
                if tc == kq:
                    colv[kv] = 0.0
                rr = [rowv[u] * d2 for u in range(4)]
                for u in range(4):
                    for v in range(4):
                        t[tid, u, v] -= rr[u] * colv[v]
                if tc == kq:
                    for u in range(4):
                        t[tid, u, kv] = rowv[u] * d
                if ib <= tr < ib + 8:
                    mrv = [mr_[(tc << 2) + v] * d for v in range(4)]
                    rm = [0.0 if (r0 + u) == k else rowv[u] * d for u in range(4)]
                    for u in range(4):
                        for v in range(4):
                            mt[tid, u, v] -= rm[u] * mrv[v]
                    if tr == ib + kq:
                        for v in range(4):
                            mt[tid, kv, v] = mrv[v]

    Qi = np.zeros((32, 32))
    for tid in range(NT):
        tr, tc = tid >> 3, tid & 7
        r0, c0 = tr << 2, kk + (tc << 2)
        for u in range(4):
            for v in range(4):
                S[r0 + u, c0 + v] = t[tid, u, v]
        if ib <= tr < ib + 8:
            mrow0 = r0 - kk
            for u in range(4):
                for v in range(4):
                    Qi[mrow0 + u, (tc << 2) + v] = mt[tid, u, v]
    return Qi


def full_block(A):
    """The whole 128x128 block: four panels plus the rank-32 trailing update."""
    S = A.copy()
    for kk in range(0, TB, 32):
        Qi = panel32(S, kk)
        L11 = S[kk:kk + 32, kk:kk + 32]
        assert np.abs(Qi @ L11 - np.eye(32)).max() < 1e-9, f"inverse wrong at {kk}"
        lw = kk + 32
        if lw < TB:
            L21 = S[lw:, kk:lw]
            S[lw:, lw:] -= L21 @ L21.T
    return np.tril(S)


def main():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(6):
        B = rng.standard_normal((TB, TB))
        A = B @ B.T + TB * np.eye(TB)
        L = full_block(A)
        ref = np.linalg.cholesky(A)
        worst = max(worst, np.abs(L - ref).max())
    print(f"max |L - reference| over the whole 128 block : {worst:.3e}")
    print("OK")


if __name__ == "__main__":
    main()
