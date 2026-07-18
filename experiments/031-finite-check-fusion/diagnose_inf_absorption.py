"""Why the last-diagonal finite check is NOT equivalent (experiment 031 refutation).

The premise recorded in the journal was: "NaN provably propagates to the last
diagonal entry". The propagation argument holds for NaN but NOT for Inf, because
the column solve divides by the pivot:

    l[i][k] = (A[i][k] - ...) / l[k][k]

If l[k][k] overflows to +Inf, then finite / Inf == 0. The Inf is *absorbed into
zeros* rather than propagated. The trailing update then subtracts a zero outer
product, leaving the remaining submatrix finite, and the last diagonal entry
comes out perfectly finite while an earlier diagonal entry is Inf.

This makes the last-diagonal-only check strictly weaker than the shipped
full-diagonal check: it silently accepts a factor the shipped code correctly
rejects and routes to the cuSOLVER fallback.
"""

import numpy as np


def unblocked_cholesky_f32(a):
    a = np.array(a, dtype=np.float32, copy=True)
    n = a.shape[0]
    l = np.zeros((n, n), dtype=np.float32)
    with np.errstate(all="ignore"):
        for k in range(n):
            s = a[k, k] - np.dot(l[k, :k], l[k, :k])
            d = np.float32(np.sqrt(np.float32(s)))
            l[k, k] = d
            if k + 1 < n:
                r = a[k + 1 :, k] - l[k + 1 :, :k] @ l[k, :k]
                l[k + 1 :, k] = (r / d).astype(np.float32)
    return l


def main():
    # Minimal 3x3 SPD matrix whose second pivot overflows float32.
    # Row 1 is scaled by ~1e20 so that A[1][1] ~ 1e40 -> +Inf in float32.
    big = np.float32(1e20)
    a = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, float(big) * float(big), 0.0],
            [0.0, 0.0, 9.0],
        ],
        dtype=np.float32,
    )

    print("input diagonal (float32):", np.diag(a))
    l = unblocked_cholesky_f32(a)

    print("\nfactor L:")
    print(l)
    print("\ndiag(L):            ", np.diag(l))
    print("isfinite(diag(L)):  ", np.isfinite(np.diag(l)))

    full = bool(np.isfinite(np.diag(l)).all())
    last = bool(np.isfinite(l[-1, -1]))

    print(f"\nshipped check   isfinite(diag).all()  -> {full}")
    print(f"exp-031 check   isfinite(L[-1,-1])     -> {last}")

    if full != last:
        print("\nREFUTED: the two checks disagree.")
        print("The shipped check rejects this factor and falls back to cuSOLVER;")
        print("the exp-031 check accepts a factor containing a +Inf diagonal.")
    else:
        print("\n(no disagreement on this input)")

    # Show the absorption directly: an Inf pivot produces a zero column.
    print("\nmechanism: pivot L[1][1] =", l[1, 1])
    print("           column below it  =", l[2, 1], "  (finite / Inf -> 0)")
    print("           so the Inf never reaches L[2][2] =", l[2, 2])


if __name__ == "__main__":
    main()
