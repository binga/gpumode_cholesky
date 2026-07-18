"""Free gate for experiment 031: last-diagonal finite check == full-diagonal check.

Experiment 031 replaces the safety-net test

    torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all()      # batch*n elements

with

    torch.isfinite(l[..., -1, -1]).all()                    # batch elements

on the split32 and 8x2048 dispatch sites. The claim is that this is *exactly*
equivalent, not merely a cheap approximation:

  Any non-finite value produced by a forward (right-looking, blocked) Cholesky
  provably propagates to the bottom-right diagonal entry l[n-1][n-1].

Argument. A non-finite entry can only originate at a diagonal step, where
l[k][k] = sqrt(A[k][k] - sum_j l[k][j]^2) goes NaN (negative radicand) or the
subsequent column solve divides by l[k][k] == 0 (-> +-Inf or 0/0 -> NaN). Every
column k < n-1 is used to solve rows below it, and row n-1 is below every such
k, so l[n-1][k] becomes non-finite. The last diagonal is then
l[n-1][n-1] = sqrt(A[n-1][n-1] - sum_{k<n-1} l[n-1][k]^2), whose radicand
contains that non-finite term. NaN/Inf never return to finite under +, -, *, /
or sqrt, so l[n-1][n-1] is non-finite. If instead k == n-1 is the first failing
step, the last diagonal is non-finite directly.

This gate checks the argument empirically on float32 with the same right-looking
blocked structure the split32 chain uses (panel solve + trailing Schur update),
over input families chosen to actually trip the failure: indefinite, singular,
near-singular and heavily ill-conditioned matrices.

Pass condition: over every trial, "some entry of L is non-finite" is TRUE
exactly when "L[n-1][n-1] is non-finite" is TRUE. A single trial where the
factor goes non-finite but the last diagonal stays finite would falsify the
change and must block the experiment.

Run: python3 gate_nan_propagation.py
"""

import numpy as np


def blocked_cholesky_f32(a, nb=32):
    """Right-looking blocked Cholesky in float32, mirroring the split32 chain.

    Deliberately performs no pivoting, no clamping and no early exit, so a
    non-finite value propagates exactly as it does in the Triton kernels.
    """
    a = np.array(a, dtype=np.float32, copy=True)
    n = a.shape[0]
    l = np.zeros((n, n), dtype=np.float32)
    for j in range(0, n, nb):
        je = min(j + nb, n)
        # Diagonal block: unblocked right-looking factorization.
        for k in range(j, je):
            with np.errstate(all="ignore"):
                s = a[k, k] - np.dot(l[k, j:k], l[k, j:k])
                d = np.sqrt(s.astype(np.float32))
                l[k, k] = d
                if k + 1 < je:
                    r = a[k + 1 : je, k] - l[k + 1 : je, j:k] @ l[k, j:k]
                    l[k + 1 : je, k] = (r / d).astype(np.float32)
        if je < n:
            with np.errstate(all="ignore"):
                # Panel solve against the just-factored diagonal block.
                lb = l[j:je, j:je]
                rhs = a[je:, j:je] - l[je:, :j] @ l[j:je, :j].T
                panel = np.zeros((n - je, je - j), dtype=np.float32)
                for c in range(je - j):
                    acc = rhs[:, c] - panel[:, :c] @ lb[c, :c]
                    panel[:, c] = (acc / lb[c, c]).astype(np.float32)
                l[je:, j:je] = panel
                # Trailing Schur update.
                a[je:, je:] -= (panel @ panel.T).astype(np.float32)
    return l


def families(rng, n):
    """Inputs engineered to drive the factorization non-finite."""
    out = {}

    # Indefinite: symmetric with negative eigenvalues -> sqrt of a negative.
    m = rng.standard_normal((n, n)).astype(np.float32)
    out["indefinite"] = ((m + m.T) / 2).astype(np.float32)

    # Exactly singular: rank-deficient Gram matrix -> division by a zero pivot.
    r = rng.standard_normal((n, max(1, n // 4))).astype(np.float32)
    out["singular"] = (r @ r.T).astype(np.float32)

    # Near-singular: tiny positive floor on a rank-deficient Gram.
    out["near_singular"] = (r @ r.T + np.float32(1e-30) * np.eye(n, dtype=np.float32)).astype(
        np.float32
    )

    # Extreme spectrum: condition number far past float32 resolution.
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.logspace(0, -30, n)
    out["spectrum"] = (q @ np.diag(ev) @ q.T).astype(np.float32)

    # Huge dynamic range: row scaling that overflows the trailing update.
    scale = np.float32(1e18) ** rng.integers(0, 2, size=n).astype(np.float32)
    base = rng.standard_normal((n, n)).astype(np.float32)
    spd = (base @ base.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)
    out["overflow"] = (spd * np.outer(scale, scale)).astype(np.float32)

    # Well-conditioned control: must stay finite under both checks.
    good = rng.standard_normal((n, n)).astype(np.float32)
    out["spd_control"] = (good @ good.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)

    return out


def main():
    rng = np.random.default_rng(31031)
    mismatches = []
    counts = {}
    trials = 0

    for n in (32, 64, 128, 160):
        for nb in (32, 64):
            if nb > n:
                continue
            for rep in range(12):
                for name, a in families(rng, n).items():
                    l = blocked_cholesky_f32(a, nb=nb)
                    trials += 1

                    any_nonfinite = not np.isfinite(l).all()
                    # The exact predicate the shipped code uses today.
                    full_diag_nonfinite = not np.isfinite(np.diag(l)).all()
                    # The exact predicate experiment 031 substitutes.
                    last_diag_nonfinite = not np.isfinite(l[-1, -1])

                    key = (name, any_nonfinite)
                    counts[key] = counts.get(key, 0) + 1

                    # The substitution must agree with the shipped predicate,
                    # and must also catch a non-finite anywhere in the factor.
                    if full_diag_nonfinite != last_diag_nonfinite or (
                        any_nonfinite != last_diag_nonfinite
                    ):
                        mismatches.append(
                            {
                                "family": name,
                                "n": n,
                                "nb": nb,
                                "rep": rep,
                                "any_nonfinite": bool(any_nonfinite),
                                "full_diag_nonfinite": bool(full_diag_nonfinite),
                                "last_diag_nonfinite": bool(last_diag_nonfinite),
                            }
                        )

    print(f"trials: {trials}")
    print("\nnon-finite factor produced, by family:")
    for name in sorted({k[0] for k in counts}):
        tripped = counts.get((name, True), 0)
        clean = counts.get((name, False), 0)
        print(f"  {name:<14} non-finite {tripped:>3} / finite {clean:>3}")

    tripped_total = sum(v for k, v in counts.items() if k[1])
    print(f"\ntrials that actually went non-finite: {tripped_total}")

    if tripped_total == 0:
        print("\nGATE INCONCLUSIVE: no trial produced a non-finite factor, so the")
        print("equivalence was never exercised. Strengthen the input families.")
        return 2

    if mismatches:
        print(f"\nGATE FAILED: {len(mismatches)} mismatch(es)")
        for m in mismatches[:10]:
            print(f"  {m}")
        return 1

    print("\nGATE PASSED: last-diagonal check agreed with the full-diagonal check")
    print("on every trial, and caught every non-finite factor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
