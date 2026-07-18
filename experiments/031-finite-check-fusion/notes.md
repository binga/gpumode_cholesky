# Experiment 031 — cheap finite check → REJECTED at the free gate

Baseline: exact ranked winner `#883174`
(`e072778cef0aec070e13e2093c7be7a98f2de74211fe6d2704cce5370fcd02e5`,
public 1084.457us). **No Modal or popcorn quota was spent** — the candidate died
on a free CPU gate.

## The premise was wrong

The recorded next-step said NaN "provably propagates to the last diagonal
entry", licensing `isfinite(l[..., -1, -1])` as an exact replacement for
`isfinite(l.diagonal(...)).all()`.

The propagation argument holds for **NaN** but fails for **Inf**, because the
column solve *divides* by the pivot:

    l[i][k] = (A[i][k] - ...) / l[k][k]

If `l[k][k]` overflows to `+Inf`, then `finite / Inf == 0`. The Inf is
**absorbed into a column of zeros** instead of propagating. The trailing update
then subtracts a zero outer product, the remaining submatrix stays finite, and
the last diagonal entry comes out finite while an earlier diagonal is `Inf`.

Minimal refutation (`diagnose_inf_absorption.py`, float32):

```
A = diag(4, 1e40, 9)        ->  L = diag(2, inf, 3)

shipped check   isfinite(diag(L)).all()  -> False   (falls back to cuSOLVER)
exp-031 check   isfinite(L[-1,-1])       -> True    (accepts an inf factor)

pivot   L[1][1] = inf
column  L[2][1] = 0.0        finite / inf -> 0
so      L[2][2] = 3.0        finite; the inf never reaches it
```

So the substitution is **strictly weaker** than the shipped predicate: it
silently accepts factors the shipped code correctly rejects. That is a
correctness gate weakened for ~1-3%, which program.md forbids outright.

## Free gate evidence (`gate_nan_propagation.py`)

504 trials of a float32 right-looking blocked Cholesky (panel solve + trailing
Schur update, mirroring the split32 structure), n ∈ {32,64,128,160},
nb ∈ {32,64}, six input families. Pass condition: the two predicates agree on
every trial.

| family | non-finite trials | predicate mismatches |
|---|---:|---:|
| indefinite | 84 | 0 |
| singular | 84 | 0 |
| near_singular | 84 | 0 |
| spectrum | 84 | 0 |
| overflow (Inf pivots) | 24 | **22** |
| spd_control | 0 | 0 |

360 trials went non-finite, so the equivalence was genuinely exercised. Every
mismatch is an Inf-pivot case (n=160, 11 at nb=32 and 11 at nb=64); the four
NaN-producing families never disagreed. That is exactly the predicted split:
**NaN propagates, Inf gets absorbed.**

## How reachable is this on the real harness?

Probably not on the *public* inputs. `reference.generate_input`'s `rowscale`
family scales **down** (`logspace(0, -0.5*cond, n)`), which drives pivots toward
zero, and a zero pivot yields `Inf`/`NaN` in the column that then *does*
propagate to the last diagonal. Exp 030 also measured 604 hits / 0 fallbacks
across all six families at n=64/128, so the check never trips there today.

That is an argument that the bug would likely go unnoticed — not an argument
that it is safe. The secret leaderboard inputs are not visible, the substitution
is provably weaker, and the prize is ~1-3% on a handful of shapes. Bad trade.

## Salvage path (untried, strictly stronger)

The version the note *should* have described: write the finiteness flag from
**inside** the kernels, at the moment each pivot is computed, before the division
that absorbs it. Each micro/diag kernel already holds the diagonal value in a
register; an atomic OR into a one-word device flag costs no extra kernel and no
extra pass, and the host then reads a single scalar.

This is both cheaper than the shipped chain (removes the isfinite + multi-stage
all-reduce entirely, leaving only the irreducible DtoH read) **and strictly more
robust than it** — it catches an Inf pivot at the instant it is produced, which
even the shipped full-diagonal check would miss if the Inf were later absorbed
and overwritten. Cost: the flag must be zeroed per call inside the CUDA-graph
capture, which is the only fiddly part.

## Disposition

Root `submission.py` is **unchanged**; `#883174` remains the ranked winner. The
false premise is corrected in `journal.md` so it does not get retried.
