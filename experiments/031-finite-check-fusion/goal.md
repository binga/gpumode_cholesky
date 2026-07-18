# Goal — Experiment 031: cheapen the per-call finite-check chain

Baseline: exact current ranked winner `#883174`, public
`1084.4572420163716us`, secret `1083.720390333199us`, commit `60b62fb`,
source SHA-256 `e072778cef0aec070e13e2093c7be7a98f2de74211fe6d2704cce5370fcd02e5`
(`baseline-883174.py`).

## Hypothesis

Per-call fixed overhead is now a top-3 cost on every sub-400us shape. Exp 030's
per-kernel profile put the finite-check chain at ~12-15us (4 kernels + a DtoH
sync) and copy-in/clone-out at ~9us, against a ~90us kernel chain at 256x128.

The finite check shipped at the split32 and 8x2048 dispatch sites is

    torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all().item()

which reduces over `batch*n` strided elements. The recorded idea was that NaN
"provably propagates to the last diagonal entry", so the same safety net could
be had from `l[..., -1, -1]` — a `batch`-element reduction — for ~1-3% on 4-5
shapes at zero kernel-code risk.

## Variant

`candidate-finite-check.py`: replace the full-diagonal predicate with
`torch.isfinite(l[..., -1, -1]).all().item()` at the two small/mid-shape sites
(split32 dispatch, 8x2048 triton path). The ms-scale `batch==1` large-n checks
are left untouched to keep the changed region tight. No kernel code changes.

## Gates

- Free gate first: the equivalence claim must hold, since the substitution is
  only admissible if it is *exactly* equivalent, not merely cheaper.
- `WINNER`: paired >= 1.02x on the small shapes, six families pass, zero
  unexpected fallbacks.
- `REJECTED`: any input on which the substituted predicate disagrees with the
  shipped predicate — this is a correctness gate, not a performance one.

## Cost guardrails

One paired probe (~$1) only after the free equivalence gate passes.

## Outcome

**REJECTED at the free gate — no GPU time spent.** The equivalence premise is
false for Inf. See `notes.md`.
