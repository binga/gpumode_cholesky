# Experiment 030 — route 1024x64 / 256x128 onto the split32 chain

Baseline: exact ranked winner `#882958`
(`3466e8f7a9ebb240924e642ef81acad054c31f5e17ceefa492c9d26f0ffe195d`,
public 1096.084us). No new kernel code: two `_SPLIT32_SHAPES` entries plus
subtile membership, with the exact ranked graph-replayed vendor paths retained
as the next dispatch layer (fallback preserved).

## Results (paired, B200, 2026-07-18, `probe-route-initial.json`)

| shape | baseline | candidate | paired speedup | families | verdict |
|---|---:|---:|---:|---|---|
| 256x128 | 157.4us | 142.8us | **1.1025x** | 6/6, zero fallbacks | **ADOPT** |
| 1024x64 | 119.3us | 119.5us | 0.9983x | 6/6 | REJECT (keep vendor route) |

Notable: tf32x3 split32 handled **all six families without a single
fallback** at both shapes (604 hits / 0 fallbacks) — at n=64/128 the
tolerance margin holds even for spectrum/lowrank.

## Why 1024x64 is a wash and 256x128 only 1.10x

Per-kernel profile at 1024x64: the one-warp micro stops being latency-hidden
at batch 1024 (2 launches x 20.5us — ~7 sequential CTAs per SM), and the
fixed per-call overhead (copy-in/clone-out ~9us + finite-check chain ~12us)
is a large fraction at this scale. At 256x128 the chain itself is ~90us
(4 micro x 14.3 + 3 inner + 3 apply) with ~25us of per-call fixed overhead —
the finite-check + copy overhead is now the biggest single lever for any
sub-200us shape.

## Disposition

`candidate-final-combined.py` = exp-029 v4 (`tl.rsqrt` micro) + the
`(256, 128)` routing only. Full-grid probe: `fullgrid-final.json`; then
popcorn test and a single ranked submission for the combination.
