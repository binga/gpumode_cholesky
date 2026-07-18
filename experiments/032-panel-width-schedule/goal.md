# Experiment 032 — per-shape non-uniform panel-width schedules (lever L2)

Baseline: ranked `#883174`, 1084.4572420163716µs public / 1083.720390333199µs
secret. Source: `docs/qr-transfer-proposal.md` lever L2.

## Hypothesis

Every split32 shape currently factors with one uniform panel width
(`_SPLIT32_NB = 128`), applied identically from the first panel to the last.
The trailing block shrinks monotonically as the panel walks the diagonal, so a
single width is necessarily mistuned at one end. Specifically, late panels pay
a full 128-wide panel factorization whose rank-128 trailing update no longer
has enough trailing rows to amortize it.

Three independent qr_v2 finishers rejected uniform widths and converged on
staircases: gau.nernst (2nd) hand-tuned `(96,96,64,32,32,192)` for n=512,
Michael Lutz (5th) shipped an `8+8+16` staircase and measured 3.3x on isolated
panel emission, zhongmingee (4th) used recursive split trees (`96 -> 48+48`).

## Implementation status: LANDED, INERT

`submission.py` now takes its panel widths from `_nb_schedule(batch, n)`, which
reads `_SPLIT32_NB_SCHEDULE`. **That table is empty**, so every shape falls
back to the uniform `_SPLIT32_NB` schedule.

Free gate passed: for all seven split32 shapes, the fallback schedule produces
panel boundaries identical to `range(0, n, 128)` — the emitted launch sequence
is unchanged from `#883174`. Enrolling a shape is therefore a one-line,
individually revertable change.

## Hard constraint discovered during implementation

`_trailing_nb` does `d = tl.arange(0, NB)`. Triton requires a power-of-two
arange bound, so **every panel width must be a power of two >= 32**.
gau.nernst's `(96,96,64,32,32,192)` is not directly expressible. Expressing
non-power-of-two widths needs a padded + masked load in `_trailing_nb`, which
wastes MMA lanes — that is a separate follow-up experiment, not this one.

`_validate_nb_schedules()` enforces sum-to-n and power-of-two at import, so a
malformed schedule fails on a free gate before any B200 time is spent.

## Candidate schedules to sweep

All validated (sum to n, power-of-two widths). `U` = current uniform baseline.

| shape | U (baseline) | A — tail taper | B — wide head + taper |
|---|---|---|---|
| 256x128 | `(128,)` | `(64,32,32)` | `(64,64)` |
| 64x256 | `(128,128)` | `(128,64,32,32)` | `(64,64,64,64)` |
| 16x512 | `(128,)*4` | `(128,128,128,64,32,32)` | `(256,128,64,32,32)` |
| 640x512 | `(128,)*4` | `(128,128,128,64,32,32)` | `(256,128,64,32,32)` |
| 4x1024 | `(128,)*8` | `(128,)*7+(64,32,32)` | `(256,256,128,128,128,64,32,32)` |
| 60x1024 | `(128,)*8` | `(128,)*7+(64,32,32)` | `(256,256,256,128,64,32,32)` |
| 8x2048 | `(128,)*16` | `(128,)*15+(64,32,32)` | `(256,)*7+(128,64,32,32)` |

Sweep A first: it is the minimal edit (only the last panel changes) and
isolates the "late panels are over-wide" hypothesis on its own. B additionally
tests whether early panels want to be *wider* than 128, which is a different
claim and confounds A if run together.

`256x128` is a special case worth running early despite its size: at n=128 the
uniform schedule is a *single* panel with no trailing update at all, so
schedule A there is a pure test of whether sub-panel staircasing helps when the
rank-NB trailing update is absent.

## Measurement protocol

Per `program.md` steps 6-8. Paired same-process B200 against the exact
`#883174` source, one shape enrolled at a time — the geometric mean hides
per-shape regressions (Mike's lesson 4: *"improve the score" is a bad
objective*). Record isolated panel-factor time, not only end-to-end, since the
hypothesis is specifically about panel emission cost.

## Expected magnitude and gate

A speedup `s` on one shape moves the geomean by `s^(1/15)`. This experiment
touches 7 of 15 shapes. At 1.05x each: `1.05^(7/15)` = **2.3%** score
reduction. At 1.02x each: 0.93%.

**Note the gate conflict.** `program.md` step 3 defaults to requiring 2.00x per
shape. No panel-width schedule will ever return 2.00x — the realistic band is
1.02-1.10x. Judge this experiment on aggregate geomean across the enrolled
shapes with no per-shape regression, or it will be rejected for clearing the
wrong bar. See `docs/qr-transfer-proposal.md` §1.

## Kill criteria

- Schedule A loses on `640x512` and `8x2048` (the two largest split32 shapes,
  most panels, most room for a taper to pay): the taper hypothesis is refuted
  and A closes for all shapes.
- Both A and B are within +/-0.5% on every shape: panel width is not a live
  axis at these sizes; record in the tracker and close the lever.
- Do not close on a single shape's negative. Mike's rule: no experiment closes
  without a hardware-level explanation.
