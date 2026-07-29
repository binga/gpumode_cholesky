# Experiment 072 — 640x512 launch fusion

## Frozen control

Ranked `#926462`, commit `1ec14b72694dceed3d91cbf5ecff303c4ca5be80`,
`submission.py` SHA-256 `582cde1648b8b3e9d77a36173dd59cd36588123ae28800ca00e5342b869ff723`.

The dispatch boundary is exclusively `(batch, n) == (640, 512)` on the eager
split32/fused-panel path. The e62 source and its shape table are out of scope.

## Amdahl ceiling

The incumbent shape diagnosis (`results/069-inc-shapediag-mid.json`) measures
1224.3 us wall, 1080.2 us device, and 144.1 us inter-launch idle across 53
launches. Deleting every idle microsecond could therefore reach only
`1224.3 / 1080.2 = 1.1334x`, so the 2.00x research target is impossible for a
pure launch-overhead lever. The chosen twelve apply+inner pairs account for an
equal-share idle ceiling of about 32.6 us, or `1.0274x`; fusion can exceed that
only by reusing panel data and reducing device work.

## N=3 variants

- V1: one CTA per matrix, 32-column update segments, four warps.
- V2: one CTA per matrix, 64-column update segments, eight warps.
- V3: one CTA per matrix, 32-column update segments, eight warps.

Every variant loads the full at-most-96x32 pre-apply panel inside one CTA,
computes its factors, consumes 32- or 64-column update segments sequentially,
and publishes factors only after every source segment has been read. This is
strictly opt-in for 640x512 and introduces no non-default stream primitive.

An initial multi-CTA draft was discarded after code audit: a column-owner CTA
could publish transformed factors while another CTA still read the pre-apply
values. Its bitwise-determinism pass is deliberately invalidated and not used.

## Mandatory strengthened gates

An initial worktree-local stress prototype exposed that `lowrank(cond=6)` was
not reliably SPD in float32: both the candidate and exact baseline returned
non-finite output. That row is invalidated. The prototype was removed from the
final diff. Authoritative evidence uses the reusable `stressgrid` from commit
`d960e08`: three same-input repeats plus guaranteed-SPD tiny-diagonal,
near-singular banded, and mixed-dynamic generators at exponents 6, 8, and 10,
all through the unchanged official checker with backend/fallback counters.

## Results

All three race-free candidates passed three-repeat same-input bitwise equality,
the unchanged official checker, active-backend counters, and all nine
guaranteed-SPD condition cases at exact `640x512`, with zero fallbacks. The
unchanged `16x512` sibling also passed as an off-target control.

| Variant | Control (us) | Candidate (us) | Speedup | CI95 | Verdict |
|---|---:|---:|---:|---:|---|
| V1, C32/W4 | 1229.776 | 1447.728 | 0.8495x | [0.8492, 0.8498] | rejected slow |
| V2, C64/W8 | 1302.080 | 1530.480 | 0.8509x | [0.8494, 0.8587] | rejected slow |
| V3, C32/W8 | 1308.256 | 1749.600 | 0.7478x | [0.7467, 0.7483] | rejected slow |

Each paired run used the exact ranked source in the same process, nine repeats,
two rotating inputs, active `_EXP072_FUSED_HITS=12`, and zero new fallbacks.
The unchanged `16x512` row stayed at parity (0.9991-1.0026x), so the regression
is isolated to the intended 640x512 dispatch.

The least-bad variant V2 was submitted to the repository ncu profiler for
register/occupancy/stall evidence. Nsight Compute failed before target execution
with `Failed to initialize the profiler: LibraryNotLoaded`; no CSV or resource
counters were available. This is recorded as a platform profiler omission, not
silently inferred. The paired loss is decisive without it.

## Classification and fast_p

`SHAPE_EXHAUSTED`: every serious variant is 15-25% slower. No candidate was
eligible for the parent-directed six-family promotion gate, integration, or
Popcorn. Stress correctness rate is `3/3 = 1.0`; `fast_1 = 0/3 = 0.0` and
`fast_2.0 = 0/3 = 0.0`. Strict all-gate `fast_0` is reported as null because
six-family was intentionally not spent after all three failed paired timing.

The lesson is structural: eliminating twelve launches does not compensate for
collapsing thousands of row/column tiles into one CTA per matrix. Four 640-CTA
waves serialize the segmented dot work, reproducing the occupancy failure class
already seen when enrolling 640x512 onto the resident e62 path.
