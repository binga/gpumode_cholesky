# Inner-loop lever ladder (QR transfer)

A standing backlog of levers for step 5 of `program.md` ("use a bounded
architecture ladder"). A shape worker picks its next variant from here rather
than free-associating, and records the outcome in the `journal.md` Optimization
Tracker.

Source: the structural evolution of the `qrproblem` kernel on the same B200
target and the same geomean-of-shapes scoring rule (see `lessons_qrproblem.md`),
plus <https://sankalp.bearblog.dev/autoresearch/>.

## Why this transfers

QR and Cholesky on this hardware share the load-bearing property: **the serial
panel/diagonal factorization is the bottleneck, not the trailing GEMM.** That is
recorded independently on both sides — `lessons_qrproblem.md` states it as the
universal finding across every size, and the memory note
`diagonal-potrf-is-the-large-n-bottleneck` measures `getrf_wo_pivot` at 52%/36%
of the two large Cholesky shapes against 7% for MXFP8 trailing work.

So the QR ladder is not an analogy. It is a list of levers already proven
against this bottleneck on this GPU under this scoring rule.

## Ladder status

| # | QR structural change | QR geomean | Cholesky equivalent | Status |
|---|---|---|---|---|
| 1 | `torch.geqrf` everywhere | >108.8k us | cuSOLVER `potrf` everywhere | shipped (baseline) |
| 2 | Blocked WY on one shape | 108.8k us | Left-looking blocked on 1x16384 | shipped (S10) |
| 3 | Blocked route on all shapes | 10.2k us | Blocked route + per-shape dispatch | shipped (S15/S20/S21) |
| 4 | Triton panels + grouped updates | 4.3k us | Triton panel-inner 64x64 | shipped (S20/S21) |
| 5 | Cholesky-ORHR for n4096 | 4.0k us | Recursive / blocked inverse | partial — `rec_inv` is n>=16384-only (exp 065) |
| 6 | CUDA graph replay | 3.4k us | CUDA Graphs | **rejected here** — graph copy cost (S16b), not capturable (S40b) |
| 7 | Fused V/T assembly: no slice copies, cats, temporaries | 2.75k us | Per-call copy-in/clone-out + finite-check chain | **UNMEASURED** — see below |
| 8 | split16 panels + tail-Gram | 2.5k us | Variable NB near the trailing edge | UNTRIED |
| 9 | Fixed-shape kernel specialization | 2.0k us | Per-shape custom CUDA | shipped for n<=256 only; mid/large shapes still run generic runtime-`n` code |
| 10 | Composed superpanels, direct-H returns | 1.80k us | Superpanel composition, direct-L return | rejected — rank-128 superpanels 0.697x (S45) |

Steps 1-5 are fully harvested here. Steps 6, 7, 9 are not, and in QR those three
carried 4.3k -> 2.0k us: **more than half the total gain**.

## Lever 7 is the largest unmeasured item on the board

`journal.md` records the cost and then stops:

> Per-call fixed overhead (copy-in/clone-out ~9us + finite-check chain ~12-15us)
> is a top-3 cost on every sub-400us shape (S28) but is not yet a column,
> because no variant has been measured.

Sensitivity against the exp059 full grid (`experiments/059-two-large-incremental/combined-v3-fullgrid.json`,
`baseline_us`), removing a flat per-call constant from every shape:

```
 4096x32      21.9us   <-- total shape cost is BELOW the recorded overhead
 1024x64      35.3us
  256x128     73.8us
   64x256    115.1us
   16x512    405.1us
  640x512   1355.5us
    4x1024   713.8us
   60x1024  1282.0us
    2x2048  1358.3us
    8x2048  1598.4us
    1x4096  1536.0us
    2x4096  3210.3us
    1x8192  5804.5us
    1x16384 15058.8us
    1x32768 42331.5us
 geomean     873.2us

 remove  9us/call ->  811.2us   1.077x   -7.1%
 remove 15us/call ->  754.5us   1.157x  -13.6%
 remove 21us/call ->  668.7us   1.306x  -23.4%
```

Because the score is a geomean, an additive constant is worth more than any
multiplicative win on 1x32768 — the shape that has absorbed most recent effort.

Three caveats, so the band is read as a band:

- The ~21us figure is from S28. Exp 061/062 have since attacked overhead on the
  large shapes, so the surviving constant today is likely nearer the 9us end.
- Full removal is not reachable. The finite check is a correctness gate. S29's
  standing constraint holds: **cheaper is allowed, weaker is not** — shrinking it
  to the last diagonal entry is invalid because `finite/Inf == 0`, so an
  overflowed pivot is absorbed into a zero column and never reaches `L[n-1][n-1]`.
- The local grid geomean (873.2us) is not the leaderboard geomean (733.5us
  public, ranked #909269). Use it for lever ranking, not for score prediction.

Even the floor of the band is a 7% geomean win on a lever with zero measured
variants against it.

### First probe

Cheap enough that it is not a research program:

1. Free gates only: fuse the finite check into the tail of the final
   factorization kernel instead of running it as a separate launch chain, and
   remove the clone-out wherever the output buffer can be written in place.
2. One `pairedgrid` run restricted to the four smallest shapes
   (4096x32, 1024x64, 256x128, 64x256), where the constant is the majority of
   the cost and the signal is largest.
3. Only if that clears, expand to the full grid.

## Lever 9 for mid shapes

Fixed-shape specialization shipped for `n<=256` (S36/S38/S39/S41, all custom
CUDA) but the mid and large shapes still run generic code with runtime `n`. QR
took 2.5k -> 2.0k us from hardcoding dimensions and fusing reductions once the
shape was known at compile time. The mid shapes are exactly where
`mid-shape-cusolver-headroom` says the geomean is still 19-260x above hardware
floors.

## Lever 6 deserves one re-test, not a reopen

CUDA Graphs were rejected here on graph copy cost (S16b) and on the CUDA micro
not being capturable (S40b). Both rejections predate the current overhead
structure. If lever 7 removes the copy-in/clone-out, the graph copy cost that
sank S16b is a different number. Re-test only after lever 7 lands, not before.

## Loop hygiene (from the autoresearch post)

Recorded here as process backlog, not adopted into `program.md`:

- **Beam search.** The post's central anti-stall device is keeping 3-5 candidate
  idea families alive rather than hill-climbing one incumbent, so structural
  changes that start slower get time to mature. `program.md` is single-incumbent
  by construction (step 11 rebases onto the latest winner; step 9 classifies
  anything slower as `REJECTED`). Exp 048's V2 won 1.167x and was discarded —
  the exact failure mode beam search exists to prevent. Changing this touches the
  promotion rules, so it needs an explicit decision.
- **Cleanup cycles.** `submission.py` is ~141KB and `journal.md` ~143KB. The post
  calls out periodic archiving/refactoring; the submission size also feeds
  directly into the cold-build budget (`program.md` step 13).
- **Cycle time.** 1x32768 is 42.3ms of the 74.9ms full-grid run (~57%); the two
  largest shapes together are ~77%. Gating them behind a 13-shape pass would cut
  integration latency substantially without weakening the final gate, since a
  candidate that regresses a small shape never needs the expensive shapes run.
