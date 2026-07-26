# The optimization loop — outer and inner together

Canonical rules live in `program.md`. This is the executable picture of them,
plus the two artifacts that keep the inner loop from free-associating:
`docs/lever-ladder.md` (what to try next) and `docs/experiment-matrix.md`
(what has already been tried, and what it moved).

```
                        THE 15-SHAPE GRID  (scripts/_gpu_runner.py BENCH_SPECS)
      small / high-batch            mid shapes                large-n / batch-1
      4096x32    21.9us         640x512  1355.5us          1x4096   1536.0us
      1024x64    35.3us           4x1024  713.8us          2x4096   3210.3us
       256x128   73.8us          60x1024 1282.0us          1x8192   5804.5us
        64x256  115.1us           2x2048 1358.3us          1x16384 15058.8us
        16x512  405.1us           8x2048 1598.4us          1x32768 42331.5us
      score = GEOMEAN over all 15 -> an additive constant costs more than a
      multiplicative win on the biggest shape

╔═══════════════════════════════════════ OUTER LOOP ═══════════════════════════════════════╗
║                                                                                          ║
║  [O1] SYNC          git fetch origin/main; read ranked incumbent id, commit,             ║
║   ^                 sha256, public/secret score. Lower is better.                        ║
║   |                                  |                                                   ║
║   |                                  v                                                   ║
║   |  [O2] PICK TARGETS  modal_verify.py benchmark -> per-shape us                        ║
║   |                     latency_budget.py -> measured = max(dram,math) + residual        ║
║   |                     docs/experiment-matrix.md -> what is already spent               ║
║   |                     rank by (residual x geomean weight), not by raw us               ║
║   |                                  |                                                   ║
║   |          +-----------------------+-----------------------+                           ║
║   |          v                       v                       v                           ║
║   |   ┌────────────┐          ┌────────────┐          ┌────────────┐                     ║
║   |   │ INNER LOOP │          │ INNER LOOP │          │ INNER LOOP │   parallel:         ║
║   |   │  shape A   │          │  shape B   │          │  shape C   │   one lease each,   ║
║   |   │ lease+wtree│          │ lease+wtree│          │ lease+wtree│   non-overlapping   ║
║   |   └─────┬──────┘          └─────┬──────┘          └─────┬──────┘   dispatch regions  ║
║   |         |  WINNER / PROMOTABLE FRONTIER only            |                            ║
║   |         +-----------------------+-----------------------+                           ║
║   |                                 v                                                    ║
║   |  [O3] INTEGRATE     new experiments/NNN-* from the LATEST ranked winner.              ║
║   |                     Merge only non-overlapping dispatch regions.                      ║
║   |                     Combined source = a NEW compile risk. Re-gate from scratch.       ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [O4] FULL GRID     15 shapes paired vs the exact ranked source.                      ║
║   |                     all pass? geomean improved? no off-target regression?             ║
║   |                     (1x32768 alone is 57% of grid wall time -- gate it last)          ║
║   |                          fail --> back to inner loops                                 ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [O5] COLD BUILD    clean sandbox, no warm extension cache,                           ║
║   |                     must finish in <80% of the Popcorn service timeout.               ║
║   |                     an exact-boundary timeout is COMPILE_BUDGET_FAILURE,              ║
║   |                     NOT evidence of bad arithmetic.                                   ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [O6] POPCORN TEST  require 17/17 on the exact source.                                ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [O7] POPCORN RANK  EXACTLY ONE in flight, globally. Monitor to terminal               ║
║   |                     public AND secret.                                                ║
║   |                                 |                                                    ║
║   |                    both splits improved?                                              ║
║   |                 no /                    \ yes                                        ║
║   |                  v                       v                                           ║
║   |          keep old winner,        [O8] ADOPT: copy to root submission.py,              ║
║   |          record rejection,             journal.md entry, matrix row,                  ║
║   |          no blind retry                commit, push main, verify remote               ║
║   |                  |                       |                                           ║
║   +------------------+-----------------------+                                           ║
║                          new incumbent becomes the baseline                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════ INNER LOOP  (one shape worker, one lease) ════════════════════════╗
║                                                                                          ║
║  lease: experiments/.leases/NNN-<shape>.json  (owner, baseline sha, heartbeat)           ║
║  worktree: isolated, branched from the exact verified incumbent                          ║
║                                    |                                                     ║
║                                    v                                                     ║
║  [I1] PICK A LEVER   from docs/lever-ladder.md, cross-checked against                    ║
║   ^                  docs/experiment-matrix.md so a closed path is not re-run.           ║
║   |                  Pe is 5-for-5 negative. CU/LP are saturated at 20/21.               ║
║   |                  Ov has 4 entries and zero clean measurements.                       ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I2] WRITE CODE                                                                     ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I3] FREE GATES    ~seconds, $0                                                     ║
║   |                     py_compile, syntax, git diff --check, source-policy scan,        ║
║   |                     artifact parse, snapshot compare                                  ║
║   |                     fail --> fix. NO GPU SPENT. (exp 031 died here, correctly)       ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I4] CORRECTNESS   ~1-2 min   modal_verify.py verify --filter <shape>               ║
║   |                     dense / spectrum / lowrank / rowscale / diagonal / tridiagonal   ║
║   |                     + official reconstruction checker + every fallback path           ║
║   |                     tolerance is NEVER weakened; closeness is enough                  ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I5] PAIRED LATENCY ~2-5 min   _gpu_runner.py pairedgrid                            ║
║   |                     candidate vs exact ranked source, ONE process, rotated inputs    ║
║   |                     MUST prove the intended backend ran: counters, load/compile      ║
║   |                     status, zero fallbacks.                                           ║
║   |                     >>> fallback timing is INVALID, not a slow data point <<<        ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I6] PROFILE       ~3-8 min    ncu_profile.py  -> dram% sm% tensor% occ%            ║
║   |                                                    stall_long_scoreboard / wait /    ║
║   |                                                    barrier / imc_miss                ║
║   |                                  latency_budget.py -> dram vs math vs residual       ║
║   |                        dram-bound -> fewer passes, smaller dtype                     ║
║   |                        math-bound -> faster pipe (tf32 -> fp8)                       ║
║   |                        residual   -> more parallelism / shorter dependent chain      ║
║   |                                      (completely DEAF to precision work)             ║
║   |                     skippable when I5 already shows a decisive win                    ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I7] CLASSIFY                                                                       ║
║   |         WINNER               >= 2.00x paired          --+                            ║
║   |         PROMOTABLE FRONTIER  < 2x, aggregate positive  --+--> hand to OUTER [O3]     ║
║   |         FRONTIER             faster, banked for later --+                            ║
║   |         REJECTED             slower / wrong / fallback-only                          ║
║   |         EXHAUSTED            6 distinct variants, no winner                          ║
║   |                                 |                                                    ║
║   |                                 v                                                    ║
║   |  [I8] RECORD        manifest + state.json (atomic) + journal.md                      ║
║   |                     + one row in docs/experiment-matrix.md                           ║
║   |                                 |                                                    ║
║   +---------------------------------+   next lever, until WINNER or EXHAUSTED            ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

## Why the gates are ordered this way

Cost per gate rises by roughly an order of magnitude at each step: free local
checks, then a single-shape Modal run, then paired + ncu, then the full grid,
then Popcorn. **A candidate never reaches a paid gate it could have failed on a
free one.** Exp 031 is the model case — its premise was refuted on a CPU gate
for zero GPU spend.

## Invariants

- **One ranked submission in flight, globally.** Exp 063 broke this (two
  concurrent ranked submissions of identical source) and it must not recur.
- **Stale-baseline check before every expensive gate.** Fetch `origin/main` and
  compare commit *and* ranked-source sha256. If the incumbent moved, stop
  spending, preserve artifacts, rebase, rerun affected gates.
- **`state.json` is the resume point**, not the chat log. On resume or
  compaction, read it and continue from `next_action`; never re-pay a completed
  paid gate.
- **Correctness gates are never weakened to promote a candidate.** Cheaper is
  allowed; weaker is not.
