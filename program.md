# Cholesky leaderboard optimization program

This is the repository's repeatable operating program for GPU MODE Cholesky
optimization on B200. It is intentionally independent of evo workflows.

## Invocation and goal parameters

Invoke this program by naming `program.md` in a goal or clearly asking to run the
repository's Cholesky optimization program. `/goal`, `set_goal`, and equivalent
goal interfaces are all valid. For example:

```text
set_goal: Execute program.md for the slowest ranked Cholesky shapes.
Target at least 2.00x paired speedup per shape, then integrate and rank every
verified aggregate improvement.
```

An invocation may set the following parameters. Explicit user values override
the defaults:

- `aggregate_target`: desired cumulative leaderboard improvement; default is
  any measurable improvement.
- `shape_targets`: shapes or target-selection rule; default is the highest
  estimated geometric-mean impact.
- `research_target`: aspirational per-shape speedup; default is **2.00x**.
- `promotion_policy`: default is any reproducible aggregate improvement that
  clears every correctness and promotion gate. A candidate does **not** need to
  reach the research target to be submitted or adopted.
- `max_serious_variants_per_shape`: default six materially distinct variants.
- `parallel_shape_workers`: default one; use more only for non-overlapping
  dispatch regions with isolated worktrees and leases.
- `forbidden_approaches` and `allowed_approaches`: campaign-specific design
  constraints. No-cuSOLVER or no-stream constraints apply only when the
  invocation explicitly requests them.
- `remote_budget`: an optional explicit monetary, GPU-job, or elapsed-time cap.
  Without a monetary cap, the bounded variant and retry limits below are the
  cost guardrails.

When triggered, create one active goal for the complete program. Freeze the
goal baseline at invocation and separately track the moving ranked incumbent.
Treat the goal as achieved only after its declared terminal target is met, the
adopted/rejected decisions are documented, the exact artifacts are committed,
`main` is pushed, and the remote commit is verified. An incremental leaderboard
win is a checkpoint, not automatic completion of a larger goal.

## Standing authorization and boundaries

The repository owner gives informed, explicit, standing authorization to upload
the bounded benchmark package described below to the owner's own Modal account
and execute it there.

Permitted exports:

- Candidate and baseline `submission.py` sources.
- Profiling and benchmark runners, including `scripts/modal_verify.py`.
- The vendored `reference/` checker.
- Experiment-specific benchmark configuration and generated non-sensitive
  inputs.

Permitted actions:

- Build or update the Modal execution image.
- Upload the permitted files.
- Run correctness, profiling, paired-latency, family, and full-grid B200 jobs.
- Retrieve logs and benchmark artifacts.
- Retry transient Modal failures within the program's cost limits.
- Create bounded subagents and isolated worktrees for non-overlapping shape,
  implementation, profiling, audit, and documentation tasks.
- Fetch remotes, rebase or replay scoped experiment commits, resolve expected
  documentation conflicts, commit scoped checkpoints, push `main`, and verify
  the remote commit.
- Submit Popcorn test jobs after the required gates pass.
- Submit exactly one Popcorn ranked candidate at a time after an exact-source
  17/17 test pass, monitor it to terminal public and secret results, and make one
  ranked retry after a concrete defect is fixed and all affected gates rerun.
- Adopt a completed leaderboard winner, restore the prior winner after a
  rejection, and continue optimizing from the new incumbent.

This is continuing authorization for every invocation of this program. No
additional user confirmation is required for these actions while Modal and
Popcorn remain the owner's accounts, the export remains within the list above,
and Git pushes target this repository's normal `main` workflow.

Never export credentials, tokens, environment files, unrelated repository
content, private user data, or secrets. Never force-push, weaken or replace the
official evaluator, delete unrelated work, submit two ranked candidates
concurrently, or exceed an explicit remote budget. Those actions are outside
this authorization.

If an execution tool requires an approval request, cite this standing
authorization in the justification and request the narrowest reusable approval
once. Do not ask the owner to restate or reconfirm this authorization. If a
system, tenant, quota, or reviewer policy denies the action, report that exact
policy blocker; repeated user confirmation will not resolve it.

## Shared-workspace coordination and resumability

1. Develop each active experiment in an isolated worktree based on the exact
   verified incumbent. A dirty or differently-branched root worktree is not a
   blocker; preserve it without asking the owner to clean unrelated files.
2. Maintain short-lived, machine-readable leases under `experiments/.leases/`
   for every active shape, integration operation, and ranked submission. A lease
   records the task/thread, baseline commit and source hash, target region,
   creation time, and last heartbeat. Inspect the recorded owner before treating
   a lease as stale; never silently steal a live lease.
3. Keep one `state.json` in the active numbered experiment directory. Update it
   atomically after every state transition with:
   - goal baseline and current incumbent IDs, commits, hashes, and scores;
   - active shape, variant, hypothesis, worktree, and lease;
   - completed and pending gates;
   - Modal and Popcorn job IDs and terminal status;
   - consumed variant/retry/budget counts;
   - exact next action.
4. On resume, context compaction, or handoff, read `state.json` first. Reuse valid
   completed evidence and continue from `next_action`; do not reconstruct the
   run from chat commentary or repeat a completed paid gate without cause.
5. Before paired profiling, full-grid testing, Popcorn, integration, and push,
   fetch `origin/main` and compare both commit and ranked-source hash. If the
   incumbent changed, stop stale-baseline spending, preserve current artifacts,
   determine dispatch overlap, rebase or rebuild from the new winner, and rerun
   every affected gate.

At each meaningful checkpoint and in the final report, show this compact user
view for all targeted or changed shapes:

| Shape | Control latency (us) | Current latency (us) | Speedup |
|---|---:|---:|---:|
| `batch x n` | exact paired control | exact paired candidate | `control/current` |

Use same-process paired evidence where available. Label missing or cross-context
numbers rather than silently mixing leaderboard and local/Modal measurements.

## The loop

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


## Workflow

1. **Synchronize the current winner**
   - Fetch `origin/main`. If the shared root is not clean, leave it untouched and
     start in an isolated worktree from the verified incumbent.
   - Record the current ranked submission ID, exact commit, public/secret score,
     full-grid evidence, and source snapshot.
   - Cholesky's leaderboard score is geometric-mean latency in seconds: **lower
     is better**. Compute improvement as
     `(incumbent - candidate) / incumbent * 100`.
   - **Verify the incumbent against `popcorn submissions list`, not against
     `main`.** The true ranked winner is whatever the leaderboard says it is,
     and it has repeatedly lived on an unmerged branch or in another worktree
     while `main` and `README.md` lagged several winners behind. Confirm by
     `shasum -a 256` of the candidate baseline against the ranked source, and
     re-check immediately before any paid gate — `origin/main` moved twice
     mid-session during experiment 064.

2. **Find the highest-impact shapes**
   - Profile or inspect the current 15-shape leaderboard grid.
   - Rank shapes by latency and estimated geometric-mean impact.
   - Prefer the slowest shapes unless another shape has a clearer higher-ROI path.

3. **Set exact shape goals**
   - Give every target shape its own bounded task and artifact directory.
   - Default research target: candidate paired mean latency at most 50% of the
     exact current ranked path, i.e. at least **2.00x speedup**.
   - Treat this as an aspirational research target, not a universal promotion
     threshold. A correct sub-2.00x frontier may be integrated and ranked when
     it yields a reproducible aggregate improvement and passes all gates.
   - State correctness, no-regression, cost, and ranked-submission guardrails.

4. **Delegate and babysit actively**
   - Inspect task checkpoints and raw evidence rather than accepting summaries.
   - Redirect stalls toward genuinely untried shape-specific Blackwell levers.
   - Reject fallback timings, missing backend evidence, cosmetic parameter
     sweeps, weakened gates, and results measured against stale baselines.
   - Use isolated worktrees and leases; never allow workers to overwrite the
     same source or benchmark the same shape against different unstated controls.

5. **Use a bounded architecture ladder**
   - Prefer materially different axes: vendor/expert APIs, per-matrix dispatch,
     blocked or left-looking factorizations, Triton, custom CUDA/tcgen05, TF32,
     BF16x9, FP8/MXFP8, CUDA Graphs, TMA, clusters/DSM, or refinement.
   - Apply no-cuSOLVER, no-stream, or other exclusions only when the invocation
     specifies them. Otherwise choose from the complete architecture ladder.
   - Measure up to `max_serious_variants_per_shape` genuinely distinct serious
     variants before declaring bounded exhaustion. Preserve every valid partial
     frontier.

6. **Run free gates first**
   - Run local property tests, compilation/syntax checks, artifact parsing,
     `git diff --check`, source-policy scans, and source/snapshot comparisons.
   - Do not spend remote GPU time on a candidate that fails a free gate.

7. **Run paired same-process Modal B200 profiling**
   - Use the owner's standing Modal authorization above.
   - Compare the candidate with the exact ranked source in one process.
   - Rotate representative inputs, retain outputs through validation, and record
     mean/best latency plus component profiles when they can identify bottlenecks.
   - Require positive proof that the intended backend executed: counters,
     readiness metadata, load/compile status, zero unexpected fallbacks, and no
     runtime error. Timing from a fallback implementation is invalid.
   - After every valid measurement, update `state.json` and the compact shape,
     control-latency, current-latency, speedup table.
   - **Compare like with like.** A probe row that calls a driver directly is not
     comparable to a control row that goes through `custom_kernel`: the wrapper
     carries dispatch plus the end-of-call `isfinite(...).all().item()` sync,
     measured at 845us on `1x32768` in experiment 064. Always include a
     reimplementation of the *shipped* logic as the probe's own control and
     quote speedups against that, or the wrapper overhead is silently booked as
     a win.
   - **Never edit a mounted file while a Modal job is building.** `submission.py`,
     the candidate source, `reference/`, and `scripts/_gpu_runner.py` are copied
     into the image; touching any of them mid-build fails the job with
     `ExecutionError: ... was modified during build process` and wastes the whole
     run. Experiment 064 lost an ~18-minute family grid this way. Edit notes and
     `state.json` while jobs are in flight; nothing else.

8. **Validate numerical closeness across the changed region**
   - Exact or bitwise equality is **not required**. Numerical closeness is enough
     when the official reconstruction checker passes; record the available
     tolerance margin as evidence.
   - Preserve the official tolerance; do not weaken the checker to promote a
     candidate. Require finite results, a valid lower-triangular factor, positive
     diagonal, and acceptable scaled reconstruction residual.
   - Cover dense, spectrum, low-rank, row-scaled, diagonal, and tridiagonal
     inputs for every changed dispatch shape. Validate all safety fallbacks.
   - **`familygrid` reports `passed: false` whenever any fallback fires, so read
     `checker_ok` per row, not the top-level flag** — and never explain a
     fallback away from memory. Rerun the identical gate against the exact
     ranked baseline and diff the rows. A fallback that the incumbent also takes
     is pre-existing behaviour; one that only the candidate takes means the fast
     path went non-finite, which is a bug signal even when the checker still
     passes through the fallback. Experiment 064 used this to confirm that three
     fallback rows at `1x16384`/`1x32768` were identical to the baseline's.
   - **`spectrum` is not generable at `n >= 16384`**: its input needs a QR of an
     `n x n` matrix, costing far more than the factorization under test and
     exceeding the sandbox timeout. Use `--families` to gate the tractable five
     (`dense`, `diagonal`, `lowrank`, `rowscale`, `tridiagonal`) and record the
     omission. Experiments before 064 simply never gated the two largest shapes.

9. **Classify every measured variant**
   - `WINNER`: correct and at least `research_target` faster on the paired
     target.
   - `FRONTIER`: correct and faster, but below the research target.
   - `PROMOTABLE FRONTIER`: a `FRONTIER` that produces a reproducible aggregate
     improvement and clears the same promotion gates as a `WINNER`; it may be
     submitted and adopted for incremental progress.
   - `REJECTED`: slower, incorrect, invalid, or fallback-only evidence.
   - `EXHAUSTED`: `max_serious_variants_per_shape` distinct measured variants
     without a research-target winner.

10. **Narrow broad searches when evidence says to**
    - Stop or archive low-value tasks after bounded exhaustion.
    - Concentrate resources on the slowest remaining shapes and strongest
      partial frontiers.

11. **Integrate only verified, non-overlapping improvements**
    - Create a new numbered experiment from the latest ranked winner.
    - Rebase candidates after any intervening leaderboard win.
    - Combine positive frontiers only when their dispatch regions do not conflict.
      Leave unimproved shapes on their shipped implementation.
    - Treat combined-source compilation as a new integration risk. A combination
      of individually ranked paths must pass cold compilation and every promotion
      gate as an exact new source before it replaces either ranked snapshot.

12. **Run the full 15-shape Modal B200 benchmark**
    - Use the owner's standing Modal authorization above, including profiler
      scripts when needed to explain regressions or unexpected dispatch costs.
    - Retain outputs as the official harness does, require every shape to pass,
      compare per-shape latency with the exact ranked baseline, and reject
      material off-target regressions.
    - Promote only when the aggregate geometric mean improves.

13. **Prove the cold build fits the service budget**
    - Build the exact candidate in a clean sandbox without relying on a warm
      extension cache. Record image setup, import, extension compilation,
      validation, and benchmark durations separately.
    - By default, require the clean build and gate to finish within 80% of
      Popcorn's observed service timeout unless the invocation supplies another
      threshold. An exact service-boundary timeout without a numerical failure is
      `COMPILE_BUDGET_FAILURE`, not evidence of incorrect arithmetic.
    - Reduce generated code or compilation work and rerun affected gates before
      Popcorn. Do not repeatedly submit an unchanged compile-time failure.

14. **Run Popcorn gates in order**
    - First run test mode and require **17/17**.
    - Audit the exact source, raw Modal artifacts, changed-family results, full
      grid, and test submission ID.
    - Then permit exactly one ranked submission at a time and monitor it until
      both public and secret runs finish.

15. **Adopt using completed leaderboard evidence**
    - Compare terminal public and secret latency scores with the previous ranked
      winner using the lower-is-better rule.
    - If both completed splits satisfy the configured promotion policy, adopt
      the exact ranked source at repository root.
    - If not improved, keep the previous winner and record the rejection without
      launching duplicate ranked retries unless a concrete defect was found.

16. **Maintain structured evidence and the journal**
    - Write a structured manifest for every measured variant immediately. Raw
      evidence and `state.json` are the live source of truth during a search.
    - Add a dated `journal.md` entry for every experiment, adopted or rejected.
      Record hypotheses, variants, component profiles, paired means/bests,
      speedups, numerical margins, fallbacks, failures, full-grid changes,
      Popcorn IDs, public/secret scores, costs, insights, and next ideas.
    - Consolidate the living **Optimization Tracker** when a variant becomes a
      frontier, an architecture closes, a shape is exhausted, or a winner is
      adopted. Mark shipped paths `✓`, tried/rejected paths `✗`, and only
      genuinely untested paths `TBD`; include the experiment/session reference
      and useful measured speedup in the cell.
    - Add columns when a new optimization family is tested. Remove stale `TBD`
      entries as soon as evidence exists. Keep the current-best line synchronized
      with the latest successful leaderboard submission.

17. **Preserve a reproducible experiment package**
    - Save the goal, exact baseline, every serious candidate, exact ranked source,
      raw paired/family/full-grid/ranked JSON, notes, and relevant harness changes
      under `experiments/NNN-*`.
    - Update the root README and `experiments/README.md` with the verdict.
    - Run final local, syntax, JSON, source-policy, whitespace, and snapshot checks.

18. **Commit, land on `main`, push, and verify**
    - Commit the complete experiment with a descriptive message.
    - Integrate it onto the latest `main`, push `main` to GitHub, and verify the
      remote branch resolves to the intended commit.
    - Only then complete the active goal and begin the next optimization cycle
      from this new ranked baseline.

## Autonomous retry and cost policy

- Retry a transient Modal infrastructure failure at most twice.
- Repair and rerun invalid or fallback-only measurements; they do not count as
  serious measured variants.
- Do not retry an unchanged numerical or performance rejection.
- Retry a failed Popcorn test only after identifying and fixing a concrete
  correctness, packaging, runtime, or compile-budget defect.
- Permit at most one ranked retry after a concrete defect, a new exact-source
  17/17 test pass, and rerunning every affected promotion gate.
- If an optional local dependency is unavailable, record the omission and
  continue when stronger authorized B200 and official evidence covers the same
  property.
- Stop new remote launches when an explicit budget is exhausted. Preserve state
  and report the exact remaining decision instead of spending beyond it.

## Terminal states

- `ACHIEVED`: the declared terminal target is present in completed leaderboard
  evidence, adopted, documented, pushed, and remotely verified.
- `PARTIAL WIN`: a verified sub-target aggregate improvement was ranked and
  adopted; checkpoint it and continue toward a larger declared goal.
- `SHAPE EXHAUSTED`: the bounded architecture ladder for one shape ended without
  its research target; preserve any promotable frontier and continue elsewhere.
- `CAMPAIGN EXHAUSTED`: every declared target and permitted transfer path is
  boundedly exhausted within budget. Document the evidence; do not claim the
  numerical target was achieved.
- `BLOCKED`: an external policy, credential, quota, destructive ambiguity, or
  irreconcilable overlapping edit prevents further authorized progress.
## The secret split: an unresolved policy gap (exp 065)

**The repository has resolved a secret-split regression both ways on the same
evidence signature. Until the owner picks one, state which rule you are applying
before you spend a ranked slot.**

| experiment | paired grid | public | secret | decision |
|---|---|---|---|---|
| exp 022 (`#882969`) | 1.0052 | +1.51% (worse) | -1.44% (better) | rejected |
| exp 035 (`#888352`) | +0.61% | **-2.94% (better)** | **+5.26% (worse)** | **adopted** |
| exp 065 (`#914341`) | +1.22% | **-3.79% (better)** | **+5.71% (worse)** | **rejected** |

Exps 035 and 065 are the same case and got opposite verdicts. `promotion_policy`
defaults to "any reproducible aggregate improvement", which does not say whether
"aggregate" means the public split, the secret split, both, or their mean. Step
15 says "if both completed splits satisfy the configured promotion policy",
which reads as *both* — but exp 035 shipped anyway and became the incumbent.

Two things follow, and neither is optional:

1. **Name the rule in the goal.** An invocation that may produce a split verdict
   must set `promotion_policy` explicitly, e.g. "adopt only if both splits
   improve" or "adopt on public, record secret". Do not infer it from precedent,
   because precedent is contradictory.
2. **A paired-grid geomean below ~1.5% does not predict the sign of the secret
   split.** Three times now — exps 022, 035, 065 — and exp 065 had the strongest
   device-time evidence of the three: CI95 [1.0112, 1.0133] excluding 1.0, all
   fifteen shapes ok, identical counters, zero new fallbacks, correctness
   bit-identical to the control. It still landed +5.71% on secret. The secret
   split evaluates inputs this repository cannot see. Treat a sub-1.5% paired
   grid as insufficient grounds to spend a ranked slot unless the goal
   explicitly accepts a coin-flip on secret.

## Non-negotiable promotion rules

- Never treat fallback latency as candidate evidence.
- Never claim "promotion requires both splits" or "public is enough" as settled
  policy — it is not; see the secret-split gap above. Name the rule first.
- Never quote a probe speedup measured against a differently-invoked control.
- Never claim a fallback is pre-existing without a baseline run that shows it.
- Never weaken correctness thresholds; approximate arithmetic is acceptable
  only when it remains close enough for the official checker.
- Never rank two candidates concurrently.
- Never rank before paired profiling, changed-family checks, the full grid, and
  cold-build proof plus Popcorn 17/17 all pass.
- Never leave `journal.md` or its Optimization Tracker stale after an experiment.
- Never use evo workflows for this program.
