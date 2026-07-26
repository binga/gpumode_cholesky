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

## Cheap gates that are still under-used

- **Compute the Amdahl ceiling before opening a shape.** It costs nothing and it
  classifies the outcome in advance. Exp 065: the diagonal is ~55% of a mid
  shape and the best predicted block speedup was 1.52x, so
  `1 / (0.45 + 0.55/1.52)` = 1.23x was the ceiling — correctly ruling out the
  2.00x research target before any GPU spend, and correctly predicting a
  frontier rather than a winner. Record the ceiling in the goal.
- **Reuse the candidate's own probe hook rather than writing a harness.**
  `mid_probe()` plus a variant-templated kernel gave, in **one** Modal run:
  per-block timing, the phase-cycle breakdown, `abs_err`/`inv_err` against an
  in-source control, and whole-shape speedups on five shapes. Restrict the
  variant tuple to (control, candidate) — probing the whole ladder every time
  multiplies the bill for rows the decision does not turn on.
- **The previous experiment's "next levers" section is load-bearing.** Exp 065
  took one variant instead of six because exp 064 had already named the lever
  and costed it. Keep those sections quantitative; they are the cheapest input
  the next experiment gets.

## Blackwell kernel technique notes

- **`bar.sync <id>, <count>` gives a partial-block barrier.** When a subset of
  warps must synchronise while other warps are elsewhere in the kernel,
  `__syncthreads()` deadlocks — it is barrier 0 over every thread in the block.
  A named barrier with an explicit participant count (a multiple of the warp
  size) is the primitive. Exp 065 used id 1 over 224 threads so warps 1-7 could
  synchronise between staging and the trailing update while warp 0 sat inside
  the triangular inverse.
- **Overlap is available wherever one warp holds a serial phase and the others
  idle** — but only after proving the memory footprints are disjoint. Exp 065's
  three regions (`S[kk:kk+32,kk:kk+32]` read-only, `S[lwid:,kk:kk+32]` -> `P`,
  `P` -> `S[lwid:,lwid:]`) provably do not alias, which is why the arithmetic
  came out bit-identical to the control. Establish that first; the win is
  scheduling, and a scheduling change that alters results is a bug.
- **Expect the overlapped serial phase to get slightly slower.** Exp 065's
  `triinv` grew 2% (37019 -> 40341 cycles) from shared-memory contention with
  the seven warps now working behind it. That is the price of the overlap, not
  a defect, and it is small next to the phases that leave the timeline.

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
