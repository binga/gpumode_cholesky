# Cholesky leaderboard optimization program

This is the repository's repeatable operating program for GPU MODE Cholesky
optimization on B200. It is intentionally independent of evo workflows.

## Trigger with `set_goal`

Invoke this program by setting a goal whose objective names this file and the
target scope, for example:

```text
set_goal: Execute program.md for the slowest ranked Cholesky shapes.
Target at least 2.00x paired speedup per shape, then integrate and rank every
verified aggregate improvement.
```

When triggered, create one active goal for the complete program. Treat the goal
as achieved only after the adopted/rejected decision is documented, the exact
artifacts are committed, `main` is pushed, and the remote commit is verified.
Use checkpoints for long searches; do not mark the goal complete merely because
a promising candidate or local commit exists.

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

This is continuing authorization for every invocation of this program. No
additional user confirmation is required for these actions while the destination
remains the owner's Modal account and the export remains within the list above.

Never export credentials, tokens, environment files, unrelated repository
content, private user data, or secrets. Popcorn test/leaderboard submissions and
GitHub pushes remain explicit workflow actions.

If an execution tool requires an approval request, cite this standing
authorization in the justification and request the narrowest reusable approval
once. Do not ask the owner to restate or reconfirm this authorization. If a
system, tenant, quota, or reviewer policy denies the action, report that exact
policy blocker; repeated user confirmation will not resolve it.

## Workflow

1. **Synchronize the current winner**
   - Start from a clean `main` and synchronize it with `origin/main`.
   - Record the current ranked submission ID, exact commit, public/secret score,
     full-grid evidence, and source snapshot.
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
   - Default target: candidate paired mean latency at most 50% of the exact
     current ranked path, i.e. at least **2.00x speedup**.
   - State correctness, no-regression, cost, and ranked-submission guardrails.

4. **Delegate and babysit actively**
   - Inspect task checkpoints and raw evidence rather than accepting summaries.
   - Redirect stalls toward genuinely untried shape-specific Blackwell levers.
   - Reject fallback timings, missing backend evidence, cosmetic parameter
     sweeps, weakened gates, and results measured against stale baselines.

5. **Use a bounded architecture ladder**
   - Prefer materially different axes: vendor/expert APIs, per-matrix dispatch,
     blocked or left-looking factorizations, Triton, custom CUDA/tcgen05, TF32,
     BF16x9, FP8/MXFP8, CUDA Graphs, TMA, clusters/DSM, or refinement.
   - For the new set of experiments, remove cuSOLVER altogether and avoid
     stream-based approaches.
   - Measure up to six genuinely distinct serious variants before declaring
     bounded exhaustion. Preserve every valid partial frontier.

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
   - `WINNER`: correct and at least 2.00x faster on the paired target.
   - `FRONTIER`: correct and faster, but below 2.00x.
   - `REJECTED`: slower, incorrect, invalid, or fallback-only evidence.
   - `EXHAUSTED`: six distinct measured variants without a 2.00x winner.

10. **Narrow broad searches when evidence says to**
    - Stop or archive low-value tasks after bounded exhaustion.
    - Concentrate resources on the slowest remaining shapes and strongest
      partial frontiers.

11. **Integrate only verified, non-overlapping improvements**
    - Create a new numbered experiment from the latest ranked winner.
    - Rebase candidates after any intervening leaderboard win.
    - Combine positive frontiers only when their dispatch regions do not conflict.
      Leave unimproved shapes on their shipped implementation.

12. **Run the full 15-shape Modal B200 benchmark**
    - Use the owner's standing Modal authorization above, including profiler
      scripts when needed to explain regressions or unexpected dispatch costs.
    - Retain outputs as the official harness does, require every shape to pass,
      compare per-shape latency with the exact ranked baseline, and reject
      material off-target regressions.
    - Promote only when the aggregate geometric mean improves.

13. **Run Popcorn gates in order**
    - First run test mode and require **17/17**.
    - Audit the exact source, raw Modal artifacts, changed-family results, full
      grid, and test submission ID.
    - Then permit exactly one ranked submission at a time and monitor it until
      both public and secret runs finish.

14. **Adopt using completed leaderboard evidence**
    - Compare public and secret scores with the previous ranked winner.
    - If improved, adopt the exact ranked source at repository root.
    - If not improved, keep the previous winner and record the rejection without
      launching duplicate ranked retries unless a concrete defect was found.

15. **Maintain the journal and optimization tracker in detail**
    - Add a dated `journal.md` entry for every experiment, adopted or rejected.
      Record hypotheses, variants, component profiles, paired means/bests,
      speedups, numerical margins, fallbacks, failures, full-grid changes,
      Popcorn IDs, public/secret scores, costs, insights, and next ideas.
    - Update the living **Optimization Tracker** table after every measured
      architecture—not only winners. Mark shipped paths `✓`, tried/rejected paths
      `✗`, and only genuinely untested paths `TBD`; include the experiment/session
      reference and useful measured speedup in the cell.
    - Add columns when a new optimization family is tested. Remove stale `TBD`
      entries as soon as evidence exists. Keep the current-best line synchronized
      with the latest successful leaderboard submission.

16. **Preserve a reproducible experiment package**
    - Save the goal, exact baseline, every serious candidate, exact ranked source,
      raw paired/family/full-grid/ranked JSON, notes, and relevant harness changes
      under `experiments/NNN-*`.
    - Update the root README and `experiments/README.md` with the verdict.
    - Run final local, syntax, JSON, source-policy, whitespace, and snapshot checks.

17. **Commit, land on `main`, push, and verify**
    - Commit the complete experiment with a descriptive message.
    - Integrate it onto the latest `main`, push `main` to GitHub, and verify the
      remote branch resolves to the intended commit.
    - Only then complete the active goal and begin the next optimization cycle
      from this new ranked baseline.

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
  Popcorn 17/17 all pass.
- Never leave `journal.md` or its Optimization Tracker stale after an experiment.
- Never use evo workflows for this program.
