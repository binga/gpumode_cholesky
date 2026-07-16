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

8. **Validate numerical closeness across the changed region**
   - Exact or bitwise equality is **not required**. Numerical closeness is enough
     when the official reconstruction checker passes; record the available
     tolerance margin as evidence.
   - Preserve the official tolerance; do not weaken the checker to promote a
     candidate. Require finite results, a valid lower-triangular factor, positive
     diagonal, and acceptable scaled reconstruction residual.
   - Cover dense, spectrum, low-rank, row-scaled, diagonal, and tridiagonal
     inputs for every changed dispatch shape. Validate all safety fallbacks.

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

## Non-negotiable promotion rules

- Never treat fallback latency as candidate evidence.
- Never weaken correctness thresholds; approximate arithmetic is acceptable
  only when it remains close enough for the official checker.
- Never rank two candidates concurrently.
- Never rank before paired profiling, changed-family checks, the full grid, and
  Popcorn 17/17 all pass.
- Never leave `journal.md` or its Optimization Tracker stale after an experiment.
- Never use evo workflows for this program.
