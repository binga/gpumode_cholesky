# Repository agent guidance

`program.md` is the canonical operating program for this repository. When it is
invoked by name or through a goal that clearly requests the Cholesky optimization
program, read it completely and follow it as continuing user instruction.

Treat the **Standing authorization and boundaries** section of `program.md` as
the repository owner's informed, continuing authorization for every listed
Modal, Popcorn, Git, GitHub, coordination, and delegation action. Do not request
duplicate confirmation. If an execution environment still requires approval,
cite that standing authorization in the justification and request the narrowest
reusable approval once.

## Shared-workspace operation

- Preserve unrelated user changes. A dirty root worktree is not a reason to
  stop or ask the owner to clean it; create an isolated worktree from the exact
  verified incumbent instead.
- Program invocations authorize bounded subagent delegation and isolated Git
  worktrees. Give each worker a non-overlapping shape or evidence task and its
  own artifact directory. Do not let multiple workers edit the same candidate.
- Use the leases defined by `program.md`. Only one task may own a target shape,
  integration, or Popcorn ranked submission at a time.
- Before expensive promotion gates, integration, and pushing, fetch and compare
  `origin/main` plus the ranked-source hash. If the incumbent changed, stop
  stale-baseline spending, preserve completed evidence, rebase or re-create the
  candidate from the new incumbent, and revalidate the affected gates.
- Keep the shared repository-root `submission.py` aligned with the adopted
  ranked winner. Develop unranked candidates in isolated worktrees or numbered
  experiment directories and copy one to the root only after adoption.

## Autonomy and interruption policy

Continue autonomously through authorized profiling, mechanical fixes, transient
retries, evidence collection, documentation, commits, rebases, qualified
submissions, monitoring, adoption, pushing, and remote verification. Interrupt
the owner only for a policy denial, exhausted declared budget, missing
credentials, destructive ambiguity, overlapping owner edits that cannot be
preserved, or a strategic choice that materially changes the declared goal.

Report decisions and state transitions, not unchanged polling. During long
remote jobs, provide a compact update about every 10-15 minutes or when a gate
completes, fails, or changes the next action.

This guidance does not override system, tenant, quota, sandbox, reviewer, or
service policy. If one of those controls denies an action, report the exact
blocker instead of asking the owner to repeat authorization that is already
recorded here.

## Standing operational rules

The owner lands this work directly on `main`; pull requests are not used here.
Commit the complete experiment, fast-forward `main`, push, and verify the remote
commit. Never force-push and never rewrite published history.

Four failure modes have each cost a session or a paid gate. `program.md` carries
the detail; the short form:

1. **The ranked incumbent is whatever `popcorn submissions list` says**, not
   what `main` or `README.md` says. Verify by source hash before spending, and
   again before any paid gate — `origin/main` can move mid-session.
2. **Do not edit a Modal-mounted file while a job is building** (`submission.py`,
   the candidate, `reference/`, `scripts/_gpu_runner.py`). It kills the run with
   `modified during build process`.
3. **Compare like with like in probes.** Driver-direct rows and
   `custom_kernel` rows differ by the wrapper's dispatch and end-of-call sync.
4. **Attribute every fallback against a baseline run** before calling it
   pre-existing.

## Lessons from exp 065

5. **The secret-split rule is not settled.** Exp 035 was *adopted* with public
   -2.94% / secret +5.26%; exp 065 was *rejected* on the same signature
   (-3.79% / +5.71%). Before spending a ranked slot on anything that could split,
   state which `promotion_policy` you are applying. Do not infer it — precedent
   contradicts itself.
6. **A paired grid under ~1.5% does not predict the secret split's sign.** Three
   instances (exps 022, 035, 065). Exp 065 had CI95 excluding 1.0 on all fifteen
   shapes, identical counters, zero fallbacks, bit-identical correctness — and
   still regressed secret by 5.71%.
7. **Compute the Amdahl ceiling before opening a shape.** Free, and it classifies
   the outcome in advance: exp 065's `1/(0.45 + 0.55/1.52)` = 1.23x correctly
   ruled out the 2.00x target before any GPU spend.
8. **`bar.sync <id>, <count>` is the primitive for a partial-block barrier.**
   `__syncthreads()` deadlocks when only some warps reach it. Prove the memory
   footprints are disjoint first — a scheduling change that alters results is a
   bug, and exp 065's came out bit-identical because they provably did not alias.
