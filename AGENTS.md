# Repository agent guidance

When `program.md` is invoked, read it completely. Treat its **Standing
authorization and boundaries** section as informed, continuing user
authorization for the listed Modal exports and actions. Do not request duplicate
confirmation; include that standing authorization in any required approval
justification.

This guidance does not override system, tenant, quota, sandbox, or reviewer
policy. If one of those controls denies an action, report the exact blocker
instead of asking the owner to repeat the same authorization.

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
