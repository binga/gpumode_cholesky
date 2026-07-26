# Repository agent guidance

`program.md` is the canonical operating program for this repository. When it is
invoked by name, or through a goal that clearly requests the Cholesky
optimization program, read it completely and follow it as continuing user
instruction.

Treat its **Standing authorization and boundaries** section as the owner's
informed, continuing authorization for every listed Modal, Popcorn, Git, GitHub,
coordination, and delegation action. Do not request duplicate confirmation. If an
execution environment still requires approval, cite that standing authorization
and request the narrowest reusable approval once.

## Where things live

One question, one file. Do not restate a fact that another file owns.

| I need to know… | Read |
|---|---|
| what is ranked right now | `docs/STATUS.md` |
| how the loop runs, and what I may do | `program.md` |
| what to try next | `docs/levers.md` |
| what has been tried, and what it moved | `docs/experiments.md` |
| what we learned the hard way | `docs/lessons.md` |
| the narrative of a past experiment | `journal.md`, `experiments/NNN-*/notes.md` |

## Standing rules

- **The owner lands work directly on `main`; pull requests are not used here.**
  Commit the complete experiment, fast-forward `main`, push, and verify the
  remote commit. Never force-push and never rewrite published history.
- **Preserve unrelated user changes.** A dirty root worktree is not a reason to
  stop or to ask the owner to clean it. Create an isolated worktree from the
  exact verified incumbent instead.
- **One worker per shape.** Program invocations authorize bounded subagents and
  isolated worktrees; give each a non-overlapping shape and its own artifact
  directory, and hold a lease under `experiments/.leases/`. Only one task may own
  a shape, an integration, or the Popcorn ranked slot at a time.
- **Re-verify the incumbent before every paid gate.** Fetch `origin/main` and
  compare commit *and* ranked-source hash; `origin/main` can move mid-session.
  Keep root `submission.py` aligned with the adopted ranked winner.
- **Six failure modes have each cost a session or a paid gate.** They are in
  `docs/lessons.md` Part 1. Read it before your first Modal spend.

## Autonomy and interruption policy

Continue autonomously through authorized profiling, mechanical fixes, transient
retries, evidence collection, documentation, commits, rebases, qualified
submissions, monitoring, adoption, pushing, and remote verification.

Interrupt the owner only for a policy denial, an exhausted declared budget,
missing credentials, destructive ambiguity, overlapping owner edits that cannot
be preserved, or a strategic choice that materially changes the declared goal.

Report decisions and state transitions, not unchanged polling. During long remote
jobs, give a compact update about every 10-15 minutes or when a gate completes,
fails, or changes the next action.

This guidance does not override system, tenant, quota, sandbox, reviewer, or
service policy. If one of those denies an action, report the exact blocker
instead of asking the owner to repeat authorization already recorded here.
