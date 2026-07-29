# Cholesky autoresearch loop (program2)

This is an **executable autoresearch program** for GPU MODE Cholesky optimization
on B200. It is a superset of `program.md`: it keeps that program's loop,
authorization, and gates unchanged, and folds in six practices drawn from two
papers that `program.md` does not yet operationalize:

- **AutoKernel** — *Autonomous GPU Kernel Optimization via Iterative Agent-Driven
  Search* (alphaXiv 2603.21331): profiling→optimize→verify phases, a tiered
  optimization playbook, an edit/benchmark/keep-revert loop, a five-stage
  verification harness, and a peak-utilization move-on rule.
- **KernelBench** — *Can LLMs Write Efficient GPU Kernels?* (arXiv 2502.10517):
  the `fast_p` correctness-and-speedup metric, repeated (parallel, high-temp)
  sampling vs. iterative refinement, and feeding compiler + execution + profiler
  feedback back into generation.

## Relationship to program.md — read this first

`program2.md` **does not replace** `program.md`. It inherits, unchanged:

- The **Standing authorization and boundaries** section (Modal, Popcorn, Git,
  subagents, isolated worktrees). Cite that authorization; do not re-request it.
- The **Non-negotiable promotion rules** and **secret-split policy** (name the
  promotion rule in the goal; a sub-1.5% paired grid does not predict the secret
  split).
- **Shared-workspace coordination**: `state.json`, leases under
  `experiments/.leases/`, one worker per shape, re-verify the incumbent before
  every paid gate.
- **Where things live**: `docs/STATUS.md` (incumbent), `docs/levers.md` (what to
  try), `docs/experiments.md` (what moved), `docs/lessons.md` (why),
  `journal.md` (narrative).

Everything below either **adds a gate**, **adds a search primitive**, or **adds a
context payload**. When `program2.md` and `program.md` agree, follow either. When
`program2.md` adds a gate, that gate is **mandatory** for work run under this
program.

## What is new here, at a glance

| # | Addition | Source | Where it plugs in |
|---|---|---|---|
| N1 | **Determinism gate** — repeated-run bitwise stability to catch races | AutoKernel stage 4 | Inner [I3.5], before paid timing |
| N2 | **Adversarial conditioning sweep** — near-singular / high-κ SPD inputs | AutoKernel stage 3 | Inner [I4], alongside families |
| N3 | **Roofline-% stop gate** — move on at ≥85% of the relevant peak | AutoKernel move-on | Inner [I7] classify |
| N4 | **Parallel best-of-N sampling** — N variants of one lever at high temp | KernelBench §5.1.1 | Inner [I2], bounded by budget |
| N5 | **G+E+P feedback payload** — codified compiler+exec+profiler turn context | KernelBench §5.1.2 | Inner refine sub-loop |
| N6 | **B200 hardware card + exemplar library** injected into every worker | KernelBench §5.2 | Worker bootstrap |
| N7 | **`fast_p` scoreboard** — correct-and-≥p-faster, reported per search | KernelBench §3.3 | Recording [I8] |

---

## N6. Worker bootstrap — the context every shape worker starts with

Before a shape worker writes any code, it loads, in order:

1. `docs/STATUS.md` — the live incumbent id, commit, SHA-256, scores.
2. `docs/levers.md` Part 1 row for its shape — shipped `✓`, rejected `✗`, `TBD`.
3. `docs/lessons.md` — Blackwell technique notes and the six operational failure
   modes.
4. **The B200 hardware card** (below).
5. **One or two exemplar kernels** matched to the lever it will pull (below).

### B200 hardware card (sm_100)

Inject verbatim into the worker's working context. These are the numbers a
Blackwell kernel is judged against.

```
GPU:                NVIDIA B200 (Blackwell, sm_100a)
SMs:                ~148
Tensor cores:       5th-gen; tcgen05.mma, block-scaled MX (MXFP8/MXFP4), 2-SM MMA
HBM3e bandwidth:    ~8 TB/s   (roofline ceiling for dram-bound work)
Shared mem / SM:    up to 228 KB (opt-in); 48 KB default per block
Registers / thread: 255 max;  register file 64K x 32-bit / SM
Legal-instruction note: work on a NON-DEFAULT CUDA STREAM is DISQUALIFIED by
  popcorn's static source scan. Cooperative grid-sync, clusters/DSM that require
  a private stream, and torch.linalg batched paths that spawn streams are all
  UNSHIPPABLE even when they profile as wins (lessons.md Part 2).
Precision order that actually shipped: TF32 > BF16x9 > native FP32 for trailing
  GEMM; FP16/BF16 plain lost to TF32; FP8/MXFP8 wins only at n>=32768.
Correctness gate: ||A - L Lᵀ||₁ <= 20·n·eps·||A||₁  (grows with n → large shapes
  have the most numerical headroom; small shapes have almost none).
```

### Exemplar library

Point the worker at the closest *shipped* winner as its worked example, not a
generic template:

- small-n custom CUDA (`4096×32`, `1024×64`, `256×128`, `64×256`): the rank-2 /
  blocked-16 warp kernels (levers.md Part 1, S36/S38/S39/S41).
- mid-n resident block: `e62_diag128` fused-block path (lessons.md; occupancy
  gate `batch ≤ ~148`).
- large-n left-looking: the `1×16384`/`1×32768` blocked path with TF32 trailing
  and the MXFP8 `1×32768` panel (levers.md Blackwell §2).

Exemplars raise the rate of *ambitious* attempts — and, per KernelBench §5.2,
also the rate of broken ones. That is exactly why N1 (determinism) and N2
(adversarial) are mandatory below.

---

## The autoresearch loop

The outer loop is `program.md`'s outer loop **unchanged** ([O1] sync → [O2] pick
targets → inner loops → [O3] integrate → [O4] full grid → [O5] cold build → [O6]
Popcorn test 17/17 → [O7] rank one → [O8] adopt). The inner loop gains N1–N5.

```
INNER LOOP (one shape, one lease, one worktree from the exact verified incumbent)

 [I1] PICK A LEVER      from docs/levers.md, cross-checked against experiments.md.
        |               Compute the Amdahl ceiling first (lessons.md): a lever
        |               whose ceiling < target is a FRONTIER at best — record it
        |               and size the effort accordingly. $0.
        v
 [I2] GENERATE (N4)     Instead of one edit, draft N distinct variants of THIS
        |               lever (default N=3–5). Vary the mechanism, not cosmetics:
        |               tile/warp shape, precision, barrier scheme, fusion
        |               boundary. Each variant is its own file in the worktree.
        |               Sequential single-edit is the N=1 special case; use it
        |               when the lever has one obvious form or budget is tight.
        v
 [I3] FREE GATES        py_compile, git diff --check, source-policy scan
        |               (NO non-default stream, no banned API), artifact parse.
        |               Kill broken variants here. NO GPU SPENT.
        v
 [I3.5] DETERMINISM (N1)  For each surviving variant, one Modal verify run that
        |               executes each changed shape's fast path >= 3 times on the
        |               SAME input and requires bitwise-identical output across
        |               repeats. A diff across repeats = a race (named barrier,
        |               missing compiler barrier, shared-mem staging) → REJECT the
        |               variant as buggy, even if it later "passes" a single
        |               closeness check. This is the cheap catch for the exp-063
        |               / lesson-5 failure class.
        v
 [I4] CORRECTNESS + ADVERSARIAL (N2)
        |               modal_verify.py verify + familygrid on the six families
        |               (dense/spectrum/lowrank/rowscale/diagonal/tridiagonal),
        |               reading per-row checker_ok, NOT the top-level flag.
        |               PLUS an adversarial conditioning sweep for every changed
        |               shape: near-singular SPD (κ ~1e6–1e10), tiny/near-zero
        |               diagonal, mixed extreme dynamic range. The official
        |               tolerance is NEVER weakened; the sweep only has to stay
        |               inside it. This is where TF32/FP8 tolerance-exploitation
        |               either survives or is caught before it reaches the secret
        |               split. spectrum is not generable at n>=16384 — gate the
        |               tractable five and record the omission.
        v
 [I5] PAIRED LATENCY    _gpu_runner.py pairedgrid, candidate vs the EXACT ranked
        |               source, one process, rotated inputs. Prove the intended
        |               backend ran (counters, zero fallbacks). Fallback timing is
        |               INVALID. Compare like with like: the probe's control must
        |               go through custom_kernel, not a bare driver call.
        v
 [I6] PROFILE          ncu_profile.py → dram% sm% tensor% occ% + stall breakdown;
        |               latency_budget.py → dram vs math vs residual.
        |               dram-bound → fewer passes / smaller dtype
        |               math-bound → faster pipe (tf32 → fp8)
        |               residual  → more parallelism / shorter dependent chain
        v
 [I5/I6 → REFINE (N5)]  If a variant is CORRECT but SLOW, or FAILS a gate, do not
        |   ^           silently discard it. Run up to R refinement turns
        |   |           (default R=3) feeding the next edit a codified payload:
        |   |             G = the previous variant source
        |   |             E = exact nvcc/py error text OR "correct, {latency}us"
        |   |             P = the dram%/sm%/tensor%/occ% + top stall reason
        |   +-----------  Stop refining a variant when it wins, when a turn makes
        |               it worse twice, or at R. Refinement turns count against
        |               the variant budget, not on top of it.
        v
 [I7] CLASSIFY + ROOFLINE STOP (N3)
        |    WINNER              >= research_target (default 2.00x) paired
        |    PROMOTABLE FRONTIER < target, reproducible aggregate positive
        |    FRONTIER            faster, banked
        |    REJECTED            slower / wrong / non-deterministic / fallback-only
        |    EXHAUSTED           max_serious_variants measured, no winner
        |    SATURATED (NEW)     achieved >= 85% of the relevant peak for the
        |                        binding resource (HBM bandwidth if dram-bound,
        |                        the shipped tensor-core TFLOP ceiling if
        |                        math-bound, or the Amdahl ceiling if serial-bound).
        |                        A SATURATED shape is DONE — stop spending on it and
        |                        move the lease, exactly as AutoKernel moves on at
        |                        ~90% of peak. Record the achieved % as the reason.
        v
 [I8] RECORD (N7)       manifest + state.json (atomic) + journal.md + one
                        experiment-matrix row. Report the fast_p line for this
                        search (below). Then next lever, until WINNER, EXHAUSTED,
                        or SATURATED.
```

### N7. The `fast_p` scoreboard for a search

At [I8], summarize the batch of variants with KernelBench's `fast_p`, adapted to
this repo's paired grid: `fast_p` = fraction of measured variants that are both
**correct (all gates incl. N1/N2)** and **>= p× the exact ranked source on the
paired target shape**. Report at least:

- `fast_0`  — correctness rate of the batch (how many even passed the gates).
- `fast_1`  — fraction faster than the incumbent at all.
- `fast_targ` — fraction reaching `research_target` (default 2.00x).

This turns "I tried some variants" into a comparable number across sessions and
makes N4's parallel sampling auditable: rising `fast_0` means the exemplar/context
is working; rising `fast_1` means the lever is real.

---

## Commands (real, executable now)

All Modal work goes through `uv` per `AGENTS.md`; needs `full_network`.

```bash
# Free correctness + families on the changed shape(s)
uv run --with modal -- python scripts/modal_verify.py verify \
    --submission experiments/NNN-x/candidate.py --shapes 16384,32768 \
    --json results/NNN-verify.json

uv run --with modal -- python scripts/modal_verify.py familygrid \
    --submission experiments/NNN-x/candidate.py --shapes 16384,32768 \
    --families dense,diagonal,lowrank,rowscale,tridiagonal \
    --json results/NNN-family.json

# Paired same-process grid: candidate vs the EXACT ranked source
uv run --with modal -- python scripts/modal_verify.py pairedgrid \
    --submission submission.py \
    --candidate experiments/NNN-x/candidate.py \
    --shapes 16384,32768 --json results/NNN-paired.json

# Profile a correct-but-slow variant
uv run --with modal -- python scripts/ncu_profile.py --shapes 32768 ...
uv run --with modal -- python scripts/latency_budget.py ...

# Popcorn gates (leaderboard name is `cholesky`; needs full_network)
popcorn submissions list --leaderboard cholesky
popcorn submit --mode test --no-tui submission.py         # require 17/17
popcorn submit --mode leaderboard --no-tui submission.py  # exactly ONE in flight
```

**N1 (determinism)** and **N2 (adversarial)** need two small harness additions —
a `--repeats K` stability check in `verify` and an `--adversarial` conditioning
generator. Until those land in `scripts/`, run N1 by calling `verify` on the same
seed twice and diffing the output tensors, and N2 by adding high-κ SPD inputs to
the family generator. Record the omission in `state.json` if a run predates the
harness change (`program.md` retry/cost policy: record and continue when stronger
authorized evidence covers the same property).

---

## Execute-now runbook

Concrete first cycle against the current board. Adjust only if `docs/STATUS.md`
has moved when you start.

1. **[O1] Sync.** `git fetch origin/main`; `popcorn submissions list
   --leaderboard cholesky`. Confirm the incumbent is `#922201`
   (SHA-256 `f108cbba…a62a429`) by `shasum -a 256` of root `submission.py`. If it
   moved, take the new winner as the baseline and rebase.

2. **[O2] Pick the target.** The measured wall is the **serial diagonal `potrf`**:
   59.6% of `1×16384` and 46.9% of `1×32768` (STATUS.md). cuSOLVER runs
   308–340 ns/row; this repo's best custom block kernel ended at 296 ns/row; the
   named lever is **named-barrier overlap inside the block kernel (~195 ns/row
   projected)**. Amdahl ceiling for `1×32768`: `1/(0.531 + 0.469·195/296)` ≈
   **1.18×** — so this is a FRONTIER/PROMOTABLE target, not a 2.00× winner.
   Record that ceiling in the goal, and **name the promotion rule** now
   (exp 065 shipped this exact lever to a +5.71% secret regression — do not spend
   a ranked slot on a sub-1.5% paired grid unless the goal accepts a secret
   coin-flip).

3. **Bootstrap the worker (N6).** Lease `experiments/.leases/NNN-large-diag.json`;
   worktree from `#922201`. Load the B200 card + the large-n left-looking
   exemplar + `lessons.md` (esp. lesson 5 compiler-barrier, and the exp-065
   named-barrier overlap notes in `experiments/065-*`).

4. **[I2] Generate N=3 variants (N4)** of the overlap lever, materially distinct:
   (a) named `bar.sync` participant split with the triangular-inverse warp behind
   the trailing update; (b) different warp-count / participant partition;
   (c) a wider `nb` schedule that changes the overlap granularity. One file each.

5. **[I3] Free gates** → **[I3.5] determinism (N1)**: 3 repeats per changed shape,
   require bitwise-identical output. This is the gate exp 065's overlap most needs
   — overlapped warps + shared memory is precisely the race surface.

6. **[I4] Correctness + adversarial (N2)**: five tractable families at
   16384/32768 (spectrum excluded, record it) **plus** near-singular SPD. Read
   per-row `checker_ok`; attribute any fallback against a baseline run.

7. **[I5] pairedgrid** vs the exact `#922201` source on `--shapes 16384,32768`;
   prove backend ran, zero new fallbacks.

8. **[I6] profile / [refine N5]** any correct-but-short variant with the G+E+P
   payload for up to 3 turns.

9. **[I7] classify** with the roofline stop: if a variant hits the ~1.18× Amdahl
   ceiling it is **SATURATED at target for this lever** — bank it as a PROMOTABLE
   FRONTIER and stop, do not chase the missing 0.8× that Amdahl already ruled out.

10. **[O3]→[O8]** only if the full 15-shape grid geomean improves and the named
    promotion rule is satisfied on **both** splits (given the secret-split
    history). Then cold build < 80% of the Popcorn timeout, test 17/17, rank
    exactly one, monitor to terminal public+secret, adopt or restore, push,
    verify remote.

---

## Retry, cost, and terminal states

Inherited verbatim from `program.md` ("Autonomous retry and cost policy" and
"Terminal states"). The only additions:

- **A non-deterministic variant (N1 fail) is `REJECTED`, not retried** as-is; fix
  the barrier/staging defect and it becomes a new variant.
- **A `SATURATED` shape closes like `SHAPE EXHAUSTED`**: preserve any promotable
  frontier, move the lease, do not keep spending against a resource ceiling you
  have already reached.
- **Parallel best-of-N (N4) counts every measured variant against
  `max_serious_variants_per_shape`.** Sampling breadth does not buy extra budget;
  it spends the same budget in parallel to raise `fast_0`/`fast_1` faster.

## Non-negotiables (unchanged, restated because they bind here too)

- Never treat fallback latency as candidate evidence.
- Never weaken the correctness threshold; approximate arithmetic is fine only
  while the official checker passes — the adversarial sweep (N2) exists to keep
  that honest.
- Never rank two candidates concurrently; never rank before paired profiling,
  changed-family + adversarial checks, the full grid, cold-build proof, and
  Popcorn 17/17 all pass.
- Never claim "both splits" or "public is enough" as settled policy — name the
  rule in the goal first.
- Never run on a non-default CUDA stream or any popcorn-banned construct, however
  well it profiles.
- Never leave `journal.md`, `docs/STATUS.md`, or the experiment matrix stale after
  an experiment.
