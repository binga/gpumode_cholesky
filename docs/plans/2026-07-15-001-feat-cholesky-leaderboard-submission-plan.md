---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
title: "feat: Land a correct ranked submission on the GPU MODE cholesky leaderboard"
type: feat
date: 2026-07-15
plan_depth: standard
leaderboard: cholesky (id 776)
gpu: B200
---

# feat: Land a correct ranked GPU MODE `cholesky` submission

## Summary

Participate in the GPU MODE **`cholesky`** competition ([leaderboard 776](https://www.gpumode.com/leaderboard/776?tab=rankings)) by landing a **correct, ranked** submission and establishing a reproducible remote submit/iterate loop. The immediate goal is a passing entry on the B200 leaderboard, not a top rank — deeper kernel optimization is explicitly deferred.

The task: batched dense Cholesky factorization. Input `A` is a `batch × n × n` `float32` CUDA tensor, symmetric positive definite up to FP32 roundoff. Return a lower-triangular `float32` tensor `L` with positive diagonal such that `A = L @ Lᵀ`. Correctness is **property-based** (structure + reconstruction residual), not elementwise against a single library. Ranking among passing submissions is the **geometric mean of runtime** across 15 benchmark shapes.

**Key constraint & verification strategy:** the development machine is macOS with **no local NVIDIA GPU**. Verification is therefore layered across three tiers: (1) a **free CPU property check** for logic/structure, (2) **real-B200 verification via Modal sandboxes** (`gpu="B200"`) for true Blackwell cuSOLVER numerics and timings without touching leaderboard quota, and (3) the official **`popcorn` CLI** submission to GPU MODE's B200 runners for ranked results. Modal is used to catch correctness/perf issues cheaply before spending scarce ranked quota.

**Product Contract preservation:** N/A — this is a direct (bootstrap) plan with no upstream `ce-brainstorm` artifact.

---

## Problem Frame

- **Who:** the user, a GPU MODE competitor, iterating from a Mac with no GPU.
- **What:** submit a single Python file implementing `custom_kernel(data) -> L` to the `cholesky` leaderboard and get a ranked, passing result on B200.
- **Why minimal-first:** landing *any* correct ranked submission de-risks the whole pipeline (auth, submission format, checker semantics, quota behavior) before investing in custom kernels. The winning trick (custom batched kernels for small-`n`/high-batch shapes) is real but is deferred follow-up work.
- **Success looks like:** the user's username appears on the `cholesky` B200 rankings with a passing geometric-mean time, produced by a command that can be re-run for future iterations.

### The evaluation contract (from `reference.py` / `eval.py`)

- **Interface:** `def custom_kernel(data: torch.Tensor) -> torch.Tensor`. Single file `submission.py`. Output is a *new* tensor (no pre-allocated output pattern for this problem).
- **Checker (`check_implementation`)** rejects unless: output is a `torch.Tensor`; `output.shape == data.shape`; `dtype == float32`; same device; all finite; strictly lower-triangular within `8 · n · eps · ‖A‖₁`; diagonal strictly positive; reconstruction `‖L·Lᵀ − A‖₁ ≤ 20 · n · eps · ‖A‖₁` (TF32 disabled during this check).
- **Reference (baseline to beat):** `torch.linalg.cholesky_ex(data, check_errors=False).L` (cuSOLVER).
- **Input families:** dense, diagonal, spectrum, lowrank, rowscale, tridiagonal SPD.
- **Modes:** `test` (correctness, every case), `benchmark` (perf, not ranked), `leaderboard` (ranked, rechecks every iteration with per-iter seed changes), `profile`.
- **Benchmark grid (15 shapes, geom-mean ranked):** `4096×32`, `1024×64`, `256×128`, `64×256`, `16×512`, `640×512`, `4×1024`, `60×1024`, `2×2048`, `8×2048`, `1×4096`, `2×4096`, `1×8192`, `1×16384`, `1×32768`.

---

## Requirements

- **R1** — A single self-contained `submission.py` implements `custom_kernel(data) -> torch.Tensor` and returns a correct `L` for all input families.
- **R2** — The submission passes `popcorn ... --mode test` (100% of test cases) on B200.
- **R3** — A `--mode leaderboard` submission completes and the entry appears on the `cholesky` B200 rankings with a passing status.
- **R4** — The submit command is reproducible and documented (directives embedded so `popcorn submit submission.py` works without re-typing flags).
- **R5** — `popcorn` CLI is authenticated and (if required) joined to the leaderboard before ranked submission.
- **R6** — A per-shape benchmark baseline is captured to a file, so future optimization can target the slowest shapes. *(Bridge to deferred work; low cost.)*
- **R7** — The submission is verified on a **real B200** via a Modal sandbox (correctness across the test grid) before any ranked popcorn submission.

Traceability: R1–R2 → U3/U7/U4; R3 → U5; R4 → U1; R5 → U2; R6 → U6; R7 → U7. (U7, the Modal B200 verification, is a new unit added on the "use Modal for local verification" revision; it slots after U3 and gates U4/U5.)

---

## Key Technical Decisions

- **KTD1 — First ranked submission = the reference one-liner.** Ship `torch.linalg.cholesky_ex(data, check_errors=False).L` as the first ranked entry. It is guaranteed correct across all input families and immediately lands on the board, satisfying the minimal goal. *Rationale:* zero correctness risk, fastest path to a ranked entry, and it establishes the loop. *Alternative considered:* start from the repo's `triton_cholesky32.py` (custom Triton for `n=32`, cuSOLVER fallback) — slightly faster but adds a correctness surface; deferred to follow-up (see Scope Boundaries).
- **KTD2 — Three-tier verification, Modal for real-GPU pre-flight.** No local GPU, so: (1) free CPU property check for logic; (2) **Modal sandbox with `gpu="B200"`** runs the real reference checker + timings on actual Blackwell hardware — this is the primary pre-submission gate, giving true cuSOLVER numerics and per-shape times without leaderboard quota; (3) `popcorn` for the official ranked result. *Rationale:* ranked quota is scarce and slow; Modal turns "submit and hope" into "verify on the exact hardware, then submit". *Cost note:* B200 sandbox time is billed per second — `verify` (small shapes) is cheap; full-grid `benchmark` (incl. 32768²) costs more, so it is run deliberately, not on every edit. *Alternative considered:* Modal `@app.function(gpu="B200")` instead of a sandbox — functionally equivalent; the sandbox form was chosen per the user's request and because it cleanly execs the same in-container runner for both verify and benchmark.
- **KTD3 — Embed `#!POPCORN` directives in `submission.py`.** Put `#!POPCORN leaderboard cholesky` and `#!POPCORN gpu B200` at the top so `popcorn submit submission.py` and `popcorn submit --mode test submission.py` work without repeating flags. *Rationale:* satisfies R4 reproducibility; matches CLI's documented directive support.
- **KTD4 — Structure `custom_kernel` as a shape dispatcher from day one.** Even though the body is initially just the cuSOLVER call, write it so shape-specialized branches can be added later without touching the interface. *Rationale:* makes the deferred optimization a localized change, not a rewrite.
- **KTD5 — Vendor the reference harness locally, read-only.** Copy `task.py`, `reference.py`, `eval.py`, and `utils.py` from `gpu-mode/reference-kernels` into the repo so the local property check and the plan are grounded in the exact checker code. *Rationale:* the checker semantics (tolerances, input families) are the real spec; keeping them local prevents drift.

---

## Output Structure

```
gpumode_cholesky/
├── submission.py                 # the entry point (custom_kernel + #!POPCORN directives)
├── reference/                    # vendored, read-only harness from reference-kernels
│   ├── task.py
│   ├── reference.py
│   ├── eval.py
│   └── utils.py
├── scripts/
│   ├── verify_local.py           # CPU property check on tiny shapes (free)
│   ├── modal_verify.py           # real-B200 verify/benchmark via Modal sandbox (driver)
│   └── _gpu_runner.py            # runs inside the Modal sandbox (not run locally)
├── results/                      # captured test/benchmark outputs (gitignored except summaries)
│   └── baseline-benchmark.json
├── docs/
│   └── plans/2026-07-15-001-feat-cholesky-leaderboard-submission-plan.md
├── .gitignore
└── README.md
```

The tree is a scope declaration; per-unit `Files` lists are authoritative.

---

## Implementation Units

### U1. Project scaffolding and popcorn project setup

- **Goal:** a committed skeleton with the entry point, vendored reference harness, and reproducible submit config.
- **Requirements:** R1 (skeleton), R4 (directives), KTD3, KTD5.
- **Dependencies:** none.
- **Files:**
  - `submission.py` (create) — `custom_kernel` dispatcher (initially cuSOLVER one-liner) + `#!POPCORN leaderboard cholesky` / `#!POPCORN gpu B200` directives.
  - `reference/task.py`, `reference/reference.py`, `reference/eval.py`, `reference/utils.py` (create) — vendored read-only from `gpu-mode/reference-kernels/problems/linalg/cholesky_py/` (`utils.py` comes from `problems/pmpp_v2/utils.py` per `task.yml`).
  - `.gitignore` (create) — ignore `results/*` except `results/*.json` summaries, `__pycache__/`, venv.
  - `README.md` (create) — one-paragraph problem statement + the exact submit/test commands.
- **Approach:** Run `popcorn setup` in the project first (it bootstraps agent skills + a submission template); reconcile its output with this structure. Download the four reference files from the pinned raw URLs. Keep `custom_kernel` as `if`-dispatch on `(batch, n)` with a single default branch calling `torch.linalg.cholesky_ex(...).L`.
- **Execution note:** mostly scaffolding/config; prefer a runtime/structure smoke check over unit tests.
- **Test scenarios:** none for the skeleton itself — `Test expectation: none -- scaffolding/config`. The behavioral check lives in U3.
- **Verification:** `python -c "import ast; ast.parse(open('submission.py').read())"` succeeds; `submission.py` contains both `#!POPCORN` directives; the four `reference/` files exist and import without error on CPU.

### U2. Authenticate the popcorn CLI and access the leaderboard

- **Goal:** the CLI is registered and able to submit to `cholesky` on B200.
- **Requirements:** R5.
- **Dependencies:** U1.
- **Files:** none (may write a credential/config file under the user's home per the CLI; do not commit secrets).
- **Approach:** Run `popcorn register` (or `reregister`) and complete the login flow. Confirm identity/access with `popcorn submissions` (should list an empty/known history rather than an auth error). If `cholesky` turns out to be closed, use `popcorn join <invite-code>` — otherwise no join needed.
- **Execution note:** interactive auth; capture whatever token/config path it produces and add it to `.gitignore`.
- **Test scenarios:** none (`Test expectation: none -- interactive auth`).
- **Verification:** `popcorn submissions` returns without an authentication error; a dry `popcorn submit --mode test submission.py --no-tui` reaches the runner (does not fail at the auth stage).
- **Open dependency:** exact auth mechanism (Discord OAuth vs token) is unconfirmed — see Open Questions Q1.

### U3. Local device-agnostic property check

- **Goal:** validate `custom_kernel` logic against the *real* checker on CPU before spending remote quota.
- **Requirements:** R1 (correctness confidence).
- **Dependencies:** U1.
- **Files:** `scripts/verify_local.py` (create).
- **Approach:** Import `generate_input` and `check_implementation` from `reference/reference.py` and `custom_kernel` from `submission.py`. Iterate the small entries of the `tests:` grid (`batch≤16`, `n≤256`) forcing CPU, and assert `check_implementation` passes for each. This exercises all input families (`dense`, `diagonal`, `spectrum`, `lowrank`, `rowscale`, `tridiagonal`). Note in the script header that cuSOLVER-specific numerics differ on GPU, but the *property* checks (structure, positive diagonal, reconstruction residual) are device-agnostic.
- **Test scenarios:**
  - Covers R1. `dense` SPD, `n=32,64,128`, `batch≤16` → `check_implementation` returns `(True, ...)`.
  - Covers R1. `diagonal` and `spectrum` families, `n=32,64,128` → passes (guards against structural/tolerance regressions on well-conditioned edge inputs).
  - Covers R1. `lowrank` and `rowscale`, `n=256` → passes (damped/ill-scaled inputs still reconstruct within tolerance).
  - Covers R1. `tridiagonal`, `n=512` → passes.
  - Edge: `n=1` single-element matrix and minimum `batch=1` → lower-triangular + positive-diagonal checks hold (guards the dispatcher's degenerate branch).
- **Verification:** `python scripts/verify_local.py` prints all-pass for every listed spec on CPU; a deliberately broken variant (e.g., returning the upper triangle) is reported as failing, proving the harness actually gates.

### U7. Real-B200 verification via Modal sandbox

*(New unit from the "use Modal for local verification" revision. Runs after U3, gates U4/U5.)*

- **Goal:** verify `custom_kernel` on an actual B200 — real cuSOLVER numerics and per-shape timings — before spending popcorn quota.
- **Requirements:** R7 (correctness on real hardware), R1.
- **Dependencies:** U1, U3.
- **Files:**
  - `scripts/modal_verify.py` (create) — Modal driver: builds an image (torch cu128 + vendored `reference/` + `submission.py` + runner), creates a `modal.Sandbox` with `gpu="B200"`, execs the runner, parses a `RESULT_JSON:` line.
  - `scripts/_gpu_runner.py` (create) — runs *inside* the sandbox; reuses `generate_input`/`check_implementation`/`custom_kernel`; `verify` mode over the test grid, `benchmark` mode over the 15-shape grid with `cuda.Event` timing + geomean.
- **Approach:** Ensure `modal` is available locally (`uv run --with modal ...` or a project venv); `~/.modal.toml` is already present so auth is set. `python scripts/modal_verify.py` runs `verify` (small shapes, cheap). The image pins/uses a torch build with Blackwell (sm_100) support; if the default wheel lacks it, fall back to an explicit `cu128` index. Keep `benchmark` mode opt-in for U6.
- **Execution note:** correctness-first; run `verify` before any ranked submission. Guard cost — do not run full `benchmark` on every edit.
- **Test scenarios:**
  - Covers R7. `verify` mode: all test-grid specs (dense/diagonal/spectrum/lowrank/rowscale/tridiagonal, n=32..2048) report `PASS` on B200; runner emits `passed: true`.
  - Failure path: a deliberately broken submission (e.g. return upper triangle) is reported `FAIL` by the in-sandbox `check_implementation`, proving the Modal path actually gates.
  - Infra: sandbox reports `torch.cuda.get_device_name(0)` containing "B200" (confirms the GPU request was honored).
- **Verification:** `python scripts/modal_verify.py` prints all-`PASS` and `passed: true`; device name confirms B200.

### U4. Remote correctness test on B200 (popcorn)

- **Goal:** confirm the submission passes the official checker/runner on the target hardware.
- **Requirements:** R2.
- **Dependencies:** U2, U3, U7.
- **Files:** `results/test-<date>.txt` (create, gitignored) — captured CLI output.
- **Approach:** `popcorn submit --mode test --no-tui submission.py` (leaderboard/gpu supplied by directives). Confirm every test case reports `pass` and the final `check: pass`.
- **Test scenarios:**
  - Covers R2. All 17 `tests:` specs report `pass` in `--mode test`.
  - Failure path: if any case fails, the captured error message (index + values) is triaged against `check_implementation` before re-submitting (avoids blind quota burn).
- **Verification:** CLI output shows `check: pass`; saved to `results/`.

### U5. Land the ranked leaderboard submission

- **Goal:** a passing, ranked entry on the `cholesky` B200 board — the plan's primary success criterion.
- **Requirements:** R3.
- **Dependencies:** U4, U7.
- **Files:** `results/leaderboard-<date>.txt` (create, gitignored).
- **Approach:** `popcorn submit --mode leaderboard --no-tui submission.py`. Wait for completion (ranked runs recheck every iteration and can take longer — `ranked_timeout` is 1200s). Then confirm the entry via `popcorn submissions` and the [rankings page](https://www.gpumode.com/leaderboard/776?tab=rankings).
- **Test scenarios:**
  - Covers R3. `--mode leaderboard` completes with `check: pass` across all 15 benchmark shapes.
  - Covers R3. The username appears in `popcorn submissions` and on the B200 rankings tab with a recorded geometric-mean time.
- **Verification:** ranked run reports pass; entry visible on the leaderboard; geom-mean time recorded in `README.md`.

### U6. Capture per-shape benchmark baseline

- **Goal:** a per-shape timing breakdown that tells future optimization which shapes are slowest. *(Bridge to deferred work.)*
- **Requirements:** R6.
- **Dependencies:** U5 (or U7 for the Modal path).
- **Files:** `results/baseline-benchmark.json` (create, committed summary).
- **Approach:** two interchangeable sources — `python scripts/modal_verify.py benchmark --json results/baseline-benchmark.json` (Modal B200, no quota cost, preferred for iteration) and/or `popcorn submit --mode benchmark --no-tui -o results/... submission.py` (official numbers). Record per-shape means; annotate in `README.md` which shapes dominate the geom mean (expectation: small-`n`/high-batch shapes like `4096×32`, `1024×64` are where cuSOLVER overhead is largest and custom kernels would win most).
- **Test scenarios:** none (`Test expectation: none -- measurement, not behavior`).
- **Verification:** `results/baseline-benchmark.json` contains a time for all 15 shapes; the 3 slowest shapes are noted in `README.md` as the future-optimization targets.

---

## Scope Boundaries

**In scope:** correct `custom_kernel`, authenticated CLI, a passing ranked B200 submission, reproducible commands, and a baseline per-shape timing capture.

### Deferred to Follow-Up Work
- **Custom small-`n` / high-batch kernels** (Triton or CUDA via `load_inline`) for `n ∈ {32, 64, 128}` where cuSOLVER's per-call overhead dominates — the primary lever for climbing the board. Start from the repo's `triton_cholesky32.py` pattern (one program per matrix, 32×32 factorized in registers).
- **Per-shape dispatch tuning** in `custom_kernel` (thresholds, warp counts, block sizes) once baseline timings identify the slow shapes.
- **Large-matrix path** (`n ≥ 4096`) — likely leave on cuSOLVER; revisit only if profiling shows headroom.
- **`profile` mode / Nsight Compute** analysis to guide kernel work.

### Out of scope (not this competition's identity)
- Non-B200 GPUs, other leaderboards, multi-GPU/distributed variants.
- Changing the evaluation harness or checker.

---

## Open Questions

- **Q1 (blocking U2):** What auth flow does `popcorn register` use on this machine (Discord OAuth in browser vs. pasted token), and is a GPU MODE / Discord account already set up? Resolve at execution time by running `popcorn register` and following prompts.
- **Q2 (non-blocking):** Is `cholesky` an open leaderboard or does it need `popcorn join <code>`? Assume open (public rankings visible); fall back to `join` if submission is rejected for access.
- **Q3 (non-blocking):** Remote quota / rate limits per day for ranked submissions — unknown. Mitigate by gating every ranked run behind a passing `--mode test` (U4) and the local CPU check (U3).

---

## Risks & Mitigations

- **No local GPU ⇒ slow feedback loop.** Mitigation: free CPU property check (U3) → real-B200 Modal verify (U7) → cheap `--mode test` (U4) before any ranked run; never debug correctness on the ranked path.
- **Modal B200 cost.** Billed per second; the full `benchmark` grid (incl. 32768²) is not free. Mitigation: default to cheap `verify`; run full `benchmark` deliberately; sandboxes are `terminate()`d in a `finally` block so nothing lingers.
- **Blackwell (sm_100) torch wheel support.** If the pulled torch wheel lacks Blackwell kernels, the Modal run errors on the GPU. Mitigation: pin an explicit `cu128` torch build in the Modal image; the error surfaces immediately in `verify`, not on the ranked path.
- **cuSOLVER numerics vs. CPU numerics differ.** The reconstruction tolerance is `20·n·eps·‖A‖₁`; the reference itself is cuSOLVER, so the one-liner is inherently within tolerance on GPU. CPU check validates *structure and logic*, not exact GPU residuals — documented in `verify_local.py`.
- **Auth / access friction (Q1, Q2).** Mitigation: isolate auth as its own unit (U2) with an explicit non-auth verification step before submitting.
- **Wasting ranked quota on avoidable failures (Q3).** Mitigation: strict U3 → U4 → U5 ordering.

---

## Definition of Done

- `submission.py` passes the CPU check (U3) and **real-B200 Modal verify** (U7, R7), then `popcorn --mode test` (R2) and a `--mode leaderboard` run reports `check: pass` (R3).
- The user's entry is visible on the `cholesky` B200 rankings with a recorded geom-mean time (R3).
- `popcorn submit submission.py` works via embedded directives, and the commands are in `README.md` (R4).
- `results/baseline-benchmark.json` holds per-shape times, with the slowest shapes flagged as deferred-optimization targets (R6).

---

## Sources & Research

- Competition page: https://www.gpumode.com/leaderboard/776?tab=rankings
- Problem source (`gpu-mode/reference-kernels`): `problems/linalg/cholesky_py/{task.yml,task.py,reference.py,eval.py,submission.py,submissions/triton_cholesky32.py}`
- Submission format & modes (DeepWiki): https://deepwiki.com/gpu-mode/reference-kernels/8-submission-guidelines
- CLI: `gpu-mode/popcorn-cli` (installed locally as `popcorn`); commands `setup`, `register`, `join`, `submit --mode {test,benchmark,leaderboard,profile}`, `submissions`.
- Modal: `modal.Sandbox.create(..., gpu="B200")` + `.exec(...)` — [Sandbox API](https://modal-labs-modal-client.mintlify.app/api/python/sandbox), [GPU guide](https://frontend.modal.com/docs/guide/gpu.md) (B200 supported). `~/.modal.toml` present locally; `modal` package installed on demand via `uv`.
- Technique context (deferred): MAGMA batched Cholesky; cuSOLVER `potrfBatched`; the repo's Triton 32×32 reference submission.
