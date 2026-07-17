# Journal — GPU MODE `cholesky` leaderboard

Running log of work, results, and findings. Newest entries at the top.

The **Optimization Tracker** immediately below is a *living* document (not a
dated session entry): update its cells as paths are shipped, rejected, or newly
identified. The dated session log starts after it.

---

## Required end-to-end experiment workflow

The canonical triggerable version of this workflow, including standing Modal
profiling authorization and promotion gates, is in `program.md`.

Every experiment, whether **adopted or rejected**, is complete only after this
entire workflow has run and the result is present on GitHub:

1. **Synchronize** — start from a clean checkout and `git pull --ff-only` so the
   experiment is based on the latest remote history.
2. **Frame the goal** — record the current ranked baseline, target shapes,
   hypothesis, success threshold, correctness constraints, cost/submission
   guardrails, and a bounded fallback ladder in `docs/goal-expNNN-*.md`.
3. **Isolate one change** — preserve the baseline and every serious candidate in
   `experiments/NNN-*/`; do not bundle unrelated optimizations before the first
   causal measurement.
4. **Run free checks first** — property/correctness tests, syntax/compile checks,
   artifact parsing, and `git diff --check` must pass before remote GPU spend.
5. **Measure progressively** — use paired same-process B200 probes on the smallest
   representative target first; expand to expensive shapes only after a credible
   win. Record raw JSON and compare against the current shipped path.
6. **Validate the changed dispatch region** — cover dense, diagonal, spectrum,
   low-rank, row-scaled, and tridiagonal families, including every fallback path.
7. **Run the full grid** — a credible finalist must pass all 15 ranked shapes and
   must not regress shapes outside its dispatch region.
8. **Use Popcorn gates in order** — require test mode 17/17, then make at most one
   justified ranked submission and wait for the leaderboard result. Rejected or
   unpromising experiments do not spend a ranked submission.
9. **Adopt or reject explicitly** — copy a winner to root `submission.py`, or
   leave it isolated if rejected. Update experiment notes/artifacts, the root
   README, `experiments/README.md`, this journal's Optimization Tracker, and a
   dated session entry with results, costs, failures, insights, and next ideas.
10. **Commit the complete experiment** — stage only the experiment's code,
    artifacts, goal, harness changes, and documentation; run final checks; create
    one descriptive commit (or a clearly documented follow-up completion commit).
11. **Push and verify GitHub** — `git push origin <branch>` and verify the remote
    branch contains the experiment commit. A local commit, successful leaderboard
    run, or journal entry alone is **not done**. Record the commit and publication
    state in the session entry.

The supervising task owns the terminal gates: prevent duplicate ranked
submissions, recover transient failures, ensure documentation is complete, and
do not report success until both the ranked/adoption decision and GitHub push are
confirmed.

---

## Optimization Tracker (living — update on progress/regress)

Rows = the 15 ranked B200 shapes. Columns = latency-reduction levers. Cells:

- **✓** — shipped / current winning path for that shape (see the referenced session).
- **TBD** — plausible lever, not yet conclusively tried (a path worth exploring).
- **✗** — tried and rejected, or not applicable / no expected benefit for this shape.

Current best: **`#881981` = 1262.9337990784535μs public geomean** (Session 15;
secret 1270.7067480724075μs). `nb` = block size.

| Shape (b×n) | Batched cuSOLVER | Per-matrix loop | Triton kernel | Custom CUDA (tcgen05/CUTLASS) | Blocked / tiled | TF32 trailing | BF16x9 FP32-emu | FP8 / MXFP8 + iter-refine | CUDA Graphs |
|---|---|---|---|---|---|---|---|---|---|
| 4096×32  | ✗ | ✗ | **✓** (S2) | TBD | ✗ | ✗ | ✗ | ✗ | TBD |
| 1024×64  | ✗ (S15) | ✗ | ✗ (S2/S15 0.67×) | ✗ (S3) | ✗ | ✗ | TBD | TBD | **✓** (S15, 1.09×) |
| 256×128  | ✗ | ✗ | ✗ (S2) | ✗ (S3) | TBD | TBD | TBD | TBD | **✓** (S9; manual capture S15) |
| 64×256   | ✗ (S15) | TBD | **✓** (S15, 1.35×) | TBD | **✓** (S15) | ✗ (tf32x3) | TBD | TBD | ✓ (in-path S15) |
| 16×512   | ✗ | TBD | **✓** (S15, 1.17×) | TBD | **✓** (S15) | ✗ (tf32x3) | TBD | TBD | ✓ (S9→S15 in-path) |
| 640×512  | ✗ (S5/S15) | ✗ (S5) | **✓** (S15, 1.71×) | TBD | **✓** (S15) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 4×1024   | ✗ | ✗ (S15) | **✓** (S15, 1.42×) | ✗ | **✓** (S15) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 60×1024  | ✗ (S15) | ✗ (S4) | **✓** (S15, 1.99×) | ✗ | **✓** (S15) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 2×2048   | ✗ | **✓** (S4) | ✗ (S15, 0.65×) | ✗ | ✗ (S15) | TBD | TBD | TBD | TBD |
| 8×2048   | ✗ (S9) | ✗ (S5) | **✓** (S15, 1.59×) | ✗ | **✓** (S15) | **✓** (S15) | TBD | TBD | ✓ (in-path S15) |
| 1×4096   | **✓** | — | ✗ (S15 cand-B 0.18–0.97×) | ✗ | ✗ (S15) | TBD | TBD | TBD | ✗ (S15, 0.97×) |
| 2×4096   | ✗ | **✓** (S4) | ✗ (S15 cand-B) | ✗ | ✗ (S15) | TBD | TBD | TBD | TBD |
| 1×8192   | **✓** | — | ✗ | ✗ | ✗ 1.07× (S6) | ✗ 1.07× (S6) | ✗ 0.95× (S7) | TBD | ✗ |
| 1×16384  | ✗ | — | ✗ | ✗ | **✓ left-looking** (S10) | **✓ active-panel** (S10) | ✗ 1.15× (S7) | TBD | ✗ |
| 1×32768  | ✗ | — | ✗ (S13/S14 no-cusolver potrf) | ✗ (S13/S14) | **✓ left-looking** (S10) | **✓ diagonal** (S10) | ✗ (S7, extrap.) | **✓ native FP8 panel + fused quantization** (S10/S14, 1.084×) | ✗ (S13 two-level) |

Notes: **CUDA streams** win several launch-bound shapes but are **banned** by
popcorn's static source scan (S4/S6) — not a column. FP16/BF16 (plain, not
BF16x9) were tried in the blocked path and **lost to TF32** on B200 (S6), so
they're folded into the TF32 result rather than tracked separately.

### Blackwell B200-specific candidate solutions (the "what else")

These are the levers that are *specific to* / most leveraged by the B200
(sm_100, 5th-gen tensor cores, `tcgen05.mma`, block-scaled MX formats). Ordered
roughly by expected ROI given the loose accuracy gate (`‖A−LLᵀ‖₁ ≤ 20·n·eps·‖A‖₁`,
which *grows with n* → the huge shapes have the most numerical headroom).

1. **BF16x9 FP32 emulation on tensor cores** — ✗ **REJECTED (S7).** Engages on the
   B200 via `CUBLAS_EMULATE_SINGLE_PRECISION=1` + `CUBLAS_FP32_EMULATED_BF16X9_MATH=1`
   (the BF16X9 var alone does nothing; the PyTorch `fp32_precision` knob only exposes
   ieee/tf32, no BF16x9). Confirmed engaged (standalone 8192 FP32 matmul 16.7→6.3ms,
   2.6×) and ≈FP32-accurate (margins 65k–139k× vs TF32's ~100–210×; and *robust* where
   TF32 NaNs on lowrank). **But slower than the shipped paths:** 8192 0.95× vs cuSOLVER,
   16384 bf16x9 1.15× vs TF32's 1.60×. Reason: BF16x9 ≈ 6–9 BF16 products per FP32
   GEMM, so ~3× slower than a single-product TF32 GEMM — TF32 tensor cores are the real
   bar, not native FP32. Speed order TF32 > BF16x9 > native FP32. See exp 007.
2. **FP8 / MXFP8 trailing update + mixed-precision iterative refinement** —
   ◐ **PARTIALLY SHIPPED (S10/S14).** Native E4M3 FP8 panel products with FP32
   accumulation in the left-looking `1×32768` path passed all six families and
   improved the exp-009 path by **1.373×**. Experiment 014 fused both operands'
   tiled amax reductions and E4M3 scale/cast passes, improving that shipped path
   another **1.084×** and producing ranked `#880770`. The dense scaled residual
   remains 4.52/20 (22.6% of tolerance). MXFP8, iterative refinement, and an FP8
   path for `1×16384` remain genuinely untested.
3. **CUTLASS 3.x Blackwell fused kernel (`tcgen05.mma`, TMA, 2-SM MMA)** — a
   warp-specialized collective kernel that fuses panel + trailing SYRK, using the
   Tensor Memory Accelerator for async bulk copies and CTA-pair (2-SM) MMA. Beats
   the PyTorch-op blocked path by avoiding per-step launch + global-memory
   materialization. See CUTLASS example 78 (`blackwell_emulated_bf16x9_gemm`).
   High effort. *Target: large-n; possibly a real n=64/128 blocked kernel.*
4. **CUDA Graphs (legal — not streams)** — capture the many small launches in the
   blocked path and the per-matrix loop into a graph to amortize launch overhead.
   Cheap, shippable, Blackwell-agnostic but complementary. *Target: launch-bound
   small-n shapes, the Python-loop blocked path, 1×32768 (2-level).*
5. **Latest cuSOLVER + expert `cusolverDnXpotrf[Batched]` API** — ensure the
   toolkit ships Blackwell-tuned `potrf`; try algorithm selection / the 64-bit
   expert API vs `cholesky_ex`. Low effort, may quietly improve mid-batch shapes.
6. **Two-level blocked scheme for 1×32768** — recurse the diagonal `potrf` so the
   FP32 diagonal factorization also becomes tensor-core work. Diminishing returns
   (S6) but 32768 is ~76% of the clock, so even a few % moves the geomean most.
7. **Thread-block clusters / distributed shared memory (sm_90+/sm_100)** — a
   cluster-wide-shared-memory panel kernel could finally crack the mid-n shapes
   (n=256–1024) currently stuck on saturated cuSOLVER. Speculative.

---

## 2026-07-17 — Session 15: mid-shape batched tensor-core factorization → ranked #881981 (NEW BEST 1262.934μs)

### Goal and result

Leader-gap analysis showed the equal-weight geomean was dominated by nine mid
shapes on stock cuSOLVER sitting 19–260× above B200 hardware floors — not by
1×32768. Experiment 015 attacked that region. **ADOPTED.** Ranked `#881981`:
**1262.9337990784535μs public** / **1270.7067480724075μs secret**, improving
`#880770` by **12.74% / 11.95%**; rank 12 → 11. Popcorn test `#881978` 17/17.
Exact ranked SHA-256
`0400f06ad250b3bef240eb5cebae10ca9c045cc227924e06185be966e7310bb5`.

### What shipped

1. **Two-level blocked tensor-core factorization** (new Triton kernels, all
   launches replayed as a per-shape CUDA graph with one shared memory pool)
   for 64×256, 16×512, 640×512, 4×1024, 60×1024, 8×2048: 1-warp rank-2
   diagonal potrf with the triangular inverse built in the same 16-step
   serial loop; panels as `tl.dot(P, Dinv^T)` tf32x3; narrow in-panel rank-32
   updates; one rank-128 trailing Schur update per outer panel (tf32 at
   n≥1024, tf32x3 at n≤512 — 4× less trailing RMW traffic than rank-32).
2. **Graph-replayed exact cuSOLVER path for 1024×64** (1.086×).
3. **256×128 moved from `make_graphed_callables` to manual static-buffer
   capture** (same vendor kernel; multi-capture-safe).

Paired same-process wins vs exact `#880770`: 64×256 1.350×, 16×512 1.171×,
640×512 **1.714×**, 4×1024 1.421×, 60×1024 **1.989×**, 8×2048 1.590×,
1024×64 1.086×. Full-grid paired aggregate **1.1859×** (1522.3→1283.6μs);
untouched shapes 0.999–1.001×. Single-module gates: verify **57/57**,
benchmark 15/15 (geomean 1325.7μs). Zero fallbacks/errors on timed runs;
worst residual 7.28/20 (640×512 tf32); lowrank takes exactly one expected
safety fallback per tf32 shape.

### Bounded ladder (measured, rounds r1–r6 + candidate B fork)

| variant | result |
|---|---|
| fused one-CTA whole-matrix potrf (r1) | rejected: 0.60–1.07×, over-tolerance at n=256 |
| split kernels, IEEE panels, rank-32 trailing, graphs (r2/r3) | rejected: micro 24μs/launch serial floor, trailing RMW ×(n/32) |
| two-level NB=128 + GJ-fused micro (r4) | frontier: 1.06–1.87× |
| rank-2 micro (r5) | frontier: 1.13–1.94× |
| ILP inverse + tiled clear (r6, TILE=128) | **adopted core** |
| TILE=256 trailing (r6) | rejected: register/smem budget |
| 2×2048 on new path | rejected 0.651× — stays on per-matrix loop |
| cand-B superpanels 1×4096/2×4096 (3 variants) | rejected 0.18–0.97× |
| cand-B fused Triton 1024×64 | rejected 0.673× |

### Defects found

- **Graph-entry use-after-free**: the `dinv` workspace wasn't kept alive with
  its captured graph — replays wrote freed memory (async illegal access that
  vanished under `CUDA_LAUNCH_BLOCKING=1`). Keep every captured buffer in the
  graph cache entry.
- **Dual-module probe artifact**: `make_graphed_callables` replays corrupt
  (residual exactly 1.42) once another module captured a manual graph earlier
  in the same process. Popcorn-like single-module runs are clean; the paired
  harness result for 256×128 was a false alarm, confirmed by
  `modal_verify.py` verify 57/57 + benchmark before ranking.

### Cost and next

~14 Modal B200 runs ≈ $6–9 (incl. 4 in the candidate-B fork); Popcorn one
test + one ranked. The board moved sharply today: leaders now 492–506μs.
Next levers: micro kernel is still ~16μs/launch (serial floor n×~500ns) —
rank-4 pivots or manual graph-node dependencies to overlap diag with
trailing; persistent per-shape kernels; FP8/BF16 trailing at mid n; 2×2048,
1×4096, 2×4096 remain open.

Artifacts: `experiments/015-mid-shape-tensorcore/` (goal, baseline, all six
candidate rounds, probe harness, paired/full-grid/verify/benchmark JSON,
ranked artifacts, notes, exact ranked `submission.py`).

---

## 2026-07-17 — Session 14: fused E4M3 quantization → ranked #880770 (NEW BEST 1447.259μs)

### Goal and result

Target the slowest ranked shape, `batch=1,n=32768`, from exact winner `#878893`
(`141d015`, public `1459.321342997556μs`, secret
`1448.3768036226527μs`). The default goal was `2.00×`; every correct positive
partial frontier was preserved and only an aggregate full-grid improvement was
eligible for ranking.

**ADOPTED.** Ranked `#880770` completed both public and secret pipelines at
**1447.2589334363144μs public** and **1443.2264907145392μs secret**. This is
`12.062410μs` / `0.8266%` better publicly and `5.150313μs` / `0.3556%` better
secretly than `#878893`. Exact ranked source SHA-256:
`78b2282d436243393897e61a5e4b8206d52c3950ec6f4495cbc71da895abd1fc`.
The complete experiment was published in commit
`48fa14a84632ee4170df6c913ad7f9e8502c2e96`; `origin/main` was independently
verified at that exact commit before this publication-state follow-up.

### Bounded architecture ladder

All target measurements were paired, same-process Modal B200 runs against the
exact source-locked exp-012 winner, with retained outputs and backend counters.

| Architecture | Baseline → candidate mean | Speedup | Verdict |
|---|---:|---:|---|
| Triton 512-microblock active superpanel, custom diagonal | 52042.6→86184.1μs | 0.604× | rejected |
| Custom CUDA128 POTRF superpanel | 51689.3→315719.2μs | 0.164× | rejected |
| CUDA128 padded-shared defect repair | 56961.4→117314.1μs | 0.486× | rejected |
| CUDA128 warp-synchronous defect repair | 52031.2→182105.3μs | 0.286× | rejected |
| Joint fused E4M3 scale/cast | 51915.0→51430.1μs | 1.009× | frontier |
| **Tiled dual amax + joint fused E4M3 scale/cast** | **51939.3→47896.9μs** | **1.084×** | **frontier/adopted** |
| Fixed-scale FP8 shadow factor | 52499.5→51293.7μs | 1.024× | superseded frontier |

The cuSOLVER-free candidates were correct (`4.53/20`) but much slower, agreeing
with independent experiment 013's diagonal-POTRF negative evidence. The adopted
frontier retains the ranked diagonal cuSOLVER calls and changes only the dynamic
quantization front end. It preserves exact scalar E4M3 decode scaling and FP32
`_scaled_mm` accumulation; both rotating dense inputs remain `4.52/20`, with 96
fused-amax hits, 96 fused-quantization hits, and zero fallback/error.

### Component evidence and numerical gates

The adopted operator profile measured `48004.3μs`. Dominant self-device costs:
triangular inverse/solve `12074.3μs`, diagonal Cholesky `11162.5μs`, TF32
diagonal update `8064.0μs`, `addmm_` `4829.8μs`, copies `4626.8μs`, panel `mm`
`4373.7μs`, FP8 `_scaled_mm` `2523.6μs`, and fused tiled amax `2449.4μs`.
The FP8 GEMM itself is no longer the primary bottleneck.

Changed-family Modal gate: **6/6**. Dense/diagonal/tridiagonal used the fast path
with zero fallback. Spectrum/low-rank/row-scaled proved the intended fused path
was attempted, then each took exactly one ranked safety fallback in both modules.
Candidate scaled residuals were respectively `4.52`, `0.000537`, `0.000507`,
`0.000042`, `0.000061`, and `0.00408`, all against the unchanged factor-20 gate.

### Full grid and harness defects

Final retained-output Modal grid: **15/15 pass**, exact baseline geomean
`1574.149644μs`, candidate `1565.545754μs`, aggregate **1.005496×**. Maximum
off-target ratio was `1.016644×` at `1×16384`, below the `1.03×` guardrail.
The target arithmetic mean was `51874.714→51198.891μs`; candidate best was
`47778.370μs`. Its mean includes one `68162.819μs` allocation outlier; the other
five samples were `47778–47824μs`, consistent with the dedicated 1.084× result.
No timed sample was removed.

Two harness defects were found and preserved rather than hidden: a single-line
full-grid JSON exceeded Modal's stdout limit, and the final warmup output
reference caused a one-time allocator expansion in reversed order. The final
harness uses exact-length 8-KiB result chunks and releases only the warmup
reference before timing; timed outputs remain retained through validation. An
initial family contract also incorrectly rejected expected safety fallbacks; the
fixed contract requires an exact fallback delta of one for those three families.

### Popcorn, cost, and next directions

Popcorn test `#880765`: **17/17**. After auditing the exact source, raw target,
family, full-grid, and test artifacts and confirming no ranked job was active,
exactly one leaderboard submission (`#880770`) was launched. All public/secret
test, benchmark, and leaderboard stages succeeded.

Approximately a dozen Modal B200 jobs covered the architecture ladder,
component profile, family gate, and full-grid repairs; exact invoice cost was not
available. Popcorn usage was one test and one ranked entry. Local CPU verification
was unavailable because the workstation Python has no `torch`; syntax, JSON,
source-policy, whitespace, hash, and stronger B200 checks all passed.

Next directions: fuse FP8 GEMM subtraction through cuBLASLt/CUTLASS, remove
product/copy traffic, or use a native Blackwell collective. Do not repeat custom
Python/Triton POTRF, smaller block sizes, or fixed-scale shadow copies; sessions
13–14 close those paths with direct negative evidence.

Artifacts: `experiments/014-fused-e4m3-quantization/` (exact baseline,
candidates, component/paired/family/full-grid JSON, Popcorn test/ranked JSON,
harness, notes, and exact ranked `submission.py`).

---

## 2026-07-16 — Session 13: 1×32768 cuSOLVER-free Triton potrf → REJECTED

### Result

**REJECTED.** Experiment 013 attempted to beat the exp-012 `1×32768` path
(`#878893`) while removing every cuSOLVER call from the fast path (Triton /
cuBLAS two-level blocked diagonal `potrf`, optional FP8 panel solve). No variant
cleared gate 1 on the cheap B200 proxies; **no ranked submission**, root
`submission.py` unchanged.

### Causal evidence

Diagonal `potrf` micro-benchmark (single 4096×4096 block — the only cuSOLVER
term on the shipped path):

| method | mean | vs cuSOLVER |
|---|---:|---:|
| **cuSOLVER `cholesky_ex`** | **1579 µs** | 1.00× |
| Triton blocked bk=64 (fastest free) | 5794 µs | **3.67× slower** |
| cuBLAS blocked bk=32 FP32 (accurate) | 13261 µs | **8.4× slower** |

Full left-looking path, candidate vs exp-012 (paired same-process):

| config | n | speedup | correctness |
|---|---:|---:|---|
| fast (Triton diag, FP8 panel) | 8192 / 16384 | **0.41× / 0.50×** | FAIL (NaN) |
| accurate (cuBLAS32 FP32 diag) | 8192 / 16384 | **0.22× / 0.30×** | PASS |

Backend counter: `nocusolver_32768_hits=48`, `fallbacks=0` — a genuine
cuSOLVER-free measurement, not a fallback.

### Verdict and cost

cuSOLVER's fused diagonal `potrf` (~1.6 ms, ~24% of the 52 ms path) is not
replaceable with PyTorch/Triton orchestration without a large net regression.
Even a perfect cuSOLVER-free `potrf` would leave ≈0% headroom; the only FP8 lever
is worth ~2–5 ms and costs accuracy. A single-launch CUTLASS/`tcgen05` `potrf`
is the only credible alternative (high effort, poor ROI). Modal spend ≈ **$2–3**
(three `nocusolverprobe` runs); ranked quota **0**.

Artifacts: `experiments/013-1x32768-no-cusolver/`, goal
`docs/goal-exp013-1x32768-no-cusolver.md`, harness `nocusolverprobe` mode in
`scripts/`.

---

## 2026-07-16 — Session 10: large left-looking frontiers → ranked #878893 (NEW BEST 1459.321μs)

### Result

**ADOPTED.** Ranked `#878893` passed public and secret validation and scored
**1459.321342997556μs public** / **1448.3768036226527μs secret**, improving
`#878273` by **2.7575% public** / **3.5342% secret**. Popcorn test `#878891`
passed 17/17.

### Integrated paths and causal evidence

The bounded searches for the three slowest shapes produced positive frontiers
at 16384 and 32768; the 8192 search produced no valid improvement and was not
integrated. Both winners were rebased onto the exact exp-009 ranked source.

| shape | path | exp 009 | exp 012 | speedup |
|---|---|---:|---:|---:|
| 1×16384 | left-looking TF32 active diagonal/panel updates | 18495.512μs | 16082.949μs | **1.150×** |
| 1×32768 | left-looking native FP8 panel products, FP32 accumulation | 71567.591μs | 52139.092μs | **1.373×** |

The paired same-process Modal probe rotated two inputs per shape, retained all
outputs, and verified native dispatch counters after timing. It recorded zero
fallbacks and no FP8 runtime error.

### Validation and workflow

- Local property checks: **10/10**, with clean compilation, whitespace,
  source-policy, snapshot, and artifact checks.
- Changed-region B200 family sweep: **12/12** across dense, spectrum, low-rank,
  row-scaled, diagonal, and tridiagonal inputs at both large sizes.
- Full 15-shape Modal benchmark: all passed; geomean
  **1652.198636→1574.881992μs** with the other 13 dispatch regions unchanged.
- Popcorn test `#878891`: **17/17**.
- Exactly one ranked run, `#878893`, was launched and monitored to success.
- No evo workflow was used. Modal uploads were made under the owner's explicit
  authorization recorded in the root README and were limited to benchmark
  source/harness files—never credentials or unrelated workspace content.

### Artifacts

See `experiments/012-large-left-looking-frontiers/` for the exp-009 baseline,
exact ranked source, paired timings, 12-family verification, full-grid result,
ranked summary, and notes. The paired/full-grid harness additions remain in
`scripts/` for reproducibility.

---

## 2026-07-15 — Session 9: combined shape frontiers → ranked #878273 (NEW BEST 1500.704μs)

### Result

**ADOPTED.** Ranked `#878273` passed public and secret validation and scored
**1500.7037765896727μs public** / **1501.4402012082579μs secret**, improving
`#878108` by **2.736% public** / **2.827% secret**. Popcorn test `#878272`
passed 17/17.

### Integrated paths and paired evidence

Three non-overlapping, shape-specific frontiers were recovered from the bounded
search tasks and integrated into exp 008:

| shape | path | exp 008 | final candidate | speedup |
|---|---|---:|---:|---:|
| 256×128 | captured vendor batched factorization + owned output | 197.311μs | 162.913μs | **1.211×** |
| 16×512 | static-buffer vendor graph + owned output | 772.362μs | 603.532μs | **1.280×** |
| 8×2048 | Triton FP32 blocks + lower TF32 Schur tiles | 5723.026μs | 3527.579μs | **1.622×** |

The paired harness mirrors Popcorn's allocation pattern: it rotates across up
to 15 inputs, retains every output, and checks each input/output pair after all
calls. All target results passed. The `8×2048` path checks its factor diagonal
and falls back to exact cuSOLVER when TF32 produces a non-finite pivot.

### Validation

- Local property checks: **10/10**, plus clean syntax, diff, snapshot, JSON, and
  forbidden queue-source checks.
- Changed-region B200 family sweep: **25/25** across dense, spectrum, diagonal,
  low-rank, row-scaled, and tridiagonal inputs.
- Full 15-shape Modal geomean: **1738.121→1652.199μs**; no material off-target
  regression.
- Popcorn test `#878272`: **17/17**.

### Ranked validation failure and correction

The first ranked attempt `#878263` failed its benchmark validation. The graph
paths returned reusable static result storage. Popcorn builds a list of outputs
for rotated inputs and verifies it afterward, so later replays overwrote earlier
entries. Returning an owned clone from both graph paths fixed the contract. The
paired harness was strengthened to reproduce retained-output validation, and all
promotion gates were repeated before the successful ranked retry `#878273`.

### Artifacts

See `experiments/009-combined-shape-frontiers/` for the exp-008 baseline, exact
ranked source, paired artifacts, family results, full-grid results, and ranked
summary. The README records explicit owner authorization to upload the bounded
verification files to Modal.

---

## 2026-07-15 — Session 8: fused in-place TF32 Schur update → ranked #878108 (NEW BEST 1542.914μs)

### Result

**ADOPTED.** Ranked `#878108` passed 17/17 and scored exactly
**1542.9137409531085μs public** / **1545.1284990962687μs secret**, improving the
previous `#878015` (~1559μs). Popcorn test `#878107` passed 17/17. One ranked
submission was used; no retry or duplicate was launched.

### Change and paired B200 evidence

Experiment 006 performed `A22 -= L21 @ L21.T`, creating a full temporary product
and launching a separate subtraction. Experiment 008 replaces only that operation
with `A22.addmm_(L21, L21.T, beta=1, alpha=-1)`, writing directly to the strided
trailing view while keeping TF32 inputs and FP32 accumulation.

| shape | exp006 separate | exp008 fused | speedup | tolerance used |
|---|---:|---:|---:|---:|
| 1×16384 | 18924.8μs | **17411.5μs** | **1.087×** | identical 0.004796 (208.5× margin) |
| 1×32768 | 73700.7μs | **68246.1μs** | **1.080×** | identical 0.002397 (417.1× margin) |

The paired same-process probe isolates the fused update and avoids cuSOLVER
run-to-run drift. The full 15-shape Modal run passed at **1738.1μs**; its changed
shapes were 18531.4μs and 73463.5μs versus experiment-006 Modal 19981.6μs and
78357.1μs. Untouched dispatch regions are source-identical.

### Correctness and fallback

- Local checker: **10/10**.
- Modal changed-size matrix: **12/12** — dense, spectrum, lowrank, rowscale,
  diagonal, and tridiagonal at both 16384 and 32768.
- Spectrum/lowrank retain the experiment-006 `isfinite` safety fallback to exact
  FP32 cuSOLVER; stable families use the fast fused path.
- Popcorn test `#878107`: **17/17**. Ranked `#878108`: **17/17**.

### Ladder decision and cost

Stage A alone achieved the ranked objective, so the lower-triangular custom
kernel, hierarchical blocking, and bounded batched pivot were not invoked. This
kept the winning change minimal and avoided repeating rejected approaches. Four
targeted/full Modal jobs consumed roughly 2–3 minutes of B200 wall time (well
under the experiment guardrail; approximately <$0.5–1 depending on billing).

Artifacts: `experiments/008-fused-triangular-schur/` contains the exact submitted
source, paired probes, all-family verify JSON, full-grid benchmark, ranked summary,
and notes. Root `submission.py` is the adopted implementation with updated metadata.

### End-to-end execution trail

1. Fast-forwarded local `main` from `566a4e9` to experiment-007 commit `1efca55`.
2. Reviewed experiments 004–007 and wrote
   `docs/goal-exp008-fused-triangular-schur.md` with a four-stage optimization
   ladder, correctness gates, one-submission limit, and `<1559μs` ranked target.
3. Ran the work in a dedicated isolated task/worktree while the supervising task
   monitored checkpoints and redirected it past an unavailable evo workspace and
   one transient task failure.
4. Isolated Stage A (`addmm_`) and passed the free local property check 10/10.
5. Established causality with paired B200 probes: 1.087× at 16384 and 1.080× at
   32768, with identical reconstruction residuals.
6. Expanded to all six input families at both changed sizes (12/12), preserving
   the difficult-input cuSOLVER fallback.
7. Ran the full 15-shape Modal grid; all shapes passed and only the intended two
   dispatch shapes changed.
8. Popcorn test `#878107` passed 17/17. Exactly one ranked run, `#878108`, was
   launched, monitored to completion, and confirmed at 1542.9137409531085μs.
9. Adopted the exact submitted source to root, saved raw artifacts and notes,
   updated both READMEs, the Optimization Tracker, and this dated entry.
10. Final checks passed: local 10/10, Python compilation, JSON parsing, and
    whitespace validation. The complete experiment was committed on `main` as
    `5604cc0` (`exp 008: fused TF32 Schur update — ranked 1542.914us`).
11. `main`, including the experiment commit and this workflow completion update,
    was pushed to `origin/main`; remote containment of `5604cc0` is the terminal
    completion check.

### Session insights

- **Fuse memory traffic before writing a custom kernel.** Replacing a materialized
  product plus subtraction with `addmm_` captured a robust 8–9% win on the two
  dominant shapes without changing the numerical algorithm. This cheap library
  formulation should precede a costly triangular CUDA/CUTLASS implementation.
- **Paired measurements are the promotion signal.** The full-grid geomean drifted
  because untouched cuSOLVER shapes vary between sessions; same-process A/B probes
  isolated the causal speedup and accurately predicted the ranked result.
- **Identical residuals sharply reduce promotion risk.** The fused call preserved
  TF32 inputs and FP32 accumulation, so the checker margins and fallback behavior
  were unchanged rather than merely still under tolerance.
- **Stop the ladder after a confirmed win.** Stage A met the ranked objective, so
  lower-triangular tensor-core code, hierarchical blocking, and the batched pivot
  remained available future ideas instead of adding risk to a proven candidate.
- **Supervision is part of correctness.** The parent task prevented a duplicate
  leaderboard submission after a worker failure, independently confirmed the
  existing job, and required adoption, documentation, commit, and push before
  declaring the experiment complete.
- **Publication is a hard gate.** A leaderboard improvement that exists only in a
  detached worktree or local `main` is not reproducible team state. Future sessions
  must finish with a verified GitHub push, including rejected experiments because
  their negative evidence prevents repeated GPU spend.

---

## 2026-07-15 — Session 7: BF16x9 FP32-emu trailing update (large-n) → REJECTED (slower than TF32)

### Result — REJECTED, nothing submitted

exp 007. BF16x9 FP32 emulation **engages** on the Modal B200 and is **≈FP32-accurate
and more robust than TF32**, but it is **decisively slower than the shipped paths**,
so no shape was adopted and no ranked slot was spent. Current best unchanged:
`#878015` (~1559μs). This closes the *BF16x9 FP32-emu* column for the large-n shapes.

### How to engage it (the API answer)

Set, **before `import torch`** (cuBLAS reads env at handle creation):

```
CUBLAS_EMULATE_SINGLE_PRECISION=1     # the MASTER switch that actually engages it
CUBLAS_FP32_EMULATED_BF16X9_MATH=1    # pins the algorithm to BF16x9
```

- `CUBLAS_FP32_EMULATED_BF16X9_MATH=1` **alone did NOT engage** — measured identical
  to native (standalone 8192 FP32 matmul 16712μs off → 16729μs with just this var).
  Adding `CUBLAS_EMULATE_SINGLE_PRECISION=1` dropped it to **6333μs (2.64×)**.
- `torch.backends.cuda.matmul.fp32_precision` exposes only `ieee`/`tf32` (no BF16x9),
  so the PyTorch knob can't reach it. `preferred_blas_library("cublaslt")` is *harmful*
  (forces the slower cuBLASLt path); the default heuristic already picks the fast
  emulated GEMM. Shipped-style config = the two env vars only, BLAS left on default.
- No non-default queues involved; nothing for popcorn's source scan to flag.

### The numbers (Modal B200 precprobe; margin = tol/residual)

| shape | current ship | best BF16x9 | verdict |
|---|---|---|---|
| 1×8192 | cuSOLVER 6410μs | 6733μs nb4096 (**0.95×**) | slower → keep cuSOLVER |
| 1×16384 | TF32 21,533μs (1.60×) | 29,990μs nb4096 (1.15× vs cuSOLVER) | slower than TF32 → keep TF32 |
| 1×32768 | TF32 ~77,200μs (2.86×) | not measured (would lose) | keep TF32 |

Accuracy: every BF16x9 variant PASSES all families (dense/spectrum/lowrank) at
8192/16384 with margins **65,000–139,000×** inside tolerance (vs TF32's ~100–210×,
which additionally **NaNs on lowrank**). The manual-split genuine-accuracy proxy
tracks the emulated residual, so the global-emulation "checker reconstructs with a
matmul" pitfall did not flatter the result.

### Why it loses

BF16x9 emulates one FP32 product with ~6–9 BF16 tensor-core products; TF32 uses one.
BF16 is only ~2× TF32 throughput on B200, so BF16x9 ≈ 3× slower than TF32 per
FP32-equivalent GEMM. Ordering: **TF32 > BF16x9 > native FP32** in speed. NVIDIA's
"3–4× faster than native FP32" is true but native FP32 isn't the bar — the shipped
TF32 tensor-core path already is, and BF16x9 can't beat it. At 8192 the FP32 diagonal
`potrf` + FP32 panel TRSM are also a large fixed cost the GEMM emulation can't touch.

### Infra kept

Added an `emuprobe` mode + `--emu` flag (finds the engaging config), `blocked_fp32`
(native/emulated) and `bf16x9split` (manual 3-way split) trailing modes, and
ill-conditioned family specs to `precprobe`. Root `submission.py` unchanged.

### Quota / cost

Ranked used: **0**. Modal spend ≈ **$3–5** (5 B200 runs; the bulk was one `emuprobe`
that hung after GPU init ~9.7 min and was killed — same transient noted in S6 — then
re-ran clean in ~40s). popcorn quota untouched.

### Next steps

- The remaining large-n lever is **FP8/MXFP8 + iterative refinement** (tracker
  candidate #2): FP8 is ~2× BF16 throughput, so a lossy FP8 trailing update + 1–2 SPD
  IR steps could beat TF32 on 32768 where the n-scaled gate has the most headroom.
  Higher effort/risk. BF16x9's accuracy edge is real but not monetizable on the
  dense ranked grid.

---

## 2026-07-15 — Session 6: large-n TF32 tensor-core blocked Cholesky → ranked #878015 (NEW BEST ~1559μs)

### Result

**New best `#878015`, ranked geomean ≈ 1559μs** — beats prior best `#877956`
(~1744μs) by **~10.6%**, and the board leader (~1924μs) by ~19%. 17/17 tests pass.
Experiment `006`, **adopted** to root `submission.py`. This closes the large
single-matrix shapes that every prior session had dismissed as "cuSOLVER near
speed-of-light, leave it" — they were never tested against tensor-core math.

### What & why

A **right-looking BLOCKED Cholesky** for `batch==1, n>=16384`: the diagonal-block
`potrf` and the panel triangular solve stay FP32 (stability), but the O(n³)
trailing Schur update `A22 -= L21·L21ᵀ` — the bulk of the FLOPs — runs on B200
tensor cores in **TF32** (FP32 accumulate). The checker gates only
`‖A−LLᵀ‖₁ ≤ 20·n·eps·‖A‖₁`, whose tolerance grows with n, so the huge shapes have
the most numerical headroom (measured residual margins 100–417× inside tolerance).

Characterization probe (Modal B200, speedup vs batched cuSOLVER):

| n     | TF32 best (nb) | speedup | FP16 best | BF16 |
|-------|----------------|---------|-----------|------|
| 8192  | 1.07× (nb2048) | marginal → **excluded** | 0.94× | 12× margin |
| 16384 | **1.80× (nb2048)** | shipped | 1.48× | 22× margin |
| 32768 | **2.94× (nb4096)** | shipped | 2.22× | rejected |

**TF32 beat FP16 everywhere** (FP16's operand-cast + fp16-output truncation
overhead outweighs its precision headroom on B200; B200 TF32 tensor cores are
already several× FP32 throughput). **Bigger nb wins as n grows** (fewer Python
steps, larger trailing GEMMs; the FP32 diagonal potrf stays a small fraction).
Dispatch: `nb = 4096 if n>=32768 else 2048`.

### Numerical safety net

TF32 error can drive a late diagonal block indefinite on ill-conditioned inputs
(`spectrum` cond=5, `lowrank` cond=4) → NaN/Inf. A post-factorization
`torch.isfinite(L).all()` check falls back to exact FP32 cuSOLVER. The ranked
shapes are well-conditioned dense (residual ~10% of tolerance — never trips it);
the fallback only fires on pathological families, so correctness holds across
every family at <1ms cost (memory-bound vs the ~75ms factorization).

### Ranked per-shape (`#877956` → `#878015`)

- **1×32768: 221000 → 77200μs (2.86×)** — the big one (76% of the clock).
- **1×16384: 34200 → 19400μs (1.76×)**.
- 1×8192: 6400 → 6390μs (unchanged, cuSOLVER).
- All other 12 shapes at/under `#877956` (low-drift run, no regressions): n=32
  61.8μs, 640×512 3.78ms, 60×1024 2.89ms, 8×2048 5.05ms, 2×4096 3.20ms.

Modal↔popcorn fidelity on the changed shapes was excellent (~2%): Modal measured
16384=19982μs / 32768=78357μs vs ranked 19400 / 77200. The huge single matrices
are compute-bound with no batched-dispatch drift, unlike the mid-batch shapes.

### Correctness

Modal verify **37/37** across all families (28/28 small/mid + 9/9 large-n incl.
16384 spectrum/lowrank/rowscale/diagonal/tridiagonal, 32768 dense/lowrank/
tridiagonal). popcorn test 17/17 (its grid maxes at n=2048, so it validates the
unchanged/fallback paths; the blocked dense large-n is validated on Modal).

### Gotcha

popcorn's static source scan flagged the literal word **"stream"** in my
docstrings (HTTP 500 "work on another stream"), exactly as the exp-004 note warned.
Removed all occurrences; re-test passed 17/17. The kernel uses no non-default
CUDA queues — pure default-queue matmuls.

### Quota / cost

Ranked used: **1 this session** (`#878015`; the "3 of 3" note was a self-imposed
per-run budget, not a platform cap — confirmed with supervisor). Modal spend this
session ≈ **~$2.5–3**; ~$2 of that was a single `verify` run that hung after GPU
init and burned the 1200s sandbox timeout (transient — the same grid re-ran in
~45s after adding `--shapes`/progress filtering to `run_verify`). popcorn
test+leaderboard run on GPU MODE infra (not billed to our Modal).

### Next steps

- Every shape is now at/near its frontier. `8192` (1.07×) and the mid-batch shapes
  (`640×512` saturated, exp-005; `8×2048` streams-banned) are the only remaining
  levers and are all poor bets. A two-level blocked scheme (recurse the diagonal
  potrf) could shave a little more off 32768 but with diminishing returns.

---

## 2026-07-15 — Session 5: `640×512` probe (REJECTED) + `8×2048` own-goal fix → ranked #877956

### Result

Two things this session, driven by "which shape is the biggest bottleneck to
progress" analysis:

1. **`640×512` (biggest attackable shape, unexplored): REJECTED.** A characterization
   probe on B200 (batched vs loop vs streamed vs chunk64/128) proved cuSOLVER's
   batched `potrf` **already saturates** the GPU for hundreds of medium matrices —
   `streamed` is **6.5× slower** than batched (25730 vs 3955μs), the exact opposite
   of the exp-004 few-large headroom signal. Chunked batched (the shippable idea) is
   1.8–2.7× slower. No non-stream approach can win; a custom kernel would have to beat
   a saturated vendor routine (exp-003 already showed naive kernels lose at n≤128).
   Shape closed, like exp-003. See `experiments/005-highbatch-mid-n/notes.md`.

2. **`8×2048` own-goal fix: SHIPPED, ranked `#877956`.** The exp-004 loop region
   `2<=batch<=8, n>=1024` had regressed `8×2048` on popcorn (5010 batched → 5370
   loop). Trimmed the region to `2<=batch<=4` so `8×2048` returns to batched cuSOLVER.
   Shipped on the last ranked slot: **`8×2048` 5370 → 5060μs (−5.8%)**, all other
   shapes unchanged (low-drift run). Ranked geomean ≈ **1744μs** (from ~1746). Small,
   clean, no regressions. 17/17 on B200 (`verify_local` 10/10, popcorn test `#877955`).

### Bottleneck analysis (the framing)

Ranking is a geometric mean, so the bottleneck to progress = the shape with the
biggest *achievable* speedup ratio, not the slowest in μs. Huge single matrices
(8192/16384/32768) dominate the clock but are compute-bound (no headroom). Small-`n`
overhead shapes + `60×1024` were already proven cuSOLVER-optimal (exp-003, exp-004
probe). `640×512` was the largest unexplored lever (~6% geomean at 2.5×) — probed
and found saturated. `8×2048` had proven headroom via streams but streams are banned;
the only realizable gain was reverting it to batched (the fix shipped here).

### Correctness

The `8×2048` fix is correct by construction (routes to the already-validated batched
path). popcorn test + leaderboard both 17/17 across all families.

### Quota / cost

**Ranked quota now fully used: 3 of 3** (`#877091`, `#877941`, `#877956`). Modal spend
this session ≈ **~$0.2–0.4** (one probe run; the `8×2048` fix needed no Modal benchmark
— see notes on the Modal↔popcorn gap). popcorn test+leaderboard run on GPU MODE infra.

### Next steps

- Ranked quota exhausted for this run. Every shape is at/near its frontier; leader
  beaten by ~9–10%. Only speculative lever left is a blocked tensor-core mid-`n`
  kernel, but the `640×512` saturation evidence makes it a poor bet.

---

## 2026-07-15 — Session 5: `640×512` probe → REJECTED (cuSOLVER-saturated)

### Goal

Attack the biggest attackable board shape, **`640×512`** (~3800μs ranked), to push
the geomean below the current best `#877941` (~1746μs) — or prove it's already
cuSOLVER-saturated and cleanly reject. Secondary: the `8×2048` own-goal (5010→5370).

### Result — REJECTED, nothing submitted

The characterization probe closes the shape. `640×512` **cuSOLVER batched is optimal**
— every decomposition is dramatically slower:

| shape | batched | loop | streamed | chunk64 | chunk128 | best |
|---|---|---|---|---|---|---|
| **640×512** | **3954.9** | 104887.1 | 25729.9 | 10494.9 | 7007.6 | **batched** |
| 2×2048 (control) | 4627.3 | **1384.2** | 1467.1 | 4674.2 | 4669.7 | loop |
| 8×2048 (control) | 5738.1 | 5427.9 | **3478.1** | 5890.7 | 5883.7 | streamed |

`streamed` (max concurrency) is **6.5× SLOWER** than batched for `640×512` — the exact
opposite of the exp-004 headroom signal. There is **no under-occupancy to capture**:
cuSOLVER's batched `potrf` already saturates the B200 for hundreds of medium matrices.
`chunk64/128` (the shippable default-stream alternative) are 1.8×–2.7× slower, ruling
out chunked batched calls **with data**. Controls reproduce exp-004 (loop/streamed win
for few-large), confirming the harness. CUDA graph capture is pointless (single batched
launch, nothing to amortize); a custom blocked kernel would have to beat a *saturated*
vendor routine — not worth it after exp-003 showed naive kernels lose at n≤128.

### `8×2048` own-goal + ranked-slot decision (flagged to supervisor)

Fix is trivial (trim loop region `2<=batch<=8` → `2<=batch<=4`, sending 8×2048 back to
batched: 5010 < 5370 on popcorn). Prepared as `experiments/005-.../submission.py`,
correct by construction, but **NOT submitted**. Since the prize is dead, the slot would
only buy a ~0.05% cleanup, against ~±20% cuSOLVER run-to-run drift risk on other shapes.
**Recommendation: keep the last ranked slot; don't burn it on 0.05%.** Root
`submission.py` stays exactly `#877941`.

### Quota / cost

Ranked used: **2 of 3** (unchanged; nothing submitted). Modal spend ≈ **~$0.2–0.4**
(1 probe run, image cached). `verify_local.py` 10/10 (repo intact).

### Insight

Two distinct high-batch failure modes now mapped: **few-large** (batch≤4, n≥1024) →
batched cuSOLVER under-occupies → loop wins (exp-004); **many-medium** (batch=640,
n=512) → batched cuSOLVER *saturates* → batched wins (exp-005). The dividing line is
total work/occupancy, not just "batched is bad." `640×512` is at its frontier.

---

## 2026-07-15 — Session 4: small-batch/large-n loop → ranked #877941, BEATS THE LEADER

### Result

**New best `#877941`, ranked geomean ≈ 1746μs** — beats prior best `#877091`
(~~2062μs) by ~15% **and the board leader (~~1924μs) by ~9%.** 17/17 tests pass.
Experiment `004`, **adopted** to root `submission.py`.

### What & why

`torch.linalg.cholesky_ex` sends batch≥2 to `cusolverDnSpotrfBatched` (tuned for
many-small matrices) — terrible for few-but-large. Confirmed on B200 with a 3-way
probe (batched vs per-matrix loop vs streamed):


| shape   | batched  | loop     | streamed |
| ------- | -------- | -------- | -------- |
| 2×4096  | 12580    | **3222** | 3391     |
| 2×2048  | 3900     | 1382     | **1132** |
| 8×2048  | 5612     | 5427     | **3477** |
| 4×1024  | 1646     | 1353     | **634**  |
| 60×1024 | **3233** | 19707    | 5782     |
| 1×4096  | **1546** | 1627     | 2447     |


Streamed was fastest but **popcorn forbids non-default streams** (static source
scan → HTTP 500 "work on another stream ... disqualification"; it even flagged the
literal word "stream" in a comment). So shipped the **loop**: dispatch
`2<=batch<=8 and n>=1024 → per-matrix loop`, keep Triton n=32 + batched cuSOLVER
elsewhere.

### Ranked per-shape (`#877091` → `#877941`)

- **2×4096: 13400 → 3200μs (4.19×)** — the big one.
- **2×2048: 3840 → 1357μs (2.83×)**.
- 4×1024: 1395 → 1297μs (1.08×).
- 8×2048: 5010 → 5370μs (**1.07× WORSE** — loop regresses here on popcorn even
though it tied/won on Modal; Modal↔popcorn fidelity gap on a marginal shape).
- others unchanged.

### Correctness

popcorn test 17/17; Modal verify 26/26 across all families (added in-region cases:
2×1024 spectrum/diagonal, 4×1024 rowscale/tridiagonal, 8×2048, 2×4096 dense/lowrank).
Loop calls the same `potrf` per slice → numerically identical to cuSOLVER.

### Quota / cost

Ranked used: **2 of 3** overall (session 2 `#877091` + this `#877941`). Test id
`#877940`. Modal spend this session ≈ **~$0.5–1**.

### Next steps

1. **Cheap fix for the 8×2048 regression:** restrict the region to `2<=batch<=4`
  (leave 8×2048 on batched, 5010 < 5370). Est. ~~1746→~~1738μs. Costs the last ranked
   submission to confirm; deferred (leader already beaten). Root keeps the exact
   `#877941` code (region `2<=batch<=8`) so it matches a confirmed ranked result.
2. Revisit whether a Triton/CUDA single-large-matrix kernel could shave 8192/16384/
  32768 (compute-bound, low ROI) — unlikely.

---

## 2026-07-15 — Session 3: CUDA n=64/128 attempt → REJECTED (cuSOLVER wins)

### Goal

Beat the board leader (~1924μs) via a **warp/block-per-matrix CUDA kernel for
n=64 and n=128** (experiment `003`), keeping Triton n=32 + cuSOLVER elsewhere.

### Infra unlocked (kept — enables all future CUDA experiments)

- Switched the Modal image to `**nvidia/cuda:13.0.0-devel-ubuntu24.04`** (has
`nvcc`) + `pip install torch numpy ninja` + `.entrypoint([])`. This lets
`torch.utils.cpp_extension.load_inline` compile CUDA on the B200 sandbox
(the plain pip-torch image has no nvcc). torch resolved to 2.13.0+cu130.
- **Gotcha:** without `ninja`, `load_inline` fails `verify_ninja_availability()`
and the try/except silently falls back to cuSOLVER. Caught it because the
n=64/128 residuals were byte-identical to cuSOLVER. Added a
`custom_cuda_loaded=<bool>` + `_CUDA_LOAD_ERROR` diagnostic to `_gpu_runner.py`
/ `submission.py` so a failed compile is never mistaken for a working kernel.

### Result — REJECTED, nothing submitted

CUDA kernel is **correct** (Modal verify 19/19, all families, `custom_cuda_loaded=True`,
residuals ~1000× inside tolerance) but **slower than cuSOLVER** at both shapes:


| shape   | cuSOLVER    | Triton | CUDA block (128-thr, `__syncthreads`) | CUDA warp (32-lane, `__syncwarp`) |
| ------- | ----------- | ------ | ------------------------------------- | --------------------------------- |
| 1024×64 | **135.7μs** | 152    | 205                                   | 214                               |
| 256×128 | **201.5μs** | 429    | 413                                   | 693                               |


Block-per-matrix is sync-bound (3N `__syncthreads`, ~3 blocks/SM at 64KB shared
for n=128); warp-per-matrix has too little per-matrix parallelism + a load-
imbalanced rank-1 update (n=128 → 693μs). Adopting either would **regress** the
geomean, so per the guardrail no ranked submission was made.

- **Ranked quota used this session: 0** (still **2 of 3** remaining overall).
- **Current best unchanged: `#877091`** (exp 002, Triton n=32, ~2062μs).
- Modal spend this session ≈ **~$1–2** (one heavy image build + ~4 short runs).

### Insight

cuSOLVER's batched `potrf` is near-optimal at n=64/128 on B200; a *naive* right-
looking kernel can't beat it. Winning would need a **blocked/recursive** kernel
(panel factorization + batched-GEMM trailing update), likely **tensor-core
(tf32/bf16) Schur updates with FP32 accumulation** (the tolerance has ~1000×
headroom), and multiple matrices per block for occupancy. That's a multi-hour
kernel effort with uncertain payoff — deferred. Details in
`experiments/003-cuda-n64-n128/notes.md`.

---

## 2026-07-15 — Session 2: first custom kernel (Triton n=32) → ranked #877091

### Goal

Beat the cuSOLVER baseline (ranked `#876988`, geomean ≈ 2080μs) by replacing
cuSOLVER with custom kernels on the highest-ROI (launch/overhead-bound) shapes:
`4096×32`, `1024×64`, `256×128`.

### What was built

- **Triton batched Cholesky kernel** (`submission.py`): one program (CTA) per
matrix, whole `N×N` matrix held in a single tile spread across the block's
threads. Right-looking factorization — at step k: `inv = 1/sqrt(A[k,k])`,
scale column k, rank-1 update of the trailing submatrix — then zero the strict
upper triangle. `N` is a `constexpr` so the k-loop is fully unrolled and the
kernel is specialized per size (Triton caches the compile at module scope).
- **Dispatcher**: `custom_kernel` routes `n==32` (CUDA, fp32) to the Triton
kernel; everything else stays on `torch.linalg.cholesky_ex` (cuSOLVER).
- **Harness upgrades** (`scripts/_gpu_runner.py`, `scripts/modal_verify.py`):
  - `--shapes` filter (e.g. `--shapes 32,64,128`) to benchmark only active shapes
  in the inner loop and save B200 cost.
  - **L2-cache clear** (256 MB buffer zeroed between timed iters) + **adaptive
  iters** (50 for n≤256, down to 8 for the huge matrices) to better mirror
  popcorn's official timing (which clears L2 via `clear_l2_cache`).
  - Extra `n=32` verify specs across all families (spectrum/diagonal/lowrank/
  rowscale/tridiagonal + high batch) to harden the correctness gate.

### The decisive experiment (Modal B200, L2-clear method — apples-to-apples)


| shape   | cuSOLVER | Triton (num_warps=1) | verdict         |
| ------- | -------- | -------------------- | --------------- |
| 4096×32 | 137.8μs  | **84→76μs**          | **Triton −39%** |
| 1024×64 | 135.7μs  | 152μs (best cfg)     | cuSOLVER wins   |
| 256×128 | 201.5μs  | 429μs                | cuSOLVER wins   |


**Key insight — `num_warps=1` is the unlock for n=32.** With one warp per matrix,
Triton's per-column reductions (`tl.sum`) compile to cheap in-warp shuffles
instead of shared-memory syncs. That beats cuSOLVER's batched-launch overhead.
For n≥64 a single warp spills registers (n=64→128 regs/thread; n=128 catastrophic
at ~5ms), and multi-warp configs re-introduce sync cost, so both lose to cuSOLVER.
→ **Triton only pays off at n=32** with the current tile-per-matrix design.

### Results — ranked submission `#877091` (17/17 pass, B200)

Custom kernel correct on **all** families at n=32 (worst scaled reconstruction
residual 0.082, tolerance is 20 — huge margin). Modal verify: 19/19.

#### Ranked per-shape (popcorn), baseline `#876988` → this run `#877091`


| shape       | #876988   | #877091    | Δ                     |
| ----------- | --------- | ---------- | --------------------- |
| **4096×32** | **113μs** | **63.7μs** | **−44%** ← the win    |
| 1024×64     | 110       | 110        | —                     |
| 256×128     | 152       | 152        | —                     |
| 64×256      | 276       | 276        | —                     |
| 16×512      | 597       | 600        | —                     |
| 640×512     | 3810      | 3800       | —                     |
| 4×1024      | 1280      | 1395       | +9% (cuSOLVER drift)  |
| 60×1024     | 2900      | 2900       | —                     |
| 2×2048      | 3220      | 3840       | +19% (cuSOLVER drift) |
| 8×2048      | 4910      | 5010       | +2% (drift)           |
| 1×4096      | 1540      | 1534       | —                     |
| 2×4096      | 11400     | 13400      | +18% (cuSOLVER drift) |
| 1×8192      | 6400      | 6410       | —                     |
| 1×16384     | 34200     | 34200      | —                     |
| 1×32768     | 221000    | 221000     | —                     |


**Geomean of this ranked run ≈ 2062μs** (computed from the per-shape means; the
`popcorn submissions list` Score column shows `-`). That is below the recorded
baseline of ~~2080μs, so the definition of done is met — but only marginally in
*absolute* terms, because several **cuSOLVER** shapes (identical code) ran
notably slower this session (`2×2048`, `2×4096`, `4×1024`). That is pure
run-to-run environment drift, not a regression. Same-environment (Modal,
L2-clear, everything but n=32 held fixed) the win is **~~3.9%**: n=32 alone moves
the geomean-monotone score from an equivalent pure-cuSOLVER ~2388μs to 2296μs.

### Findings & insights

- **Confirmed the launch/overhead-bound hypothesis for n=32.** 113→63.7μs from a
single fused Triton launch vs cuSOLVER's batched dispatch across 4096 tiny
matrices. The floor is ~~memory-bound (~~5μs for 32 MB R/W); 63.7μs is still
mostly fixed overhead, so there may be a little more with a multi-matrix-per-
program design, but returns are small.
- **Triton's tile-per-matrix model caps out at n=32 here.** The right-looking
loop needs per-step column extraction (a reduction). One warp keeps that as
shuffles (fast) but limits registers; more warps add sync cost. n=64/128 need
a **warp-per-matrix CUDA kernel** (register-blocked columns + `__shfl`), which
needs nvcc — not available in our pip-torch Modal image (would require a CUDA
*devel* base image to test). Deferred: higher effort + risk.
- **cuSOLVER shapes drift run-to-run** on the board (~±20% on some mid shapes),
so absolute geomean deltas < a few % are noisy. Trust per-shape same-seed
deltas (n=32: 113→63.7 is rock-solid) over the raw geomean number.
- **Accuracy is a non-issue** for this simple FP32 right-looking factorization —
residuals are 100–1000× inside tolerance across all families.

### Cost

~~9 Modal B200 sandbox runs this session (verify/benchmark, ~40–65s each) ≈ ~10 min
B200 wall ≈ **~~$1–2** Modal spend. popcorn test+leaderboard run on GPU MODE infra
(not billed to our Modal). Ranked submissions used this session: **1 of 3**.

### Next steps (to chase the board leader ~1924μs)

1. **Warp-per-matrix CUDA kernel for n∈{64,128}** via `load_inline` (nvcc is on
  popcorn's runner per the brief). Design: block-per-matrix, `n` threads, thread
   `j` owns column `j`; right-looking with a shared-mem pivot-column broadcast;
   ~2n syncs, O(n³/n) work/thread. To iterate on Modal, switch the image to an
   `nvidia/cuda:*-devel` base so `load_inline` can compile there. Wrap in
   try/except → fall back to cuSOLVER so a compile failure never breaks ranking.
   Potential: if 64/128 also reach ~0.5× cuSOLVER, geomean → ~1810μs (beats leader).
2. **Multi-matrix-per-program Triton for n=32** to shave the remaining launch
  overhead (63.7 → maybe ~50μs). Small ROI but cheap and low-risk.
3. Leave `n≥256` on cuSOLVER (compute-bound; cuSOLVER already near SOL).

---

## 2026-07-15 — Session 1: setup → first ranked submission

### Goal

Participate in the GPU MODE `[cholesky` leaderboard (776)](https://www.gpumode.com/leaderboard/776?tab=rankings)
— batched dense Cholesky factorization on **B200**, ranked by geometric mean of
runtime across 15 benchmark shapes. Ambition for this session: **land a correct
ranked submission first**, defer deep optimization.

### Environment

- Dev machine: macOS, **no local NVIDIA GPU**.
- `popcorn` CLI installed; authenticated via **GitHub** this session.
- `modal` used on-demand via `uv run --with modal` (`~/.modal.toml` already present).

### What was built

- `submission.py` — cuSOLVER baseline (`torch.linalg.cholesky_ex(...).L`) with
`#!POPCORN leaderboard cholesky` / `#!POPCORN gpu B200` directives and a
shape-dispatcher structure for future custom kernels.
- `reference/` — vendored read-only harness (`task.py`, `reference.py`, `eval.py`,
`utils.py`); the checker here is the real spec.
- Three-tier verification:
  - `scripts/verify_local.py` — free CPU property check.
  - `scripts/modal_verify.py` + `scripts/_gpu_runner.py` — real **B200** via a Modal sandbox.
- Plan: `docs/plans/2026-07-15-001-feat-cholesky-leaderboard-submission-plan.md`.

### Results


| Check                                        | Result                                              |
| -------------------------------------------- | --------------------------------------------------- |
| CPU property check (`verify_local.py`)       | **10/10 pass**                                      |
| Modal B200 verify (`modal_verify.py verify`) | **13/13 pass** on `NVIDIA B200`, torch 2.12.0+cu130 |
| popcorn `--mode test`                        | **17/17 pass** on B200                              |
| popcorn `--mode leaderboard` (`#876988`)     | **done, 17/17 pass**, ranked geomean ≈ **2080μs**   |


Reference points on the board at submission time: xuan9938 ~~1924μs, msaroufim ~2041μs.
The raw cuSOLVER baseline (~~2080μs) is already competitive — roughly ~2% behind 2nd.

#### Ranked per-shape times (popcorn, B200)


| shape   | mean    |     | shape   | mean    |
| ------- | ------- | --- | ------- | ------- |
| 4096×32 | 113 µs  |     | 60×1024 | 2.90 ms |
| 1024×64 | 110 µs  |     | 2×2048  | 3.22 ms |
| 256×128 | 152 µs  |     | 8×2048  | 4.91 ms |
| 64×256  | 276 µs  |     | 1×4096  | 1.54 ms |
| 16×512  | 597 µs  |     | 2×4096  | 11.4 ms |
| 640×512 | 3.81 ms |     | 1×8192  | 6.40 ms |
| 4×1024  | 1.28 ms |     | 1×16384 | 34.2 ms |
|         |         |     | 1×32768 | 221 ms  |


Raw logs: `results/leaderboard-*.txt`, `results/test-*.txt`.
Summaries (committed): `results/ranked-submission-876988.json`, `results/baseline-benchmark.json`.

### Findings & insights

- **The baseline is already strong.** Plain `torch.linalg.cholesky_ex` (cuSOLVER) on
B200 lands within ~2% of 2nd place. The competition is tight at the top; wins are marginal.
- **Only soft spots are small-`n` / high-batch shapes** (`4096×32`=113μs, `1024×64`=110μs,
`256×128`=152μs). These are **launch/overhead-bound**, not compute-bound — a 32×32
factorization is trivial, so ~110μs is almost pure per-call + dispatch overhead across
thousands of tiny matrices. This is exactly where custom batched kernels win, and matches
the leaders' known trick (cf. the repo's `triton_cholesky32.py`, one program per matrix).
- **Large single matrices are compute-bound** (`32768²`=221ms, `16384²`=34ms). cuSOLVER is
already near speed-of-light here; low ROI — leave on cuSOLVER.
- **Property-based checker is forgiving on accuracy** — scaled reconstruction residuals were
~0.0006–0.024 (tolerance is `20·n·eps·‖A‖₁`). There's headroom to trade a little accuracy
for speed (e.g., TF32 in intermediate steps) *if* it doesn't break the FP32 reconstruction gate.
- **Modal verification paid off as a pre-flight.** Both the Modal B200 verify and the popcorn
test reported identical residuals — Modal caught nothing broken here, but it means future
kernel work can be validated on the exact hardware without burning ranked quota.

### Gotchas

- `modal.Sandbox.exec()` timed out connecting to Modal's newer per-task command-router
(blocked egress here). Fix: run the command as the sandbox **entrypoint** and stream
`sandbox.stdout` — the documented pattern, works over the standard control channel.
- Default torch wheel (2.12.0+cu130) already ships Blackwell/sm_100 kernels — no cu128 pin needed.
- `popcorn register` is OAuth (github/discord); must be completed in a browser.

### Next steps (deferred optimization)

1. Custom batched kernel for `n ∈ {32, 64, 128}` (Triton or CUDA `load_inline`), starting from
  the `triton_cholesky32.py` pattern; dispatch on `(batch, n)` in `custom_kernel`.
2. Re-benchmark on Modal (`modal_verify.py benchmark`) before each ranked submission.
3. Tune high-batch mid-size shapes (`640×512`, `8×2048`, `2×4096`) for occupancy.
4. Leave `n ≥ 8192` on cuSOLVER.
