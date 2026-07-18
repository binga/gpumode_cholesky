# Technical levers transferred from the qr_v2 podium

Source material: [sankalp, 12th place](https://sankalp.bearblog.dev/autoresearch/)
and [Michael Lutz, 5th place](https://ml-mike.com/writing/qr_v2/), plus the
podium comparison table in the latter (1st `10billiontokens`, 2nd `gau.nernst`,
3rd `dhu.randhar`, 4th `zhongmingee`).

Scope: technical levers only. Current ranked baseline `#883174`,
**1084.457µs public / 1083.720µs secret**, 15-shape geomean.

---

## 0. What does not transfer, and why

Read this first — it removes the single most-cited technique in both writeups.

**Input-structure detection is worthless here.** In qr_v2, five of twelve scored
cases were non-dense (`rank-deficient`, `clustered`, `mixed`, `near-rank`), and
1st/3rd/5th place all built cheap detectors to route them. Our
`BENCH_SPECS` in `scripts/_gpu_runner.py` is fifteen entries, every one of them
`cond=2` with no `case` key — i.e. **uniformly `dense`**. The `spectrum`,
`diagonal`, `lowrank`, `rowscale`, and `tridiagonal` families appear only in the
*correctness* set. A `diagonal` fast path would be an O(n) factorization and is
tempting, but it can only ever be a correctness-set win and would add a detector
to the timed path for zero ranked benefit.

**Consequence:** the transferable template is **gau.nernst's 2nd place**, not the
winner's. He used *no* input detection at all — pure blocked factorization,
warp-specialized panel kernels, per-shape precision, per-shape panel-width
schedules, custom triangular inverse. Every one of those axes is a live gap for
us.

Also dead on arrival: more BF16x9 (S7 measured it at 6–9 products/GEMM, ~3× a
TF32 product — the arithmetic does not improve), and anything stream-based
(banned by popcorn's source scan, S4/S6).

---

## 1. Do the geomean arithmetic before picking a lane

Mike's `w324`: he computed the cumulative improvement the podium required and
*proved* single-bucket wins could not reach it — that arithmetic, not any
profile, forced his architectural pivot. Ours:

A speedup `s` on one shape moves the score by `s^(1/15)`.

| single-shape speedup | geomean reduction |
|---|---|
| 1.07× | 0.45% |
| 1.10× | 0.63% |
| 1.50× | 2.67% |
| 2.00× | 4.50% |

**A 2× on any one shape buys 4.5%.** That is the ceiling on single-shape work,
and it explains why the tracker's `✗` marks cluster around the 1.0–1.1× band —
the program's default 2.00× gate is asking each shape for the maximum a shape
can structurally deliver.

Where the mass actually is:

- **Four of fifteen shapes are still on cuSOLVER**: `2×2048` and `2×4096`
  (per-matrix loop, S4), `1×4096` and `1×8192` (batched). That is 4/15 of the
  log-score sitting on the vendor path. At 1.5× each: `1.5^(4/15)` = **10.2%
  score reduction**. At 2× each: 18.9%.
- **Seven shapes share one hard-coded panel width** (`_SPLIT32_NB = 128`). At
  1.05× each: `1.05^(7/15)` = **2.3%**, for what is a parameter-schedule change.
- **Four shapes are under 400µs** and carry ~21–24µs of per-call fixed overhead
  (copy-in/clone-out ~9µs + finite-check ~12–15µs, per S28). That is 6–17% of
  those runtimes: **~1.5–2.5%** of score.

Everything below is ordered by that arithmetic, not by how interesting it is.

---

## 2. Backlog

### L1 — Escape cuSOLVER on the four vendor-resident shapes
**Expected: 5–10% score. Highest ROI item by a wide margin. Also the hardest.**

Every writeup's day-one lesson was "escape the library," and every podium
finisher had *zero* library factorization left in the timed path by the end.
We still have four.

The reason they're stuck is Mike's bottleneck #2, verbatim: *"n=2048 and n=4096
have tiny batches. During panel work, there are far fewer independent jobs than
148 SMs."* At `batch=1..2` on `n=4096`, the panel phase runs on 1–2 of 148 SMs.

We already probed this and failed: **exp 028's persistent dual-matrix kernel
measured 0.40–0.49×, attributed to a spin-barrier floor.** That is the correct
diagnosis of the wrong mechanism. gau.nernst's answer was three *different*
mappings, chosen by shape:

- 1 SM per matrix for `n32`/`n512`,
- **2 SMs per matrix via thread-block clusters, with a shared-to-cluster TMA
  broadcast** of the panel — hardware cluster barrier and distributed shared
  memory, not a software global spin,
- multi-SM with panel data through global memory *and* a polled flag, reserved
  for `n2048`/`n4096` only.

Mike tried clusters/DSM too and "stopped at parity"; his stated read on why
gau.nernst got past him is that the difference was *warp-specialized broadcast
of the panel*, rather than trying to widen the serial recurrence itself. That
is the specific thing to build: one warp produces the pivot block while others
consume it, with the handoff over DSM/TMA inside a cluster.

Note the repo constraint: cooperative-launch and grid-sync variants were
non-submittable in the QR project for stream reasons; verify cluster launch is
clean against popcorn's scanner on a *free* gate before spending B200 time.

Kill criterion: if a 2-SM cluster panel cannot beat 1-SM on `2×2048` in
isolation, the mechanism is refuted for all four shapes and this closes.

---

### L2 — Per-shape, non-uniform panel-width schedules
**Expected: 1.5–3% score. Cheapest experiment in this document.**

`submission.py:848` is `_SPLIT32_NB = 128`, one constant applied to all seven
split32 shapes, uniform across every panel of every factorization.

All three of the relevant finishers rejected uniform widths:

- gau.nernst hand-tuned a *schedule per shape* — his `n=512` was
  `(96, 96, 64, 32, 32, 192)`. Wide early, narrow through the middle, one huge
  final block.
- Mike's shipped panel is an **8+8+16 staircase**, and he explains why two
  stages beat one: *"A 32-wide factor keeps twice the live state per thread, and
  that costs occupancy exactly where latency-hiding matters most."* Isolated,
  the move was 3.3× on panel emission.
- zhongmingee used recursive split trees (`96 → 48+48`). Mike calls the
  convergence out explicitly — three people found the same recursion
  independently.

We have the sub-blocking idea already (S20/S21's 64×64 panel-inner subtiling,
1.047–1.128×), which is the same insight applied one level down. What is
untried is letting the *width vary across panels within a factorization*. The
trailing block shrinks monotonically as the factorization walks the diagonal,
so a fixed 128 is necessarily wrong at one end or the other — almost certainly
wasteful on the late panels, where the trailing update no longer amortizes a
128-wide panel factor.

Implementation: replace the constant with a per-shape tuple in
`_SPLIT32_SHAPES`, which already carries per-shape `(panel_prec, trailing_prec,
tile, mode, first_touch)`. Sweep is cheap and free-gateable; measure isolated
panel time, not just end-to-end.

Risk: this multiplies Triton constexpr specializations. See L6 note on compile
budget.

---

### L3 — Unblock the empty "Custom CUDA (tcgen05)" column via NVRTC
**Expected: unknown, plausibly large. Both writeups name this as their top regret.**

Every row of that tracker column is `TBD`. Meanwhile:

- 3rd place: 14.5k-line machine-generated file inlining ~170 CUDA kernels using
  Blackwell **tcgen05** tensor-core MMA.
- 4th place: 25.8k lines of **NVRTC source-JIT**, sm_100 tcgen05/TMEM.
- Mike's regret list: *"first-class use of tcgen05/TMEM, the Blackwell-native
  MMA path my Triton kernels never touched."*
- sankalp's regret list: *"Wasn't able to use tcgen05."*

Our blocker was recorded in S2 as *"needs nvcc — not available in our pip-torch
Modal image (would require a CUDA devel base image to test)."* **NVRTC removes
that blocker entirely**: it compiles CUDA source to cubin at runtime on the
target GPU via `cuda-python`, no nvcc, no devel image, and it is exactly what
zhongmingee shipped. It also sidesteps the `load_inline` compile-at-import
problem on popcorn's runner.

Start narrow: one tcgen05 trailing Schur GEMM on `640×512` or `60×1024`, where
we currently hand the trailing update to Triton `tl.dot` at plain TF32, and
compare against the shipped kernel in the same process. Do not attempt a
rewrite; establish that the toolchain works and that TMEM-resident accumulation
beats `tl.dot` on one tile shape.

---

### L4 — Emulated precision we have not actually tried
**Expected: 1–3%. Directly targets two prior rejections.**

Our precision ladder today: `tf32x3` panels, plain `tf32` trailing at n≥1024,
`fp16` trailing on five split32 shapes (S19), native E4M3 at `1×32768`. We
rejected BF16x9 (S7) and tile-local E4M3 at `8×2048` (S25, incorrect/fallback).

The winner's precision stack was different from all of these:

> *"Survives fp16 precision with compensated arithmetic: three-fp16-MMA
> emulated fp32 dots, TF32×2 splits, MXFP8 two-term dots."*

Two concrete untried items:

1. **fp16x3 (three-fp16-MMA emulated fp32).** Three products, like tf32x3, but
   fp16 MMA has roughly double TF32 throughput on B200 — so three fp16 products
   can land under three tf32 products at comparable accuracy. This is the
   natural replacement for `tf32x3` on the panel dots, where we spend accuracy
   budget precisely because the n-scaled tolerance is tight. It is *not* BF16x9;
   the S7 rejection (9 products) does not apply.
2. **MXFP8 two-term dots** at `8×2048`. S25 failed with *tile-local* E4M3 —
   accuracy, not throughput. MXFP8's per-block scale factors are the standard
   fix for exactly that failure mode, and 4th place paired FP8 with explicit
   precision emulation rather than plain casts.

Also worth one measurement: `TF32×2` splits, a cheaper rung between plain tf32
and tf32x3 that we have never sampled.

Constraint: reconstruction tolerance is `20·n·eps·‖A‖₁` and our margins have run
100–1000× wide on dense inputs — but `cond=2` dense is the *easy* family, and
Mike's note that *"the n512 tests are the stress-tested ones"* has an analogue
here. Validate against `spectrum` and `lowrank` at every changed shape, not just
`dense`.

---

### L5 — Delete the per-call fixed overhead on the four small shapes
**Expected: 1.5–2.5%. Already half-scoped in S28/S29.**

S28 identified ~9µs copy-in/clone-out plus ~12–15µs finite-check chain as a
top-3 cost on every sub-400µs shape, and S29 correctly refuted the cheap
variant on a free gate (`finite/Inf == 0` absorbs an overflowed pivot into a
zero column, so checking `L[n-1][n-1]` is unsound).

The QR analogue is gau.nernst's `n32` route, which Mike describes as the one
shape that *"escapes the pattern, by never solving anything at all."* The
principle is that at these sizes any fixed cost on the timed path is the whole
problem.

The open variant from the tracker is the right one: **an in-kernel flag written
at pivot time** — cheaper than the full-diagonal reduction and strictly
stronger, since it observes the pivot before the zero-column absorption can
occur. Pair it with eliminating the clone-out where the graph-replay path
already owns its output buffer.

---

### L6 — PDL, and an audit of the compile-time budget
**Expected: <1% each. Cheap, and one of them is a latent submission risk.**

**PDL** (programmatic dependent launch) overlaps the tail of one kernel with the
prologue of the next. zhongmingee enabled it *"only where it pays."* Our split32
chain is ~10 launches per shape inside a replayed graph — a reasonable
candidate, and orthogonal to everything above.

**Compile budget is the risk item.** Mike's `w1406`: his constexpr-specialized
kernels needed 267 Triton compiles = 354s cold, against a ranked evaluator with
a **240-second test phase**. Making panel offsets runtime arguments where
specialization was not paying cut it to 96s. *"It barely moved the score, but
every later ranked submission depended on it."*

We are now constexpr-specializing across 7 shapes with per-shape precision,
tile, and mode — and L2 above would multiply that further. Measure cold-start
compile time before it becomes a mysterious ranked failure.

---

### L7 — Audit remaining library triangular solves at mid shapes
**Expected: <1%, but it is the single most-repeated finding in the source material.**

Mike replaced the library triangular solve **four separate times** and called it
*"one thread [that] ran the whole campaign: the library triangular solve was
never the right tool, at any scale."* gau.nernst independently found the same
thing (16×16 diagonal inverses, doubled 32→64 by block substitution). 2nd place
confirmed it in Discord after the deadline. Mike's final instance:
**202µs vs cuSOLVER's 619µs at batch 8.**

We did recursive GEMM triangular inversion at `1×16384`/`1×32768` (1.028–1.055×,
S16a/S17) and it was rejected at `1×8192` (0.954×, S26). Worth confirming that
no library `trsm`/`triangular_solve` remains anywhere in the mid-shape panel
paths — this is a grep, not an experiment, and the prior is unusually strong.

---

## 3. One measurement-validity note

Mike, on his `w466`: a route that *"passed 6/6 runs with a public score of
2.59ms and a secret score of 9.81ms."* And separately, that his eval ledger
recorded failures for submissions that had actually landed and scored, because a
polling timeout looks like a failure.

We have already seen the first phenomenon: **exp 022's rank-4 pivot regressed
public by 1.510% while improving secret by 1.440%**, and it was dropped. That
divergence pattern is worth treating as a signal about route stability rather
than noise, particularly for any candidate from L4 that spends numerical margin.
Public/secret disagreement above ~1% in *opposite directions* should trigger a
re-measurement rather than a verdict.

---

## 4. Suggested order

1. **L2** (panel-width schedules) — cheapest, touches 7/15 shapes, free-gateable.
2. **L5** (fixed overhead) — already scoped, bounded, 4/15 shapes.
3. **L1** (cluster/warp-specialized panels on the four cuSOLVER shapes) — the
   real prize, start it in parallel because it will take double-digit failed
   iterations. Both writeups say so about their equivalents.
4. **L4** (fp16x3, MXFP8) — needs L2 settled first so precision and width are
   not confounded.
5. **L3** (NVRTC/tcgen05) — highest ceiling, highest cost; begin as a toolchain
   spike, not a rewrite.
6. **L6/L7** — opportunistic; run the compile-time audit before the next ranked
   submission regardless.
