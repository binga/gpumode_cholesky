# Experiment 044 — mid-shape split32 diagonal chain

**Verdict: ADOPTED.** Ranked `#889994` = **852.746us public / 847.396us
secret**, improving `#888996` (916.577 / 863.850) by **6.966% public /
1.905% secret** and beating the previous best-ever public score `#888867`
(899.125us) by 5.161%.

## What shipped

`micro_potrf32_rank4` — a warp-synchronous rank-4 CUDA factorization of the
32x32 diagonal block, replacing Triton's `_micro_potrf_gj32` on the two
eager-mode split32 shapes. One warp owns one matrix, lane `r` holds row `r` in
registers, four pivots share one rendezvous, and the triangular inverse is
solved one column per lane out of the finished factor. It is compiled into the
experiment-042 extension rather than its own, so the submission still performs
three nvcc invocations (see *Compile budget* below).

Measured full 15-shape paired grid vs the exact ranked source:
**1.012977x CI [1.012531, 1.013423]**, 15/15 pass, no shape outside the
dispatch region moved more than the 0.55% A-vs-A noise floor.

| shape | baseline | candidate | ratio |
|---|---|---|---|
| 640x512 | 1531.6us | 1394.6us | **1.0982x** |
| 60x1024 | 1382.7us | 1250.3us | **1.1061x** |
| other 13 | — | — | 0.9995–1.0005x (parity) |

Residuals are byte-identical to the baseline on every shape; all six input
families pass the official checker at both changed shapes with no new
fallback.

## Why not 2x — the measured floor

The kernel design space was searched with a `midprobe` harness that times
competing architectures in one Modal run. Per-launch cost of one 32x32
diagonal factorization at 16x512 (3.5us eager launch floor included):

| architecture | us/launch | ns/pivot |
|---|---|---|
| Triton `_micro_potrf_gj32` (shipped) | 13.56 | 424 |
| CUDA rank-1, uncoalesced staging | 12.31 | 275 |
| CUDA rank-1, coalesced staging | 11.25 | 242 |
| CUDA rank-2, coalesced | 11.27 | 242 |
| **CUDA rank-4, coalesced (shipped)** | **10.26** | **211** |
| CUDA rank-1, inverse omitted | 8.20 | 147 |

Fused whole-block architectures, one CTA per matrix over an NB-wide diagonal
block, were all worse per column:

| architecture | us/block | chain for n=512 |
|---|---|---|
| block64, BK=16, 8 warps | 20.73 | 165.8us |
| block128, BK=16, 8 warps | 47.36 | 189.4us |
| block128, BK=32, 8 warps | 67.78 | 271.1us |
| block128 + full 128x128 inverse (hybrid warp-sync) | 57.55 | 230.2us |

All are **batch-independent** (block64 measured 20.730 / 20.698 / 20.730us at
batch 16 / 4 / 2), confirming exposed serial latency rather than throughput.
An eight-warp `__syncthreads` chain costs ~324ns/pivot against ~134ns/pivot
for a single warp, so widening the CTA makes the chain *slower*.

Cholesky needs `n` sequential square roots. At the best measured 134ns/pivot
that is 69us for n=512, 137us for n=1024 and 274us for n=2048 before any
panel, trailing, copy or gate work — against 2x targets of 199 / 403 / 681us.
2x on these shapes is therefore not reachable by attacking the diagonal chain
alone; it needs the panel and trailing launches collapsed as well, and the
fused-block measurements above show that collapsing them costs more than it
saves at this size.

## The blocked 4.3% — CUDA graph capture

The same kernel measures **1.1910x on 16x512, 1.2214x on 4x1024 and 1.1857x
on 8x2048** (full grid, `variant-05-fullgrid.json`, aggregate **1.055953x**).
Those three shapes replay their split32 chain from a CUDA graph, and a
`<<<grid, block>>>` launch with no queue argument cannot be captured into one:
measured 0.38–0.52x there, falling through the finiteness guard. Naming the
current work queue in the launch makes capture work — that is what
`variant-05` did — but popcorn's source policy rejects the submission
("Your code contains work on another stream"), so it was not shipped and no
attempt was made to disguise it. Recovering those three shapes is worth
another ~4.3% geomean if the policy ever admits an explicit current-queue
launch.

## Rejected variants

- **v8, 60x1024 eager -> CUDA graph**: 1.0056x. The 583.5us `shapediag`
  reported as idle at this shape is eager-launch pipelining, not recoverable
  dead time; graph mode also re-introduces the copy-in/clone-out pair that
  first-touch eager (S17) was chosen to avoid. Eager + CUDA micro (1.1041x)
  is far better.
- **v6, default-queue launch on all five split32 shapes**: 0.6836x geomean —
  the three graph shapes regress to 0.38–0.52x as above.

## Compile budget

Shipping the micro as a fourth `load_inline` extension made the official
runner's compile marginal: test `#889943` failed at exactly the six-minute
limit, `#889955`/`#889978` passed, and ranked `#889979` **failed secret
validation**. Folding the kernel into the experiment-042 extension (three nvcc
invocations, as `#888996`) cut the whole test run to 94 seconds and ranked
cleanly. This repeats the exp-042 V4 -> V5 lesson: a correct, fast kernel can
still be rejected for compile cost alone.

## Artifacts

- `baseline-888996.py` — exact frozen control.
- `candidate-v1.py` — first drop-in (rank-1, uncoalesced), 1.0895x on 16x512.
- `probe-v2.py` / `probe-v3.py` / `probe-v4.py` — the six-variant micro probe,
  the fused-block probe and the hybrid 128-block probe.
- `candidate-v5.py` — explicit-queue version, all five split32 shapes,
  full grid 1.055953x. Not shippable under popcorn source policy.
- `candidate-v7.py` — eager shapes only, four nvcc extensions (compile-fail).
- `candidate-v9.py` — **ranked `#889994`**, merged extension.
- `variant-*.json` — every raw measurement referenced above.
