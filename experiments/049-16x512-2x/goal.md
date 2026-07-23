# Experiment 049 — `16x512` 2x latency target

Control: exact ranked `#890798`, source SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
Experiment 047's paired full grid measured `batch=16, n=512` at `389.408us`
for the now-current source, versus `389.272us` for the prior control (parity).
The strict 2x target is therefore **at most `194.704us`**.

The target is a paired same-process B200 candidate latency at most 50% of the
exact control for `batch=16, n=512`, with the official checker unchanged,
positive intended-backend proof, zero new fallback, and no off-target dispatch.
New experiment paths contain no cuSOLVER factorization and no auxiliary CUDA
queue. Preserve every correct frontier, but classify only a correct candidate
at or above 2.00x as a winner.

Start with a fresh constituent profile: wall/device idle, launch/dependency
span, diagonal micro, panel/apply, inner update, trailing matmul, copies/gates,
and HBM/arithmetic floors. Measure a bounded ladder of at most six materially
distinct serious Blackwell architectures one after another. Record in-kernel
phase constituents for persistent candidates. Root owns the full-grid, Popcorn,
and leaderboard gates; this experiment does not run either Popcorn mode.

Current state: **PAUSED** after four valid measured architectures. V5 is a
locally-gated but unmeasured exact-shape fused-panel overlay; its B200 gate was
denied by the external approval-review usage limit, with reset reported as
July 27, 2026 at 2:10 PM. This is not bounded exhaustion and does not authorize
V6, Popcorn, leaderboard, or root-source edits.
