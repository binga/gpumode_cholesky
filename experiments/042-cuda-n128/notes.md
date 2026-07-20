# Experiment 042 result — `256x128` reaches 2x and completes the campaign

Status: **V5 WINNER / ADOPTED.** Exact ranked source `ranked-888996.py`,
SHA-256 `5bbb8e8de3eedfadfb70fdcfaa902723fab03c2998bc18e9cb80512e82cce80c`.

## Control and constituents

Revision-5 exact `#888867` measured 154.824us, so the contract threshold was
77.412us. A fresh profile of its 18-operation split32 graph measured 143.3us
wall / 115.2us device: 55.13us diagonal micros, 19.14us panel-inner, 13.67us
panel apply, 8.90us copies, 5.54us finite reduction, 3.59us device-to-host
gate, 9.18us other elementwise work, and 28.1us wall-minus-device.

V1's four-warp register-row kernel was correct but only 1.081x. Device clocks
split it into 6.50us staging, 89.25us scalar factor, and 2.43us output; a forced
256-barrier floor was only 2.88us. V2/V3 shared element ownership measured
0.988x/0.983x. The evidence ruled out launch, copy, synchronization, and
occupancy scheduling as standalone levers.

## Winning architecture and gates

V4 introduced FP32 blocked-16 Cholesky in one eight-warp CTA per matrix. A
padded 128x129 shared tile holds the matrix; each round factors one 16x16
diagonal block, solves panel rows in registers, and applies one rank-16
trailing dot. V5 forces the outer eight-round loop not to unroll, retaining the
speed and removing the official compile timeout.

- Isolated V5: 142.920 -> 70.356us = **2.0314x**.
- Six-family gate: 6/6 active, no fallback, worst residual 0.0176/20.
- Exact V5 full grid: 15/15 correct, target **140.932 -> 69.852us = 2.0191x**,
  all other shapes at parity, aggregate **1.047866x**, CI
  [1.047341,1.048391].
- Stage-specific control: 154.824 -> 69.852us = **2.2165x**.
- Popcorn test `#888995`: 17/17.
- Ranked `#888996`: **916.5768129471865us public / 863.8500740634134us
  secret**. Versus `#888867`, public drifted 1.904% slower while secret
  improved 4.812%.

Instrumented V5 device span was 59.55us: 3.30us staging, 21.28us diagonal,
6.05us panel, 23.62us trailing, 1.12us output, and ~4.18us boundary overhead.
V6 dual accumulation was noise (1.0018x, CI crosses 1), V7 BK=32 regressed
21.8%, and V8 16-warps regressed 5.2%.

Popcorn V4 test `#888971` ended at the exact six-minute runner limit with no
failed-case output; V5's compact code generation fixed it.
