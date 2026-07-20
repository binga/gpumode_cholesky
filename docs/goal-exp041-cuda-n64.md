# Goal — Experiment 041: cuSOLVER-free CUDA `1024x64`

## Frozen baseline and target

- Ranked source: `#888636`, commit `20d3deb`, SHA-256
  `e6672b39a324a4d6247d803fdf4bf62422b7afb66d1aac09063a55e5990770d1`.
- Revision-4 byte-identical audit: **124.540us** candidate-side latency; exact
  2x threshold **62.270us**. Null A-vs-A spread was 0.15% on this shape.
- Off-target regression limit: 3%; aggregate three-shape audit must improve.

## Constituent diagnosis

The shipped graph path is 119.8us wall / 119.4us device with only 0.4us idle
across 17 operations. Vendor factor/SYRK/TRSM kernels cost 73.56us, output
elementwise plus strict-lower cleanup 33.08us, two device copies 9.53us, and
info setup 2.26us. The 32MiB compulsory input/output traffic floor is about
4.4us. Replaying the graph cannot reach 2x because its constituent kernels,
copies, and output cleanup already exceed the target.

## Falsifiable architecture

Use one warp per 64x64 matrix. Each lane owns two complete rows in registers;
the warp loads/stores through a padded 64x65 shared tile for coalescing. Two
pivots are processed per iteration, with only the two pivot columns exchanged
through shared memory. This generalizes Experiment 039's winning rank-2
register-row mechanism while keeping 1024 independent matrices available for
B200 occupancy. One launch writes the required lower-triangular output
directly, removing the graph's 17-operation factor/copy/cleanup chain.

## Bounded ladder and gates

1. Rank-2 register rows, one warp and one matrix per CTA.
2. Two warps / one matrix if register latency dominates.
3. Two matrices per CTA if launch/occupancy dominates.
4. Rank-4 pivots if the serial pivot chain remains visible.
5. Vectorized shared staging or direct global staging based on the profile.
6. Stop after six distinct correct measured architectures.

Require active `_CUDA64_HITS`, all six input families, paired latency at or
below 62.270us, full 15-shape parity, Popcorn test 17/17, then one serial ranked
submission. The new fast path may use neither cuSOLVER nor an
auxiliary/concurrent CUDA queue API.

## Result — 2x achieved, refined, and ranked

V1 crossed the target immediately: isolated paired latency 122.324 ->
53.896us (**2.2696x**), six families 6/6 with worst residual 0.0376/20, and
full-grid target 2.2130x with aggregate 1.05441x. Popcorn test `#888798` passed
17/17; ranked `#888803` scored 928.078us public / 921.303us secret. Exact
ranked SHA-256 is
`aa7a5badc577ba365f468773f9516b18b1f470809de077934d8e88c2f2317b42`.

The authorized post-target ladder then rejected correct rank-4 V2 at 0.9353x
and adopted two-warp V3. One register row per thread plus a four-rendezvous
rank-2 handoff improved exact V1 another 53.584 -> 32.192us (**1.6640x**) on
the full paired grid; aggregate latency improved 1.034641x with all 15 shapes
correct. Popcorn test `#888864` passed 17/17, and ranked `#888867` scored
899.125us public / 905.417us secret. Exact latest ranked SHA-256 is
`7380e038441b55666819d6685ff3ddd68776c7571757afced15c29b3656ac9c2`.
