# Experiment 049 V5 outcome

The previously blocked fused-panel overlay was resumed in a clean worktree at
exact ranked `#890798`. Free gates and the repository audit contract passed.

The paired B200 gate proved the intended route (`_FUSED512_HITS=9`, readiness
positive, no new fallback/error) and retained-output correctness. It regressed
the target from `397.616us` to `423.632us`: `0.938446x`, CI95
`[0.937938, 0.938906]`. The off-target `640x512` control remained parity at
`0.999076x`.

V5 is rejected before family or full-grid spend. Together with the four prior
negative persistent architectures, this exhausts Experiment 049. No Popcorn
submission and no root-source change occurred.
