# Experiment 037 goal

Test the specific Experiment 036 premise that rewriting `_micro_potrf_gj32`
from Triton to CUDA can halve its approximately 13.5us per-call latency.

The frozen ranked source is `#888352` / commit `f84e1de`. The mechanism passes
only if assembly evidence identifies removable instruction work and a synthetic
one-warp kernel with the same launch geometry demonstrates a latency floor at
or below 6.75us. This is a diagnosis experiment: it does not change the ranked
source and spends no leaderboard submission.
