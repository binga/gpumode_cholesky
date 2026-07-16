# Experiment 014 historical checkpoint

Status: resolved. The owner resumed the workflow in an environment that allowed
the standing-authorized Modal export. Family and full-grid gates passed; Popcorn
test `#880765` passed 17/17; leaderboard `#880770` improved public and secret
scores; the candidate was adopted at the repository root.

## Exact baseline

- Ranked submission: `#878893`
- Commit: `141d015aa54dee65109722f9a59742588f20926d`
- Source SHA-256:
  `112ee017f96f4dafb95a173cf51bb59190c2ded1e7702a64992e6795985759dd`
- Public / secret score: `1459.321342997556 us` / `1448.3768036226527 us`
- Slowest full-grid shape: `batch=1, n=32768`, `51909.292 us` mean

`baseline-exp012.py` and the source configured in
`profiling-harness/profile-config.json` both reproduce the exact source hash.

## Measured architecture ladder

All target numbers are paired, same-process Modal B200 CUDA-event means against
the exact source lock. Each JSON artifact contains the raw samples, official
checker results, and backend-counter contract.

| Variant | Baseline (us) | Candidate (us) | Speedup | Classification |
| --- | ---: | ---: | ---: | --- |
| Active 512-microblock superpanel | 52042.613 | 86184.138 | 0.604x | REJECTED |
| Custom CUDA 128 POTRF | 51689.344 | 315719.248 | 0.164x | REJECTED |
| CUDA 128 POTRF, padded shared memory | 56961.418 | 117314.149 | 0.486x | REJECTED |
| CUDA 128 POTRF, warp synchronous | 52031.163 | 182105.324 | 0.286x | REJECTED |
| Fused E4M3 scale/cast | 51915.019 | 51430.054 | 1.009x | FRONTIER |
| Fused tiled amax plus E4M3 scale/cast | 51939.254 | 47896.867 | **1.084x** | **FRONTIER** |
| Fixed-scale shadow FP8 factor | 52499.491 | 51293.734 | 1.024x | FRONTIER |

The strongest frontier is `candidate-fused-e4m3-amax-quant.py`. Its two
rotating dense inputs pass the unmodified official checker with the same
`4.52 / 20.0` scaled residual as the baseline. The backend contract recorded
16 target hits, 96 fused-amax hits, 96 fused-quantization hits, no large-path
fallback, and no runtime error. It is 8.44% faster but does not meet the 2.00x
program target.

The best new implementations which removed cuSOLVER were correct but slower.
The strongest partial frontier only changes the quantization front end of the
shipped left-looking path, so it still inherits the shipped diagonal cuSOLVER
calls. It must not be described as a non-cuSOLVER winner.

## Current bottleneck profile

The best frontier's component profile reports `48004.254 us` total. The main
self-device costs are triangular inverse/solve (`12074.284 us`), diagonal
Cholesky (`11162.507 us`), TF32 diagonal update kernels (`8063.995 us`),
`addmm_` (`4829.760 us`), copies (`4626.789 us`), panel `mm`
(`4373.670 us`), FP8 `_scaled_mm` (`2523.562 us`), and the fused tiled amax
kernel (`2449.351 us`). The FP8 GEMM is not the primary remaining bottleneck.

## Original continuation command

Free gates pass: Python compilation, JSON parsing, exact baseline source lock,
source-policy scan, and `git diff --check`.

The next authorized command is the changed-family correctness gate:

```shell
UV_CACHE_DIR=.uv-cache uv run --with modal python \
  experiments/014-fused-e4m3-quantization/profiling-harness/modal_profile_013.py \
  families \
  --candidate experiments/014-fused-e4m3-quantization/candidate-fused-e4m3-amax-quant.py \
  --json experiments/014-fused-e4m3-quantization/variant-04-families.json
```

This checkpoint records the earlier interruption for provenance. The final
evidence is in `variant-04-families.json`, `variant-04-full-grid.json`,
`popcorn-test-880765.json`, `ranked-880770.json`, and `notes.md`.
