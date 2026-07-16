# Experiment 014 — fused E4M3 quantization for `1x32768`

Status: **ADOPTED — ranked winner `#880770`.** Public geomean
`1447.2589334363144 us`; secret geomean `1443.2264907145392 us`. Root
`submission.py` is the exact ranked source.

## Baseline and goal

The exact control is ranked submission `#878893` at commit
`141d015aa54dee65109722f9a59742588f20926d`. Its source SHA-256 is
`112ee017f96f4dafb95a173cf51bb59190c2ded1e7702a64992e6795985759dd`.
It scored `1459.321342997556 us` public and `1448.3768036226527 us` secret.
The slowest full-grid shape is `batch=1,n=32768`: `51909.292 us` mean and
`51844.383 us` best in the retained-output exp-012 Modal run.

The program's default target is at least `2.00x` paired speedup. Every new
architecture is default-queue-only. New custom factorization candidates remove
cuSOLVER; precision-front-end candidates which retain the ranked diagonal path
are recorded as partial frontiers, never as non-cuSOLVER winners.

## B200 environment and evidence contract

All measured target variants ran in one process with the exact source-locked
control on an NVIDIA B200 (`SM 10.0`, 191,503,138,816 bytes), PyTorch
`2.13.0+cu130`, and CUDA runtime 13.0. Two dense inputs rotate, invocation order
alternates, L2 is flushed, CUDA events measure latency, and timed outputs remain
live through the unmodified official checker. Candidate-specific counters must
increase; fallbacks and runtime errors invalidate timing.

## Architecture ladder

| ID | Architecture | Baseline mean | Candidate mean | Speedup | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| 01 | Triton 512-microblock active superpanel, custom diagonal factor | 52042.613 us | 86184.138 us | 0.604x | REJECTED |
| 02 | Custom CUDA 128-wide POTRF active superpanel | 51689.344 us | 315719.248 us | 0.164x | REJECTED |
| 02b | Same CUDA architecture, padded shared stride and row-owned updates | 56961.418 us | 117314.149 us | 0.486x | REJECTED defect iteration |
| 02c | Same CUDA architecture, warp-synchronous update | 52031.163 us | 182105.324 us | 0.286x | REJECTED defect iteration |
| 03 | Joint fused E4M3 scale/cast, ranked factor schedule | 51915.019 us | 51430.054 us | 1.009x | FRONTIER |
| 04 | Tiled dual amax plus joint fused E4M3 scale/cast | 51939.254 us | **47896.867 us** | **1.084x** | **FRONTIER** |
| 05 | Fixed-scale FP8 shadow factor | 52499.491 us | 51293.734 us | 1.024x | FRONTIER; superseded |

Variants 02b and 02c are repairs within architecture 02, not extra serious
architecture slots. Variants 03–05 preserve the ranked diagonal cuSOLVER calls,
so they provide useful precision/data-movement evidence but do not satisfy the
new non-cuSOLVER architecture requirement. The active-superpanel and custom
CUDA paths are the two measured cuSOLVER-free architectures. The bounded search
is not declared exhausted.

### Variant 01: active superpanel

The candidate factors each 4096-wide active superpanel in 512-wide microblocks
with the existing custom Triton diagonal factorization, applies a triangular
inverse with vendor triangular solve, and updates only the remaining rectangular
superpanel. It removes cuSOLVER and avoids a full trailing square. Both rotating
inputs pass (`4.53 / 20` candidate versus `4.52 / 20` control), but the custom
diagonal and panel solve cost makes it `39.6%` slower than the control.

### Variant 02: custom CUDA POTRF

The first 128-wide CUDA POTRF serialized too much work and took about `258 ms`
inside a `315.7 ms` factorization. Padded shared memory and row-owned updates
reduced POTRF to about `55 ms` and end-to-end time to `117.3 ms`, still over 2x
slower than the ranked path. A warp-synchronous rewrite regressed. All versions
were correct (`4.53 / 20`) and proved their custom backend; the architecture is
rejected on latency rather than correctness.

### Variants 03–04: fused dynamic quantization

The ranked path separately computes whole-tensor `abs().amax()`, scales, casts,
and invokes `_scaled_mm` for both operands on six previous-panel products.
Variant 03 fuses the two scale/cast passes. Variant 04 additionally computes
both operand maxima with one tiled Triton kernel plus tiny final reductions.
It retains the exact scalar E4M3 scaling and FP32 `_scaled_mm` output, so it does
not spend additional numerical margin.

The robust 10-round variant-04 artifact reports:

- control mean/best: `51939.254 / 51752.354 us`;
- candidate mean/best: `47896.867 / 47687.393 us`;
- paired mean speedup: `1.0843977x`;
- both rotating inputs: candidate and control `4.52 / 20` residual;
- backend deltas: 16 left-looking target hits, 96 tiled-amax hits, 96 fused
  quantization hits, zero fallback, and no errors;
- candidate source SHA-256:
  `78b2282d436243393897e61a5e4b8206d52c3950ec6f4495cbc71da895abd1fc`.

The operator profile independently proves `_scaled_mm`, `mm`, `addmm`, diagonal
Cholesky, and triangular solve engagement. It reports `48004.254 us` total.
The largest costs are triangular inverse/solve (`12074.284 us`), diagonal
Cholesky (`11162.507 us`), TF32 diagonal update (`8063.995 us`), `addmm_`
(`4829.760 us`), copies (`4626.789 us`), panel `mm` (`4373.670 us`), FP8
`_scaled_mm` (`2523.562 us`), and tiled amax (`2449.351 us`). This rules out the
FP8 GEMM itself as the dominant remaining bottleneck.

### Variant 05: fixed-scale shadow factor

This architecture quantizes completed panel data into a reusable shadow factor
with a guarded fixed scale, avoiding repeated dynamic reductions. It passes with
`4.51 / 20` residual and zero fallback. Its steady-state samples are faster than
the ranked source, but shadow copies and layout conversions make it slower than
variant 04, so it is a superseded frontier.

## Free gates

The exact baseline snapshot reproduces the source lock. Python compilation,
JSON parsing, source-policy scans, and `git diff --check` pass. No credentials or
unrelated repository content are part of the Modal image.

The family audit deliberately retains the repository's unmodified
`reference.generate_input` and `check_implementation`. The sweep covers dense,
spectrum, low-rank, row-scaled, diagonal, and tridiagonal at the exact changed
`1x32768` dispatch. The 32768 spectrum QR is costly but is not replaced by an
easier structured proxy; changing it would weaken comparability with the
repository's prior changed-region evidence.

## Changed-family correctness

The corrected Modal family artifact passes **6/6** with the unmodified official
generator and checker. Dense, diagonal, and tridiagonal use the target path with
zero fallback. Spectrum, low-rank, and row-scaled each record the intended
left-looking, fused-amax, and fused-quantization hits, then take exactly one
expected safety fallback in both baseline and candidate.

| Family | Candidate scaled residual / 20 | Safety fallback |
| --- | ---: | ---: |
| dense | 4.52 / 20 | 0 |
| spectrum | 0.000537 / 20 | 1 expected |
| low-rank | 0.000507 / 20 | 1 expected |
| row-scaled | 0.000042 / 20 | 1 expected |
| diagonal | 0.000061 / 20 | 0 |
| tridiagonal | 0.00408 / 20 | 0 |

The first family attempt is preserved because it exposed a harness-contract
defect: the old contract rejected all fallbacks even though the ranked baseline
intentionally falls back on those three ill-conditioned families. The corrected
contract requires those deltas to equal one rather than silently allowing them.

## Full 15-shape Modal gate

All **15/15** outputs pass. The exact source-locked control geomean is
`1574.149644 us`; candidate geomean is `1565.545754 us`, an aggregate
`1.0054958x` improvement. Maximum off-target candidate/control mean is
`1.016644x` at `1x16384`, below the `1.03x` limit. The exact changed shape is:

| Shape | Baseline mean | Candidate mean | Candidate best | Mean speedup | Residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1x32768` | 51874.714 us | 51198.891 us | 47778.370 us | 1.0132x | 4.52 / 20 |

The arithmetic mean conservatively includes one 68.16 ms candidate allocation
outlier. The other five candidate samples are 47.78–47.82 ms, consistent with
the dedicated 10-round target result (`1.0844x`). No timed sample was removed.
Backend proof records 60 fused-amax hits, 60 fused-quantization hits, ten target
calls, zero unexpected fallback, and no runtime error.

Two failed harness attempts are documented in `full-grid-initial-run.md`. The
first exceeded Modal's stdout-line limit. The second proved a warmup allocation
reference polluted one reversed-order sample. The final run uses exact-length
chunked transport and releases only the warmup reference; all timed outputs stay
live through validation.

## Popcorn and ranked result

- Popcorn test `#880765`: **17/17**, B200, succeeded.
- Exactly one leaderboard submission: `#880770`, all public and secret test,
  benchmark, and leaderboard stages succeeded.
- Public: `1459.321342997556 -> 1447.2589334363144 us`, an improvement of
  `12.062410 us` (`0.8266%`).
- Secret: `1448.3768036226527 -> 1443.2264907145392 us`, an improvement of
  `5.150313 us` (`0.3556%`).
- Ranked source SHA-256:
  `78b2282d436243393897e61a5e4b8206d52c3950ec6f4495cbc71da895abd1fc`.

## Verdict, cost, and next ideas

**ADOPTED.** The 2.00x target was not reached, and the measured cuSOLVER-free
architectures were decisively slower. The best verified partial frontier still
improves the aggregate and both leaderboard scores, so `program.md` requires it
to ship. Experiment 013's independently measured no-cuSOLVER rejection remains
intact; this winner is numbered 014 because the remote experiment 013 landed
while this search was in progress.

The run used the existing Modal image plus the bounded baseline, candidates,
harness, and checker. Approximately a dozen B200 jobs covered architecture
probes, component profiling, family validation, and repeated full-grid harness
repairs; exact billing was not available. Popcorn consumed one test and exactly
one ranked submission.

The remaining target cost is dominated by triangular inverse/solve, diagonal
Cholesky, TF32 diagonal update, copies, and panel `mm`, not the FP8 GEMM. The
next credible high-effort direction is a fused cuBLASLt/CUTLASS epilogue or a
native Blackwell collective which removes product temporaries while preserving
the ranked diagonal path. Repeating custom Python/Triton POTRF, smaller block
sizes, or fixed-scale shadow copies is closed by experiments 013–014.
