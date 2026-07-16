# Experiment 014 — fused E4M3 quantization for 1x32768

## Exact baseline

- Ranked submission: `#878893`
- Git commit: `141d015aa54dee65109722f9a59742588f20926d`
- Public score: `1459.321342997556 us`
- Secret score: `1448.3768036226527 us`
- Exact source snapshot: `baseline-exp012.py`
- Paired exp-012 latency: `52139.092 us` mean, `51606.770 us` best
- Retained-output full-grid latency: `51909.292 us` mean, `51844.383 us` best

## Target

Improve `batch=1, n=32768` against the exact exp-012 ranked path. The default
winner threshold is a paired candidate mean at most 50% of the baseline mean
(`<= 26069.546 us`, at least `2.00x`). Correct positive improvements below that
threshold are retained as frontiers.

## Boundaries and gates

- Remove cuSOLVER from every new candidate path; do not use stream-based designs.
- Measure at most six genuinely distinct serious architectures before declaring
  bounded exhaustion. Parameter-only sweeps do not count as architectures.
- Require official reconstruction tolerance, finite lower-triangular output,
  positive diagonal, and changed-family coverage for dense, spectrum, low-rank,
  row-scaled, diagonal, and tridiagonal inputs.
- Paired timing must use the exact ranked baseline in the same B200 process,
  rotate inputs, retain outputs through validation, and prove the candidate
  backend ran without an unexpected fallback or runtime error.
- Run free local checks before Modal. Run the full retained-output 15-shape grid
  only for a positive target frontier. Require aggregate geometric-mean
  improvement and no material off-target regression before Popcorn.
- Run Popcorn test before exactly one ranked submission. Adopt only completed
  public and secret leaderboard evidence.
- Preserve all artifacts and update `journal.md`, the Optimization Tracker,
  `README.md`, and `experiments/README.md` for either adoption or rejection.
