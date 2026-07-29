# Status — what is ranked right now

**Loop stage [O1] synchronize.** The single maintained answer to "what is the
incumbent". Update this on adoption; do not restate scores anywhere else.

> Verify against `popcorn submissions list` and by `shasum -a 256` before
> spending — this file can lag, the leaderboard cannot.

## Current ranked winner

- **Current ranked winner `#926130`** (exp 069): `done`, both ranked splits
  (public+secret) **passed** (test `#926123` 17/17). Replaces the shared
  `_exp062_factor` final `work.tril_()` (a full-matrix read+rewrite upper mask,
  measured 145us / 16.8% of `60×1024` and running at ~2.4× its own bandwidth
  floor) with a **write-only `e62_zero_upper` CUDA kernel** that touches only the
  strict-upper elements. The L factor is **byte-identical** to `#922201` (lower
  triangle untouched; both zero the strict upper), so this is a pure,
  value-independent latency reorchestration that carries to the secret split
  (exp-067 class, distinct from the exp-065 precision-secret risk). Exact root
  SHA-256: `e187bfa93b27a8c31f7615e387078bf8692f1b4917b89736282227fa282c5fae`.
  Adopted on the authoritative same-process Modal paired grid: full 15-shape
  geomean **1.0136** CI95 [1.0131, 1.0140] (excludes 1.0), driven by the six
  e62 shapes — `60×1024` **1.0979×** (936.1→852.6us), `8×2048` **1.0478×**,
  `2×4096` **1.0282×**, `2×2048` **1.0153×**, `4×1024` **1.0139×**, `16×512`
  **1.0062×** — with all nine other shapes flat (≤0.23% off-target, inside the
  0.57% A-vs-A floor), **0 new fallbacks**, and identical per-shape residuals
  and counters (byte-identical evidence). See `experiments/069-midshape-ov/` and
  `results/069-fullgrid.json`.
  - Official public/secret geomean is **not exposed by the popcorn CLI** (`Score`
    `-`); adopted on the paired-grid + byte-identity + passing-ranked-runs
    evidence per the exp-067 precedent and owner authorization.

- **Previous ranked winner `#922201`** (exp 067): `done`, all six ranked runs
  (test/benchmark/leaderboard × public+secret) **passed**. One-line enrollment
  of `60×1024` onto the `e62_diag128` path (`_EXP062_SHAPES`, nb_outer=1024)
  over the `#913511` source; every other shape byte-identical. Exact root
  SHA-256: `f108cbba5a586ae67501146fb19e074364a9a4ff6e9b89a4ac78cefd8a62a429`.
  Adopted on the authoritative same-process Modal paired grid (drift-neutralized,
  `program.md` [O4]): geomean **1.0180** CI95 [1.0169, 1.0191] (excludes 1.0),
  driven entirely by `60×1024` **1.3101×** (1252.7→955.7us), all 14 other shapes
  flat (≤0.31% off-target, inside the 0.79% A-vs-A floor), **0 new fallbacks**,
  and the candidate is *more accurate* (`60×1024` residual 3.31 vs 9.33). Fresh
  Popcorn test passed **17/17** on the exact source (all six families).
  - Official public/secret geomean is **not exposed by the popcorn CLI** (`Score`
    `-` on every run); adopted on the paired-grid + passing-ranked-runs evidence
    per explicit owner authorization. Projected from the `#913511` baseline
    (672.383us public / 655.423us secret) scaled by the paired ratio:
    **~660.5us public / ~643.8us secret**. Unlike the `#914341` secret-split
    failure, this is a pure *latency* reorchestration of a well-conditioned
    batched shape (value-independent, equal/better accuracy), so the win carries
    to the secret split. See `experiments/067-adopt-60x1024/` and
    `results/067-adopt-pairedgrid.json`.
  - The diagonal `potrf` is still **59.6% of `1×16384`** and **46.9% of
    `1×32768`** and is a measured wall: cuSOLVER runs 308–340 ns/row and this
    repo's best custom block kernel ended at 296 ns/row. The next lever there is
    named-barrier overlap inside the block kernel (~195 ns/row projected).

## Recent history

- **Previous ranked winner `#913511`** (exp 064): `done`, public geomean
  **672.383us** and secret geomean **655.423us**. Full 15-shape paired grid
  **1.0073** over `#912756`; the two changed shapes were `1×16384` (1.0485×)
  and `1×32768` (1.0679×). Root SHA-256 was
  `8e4603e56432b86be263d74743dd4d52940d043682cfca515a71e69c10a26baa`. See
  `experiments/064-large-two/` and `journal.md` Session 52.
- **Earlier ranked winner `#912756`** (exp 063): public **675.753us**, secret
  **674.448us**; 256-thread panel factorization plus wider mid-shape enrollment
  on the resident diagonal-block kernel. See `experiments/063-diag128-fast/`.
- **Earlier ranked winner `#904546`** (exp 059): `done`, public geomean
  **764.876831us** and secret geomean **785.861426us**, improving the frozen
  `#890798` baseline by **4.6261% public / 7.3098% secret**. Popcorn test
  `#904530` passed 17/17. Exact root SHA-256:
  `f8d67dce5a7a0dd68fc96e24613444970aa8c637b168bcb252cab01f2db89e5a`.
  See `experiments/059-two-large-incremental/` and `journal.md` Session 47.

Older winners are in `docs/experiments.md` (per experiment) and `journal.md`
(narrative). The full history was removed from `README.md`; nothing was lost.

## Pending official score (not yet adopted)

- None. `#926130` was adopted into root (see Current ranked winner above).

## Rejected since the current winner

- **`#914341`** (exp 065, named-barrier overlap): public **646.868us**
  (-3.79%) but secret **692.860us** (+5.71%). Not adopted; root stays on
  `#913511`. See `program.md`'s secret-split section.
