# Status — what is ranked right now

**Loop stage [O1] synchronize.** The single maintained answer to "what is the
incumbent". Update this on adoption; do not restate scores anywhere else.

> Verify against `popcorn submissions list` and by `shasum -a 256` before
> spending — this file can lag, the leaderboard cannot.

## Current ranked winner

- **Current ranked winner `#913511`** (exp 064): `done`, public geomean
  **672.383us** and secret geomean **655.423us**, improving the previous
  `#912756` winner by **0.499% public / 2.821% secret**. Popcorn test
  `#913422` passed on the exact source. Exact root SHA-256:
  `8e4603e56432b86be263d74743dd4d52940d043682cfca515a71e69c10a26baa`.
  Full 15-shape paired grid **1.0073** CI95 [1.0068, 1.0079]; the two changed
  shapes are `1×16384` (1.0485×) and `1×32768` (1.0679×), everything else flat.
  See `experiments/064-large-two/` and `journal.md` Session 52.
  - The diagonal `potrf` is now **59.6% of `1×16384`** and **46.9% of
    `1×32768`** and is a measured wall: cuSOLVER runs 308–340 ns/row and this
    repo's best custom block kernel ended at 296 ns/row. The next lever there is
    named-barrier overlap inside the block kernel (~195 ns/row projected).

## Recent history

- **Previous ranked winner `#912756`** (exp 063): public **675.753us**, secret
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

## Rejected since the current winner

- **`#914341`** (exp 065, named-barrier overlap): public **646.868us**
  (-3.79%) but secret **692.860us** (+5.71%). Not adopted; root stays on
  `#913511`. See `program.md`'s secret-split section.
