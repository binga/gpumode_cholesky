# Status — what is ranked right now

**Loop stage [O1] synchronize.** The single maintained answer to "what is the
incumbent". Update this on adoption; do not restate scores anywhere else.

> Verify against `popcorn submissions list` and by `shasum -a 256` before
> spending — this file can lag, the leaderboard cannot.

## Current ranked winner

- **Current ranked winner `#926462`** (exp 070): `done`, both ranked splits
  (public+secret) **passed** (test `#926455` 17/17). **Official geomeans:
  630.403us public / 670.301us secret.** The public score was verified on the
  gpumode.com board (rank #31), up from #32 /
  646.868us** (kdpisda fell to #32 at 632.306us). Exact root SHA-256:
  `582cde1648b8b3e9d77a36173dd59cd36588123ae28800ca00e5342b869ff723`.
  - **What it is: the public-LB stack.** Layers exp 065's **named-barrier
    overlap** (VAR=4) in the e62 128×128 diagonal block kernel onto the current
    root, which already carried exp 067 (`60×1024` e62 enroll) and exp 069
    (write-only upper mask). The three touch disjoint code paths, so they
    compose. Authoritative same-process Modal paired grid vs `#926130`: full
    15-shape geomean **1.0171** CI95 [1.0167, 1.0176] (excludes 1.0), driven by
    the six e62 shapes — `2×2048` **1.0507×**, `2×4096` **1.0474×**, `4×1024`
    **1.0455×**, `16×512` **1.0393×**, `8×2048` **1.0393×**, `60×1024`
    **1.0369×** — with all nine other shapes flat (≤0.06%, incl. the untouched
    large shapes), **0 new fallbacks**, identical per-shape counters and
    accuracy. See `experiments/070-lb-stack/` and `results/070-fullgrid.json`.
  - **Promotion rule (owner-selected 2026-07-29): optimize public, ACCEPT
    secret.** exp 065's overlap regressed the secret split (+5.71% on `#914341`),
    so this winner is **not secret-safe**. It was adopted because the live board
    ranks by *public* and the competition closed the same day; the owner chose
    the visible rank over the secret split. `#926130` remains the **secret-safe
    fallback** (revert to it if final standings recompute on secret). This
    reverses the exp-065 rejection under the named rule — see `program.md`'s
    secret-split section.

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

## Secret-safe fallback (not the live ranked winner)

- **`#926130`** (exp 069): public ~651us / secret ~643us, byte-identical to
  `#922201` on all shapes. This is the last **secret-safe** incumbent; revert
  root to its SHA-256
  `e187bfa93b27a8c31f7615e387078bf8692f1b4917b89736282227fa282c5fae` if the goal
  returns to protecting the secret split (e.g. final standings recompute on
  secret). See `experiments/069-midshape-ov/`.

## Note on exp 065

- **`#914341`** (exp 065, named-barrier overlap): public **646.868us** (-3.79%)
  but secret **692.860us** (+5.71%). Originally rejected (secret). Its overlap
  kernel was **re-adopted inside exp 070** under the owner's public-optimization
  rule — the public win it delivers is what set our board rank. See
  `experiments/070-lb-stack/` and `program.md`'s secret-split section.
