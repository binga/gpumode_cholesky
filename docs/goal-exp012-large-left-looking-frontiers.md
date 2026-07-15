# Goal — experiment 012 large left-looking frontiers

Start from ranked winner `#878273` (`4b4d557`, public `1500.704 us`) and combine
the two positive frontiers recovered from the bounded slow-shape searches:

- `1x16384`: left-looking TF32 active diagonal/panel updates, measured
  `18512.6 -> 15882.0 us` (`1.166x`);
- `1x32768`: left-looking factorization with native Blackwell FP8 panel products,
  measured `72535.4 -> 52349.6 us` (`1.386x`).

The `1x8192` search produced no faster valid path, so that dispatch remains
byte-for-byte on the ranked cuSOLVER implementation.

## Promotion gates

1. Local property, syntax, JSON, diff, snapshot, and forbidden-source checks pass.
2. A paired same-process Modal B200 probe uses rotating inputs, retains all
   outputs through validation, confirms the intended large-shape backends ran
   without fallback, and shows both exact targets faster than `#878273`.
3. Dense, spectrum, diagonal, lowrank, rowscale, and tridiagonal inputs pass for
   both changed shapes, including any exact fallback.
4. The complete 15-shape Modal grid passes with no material off-target regression
   and a lower geometric mean than experiment 009.
5. Popcorn test mode passes 17/17. Only then launch one ranked submission and
   adopt only if the completed public/secret leaderboard scores improve.

The README's Modal upload authorization applies to the bounded experiment and
verification files. No evo workflow is used.
