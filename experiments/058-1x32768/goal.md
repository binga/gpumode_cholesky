# Experiment 058 — `1×32768`

Optimize only `batch=1,n=32768` from ranked submission `#890798` at commit
`f358e879b1287ca50d29115ad9a403c6bd10a69d` and source SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.

Research target: at least `2.00×` paired speedup. Preserve the official
correctness threshold, require positive backend/fallback evidence, and measure
at most six materially distinct serious variants. Do not edit any other shape
dispatch. Popcorn and integration remain orchestrator-owned.
