# Goal — Experiment 044: mid-shape split32 diagonal chain

## Frozen control

Exact ranked `#888996`, commit `ec33b31`, SHA-256
`5bbb8e8de3eedfadfb70fdcfaa902723fab03c2998bc18e9cb80512e82cce80c`
(`experiments/044-midshape-2x/baseline-888996.py`).

`64x256` was deliberately excluded: a concurrent session held that shape
(experiment 043) throughout this run.

## Constituent diagnosis (`shapediag`, B200)

| shape | wall | device | idle | launches | dominant kernel |
|---|---|---|---|---|---|
| 16x512  | 398.5us | 376.4us | 22.1us (5.6%)  | 54  | `_micro_potrf_gj32` 217.0us / 16 calls = **57.7%** |
| 640x512 | 1550.0us | 1447.1us | 102.9us (6.6%) | 53  | `_panel_inner32_subtile64` 433.7us = 30.0% |
| 4x1024  | 805.9us | 692.6us | 113.3us (14.1%) | 102 | `_micro_potrf_gj32` 433.6us / 32 calls = **62.6%** |
| 60x1024 | 1813.8us | 1230.4us | 583.5us (32.2%) | 100 | `_micro_potrf_gj32` 435.3us = 35.4% |
| 2x2048  | 1362.8us | 1351.6us | 11.1us (0.8%) | 13 | cuSOLVER `getrf_wo_pivot` 1233.3us / 2 calls = **91.2%** |
| 8x2048  | 1828.4us | 1668.2us | 160.2us (8.8%) | 198 | `_micro_potrf_gj32` 869.3us / 64 calls = **52.1%** |

`_micro_potrf_gj32` costs **13.55us per launch independent of batch**
(13.563 / 13.548 / 13.583us at batch 16 / 4 / 8). It is not bandwidth,
arithmetic or launch bound: it is a 32-step serial pivot chain paying one
Triton block rendezvous per pivot, and it is the single largest constituent of
three of the six shapes.

## Hypothesis

Replace that chain with a warp-synchronous CUDA kernel keeping the same
contract (factor the 32x32 block at `(k, k)` in place, publish `L^-1` into
`dinv`), so the surrounding split32 schedule is untouched.

## Gates

Free property checks, paired same-process B200 probes against the exact ranked
source, all six input families over every changed shape, the full 15-shape
paired grid, Popcorn test 17/17, then one ranked submission.
