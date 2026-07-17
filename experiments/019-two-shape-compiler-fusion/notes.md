# Experiment 019 — two-shape compiler and precision probe

**Status: ADOPTED — ranked winner `#882825`.** The search began with
`4x1024` and `8x2048` from exact ranked exp-017 source
(`5f29c6a15241a62f7a34e2580070e057777ca96c4a95ed64a245908b753d9a56`),
then promoted the strongest result through all affected shapes and the full
leaderboard grid. Exact ranked/adopted SHA-256:
`ad8bce6fdc3d037dbdc91912ddfec802d5eea844a4b6e18e4cc8552c45f66dcd`.

## Current-path profile

Fresh B200 component profile of the final exp-017 source:

| shape | end-to-end | micro | panel inner | panel apply | trailing |
|---|---:|---:|---:|---:|---:|
| 4x1024 | 827.9us | 437.0us | 156.6us | 122.7us | 71.5us |
| 8x2048 | 2396.4us | 879.9us | 326.2us | 258.5us | 426.3us |

The 8x2048 end-to-end number varies materially across sandbox runs; all final
claims below use alternating-order same-process paired rounds.

Artifact capture succeeded for the exact B200 (`sm100`, Triton 3.7.1)
specializations. Baseline micro PTX contains four `sqrt.approx` + reciprocal
`div.full` instructions and four additional inverse-row `div.full`
instructions per rank-4 loop iteration. No PTX local-memory operations appear
in the micro, but `cuobjdump` reports 236 registers for the divide-to-multiply
variant. The panel-inner kernel is at the 255-register ceiling with a 408-byte
stack; SASS contains 122 `LDL`/`STL` instructions. This is the clearest new
compiler-level bottleneck.

Artifacts: `baseline-current-profile.json`, `baseline-artifact-probe.json`,
`artifacts-baseline.zip`.

## Variant 1 — inverse-row divide to multiply

Reuses the already-computed `inv=1/sqrt(d)` and replaces four inverse-row
divisions with multiplies. PTX confirms the four late `div.full` instructions
are gone.

| shape | baseline | candidate | paired speedup |
|---|---:|---:|---:|
| 4x1024 | 857.29us | 851.35us | **1.00699x** |
| 8x2048 | 2042.98us | 2033.27us | **1.00478x** |

All six families pass on both shapes. Classification: small positive frontier.

Artifacts: `candidate-divmul.py`, `divmul-probe.json`,
`divmul-interleaved.json`, `artifacts-divmul.zip`.

## Variant 2 — constexpr panel-inner width

`WIDTH` is specialized to 32/64/96. It adds no reliable gain over the divide
rewrite:

| shape | baseline | candidate | paired speedup |
|---|---:|---:|---:|
| 4x1024 | 855.22us | 852.97us | **1.00264x** |
| 8x2048 | 2039.28us | 2026.71us | **1.00620x** |

All six families passed in the first run. Artifact evidence explains the weak
result: every width specialization remains at 255 registers and a 408-byte
stack. Classification: rejected as a standalone compiler hint; it does not
address the spill bottleneck.

Artifacts: `candidate-static.py`, `static-width-probe.json`,
`static-width-interleaved.json`, `artifacts-static-width.zip`.

## Variant 3 — FP16 trailing operands with FP32 accumulation

Starting from the positive divide rewrite, `_trailing_nb` casts loaded factor
tiles to FP16 in registers, executes `tcgen05.mma ... kind::f16`, and retains
FP32 accumulation and output.

| shape | baseline | candidate | paired speedup |
|---|---:|---:|---:|
| 4x1024 | 821.62us | 810.61us | **1.01359x** |
| 8x2048 | 1992.74us | 1958.13us | **1.01767x** |

Component trailing time was 66.5us at 4x1024 and 403.2us at 8x2048 versus
fresh baseline-profile values of 71.5us and 426.3us. Dense scaled residuals
are 1.35-1.43/20 at 4x1024 and 0.73-0.737/20 at 8x2048. All six families pass
on both shapes; lowrank takes the same expected safety fallback as the shipped
path. The compiled trailing kernel remains register-limited (255 registers)
and its stack grows from 88 to 112 bytes, but faster FP16 tensor instructions
still produce a net win.

Classification: strongest frontier. It was cleaned of artifact instrumentation
and promoted to `candidate-final.py`. Five profitable shapes use the FP16
specialization; `60x1024` gets a compile-time false signal that preserves the
ranked TF32/divide specialization exactly.

Artifacts: `candidate-fp16-trailing.py`, `fp16-trailing-probe.json`,
`fp16-trailing-interleaved.json`, `artifacts-fp16-trailing.zip`.

## Promotion and ranked result

The clean final candidate passed six families on every affected shape twice
(36/36 each sweep). The final exact-baseline `60x1024` route passed another
6/6. Its paired 15-shape grid passed 15/15 and improved geometric mean
1174.1us -> 1163.3us (**1.0093x**). The five intentional routes gained
1.5-4.9%; unchanged routes were timing-flat.

Local properties passed 10/10. Popcorn test `#882824` passed **17/17**. Exactly
one ranked job was launched: `#882825` passed public and secret test, benchmark,
and leaderboard stages, scoring **1122.570us public / 1128.511us secret** versus
`#882706` at 1205.336us / 1197.791us. This is a **6.867% public / 5.784% secret**
improvement. The exact submitted source is adopted at root `submission.py`.

Artifacts: `fp16-final-six-shapes.json`,
`fp16-static-gated-six-shapes.json`, `final-60x1024-baseline-route.json`,
`final-fullgrid.json`, `test-882824.json`, `ranked-882825.json`, and
`candidate-final.py`.

## Assessment / next move

The ordered compiler/micro pass produced a substantial ranked win despite a
modest paired-grid forecast. The next compiler experiment should target the measured
panel-inner spill problem (smaller `TILE_R`, epilogue subtiling, or staged
apply+inner fusion), rather than specializing more scalar offsets. Before any
integration, restrict the FP16 path to these exact shapes or verify all six
split32 shapes, then run a full-grid paired benchmark.

One harness defect was fixed during the work: large single-line artifact
transport was truncated by Modal stdout. Artifact bundles now use bounded
chunks and omit redundant source/LLVM/SASS copies for later variants while
retaining TTIR, TTGIR, PTX, cubin, and resource usage.
