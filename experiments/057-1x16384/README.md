# Experiment 057 — exact `1×16384` structural search

Status: **active; V2 frontier preserved**.

Baseline is ranked submission `#890798`, commit
`f358e879b1287ca50d29115ad9a403c6bd10a69d`, source SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
Every candidate in this directory changes only the exact
`batch=1,n=16384` dispatch.

## Paired B200 results

| Variant | Architecture | Baseline | Candidate | Speedup | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| V1 | base-256 leaf-batched breadth-first inverse + merged block-column update | 15167.872us | 14336.464us | `1.058135×` | frontier, superseded |
| **V2** | **scalar-leaf trsm-free breadth-first inverse + merged block-column update** | **15286.256us** | **10737.440us** | **`1.423787×`** | **frontier** |
| V3 | FP16-resident factor shadow + V2 inverse | 15140.352us | 10672.192us | `1.418604×` | rejected below V2 |
| **V4** | **custom Triton base-32 inverse leaves + GEMM combines** | **15137.920us** | **10187.200us** | **`1.486074×`** | **validated frontier** |

V2's per-shape 95% interval is `[1.422166,1.424248]`. Both exact-source
paired checks passed the official reconstruction checker. V2 reported one
`_EXP057_V2_HITS` increment and no new fallback on the ranked dense input.

## V2 family envelope

The candidate and exact baseline each pass the official checker on all six
families. V2 stays active without fallback on dense, diagonal, row-scaled, and
tridiagonal inputs. Spectrum and low-rank take V2's explicit safety handoff to
the exact ranked implementation; the exact control also takes its existing
safety path on those same two families. Residuals are identical on both safety
families (`0.107` and `0.000822`).

The raw generic `familygrid` artifacts say `passed=false` because that harness
rejects *all* fallback. The reviewed candidate-vs-control contract in
`variant-02-family-comparison.json` passes: 6/6 official checks, four active V2
families, two incumbent-matched safety families, zero unexpected fallback.

## Closed prior architectures

The search did not repeat measured negative designs: `nb=1024/4096`, one direct
2048 triangular solve per panel, split32 diagonal POTRF, cuSOLVER-free
Triton/CUDA diagonal POTRF, dynamic FP8 panels at 16384, BF16x9, and the
right-looking FP16/BF16 trailing path. The preserved experiment-052 V1 design
was first narrowed to this exact shape and reproduced before removing its
remaining triangular solves in V2.

V2 remains short of the campaign's approximately `2.204×` balanced per-shape
requirement. V3's FP16 shadow passed dense correctness but its paired ratio was
significantly below V2, closing that precision/storage combination. The search
then used V2's component profile to target 122 elementwise inverse kernels:
V4 replaces the first five scalar/GEMM inverse levels with one parallel Triton
base-32 leaf launch per panel and improves the frontier to `1.486074×`. Its
six-family envelope matches V2: 6/6 official checks, four active fast-path
families with seven Triton leaf hits each, and the same two incumbent-matched
safety families.
