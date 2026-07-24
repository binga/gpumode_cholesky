# Experiment 058 — `1×32768`

Status at V4 checkpoint: **promotable frontier for integration**.

The exact ranked `#890798` source at SHA-256 `fd3072b5…` recursively inverts
each 4096-wide diagonal factor and serializes 112 cuBLAS TRSM launches. V1
replaces only the `batch=1,n=32768` inverse with a breadth-first tree:
independent 256-wide diagonal leaves are inverted in one batched TRSM, and
successive levels are assembled with batched GEMMs. V4 additionally moves the
seven panel-apply GEMMs to FP16 inputs with FP32 accumulation/output. FP16 and
TF32 have the same 10-bit significand; the wider FP16 exponent avoided V3's
E4M3 failure. MXFP8 left-looking updates, the 4096 schedule, official checker,
and all other dispatches remain unchanged.

| Shape | Control latency (us) | Current latency (us) | Speedup |
|---|---:|---:|---:|
| `1×32768` | 42,762.352 | 31,473.087 | `1.358510×` |

The same-process B200 CI95 is `[1.356413, 1.359330]`. Dense reconstruction
residual was `5.28` under the official checker. `_BLOCKED_INV_32768_HITS=1`,
`_FP16_SOLVE_32768_HITS=7`, `_MXFP8_HITS=6`, and no new fallback occurred in
the paired gate.

Both V1 and the byte-exact control pass the official checker on all six
families. Their fallback patterns are identical: dense, diagonal, and
tridiagonal complete on the primary large path; spectrum, low-rank, and
row-scaled use the incumbent's existing safety chain. V1 therefore introduces
no safety regression.

V2's 2048 schedule and V3's MXFP8 panel application both became non-finite on
dense input and were rejected without using their fallback-contaminated timing.
V4 remains below the aspirational `2.00×` checkpoint and the two-shape campaign
product requirement, so the bounded search continues from this protected
frontier.
