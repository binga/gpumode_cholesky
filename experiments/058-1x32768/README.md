# Experiment 058 — `1×32768`

Status at V1 checkpoint: **promotable frontier for integration**.

The exact ranked `#890798` source at SHA-256 `fd3072b5…` recursively inverts
each 4096-wide diagonal factor and serializes 112 cuBLAS TRSM launches. V1
replaces only the `batch=1,n=32768` inverse with a breadth-first tree:
independent 256-wide diagonal leaves are inverted in one batched TRSM, and
successive levels are assembled with batched GEMMs. MXFP8 updates, the
4096-wide panel schedule, official checker, and all other dispatches are
unchanged.

| Shape | Control latency (us) | Current latency (us) | Speedup |
|---|---:|---:|---:|
| `1×32768` | 42,769.024 | 33,043.633 | `1.294300×` |

The same-process B200 CI95 is `[1.293613, 1.294789]`. Dense reconstruction
residual was `5.28` under the official checker. `_BLOCKED_INV_32768_HITS=1`,
`_MXFP8_HITS=6`, and no new fallback occurred in the paired gate.

Both V1 and the byte-exact control pass the official checker on all six
families. Their fallback patterns are identical: dense, diagonal, and
tridiagonal complete on the primary large path; spectrum, low-rank, and
row-scaled use the incumbent's existing safety chain. V1 therefore introduces
no safety regression.

This is below the aspirational `2.00×` shape checkpoint and well below the
two-shape campaign product requirement. The next structural variant reopens
the `n=32768` panel-width tradeoff at `nb=2048` only after the blocked inverse
has removed the old schedule's dominant serialized-inverse cost.
