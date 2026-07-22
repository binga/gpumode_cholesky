# Experiment 050 compile-budget rescue

Exact ranked control is `#890798`, commit `f90ef90`, SHA-256
`fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1`.
The frozen campaign target is a 2x public-geomean improvement:
`801.977179us -> 400.988590us` or lower.

The recovered `integrated-v8.py` source is byte-for-byte the source stored by
Popcorn for submissions `#893978`, `#893983`, and `#893995` (the CLI adds one
display newline; the corresponding extracted hash is recorded in `state.json`).
It is a real promotable frontier:

| Shape | Control latency (us) | Current latency (us) | Speedup |
|---|---:|---:|---:|
| `4x1024` | 690.388 | 490.732 | 1.40859x |

The same-process 15-shape paired grid improves 1.022989x with CI95
`[1.022574, 1.023405]`, and all 15 dense rows pass. Popcorn test `#893978`
passed. Ranked attempt `#893983` completed its secret split at `780.334us`
(better than incumbent secret `847.836us`) but its public validation hit the
six-minute boundary. The exact-source retry `#893995` timed out on both test
splits. This is classified as `COMPILE_BUDGET_FAILURE`, not numerical failure.

The family grid passes the unchanged official checker on all rows. The new
cooperative route actively handles dense, diagonal, rowscale, and tridiagonal;
its numerical safety gate falls back to the ranked path on spectrum and
lowrank. Those fallbacks are preserved as safety behavior and must be proved
again for the compile-fix candidate.

V10 changes only cold-build cost: remove diagnostic-only timing plumbing and
reduce compile-heavy generated code while preserving the cooperative kernel's
arithmetic, dispatch, safety fallback, and output contract. Performance and
correctness claims remain unproven until the affected B200 gates rerun.

## V10 checkpoint

Fresh B200 import/build finished in `88.029s`; total runner initialization was
`91.099s`, well below the program's `288s` cold-build threshold. Paired
same-process evidence then proved the intended route executed with
`_COOP1024_HITS=1`, no new fallback, and a tighter target win:

| Shape | Control latency (us) | Current latency (us) | Speedup |
|---|---:|---:|---:|
| `4x1024` | 713.404 | 493.540 | 1.44489x |

The unchanged `60x1024` row measured `1209.216us -> 1214.640us` (0.99570x),
inside the 3% per-case regression guardrail. V10 remains a frontier pending
the six-family safety gate.

The six-family artifact is byte-identical to V8's family artifact. All 12
`4x1024` and `60x1024` rows pass the unchanged official checker. The
cooperative route is active on dense, diagonal, rowscale, and tridiagonal;
its generic finite-diagonal safety check sends spectrum and lowrank to the
exact ranked fallback. No fallback latency is used as candidate evidence.

The full 15-shape paired grid passes with aggregate speedup `1.023422x`,
CI95 `[1.022863, 1.023981]`, and 15/15 correct rows. The target row is
`679.376us -> 481.032us` (`1.412483x`); all 14 off-target rows are within
0.11% of parity.

Popcorn test `#896863` nevertheless failed at the exact six-minute service
boundary (`360.067s`). It emitted no arithmetic error and is classified as a
compile-budget failure. V10 is not rankable. V11 will lower only the combined
CUDA128/cooperative translation unit from `-O3` to `-O2`; performance and cold
build must be remeasured before another official test.

V11 changes only that module's optimization level to `-O2`. Fresh import fell
to `79.625s` (9.5% below V10), while paired target latency remained a strong
`695.416us -> 495.028us` (`1.40513x`) with active backend and zero fallback.
This is a valid compile/performance frontier but not yet enough margin to spend
another official test; V12 tests `-O1` next.

## V12-V16 compile decomposition and rescue

Lowering CUDA128 to `-O1` did not improve the frontier: V12 imported in
`82.135s`, slower than V11. Adding `--threads=4` was actively harmful: V13
needed `120.233s`. V14 partially unrolled the template-instantiated rank-32
trailing update, but the per-extension profiler showed the underlying issue:
CUDA32 took `26.405s`, CUDA64 `26.208s`, and CUDA128 plus cooperative code
`27.851s`. Fixed NVCC/PyBind startup dominated generated-code optimization.
A separate clean incumbent sample made the same point at a slower worker
allocation: `51.769s`, `51.326s`, and `58.827s` respectively.

V15 consolidated the three eager extensions, but the strengthened cold-import
gate correctly rejected it: CUDA64 and CUDA128 both declared a translation-unit
constant named `N`. V16 makes the minimal source-level repair (`N` -> `N64` in
the unchanged CUDA64 source) and preserves every kernel, launch, dispatch, and
`-O3` setting in one extension. Its clean B200 build is valid:

- one combined extension build: `56.157s`;
- submission import: `56.583s`;
- `_CUDA32`, `_CUDA64`, and `_CUDA128` all ready;
- no extension load errors.

Paired target evidence proves the intended cooperative route, not a fallback:

| Shape | Control latency (us) | Current latency (us) | Speedup |
|---|---:|---:|---:|
| `4x1024` | 685.392 | 473.692 | 1.44757x |
| `60x1024` | 1128.832 | 1128.260 | 1.00073x |

The `4x1024` CI95 is `[1.445987, 1.448390]`, `_COOP1024_HITS=1`, and there
are no new fallbacks. The repackaged ranked CUDA32/64/128 rows all pass with
their intended backend counters; their aggregate ratio is `0.999419x` and the
largest deviation is `0.278%`, within noise.

All 12 six-family rows at `4x1024` and `60x1024` pass the unchanged official
checker with no extension errors. The only fallbacks reproduce shipped safety
behavior: cooperative spectrum and lowrank at `4x1024`, plus the ranked fused
CTA lowrank fallback at `60x1024`. No fallback timing is used as candidate
evidence. A fresh pre-promotion fetch still resolves `origin/main` to
`f90ef909`, and its `submission.py` still hashes to the frozen incumbent
`fd3072b5...4244c1`.

The exact V16 source then passed the full 15-shape paired grid: 15/15 official
checker passes, aggregate `1.021085x` with CI95 `[1.020413, 1.021757]`, and
`4x1024` at `715.120us -> 509.536us` (`1.403473x`) with the cooperative
backend active. Every off-target row remains inside the 3% guardrail. The
smallest rows showed `0.9876x` (32) and `0.9857x` (64) in this long run, while
the dedicated affected-shape gate was at parity; both are retained as raw
evidence rather than hidden. Final Python syntax, source-policy, protected
evaluator diff, source hash, and `kernel-audit.json` contract validation pass.
V16 is qualified for exactly one Popcorn test-mode submission.

Popcorn test `#897104` passed all 17/17 rows in `64.410s`, confirming that the
combined extension fixed the service compile budget. The unchanged exact source
was then submitted once as ranked `#897112`. Both scored splits completed, but
the result is asymmetric:

- public: `812.134577us`, a 1.27% regression from incumbent `801.977179us`;
- secret: `780.509815us`, a 7.94% improvement from incumbent `847.836us`.

The candidate is therefore rejected by the two-sided adoption rule despite its
secret improvement. Root `submission.py` remains the exact `#890798` source;
V16 must not be reranked unchanged. Its secret-positive result remains useful
frontier evidence for a future candidate with a larger and more robust public
margin.
