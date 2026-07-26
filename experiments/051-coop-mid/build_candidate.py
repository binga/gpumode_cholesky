"""Build experiment 051 candidates from the exact ranked baseline.

Every transformation is an exact-substring replacement against
`baseline-890798.py` (sha256 fd3072b5...4244c1) so the diff stays auditable and
the script fails loudly if the baseline ever moves.

Two independent transformations:

  MERGE  - collapse the four `load_inline` extensions into one. Experiment 050
           proved the popcorn build cache is keyed by extension name and that a
           cold four-extension build exceeds the 360s service timeout, while the
           merged build passes 17/17 in 36s. Experiment 050 V6 also measured the
           merge itself neutral on all four CUDA shapes (ratios 0.9996-1.0040).
  COOP   - add the repaired cooperative tile-32 kernel and route the enrolled
           tiny-batch mid shapes to it behind a finiteness gate.

Usage: python build_candidate.py <variant> <output.py>
"""

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline-890798.py"
BASELINE_SHA = "fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1"

# variant -> (kernel source file, enrolled shapes, extra CUDA source edits)
VARIANTS = {
    "v1": (
        "coop_kernel.cu",
        "{(4, 1024), (16, 512), (8, 2048), (2, 2048), (1, 4096), (2, 4096)}",
        (),
    ),
    # v1b isolates one question the v1 grid could not answer: is the loss from
    # the trailing update itself, or from grid-barrier cost scaling with the CTA
    # count? v1 runs ~148 CTAs per matrix where experiment 048 V2 ran exactly 32
    # and measured 1.167x on 4x1024. Same kernel, V2's CTA count.
    "v1b": (
        "coop_kernel.cu",
        "{(4, 1024), (16, 512), (8, 2048), (2, 2048), (1, 4096), (2, 4096)}",
        (("    if (ctas > 256) ctas = 256;", "    if (ctas > 32) ctas = 32;"),),
    ),
    # v2: two-level NB=128 blocking. The rank-32 trailing update of v1 is capped
    # near 28 TFLOP/s by arithmetic intensity alone; a rank-128 update staged
    # through shared memory as 64x64 blocks reaches ~16 flops/byte.
    "v2": (
        "coop_kernel_v2.cu",
        "{(4, 1024), (16, 512), (8, 2048), (2, 2048), (1, 4096), (2, 4096)}",
        (),
    ),
    # v2b: v1b proved co-resident CTA count is the binding constraint, not
    # barrier cost (pinning it to 32 per matrix cost 1x4096 a further 2.8x). The
    # arbitrary 256 cap therefore truncates exactly the batch-1 shapes, which
    # can use the whole occupancy-derived grid. Let the occupancy query decide.
    "v2b": (
        "coop_kernel_v2.cu",
        "{(4, 1024), (16, 512), (8, 2048), (2, 2048), (1, 4096), (2, 4096)}",
        (("constexpr int COOP_MAX_CTAS = 256;", "constexpr int COOP_MAX_CTAS = 4096;"),),
    ),
    # v2p: v2 instrumented with %globaltimer phase accumulators. v1 and v2 make
    # contradictory predictions and both lose, so the next spend has to buy
    # attribution, not another architecture guess. Exposes the entry points the
    # existing `coopphase` runner mode expects.
    "v2p": (
        "coop_kernel_v2p.cu",
        "{(4, 1024), (16, 512), (8, 2048), (2, 2048), (1, 4096), (2, 4096)}",
        (),
    ),
}

# Variants exposing the phase-probe entry point must declare it to pybind.
PROBE_VARIANTS = {"v2p"}


def require_replace(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"anchor {label!r} matched {count} times, expected 1")
    return text.replace(old, new)


# --------------------------------------------------------------------------
# MERGE: one extension for every CUDA kernel.
# --------------------------------------------------------------------------

LOAD32 = '''
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA32 = load_inline(
            name="chol32_exp039_final",
            cpp_sources="void chol32_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA32_SOURCE,
            functions=["chol32_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA32_ERROR = repr(exc)
'''

LOAD64 = '''
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA64 = load_inline(
            name="chol64_exp041_v3_final",
            cpp_sources="void chol64_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA64_SOURCE,
            functions=["chol64_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA64_ERROR = repr(exc)
'''

LOAD128 = '''
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA128 = load_inline(
            name="chol128_exp042_v5_with_exp044_micro",
            cpp_sources=(
                "void chol128_launch(torch::Tensor, torch::Tensor);\\n"
                "void micro32_launch(torch::Tensor, torch::Tensor, "
                "torch::Tensor, int64_t, int64_t, int64_t);"
            ),
            cuda_sources=_CUDA128_SOURCE,
            functions=["chol128_launch", "micro32_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA128_ERROR = repr(exc)
'''

LOAD256 = '''
def _load_cuda256() -> None:
    global _CUDA256, _CUDA256_ERROR
    if _CUDA256 is not None or _CUDA256_ERROR is not None:
        return
    if not torch.cuda.is_available():
        return
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA256 = load_inline(
            name="chol256_exp043_v35_scalar_accurate",
            cpp_sources="void chol256_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA256_SOURCE,
            functions=["chol256_launch"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA256_ERROR = repr(exc)
'''

MERGED_LOADER = '''
# ---------------------------------------------------------------------------
# Experiment 051 (from experiment 050 L5): ONE extension for every CUDA kernel.
#
# The official runner keeps a build cache keyed by extension name, so the real
# budget is the COLD build: a probe of the exact ranked source with nothing
# changed but the extension names hit the 360s service timeout (#898675) while
# the merged build passes 17/17 in 36s (#898689). Four `load_inline` calls means
# compiling the expensive `torch/extension.h` pybind glue four times.
#
# `_CUDA64_SOURCE`'s `N` was renamed `N64` to clear the only symbol collision;
# the experiment-051 kernel keeps its own symbols in `namespace coop051`.
_CUDA_ALL_SOURCE = (
    _CUDA32_SOURCE + _CUDA64_SOURCE + _CUDA128_SOURCE + _CUDA256_SOURCE
    + _COOP_SOURCE
)

_CUDA_ALL = None
_CUDA_ALL_ERROR = None

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA_ALL = load_inline(
            name="chol_all_exp051_v1",
            cpp_sources=(
                "void chol32_launch(torch::Tensor, torch::Tensor);\\n"
                "void chol64_launch(torch::Tensor, torch::Tensor);\\n"
                "void chol128_launch(torch::Tensor, torch::Tensor);\\n"
                "void chol256_launch(torch::Tensor, torch::Tensor);\\n"
                "void micro32_launch(torch::Tensor, torch::Tensor, "
                "torch::Tensor, int64_t, int64_t, int64_t);\\n"
                "void coop051_launch(torch::Tensor);\\n"
                "int64_t coop051_ctas(int64_t, int64_t);"__PROBE_DECL__
            ),
            cuda_sources=_CUDA_ALL_SOURCE,
            functions=[
                "chol32_launch",
                "chol64_launch",
                "chol128_launch",
                "chol256_launch",
                "micro32_launch",
                "coop051_launch",
                "coop051_ctas",__PROBE_FUNC__
            ],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA_ALL_ERROR = repr(exc)

_CUDA32 = _CUDA_ALL
_CUDA64 = _CUDA_ALL
_CUDA128 = _CUDA_ALL
_CUDA256 = _CUDA_ALL
_CUDA32_ERROR = _CUDA_ALL_ERROR
_CUDA64_ERROR = _CUDA_ALL_ERROR
_CUDA128_ERROR = _CUDA_ALL_ERROR
_CUDA256_ERROR = _CUDA_ALL_ERROR


def _load_cuda256() -> None:
    """The merged extension is already resident; kept for call-site parity."""
    return
'''

COOP_GLUE = '''
# The `coopphase` runner mode reads these two module attributes.
_COOP4096 = _CUDA_ALL
_COOP4096_ERROR = _CUDA_ALL_ERROR
_COOP_HITS = 0
_COOP_FALLBACKS = 0
_COOP_ERROR = None
_COOP_SHAPES = __COOP_SHAPES__


def _coop_factor(data: torch.Tensor) -> torch.Tensor:
    """Single-launch cooperative Cholesky. The kernel factors in place, so the
    caller owns the clone; on a non-finite result the caller keeps `data` and
    falls through to the ranked dispatch."""
    out = data.clone()
    _CUDA_ALL.coop051_launch(out)
    return out
'''

COOP_DISPATCH = '''    # Experiment 051: single-launch cooperative tile-32 factorization for the
    # tiny-batch mid shapes, where the ranked split32 chain is dependency bound
    # rather than throughput bound. One launch replaces 54-198 of them, so the
    # eager-mode launch tax that sank experiment 050 does not apply. Any
    # non-finite diagonal (ill-conditioned low-rank families) falls through to
    # the exact ranked dispatch below.
    if (
        is_f32_cuda
        and _CUDA_ALL is not None
        and (batch, n) in _COOP_SHAPES
        and data.is_contiguous()
    ):
        try:
            l = _coop_factor(data)
            if torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all().item():
                _COOP_HITS += 1
                return l
            _COOP_FALLBACKS += 1
        except Exception as exc:
            _COOP_ERROR = repr(exc)
            _COOP_FALLBACKS += 1

'''

DISPATCH_ANCHOR = """    # Experiment 015 round 4: two-level blocked tensor-core potrf with
    # per-shape graph replay for the mid shapes."""

GLOBALS_ANCHOR = """    global _LEFT_32768_ERROR, _LEFT_LARGE_FALLBACKS
    global _LARGE_FP8_HITS, _LARGE_FP8_FALLBACKS, _LARGE_FP8_ERROR
    global _FUSED_CTA_HITS, _FUSED_CTA_FALLBACKS, _FUSED_CTA_ERROR
"""


def build(variant: str) -> str:
    text = BASELINE.read_text()
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if digest != BASELINE_SHA:
        raise SystemExit(f"baseline sha256 {digest} != expected {BASELINE_SHA}")

    # -- MERGE -------------------------------------------------------------
    # `N` is declared by both _CUDA64_SOURCE (64) and _CUDA128_SOURCE (128), so
    # rename it inside the n=64 source only. Scope the edit to that literal
    # rather than the whole file: `blockIdx.x * N * N` appears in both.
    head, marker, rest = text.partition('_CUDA64_SOURCE = r"""')
    if not marker:
        raise SystemExit("could not find _CUDA64_SOURCE")
    body, end, tail = rest.partition('"""')
    if not end:
        raise SystemExit("could not find the end of _CUDA64_SOURCE")
    body = require_replace(body, "constexpr int N = 64;", "constexpr int N64 = 64;", "N64 decl")
    body = require_replace(
        body,
        "    const size_t base = (size_t)blockIdx.x * N * N;",
        "    const size_t base = (size_t)blockIdx.x * N64 * N64;",
        "N64 base",
    )
    if body.count("    for (int linear = row; linear < N * N; linear += 64) {") != 2:
        raise SystemExit("expected exactly 2 N*N loop headers in _CUDA64_SOURCE")
    body = body.replace(
        "    for (int linear = row; linear < N * N; linear += 64) {",
        "    for (int linear = row; linear < N64 * N64; linear += 64) {",
    )
    if "\bN\b" in body:
        raise SystemExit("unexpected")
    text = head + marker + body + end + tail

    for label, block in (("load32", LOAD32), ("load64", LOAD64), ("load128", LOAD128)):
        text = require_replace(text, block, "\n", label)

    kernel_file, shapes, cuda_edits = VARIANTS[variant]
    coop_source = (HERE / kernel_file).read_text()
    for old, new in cuda_edits:
        coop_source = require_replace(coop_source, old, new, f"{variant} cuda edit")
    loader = MERGED_LOADER.replace("chol_all_exp051_v1", f"chol_all_exp051_{variant}")
    if variant in PROBE_VARIANTS:
        loader = loader.replace(
            "__PROBE_DECL__",
            '\n                "\\ntorch::Tensor chol4096_profile(torch::Tensor);"')
        loader = loader.replace("__PROBE_FUNC__", '\n                "chol4096_profile",')
    else:
        loader = loader.replace("__PROBE_DECL__", "").replace("__PROBE_FUNC__", "")
    coop_block = '_COOP_SOURCE = r"""\n' + coop_source + '"""\n' + loader
    text = require_replace(text, LOAD256, coop_block, "load256 -> merged")

    # -- COOP --------------------------------------------------------------
    glue = COOP_GLUE.replace("__COOP_SHAPES__", shapes)
    text = require_replace(
        text,
        "\ndef custom_kernel(data: input_t) -> output_t:\n",
        glue + "\ndef custom_kernel(data: input_t) -> output_t:\n",
        "custom_kernel glue",
    )
    text = require_replace(
        text,
        GLOBALS_ANCHOR,
        GLOBALS_ANCHOR + "    global _COOP_HITS, _COOP_FALLBACKS, _COOP_ERROR\n",
        "custom_kernel globals",
    )
    text = require_replace(
        text, DISPATCH_ANCHOR, COOP_DISPATCH + DISPATCH_ANCHOR, "coop dispatch"
    )
    return text


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    variant, out = sys.argv[1], Path(sys.argv[2])
    text = build(variant)
    out.write_text(text)
    compile(text, str(out), "exec")
    print(f"wrote {out} ({len(text.splitlines())} lines, sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
