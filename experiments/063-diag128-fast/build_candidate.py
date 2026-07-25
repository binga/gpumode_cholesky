"""Build an experiment-062 candidate from the exact ranked baseline.

Only exact-substring edits are applied, and the baseline hash is verified
first, so a candidate can never silently drift from the incumbent source.

    python build_candidate.py <variant> <out.py>

Variants
  probe   baseline + the new kernel in its own extension + `mid_probe()`
          (no dispatch change; for `midprobe` runs only)
  ship    probe layout, but the new CUDA source is folded into the existing
          combined `load_inline` (one extension for every CUDA kernel -- the
          exp-050 finding that makes a cold build fit popcorn's 360s budget)
          and `custom_kernel` dispatches the two enrolled shapes.
"""

import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE.parent / "062-midshape-2x" / "baseline-907267.py"
BASELINE_SHA = "06799fb095b9fbccb476e7da2c0567a3d36ba57ccb09ccf278b49149db8814c2"
TAIL_DEFAULT = "tail-v5.py"

# --- anchors that must appear exactly once in the baseline -------------------

ANCHOR_EXT_NAME = '            name="chol3264128_exp055_combined_o3",'
ANCHOR_EXT_CPP = (
    '                "void micro32_launch(torch::Tensor, torch::Tensor, "\n'
    '                "torch::Tensor, int64_t, int64_t, int64_t);"\n'
)
ANCHOR_EXT_SRC = (
    "            cuda_sources=(\n"
    "                _CUDA32_SOURCE + \"\\n\" + _CUDA64_SOURCE + \"\\n\" +\n"
    "                _CUDA128_SOURCE\n"
    "            ),\n"
)
ANCHOR_EXT_FN = (
    '            functions=["chol32_launch", "chol64_launch", "chol128_launch",\n'
    '                       "micro32_launch"],\n'
)
ANCHOR_DISPATCH = (
    "    # Few-but-large matrices: avoid cusolverDnSpotrfBatched"
)
ANCHOR_DISPATCH_EARLY = (
    "    # Experiment 015 round 4: two-level blocked tensor-core potrf with"
)
ANCHOR_GLOBALS = (
    "    global _EXP057_V2_HITS, _EXP058_V1_HITS, _EXP061_16384_HITS\n"
)

ANCHOR_HOIST = (
    "if torch.cuda.is_available():\n"
    "    try:\n"
    "        from torch.utils.cpp_extension import load_inline\n"
    "\n"
    "        _CUDA128 = load_inline(\n"
)

DISPATCH_BLOCK = """    # Experiment 062: tiny-batch mid shapes. The vendor factorization runs
    # once per matrix and is dependent-pivot-latency bound, so it costs c*n per
    # matrix regardless of batch. The blocked path factors both matrices with
    # two co-resident CTAs, paying the pivot chain once for the whole batch.
    if (
        is_f32_cuda
        and (batch, n) in _EXP062_SHAPES
        and data.is_contiguous()
    ):
        _load_exp062()
        if _EXP062 is not None:
            try:
                l = _exp062_factor(data, _EXP062_SHAPES[(batch, n)])
                if torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all().item():
                    _EXP062_HITS += 1
                    return l
                _EXP062_FALLBACKS += 1
            except Exception as exc:
                _EXP062_ERROR = repr(exc)
                _EXP062_FALLBACKS += 1

"""

GLOBALS_LINE = "    global _EXP062_HITS, _EXP062_FALLBACKS, _EXP062_ERROR\n"


def _require(src, anchor, label):
    if src.count(anchor) != 1:
        raise SystemExit(
            f"anchor {label!r} appears {src.count(anchor)} times, expected 1"
        )


def build(variant: str, tail_name: str = TAIL_DEFAULT) -> str:
    base = BASELINE.read_text()
    got = hashlib.sha256(base.encode()).hexdigest()
    if got != BASELINE_SHA:
        raise SystemExit(f"baseline hash mismatch: {got}")
    tail = (HERE / tail_name).read_text()

    if variant == "probe":
        return base + tail

    if variant == "ship_sep":
        # Dispatch patch only. The new kernel keeps its own load_inline, which
        # is the exact arrangement every measured probe candidate used.
        _require(base, ANCHOR_DISPATCH, "dispatch")
        _require(base, ANCHOR_GLOBALS, "globals")
        out = base.replace(ANCHOR_GLOBALS, ANCHOR_GLOBALS + GLOBALS_LINE)
        out = out.replace(ANCHOR_DISPATCH, DISPATCH_BLOCK + ANCHOR_DISPATCH)
        return out + tail

    if variant == "ship_sep_early":
        # Same layout, but the exp-062 branch is hoisted ABOVE the split32
        # dispatch. 16x512, 4x1024 and 8x2048 all appear in `_SPLIT32_SHAPES`
        # and would otherwise return before the resident-block path is ever
        # consulted. Shapes that are not in `_EXP062_SHAPES` are untouched, so
        # nothing else changes position.
        _require(base, ANCHOR_DISPATCH_EARLY, "early dispatch")
        _require(base, ANCHOR_GLOBALS, "globals")
        out = base.replace(ANCHOR_GLOBALS, ANCHOR_GLOBALS + GLOBALS_LINE)
        out = out.replace(ANCHOR_DISPATCH_EARLY,
                          DISPATCH_BLOCK + ANCHOR_DISPATCH_EARLY)
        return out + tail

    if variant != "ship":
        raise SystemExit(f"unknown variant {variant!r}")

    # 1. Fold the new CUDA source into the single combined extension so a cold
    #    build compiles the pybind glue once (exp-050 blocker 2).
    for anchor, label in (
        (ANCHOR_EXT_NAME, "ext name"),
        (ANCHOR_EXT_CPP, "ext cpp"),
        (ANCHOR_EXT_SRC, "ext sources"),
        (ANCHOR_EXT_FN, "ext functions"),
        (ANCHOR_DISPATCH, "dispatch"),
        (ANCHOR_GLOBALS, "globals"),
    ):
        _require(base, anchor, label)

    out = base.replace(
        ANCHOR_EXT_NAME,
        '            name="chol3264128_exp062_combined_o3",',
    )
    out = out.replace(
        ANCHOR_EXT_CPP,
        ANCHOR_EXT_CPP
        + '                "\\nvoid e62_diag128_launch(torch::Tensor, '
        'torch::Tensor, int64_t, int64_t, torch::Tensor);"\n',
    )
    out = out.replace(
        ANCHOR_EXT_SRC,
        "            cuda_sources=(\n"
        "                _CUDA32_SOURCE + \"\\n\" + _CUDA64_SOURCE + \"\\n\" +\n"
        "                _CUDA128_SOURCE + \"\\n\" + _EXP062_SOURCE\n"
        "            ),\n",
    )
    out = out.replace(
        ANCHOR_EXT_FN,
        '            functions=["chol32_launch", "chol64_launch", "chol128_launch",\n'
        '                       "micro32_launch", "e62_diag128_launch"],\n',
    )
    out = out.replace(ANCHOR_GLOBALS, ANCHOR_GLOBALS + GLOBALS_LINE)
    tail = tail.replace(
        "_EXP062_COMBINED = None", "_EXP062_COMBINED = _CUDA128"
    )
    # The combined load_inline runs at import time, long before the appended
    # tail. Hoist the CUDA source string above it, or the merged build dies
    # with NameError and every pre-existing CUDA fast path silently vanishes.
    key = '_EXP062_SOURCE = r"""'
    i = tail.index(key)
    j = tail.index('"""', i + len(key)) + 3
    source_decl = tail[i:j]
    tail = tail[:i] + tail[j:]
    _require(out, ANCHOR_HOIST, "hoist")
    out = out.replace(ANCHOR_HOIST, source_decl + "\n\n" + ANCHOR_HOIST)
    out = out.replace(ANCHOR_DISPATCH, DISPATCH_BLOCK + ANCHOR_DISPATCH)
    return out + tail


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    text = build(sys.argv[1], sys.argv[3] if len(sys.argv) == 4 else TAIL_DEFAULT)
    pathlib.Path(sys.argv[2]).write_text(text)
    print(
        f"wrote {sys.argv[2]} ({len(text.splitlines())} lines, "
        f"sha256 {hashlib.sha256(text.encode()).hexdigest()[:12]})"
    )
