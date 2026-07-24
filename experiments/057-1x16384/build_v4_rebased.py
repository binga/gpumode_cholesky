"""Build V4 on the exact ranked #904546 source.

The generated file is standalone: exact source SHA
f8d67dce5a7a0dd68fc96e24613444970aa8c637b168bcb252cab01f2db89e5a
plus one mechanically checked replacement of the 1x16384 inverse helper.
"""

import hashlib
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE = ROOT / "submission.py"
OUTPUT = pathlib.Path(__file__).resolve().parent / "candidate-v4-rebased-904546.py"
BASELINE_SHA256 = (
    "f8d67dce5a7a0dd68fc96e24613444970aa8c637b168bcb252cab01f2db89e5a"
)

COUNTER_ANCHOR = """_EXP057_V2_HITS = 0
_EXP057_V2_INVERSE_CALLS = 0
_EXP058_V1_HITS = 0"""

COUNTER_AND_KERNEL = r"""_EXP057_V2_HITS = 0
_EXP057_V2_INVERSE_CALLS = 0
_EXP057_V4_TRITON_LEAF_HITS = 0
_EXP058_V1_HITS = 0


if _HAVE_TRITON:

    @triton.jit
    def _exp057_tri_inv_leaf32_kernel(
        lower_ptr,
        inverse_ptr,
        n: tl.constexpr,
        base: tl.constexpr,
    ):
        # One program solves one column of one 32x32 diagonal block.
        pid = tl.program_id(0)
        block = pid // base
        column = pid % base
        rows = tl.arange(0, base)
        row0 = block * base
        values = tl.zeros((base,), dtype=tl.float32)
        for row in tl.static_range(0, base):
            diagonal = tl.load(
                lower_ptr + (row0 + row) * n + row0 + row
            )
            coefficients = tl.load(
                lower_ptr + (row0 + row) * n + row0 + rows,
                mask=rows < row,
                other=0.0,
            )
            rhs = tl.where(column == row, 1.0, 0.0)
            solved = (
                rhs - tl.sum(coefficients * values, axis=0)
            ) / diagonal
            values = tl.where(rows == row, solved, values)
        tl.store(
            inverse_ptr + (row0 + rows) * n + row0 + column,
            values,
            mask=rows >= column,
        )
"""

OLD_HELPER = """def _trsm_free_inverse_16384(lower: torch.Tensor) -> torch.Tensor:
    global _EXP057_V2_INVERSE_CALLS
    _EXP057_V2_INVERSE_CALLS += 1
    n = lower.shape[0]
    if n & (n - 1):
        raise ValueError("trsm-free inverse requires a power-of-two order")
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    inverse.diagonal().copy_(lower.diagonal().reciprocal())
    size = 1
    while size < n:
        step = 2 * size
        shape = (n // step, size, size)
        stride = (step * n + step, n, 1)
        inv11 = inverse.as_strided(shape, stride, 0)
        inv22 = inverse.as_strided(shape, stride, size * n + size)
        low21 = lower.as_strided(shape, stride, size * n)
        inverse.as_strided(shape, stride, size * n).copy_(
            torch.bmm(inv22, torch.bmm(low21, inv11)).neg_()
        )
        size = step
    return inverse
"""

NEW_HELPER = """def _trsm_free_inverse_16384(lower: torch.Tensor) -> torch.Tensor:
    global _EXP057_V2_INVERSE_CALLS, _EXP057_V4_TRITON_LEAF_HITS
    _EXP057_V2_INVERSE_CALLS += 1
    n = lower.shape[0]
    if not _HAVE_TRITON or n % 32 or (n & (n - 1)):
        raise RuntimeError("exp057 Triton base-32 inverse precondition failed")
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    count = n // 32
    _exp057_tri_inv_leaf32_kernel[(count * 32,)](
        lower,
        inverse,
        n=n,
        base=32,
        num_warps=1,
    )
    _EXP057_V4_TRITON_LEAF_HITS += 1
    size = 32
    while size < n:
        step = 2 * size
        shape = (n // step, size, size)
        stride = (step * n + step, n, 1)
        inv11 = inverse.as_strided(shape, stride, 0)
        inv22 = inverse.as_strided(shape, stride, size * n + size)
        low21 = lower.as_strided(shape, stride, size * n)
        inverse.as_strided(shape, stride, size * n).copy_(
            torch.bmm(inv22, torch.bmm(low21, inv11)).neg_()
        )
        size = step
    return inverse
"""


source = BASELINE.read_text()
digest = hashlib.sha256(source.encode()).hexdigest()
if digest != BASELINE_SHA256:
    raise SystemExit(f"baseline hash mismatch: {digest}")
if source.count(COUNTER_ANCHOR) != 1:
    raise SystemExit("counter anchor is not unique")
if source.count(OLD_HELPER) != 1:
    raise SystemExit("1x16384 helper anchor is not unique")
source = source.replace(COUNTER_ANCHOR, COUNTER_AND_KERNEL, 1)
source = source.replace(OLD_HELPER, NEW_HELPER, 1)
OUTPUT.write_text(source)
print(f"wrote {OUTPUT}")
print(f"candidate_sha256={hashlib.sha256(source.encode()).hexdigest()}")
