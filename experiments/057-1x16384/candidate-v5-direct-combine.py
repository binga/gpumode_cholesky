"""Experiment 057 V5: direct strided baddbmm inverse combines.

Scope is exactly ``batch=1, n=16384``. Every other input delegates byte-for-byte
to ranked submission #890798.
"""

import torch
import submission as _ranked

try:
    import triton
    import triton.language as tl

    _EXP057_V5_HAVE_TRITON = True
except Exception:
    triton = None
    tl = None
    _EXP057_V5_HAVE_TRITON = False


_EXP057_V5_ATTEMPTS = 0
_EXP057_V5_HITS = 0
_EXP057_V5_FALLBACKS = 0
_EXP057_V5_ERROR = None
_EXP057_V5_INVERSE_CALLS = 0
_EXP057_V5_TRITON_LEAF_HITS = 0
_EXP057_V5_COMBINE_HITS = 0


if _EXP057_V5_HAVE_TRITON:

    @triton.jit
    def _tri_inv_leaf32_kernel(
        lower_ptr,
        inverse_ptr,
        n: tl.constexpr,
        base: tl.constexpr,
    ):
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
            solved = (rhs - tl.sum(coefficients * values, axis=0)) / diagonal
            values = tl.where(rows == row, solved, values)
        tl.store(
            inverse_ptr + (row0 + rows) * n + row0 + column,
            values,
            mask=rows >= column,
        )


def _direct_combine_inverse(lower: torch.Tensor) -> torch.Tensor:
    """Use Triton leaves and write outer GEMMs directly into strided targets."""
    global _EXP057_V5_INVERSE_CALLS, _EXP057_V5_TRITON_LEAF_HITS
    global _EXP057_V5_COMBINE_HITS
    _EXP057_V5_INVERSE_CALLS += 1
    n = lower.shape[0]
    if not _EXP057_V5_HAVE_TRITON or n % 32 or (n & (n - 1)):
        raise RuntimeError("Triton base-32 inverse precondition failed")
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    count = n // 32
    _tri_inv_leaf32_kernel[(count * 32,)](
        lower,
        inverse,
        n=n,
        base=32,
        num_warps=1,
    )
    _EXP057_V5_TRITON_LEAF_HITS += 1
    size = 32
    while size < n:
        step = 2 * size
        shape = (n // step, size, size)
        stride = (step * n + step, n, 1)
        inv11 = inverse.as_strided(shape, stride, 0)
        inv22 = inverse.as_strided(shape, stride, size * n + size)
        low21 = lower.as_strided(shape, stride, size * n)
        target = inverse.as_strided(shape, stride, size * n)
        middle = torch.bmm(low21, inv11)
        target.baddbmm_(inv22, middle, beta=0.0, alpha=-1.0)
        _EXP057_V5_COMBINE_HITS += 1
        size = step
    return inverse


def _factor_1x16384(mat: torch.Tensor) -> torch.Tensor:
    n = 16384
    nb = 2048
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            j = k + nb
            block = mat[k:, k:j].contiguous()
            if k:
                block.addmm_(
                    factor[k:, :k],
                    factor[k:j, :k].transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            lkk = torch.linalg.cholesky_ex(
                block[:nb], check_errors=False
            ).L
            factor[k:j, k:j] = lkk
            if j >= n:
                break
            inverse = _direct_combine_inverse(lkk)
            factor[j:, k:j] = block[nb:] @ inverse.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _EXP057_V5_ATTEMPTS, _EXP057_V5_HITS, _EXP057_V5_FALLBACKS
    global _EXP057_V5_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    if not target:
        return _ranked.custom_kernel(data)
    _EXP057_V5_ATTEMPTS += 1
    combines_before = _EXP057_V5_COMBINE_HITS
    try:
        factor = _factor_1x16384(data[0])
        if (
            _EXP057_V5_TRITON_LEAF_HITS > 0
            and _EXP057_V5_COMBINE_HITS - combines_before == 42
            and torch.isfinite(factor.diagonal()).all().item()
        ):
            _EXP057_V5_HITS += 1
            return factor.unsqueeze(0)
        _EXP057_V5_ERROR = "missing V5 backend hits or non-finite diagonal"
    except Exception as exc:
        _EXP057_V5_ERROR = repr(exc)
    _EXP057_V5_FALLBACKS += 1
    return _ranked.custom_kernel(data)
