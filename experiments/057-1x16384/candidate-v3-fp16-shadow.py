"""Experiment 057 V3: FP16-resident factor shadow plus trsm-free inverse.

Scope is exactly ``batch=1, n=16384``. Every other input delegates byte-for-byte
to ranked submission #890798.
"""

import torch
import submission as _ranked


_EXP057_V3_ATTEMPTS = 0
_EXP057_V3_HITS = 0
_EXP057_V3_FALLBACKS = 0
_EXP057_V3_ERROR = None
_EXP057_V3_INVERSE_CALLS = 0
_EXP057_V3_FP16_UPDATES = 0


def _trsm_free_inverse(lower: torch.Tensor) -> torch.Tensor:
    """Invert a power-of-two lower triangle from scalar reciprocal leaves."""
    global _EXP057_V3_INVERSE_CALLS
    _EXP057_V3_INVERSE_CALLS += 1
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


def _factor_1x16384(mat: torch.Tensor) -> torch.Tensor:
    global _EXP057_V3_FP16_UPDATES
    n = 16384
    nb = 2048
    factor = torch.zeros_like(mat)
    shadow = torch.empty(n, n, device=mat.device, dtype=torch.float16)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            j = k + nb
            block = mat[k:, k:j].contiguous()
            if k:
                block.sub_(
                    torch.mm(
                        shadow[k:, :k],
                        shadow[k:j, :k].transpose(-1, -2),
                    )
                )
                _EXP057_V3_FP16_UPDATES += 1
            lkk = torch.linalg.cholesky_ex(
                block[:nb], check_errors=False
            ).L
            factor[k:j, k:j] = lkk
            shadow[k:j, k:j] = lkk
            if j >= n:
                break
            inverse = _trsm_free_inverse(lkk)
            solved = block[nb:] @ inverse.transpose(-1, -2)
            factor[j:, k:j] = solved
            shadow[j:, k:j] = solved
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _EXP057_V3_ATTEMPTS, _EXP057_V3_HITS, _EXP057_V3_FALLBACKS
    global _EXP057_V3_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    if not target:
        return _ranked.custom_kernel(data)
    _EXP057_V3_ATTEMPTS += 1
    try:
        factor = _factor_1x16384(data[0])
        if torch.isfinite(factor.diagonal()).all().item():
            _EXP057_V3_HITS += 1
            return factor.unsqueeze(0)
        _EXP057_V3_ERROR = "non-finite diagonal"
    except Exception as exc:
        _EXP057_V3_ERROR = repr(exc)
    _EXP057_V3_FALLBACKS += 1
    return _ranked.custom_kernel(data)
