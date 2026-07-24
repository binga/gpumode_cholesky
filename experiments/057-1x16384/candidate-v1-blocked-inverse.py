"""Experiment 057 V1: blocked leaf-batched inverse plus merged block-column update.

Scope is exactly ``batch=1, n=16384``. Every other input delegates byte-for-byte
to ranked submission #890798.
"""

import torch
import submission as _ranked


_EXP057_V1_ATTEMPTS = 0
_EXP057_V1_HITS = 0
_EXP057_V1_FALLBACKS = 0
_EXP057_V1_ERROR = None
_EXP057_V1_INVERSE_CALLS = 0


def _blocked_tri_inv(lower: torch.Tensor, base: int = 256) -> torch.Tensor:
    """Invert a power-of-two lower triangle breadth-first."""
    global _EXP057_V1_INVERSE_CALLS
    _EXP057_V1_INVERSE_CALLS += 1
    n = lower.shape[0]
    if n <= base or n % base or (n & (n - 1)):
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    count = n // base
    leaf_shape = (count, base, base)
    leaf_stride = (base * n + base, n, 1)
    blocks = lower.as_strided(leaf_shape, leaf_stride).contiguous()
    identity = torch.eye(base, device=lower.device, dtype=lower.dtype)
    inverse.as_strided(leaf_shape, leaf_stride).copy_(
        torch.linalg.solve_triangular(
            blocks, identity.expand(leaf_shape).contiguous(), upper=False
        )
    )
    size = base
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
            inverse = _blocked_tri_inv(lkk)
            factor[j:, k:j] = block[nb:] @ inverse.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _EXP057_V1_ATTEMPTS, _EXP057_V1_HITS, _EXP057_V1_FALLBACKS
    global _EXP057_V1_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    if not target:
        return _ranked.custom_kernel(data)
    _EXP057_V1_ATTEMPTS += 1
    try:
        factor = _factor_1x16384(data[0])
        if torch.isfinite(factor.diagonal()).all().item():
            _EXP057_V1_HITS += 1
            return factor.unsqueeze(0)
        _EXP057_V1_ERROR = "non-finite diagonal"
    except Exception as exc:
        _EXP057_V1_ERROR = repr(exc)
    _EXP057_V1_FALLBACKS += 1
    return _ranked.custom_kernel(data)
