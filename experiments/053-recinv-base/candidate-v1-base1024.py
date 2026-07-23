"""Experiment 053 V1: use 1024 rather than 512 as recursive-inverse base."""

import torch
import submission as _ranked


_RECINV1024_TARGET_HITS = 0
_RECINV1024_BASE_HITS = 0
_RECINV1024_FALLBACKS = 0
_RECINV1024_ERROR = None


def _tri_inv_recursive_1024(lower: torch.Tensor) -> torch.Tensor:
    global _RECINV1024_BASE_HITS
    n = lower.shape[0]
    if n <= 1024:
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        _RECINV1024_BASE_HITS += 1
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    m = n // 2
    inv11 = _tri_inv_recursive_1024(lower[:m, :m])
    inv22 = _tri_inv_recursive_1024(lower[m:, m:])
    out = torch.zeros_like(lower)
    out[:m, :m] = inv11
    out[m:, m:] = inv22
    out[m:, :m] = -(inv22 @ (lower[m:, :m] @ inv11))
    return out


_ranked._tri_inv_recursive = _tri_inv_recursive_1024


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _RECINV1024_TARGET_HITS, _RECINV1024_FALLBACKS, _RECINV1024_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    before_base = _RECINV1024_BASE_HITS
    before_fallback = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    if target:
        base_delta = _RECINV1024_BASE_HITS - before_base
        fallback_delta = int(_ranked._LARGE_FP8_FALLBACKS) - before_fallback
        _RECINV1024_TARGET_HITS += int(base_delta > 0 and fallback_delta == 0)
        _RECINV1024_FALLBACKS += fallback_delta
        _RECINV1024_ERROR = _ranked._LARGE_FP8_ERROR
    return output
