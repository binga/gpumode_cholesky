"""Experiment 053 V2: use one direct 2048 triangular inverse per panel."""

import torch
import submission as _ranked


_RECINV2048_TARGET_HITS = 0
_RECINV2048_BASE_HITS = 0
_RECINV2048_FALLBACKS = 0
_RECINV2048_ERROR = None


def _tri_inv_recursive_2048(lower: torch.Tensor) -> torch.Tensor:
    global _RECINV2048_BASE_HITS
    n = lower.shape[0]
    if n <= 2048:
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        _RECINV2048_BASE_HITS += 1
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    m = n // 2
    inv11 = _tri_inv_recursive_2048(lower[:m, :m])
    inv22 = _tri_inv_recursive_2048(lower[m:, m:])
    out = torch.zeros_like(lower)
    out[:m, :m] = inv11
    out[m:, m:] = inv22
    out[m:, :m] = -(inv22 @ (lower[m:, :m] @ inv11))
    return out


_ranked._tri_inv_recursive = _tri_inv_recursive_2048


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _RECINV2048_TARGET_HITS, _RECINV2048_FALLBACKS, _RECINV2048_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    before_base = _RECINV2048_BASE_HITS
    before_fallback = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    if target:
        base_delta = _RECINV2048_BASE_HITS - before_base
        fallback_delta = int(_ranked._LARGE_FP8_FALLBACKS) - before_fallback
        _RECINV2048_TARGET_HITS += int(base_delta > 0 and fallback_delta == 0)
        _RECINV2048_FALLBACKS += fallback_delta
        _RECINV2048_ERROR = _ranked._LARGE_FP8_ERROR
    return output
