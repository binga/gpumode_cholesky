"""Experiment 052 V1: transfer shipped MXFP8 panel updates to 1x8192."""

import torch
import submission as _ranked


_MX8192_HITS = 0
_MX8192_READY_HITS = 0
_MX8192_FALLBACKS = 0
_MX8192_ERROR = None

_ranked._LARGE_CFG = dict(_ranked._LARGE_CFG)
_ranked._LARGE_CFG[8192] = dict(
    nb=2048,
    panel_mode="mxfp8",
    diag_mode="tf32",
    rec_inv=False,
    shadow=False,
)


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _MX8192_HITS, _MX8192_READY_HITS, _MX8192_FALLBACKS, _MX8192_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 8192, 8192)
        and data.is_contiguous()
    )
    before_hits = int(_ranked._MXFP8_HITS)
    before_fallbacks = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    if target:
        hit_delta = int(_ranked._MXFP8_HITS) - before_hits
        fallback_delta = int(_ranked._LARGE_FP8_FALLBACKS) - before_fallbacks
        _MX8192_HITS += hit_delta
        _MX8192_FALLBACKS += fallback_delta
        _MX8192_READY_HITS += int(hit_delta > 0 and fallback_delta == 0)
        _MX8192_ERROR = _ranked._LARGE_FP8_ERROR
    return output
