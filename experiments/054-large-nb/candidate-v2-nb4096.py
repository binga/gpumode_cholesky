"""Experiment 054 V2: run the incumbent 16384 path with nb=4096."""

import torch
import submission as _ranked


_NB4096_ATTEMPTS = 0
_NB4096_HITS = 0
_NB4096_FALLBACKS = 0
_NB4096_ERROR = None

_ranked._LARGE_CFG = dict(_ranked._LARGE_CFG)
_ranked._LARGE_CFG[16384] = dict(_ranked._LARGE_CFG[16384], nb=4096)
_original_left_looking_large = _ranked._left_looking_large


def _left_looking_large_nb4096(mat, nb, panel_mode, diag_mode, rec_inv, shadow):
    global _NB4096_ATTEMPTS, _NB4096_HITS
    target = mat.shape[0] == 16384 and nb == 4096
    _NB4096_ATTEMPTS += int(target)
    result = _original_left_looking_large(
        mat, nb, panel_mode, diag_mode, rec_inv, shadow
    )
    _NB4096_HITS += int(target)
    return result


_ranked._left_looking_large = _left_looking_large_nb4096


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _NB4096_FALLBACKS, _NB4096_ERROR
    before = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    _NB4096_FALLBACKS += int(_ranked._LARGE_FP8_FALLBACKS) - before
    _NB4096_ERROR = _ranked._LARGE_FP8_ERROR
    return output
