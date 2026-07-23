"""Experiment 054 V1: run the incumbent 16384 path with nb=1024."""

import torch
import submission as _ranked


_NB1024_ATTEMPTS = 0
_NB1024_HITS = 0
_NB1024_FALLBACKS = 0
_NB1024_ERROR = None

_ranked._LARGE_CFG = dict(_ranked._LARGE_CFG)
_ranked._LARGE_CFG[16384] = dict(_ranked._LARGE_CFG[16384], nb=1024)
_original_left_looking_large = _ranked._left_looking_large


def _left_looking_large_nb1024(mat, nb, panel_mode, diag_mode, rec_inv, shadow):
    global _NB1024_ATTEMPTS, _NB1024_HITS
    target = mat.shape[0] == 16384 and nb == 1024
    _NB1024_ATTEMPTS += int(target)
    result = _original_left_looking_large(
        mat, nb, panel_mode, diag_mode, rec_inv, shadow
    )
    _NB1024_HITS += int(target)
    return result


_ranked._left_looking_large = _left_looking_large_nb1024


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _NB1024_FALLBACKS, _NB1024_ERROR
    before = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    _NB1024_FALLBACKS += int(_ranked._LARGE_FP8_FALLBACKS) - before
    _NB1024_ERROR = _ranked._LARGE_FP8_ERROR
    return output
