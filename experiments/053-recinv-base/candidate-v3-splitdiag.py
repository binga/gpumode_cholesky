"""Experiment 053 V3: factor 16384's 2048 diagonal blocks with split32."""

import torch
import submission as _ranked


_SPLITDIAG_TARGET_HITS = 0
_SPLITDIAG_BLOCK_HITS = 0
_SPLITDIAG_FALLBACKS = 0
_SPLITDIAG_ERROR = None

_ranked._SPLIT32_SHAPES = dict(_ranked._SPLIT32_SHAPES)
_ranked._SPLIT32_SHAPES[(1, 2048)] = (
    "tf32",
    "tf32",
    128,
    "graph",
    True,
)
_ranked._SPLIT32_NB_SCHEDULE = dict(_ranked._SPLIT32_NB_SCHEDULE)
_ranked._SPLIT32_NB_SCHEDULE[(1, 2048)] = (256,) * 8
_ranked._PANEL_INNER_SUBTILE64_SHAPES = set(
    _ranked._PANEL_INNER_SUBTILE64_SHAPES
)
_ranked._PANEL_INNER_SUBTILE64_SHAPES.add((1, 2048))
_ranked._BMM_TRAILING_SHAPES = set(_ranked._BMM_TRAILING_SHAPES)
_ranked._BMM_TRAILING_SHAPES.add((1, 2048))

_original_left_looking_large = _ranked._left_looking_large


def _left_looking_large_splitdiag(
    mat: torch.Tensor,
    nb: int,
    panel_mode: str,
    diag_mode: str,
    rec_inv: bool,
    shadow: bool,
) -> torch.Tensor:
    global _SPLITDIAG_BLOCK_HITS
    if mat.shape[0] != 16384 or nb != 2048 or shadow:
        return _original_left_looking_large(
            mat, nb, panel_mode, diag_mode, rec_inv, shadow
        )
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = mat[k : k + kb, k : k + kb].clone()
            if k:
                row = factor[k : k + kb, :k]
                diagonal.addmm_(
                    row, row.transpose(-1, -2), beta=1.0, alpha=-1.0
                )
            lkk = _ranked._split32_factor(diagonal.unsqueeze(0))[0]
            _SPLITDIAG_BLOCK_HITS += 1
            factor[k : k + kb, k : k + kb] = lkk
            j = k + kb
            if j >= n:
                break
            panel = mat[j:, k : k + kb].clone()
            if k:
                panel.addmm_(
                    factor[j:, :k],
                    factor[k : k + kb, :k].transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            if rec_inv:
                inverse = _ranked._tri_inv_recursive(lkk)
                factor[j:, k : k + kb] = panel @ inverse.transpose(-1, -2)
            else:
                factor[j:, k : k + kb] = torch.linalg.solve_triangular(
                    lkk.transpose(-1, -2), panel, upper=True, left=False
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


_ranked._left_looking_large = _left_looking_large_splitdiag


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _SPLITDIAG_TARGET_HITS, _SPLITDIAG_FALLBACKS, _SPLITDIAG_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    before_blocks = _SPLITDIAG_BLOCK_HITS
    before_fallback = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    if target:
        block_delta = _SPLITDIAG_BLOCK_HITS - before_blocks
        fallback_delta = int(_ranked._LARGE_FP8_FALLBACKS) - before_fallback
        _SPLITDIAG_TARGET_HITS += int(block_delta == 8 and fallback_delta == 0)
        _SPLITDIAG_FALLBACKS += fallback_delta
        _SPLITDIAG_ERROR = _ranked._LARGE_FP8_ERROR
    return output
