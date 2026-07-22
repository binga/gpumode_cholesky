"""Experiment 049 V5: enroll only 16x512 in ranked fused-panel graph path."""

import torch
import submission as _ranked


_FUSED512_HITS = 0
_FUSED512_READY_HITS = 0
_FUSED512_FALLBACKS = 0
_FUSED512_ERROR = None

# Exact-shape overlay. The ranked module owns the Triton implementation and
# graph lifecycle; every other dispatch configuration remains byte-for-byte
# the current #890798 behavior.
_ranked._FUSED_PANEL_SHAPES = dict(_ranked._FUSED_PANEL_SHAPES)
_ranked._FUSED_PANEL_SHAPES[(16, 512)] = (128, 8, False)


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _FUSED512_HITS, _FUSED512_READY_HITS
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (16, 512, 512)
        and data.is_contiguous()
    )
    before = int(_ranked._FUSED_PANEL_HITS)
    output = _ranked.custom_kernel(data)
    if target:
        _FUSED512_HITS += int(_ranked._FUSED_PANEL_HITS) - before
        _FUSED512_READY_HITS += 1
    return output
