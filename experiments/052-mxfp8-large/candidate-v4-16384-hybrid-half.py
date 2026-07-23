"""Experiment 052 V4: clip-safe MXFP8 first half, TF32 second half at 1x16384."""

import torch
import triton
import triton.language as tl
import submission as _ranked


_MX16384_HITS = 0
_MX16384_READY_HITS = 0
_MX16384_FALLBACKS = 0
_MX16384_ERROR = None
_TF32_TAIL_PANEL_HITS = 0


@triton.jit
def _mx_quant_e4m3_blocked_clipfix(
    x_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_xk,
    columns,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Emit MXFP8 values and blocked E8M0 scales without E4M3 clipping.

    The shipped floor(log2(amax)) scale maps a block maximum with normalized
    mantissa above 1.75 past E4M3's maximum 448. For only those blocks, raise
    the shared exponent by one. That halves the normalized values, preventing
    saturation at the cost of one mantissa bit only where clipping would occur.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk)
    grouped = tl.reshape(x, (BLOCK_M, BLOCK_K // 32, 32))
    amax = tl.max(tl.abs(grouped), axis=2)
    amax_bits = amax.to(tl.int32, bitcast=True)
    exp_bits = (amax_bits >> 23) & 0xFF
    mantissa_bits = amax_bits & 0x7FFFFF
    would_clip = mantissa_bits > 0x600000
    sbyte = tl.maximum(exp_bits - 8 + would_clip.to(tl.int32), 0)
    inv_scale = tl.exp2((127 - sbyte).to(tl.float32))
    q = grouped * inv_scale[:, :, None]
    tl.store(
        q_ptr + rows[:, None] * columns + cols[None, :],
        tl.reshape(q, (BLOCK_M, BLOCK_K)).to(tl.float8e4nv),
    )
    tile = (pid_m // 4) * (columns // 128) + pid_k
    b = tl.arange(0, BLOCK_M)
    c_in = tl.arange(0, BLOCK_K // 32)
    tl.store(
        s_ptr
        + tile * 512
        + b[:, None] * 16
        + (pid_m % 4) * 4
        + c_in[None, :],
        sbyte.to(tl.uint8),
    )


_ranked._mx_quant_e4m3_blocked_kernel = _mx_quant_e4m3_blocked_clipfix
_original_mxfp8_panel_update = _ranked._mxfp8_panel_update


def _hybrid_panel_update(out: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    global _TF32_TAIL_PANEL_HITS
    if lhs.shape[1] >= 8192:
        out.addmm_(
            lhs,
            rhs.transpose(-1, -2),
            beta=1.0,
            alpha=-1.0,
        )
        _TF32_TAIL_PANEL_HITS += 1
        return
    _original_mxfp8_panel_update(out, lhs, rhs)


_ranked._mxfp8_panel_update = _hybrid_panel_update
_ranked._LARGE_CFG = dict(_ranked._LARGE_CFG)
_ranked._LARGE_CFG[16384] = dict(
    nb=2048,
    panel_mode="mxfp8",
    diag_mode="tf32",
    rec_inv=True,
    shadow=False,
)


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _MX16384_HITS, _MX16384_READY_HITS
    global _MX16384_FALLBACKS, _MX16384_ERROR
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (1, 16384, 16384)
        and data.is_contiguous()
    )
    before_hits = int(_ranked._MXFP8_HITS)
    before_fallbacks = int(_ranked._LARGE_FP8_FALLBACKS)
    output = _ranked.custom_kernel(data)
    if target:
        hit_delta = int(_ranked._MXFP8_HITS) - before_hits
        fallback_delta = int(_ranked._LARGE_FP8_FALLBACKS) - before_fallbacks
        _MX16384_HITS += hit_delta
        _MX16384_FALLBACKS += fallback_delta
        _MX16384_READY_HITS += int(hit_delta > 0 and fallback_delta == 0)
        _MX16384_ERROR = _ranked._LARGE_FP8_ERROR
    return output
