"""Architectural candidates for experiment 010's exact 1x16384 shape.

This file is uploaded only by the bounded Modal harness.  Non-target calls are
delegated byte-for-byte to the exp009 ranked baseline.  A finalist is copied
into a complete submission snapshot only after it clears every local and B200
gate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import triton
import triton.language as tl


def _load_baseline():
    path = Path(__file__).with_name("baseline-exp009.py")
    spec = importlib.util.spec_from_file_location("exp010_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASELINE = _load_baseline()
TARGET_N = 16384
OUTER_NB = 2048
TILE = 128
DEPTH_TILE = 64


@triton.jit
def _lower_tf32_update_kernel(
    a_ptr,
    l_ptr,
    rows_count,
    depth_count: tl.constexpr,
    stride_ar,
    stride_ac,
    stride_lr,
    stride_lk,
    tile: tl.constexpr,
    depth_tile: tl.constexpr,
):
    triangular_id = tl.program_id(0)
    block_row = ((tl.sqrt(8.0 * triangular_id + 1.0) - 1.0) * 0.5).to(tl.int32)
    block_col = triangular_id - block_row * (block_row + 1) // 2
    rows = block_row * tile + tl.arange(0, tile)
    cols = block_col * tile + tl.arange(0, tile)
    acc = tl.zeros((tile, tile), tl.float32)
    for kk in range(0, depth_count, depth_tile):
        depth = kk + tl.arange(0, depth_tile)
        lhs = tl.load(
            l_ptr + rows[:, None] * stride_lr + depth[None, :] * stride_lk,
            mask=(rows[:, None] < rows_count) & (depth[None, :] < depth_count),
            other=0.0,
        )
        rhs = tl.load(
            l_ptr + cols[None, :] * stride_lr + depth[:, None] * stride_lk,
            mask=(cols[None, :] < rows_count) & (depth[:, None] < depth_count),
            other=0.0,
        )
        acc += tl.dot(lhs, rhs, input_precision="tf32", out_dtype=tl.float32)
    out_ptrs = a_ptr + rows[:, None] * stride_ar + cols[None, :] * stride_ac
    valid = (rows[:, None] < rows_count) & (cols[None, :] < rows_count)
    valid &= (block_row != block_col) | (cols[None, :] <= rows[:, None])
    old = tl.load(out_ptrs, mask=valid, other=0.0)
    tl.store(out_ptrs, old - acc, mask=valid)


@triton.jit
def _lower_fp8_update_kernel(
    a_ptr,
    q_ptr,
    scale_ptr,
    rows_count,
    depth_count: tl.constexpr,
    stride_ar,
    stride_ac,
    stride_qr,
    stride_qk,
    tile: tl.constexpr,
    depth_tile: tl.constexpr,
):
    triangular_id = tl.program_id(0)
    block_row = ((tl.sqrt(8.0 * triangular_id + 1.0) - 1.0) * 0.5).to(tl.int32)
    block_col = triangular_id - block_row * (block_row + 1) // 2
    rows = block_row * tile + tl.arange(0, tile)
    cols = block_col * tile + tl.arange(0, tile)
    acc = tl.zeros((tile, tile), tl.float32)
    for kk in range(0, depth_count, depth_tile):
        depth = kk + tl.arange(0, depth_tile)
        lhs = tl.load(
            q_ptr + rows[:, None] * stride_qr + depth[None, :] * stride_qk,
            mask=(rows[:, None] < rows_count) & (depth[None, :] < depth_count),
            other=0.0,
        )
        rhs = tl.load(
            q_ptr + cols[None, :] * stride_qr + depth[:, None] * stride_qk,
            mask=(cols[None, :] < rows_count) & (depth[:, None] < depth_count),
            other=0.0,
        )
        acc += tl.dot(lhs, rhs, out_dtype=tl.float32)
    scale = tl.load(scale_ptr)
    acc *= scale * scale
    out_ptrs = a_ptr + rows[:, None] * stride_ar + cols[None, :] * stride_ac
    valid = (rows[:, None] < rows_count) & (cols[None, :] < rows_count)
    valid &= (block_row != block_col) | (cols[None, :] <= rows[:, None])
    old = tl.load(out_ptrs, mask=valid, other=0.0)
    tl.store(out_ptrs, old - acc, mask=valid)


def _triangular_grid(rows: int) -> tuple[int]:
    blocks = triton.cdiv(rows, TILE)
    return (blocks * (blocks + 1) // 2,)


def _lower_tf32_update(a22: torch.Tensor, panel: torch.Tensor) -> None:
    _lower_tf32_update_kernel[_triangular_grid(a22.shape[0])](
        a22,
        panel,
        a22.shape[0],
        panel.shape[1],
        a22.stride(0),
        a22.stride(1),
        panel.stride(0),
        panel.stride(1),
        tile=TILE,
        depth_tile=DEPTH_TILE,
        num_warps=8,
        num_stages=3,
    )


def _quantize_e4m3(panel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-panel dynamic scale.  Keeping the scalar on-device avoids a host sync.
    scale = (panel.abs().amax() / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (panel / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, scale


def _lower_fp8_update(a22: torch.Tensor, panel: torch.Tensor) -> None:
    quantized, scale = _quantize_e4m3(panel)
    _lower_fp8_update_kernel[_triangular_grid(a22.shape[0])](
        a22,
        quantized,
        scale,
        a22.shape[0],
        panel.shape[1],
        a22.stride(0),
        a22.stride(1),
        quantized.stride(0),
        quantized.stride(1),
        tile=TILE,
        depth_tile=DEPTH_TILE,
        num_warps=8,
        num_stages=3,
    )


def _scaled_mm_fp8(panel: torch.Tensor) -> torch.Tensor:
    quantized, scale = _quantize_e4m3(panel)
    return torch._scaled_mm(
        quantized,
        quantized.transpose(0, 1),
        scale_a=scale,
        scale_b=scale,
        out_dtype=torch.float32,
        use_fast_accum=True,
    )


def _factor_right_looking(
    mat: torch.Tensor,
    update,
    *,
    diagonal_factor=None,
    outer_nb: int = OUTER_NB,
) -> torch.Tensor:
    a = mat.clone()
    n = a.shape[0]
    for k in range(0, n, outer_nb):
        kb = min(outer_nb, n - k)
        a11 = a[k : k + kb, k : k + kb]
        if diagonal_factor is None:
            l11 = torch.linalg.cholesky_ex(a11, check_errors=False).L
        else:
            l11 = diagonal_factor(a11)
        a[k : k + kb, k : k + kb] = l11
        j = k + kb
        if j >= n:
            break
        a21 = a[j:, k : k + kb]
        l21 = torch.linalg.solve_triangular(
            l11.transpose(0, 1), a21, upper=True, left=False
        )
        a[j:, k : k + kb] = l21
        update(a[j:, j:], l21)
    return torch.tril(a)


def v1_lower_tf32(data: torch.Tensor) -> torch.Tensor:
    return _factor_right_looking(data[0], _lower_tf32_update).unsqueeze(0)


_SYRK_MOD = None
_SYRK_ERROR = None
_SYRK_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>

torch::Tensor syrk_lower_(torch::Tensor c, torch::Tensor a) {
    TORCH_CHECK(c.is_cuda() && a.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(c.scalar_type() == torch::kFloat32, "c must be float32");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(c.is_contiguous() || c.stride(1) == 1, "row-major c required");
    TORCH_CHECK(a.is_contiguous(), "contiguous a required");
    const int n = static_cast<int>(c.size(0));
    const int k = static_cast<int>(a.size(1));
    const float alpha = -1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasMath_t previous;
    TORCH_CHECK(cublasGetMathMode(handle, &previous) == CUBLAS_STATUS_SUCCESS,
                "cublasGetMathMode failed");
    TORCH_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) == CUBLAS_STATUS_SUCCESS,
                "cublasSetMathMode failed");
    cublasStatus_t status = cublasSsyrk(
        handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k, &alpha,
        a.data_ptr<float>(), k, &beta, c.data_ptr<float>(), static_cast<int>(c.stride(0)));
    cublasSetMathMode(handle, previous);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasSsyrk failed: ", int(status));
    return c;
}
"""


def initialize_syrk_backend() -> None:
    global _SYRK_MOD, _SYRK_ERROR
    if _SYRK_MOD is not None or _SYRK_ERROR is not None:
        return
    try:
        from torch.utils.cpp_extension import load_inline

        _SYRK_MOD = load_inline(
            name="exp010_syrk_ext_v1",
            cpp_sources=_SYRK_CPP,
            functions=["syrk_lower_"],
            with_cuda=True,
            extra_ldflags=["-lcublas"],
            verbose=False,
        )
    except Exception as exc:  # recorded by the harness; never silently fallback
        _SYRK_ERROR = repr(exc)


def _syrk_update(a22: torch.Tensor, panel: torch.Tensor) -> None:
    initialize_syrk_backend()
    if _SYRK_MOD is None:
        raise RuntimeError(f"SYRK backend unavailable: {_SYRK_ERROR}")
    _SYRK_MOD.syrk_lower_(a22, panel.contiguous())


def v2_cublas_syrk(data: torch.Tensor) -> torch.Tensor:
    return _factor_right_looking(data[0], _syrk_update).unsqueeze(0)


def _fp8_full_update(a22: torch.Tensor, panel: torch.Tensor) -> None:
    a22.sub_(_scaled_mm_fp8(panel))


def v3_fp8_full(data: torch.Tensor) -> torch.Tensor:
    return _factor_right_looking(data[0], _fp8_full_update).unsqueeze(0)


def v4_fp8_lower(data: torch.Tensor) -> torch.Tensor:
    return _factor_right_looking(data[0], _lower_fp8_update).unsqueeze(0)


def _hierarchical_diagonal(block: torch.Tensor) -> torch.Tensor:
    return _factor_right_looking(
        block,
        _lower_tf32_update,
        outer_nb=256,
    )


def v5_hierarchical(data: torch.Tensor) -> torch.Tensor:
    return _factor_right_looking(
        data[0],
        _lower_tf32_update,
        diagonal_factor=_hierarchical_diagonal,
    ).unsqueeze(0)


_GRAPH = None
_GRAPH_INPUT = None
_GRAPH_OUTPUT = None
_GRAPH_ERROR = None


def v6_graph_lower_tf32(data: torch.Tensor) -> torch.Tensor:
    global _GRAPH, _GRAPH_INPUT, _GRAPH_OUTPUT, _GRAPH_ERROR
    if _GRAPH is None and _GRAPH_ERROR is None:
        try:
            _GRAPH_INPUT = torch.empty_like(data)
            _GRAPH_INPUT.copy_(data)
            for _ in range(2):
                _GRAPH_OUTPUT = v1_lower_tf32(_GRAPH_INPUT)
            torch.cuda.synchronize()
            _GRAPH = torch.cuda.CUDAGraph()
            with torch.cuda.graph(_GRAPH):
                _GRAPH_OUTPUT = v1_lower_tf32(_GRAPH_INPUT)
            _GRAPH.replay()
        except Exception as exc:
            _GRAPH_ERROR = repr(exc)
            raise RuntimeError(f"graph capture failed: {_GRAPH_ERROR}") from exc
    if _GRAPH is None:
        raise RuntimeError(f"graph capture unavailable: {_GRAPH_ERROR}")
    _GRAPH_INPUT.copy_(data)
    _GRAPH.replay()
    # Popcorn retains outputs, so reusable graph storage must not escape.
    return _GRAPH_OUTPUT.clone()


def _hierarchical_right_trsm(
    upper: torch.Tensor, rhs: torch.Tensor, inner: int = 256
) -> torch.Tensor:
    """Solve X @ upper = rhs using tensor-core cross-block updates.

    The diagonal 256-wide solves remain FP32.  All O(m*k^2) dependencies between
    those solves become TF32 GEMMs, directly attacking the component profile's
    dominant 7.2 ms vendor TRSM cost.
    """
    out = rhs.clone()
    width = upper.shape[0]
    for q in range(0, width, inner):
        qb = min(inner, width - q)
        current = out[:, q : q + qb]
        if q:
            current.addmm_(out[:, :q], upper[:q, q : q + qb], beta=1.0, alpha=-1.0)
        solved = torch.linalg.solve_triangular(
            upper[q : q + qb, q : q + qb],
            current,
            upper=True,
            left=False,
        )
        out[:, q : q + qb] = solved
    return out


def _hierarchical_diagonal_fast(block: torch.Tensor, inner: int = 256) -> torch.Tensor:
    """Factor one 2048 diagonal block with tensor-core cross-block work."""
    a = block.clone()
    width = a.shape[0]
    for k in range(0, width, inner):
        kb = min(inner, width - k)
        l11 = torch.linalg.cholesky_ex(
            a[k : k + kb, k : k + kb], check_errors=False
        ).L
        a[k : k + kb, k : k + kb] = l11
        j = k + kb
        if j >= width:
            break
        panel = torch.linalg.solve_triangular(
            l11.transpose(0, 1),
            a[j:, k : k + kb],
            upper=True,
            left=False,
        )
        a[j:, k : k + kb] = panel
        a[j:, j:].addmm_(panel, panel.transpose(0, 1), beta=1.0, alpha=-1.0)
    return torch.tril(a)


def _fast_clear_upper(a: torch.Tensor) -> None:
    total = a.numel()
    grid = 4096
    candidates_kernel = getattr(BASELINE, "_clear_upper_8x2048", None)
    if candidates_kernel is None:
        raise RuntimeError("ranked Triton upper-clear kernel unavailable")
    candidates_kernel[(grid,)](
        a,
        total=total,
        n=a.shape[0],
        BLOCK=256,
        GRID=grid,
        num_warps=8,
    )


def v7_hierarchical_panel(data: torch.Tensor) -> torch.Tensor:
    a = data[0].clone()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, TARGET_N, OUTER_NB):
            l11 = _hierarchical_diagonal_fast(
                a[k : k + OUTER_NB, k : k + OUTER_NB]
            )
            a[k : k + OUTER_NB, k : k + OUTER_NB] = l11
            j = k + OUTER_NB
            if j >= TARGET_N:
                break
            panel = _hierarchical_right_trsm(
                l11.transpose(0, 1),
                a[j:, k : k + OUTER_NB],
            )
            a[j:, k : k + OUTER_NB] = panel
            a[j:, j:].addmm_(panel, panel.transpose(0, 1), beta=1.0, alpha=-1.0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    _fast_clear_upper(a)
    return a.unsqueeze(0)


def v8_inverse_panel_gemm(data: torch.Tensor) -> torch.Tensor:
    """Replace each large right-side TRSM with inverse formation + TF32 GEMM.

    A 1024-wide outer panel reduces serial POTRF work.  The compact triangular
    inverse is formed in FP32; applying it to the tall panel is a tensor-core
    GEMM rather than the component profile's dominant vendor TRSM.
    """
    nb = 1024
    a = data[0].clone()
    identity = torch.eye(nb, dtype=a.dtype, device=a.device)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, TARGET_N, nb):
            l11 = torch.linalg.cholesky_ex(
                a[k : k + nb, k : k + nb], check_errors=False
            ).L
            a[k : k + nb, k : k + nb] = l11
            j = k + nb
            if j >= TARGET_N:
                break
            upper = l11.transpose(0, 1)
            inverse = torch.linalg.solve_triangular(
                upper, identity, upper=True, left=True
            )
            panel = a[j:, k : k + nb] @ inverse
            a[j:, k : k + nb] = panel
            a[j:, j:].addmm_(panel, panel.transpose(0, 1), beta=1.0, alpha=-1.0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    _fast_clear_upper(a)
    return a.unsqueeze(0)


def v9_left_looking(data: torch.Tensor) -> torch.Tensor:
    """Left-looking blocked Cholesky: update only the active diagonal and panel."""
    a = data[0].clone()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, TARGET_N, OUTER_NB):
            kb = min(OUTER_NB, TARGET_N - k)
            diagonal = a[k : k + kb, k : k + kb]
            if k:
                left = a[k : k + kb, :k]
                diagonal.addmm_(left, left.transpose(0, 1), beta=1.0, alpha=-1.0)
            l11 = torch.linalg.cholesky_ex(diagonal, check_errors=False).L
            a[k : k + kb, k : k + kb] = l11
            j = k + kb
            if j >= TARGET_N:
                break
            panel = a[j:, k : k + kb]
            if k:
                panel.addmm_(
                    a[j:, :k],
                    a[k : k + kb, :k].transpose(0, 1),
                    beta=1.0,
                    alpha=-1.0,
                )
            solved = torch.linalg.solve_triangular(
                l11.transpose(0, 1), panel, upper=True, left=False
            )
            a[j:, k : k + kb] = solved
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    _fast_clear_upper(a)
    return a.unsqueeze(0)


VARIANTS = {
    "v1_lower_tf32": v1_lower_tf32,
    "v2_cublas_syrk": v2_cublas_syrk,
    "v3_fp8_full": v3_fp8_full,
    "v4_fp8_lower": v4_fp8_lower,
    "v5_hierarchical": v5_hierarchical,
    "v6_graph_lower_tf32": v6_graph_lower_tf32,
    "v7_hierarchical_panel": v7_hierarchical_panel,
    "v8_inverse_panel_gemm": v8_inverse_panel_gemm,
    "v9_left_looking": v9_left_looking,
}


def backend_status() -> dict:
    return {
        "triton": True,
        "syrk_loaded": _SYRK_MOD is not None,
        "syrk_error": _SYRK_ERROR,
        "fp8_dtype": hasattr(torch, "float8_e4m3fn"),
        "scaled_mm": hasattr(torch, "_scaled_mm"),
        "graph_captured": _GRAPH is not None,
        "graph_error": _GRAPH_ERROR,
    }


def candidate_call(name: str, data: torch.Tensor) -> torch.Tensor:
    if tuple(data.shape) != (1, TARGET_N, TARGET_N):
        return BASELINE.custom_kernel(data)
    return VARIANTS[name](data)
