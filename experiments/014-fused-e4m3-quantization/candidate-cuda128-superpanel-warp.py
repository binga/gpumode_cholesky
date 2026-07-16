#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission — experiment 012 ranked winner.

Builds on exp 006 (`#878015`) by fusing its TF32 trailing Schur product and
subtraction into an in-place `addmm_` on the trailing view. This removes the full
temporary product and subtraction launch while preserving identical TF32/FP32
numerics. Ranked `#878108`: 17/17, public geomean 1542.914 us (secret 1545.128
us), improving the prior ~1559 us. Experiment 009 adds three exact-shape paths
that were independently measured on the same B200 as their shipped control.
Ranked `#878273`: public 1500.704 us, secret 1501.440 us.
Experiment 012 replaces only the 1x16384 and 1x32768 paths with left-looking
frontiers. Ranked `#878893`: public 1459.321 us, secret 1448.377 us.

Shape dispatcher:
  * n == 32                         -> Triton batched kernel, one warp per matrix
    (experiment 002). Beats cuSOLVER's batched-launch overhead for tiny matrices.
  * batch == 256 and n == 128       -> captured vendor batched factorization
    (1.177x paired speedup, exact numerics).
  * batch == 16 and n == 512        -> static-buffer captured vendor batched
    factorization (1.291x paired speedup, exact numerics). The buffer refresh
    remains fast when the official harness rotates among input allocations.
  * batch == 8 and n == 2048        -> Triton blocked factorization with FP32
    diagonal/panel work and grouped lower TF32 Schur updates (1.619x paired).
  * batch == 1 and n == 16384       -> left-looking TF32 factorization that
    updates only the active diagonal and panel (1.166x paired frontier).
  * batch == 1 and n == 32768       -> left-looking factorization with native
    Blackwell FP8 panel products and FP32 accumulation (1.386x paired frontier).
  * other batch == 1 and n >= 16384 -> blocked right-looking Cholesky with a
    fused in-place TF32 tensor-core trailing update (experiment 008).
    8192 (only ~1.07x in exp 006) stays on cuSOLVER.
  * 2 <= batch <= 4 and n >= 1024   -> per-matrix factorization in a sequential
    loop (experiment 004, region trimmed by exp 005). `torch.linalg` routes
    batch>=2 to `cusolverDnSpotrfBatched`, which is tuned for many-small matrices
    and is ~1.2-4x too slow for few-large ones; factorizing each matrix on its own
    with the fast single-matrix blocked `potrf` is much faster. batch>=8 (e.g.
    8×2048) stays on batched cuSOLVER (faster on popcorn).
  * everything else                 -> batched cuSOLVER via cholesky_ex (best for
    batch=1 mid-n and high-batch small/mid-n, incl. the saturated 640×512).
"""

import torch

from task import input_t, output_t

# ---------------------------------------------------------------------------
# Triton kernel for n == 32 (adopted experiment 002).
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _chol_batched_kernel(
        A_ptr,
        L_ptr,
        stride_ab,
        stride_ai,
        stride_aj,
        stride_lb,
        stride_li,
        stride_lj,
        N: tl.constexpr,
    ):
        """One program (CTA) factorizes one N x N SPD matrix (right-looking)."""
        pid = tl.program_id(0)
        rows = tl.arange(0, N)
        cols = tl.arange(0, N)
        a_ptrs = (
            A_ptr
            + pid * stride_ab
            + rows[:, None] * stride_ai
            + cols[None, :] * stride_aj
        )
        a = tl.load(a_ptrs)

        for k in range(N):
            akk = tl.sum(
                tl.where((rows[:, None] == k) & (cols[None, :] == k), a, 0.0)
            )
            inv = 1.0 / tl.sqrt(akk)
            col_k = (cols[None, :] == k) & (rows[:, None] >= k)
            a = tl.where(col_k, a * inv, a)
            lk = tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)
            trail = (rows[:, None] > k) & (cols[None, :] > k)
            a = tl.where(trail, a - lk[:, None] * lk[None, :], a)

        a = tl.where(cols[None, :] > rows[:, None], 0.0, a)
        l_ptrs = (
            L_ptr
            + pid * stride_lb
            + rows[:, None] * stride_li
            + cols[None, :] * stride_lj
        )
        tl.store(l_ptrs, a)

    _NUM_WARPS = {32: 1}

    def _triton_cholesky32(data: torch.Tensor) -> torch.Tensor:
        batch, n, _ = data.shape
        data = data.contiguous()
        out = torch.empty_like(data)
        _chol_batched_kernel[(batch,)](
            data,
            out,
            data.stride(0),
            data.stride(1),
            data.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            N=n,
            num_warps=_NUM_WARPS.get(n, 4),
        )
        return out


    _BK_8X2048 = 64
    _UPDATE_TILE_8X2048 = 128

    @triton.jit
    def _diag_factor_8x2048(
        a_ptr,
        n: tl.constexpr,
        k,
        BK_CONST: tl.constexpr,
    ):
        batch = tl.program_id(0)
        rows = tl.arange(0, BK_CONST)
        cols = tl.arange(0, BK_CONST)
        base = batch * n * n
        ptrs = a_ptr + base + (k + rows[:, None]) * n + k + cols[None, :]
        tile = tl.load(ptrs)

        for p in range(0, BK_CONST):
            diag_mask = (rows[:, None] == p) & (cols[None, :] == p)
            diagonal = tl.sum(tl.where(diag_mask, tile, 0.0))
            inv_sqrt = 1.0 / tl.sqrt(diagonal)
            column_mask = (cols[None, :] == p) & (rows[:, None] >= p)
            tile = tl.where(column_mask, tile * inv_sqrt, tile)
            column = tl.sum(
                tl.where(cols[None, :] == p, tile, 0.0), axis=1
            )
            trailing = (rows[:, None] > p) & (cols[None, :] > p)
            tile = tl.where(
                trailing,
                tile - column[:, None] * column[None, :],
                tile,
            )

        tl.store(ptrs, tile, mask=cols[None, :] <= rows[:, None])

    @triton.jit
    def _panel_solve_8x2048(
        a_ptr,
        n: tl.constexpr,
        k,
        remaining,
        BK_CONST: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = row_tile * BK_CONST + tl.arange(0, BK_CONST)
        cols = tl.arange(0, BK_CONST)
        base = batch * n * n
        row_mask = rows < remaining

        diag_ptrs = (
            a_ptr
            + base
            + (k + cols[:, None]) * n
            + k
            + cols[None, :]
        )
        diagonal = tl.load(diag_ptrs)
        panel_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + rows[:, None]) * n
            + k
            + cols[None, :]
        )
        panel = tl.load(panel_ptrs, mask=row_mask[:, None], other=0.0)

        for p in range(0, BK_CONST):
            diag_column = tl.sum(
                tl.where(cols[None, :] == p, diagonal, 0.0), axis=1
            )
            diag_pp = tl.sum(
                tl.where(cols == p, diag_column, 0.0), axis=0
            )
            value = tl.sum(
                tl.where(cols[None, :] == p, panel, 0.0), axis=1
            ) / diag_pp
            panel = tl.where(cols[None, :] == p, value[:, None], panel)
            panel = tl.where(
                cols[None, :] > p,
                panel - value[:, None] * diag_column[None, :],
                panel,
            )

        tl.store(panel_ptrs, panel, mask=row_mask[:, None])

    @triton.jit
    def _lower_schur_8x2048(
        a_ptr,
        n: tl.constexpr,
        k,
        remaining,
        BK_CONST: tl.constexpr,
        TILE: tl.constexpr,
    ):
        triangular_id = tl.program_id(0)
        batch = tl.program_id(1)
        block_row = (
            (tl.sqrt(8.0 * triangular_id + 1.0) - 1.0) * 0.5
        ).to(tl.int32)
        block_col = triangular_id - block_row * (block_row + 1) // 2

        rows = block_row * TILE + tl.arange(0, TILE)
        cols = block_col * TILE + tl.arange(0, TILE)
        depth = tl.arange(0, BK_CONST)
        base = batch * n * n
        lhs_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + rows[:, None]) * n
            + k
            + depth[None, :]
        )
        rhs_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + cols[None, :]) * n
            + k
            + depth[:, None]
        )
        lhs = tl.load(lhs_ptrs, mask=rows[:, None] < remaining, other=0.0)
        rhs = tl.load(rhs_ptrs, mask=cols[None, :] < remaining, other=0.0)
        product = tl.dot(lhs, rhs, input_precision="tf32", out_dtype=tl.float32)

        out_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + rows[:, None]) * n
            + k
            + BK_CONST
            + cols[None, :]
        )
        valid = (rows[:, None] < remaining) & (cols[None, :] < remaining)
        valid = valid & (
            (block_row != block_col) | (cols[None, :] <= rows[:, None])
        )
        old = tl.load(out_ptrs, mask=valid, other=0.0)
        tl.store(out_ptrs, old - product, mask=valid)

    @triton.jit
    def _clear_upper_8x2048(
        a_ptr,
        total: tl.constexpr,
        n: tl.constexpr,
        BLOCK: tl.constexpr,
        GRID: tl.constexpr,
    ):
        first = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        for step in range(0, total, GRID * BLOCK):
            offsets = first + step
            valid = offsets < total
            matrix_offset = offsets % (n * n)
            row = matrix_offset // n
            col = matrix_offset - row * n
            tl.store(a_ptr + offsets, 0.0, mask=valid & (col > row))

    def _triton_cholesky_8x2048(data: torch.Tensor) -> torch.Tensor:
        out = data.contiguous().clone()
        batch, n, _ = out.shape
        for k in range(0, n, _BK_8X2048):
            _diag_factor_8x2048[(batch,)](
                out,
                n=n,
                k=k,
                BK_CONST=_BK_8X2048,
                num_warps=8,
            )
            remaining = n - k - _BK_8X2048
            if remaining <= 0:
                break
            panel_tiles = triton.cdiv(remaining, _BK_8X2048)
            _panel_solve_8x2048[(panel_tiles, batch)](
                out,
                n=n,
                k=k,
                remaining=remaining,
                BK_CONST=_BK_8X2048,
                num_warps=8,
            )
            update_tiles = triton.cdiv(remaining, _UPDATE_TILE_8X2048)
            triangular_tiles = update_tiles * (update_tiles + 1) // 2
            _lower_schur_8x2048[(triangular_tiles, batch)](
                out,
                n=n,
                k=k,
                remaining=remaining,
                BK_CONST=_BK_8X2048,
                TILE=_UPDATE_TILE_8X2048,
                num_warps=8,
                num_stages=3,
            )

        total = batch * n * n
        clear_grid = 4096
        _clear_upper_8x2048[(clear_grid,)](
            out,
            total=total,
            n=n,
            BLOCK=256,
            GRID=clear_grid,
            num_warps=8,
        )
        return out


# ---------------------------------------------------------------------------
# Exact graph-replay paths for two overhead-bound ranked shapes.
# ---------------------------------------------------------------------------
_GRAPH_16X512 = None
_GRAPH_INPUT_16X512 = None
_GRAPH_OUTPUT_16X512 = None
_GRAPH_ERROR_16X512 = None

_GRAPH_256X128 = None
_GRAPH_ERROR_256X128 = None


def _graph_cholesky_16x512(data: torch.Tensor) -> torch.Tensor:
    global _GRAPH_16X512, _GRAPH_INPUT_16X512, _GRAPH_OUTPUT_16X512
    global _GRAPH_ERROR_16X512

    if _GRAPH_16X512 is None and _GRAPH_ERROR_16X512 is None:
        try:
            static_input = torch.empty_like(data)
            static_input.copy_(data)
            for _ in range(3):
                torch.linalg.cholesky_ex(
                    static_input, check_errors=False
                ).L
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = torch.linalg.cholesky_ex(
                    static_input, check_errors=False
                ).L
            graph.replay()
            _GRAPH_INPUT_16X512 = static_input
            _GRAPH_OUTPUT_16X512 = static_output
            _GRAPH_16X512 = graph
            return static_output.clone()
        except Exception as exc:  # pragma: no cover
            _GRAPH_ERROR_16X512 = repr(exc)
            return torch.linalg.cholesky_ex(data, check_errors=False).L

    if _GRAPH_16X512 is None:
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    _GRAPH_INPUT_16X512.copy_(data)
    _GRAPH_16X512.replay()
    return _GRAPH_OUTPUT_16X512.clone()


def _graph_cholesky_256x128(data: torch.Tensor) -> torch.Tensor:
    global _GRAPH_256X128, _GRAPH_ERROR_256X128
    if _GRAPH_256X128 is None:
        try:
            def _factor(x: torch.Tensor) -> torch.Tensor:
                return torch.linalg.cholesky_ex(x, check_errors=False).L

            _GRAPH_256X128 = torch.cuda.make_graphed_callables(
                _factor,
                (data,),
                num_warmup_iters=5,
            )
        except Exception as exc:  # pragma: no cover
            _GRAPH_ERROR_256X128 = repr(exc)
            _GRAPH_256X128 = False

    if _GRAPH_256X128 is False:
        return torch.linalg.cholesky_ex(data, check_errors=False).L
    return _GRAPH_256X128(data).clone()


# ---------------------------------------------------------------------------
# Large single-matrix left-looking paths (experiment 012).
# ---------------------------------------------------------------------------
_LEFT_16384_HITS = 0
_LEFT_32768_HITS = 0
_LEFT_32768_ERROR = None
_LEFT_LARGE_FALLBACKS = 0
_SUPERPANEL_32768_HITS = 0
_SUPERPANEL_32768_MICRO_POTRF_HITS = 0
_SUPERPANEL_32768_PANEL_MM_HITS = 0
_SUPERPANEL_32768_FP8_UPDATE_HITS = 0
_SUPERPANEL_32768_ERROR = None
_SUPERPANEL_32768_FALLBACKS = 0
_SUPERPANEL_32768_EXTENSION_READY = False
_SUPERPANEL_32768_EXTENSION_ERROR = None
_SUPERPANEL_32768_EXTENSION = None


def _clear_upper_large(matrix: torch.Tensor) -> torch.Tensor:
    if not _HAVE_TRITON:
        return torch.tril(matrix)
    grid = 4096
    _clear_upper_8x2048[(grid,)](
        matrix,
        total=matrix.numel(),
        n=matrix.shape[0],
        BLOCK=256,
        GRID=grid,
        num_warps=8,
    )
    return matrix


def _left_looking_cholesky_16384(mat: torch.Tensor) -> torch.Tensor:
    global _LEFT_16384_HITS

    nb = 2048
    n = mat.shape[0]
    a = mat.clone()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = a[k : k + kb, k : k + kb]
            if k:
                left = a[k : k + kb, :k]
                diagonal.addmm_(
                    left,
                    left.transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            diagonal_factor = torch.linalg.cholesky_ex(
                diagonal, check_errors=False
            ).L
            a[k : k + kb, k : k + kb] = diagonal_factor
            j = k + kb
            if j >= n:
                break
            panel = a[j:, k : k + kb]
            if k:
                panel.addmm_(
                    a[j:, :k],
                    a[k : k + kb, :k].transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            solved = torch.linalg.solve_triangular(
                diagonal_factor.transpose(-1, -2),
                panel,
                upper=True,
                left=False,
            )
            a[j:, k : k + kb] = solved
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _LEFT_16384_HITS += 1
    return _clear_upper_large(a)


def _scaled_mm_fp8_32768(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale_lhs: torch.Tensor,
    scale_rhs: torch.Tensor,
) -> torch.Tensor:
    try:
        result = torch._scaled_mm(
            lhs,
            rhs,
            scale_a=scale_lhs,
            scale_b=scale_rhs,
            out_dtype=torch.float32,
            use_fast_accum=True,
        )
    except TypeError:
        result = torch._scaled_mm(
            lhs,
            rhs,
            scale_a=scale_lhs,
            scale_b=scale_rhs,
            out_dtype=torch.float32,
        )
    return result[0] if isinstance(result, tuple) else result


def _fp8_product_32768(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    max_value = torch.finfo(torch.float8_e4m3fn).max
    scale_lhs = (
        max_value / lhs.abs().amax().clamp_min(2.0**-24)
    ).float()
    scale_rhs = (
        max_value / rhs.abs().amax().clamp_min(2.0**-24)
    ).float()
    quantized_lhs = (lhs * scale_lhs).to(torch.float8_e4m3fn)
    quantized_rhs = (rhs * scale_rhs).to(torch.float8_e4m3fn)
    return _scaled_mm_fp8_32768(
        quantized_lhs,
        quantized_rhs,
        scale_lhs.reciprocal(),
        scale_rhs.reciprocal(),
    )


def _left_looking_cholesky_32768(mat: torch.Tensor) -> torch.Tensor:
    global _LEFT_32768_HITS

    nb = 4096
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = mat[k : k + kb, k : k + kb].clone()
            if k:
                previous_row = factor[k : k + kb, :k]
                diagonal.addmm_(
                    previous_row,
                    previous_row.transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            diagonal_factor = torch.linalg.cholesky_ex(
                diagonal, check_errors=False
            ).L
            factor[k : k + kb, k : k + kb] = diagonal_factor
            j = k + kb
            if j >= n:
                break
            panel = mat[j:, k : k + kb].clone()
            if k:
                panel.sub_(
                    _fp8_product_32768(
                        factor[j:, :k],
                        factor[k : k + kb, :k].transpose(-1, -2),
                    )
                )
            identity = torch.eye(
                kb, device=mat.device, dtype=mat.dtype
            )
            inverse_transpose = torch.linalg.solve_triangular(
                diagonal_factor.transpose(-1, -2),
                identity,
                upper=True,
            )
            factor[j:, k : k + kb] = panel @ inverse_transpose
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _LEFT_32768_HITS += 1
    return factor


def _factor_active_superpanel_32768(
    superpanel: torch.Tensor,
    width: int,
) -> None:
    """Factor/solve a tall active superpanel with CUDA 128x128 POTRF tiles."""
    global _SUPERPANEL_32768_MICRO_POTRF_HITS
    global _SUPERPANEL_32768_PANEL_MM_HITS

    micro = 128
    for p in range(0, width, micro):
        pb = min(micro, width - p)
        diagonal = superpanel[p : p + pb, p : p + pb]
        diagonal_factor = diagonal.contiguous().clone()
        _load_superpanel_extension_32768().potrf128_cuda(diagonal_factor)
        superpanel[p : p + pb, p : p + pb] = diagonal_factor
        _SUPERPANEL_32768_MICRO_POTRF_HITS += 1

        row_start = p + pb
        if row_start >= superpanel.shape[0]:
            break
        active_rows = superpanel[row_start:, p : p + pb]
        identity = torch.eye(pb, device=superpanel.device, dtype=superpanel.dtype)
        inverse_transpose = torch.linalg.solve_triangular(
            diagonal_factor.transpose(-1, -2),
            identity,
            upper=True,
        )
        solved = active_rows @ inverse_transpose
        superpanel[row_start:, p : p + pb] = solved
        _SUPERPANEL_32768_PANEL_MM_HITS += 1

        remaining_columns = width - row_start
        if remaining_columns:
            superpanel[row_start:, row_start:width].addmm_(
                solved,
                solved[:remaining_columns].transpose(-1, -2),
                beta=1.0,
                alpha=-1.0,
            )


_POTRF128_CPP = r"""
#include <torch/extension.h>

torch::Tensor potrf128_cuda(torch::Tensor input);
"""


_POTRF128_CUDA = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void potrf128_kernel(float* matrix) {
    constexpr int N = 128;
    constexpr int STRIDE = 129;
    extern __shared__ float tile[];
    const int tid = threadIdx.x;

    for (int offset = tid; offset < N * N; offset += blockDim.x) {
        const int row = offset / N;
        const int column = offset - row * N;
        tile[row * STRIDE + column] = matrix[offset];
    }
    __syncwarp();

    for (int k = 0; k < N; ++k) {
        if (tid == 0) {
            tile[k * STRIDE + k] = sqrtf(
                fmaxf(tile[k * STRIDE + k], 1.0e-20f));
        }
        __syncwarp();
        const float diagonal = tile[k * STRIDE + k];
        for (int row = k + 1 + tid; row < N; row += blockDim.x) {
            tile[row * STRIDE + k] /= diagonal;
        }
        __syncwarp();
        for (int row = k + 1 + tid; row < N; row += blockDim.x) {
            const float column_value = tile[row * STRIDE + k];
            for (int column = k + 1; column <= row; ++column) {
                tile[row * STRIDE + column] -=
                    column_value * tile[column * STRIDE + k];
            }
        }
        __syncwarp();
    }

    for (int offset = tid; offset < N * N; offset += blockDim.x) {
        const int row = offset / N;
        const int column = offset - row * N;
        matrix[offset] = column <= row
            ? tile[row * STRIDE + column]
            : 0.0f;
    }
}

torch::Tensor potrf128_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "potrf128 input must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "potrf128 requires FP32");
    TORCH_CHECK(input.is_contiguous(), "potrf128 input must be contiguous");
    TORCH_CHECK(input.dim() == 2 && input.size(0) == 128 && input.size(1) == 128,
                "potrf128 input must be 128x128");
    static bool configured = false;
    if (!configured) {
        const cudaError_t status = cudaFuncSetAttribute(
            potrf128_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            128 * 129 * static_cast<int>(sizeof(float)));
        TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
        configured = true;
    }
    potrf128_kernel<<<1, 32, 128 * 129 * sizeof(float)>>>(
        input.data_ptr<float>());
    const cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    return input;
}
"""


def _load_superpanel_extension_32768():
    global _SUPERPANEL_32768_EXTENSION
    global _SUPERPANEL_32768_EXTENSION_READY
    global _SUPERPANEL_32768_EXTENSION_ERROR

    if _SUPERPANEL_32768_EXTENSION is None:
        try:
            from torch.utils.cpp_extension import load_inline

            _SUPERPANEL_32768_EXTENSION = load_inline(
                name="chol_exp013_potrf128_v3_warp",
                cpp_sources=_POTRF128_CPP,
                cuda_sources=_POTRF128_CUDA,
                functions=["potrf128_cuda"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=False,
            )
            _SUPERPANEL_32768_EXTENSION_READY = True
            _SUPERPANEL_32768_EXTENSION_ERROR = None
        except Exception as exc:
            _SUPERPANEL_32768_EXTENSION_ERROR = repr(exc)
            raise
    return _SUPERPANEL_32768_EXTENSION


def _active_superpanel_cholesky_32768(mat: torch.Tensor) -> torch.Tensor:
    """Left-looking active-superpanel Cholesky for exactly 1x32768."""
    global _SUPERPANEL_32768_HITS, _SUPERPANEL_32768_FP8_UPDATE_HITS

    nb = 4096
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            superpanel = mat[k:, k : k + kb].clone()
            if k:
                previous_columns = factor[k:, :k]
                previous_row = factor[k : k + kb, :k]
                superpanel[:kb].addmm_(
                    previous_row,
                    previous_row.transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
                if superpanel.shape[0] > kb:
                    superpanel[kb:].sub_(
                        _fp8_product_32768(
                            previous_columns[kb:],
                            previous_row.transpose(-1, -2),
                        )
                    )
                    _SUPERPANEL_32768_FP8_UPDATE_HITS += 1
            _factor_active_superpanel_32768(superpanel, kb)
            factor[k:, k : k + kb] = superpanel
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _SUPERPANEL_32768_HITS += 1
    return _clear_upper_large(factor)


# ---------------------------------------------------------------------------
# Small-batch / large-n path (experiment 004, region trimmed by exp 005).
# ---------------------------------------------------------------------------
def _loop_cholesky(data: torch.Tensor) -> torch.Tensor:
    """Sequential per-matrix single-matrix potrf, then stack. Avoids the slow
    batched cuSOLVER path for few-but-large matrices."""
    batch = data.shape[0]
    return torch.stack(
        [
            torch.linalg.cholesky_ex(data[i], check_errors=False).L
            for i in range(batch)
        ]
    )


# ---------------------------------------------------------------------------
# Large single-matrix path (experiments 006 + 008): blocked right-looking
# Cholesky with a fused in-place TF32 trailing update. Diagonal block + panel
# solve stay FP32.
# ---------------------------------------------------------------------------
def _blocked_cholesky_tf32(mat: torch.Tensor, nb: int) -> torch.Tensor:
    """Right-looking blocked Cholesky of a single (n, n) FP32 SPD matrix.

    The trailing Schur update (the O(n^3) cost) runs on tensor cores in TF32;
    the diagonal block factorization and the panel triangular solve stay FP32.
    Returns an FP32 lower-triangular factor. Default-queue only.
    """
    a = mat.clone()
    n = a.shape[0]
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            a11 = a[k : k + kb, k : k + kb]
            l11 = torch.linalg.cholesky_ex(a11, check_errors=False).L
            a[k : k + kb, k : k + kb] = l11
            j = k + kb
            if j >= n:
                break
            a21 = a[j:, k : k + kb]
            # Solve L21 @ L11^T = A21 for the panel factor (FP32 TRSM).
            l21 = torch.linalg.solve_triangular(
                l11.transpose(-1, -2), a21, upper=True, left=False
            )
            a[j:, k : k + kb] = l21
            # Fused trailing Schur update on TF32 tensor cores (FP32 accumulate).
            # Writing directly into the strided trailing view avoids materializing
            # a full product followed by a separate subtraction kernel.
            a[j:, j:].addmm_(
                l21, l21.transpose(-1, -2), beta=1.0, alpha=-1.0
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return torch.tril(a)


def custom_kernel(data: input_t) -> output_t:
    global _LEFT_32768_ERROR, _LEFT_LARGE_FALLBACKS
    global _SUPERPANEL_32768_ERROR

    batch, n, _ = data.shape
    is_f32_cuda = data.is_cuda and data.dtype == torch.float32

    if is_f32_cuda and _HAVE_TRITON and n == 32:
        return _triton_cholesky32(data)

    if is_f32_cuda and batch == 256 and n == 128:
        return _graph_cholesky_256x128(data)

    if is_f32_cuda and batch == 16 and n == 512:
        return _graph_cholesky_16x512(data)

    if is_f32_cuda and _HAVE_TRITON and batch == 8 and n == 2048:
        l = _triton_cholesky_8x2048(data)
        if torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all().item():
            return l
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    if is_f32_cuda and batch == 1 and n == 16384:
        try:
            l = _left_looking_cholesky_16384(data[0])
            if torch.isfinite(l.diagonal()).all().item():
                return l.unsqueeze(0)
        except Exception:
            pass
        _LEFT_LARGE_FALLBACKS += 1
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    if is_f32_cuda and batch == 1 and n == 32768:
        try:
            l = _active_superpanel_cholesky_32768(data[0])
            _SUPERPANEL_32768_ERROR = None
            return l.unsqueeze(0)
        except Exception as exc:
            _SUPERPANEL_32768_ERROR = repr(exc)
            raise

    # Large single matrices: blocked Cholesky with a TF32 tensor-core trailing
    # update beats cuSOLVER's all-FP32 potrf (exp 006), with the product and
    # subtraction fused in-place by exp 008. Only the measured-win
    # region (batch==1, n>=16384); 8192 was only ~1.07x so it stays on cuSOLVER.
    if is_f32_cuda and batch == 1 and n >= 16384:
        nb = 4096 if n >= 32768 else 2048
        l = _blocked_cholesky_tf32(data[0], nb)
        # Numerical safety net: TF32 error can drive a late diagonal block
        # indefinite on ill-conditioned inputs (spectrum/lowrank), yielding
        # NaN/Inf. The ranked shapes are well-conditioned dense (huge margin,
        # never trips this), but fall back to exact FP32 cuSOLVER otherwise so
        # correctness holds across every family. isfinite is ~memory-bound and
        # negligible vs the O(n^3) factorization.
        if torch.isfinite(l).all().item():
            return l.unsqueeze(0)
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    # Few-but-large matrices: avoid cusolverDnSpotrfBatched (see module docstring).
    # exp 005: upper bound trimmed 8->4 so 8x2048 stays on batched cuSOLVER.
    if is_f32_cuda and 2 <= batch <= 4 and n >= 1024:
        return _loop_cholesky(data)

    # Default: batched cuSOLVER. Correct for every input family.
    return torch.linalg.cholesky_ex(data, check_errors=False).L
