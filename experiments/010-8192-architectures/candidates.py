"""Compiled experiment-010 candidates for the exact 1 x 8192 shape.

This module is isolated from the ranked source. A failed build is surfaced to
the probe runner and is never treated as a valid timing.
"""

import traceback

import torch


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mutex>
#include <vector>

#define CUBLAS_OK(expr) TORCH_CHECK((expr) == CUBLAS_STATUS_SUCCESS, #expr)
#define CUSOLVER_OK(expr) TORCH_CHECK((expr) == CUSOLVER_STATUS_SUCCESS, #expr)

namespace {

cublasHandle_t g_blas;
cusolverDnHandle_t g_solver;
std::once_flag g_once;

void init_handles() {
    std::call_once(g_once, [] {
        CUBLAS_OK(cublasCreate(&g_blas));
        CUSOLVER_OK(cusolverDnCreate(&g_solver));
    });
}

__global__ void clear_row_upper(float* a, int n) {
    const long long total = static_cast<long long>(n) * n;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / n);
        const int col = static_cast<int>(idx - static_cast<long long>(row) * n);
        if (col > row) a[idx] = 0.0f;
    }
}

__global__ void set_one_pointer(float** slot, float* value) {
    if (threadIdx.x == 0) *slot = value;
}

void check_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3 && input.size(0) == 1, "expected batch one");
    TORCH_CHECK(input.size(1) == input.size(2), "expected square matrix");
}

void finish_lower(torch::Tensor& out, int n) {
    clear_row_upper<<<4096, 256>>>(out.data_ptr<float>(), n);
}

}  // namespace

torch::Tensor legacy_potrf(torch::Tensor input) {
    check_input(input);
    init_handles();
    auto out = input.clone();
    const int n = static_cast<int>(input.size(1));
    int lwork = 0;
    CUSOLVER_OK(cusolverDnSpotrf_bufferSize(
        g_solver, CUBLAS_FILL_MODE_UPPER, n, out.data_ptr<float>(), n, &lwork));
    auto workspace = torch::empty({lwork}, input.options());
    auto info = torch::empty({1}, input.options().dtype(torch::kInt32));
    CUSOLVER_OK(cusolverDnSpotrf(
        g_solver, CUBLAS_FILL_MODE_UPPER, n, out.data_ptr<float>(), n,
        workspace.data_ptr<float>(), lwork, info.data_ptr<int>()));
    finish_lower(out, n);
    return out;
}

torch::Tensor expert_potrf(torch::Tensor input) {
    check_input(input);
    init_handles();
    auto out = input.clone();
    const int64_t n = input.size(1);
    cusolverDnParams_t params = nullptr;
    CUSOLVER_OK(cusolverDnCreateParams(&params));
    size_t device_bytes = 0;
    size_t host_bytes = 0;
    CUSOLVER_OK(cusolverDnXpotrf_bufferSize(
        g_solver, params, CUBLAS_FILL_MODE_UPPER, n, CUDA_R_32F,
        out.data_ptr<float>(), n, CUDA_R_32F, &device_bytes, &host_bytes));
    auto device_workspace = torch::empty(
        {static_cast<int64_t>(device_bytes)}, input.options().dtype(torch::kUInt8));
    std::vector<unsigned char> host_workspace(host_bytes);
    auto info = torch::empty({1}, input.options().dtype(torch::kInt32));
    CUSOLVER_OK(cusolverDnXpotrf(
        g_solver, params, CUBLAS_FILL_MODE_UPPER, n, CUDA_R_32F,
        out.data_ptr<float>(), n, CUDA_R_32F,
        device_workspace.data_ptr(), device_bytes,
        host_workspace.data(), host_bytes, info.data_ptr<int>()));
    CUSOLVER_OK(cusolverDnDestroyParams(params));
    finish_lower(out, static_cast<int>(n));
    return out;
}

torch::Tensor batched_potrf(torch::Tensor input) {
    check_input(input);
    init_handles();
    auto out = input.clone();
    const int n = static_cast<int>(input.size(1));
    auto pointer = torch::empty({1}, input.options().dtype(torch::kInt64));
    auto info = torch::empty({1}, input.options().dtype(torch::kInt32));
    auto slot = reinterpret_cast<float**>(pointer.data_ptr<int64_t>());
    set_one_pointer<<<1, 1>>>(slot, out.data_ptr<float>());
    CUSOLVER_OK(cusolverDnSpotrfBatched(
        g_solver, CUBLAS_FILL_MODE_UPPER, n, slot, n, info.data_ptr<int>(), 1));
    finish_lower(out, n);
    return out;
}

torch::Tensor blocked_syrk(torch::Tensor input, int64_t block, bool tf32) {
    check_input(input);
    init_handles();
    auto out = input.clone();
    const int n = static_cast<int>(input.size(1));
    const int nb = static_cast<int>(block);
    TORCH_CHECK(nb > 0 && nb <= n, "invalid block size");

    CUBLAS_OK(cublasSetMathMode(
        g_blas, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));

    int max_lwork = 0;
    CUSOLVER_OK(cusolverDnSpotrf_bufferSize(
        g_solver, CUBLAS_FILL_MODE_UPPER, std::min(nb, n),
        out.data_ptr<float>(), n, &max_lwork));
    auto workspace = torch::empty({max_lwork}, input.options());
    auto info = torch::empty({1}, input.options().dtype(torch::kInt32));
    float* base = out.data_ptr<float>();
    const float one = 1.0f;
    const float minus_one = -1.0f;

    for (int k = 0; k < n; k += nb) {
        const int kb = std::min(nb, n - k);
        float* diagonal = base + static_cast<long long>(k) * n + k;
        int lwork = max_lwork;
        if (kb != std::min(nb, n)) {
            CUSOLVER_OK(cusolverDnSpotrf_bufferSize(
                g_solver, CUBLAS_FILL_MODE_UPPER, kb, diagonal, n, &lwork));
        }
        CUSOLVER_OK(cusolverDnSpotrf(
            g_solver, CUBLAS_FILL_MODE_UPPER, kb, diagonal, n,
            workspace.data_ptr<float>(), lwork, info.data_ptr<int>()));

        const int j = k + kb;
        const int remaining = n - j;
        if (remaining == 0) break;
        float* panel = base + static_cast<long long>(j) * n + k;
        CUBLAS_OK(cublasStrsm(
            g_blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            kb, remaining, &one, diagonal, n, panel, n));
        float* trailing = base + static_cast<long long>(j) * n + j;
        CUBLAS_OK(cublasSsyrk(
            g_blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            remaining, kb, &minus_one, panel, n, &one, trailing, n));
    }

    finish_lower(out, n);
    return out;
}

torch::Tensor blocked_batched_diag_syrk(torch::Tensor input, int64_t block) {
    check_input(input);
    init_handles();
    auto out = input.clone();
    const int n = static_cast<int>(input.size(1));
    const int nb = static_cast<int>(block);
    auto pointer = torch::empty({1}, input.options().dtype(torch::kInt64));
    auto info = torch::empty({1}, input.options().dtype(torch::kInt32));
    auto slot = reinterpret_cast<float**>(pointer.data_ptr<int64_t>());
    float* base = out.data_ptr<float>();
    const float one = 1.0f;
    const float minus_one = -1.0f;
    CUBLAS_OK(cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH));

    for (int k = 0; k < n; k += nb) {
        const int kb = std::min(nb, n - k);
        float* diagonal = base + static_cast<long long>(k) * n + k;
        set_one_pointer<<<1, 1>>>(slot, diagonal);
        CUSOLVER_OK(cusolverDnSpotrfBatched(
            g_solver, CUBLAS_FILL_MODE_UPPER, kb, slot, n,
            info.data_ptr<int>(), 1));
        const int j = k + kb;
        const int remaining = n - j;
        if (remaining == 0) break;
        float* panel = base + static_cast<long long>(j) * n + k;
        CUBLAS_OK(cublasStrsm(
            g_blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            kb, remaining, &one, diagonal, n, panel, n));
        float* trailing = base + static_cast<long long>(j) * n + j;
        CUBLAS_OK(cublasSsyrk(
            g_blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            remaining, kb, &minus_one, panel, n, &one, trailing, n));
    }
    finish_lower(out, n);
    return out;
}
"""


_CUDA_MOD = None
_CUDA_LOAD_ERROR = None
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA_MOD = load_inline(
            name="chol_exp010_arch_v1",
            cpp_sources="""
                torch::Tensor legacy_potrf(torch::Tensor input);
                torch::Tensor expert_potrf(torch::Tensor input);
                torch::Tensor batched_potrf(torch::Tensor input);
                torch::Tensor blocked_syrk(torch::Tensor input, int64_t block, bool tf32);
                torch::Tensor blocked_batched_diag_syrk(torch::Tensor input, int64_t block);
            """,
            cuda_sources=_CUDA_SRC,
            functions=[
                "legacy_potrf",
                "expert_potrf",
                "batched_potrf",
                "blocked_syrk",
                "blocked_batched_diag_syrk",
            ],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_ldflags=["-lcublas", "-lcusolver"],
            verbose=False,
        )
    except Exception as exc:  # surfaced as a hard probe failure
        _CUDA_LOAD_ERROR = "".join(traceback.format_exception_only(type(exc), exc))
        _CUDA_LOAD_ERROR += traceback.format_exc()[-4000:]


def require_compiled() -> None:
    if _CUDA_MOD is None:
        raise RuntimeError(f"experiment-010 extension unavailable:\n{_CUDA_LOAD_ERROR}")


def legacy_potrf(data: torch.Tensor) -> torch.Tensor:
    require_compiled()
    return _CUDA_MOD.legacy_potrf(data.contiguous())


def expert_potrf(data: torch.Tensor) -> torch.Tensor:
    require_compiled()
    return _CUDA_MOD.expert_potrf(data.contiguous())


def batched_potrf(data: torch.Tensor) -> torch.Tensor:
    require_compiled()
    return _CUDA_MOD.batched_potrf(data.contiguous())


def blocked_syrk_tf32(data: torch.Tensor, block: int) -> torch.Tensor:
    require_compiled()
    return _CUDA_MOD.blocked_syrk(data.contiguous(), block, True)


def blocked_syrk_fp32(data: torch.Tensor, block: int) -> torch.Tensor:
    require_compiled()
    return _CUDA_MOD.blocked_syrk(data.contiguous(), block, False)


def blocked_batched_diag_syrk(data: torch.Tensor, block: int) -> torch.Tensor:
    require_compiled()
    return _CUDA_MOD.blocked_batched_diag_syrk(data.contiguous(), block)


_GRAPH_BLOCKED = None


def graph_blocked_syrk_tf32(data: torch.Tensor) -> torch.Tensor:
    """Capture the compiled lower-only pipeline and return owned output."""
    global _GRAPH_BLOCKED
    require_compiled()
    if _GRAPH_BLOCKED is None:
        static_input = data.clone()
        for _ in range(3):
            static_output = _CUDA_MOD.blocked_syrk(static_input, 4096, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = _CUDA_MOD.blocked_syrk(static_input, 4096, True)
        _GRAPH_BLOCKED = (static_input, static_output, graph)
    static_input, static_output, graph = _GRAPH_BLOCKED
    static_input.copy_(data)
    graph.replay()
    return static_output.clone()


try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


if _HAVE_TRITON:
    _BK = 64
    _TILE = 128

    @triton.jit
    def _diag_factor(a_ptr, n: tl.constexpr, k, BK: tl.constexpr):
        rows = tl.arange(0, BK)
        cols = tl.arange(0, BK)
        ptrs = a_ptr + (k + rows[:, None]) * n + k + cols[None, :]
        tile = tl.load(ptrs)
        for p in range(0, BK):
            diagonal = tl.sum(
                tl.where((rows[:, None] == p) & (cols[None, :] == p), tile, 0.0)
            )
            inv_sqrt = 1.0 / tl.sqrt(diagonal)
            tile = tl.where(
                (cols[None, :] == p) & (rows[:, None] >= p),
                tile * inv_sqrt,
                tile,
            )
            column = tl.sum(tl.where(cols[None, :] == p, tile, 0.0), axis=1)
            tile = tl.where(
                (rows[:, None] > p) & (cols[None, :] > p),
                tile - column[:, None] * column[None, :],
                tile,
            )
        tl.store(ptrs, tile, mask=cols[None, :] <= rows[:, None])

    @triton.jit
    def _panel_solve(a_ptr, n: tl.constexpr, k, remaining, BK: tl.constexpr):
        tile_id = tl.program_id(0)
        rows = tile_id * BK + tl.arange(0, BK)
        cols = tl.arange(0, BK)
        valid_rows = rows < remaining
        diag_ptrs = a_ptr + (k + cols[:, None]) * n + k + cols[None, :]
        diagonal = tl.load(diag_ptrs)
        panel_ptrs = a_ptr + (k + BK + rows[:, None]) * n + k + cols[None, :]
        panel = tl.load(panel_ptrs, mask=valid_rows[:, None], other=0.0)
        for p in range(0, BK):
            diag_column = tl.sum(
                tl.where(cols[None, :] == p, diagonal, 0.0), axis=1
            )
            diag_pp = tl.sum(tl.where(cols == p, diag_column, 0.0), axis=0)
            value = tl.sum(
                tl.where(cols[None, :] == p, panel, 0.0), axis=1
            ) / diag_pp
            panel = tl.where(cols[None, :] == p, value[:, None], panel)
            panel = tl.where(
                cols[None, :] > p,
                panel - value[:, None] * diag_column[None, :],
                panel,
            )
        tl.store(panel_ptrs, panel, mask=valid_rows[:, None])

    @triton.jit
    def _lower_update(
        a_ptr,
        n: tl.constexpr,
        k,
        remaining,
        BK: tl.constexpr,
        TILE: tl.constexpr,
        USE_BF16: tl.constexpr,
    ):
        triangular_id = tl.program_id(0)
        block_row = ((tl.sqrt(8.0 * triangular_id + 1.0) - 1.0) * 0.5).to(tl.int32)
        block_col = triangular_id - block_row * (block_row + 1) // 2
        rows = block_row * TILE + tl.arange(0, TILE)
        cols = block_col * TILE + tl.arange(0, TILE)
        depth = tl.arange(0, BK)
        lhs_ptrs = a_ptr + (k + BK + rows[:, None]) * n + k + depth[None, :]
        rhs_ptrs = a_ptr + (k + BK + cols[None, :]) * n + k + depth[:, None]
        lhs = tl.load(lhs_ptrs, mask=rows[:, None] < remaining, other=0.0)
        rhs = tl.load(rhs_ptrs, mask=cols[None, :] < remaining, other=0.0)
        if USE_BF16:
            product = tl.dot(
                lhs.to(tl.bfloat16), rhs.to(tl.bfloat16), out_dtype=tl.float32
            )
        else:
            product = tl.dot(
                lhs, rhs, input_precision="tf32", out_dtype=tl.float32
            )
        out_ptrs = a_ptr + (k + BK + rows[:, None]) * n + k + BK + cols[None, :]
        valid = (rows[:, None] < remaining) & (cols[None, :] < remaining)
        valid = valid & (
            (block_row != block_col) | (cols[None, :] <= rows[:, None])
        )
        old = tl.load(out_ptrs, mask=valid, other=0.0)
        tl.store(out_ptrs, old - product, mask=valid)

    @triton.jit
    def _clear_upper(
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
            row = offsets // n
            col = offsets - row * n
            tl.store(a_ptr + offsets, 0.0, mask=valid & (col > row))


def triton_lower_blocked(data: torch.Tensor, use_bf16: bool) -> torch.Tensor:
    if not _HAVE_TRITON:
        raise RuntimeError("Triton unavailable")
    out = data.contiguous().clone()
    n = out.shape[-1]
    for k in range(0, n, _BK):
        _diag_factor[(1,)](out, n=n, k=k, BK=_BK, num_warps=8)
        remaining = n - k - _BK
        if remaining <= 0:
            break
        _panel_solve[(triton.cdiv(remaining, _BK),)](
            out, n=n, k=k, remaining=remaining, BK=_BK, num_warps=8
        )
        blocks = triton.cdiv(remaining, _TILE)
        triangular = blocks * (blocks + 1) // 2
        _lower_update[(triangular,)](
            out,
            n=n,
            k=k,
            remaining=remaining,
            BK=_BK,
            TILE=_TILE,
            USE_BF16=use_bf16,
            num_warps=8,
            num_stages=3,
        )
    total = n * n
    _clear_upper[(4096,)](
        out, total=total, n=n, BLOCK=256, GRID=4096, num_warps=8
    )
    return out
