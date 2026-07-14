#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission.

Batched dense Cholesky factorization. Input `A` is a `batch x n x n` float32
CUDA tensor, SPD up to FP32 roundoff. Return lower-triangular float32 `L` with
positive diagonal such that `A = L @ L.T`.

Shape dispatcher:
  * n == 32          -> Triton batched kernel (one warp per matrix). Adopted in
    experiment 002; beats cuSOLVER's batched-launch overhead.
  * n in {64, 128}   -> custom CUDA kernel (one thread-block per matrix, matrix
    in shared memory, right-looking factorization). Experiment 003. Compiled on
    the runner via load_inline and cached at module scope. Falls back to cuSOLVER
    if nvcc/compilation is unavailable so correctness is never at risk.
  * everything else  -> cuSOLVER via torch.linalg.cholesky_ex (already strong).
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

    def _triton_cholesky(data: torch.Tensor) -> torch.Tensor:
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


# ---------------------------------------------------------------------------
# CUDA kernel for n in {64, 128} (experiment 003).
# One thread-block per matrix, N threads. The matrix lives in shared memory;
# thread t owns column t. Right-looking Cholesky: at step k scale column k by
# 1/sqrt(A[k,k]) then rank-1 update the trailing lower triangle. Dynamic shared
# memory (n=128 needs 64KB, above the 48KB static limit) via a raw byte buffer
# to dodge templated `extern __shared__` type conflicts.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// One WARP (32 lanes) per matrix. The matrix lives in shared memory; the warp
// runs the right-looking factorization with cheap __syncwarp() ordering instead
// of block-wide __syncthreads (the block-per-matrix variant was sync-bound).
template <int N>
__global__ void chol_warp(const float* __restrict__ A, float* __restrict__ L) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* s = reinterpret_cast<float*>(smem_raw);

    const int lane = threadIdx.x;                  // 0..31
    const float* Ab = A + (size_t)blockIdx.x * N * N;
    float* Lb = L + (size_t)blockIdx.x * N * N;

    for (int idx = lane; idx < N * N; idx += 32) s[idx] = Ab[idx];
    __syncwarp();

    for (int k = 0; k < N; ++k) {
        const float d = sqrtf(s[k * N + k]);       // every lane reads pivot
        __syncwarp();
        for (int i = k + 1 + lane; i < N; i += 32) s[i * N + k] /= d;
        if (lane == 0) s[k * N + k] = d;           // finalize L[k,k]
        __syncwarp();
        for (int i = k + 1 + lane; i < N; i += 32) {
            const float lik = s[i * N + k];
            for (int j = k + 1; j <= i; ++j) {     // rank-1 update, lower part
                s[i * N + j] -= lik * s[j * N + k];
            }
        }
        __syncwarp();
    }

    for (int r = 0; r < N; ++r) {
        for (int c = lane; c < N; c += 32) {
            Lb[r * N + c] = (c <= r) ? s[r * N + c] : 0.0f;
        }
    }
}

void chol_launch(torch::Tensor A, torch::Tensor L) {
    const int batch = A.size(0);
    const int n = A.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(batch);
    dim3 block(32);                                 // one warp per matrix
    size_t shmem = (size_t)n * n * sizeof(float);
    if (n == 64) {
        chol_warp<64><<<grid, block, shmem, stream>>>(
            A.data_ptr<float>(), L.data_ptr<float>());
    } else if (n == 128) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(chol_warp<128>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
            configured = true;
        }
        chol_warp<128><<<grid, block, shmem, stream>>>(
            A.data_ptr<float>(), L.data_ptr<float>());
    } else {
        TORCH_CHECK(false, "chol_launch: unsupported n=", n);
    }
}
"""

_CUDA_MOD = None
_CUDA_LOAD_ERROR = None
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA_MOD = load_inline(
            name="chol_cuda_ext",
            cpp_sources="void chol_launch(torch::Tensor A, torch::Tensor L);",
            cuda_sources=_CUDA_SRC,
            functions=["chol_launch"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - fall back to cuSOLVER if nvcc missing
        import traceback

        _CUDA_MOD = None
        _CUDA_LOAD_ERROR = "".join(
            traceback.format_exception_only(type(exc), exc)
        ) + (traceback.format_exc()[-2000:])


def _cuda_cholesky(data: torch.Tensor) -> torch.Tensor:
    data = data.contiguous()
    out = torch.empty_like(data)
    _CUDA_MOD.chol_launch(data, out)
    return out


_CUDA_NS = (64, 128)


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    is_f32_cuda = data.is_cuda and data.dtype == torch.float32
    if is_f32_cuda and _HAVE_TRITON and n == 32:
        return _triton_cholesky(data)
    if is_f32_cuda and _CUDA_MOD is not None and n in _CUDA_NS:
        return _cuda_cholesky(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
