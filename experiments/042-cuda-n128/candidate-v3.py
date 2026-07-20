"""Experiment 042 V3: compact-loop eight-warp shared CUDA Cholesky."""

import torch
import submission as _ranked


_CUDA128_HITS = 0
_CUDA128_ERROR = None
_CUDA128 = None

_CUDA128_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int N = 128;
constexpr int TILE_STRIDE = 129;
constexpr int THREADS = 256;
constexpr int SHARED_BYTES = N * TILE_STRIDE * sizeof(float);

__global__ void cholesky128_shared(const float* input, float* output) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t base = (size_t)blockIdx.x * N * N;
    extern __shared__ float tile[];
    __shared__ float reciprocal;

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 7;
        const int column = linear & 127;
        tile[row * TILE_STRIDE + column] = input[base + linear];
    }
    __syncthreads();

    for (int k = 0; k < N; ++k) {
        if (tid == 0) {
            reciprocal = rsqrtf(tile[k * TILE_STRIDE + k]);
            tile[k * TILE_STRIDE + k] *= reciprocal;
        }
        __syncthreads();

        for (int row = k + 1 + tid; row < N; row += THREADS) {
            tile[row * TILE_STRIDE + k] *= reciprocal;
        }
        __syncthreads();

        for (int row = k + 1 + warp; row < N; row += 8) {
            const float scale = tile[row * TILE_STRIDE + k];
            for (int column = k + 1 + lane; column <= row; column += 32) {
                const int offset = row * TILE_STRIDE + column;
                tile[offset] = fmaf(
                    -scale,
                    tile[column * TILE_STRIDE + k],
                    tile[offset]);
            }
        }
        __syncthreads();
    }

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 7;
        const int column = linear & 127;
        output[base + linear] =
            column <= row ? tile[row * TILE_STRIDE + column] : 0.0f;
    }
}

void chol128_launch(torch::Tensor input, torch::Tensor output) {
    static bool configured = false;
    if (!configured) {
        cudaError_t attr = cudaFuncSetAttribute(
            cholesky128_shared,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHARED_BYTES);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    const int batch = (int)input.size(0);
    cholesky128_shared<<<dim3(batch), dim3(THREADS), SHARED_BYTES>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA128 = load_inline(
            name="chol128_exp042_v3",
            cpp_sources="void chol128_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA128_SOURCE,
            functions=["chol128_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA128_ERROR = repr(exc)


def custom_kernel(data):
    global _CUDA128_HITS
    if (
        _CUDA128 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and data.shape == (256, 128, 128)
    ):
        out = torch.empty_like(data)
        _CUDA128.chol128_launch(data, out)
        _CUDA128_HITS += 1
        return out
    return _ranked.custom_kernel(data)
