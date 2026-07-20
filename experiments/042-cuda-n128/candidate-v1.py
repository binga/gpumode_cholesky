"""Experiment 042 V1: four-warp rank-2 CUDA Cholesky for 256x128."""

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
constexpr int SHARED_BYTES = N * TILE_STRIDE * sizeof(float);

__global__ void cholesky128_rank2(const float* input, float* output) {
    const int row = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * N * N;
    extern __shared__ float shared[];
    float* tile = shared;
    float* pivot0 = shared;
    float* pivot1 = shared + N;
    float* reciprocal0 = shared + 2 * N;
    float* reciprocal1 = shared + 2 * N + 1;

    for (int linear = row; linear < N * N; linear += N) {
        const int r = linear >> 7;
        const int c = linear & 127;
        tile[r * TILE_STRIDE + c] = input[base + linear];
    }
    __syncthreads();

    float values[N];
    #pragma unroll
    for (int column = 0; column < N; ++column) {
        values[column] = tile[row * TILE_STRIDE + column];
    }

    #pragma unroll
    for (int iteration = 0; iteration < N / 2; ++iteration) {
        const int k = 2 * iteration;
        const int q = k + 1;
        if (row == k) *reciprocal0 = rsqrtf(values[k]);
        __syncthreads();

        if (row >= k) values[k] *= *reciprocal0;
        pivot0[row] = values[k];
        if (row == q) {
            values[q] = fmaf(-values[k], values[k], values[q]);
            *reciprocal1 = rsqrtf(values[q]);
        }
        __syncthreads();

        if (row >= q) {
            if (row != q) {
                values[q] = fmaf(-values[k], pivot0[q], values[q]);
            }
            values[q] *= *reciprocal1;
        }
        pivot1[row] = values[q];
        __syncthreads();

        if (row > q) {
            const float scale0 = values[k];
            const float scale1 = values[q];
            #pragma unroll
            for (int column = 0; column < N; ++column) {
                if (column > q && column <= row) {
                    float value = fmaf(
                        -scale0, pivot0[column], values[column]);
                    values[column] = fmaf(
                        -scale1, pivot1[column], value);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int column = 0; column < N; ++column) {
        tile[row * TILE_STRIDE + column] =
            column <= row ? values[column] : 0.0f;
    }
    __syncthreads();
    for (int linear = row; linear < N * N; linear += N) {
        const int r = linear >> 7;
        const int c = linear & 127;
        output[base + linear] = tile[r * TILE_STRIDE + c];
    }
}

void chol128_launch(torch::Tensor input, torch::Tensor output) {
    static bool configured = false;
    if (!configured) {
        cudaError_t attr = cudaFuncSetAttribute(
            cholesky128_rank2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHARED_BYTES);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    const int batch = (int)input.size(0);
    cholesky128_rank2<<<dim3(batch), dim3(N), SHARED_BYTES>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA128 = load_inline(
            name="chol128_exp042_v1",
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
