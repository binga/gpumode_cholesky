"""Experiment 041 V3: two-warp rank-2 CUDA Cholesky for 1024x64."""

import torch
import submission as _ranked


_CUDA64_HITS = 0
_CUDA64_ERROR = None
_CUDA64 = None

_CUDA64_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int N = 64;

__global__ void cholesky64_rank2(const float* input, float* output) {
    const int row = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * N * N;
    __shared__ float tile[64][65];
    __shared__ float pivot0[64];
    __shared__ float pivot1[64];
    __shared__ float reciprocal0;
    __shared__ float reciprocal1;

    for (int linear = row; linear < N * N; linear += 64) {
        tile[linear >> 6][linear & 63] = input[base + linear];
    }
    __syncthreads();

    float values[64];
    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        values[column] = tile[row][column];
    }

    #pragma unroll
    for (int iteration = 0; iteration < 32; ++iteration) {
        const int k = 2 * iteration;
        const int q = k + 1;
        if (row == k) reciprocal0 = rsqrtf(values[k]);
        __syncthreads();
        if (row >= k) values[k] *= reciprocal0;
        pivot0[row] = values[k];
        if (row == q) {
            values[q] = fmaf(-values[k], values[k], values[q]);
            reciprocal1 = rsqrtf(values[q]);
        }
        __syncthreads();

        if (row >= q) {
            if (row != q) {
                values[q] = fmaf(-values[k], pivot0[q], values[q]);
            }
            values[q] *= reciprocal1;
        }
        pivot1[row] = values[q];
        __syncthreads();

        if (row > q) {
            const float scale0 = values[k];
            const float scale1 = values[q];
            #pragma unroll
            for (int column = 0; column < 64; ++column) {
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
    for (int column = 0; column < 64; ++column) {
        tile[row][column] = column <= row ? values[column] : 0.0f;
    }
    __syncthreads();
    for (int linear = row; linear < N * N; linear += 64) {
        output[base + linear] = tile[linear >> 6][linear & 63];
    }
}

void chol64_launch(torch::Tensor input, torch::Tensor output) {
    const int batch = (int)input.size(0);
    cholesky64_rank2<<<dim3(batch), dim3(64)>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA64 = load_inline(
            name="chol64_exp041_v3",
            cpp_sources="void chol64_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA64_SOURCE,
            functions=["chol64_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA64_ERROR = repr(exc)


def custom_kernel(data):
    global _CUDA64_HITS
    if (
        _CUDA64 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (1024, 64, 64)
        and data.is_contiguous()
    ):
        output = torch.empty_like(data)
        _CUDA64.chol64_launch(data, output)
        _CUDA64_HITS += 1
        return output
    return _ranked.custom_kernel(data)
