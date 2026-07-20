"""Experiment 041 V1: rank-2 register-row CUDA Cholesky for 1024x64."""

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
    const int lane = threadIdx.x;
    const int row0 = lane;
    const int row1 = lane + 32;
    const size_t base = (size_t)blockIdx.x * N * N;
    __shared__ float tile[64][65];
    __shared__ float pivot0[64];
    __shared__ float pivot1[64];

    for (int linear = lane; linear < N * N; linear += 32) {
        tile[linear >> 6][linear & 63] = input[base + linear];
    }
    __syncwarp();

    float values0[64];
    float values1[64];
    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        values0[column] = tile[row0][column];
        values1[column] = tile[row1][column];
    }

    #pragma unroll
    for (int iteration = 0; iteration < 32; ++iteration) {
        const int k = 2 * iteration;
        const int q = k + 1;
        const int owner0 = k & 31;
        float inverse0 = 0.0f;
        if (lane == owner0) {
            inverse0 = rsqrtf(k < 32 ? values0[k] : values1[k]);
        }
        inverse0 = __shfl_sync(0xffffffffu, inverse0, owner0);
        if (row0 >= k) values0[k] *= inverse0;
        if (row1 >= k) values1[k] *= inverse0;
        pivot0[row0] = values0[k];
        pivot0[row1] = values1[k];
        __syncwarp();

        if (row0 >= q) {
            values0[q] = fmaf(-values0[k], pivot0[q], values0[q]);
        }
        if (row1 >= q) {
            values1[q] = fmaf(-values1[k], pivot0[q], values1[q]);
        }
        const int owner1 = q & 31;
        float inverse1 = 0.0f;
        if (lane == owner1) {
            inverse1 = rsqrtf(q < 32 ? values0[q] : values1[q]);
        }
        inverse1 = __shfl_sync(0xffffffffu, inverse1, owner1);
        if (row0 >= q) values0[q] *= inverse1;
        if (row1 >= q) values1[q] *= inverse1;
        pivot1[row0] = values0[q];
        pivot1[row1] = values1[q];
        __syncwarp();

        if (row0 > q) {
            const float scale0 = values0[k];
            const float scale1 = values0[q];
            #pragma unroll
            for (int column = 0; column < 64; ++column) {
                if (column > q && column <= row0) {
                    float value = fmaf(
                        -scale0, pivot0[column], values0[column]);
                    values0[column] = fmaf(
                        -scale1, pivot1[column], value);
                }
            }
        }
        if (row1 > q) {
            const float scale0 = values1[k];
            const float scale1 = values1[q];
            #pragma unroll
            for (int column = 0; column < 64; ++column) {
                if (column > q && column <= row1) {
                    float value = fmaf(
                        -scale0, pivot0[column], values1[column]);
                    values1[column] = fmaf(
                        -scale1, pivot1[column], value);
                }
            }
        }
    }

    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        tile[row0][column] = column <= row0 ? values0[column] : 0.0f;
        tile[row1][column] = column <= row1 ? values1[column] : 0.0f;
    }
    __syncwarp();
    for (int linear = lane; linear < N * N; linear += 32) {
        output[base + linear] = tile[linear >> 6][linear & 63];
    }
}

void chol64_launch(torch::Tensor input, torch::Tensor output) {
    const int batch = (int)input.size(0);
    cholesky64_rank2<<<dim3(batch), dim3(32)>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA64 = load_inline(
            name="chol64_exp041_v1",
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
