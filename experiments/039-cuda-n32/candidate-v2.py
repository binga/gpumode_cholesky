"""Experiment 039 V2 overlay: one row per lane in conflict-free shared memory."""

import torch
import submission as _ranked


_CUDA32_HITS = 0
_CUDA32_ERROR = None
_CUDA32 = None

_CUDA32_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void chol32_rows(const float* __restrict__ src,
                            float* __restrict__ dst) {
    const int lane = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 1024;
    __shared__ float matrix[32][33];
    __shared__ float reciprocal;

    #pragma unroll
    for (int item = 0; item < 32; ++item) {
        const int linear = item * 32 + lane;
        matrix[linear >> 5][linear & 31] = src[base + linear];
    }
    __syncwarp();

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        if (lane == k) {
            const float diagonal = sqrtf(matrix[k][k]);
            matrix[k][k] = diagonal;
            reciprocal = 1.0f / diagonal;
        }
        __syncwarp();
        if (lane > k) matrix[lane][k] *= reciprocal;
        __syncwarp();
        if (lane > k) {
            const float scale = matrix[lane][k];
            #pragma unroll
            for (int column = 0; column < 32; ++column) {
                if (column > k && column <= lane) {
                    matrix[lane][column] -= scale * matrix[column][k];
                }
            }
        }
        __syncwarp();
    }

    #pragma unroll
    for (int item = 0; item < 32; ++item) {
        const int linear = item * 32 + lane;
        const int row = linear >> 5;
        const int column = linear & 31;
        dst[base + linear] = column <= row ? matrix[row][column] : 0.0f;
    }
}

void chol32_launch(torch::Tensor src, torch::Tensor dst) {
    chol32_rows<<<(int)src.size(0), 32>>>(src.data_ptr<float>(), dst.data_ptr<float>());
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA32 = load_inline(
            name="chol32_exp039_v2",
            cpp_sources="void chol32_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA32_SOURCE,
            functions=["chol32_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA32_ERROR = repr(exc)


def custom_kernel(data):
    global _CUDA32_HITS
    if (
        _CUDA32 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (4096, 32, 32)
        and data.is_contiguous()
    ):
        out = torch.empty_like(data)
        _CUDA32.chol32_launch(data, out)
        _CUDA32_HITS += 1
        return out
    return _ranked.custom_kernel(data)
