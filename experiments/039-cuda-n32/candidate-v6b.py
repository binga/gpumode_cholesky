"""Experiment 039 V6b: V6 with shuffle-rendezvous pivot-buffer ordering."""

import torch
import submission as _ranked


_CUDA32_HITS = 0
_CUDA32_ERROR = None
_CUDA32 = None

_CUDA32_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void chol32_register_shared(const float* __restrict__ src,
                                       float* __restrict__ dst) {
    const int lane = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 1024;
    __shared__ float staging[32][33];
    __shared__ float pivot[32];
    float row_values[32];

    #pragma unroll
    for (int item = 0; item < 32; ++item) {
        const int linear = item * 32 + lane;
        staging[linear >> 5][linear & 31] = src[base + linear];
    }
    __syncwarp();
    #pragma unroll
    for (int column = 0; column < 32; ++column) {
        row_values[column] = staging[lane][column];
    }

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        float inverse = lane == k ? rsqrtf(row_values[k]) : 0.0f;
        // This full-mask rendezvous is also the read-before-reuse ordering
        // point for the preceding iteration's shared pivot buffer.
        inverse = __shfl_sync(0xffffffffu, inverse, k);
        if (lane >= k) row_values[k] *= inverse;
        pivot[lane] = row_values[k];
        __syncwarp();
        if (lane > k) {
            const float scale = row_values[k];
            #pragma unroll
            for (int column = 0; column < 32; ++column) {
                if (column > k && column <= lane) {
                    row_values[column] = fmaf(
                        -scale, pivot[column], row_values[column]);
                }
            }
        }
    }

    #pragma unroll
    for (int column = 0; column < 32; ++column) {
        staging[lane][column] = column <= lane ? row_values[column] : 0.0f;
    }
    __syncwarp();
    #pragma unroll
    for (int item = 0; item < 32; ++item) {
        const int linear = item * 32 + lane;
        dst[base + linear] = staging[linear >> 5][linear & 31];
    }
}

void chol32_launch(torch::Tensor src, torch::Tensor dst) {
    chol32_register_shared<<<(int)src.size(0), 32>>>(
        src.data_ptr<float>(), dst.data_ptr<float>());
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA32 = load_inline(
            name="chol32_exp039_v6b",
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
