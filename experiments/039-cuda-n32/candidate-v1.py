"""Experiment 039 V1 overlay: CUDA register-column n=32 kernel.

This paired-probe candidate imports the frozen uploaded submission for every
unchanged dispatch. The accepted finalist will be integrated into a standalone
submission before repository-wide and Popcorn gates.
"""

import torch
import submission as _ranked


_CUDA32_HITS = 0
_CUDA32_ERROR = None
_CUDA32 = None

_CUDA32_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void chol32_columns(const float* __restrict__ src,
                               float* __restrict__ dst) {
    const int lane = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 1024;
    float x[32];
    __shared__ float pivot[32];

    #pragma unroll
    for (int row = 0; row < 32; ++row) {
        x[row] = src[base + row * 32 + lane];
    }

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        if (lane == k) {
            const float diagonal = sqrtf(x[k]);
            pivot[k] = diagonal;
            #pragma unroll
            for (int row = k + 1; row < 32; ++row) {
                pivot[row] = x[row] / diagonal;
            }
        }
        __syncwarp();

        if (lane == k) {
            x[k] = pivot[k];
            #pragma unroll
            for (int row = k + 1; row < 32; ++row) {
                x[row] = pivot[row];
            }
        } else if (lane > k) {
            const float scale = pivot[lane];
            #pragma unroll
            for (int row = 0; row < 32; ++row) {
                if (row >= lane) x[row] -= pivot[row] * scale;
            }
        }
        __syncwarp();
    }

    #pragma unroll
    for (int row = 0; row < 32; ++row) {
        dst[base + row * 32 + lane] = row >= lane ? x[row] : 0.0f;
    }
}

void chol32_launch(torch::Tensor src, torch::Tensor dst) {
    const int batch = (int)src.size(0);
    chol32_columns<<<batch, 32>>>(src.data_ptr<float>(), dst.data_ptr<float>());
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA32 = load_inline(
            name="chol32_exp039_v1",
            cpp_sources="void chol32_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA32_SOURCE,
            functions=["chol32_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # correctness falls back to the frozen source
        _CUDA32_ERROR = repr(exc)


def custom_kernel(data):
    global _CUDA32_HITS
    if (
        _CUDA32 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.ndim == 3
        and data.shape == (4096, 32, 32)
        and data.is_contiguous()
    ):
        out = torch.empty_like(data)
        _CUDA32.chol32_launch(data, out)
        _CUDA32_HITS += 1
        return out
    return _ranked.custom_kernel(data)
