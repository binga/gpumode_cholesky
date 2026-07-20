"""Experiment 042 V5: compile-compact blocked-16 CUDA Cholesky."""

import torch
import submission as _ranked


_CUDA128_HITS = 0
_CUDA128_ERROR = None
_CUDA128 = None

_CUDA128_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int N = 128;
constexpr int BK = 16;
constexpr int TILE_STRIDE = 129;
constexpr int THREADS = 256;
constexpr int SHARED_BYTES = N * TILE_STRIDE * sizeof(float);

__global__ void cholesky128_block16(const float* input, float* output) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t base = (size_t)blockIdx.x * N * N;
    extern __shared__ float tile[];
    __shared__ float reciprocal;
    __shared__ float inverse_diag[BK];

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 7;
        const int column = linear & 127;
        tile[row * TILE_STRIDE + column] = input[base + linear];
    }
    __syncthreads();

    #pragma unroll 1
    for (int block = 0; block < N; block += BK) {
        const int block_end = block + BK;

        #pragma unroll
        for (int local = 0; local < BK; ++local) {
            const int pivot = block + local;
            if (tid == 0) {
                reciprocal = rsqrtf(tile[pivot * TILE_STRIDE + pivot]);
                inverse_diag[local] = reciprocal;
                tile[pivot * TILE_STRIDE + pivot] *= reciprocal;
            }
            __syncthreads();

            const int panel_row = pivot + 1 + tid;
            if (panel_row < block_end) {
                tile[panel_row * TILE_STRIDE + pivot] *= reciprocal;
            }
            __syncthreads();

            for (int linear = tid; linear < BK * BK; linear += THREADS) {
                const int row = block + (linear >> 4);
                const int column = block + (linear & 15);
                if (row > pivot && column > pivot && column <= row) {
                    const int offset = row * TILE_STRIDE + column;
                    tile[offset] = fmaf(
                        -tile[row * TILE_STRIDE + pivot],
                        tile[column * TILE_STRIDE + pivot],
                        tile[offset]);
                }
            }
            __syncthreads();
        }

        const int row = block_end + tid;
        if (row < N) {
            #pragma unroll
            for (int local = 0; local < BK; ++local) {
                const int column = block + local;
                float value = tile[row * TILE_STRIDE + column];
                #pragma unroll
                for (int prior = 0; prior < local; ++prior) {
                    value = fmaf(
                        -tile[row * TILE_STRIDE + block + prior],
                        tile[column * TILE_STRIDE + block + prior],
                        value);
                }
                tile[row * TILE_STRIDE + column] =
                    value * inverse_diag[local];
            }
        }
        __syncthreads();

        for (int trailing_row = block_end + warp;
             trailing_row < N;
             trailing_row += 8) {
            for (int column = block_end + lane;
                 column <= trailing_row;
                 column += 32) {
                float update = 0.0f;
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    update = fmaf(
                        tile[trailing_row * TILE_STRIDE + block + k],
                        tile[column * TILE_STRIDE + block + k],
                        update);
                }
                tile[trailing_row * TILE_STRIDE + column] -= update;
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
            cholesky128_block16,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHARED_BYTES);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    const int batch = (int)input.size(0);
    cholesky128_block16<<<dim3(batch), dim3(THREADS), SHARED_BYTES>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA128 = load_inline(
            name="chol128_exp042_v5",
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
