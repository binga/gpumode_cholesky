"""Experiment 040 V2 overlay: cooperative tile-64 CUDA Cholesky for 1x4096."""

import torch
import submission as _ranked


_COOP4096_HITS = 0
_COOP4096_ERROR = None
_COOP4096 = None

_COOP4096_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int N = 4096;
constexpr int TILE = 64;
constexpr int NT = N / TILE;

__global__ void cooperative_cholesky_4096(float* matrix) {
    cg::grid_group grid = cg::this_grid();
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    __shared__ float tile[64][65];
    __shared__ float diagonal[64][65];
    __shared__ float reciprocal;

    for (int kt = 0; kt < NT; ++kt) {
        const int k0 = kt * TILE;

        // Phase 1: CTA 0 factors the current 64x64 diagonal tile in shared
        // memory. The remaining CTAs wait only at the global phase boundary.
        if (blockIdx.x == 0) {
            for (int linear = thread; linear < 4096; linear += blockDim.x) {
                const int row = linear >> 6;
                const int column = linear & 63;
                tile[row][column] =
                    matrix[(size_t)(k0 + row) * N + k0 + column];
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 64; ++k) {
                if (thread == 0) {
                    reciprocal = rsqrtf(tile[k][k]);
                    tile[k][k] *= reciprocal;
                }
                __syncthreads();
                if (thread < 64 && thread > k) {
                    tile[thread][k] *= reciprocal;
                }
                __syncthreads();
                for (int linear = thread; linear < 4096;
                     linear += blockDim.x) {
                    const int row = linear >> 6;
                    const int column = linear & 63;
                    if (row > k && column > k && column <= row) {
                        tile[row][column] = fmaf(
                            -tile[row][k], tile[column][k],
                            tile[row][column]);
                    }
                }
                __syncthreads();
            }
            for (int linear = thread; linear < 4096; linear += blockDim.x) {
                const int row = linear >> 6;
                const int column = linear & 63;
                if (column <= row) {
                    matrix[(size_t)(k0 + row) * N + k0 + column] =
                        tile[row][column];
                }
            }
        }
        grid.sync();

        // Phase 2: active CTAs solve independent 64x64 panel tiles against
        // the new diagonal factor. One thread owns each complete panel row.
        for (int bi = kt + 1 + blockIdx.x; bi < NT; bi += gridDim.x) {
            const int row0 = bi * TILE;
            for (int linear = thread; linear < 4096; linear += blockDim.x) {
                const int row = linear >> 6;
                const int column = linear & 63;
                tile[row][column] =
                    matrix[(size_t)(row0 + row) * N + k0 + column];
                diagonal[row][column] =
                    matrix[(size_t)(k0 + row) * N + k0 + column];
            }
            __syncthreads();
            if (thread < 64) {
                float values[64];
                #pragma unroll
                for (int column = 0; column < 64; ++column) {
                    values[column] = tile[thread][column];
                }
                #pragma unroll
                for (int column = 0; column < 64; ++column) {
                    float value = values[column];
                    #pragma unroll
                    for (int p = 0; p < 64; ++p) {
                        if (p < column) value = fmaf(
                            -values[p], diagonal[column][p], value);
                    }
                    values[column] = value / diagonal[column][column];
                }
                #pragma unroll
                for (int column = 0; column < 64; ++column) {
                    tile[thread][column] = values[column];
                }
            }
            __syncthreads();
            for (int linear = thread; linear < 4096; linear += blockDim.x) {
                const int row = linear >> 6;
                const int column = linear & 63;
                matrix[(size_t)(row0 + row) * N + k0 + column] =
                    tile[row][column];
            }
        }
        grid.sync();

        // Phase 3: eight resident warps cover the sixteen 16x16 quarters of
        // each 64x64 trailing tile in two passes, using TF32 MMA with FP32
        // accumulation over the new 64-column panel.
        const int remaining_tiles = NT - kt - 1;
        const int pair_count = remaining_tiles * (remaining_tiles + 1) / 2;
        for (int pair = blockIdx.x; pair < pair_count; pair += gridDim.x) {
            const int local_row = (int)((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
            const int local_column = pair - local_row * (local_row + 1) / 2;
            const int bi = kt + 1 + local_row;
            const int bj = kt + 1 + local_column;
            #pragma unroll
            for (int quarter = warp; quarter < 16; quarter += 8) {
                const int warp_row = quarter >> 2;
                const int warp_column = quarter & 3;
                float* cptr = matrix
                    + (size_t)(bi * TILE + warp_row * 16) * N
                    + bj * TILE + warp_column * 16;
                const float* aptr = matrix
                    + (size_t)(bi * TILE + warp_row * 16) * N + k0;
                const float* bptr = matrix
                    + (size_t)(bj * TILE + warp_column * 16) * N + k0;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
                wmma::fill_fragment(accumulator, 0.0f);
                #pragma unroll
                for (int depth = 0; depth < 64; depth += 8) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 8,
                                   wmma::precision::tf32, wmma::row_major> lhs;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                   wmma::precision::tf32, wmma::col_major> rhs;
                    wmma::load_matrix_sync(lhs, aptr + depth, N);
                    wmma::load_matrix_sync(rhs, bptr + depth, N);
                    wmma::mma_sync(accumulator, lhs, rhs, accumulator);
                }
                #pragma unroll
                for (int element = 0; element < accumulator.num_elements;
                     ++element) {
                    accumulator.x[element] = -accumulator.x[element];
                }
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
                wmma::load_matrix_sync(original, cptr, N, wmma::mem_row_major);
                #pragma unroll
                for (int element = 0; element < accumulator.num_elements;
                     ++element) {
                    accumulator.x[element] += original.x[element];
                }
                wmma::store_matrix_sync(
                    cptr, accumulator, N, wmma::mem_row_major);
            }
        }
        grid.sync();
    }

    // Clear the strict upper triangle for the required output representation.
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t linear = (size_t)blockIdx.x * blockDim.x + thread;
         linear < (size_t)N * N; linear += stride) {
        const int row = (int)(linear / N);
        const int column = (int)(linear - (size_t)row * N);
        if (column > row) matrix[linear] = 0.0f;
    }
}

void chol4096_launch(torch::Tensor matrix) {
    float* ptr = matrix.data_ptr<float>();
    void* args[] = {&ptr};
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    cudaError_t status = cudaLaunchCooperativeKernel(
        (void*)cooperative_cholesky_4096,
        dim3(properties.multiProcessorCount), dim3(256), args, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _COOP4096 = load_inline(
            name="chol4096_exp040_v2",
            cpp_sources="void chol4096_launch(torch::Tensor);",
            cuda_sources=_COOP4096_SOURCE,
            functions=["chol4096_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _COOP4096_ERROR = repr(exc)


def custom_kernel(data):
    global _COOP4096_HITS
    if (
        _COOP4096 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (1, 4096, 4096)
        and data.is_contiguous()
    ):
        output = data.clone()
        _COOP4096.chol4096_launch(output)
        _COOP4096_HITS += 1
        return output
    return _ranked.custom_kernel(data)
