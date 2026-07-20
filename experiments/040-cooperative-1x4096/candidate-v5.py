"""Experiment 040 V5: cooperative left-looking tile-32 CUDA Cholesky."""

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
constexpr int TILE = 32;
constexpr int NT = N / TILE;

__global__ void cooperative_cholesky_4096(float* matrix) {
    cg::grid_group grid = cg::this_grid();
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int warp_row = warp >> 1;
    const int warp_column = warp & 1;
    __shared__ float tile[32][36];
    __shared__ float diagonal[32][36];
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];

    for (int kt = 0; kt < NT; ++kt) {
        const int k0 = kt * TILE;

        // Phase 1: form A[k,k] - L[k,:k]L[k,:k]^T exactly once, then
        // factor the resulting 32x32 tile with the rank-2 warp microkernel.
        if (blockIdx.x == 0) {
            float* cptr = matrix
                + (size_t)(k0 + warp_row * 16) * N
                + k0 + warp_column * 16;
            const float* aptr = matrix
                + (size_t)(k0 + warp_row * 16) * N;
            const float* bptr = matrix
                + (size_t)(k0 + warp_column * 16) * N;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
            for (int depth = 0; depth < k0; depth += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> lhs;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::col_major> rhs;
                wmma::load_matrix_sync(lhs, aptr + depth, N);
                wmma::load_matrix_sync(rhs, bptr + depth, N);
                wmma::mma_sync(accumulator, lhs, rhs, accumulator);
            }
            #pragma unroll
            for (int element = 0; element < accumulator.num_elements; ++element) {
                accumulator.x[element] = -accumulator.x[element];
            }
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
            wmma::load_matrix_sync(original, cptr, N, wmma::mem_row_major);
            #pragma unroll
            for (int element = 0; element < accumulator.num_elements; ++element) {
                accumulator.x[element] += original.x[element];
            }
            wmma::store_matrix_sync(
                &tile[warp_row * 16][warp_column * 16],
                accumulator, 36, wmma::mem_row_major);
            __syncthreads();

            if (warp == 0) {
                float row_values[32];
                #pragma unroll
                for (int column = 0; column < 32; ++column) {
                    row_values[column] = tile[lane][column];
                }
                #pragma unroll
                for (int iteration = 0; iteration < 16; ++iteration) {
                    const int k = 2 * iteration;
                    const int q = k + 1;
                    float inverse0 = lane == k ? rsqrtf(row_values[k]) : 0.0f;
                    inverse0 = __shfl_sync(0xffffffffu, inverse0, k);
                    if (lane >= k) row_values[k] *= inverse0;
                    pivot0[lane] = row_values[k];
                    __syncwarp();
                    if (lane >= q) {
                        row_values[q] = fmaf(
                            -row_values[k], pivot0[q], row_values[q]);
                    }
                    float inverse1 = lane == q ? rsqrtf(row_values[q]) : 0.0f;
                    inverse1 = __shfl_sync(0xffffffffu, inverse1, q);
                    if (lane >= q) row_values[q] *= inverse1;
                    pivot1[lane] = row_values[q];
                    __syncwarp();
                    if (lane > q) {
                        const float scale0 = row_values[k];
                        const float scale1 = row_values[q];
                        #pragma unroll
                        for (int column = 0; column < 32; ++column) {
                            if (column > q && column <= lane) {
                                float value = fmaf(
                                    -scale0, pivot0[column], row_values[column]);
                                row_values[column] = fmaf(
                                    -scale1, pivot1[column], value);
                            }
                        }
                    }
                }
                #pragma unroll
                for (int column = 0; column < 32; ++column) {
                    tile[lane][column] =
                        column <= lane ? row_values[column] : 0.0f;
                }
                __syncwarp();
                #pragma unroll
                for (int item = 0; item < 32; ++item) {
                    const int linear = item * 32 + lane;
                    matrix[(size_t)(k0 + (linear >> 5)) * N
                           + k0 + (linear & 31)] =
                        tile[linear >> 5][linear & 31];
                }
            }
        }
        grid.sync();

        // Phase 2: every active CTA forms one left-looking panel tile, then
        // performs the FP32 triangular substitution against the new factor.
        for (int bi = kt + 1 + blockIdx.x; bi < NT; bi += gridDim.x) {
            const int row0 = bi * TILE;
            float* cptr = matrix
                + (size_t)(row0 + warp_row * 16) * N
                + k0 + warp_column * 16;
            const float* aptr = matrix
                + (size_t)(row0 + warp_row * 16) * N;
            const float* bptr = matrix
                + (size_t)(k0 + warp_column * 16) * N;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
            for (int depth = 0; depth < k0; depth += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> lhs;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::col_major> rhs;
                wmma::load_matrix_sync(lhs, aptr + depth, N);
                wmma::load_matrix_sync(rhs, bptr + depth, N);
                wmma::mma_sync(accumulator, lhs, rhs, accumulator);
            }
            #pragma unroll
            for (int element = 0; element < accumulator.num_elements; ++element) {
                accumulator.x[element] = -accumulator.x[element];
            }
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
            wmma::load_matrix_sync(original, cptr, N, wmma::mem_row_major);
            #pragma unroll
            for (int element = 0; element < accumulator.num_elements; ++element) {
                accumulator.x[element] += original.x[element];
            }
            wmma::store_matrix_sync(
                &tile[warp_row * 16][warp_column * 16],
                accumulator, 36, wmma::mem_row_major);
            __syncthreads();
            for (int linear = thread; linear < 1024; linear += blockDim.x) {
                const int row = linear >> 5;
                const int column = linear & 31;
                diagonal[row][column] =
                    matrix[(size_t)(k0 + row) * N + k0 + column];
            }
            __syncthreads();
            if (warp == 0) {
                float values[32];
                #pragma unroll
                for (int column = 0; column < 32; ++column) {
                    values[column] = tile[lane][column];
                }
                #pragma unroll
                for (int column = 0; column < 32; ++column) {
                    float value = values[column];
                    #pragma unroll
                    for (int p = 0; p < 32; ++p) {
                        if (p < column) value = fmaf(
                            -values[p], diagonal[column][p], value);
                    }
                    values[column] = value / diagonal[column][column];
                }
                #pragma unroll
                for (int column = 0; column < 32; ++column) {
                    tile[lane][column] = values[column];
                }
            }
            __syncthreads();
            for (int linear = thread; linear < 1024; linear += blockDim.x) {
                const int row = linear >> 5;
                const int column = linear & 31;
                matrix[(size_t)(row0 + row) * N + k0 + column] =
                    tile[row][column];
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
        dim3(properties.multiProcessorCount), dim3(128), args, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _COOP4096 = load_inline(
            name="chol4096_exp040_v5",
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
