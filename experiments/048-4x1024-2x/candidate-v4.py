"""Experiment 048 V4: cooperative rank-4 diagonal plus reciprocal panels."""

import torch
import submission as _ranked


_COOP1024_HITS = 0
_COOP1024_FALLBACKS = 0
_COOP1024_ERROR = None
_COOP1024 = None

_COOP1024_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int N = 1024;
constexpr int TILE = 32;
constexpr int NT = N / TILE;
constexpr int MATRICES = 4;
constexpr int GROUP_BLOCKS = 32;

__global__ void cooperative_cholesky_1024(float* matrices) {
    cg::grid_group grid = cg::this_grid();
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int matrix_id = static_cast<int>(blockIdx.x) / GROUP_BLOCKS;
    const int local_block = static_cast<int>(blockIdx.x) % GROUP_BLOCKS;
    float* matrix = matrices + static_cast<size_t>(matrix_id) * N * N;
    __shared__ float tile[32][33];
    __shared__ float diagonal[32][33];
    __shared__ float pivots[4][32];

    for (int kt = 0; kt < NT; ++kt) {
        const int k0 = kt * TILE;

        // Phase 1: one warp factors the current 32x32 diagonal tile.  Four
        // pivots share each rendezvous, transferring the graph-capture-safe
        // rank-4 mechanism established by experiment 044 into this persistent
        // cooperative path.
        if (local_block == 0 && warp == 0) {
            float row_values[32];
            #pragma unroll
            for (int item = 0; item < 32; ++item) {
                const int linear = item * 32 + lane;
                tile[linear >> 5][linear & 31] =
                    matrix[(size_t)(k0 + (linear >> 5)) * N + k0 + (linear & 31)];
            }
            __syncwarp();
            #pragma unroll
            for (int column = 0; column < 32; ++column) {
                row_values[column] = tile[lane][column];
            }
            #pragma unroll
            for (int iteration = 0; iteration < 8; ++iteration) {
                const int p0 = 4 * iteration;
                const int p1 = p0 + 1;
                const int p2 = p0 + 2;
                const int p3 = p0 + 3;

                const float rec0 = rsqrtf(
                    __shfl_sync(0xffffffffu, row_values[p0], p0));
                const float s0 = lane >= p0 ? row_values[p0] * rec0 : 0.0f;
                row_values[p0] = s0;
                pivots[0][lane] = s0;
                const float c01 = __shfl_sync(0xffffffffu, s0, p1);
                const float c02 = __shfl_sync(0xffffffffu, s0, p2);
                const float c03 = __shfl_sync(0xffffffffu, s0, p3);
                if (lane >= p1) row_values[p1] =
                    fmaf(-s0, c01, row_values[p1]);
                if (lane >= p2) row_values[p2] =
                    fmaf(-s0, c02, row_values[p2]);
                if (lane >= p3) row_values[p3] =
                    fmaf(-s0, c03, row_values[p3]);

                const float rec1 = rsqrtf(
                    __shfl_sync(0xffffffffu, row_values[p1], p1));
                const float s1 = lane >= p1 ? row_values[p1] * rec1 : 0.0f;
                row_values[p1] = s1;
                pivots[1][lane] = s1;
                const float c12 = __shfl_sync(0xffffffffu, s1, p2);
                const float c13 = __shfl_sync(0xffffffffu, s1, p3);
                if (lane >= p2) row_values[p2] =
                    fmaf(-s1, c12, row_values[p2]);
                if (lane >= p3) row_values[p3] =
                    fmaf(-s1, c13, row_values[p3]);

                const float rec2 = rsqrtf(
                    __shfl_sync(0xffffffffu, row_values[p2], p2));
                const float s2 = lane >= p2 ? row_values[p2] * rec2 : 0.0f;
                row_values[p2] = s2;
                pivots[2][lane] = s2;
                const float c23 = __shfl_sync(0xffffffffu, s2, p3);
                if (lane >= p3) row_values[p3] =
                    fmaf(-s2, c23, row_values[p3]);

                const float rec3 = rsqrtf(
                    __shfl_sync(0xffffffffu, row_values[p3], p3));
                const float s3 = lane >= p3 ? row_values[p3] * rec3 : 0.0f;
                row_values[p3] = s3;
                pivots[3][lane] = s3;
                __syncwarp();

                #pragma unroll
                for (int column = 0; column < 32; ++column) {
                    if (column > p3 && column <= lane) {
                        float value = row_values[column];
                        value = fmaf(-s0, pivots[0][column], value);
                        value = fmaf(-s1, pivots[1][column], value);
                        value = fmaf(-s2, pivots[2][column], value);
                        row_values[column] =
                            fmaf(-s3, pivots[3][column], value);
                    }
                }
                __syncwarp();
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
                matrix[(size_t)(k0 + (linear >> 5)) * N + k0 + (linear & 31)] =
                    tile[linear >> 5][linear & 31];
            }
        }
        grid.sync();

        // Phase 2: active CTAs solve independent 32x32 panel tiles against
        // the new diagonal factor. Each lane owns one complete panel row.
        for (int bi = kt + 1 + local_block; bi < NT; bi += GROUP_BLOCKS) {
            const int row0 = bi * TILE;
            for (int linear = thread; linear < 1024; linear += blockDim.x) {
                const int row = linear >> 5;
                const int column = linear & 31;
                tile[row][column] =
                    matrix[(size_t)(row0 + row) * N + k0 + column];
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
                    values[column] = value * __frcp_rn(diagonal[column][column]);
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

        // Phase 3: four warps distribute the lower-triangular trailing tiles.
        // Every warp owns one 16x16 quarter and uses TF32 MMA with FP32
        // accumulation over the new 32-column panel.
        const int remaining_tiles = NT - kt - 1;
        const int pair_count = remaining_tiles * (remaining_tiles + 1) / 2;
        for (int pair = local_block; pair < pair_count; pair += GROUP_BLOCKS) {
            const int local_row = (int)((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
            const int local_column = pair - local_row * (local_row + 1) / 2;
            const int bi = kt + 1 + local_row;
            const int bj = kt + 1 + local_column;
            const int warp_row = warp >> 1;
            const int warp_column = warp & 1;
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
            for (int depth = 0; depth < 32; depth += 8) {
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
            // Reload C and add it after negating the product. This keeps the
            // product on tensor cores without allocating a second tile.
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
            wmma::load_matrix_sync(original, cptr, N, wmma::mem_row_major);
            #pragma unroll
            for (int element = 0; element < accumulator.num_elements; ++element) {
                accumulator.x[element] += original.x[element];
            }
            wmma::store_matrix_sync(cptr, accumulator, N, wmma::mem_row_major);
        }
        grid.sync();
    }

    // Clear the strict upper triangle for the required output representation.
    const size_t stride = (size_t)GROUP_BLOCKS * blockDim.x;
    for (size_t linear = (size_t)local_block * blockDim.x + thread;
         linear < (size_t)N * N; linear += stride) {
        const int row = (int)(linear / N);
        const int column = (int)(linear - (size_t)row * N);
        if (column > row) matrix[linear] = 0.0f;
    }
}

void chol1024_launch(torch::Tensor matrix) {
    float* ptr = matrix.data_ptr<float>();
    void* args[] = {&ptr};
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    TORCH_CHECK(matrix.dim() == 3 && matrix.size(0) == MATRICES &&
                matrix.size(1) == N && matrix.size(2) == N,
                "cooperative_cholesky_1024 requires 4x1024x1024");
    cudaError_t status = cudaLaunchCooperativeKernel(
        (void*)cooperative_cholesky_1024,
        dim3(MATRICES * GROUP_BLOCKS), dim3(128), args, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _COOP1024 = load_inline(
            name="chol1024_exp048_v4_rank4_recip",
            cpp_sources="void chol1024_launch(torch::Tensor);",
            cuda_sources=_COOP1024_SOURCE,
            functions=["chol1024_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _COOP1024_ERROR = repr(exc)


def custom_kernel(data):
    global _COOP1024_HITS, _COOP1024_FALLBACKS
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (4, 1024, 1024)
        and data.is_contiguous()
    )
    if target and _COOP1024 is not None:
        output = data.clone()
        _COOP1024.chol1024_launch(output)
        _COOP1024_HITS += 1
        return output
    if target:
        _COOP1024_FALLBACKS += 1
    return _ranked.custom_kernel(data)
