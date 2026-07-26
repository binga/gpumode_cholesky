
// ---------------------------------------------------------------------------
// Experiment 051: cooperative tile-32 blocked Cholesky for the tiny-batch mid
// shapes, repaired and completed from experiment 048 V2.
//
// 048 V2 measured 1.167x on 4x1024 but was rejected for a low-rank NaN and
// spent 41.5% of its runtime in an unvectorized scalar panel solve. Three
// changes here:
//
//   1. Every CTA factors the current 32x32 diagonal tile redundantly instead of
//      one warp publishing it through global memory. The factor and its
//      reciprocal diagonal stay in shared memory, which removes one of the
//      three grid-wide barriers per block step and the global round trip.
//   2. The panel solve is right-looking over 32 registers per lane: column j is
//      finalized with one multiply and then broadcast into the remaining
//      columns. The dependent chain per 32x32 tile falls from 528 serial FMAs
//      (048 V2's left-looking `if (p < column)` loop) to 32, and all four warps
//      solve independent tiles instead of only warp 0.
//   3. The co-resident CTA count is chosen from the measured occupancy rather
//      than fixed at 32 per matrix, so the trailing update gets the whole GPU.
//
// Non-finite output on ill-conditioned families is caught by the caller's
// finiteness gate, which falls through to the ranked dispatch.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace coop051 {

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int COOP_THREADS = 128;
constexpr int COOP_WARPS = COOP_THREADS / 32;
// A cooperative launch can only use as many CTAs as are co-resident, so the
// register budget directly caps the trailing update's parallelism. Pinning four
// blocks per SM gives 128 registers per thread -- enough for the 32-register
// panel row without spilling -- and ~4x experiment 048 V2's fixed 32 CTAs per
// matrix.
constexpr int COOP_MIN_BLOCKS = 4;

__global__ __launch_bounds__(COOP_THREADS, COOP_MIN_BLOCKS) void coop_cholesky_tile32(
    float* __restrict__ matrices, int n, int nt, int ctas_per_matrix) {
    cg::grid_group grid = cg::this_grid();
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int matrix_id = static_cast<int>(blockIdx.x) / ctas_per_matrix;
    const int local_block = static_cast<int>(blockIdx.x) % ctas_per_matrix;
    float* matrix = matrices + static_cast<size_t>(matrix_id) * n * n;

    __shared__ float dtile[32][33];
    __shared__ float rdiag[32];
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];
    __shared__ float ptile[COOP_WARPS][32][33];

    for (int kt = 0; kt < nt; ++kt) {
        const int k0 = kt * 32;

        // -- Phase 1: redundant 32x32 diagonal factorization (no barrier).
        for (int linear = thread; linear < 1024; linear += COOP_THREADS) {
            const int row = linear >> 5;
            const int column = linear & 31;
            dtile[row][column] =
                matrix[static_cast<size_t>(k0 + row) * n + k0 + column];
        }
        __syncthreads();
        if (warp == 0) {
            float row_values[32];
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                row_values[column] = dtile[lane][column];
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
                    row_values[q] =
                        fmaf(-row_values[k], pivot0[q], row_values[q]);
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
                            float value =
                                fmaf(-scale0, pivot0[column], row_values[column]);
                            row_values[column] =
                                fmaf(-scale1, pivot1[column], value);
                        }
                    }
                }
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                dtile[lane][column] = column <= lane ? row_values[column] : 0.0f;
            }
            rdiag[lane] = 1.0f / row_values[lane];
        }
        __syncthreads();

        // -- Phase 2: right-looking triangular panel solve, one tile per warp.
        // X L^T = P, so column j is final after one multiply by 1/L[j][j] and
        // is then subtracted out of every remaining column. The 32 columns form
        // the only dependent chain; the 31-wide update inside each step is
        // independent ILP.
        const int panel_tiles = nt - kt - 1;
        for (int t = local_block * COOP_WARPS + warp; t < panel_tiles;
             t += ctas_per_matrix * COOP_WARPS) {
            const int row0 = (kt + 1 + t) * 32;
            for (int i = 0; i < 32; ++i) {
                ptile[warp][i][lane] =
                    matrix[static_cast<size_t>(row0 + i) * n + k0 + lane];
            }
            __syncwarp();
            float v[32];
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                v[c] = ptile[warp][lane][c];
            }
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float xj = v[j] * rdiag[j];
                v[j] = xj;
#pragma unroll
                for (int c = j + 1; c < 32; ++c) {
                    v[c] = fmaf(-xj, dtile[c][j], v[c]);
                }
            }
#pragma unroll
            for (int c = 0; c < 32; ++c) {
                ptile[warp][lane][c] = v[c];
            }
            __syncwarp();
            for (int i = 0; i < 32; ++i) {
                matrix[static_cast<size_t>(row0 + i) * n + k0 + lane] =
                    ptile[warp][i][lane];
            }
            __syncwarp();
        }
        grid.sync();

        // One CTA per matrix publishes the diagonal factor. It runs after the
        // barrier so it cannot race the phase-1 reads of the other CTAs.
        if (local_block == 0) {
            for (int linear = thread; linear < 1024; linear += COOP_THREADS) {
                const int row = linear >> 5;
                const int column = linear & 31;
                matrix[static_cast<size_t>(k0 + row) * n + k0 + column] =
                    dtile[row][column];
            }
        }

        // -- Phase 3: TF32 tensor-core trailing update over the lower tiles.
        const int rem = nt - kt - 1;
        const int pair_count = rem * (rem + 1) / 2;
        for (int pair = local_block; pair < pair_count; pair += ctas_per_matrix) {
            int local_row =
                static_cast<int>((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
            while ((local_row + 1) * (local_row + 2) / 2 <= pair) ++local_row;
            while (local_row > 0 && local_row * (local_row + 1) / 2 > pair) {
                --local_row;
            }
            const int local_column = pair - local_row * (local_row + 1) / 2;
            const int bi = kt + 1 + local_row;
            const int bj = kt + 1 + local_column;
            const int warp_row = warp >> 1;
            const int warp_column = warp & 1;
            float* cptr = matrix
                + static_cast<size_t>(bi * 32 + warp_row * 16) * n
                + bj * 32 + warp_column * 16;
            const float* aptr = matrix
                + static_cast<size_t>(bi * 32 + warp_row * 16) * n + k0;
            const float* bptr = matrix
                + static_cast<size_t>(bj * 32 + warp_column * 16) * n + k0;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
            wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
            for (int depth = 0; depth < 32; depth += 8) {
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> lhs;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::col_major> rhs;
                wmma::load_matrix_sync(lhs, aptr + depth, n);
                wmma::load_matrix_sync(rhs, bptr + depth, n);
                wmma::mma_sync(accumulator, lhs, rhs, accumulator);
            }
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
            wmma::load_matrix_sync(original, cptr, n, wmma::mem_row_major);
#pragma unroll
            for (int element = 0; element < accumulator.num_elements; ++element) {
                accumulator.x[element] =
                    original.x[element] - accumulator.x[element];
            }
            wmma::store_matrix_sync(cptr, accumulator, n, wmma::mem_row_major);
        }
        grid.sync();
    }

    // Required output representation: strictly-upper entries are zero.
    for (int row = local_block; row < n; row += ctas_per_matrix) {
        for (int column = row + 1 + thread; column < n; column += COOP_THREADS) {
            matrix[static_cast<size_t>(row) * n + column] = 0.0f;
        }
    }
}

}  // namespace coop051

int64_t coop051_ctas(int64_t batch, int64_t nt) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    int per_sm = 0;
    cudaError_t occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, (void*)coop051::coop_cholesky_tile32, coop051::COOP_THREADS, 0);
    if (occupancy != cudaSuccess || per_sm <= 0) return 0;
    int64_t ctas = (int64_t)per_sm * properties.multiProcessorCount / batch;
    const int64_t useful = nt * (nt + 1) / 2;
    if (ctas > useful) ctas = useful;
    if (ctas > 256) ctas = 256;
    return ctas;
}

void coop051_launch(torch::Tensor matrix) {
    TORCH_CHECK(matrix.is_cuda() && matrix.dim() == 3, "coop051 needs a 3-D CUDA tensor");
    TORCH_CHECK(matrix.scalar_type() == torch::kFloat32, "coop051 needs float32");
    TORCH_CHECK(matrix.is_contiguous(), "coop051 needs a contiguous tensor");
    int batch = (int)matrix.size(0);
    int n = (int)matrix.size(1);
    TORCH_CHECK(matrix.size(2) == n && n % 32 == 0, "coop051 needs square n%32==0");
    int nt = n / 32;
    int ctas = (int)coop051_ctas(batch, nt);
    TORCH_CHECK(ctas >= 1, "coop051 has no co-resident cooperative grid");
    float* ptr = matrix.data_ptr<float>();
    void* args[] = {&ptr, &n, &nt, &ctas};
    cudaError_t status = cudaLaunchCooperativeKernel(
        (void*)coop051::coop_cholesky_tile32, dim3(batch * ctas),
        dim3(coop051::COOP_THREADS), args, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
