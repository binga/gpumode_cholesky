
// ---------------------------------------------------------------------------
// Experiment 051 V2: two-level cooperative Cholesky (NB=128 over tile 32).
//
// V1 fixed the panel solve and stayed correct on every shape, but its paired
// grid showed the rank-32 trailing update running at only 2.6-8.7 TFLOP/s, so
// it lost badly wherever the trailing volume dominates (1x4096 0.352x). The
// cause is structural, not a tuning miss: a rank-32 update of 32x32 tiles reads
// three tiles and writes one for 65 kFLOP -- about 4 flops per byte, which caps
// it near 28 TFLOP/s even at full HBM bandwidth, and loading the MMA fragments
// straight from global memory gives up most of that.
//
// V2 therefore blocks the factorization at NB=128, exactly like the ranked
// split32 pipeline:
//
//   * The 128 columns of a panel are factored by four dependent tile-32 steps
//     whose Schur updates stay INSIDE the panel (at most 96 columns wide).
//   * The trailing submatrix then takes a single rank-128 update per panel,
//     staged through shared memory as 64x64 output blocks. That is 16 flops per
//     byte -- a ~75 TFLOP/s ceiling instead of ~28.
//
// The dependent pivot chain is unchanged (it is the irreducible part), but the
// bandwidth-bound work now runs at a rate the shape budgets can absorb.
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
constexpr int COOP_MIN_BLOCKS = 4;
constexpr int COOP_NB = 128;          // outer panel width
constexpr int COOP_MAX_CTAS = 256;

// Shared scratch is reused between the panel phase and the trailing phase.
//   panel:    ptile[COOP_WARPS][32][33]      = 4224 floats
//   trailing: As[64][36] followed by Bs[64][36] = 4608 floats
constexpr int COOP_POOL = 4608;
constexpr int COOP_LDS = 36;

// One 32x32 <- 32x32 * 32x32^T rank-32 Schur update by a single warp. Used only
// for the narrow in-panel updates, whose total volume is O(n^2 * NB).
__device__ __forceinline__ void schur_tile32(float* __restrict__ matrix, int ld,
                                             int bi, int bj, int k0) {
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
        const int qr = quarter >> 1;
        const int qc = quarter & 1;
        float* cptr =
            matrix + static_cast<size_t>(bi * 32 + qr * 16) * ld + bj * 32 + qc * 16;
        const float* aptr =
            matrix + static_cast<size_t>(bi * 32 + qr * 16) * ld + k0;
        const float* bptr =
            matrix + static_cast<size_t>(bj * 32 + qc * 16) * ld + k0;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int depth = 0; depth < 32; depth += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8,
                           wmma::precision::tf32, wmma::row_major> lhs;
            wmma::fragment<wmma::matrix_b, 16, 16, 8,
                           wmma::precision::tf32, wmma::col_major> rhs;
            wmma::load_matrix_sync(lhs, aptr + depth, ld);
            wmma::load_matrix_sync(rhs, bptr + depth, ld);
            wmma::mma_sync(acc, lhs, rhs, acc);
        }
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
        wmma::load_matrix_sync(original, cptr, ld, wmma::mem_row_major);
#pragma unroll
        for (int e = 0; e < acc.num_elements; ++e) {
            acc.x[e] = original.x[e] - acc.x[e];
        }
        wmma::store_matrix_sync(cptr, acc, ld, wmma::mem_row_major);
    }
}

// Warp-synchronous rank-2 Cholesky of the 32x32 tile in `tile`; leaves L there
// (strict upper zeroed) and 1/L[j][j] in `rdiag`.
__device__ __forceinline__ void factor_tile32(float (*tile)[33], float* rdiag,
                                              float* pivot0, float* pivot1,
                                              int lane) {
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
            row_values[q] = fmaf(-row_values[k], pivot0[q], row_values[q]);
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
                    float value = fmaf(-scale0, pivot0[column], row_values[column]);
                    row_values[column] = fmaf(-scale1, pivot1[column], value);
                }
            }
        }
    }
#pragma unroll
    for (int column = 0; column < 32; ++column) {
        tile[lane][column] = column <= lane ? row_values[column] : 0.0f;
    }
    rdiag[lane] = 1.0f / row_values[lane];
}

__global__ __launch_bounds__(COOP_THREADS, COOP_MIN_BLOCKS) void coop_cholesky_v2(
    float* __restrict__ matrices, int n, int nt, int ctas_per_matrix) {
    cg::grid_group grid = cg::this_grid();
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int matrix_id = static_cast<int>(blockIdx.x) / ctas_per_matrix;
    const int local_block = static_cast<int>(blockIdx.x) % ctas_per_matrix;
    float* matrix = matrices + static_cast<size_t>(matrix_id) * n * n;
    const int slot = local_block * COOP_WARPS + warp;
    const int slots = ctas_per_matrix * COOP_WARPS;

    __shared__ float dtile[32][33];
    __shared__ float rdiag[32];
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];
    __shared__ __align__(16) float pool[COOP_POOL];
    float* const ptile = pool + warp * 32 * 33;
    float* const smem_a = pool;
    float* const smem_b = pool + 64 * COOP_LDS;

    for (int jb = 0; jb < n; jb += COOP_NB) {
        const int panel_end_tile = (jb + COOP_NB) / 32;

        // ================= panel columns [jb, jb+128) =====================
        for (int kt = jb / 32; kt < panel_end_tile; ++kt) {
            const int k0 = kt * 32;

            // -- diagonal tile, factored redundantly by every CTA.
            for (int linear = thread; linear < 1024; linear += COOP_THREADS) {
                const int row = linear >> 5;
                const int column = linear & 31;
                dtile[row][column] =
                    matrix[static_cast<size_t>(k0 + row) * n + k0 + column];
            }
            __syncthreads();
            if (warp == 0) factor_tile32(dtile, rdiag, pivot0, pivot1, lane);
            __syncthreads();

            // -- right-looking triangular panel solve, one tile per warp.
            const int panel_tiles = nt - kt - 1;
            for (int t = slot; t < panel_tiles; t += slots) {
                const int row0 = (kt + 1 + t) * 32;
                for (int i = 0; i < 32; ++i) {
                    ptile[i * 33 + lane] =
                        matrix[static_cast<size_t>(row0 + i) * n + k0 + lane];
                }
                __syncwarp();
                float v[32];
#pragma unroll
                for (int c = 0; c < 32; ++c) v[c] = ptile[lane * 33 + c];
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
                for (int c = 0; c < 32; ++c) ptile[lane * 33 + c] = v[c];
                __syncwarp();
                for (int i = 0; i < 32; ++i) {
                    matrix[static_cast<size_t>(row0 + i) * n + k0 + lane] =
                        ptile[i * 33 + lane];
                }
                __syncwarp();
            }
            grid.sync();

            // One CTA per matrix publishes the diagonal factor, after the
            // barrier so it cannot race the other CTAs' phase-1 reads.
            if (local_block == 0) {
                for (int linear = thread; linear < 1024; linear += COOP_THREADS) {
                    const int row = linear >> 5;
                    const int column = linear & 31;
                    matrix[static_cast<size_t>(k0 + row) * n + k0 + column] =
                        dtile[row][column];
                }
            }

            // -- narrow Schur update: rows below the diagonal tile, columns
            // only as far as the end of this panel. At most three tile columns.
            const int ct_lo = kt + 1;
            const int ncols = panel_end_tile - ct_lo;
            const int nrows = nt - ct_lo;
            for (int ci = 0; ci < ncols; ++ci) {
                for (int ri = ci + slot; ri < nrows; ri += slots) {
                    schur_tile32(matrix, n, ct_lo + ri, ct_lo + ci, k0);
                }
            }
            grid.sync();
        }

        // ================= rank-128 trailing update =======================
        // C[R,C] -= L[R, jb:jb+128] @ L[C, jb:jb+128]^T over 64x64 blocks.
        const int trail = n - (jb + COOP_NB);
        if (trail > 0) {
            const int mb = trail / 64;
            const int base = (jb + COOP_NB) / 64;
            const int npairs = mb * (mb + 1) / 2;
            const int qr = (warp >> 1) * 32;
            const int qc = (warp & 1) * 32;
            for (int pair = local_block; pair < npairs; pair += ctas_per_matrix) {
                int br = static_cast<int>((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
                while ((br + 1) * (br + 2) / 2 <= pair) ++br;
                while (br > 0 && br * (br + 1) / 2 > pair) --br;
                const int bc = pair - br * (br + 1) / 2;
                const int row0 = (base + br) * 64;
                const int col0 = (base + bc) * 64;

                wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
#pragma unroll
                for (int a = 0; a < 2; ++a) {
#pragma unroll
                    for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[a][b], 0.0f);
                }

                for (int kc = 0; kc < COOP_NB; kc += 32) {
                    __syncthreads();
                    for (int idx = thread; idx < 64 * 32; idx += COOP_THREADS) {
                        const int i = idx >> 5;
                        const int j = idx & 31;
                        smem_a[i * COOP_LDS + j] =
                            matrix[static_cast<size_t>(row0 + i) * n + jb + kc + j];
                        smem_b[i * COOP_LDS + j] =
                            matrix[static_cast<size_t>(col0 + i) * n + jb + kc + j];
                    }
                    __syncthreads();
#pragma unroll
                    for (int d = 0; d < 32; d += 8) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 8,
                                       wmma::precision::tf32, wmma::row_major> fa[2];
                        wmma::fragment<wmma::matrix_b, 16, 16, 8,
                                       wmma::precision::tf32, wmma::col_major> fb[2];
#pragma unroll
                        for (int a = 0; a < 2; ++a) {
                            wmma::load_matrix_sync(
                                fa[a], smem_a + (qr + a * 16) * COOP_LDS + d,
                                COOP_LDS);
                        }
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            wmma::load_matrix_sync(
                                fb[b], smem_b + (qc + b * 16) * COOP_LDS + d,
                                COOP_LDS);
                        }
#pragma unroll
                        for (int a = 0; a < 2; ++a) {
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                wmma::mma_sync(acc[a][b], fa[a], fb[b], acc[a][b]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int a = 0; a < 2; ++a) {
#pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        float* cptr = matrix
                            + static_cast<size_t>(row0 + qr + a * 16) * n
                            + col0 + qc + b * 16;
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
                        wmma::load_matrix_sync(original, cptr, n, wmma::mem_row_major);
#pragma unroll
                        for (int e = 0; e < original.num_elements; ++e) {
                            original.x[e] -= acc[a][b].x[e];
                        }
                        wmma::store_matrix_sync(cptr, original, n, wmma::mem_row_major);
                    }
                }
            }
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
        &per_sm, (void*)coop051::coop_cholesky_v2, coop051::COOP_THREADS, 0);
    if (occupancy != cudaSuccess || per_sm <= 0) return 0;
    int64_t ctas = (int64_t)per_sm * properties.multiProcessorCount / batch;
    const int64_t useful = nt * (nt + 1) / 2;
    if (ctas > useful) ctas = useful;
    if (ctas > coop051::COOP_MAX_CTAS) ctas = coop051::COOP_MAX_CTAS;
    return ctas;
}

void coop051_launch(torch::Tensor matrix) {
    TORCH_CHECK(matrix.is_cuda() && matrix.dim() == 3, "coop051 needs a 3-D CUDA tensor");
    TORCH_CHECK(matrix.scalar_type() == torch::kFloat32, "coop051 needs float32");
    TORCH_CHECK(matrix.is_contiguous(), "coop051 needs a contiguous tensor");
    int batch = (int)matrix.size(0);
    int n = (int)matrix.size(1);
    TORCH_CHECK(matrix.size(2) == n, "coop051 needs a square matrix");
    TORCH_CHECK(n % coop051::COOP_NB == 0, "coop051 needs n % 128 == 0");
    int nt = n / 32;
    int ctas = (int)coop051_ctas(batch, nt);
    TORCH_CHECK(ctas >= 1, "coop051 has no co-resident cooperative grid");
    float* ptr = matrix.data_ptr<float>();
    void* args[] = {&ptr, &n, &nt, &ctas};
    cudaError_t status = cudaLaunchCooperativeKernel(
        (void*)coop051::coop_cholesky_v2, dim3(batch * ctas),
        dim3(coop051::COOP_THREADS), args, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
