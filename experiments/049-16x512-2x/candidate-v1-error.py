"""Experiment 049 V1: full-resident cluster/DSM Cholesky for 16x512.

Sixteen independent Blackwell clusters each own one matrix.  Within a cluster,
sixteen CTAs keep one 32x512 row tile apiece in distributed shared memory for
the complete factorization.  No auxiliary CUDA queue is named.
"""

import statistics

import torch
import submission as _ranked


_COOP512_HITS = 0
_COOP512_READY_HITS = 0
_COOP512_FALLBACKS = 0
_COOP512_ERROR = None
_COOP512 = None
_COOP512_PHASE = None
_COOP512_PROFILE_PENDING = True

# The paired harness records only integer *_HITS/*_FALLBACKS deltas.  These
# one-shot counters therefore carry median device-globaltimer nanoseconds from
# the correctness call; timing calls reuse the buffer without a host readback.
_COOP512_PHASE_LOAD_NS_HITS = 0
_COOP512_PHASE_DIAG_NS_HITS = 0
_COOP512_PHASE_PANEL_NS_HITS = 0
_COOP512_PHASE_TRAILING_NS_HITS = 0
_COOP512_PHASE_STORE_NS_HITS = 0
_COOP512_PHASE_TOTAL_NS_HITS = 0


_COOP512_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int N = 512;
constexpr int TILE = 32;
constexpr int NT = N / TILE;
constexpr int MATRICES = 16;
constexpr int GROUP_BLOCKS = 16;
constexpr int THREADS = 128;
constexpr int ROW_FLOATS = TILE * N;
constexpr int RESIDUAL_FLOATS = TILE * TILE;
constexpr int SHARED_BYTES = (ROW_FLOATS + RESIDUAL_FLOATS) * sizeof(float);
static_assert(SHARED_BYTES == 69632, "unexpected resident shared-memory size");

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void __cluster_dims__(GROUP_BLOCKS, 1, 1)
cluster_cholesky_512(float* matrices, unsigned long long* phase) {
    cg::cluster_group cluster = cg::this_cluster();
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int matrix_id = static_cast<int>(blockIdx.x) / GROUP_BLOCKS;
    const int local_block = static_cast<int>(cluster.block_rank());
    float* matrix = matrices + static_cast<size_t>(matrix_id) * N * N;

    extern __shared__ float dynamic_shared[];
    float* row_tile = dynamic_shared;
    float* residual = dynamic_shared + ROW_FLOATS;
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];

    unsigned long long total_start = 0;
    unsigned long long phase_start = 0;
    unsigned long long load_ns = 0;
    unsigned long long diagonal_ns = 0;
    unsigned long long panel_ns = 0;
    unsigned long long trailing_ns = 0;
    unsigned long long store_ns = 0;

    if (local_block == 0 && thread == 0) {
        total_start = global_nanoseconds();
        phase_start = total_start;
    }
    const int row0 = local_block * TILE;
    for (int linear = thread; linear < ROW_FLOATS; linear += blockDim.x) {
        const int row = linear / N;
        const int column = linear - row * N;
        row_tile[linear] = matrix[(size_t)(row0 + row) * N + column];
    }
    cluster.sync();
    if (local_block == 0 && thread == 0) {
        const unsigned long long now = global_nanoseconds();
        load_ns = now - phase_start;
    }

    for (int kt = 0; kt < NT; ++kt) {
        const int k0 = kt * TILE;
        if (local_block == 0 && thread == 0) phase_start = global_nanoseconds();

        // Pivot CTA: FP32 register-row Cholesky, two pivots per iteration.
        if (local_block == kt && warp == 0) {
            float values[32];
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                values[column] = row_tile[lane * N + k0 + column];
            }
#pragma unroll
            for (int iteration = 0; iteration < 16; ++iteration) {
                const int k = 2 * iteration;
                const int q = k + 1;
                float inverse0 = lane == k
                    ? rsqrtf(fmaxf(values[k], 1.0e-20f)) : 0.0f;
                inverse0 = __shfl_sync(0xffffffffu, inverse0, k);
                if (lane >= k) values[k] *= inverse0;
                pivot0[lane] = values[k];
                __syncwarp();
                if (lane >= q) {
                    values[q] = fmaf(-values[k], pivot0[q], values[q]);
                }
                float inverse1 = lane == q
                    ? rsqrtf(fmaxf(values[q], 1.0e-20f)) : 0.0f;
                inverse1 = __shfl_sync(0xffffffffu, inverse1, q);
                if (lane >= q) values[q] *= inverse1;
                pivot1[lane] = values[q];
                __syncwarp();
                if (lane > q) {
                    const float scale0 = values[k];
                    const float scale1 = values[q];
#pragma unroll
                    for (int column = 0; column < 32; ++column) {
                        if (column > q && column <= lane) {
                            float value = fmaf(-scale0, pivot0[column], values[column]);
                            values[column] = fmaf(-scale1, pivot1[column], value);
                        }
                    }
                }
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                row_tile[lane * N + k0 + column] =
                    column <= lane ? values[column] : 0.0f;
            }
        }

        // Diagonal publication barrier: every active panel CTA reads the
        // pivot CTA's resident row tile through DSM after this point.
        cluster.sync();
        if (local_block == 0 && thread == 0) {
            const unsigned long long now = global_nanoseconds();
            diagonal_ns += now - phase_start;
            phase_start = now;
        }
        float* diagonal = cluster.map_shared_rank(row_tile, kt);

        // One warp solves the complete 32x32 panel tile.  Residual contains
        // x1 = x - tf32(x) for the three-term TF32x3 trailing product.
        if (local_block > kt && warp == 0) {
            float values[32];
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                values[column] = row_tile[lane * N + k0 + column];
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                float value = values[column];
#pragma unroll
                for (int p = 0; p < 32; ++p) {
                    if (p < column) {
                        value = fmaf(
                            -values[p], diagonal[column * N + k0 + p], value);
                    }
                }
                values[column] = value / diagonal[column * N + k0 + column];
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                const float value = values[column];
                row_tile[lane * N + k0 + column] = value;
                residual[lane * TILE + column] = value - __float_to_tf32(value);
            }
        }

        // Panel publication barrier: trailing CTAs may now read both the main
        // and residual panel tiles from any active CTA's DSM allocation.
        cluster.sync();
        if (local_block == 0 && thread == 0) {
            const unsigned long long now = global_nanoseconds();
            panel_ns += now - phase_start;
            phase_start = now;
        }

        if (local_block > kt) {
            const int warp_row = warp >> 1;
            const int warp_column = warp & 1;
            const float* lhs0_ptr =
                row_tile + (warp_row * 16) * N + k0;
            const float* lhs1_ptr =
                residual + (warp_row * 16) * TILE;

            // This CTA exclusively owns every lower 32x32 tile in its row.
            for (int bj = kt + 1; bj <= local_block; ++bj) {
                float* remote_row = cluster.map_shared_rank(row_tile, bj);
                float* remote_residual = cluster.map_shared_rank(residual, bj);
                const float* rhs0_ptr =
                    remote_row + (warp_column * 16) * N + k0;
                const float* rhs1_ptr =
                    remote_residual + (warp_column * 16) * TILE;
                float* target = row_tile
                    + (warp_row * 16) * N + bj * TILE + warp_column * 16;

                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> lhs0;
                wmma::fragment<wmma::matrix_a, 16, 16, 8,
                               wmma::precision::tf32, wmma::row_major> lhs1;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::col_major> rhs0;
                wmma::fragment<wmma::matrix_b, 16, 16, 8,
                               wmma::precision::tf32, wmma::col_major> rhs1;
                wmma::fragment<wmma::accumulator, 16, 16, 8, float> accumulator;
                wmma::fill_fragment(accumulator, 0.0f);

#pragma unroll
                for (int depth = 0; depth < 32; depth += 8) {
                    wmma::load_matrix_sync(lhs0, lhs0_ptr + depth, N);
                    wmma::load_matrix_sync(lhs1, lhs1_ptr + depth, TILE);
                    wmma::load_matrix_sync(rhs0, rhs0_ptr + depth, N);
                    wmma::load_matrix_sync(rhs1, rhs1_ptr + depth, TILE);
                    wmma::mma_sync(accumulator, lhs0, rhs0, accumulator);
                    wmma::mma_sync(accumulator, lhs1, rhs0, accumulator);
                    wmma::mma_sync(accumulator, lhs0, rhs1, accumulator);
                }

                wmma::fragment<wmma::accumulator, 16, 16, 8, float> original;
                wmma::load_matrix_sync(original, target, N, wmma::mem_row_major);
#pragma unroll
                for (int element = 0; element < accumulator.num_elements; ++element) {
                    original.x[element] -= accumulator.x[element];
                }
                wmma::store_matrix_sync(target, original, N, wmma::mem_row_major);
            }
        }

        // Trailing publication barrier: the next pivot block is read only
        // after all owner CTAs have completed the current Schur update.
        cluster.sync();
        if (local_block == 0 && thread == 0) {
            const unsigned long long now = global_nanoseconds();
            trailing_ns += now - phase_start;
        }
    }

    if (local_block == 0 && thread == 0) phase_start = global_nanoseconds();
    for (int linear = thread; linear < ROW_FLOATS; linear += blockDim.x) {
        const int row = linear / N;
        const int column = linear - row * N;
        const int global_row = row0 + row;
        matrix[(size_t)global_row * N + column] =
            column <= global_row ? row_tile[linear] : 0.0f;
    }
    cluster.sync();
    if (local_block == 0 && thread == 0) {
        const unsigned long long end = global_nanoseconds();
        store_ns = end - phase_start;
        unsigned long long* row = phase + (size_t)matrix_id * 6;
        row[0] = load_ns;
        row[1] = diagonal_ns;
        row[2] = panel_ns;
        row[3] = trailing_ns;
        row[4] = store_ns;
        row[5] = end - total_start;
    }
}

void chol512_launch(torch::Tensor matrix, torch::Tensor phase) {
    TORCH_CHECK(matrix.is_cuda() && matrix.scalar_type() == torch::kFloat32,
                "chol512 requires a CUDA float32 matrix");
    TORCH_CHECK(matrix.is_contiguous() && matrix.dim() == 3 &&
                matrix.size(0) == MATRICES && matrix.size(1) == N &&
                matrix.size(2) == N, "chol512 requires 16x512x512 contiguous");
    TORCH_CHECK(phase.is_cuda() && phase.scalar_type() == torch::kInt64 &&
                phase.is_contiguous() && phase.numel() == MATRICES * 6,
                "phase must be CUDA int64[16,6]");

    static bool configured = false;
    if (!configured) {
        cudaError_t status = cudaFuncSetAttribute(
            cluster_cholesky_512,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
        status = cudaFuncSetAttribute(
            cluster_cholesky_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES);
        TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
        configured = true;
    }
    float* ptr = matrix.data_ptr<float>();
    unsigned long long* phase_ptr =
        reinterpret_cast<unsigned long long*>(phase.data_ptr<int64_t>());
    cluster_cholesky_512<<<MATRICES * GROUP_BLOCKS, THREADS, SHARED_BYTES>>>(
        ptr, phase_ptr);
    cudaError_t status = cudaPeekAtLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""


if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _COOP512 = load_inline(
            name="chol512_exp049_v1_error_cluster16_resident_tf32x3",
            cpp_sources="void chol512_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_COOP512_SOURCE,
            functions=["chol512_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    except Exception as exc:
        _COOP512_ERROR = repr(exc)
        raise RuntimeError(
            "EXPERIMENT_049_V1_EXTENSION_ERROR: " + _COOP512_ERROR
        ) from exc


def _capture_phase_once() -> None:
    global _COOP512_PROFILE_PENDING
    global _COOP512_PHASE_LOAD_NS_HITS, _COOP512_PHASE_DIAG_NS_HITS
    global _COOP512_PHASE_PANEL_NS_HITS, _COOP512_PHASE_TRAILING_NS_HITS
    global _COOP512_PHASE_STORE_NS_HITS, _COOP512_PHASE_TOTAL_NS_HITS
    if not _COOP512_PROFILE_PENDING:
        return
    torch.cuda.synchronize()
    rows = _COOP512_PHASE.cpu().tolist()
    medians = [int(statistics.median(row[index] for row in rows)) for index in range(6)]
    (
        _COOP512_PHASE_LOAD_NS_HITS,
        _COOP512_PHASE_DIAG_NS_HITS,
        _COOP512_PHASE_PANEL_NS_HITS,
        _COOP512_PHASE_TRAILING_NS_HITS,
        _COOP512_PHASE_STORE_NS_HITS,
        _COOP512_PHASE_TOTAL_NS_HITS,
    ) = medians
    _COOP512_PROFILE_PENDING = False


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _COOP512_HITS, _COOP512_READY_HITS, _COOP512_FALLBACKS
    global _COOP512_PHASE
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (16, 512, 512)
        and data.is_contiguous()
    )
    if target and _COOP512 is not None:
        if _COOP512_PHASE is None:
            _COOP512_PHASE = torch.empty((16, 6), dtype=torch.int64, device=data.device)
        output = data.clone()
        _COOP512.chol512_launch(output, _COOP512_PHASE)
        _COOP512_HITS += 1
        _COOP512_READY_HITS += 1
        _capture_phase_once()
        return output
    if target:
        _COOP512_FALLBACKS += 1
    return _ranked.custom_kernel(data)
