"""Experiment 049 V4: occupancy-gated rank-128 atomic superpanels."""

import statistics

import torch
import submission as _ranked


_SUPER512_HITS = 0
_SUPER512_READY_HITS = 0
_SUPER512_OCCUPANCY_HITS = 0
_SUPER512_FALLBACKS = 0
_SUPER512_ERROR = None
_SUPER512 = None
_SUPER512_PHASE = None
_SUPER512_PANEL = None
_SUPER512_RESIDUAL = None
_SUPER512_BARRIER = None
_SUPER512_PROFILE_PENDING = True
_SUPER512_PHASE_LOAD_NS_HITS = 0
_SUPER512_PHASE_DIAG_NS_HITS = 0
_SUPER512_PHASE_BARRIER_NS_HITS = 0
_SUPER512_PHASE_PANEL_NS_HITS = 0
_SUPER512_PHASE_TRAILING_NS_HITS = 0
_SUPER512_PHASE_STORE_NS_HITS = 0
_SUPER512_PHASE_TOTAL_NS_HITS = 0


_SUPER512_SOURCE_V3_UNUSED = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int N = 512;
constexpr int TILE = 32;
constexpr int NT = N / TILE;
constexpr int MATRICES = 16;
constexpr int GROUP_BLOCKS = 16;
constexpr int TOTAL_BLOCKS = MATRICES * GROUP_BLOCKS;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int PHASES = 7;

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ float round_tf32(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00000fffu + ((bits >> 13) & 1u);
    return __uint_as_float(bits & 0xffffe000u);
}

__device__ __forceinline__ void matrix_barrier(
    int* arrivals, int* epochs, int matrix_id, int& local_epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int target_epoch = local_epoch + 1;
        const int ticket = atomicAdd(arrivals + matrix_id, 1);
        if (ticket == GROUP_BLOCKS - 1) {
            atomicExch(arrivals + matrix_id, 0);
            __threadfence();
            atomicExch(epochs + matrix_id, target_epoch);
        } else {
            while (atomicAdd(epochs + matrix_id, 0) < target_epoch) {
                __nanosleep(64);
            }
        }
        __threadfence();
        local_epoch = target_epoch;
    }
    __syncthreads();
}

__global__ void atomic_group_cholesky_512(
    float* matrices, float* panel_workspace, float* residual_workspace,
    int* arrivals, int* epochs, unsigned long long* phase) {
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int matrix_id = static_cast<int>(blockIdx.x) / GROUP_BLOCKS;
    const int local_block = static_cast<int>(blockIdx.x) % GROUP_BLOCKS;
    float* matrix = matrices + static_cast<size_t>(matrix_id) * N * N;
    float* panel = panel_workspace + static_cast<size_t>(matrix_id) * N * TILE;
    float* residual = residual_workspace + static_cast<size_t>(matrix_id) * N * TILE;
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];

    int local_epoch = atomicAdd(epochs + matrix_id, 0);
    unsigned long long total_start = 0;
    unsigned long long phase_start = 0;
    unsigned long long barrier_start = 0;
    unsigned long long load_ns = 0;
    unsigned long long diagonal_ns = 0;
    unsigned long long barrier_ns = 0;
    unsigned long long panel_ns = 0;
    unsigned long long trailing_ns = 0;
    unsigned long long store_ns = 0;
    if (thread == 0) total_start = global_nanoseconds();

    for (int kt = 0; kt < NT; ++kt) {
        const int k0 = kt * TILE;

        if (local_block == 0 && warp == 0) {
            float values[32];
            if (lane == 0) phase_start = global_nanoseconds();
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                values[column] = matrix[(size_t)(k0 + lane) * N + k0 + column];
            }
            __syncwarp();
            if (lane == 0) {
                const unsigned long long now = global_nanoseconds();
                load_ns += now - phase_start;
                phase_start = now;
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
                if (lane >= q) values[q] = fmaf(-values[k], pivot0[q], values[q]);
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
                matrix[(size_t)(k0 + lane) * N + k0 + column] =
                    column <= lane ? values[column] : 0.0f;
            }
            if (lane == 0) diagonal_ns += global_nanoseconds() - phase_start;
        }

        if (thread == 0) barrier_start = global_nanoseconds();
        matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
        if (thread == 0) {
            barrier_ns += global_nanoseconds() - barrier_start;
            phase_start = global_nanoseconds();
        }

        // One active CTA solves one complete 32x32 panel tile.
        if (local_block > kt && warp == 0) {
            const int global_row = local_block * TILE + lane;
            float values[32];
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                values[column] = matrix[(size_t)global_row * N + k0 + column];
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                float value = values[column];
#pragma unroll
                for (int p = 0; p < 32; ++p) {
                    if (p < column) {
                        value = fmaf(
                            -values[p], matrix[(size_t)(k0 + column) * N + k0 + p],
                            value);
                    }
                }
                values[column] =
                    value / matrix[(size_t)(k0 + column) * N + k0 + column];
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                const float value = values[column];
                matrix[(size_t)global_row * N + k0 + column] = value;
                panel[global_row * TILE + column] = value;
                residual[global_row * TILE + column] = value - round_tf32(value);
            }
        }
        if (thread == 0) panel_ns += global_nanoseconds() - phase_start;

        if (thread == 0) barrier_start = global_nanoseconds();
        matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
        if (thread == 0) {
            barrier_ns += global_nanoseconds() - barrier_start;
            phase_start = global_nanoseconds();
        }

        const int remaining_tiles = NT - kt - 1;
        const int pair_count = remaining_tiles * (remaining_tiles + 1) / 2;
        for (int pair = local_block; pair < pair_count; pair += GROUP_BLOCKS) {
            const int local_bi =
                (int)((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
            const int local_bj = pair - local_bi * (local_bi + 1) / 2;
            const int bi = kt + 1 + local_bi;
            const int bj = kt + 1 + local_bj;
            const int warp_row = warp >> 1;
            const int warp_column = warp & 1;
            const int panel_row_a = bi * TILE + warp_row * 16;
            const int panel_row_b = bj * TILE + warp_column * 16;
            const float* lhs0_ptr = panel + panel_row_a * TILE;
            const float* lhs1_ptr = residual + panel_row_a * TILE;
            const float* rhs0_ptr = panel + panel_row_b * TILE;
            const float* rhs1_ptr = residual + panel_row_b * TILE;
            float* target = matrix
                + (size_t)panel_row_a * N + bj * TILE + warp_column * 16;

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
                wmma::load_matrix_sync(lhs0, lhs0_ptr + depth, TILE);
                wmma::load_matrix_sync(lhs1, lhs1_ptr + depth, TILE);
                wmma::load_matrix_sync(rhs0, rhs0_ptr + depth, TILE);
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
        if (thread == 0) trailing_ns += global_nanoseconds() - phase_start;

        if (thread == 0) barrier_start = global_nanoseconds();
        matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
        if (thread == 0) barrier_ns += global_nanoseconds() - barrier_start;
    }

    if (thread == 0) phase_start = global_nanoseconds();
    const int row0 = local_block * TILE;
    for (int linear = thread; linear < TILE * N; linear += blockDim.x) {
        const int row = row0 + linear / N;
        const int column = linear % N;
        if (column > row) matrix[(size_t)row * N + column] = 0.0f;
    }
    if (thread == 0) store_ns = global_nanoseconds() - phase_start;
    if (thread == 0) barrier_start = global_nanoseconds();
    matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
    if (thread == 0) {
        barrier_ns += global_nanoseconds() - barrier_start;
        const unsigned long long end = global_nanoseconds();
        unsigned long long* row = phase
            + ((size_t)matrix_id * GROUP_BLOCKS + local_block) * PHASES;
        row[0] = load_ns;
        row[1] = diagonal_ns;
        row[2] = barrier_ns;
        row[3] = panel_ns;
        row[4] = trailing_ns;
        row[5] = store_ns;
        row[6] = end - total_start;
    }
}

int64_t chol512_capacity() {
    int active = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, atomic_group_cholesky_512, THREADS, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    int device = 0;
    status = cudaGetDevice(&device);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    return (int64_t)active * properties.multiProcessorCount;
}

void chol512_launch(
    torch::Tensor matrix, torch::Tensor panel, torch::Tensor residual,
    torch::Tensor barrier, torch::Tensor phase) {
    TORCH_CHECK(matrix.is_cuda() && matrix.scalar_type() == torch::kFloat32 &&
                matrix.is_contiguous() && matrix.dim() == 3 &&
                matrix.size(0) == MATRICES && matrix.size(1) == N &&
                matrix.size(2) == N,
                "matrix must be CUDA float32 16x512x512 contiguous");
    TORCH_CHECK(panel.is_cuda() && panel.scalar_type() == torch::kFloat32 &&
                panel.is_contiguous() && panel.numel() == MATRICES * N * TILE,
                "panel workspace mismatch");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kFloat32 &&
                residual.is_contiguous() && residual.numel() == MATRICES * N * TILE,
                "residual workspace mismatch");
    TORCH_CHECK(barrier.is_cuda() && barrier.scalar_type() == torch::kInt32 &&
                barrier.is_contiguous() && barrier.numel() == MATRICES * 2,
                "barrier state mismatch");
    TORCH_CHECK(phase.is_cuda() && phase.scalar_type() == torch::kInt64 &&
                phase.is_contiguous() && phase.numel() == TOTAL_BLOCKS * PHASES,
                "phase workspace mismatch");
    const int64_t capacity = chol512_capacity();
    TORCH_CHECK(capacity >= TOTAL_BLOCKS,
                "atomic CTA barrier unsafe: resident capacity ", capacity,
                " < required ", TOTAL_BLOCKS);
    float* matrix_ptr = matrix.data_ptr<float>();
    float* panel_ptr = panel.data_ptr<float>();
    float* residual_ptr = residual.data_ptr<float>();
    int* arrivals = barrier.data_ptr<int>();
    int* epochs = arrivals + MATRICES;
    unsigned long long* phase_ptr =
        reinterpret_cast<unsigned long long*>(phase.data_ptr<int64_t>());
    atomic_group_cholesky_512<<<TOTAL_BLOCKS, THREADS>>>(
        matrix_ptr, panel_ptr, residual_ptr, arrivals, epochs, phase_ptr);
    cudaError_t status = cudaPeekAtLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""


_SUPER512_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int N = 512;
constexpr int SUPER = 128;
constexpr int NS = N / SUPER;
constexpr int BK = 16;
constexpr int TILE = 32;
constexpr int DIAG_STRIDE = 129;
constexpr int MATRICES = 16;
constexpr int GROUP_BLOCKS = 16;
constexpr int TOTAL_BLOCKS = MATRICES * GROUP_BLOCKS;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int PHASES = 7;
constexpr int SHARED_BYTES = SUPER * DIAG_STRIDE * sizeof(float);
static_assert(SHARED_BYTES == 66048, "unexpected rank-128 shared size");

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ __forceinline__ float round_tf32(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00000fffu + ((bits >> 13) & 1u);
    return __uint_as_float(bits & 0xffffe000u);
}

__device__ __forceinline__ void matrix_barrier(
    int* arrivals, int* epochs, int matrix_id, int& local_epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const int target_epoch = local_epoch + 1;
        const int ticket = atomicAdd(arrivals + matrix_id, 1);
        if (ticket == GROUP_BLOCKS - 1) {
            atomicExch(arrivals + matrix_id, 0);
            __threadfence();
            atomicExch(epochs + matrix_id, target_epoch);
        } else {
            while (atomicAdd(epochs + matrix_id, 0) < target_epoch) {
                __nanosleep(64);
            }
        }
        __threadfence();
        local_epoch = target_epoch;
    }
    __syncthreads();
}

__device__ __noinline__ void factor_diagonal128(
    float* matrix, float* diagonal, int k0) {
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    __shared__ float reciprocal;
    __shared__ float inverse_diag[BK];

#pragma unroll 1
    for (int block = 0; block < SUPER; block += BK) {
        const int block_end = block + BK;
#pragma unroll
        for (int local = 0; local < BK; ++local) {
            const int pivot = block + local;
            if (thread == 0) {
                reciprocal = rsqrtf(fmaxf(
                    diagonal[pivot * DIAG_STRIDE + pivot], 1.0e-20f));
                inverse_diag[local] = reciprocal;
                diagonal[pivot * DIAG_STRIDE + pivot] *= reciprocal;
            }
            __syncthreads();
            const int panel_row = pivot + 1 + thread;
            if (panel_row < block_end) {
                diagonal[panel_row * DIAG_STRIDE + pivot] *= reciprocal;
            }
            __syncthreads();
            for (int linear = thread; linear < BK * BK; linear += blockDim.x) {
                const int row = block + (linear >> 4);
                const int column = block + (linear & 15);
                if (row > pivot && column > pivot && column <= row) {
                    const int offset = row * DIAG_STRIDE + column;
                    diagonal[offset] = fmaf(
                        -diagonal[row * DIAG_STRIDE + pivot],
                        diagonal[column * DIAG_STRIDE + pivot], diagonal[offset]);
                }
            }
            __syncthreads();
        }

        const int row = block_end + thread;
        if (row < SUPER) {
#pragma unroll
            for (int local = 0; local < BK; ++local) {
                const int column = block + local;
                float value = diagonal[row * DIAG_STRIDE + column];
#pragma unroll
                for (int prior = 0; prior < BK; ++prior) {
                    if (prior < local) {
                        value = fmaf(
                            -diagonal[row * DIAG_STRIDE + block + prior],
                            diagonal[column * DIAG_STRIDE + block + prior], value);
                    }
                }
                diagonal[row * DIAG_STRIDE + column] =
                    value * inverse_diag[local];
            }
        }
        __syncthreads();

        for (int trailing_row = block_end + warp;
             trailing_row < SUPER; trailing_row += WARPS) {
            for (int column = block_end + lane;
                 column <= trailing_row; column += 32) {
                float update = 0.0f;
#pragma unroll
                for (int depth = 0; depth < BK; ++depth) {
                    update = fmaf(
                        diagonal[trailing_row * DIAG_STRIDE + block + depth],
                        diagonal[column * DIAG_STRIDE + block + depth], update);
                }
                diagonal[trailing_row * DIAG_STRIDE + column] -= update;
            }
        }
        __syncthreads();
    }

    for (int linear = thread; linear < SUPER * SUPER; linear += blockDim.x) {
        const int row = linear / SUPER;
        const int column = linear - row * SUPER;
        matrix[(size_t)(k0 + row) * N + k0 + column] =
            column <= row ? diagonal[row * DIAG_STRIDE + column] : 0.0f;
    }
}

__device__ __noinline__ void solve_panel128(
    float* matrix, float* panel, float* residual, int k0, int local_block) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int tile_index = local_block - 1;
    const int global_row = k0 + SUPER + tile_index * TILE + lane;
    if (local_block < 1 || global_row >= N || warp != 0) return;

    // Four register-32 segments avoid a 128-value live range while retaining
    // an exact FP32 triangular solve against the completed superdiagonal.
    for (int segment = 0; segment < SUPER; segment += TILE) {
        float values[TILE];
#pragma unroll
        for (int column = 0; column < TILE; ++column) {
            values[column] =
                matrix[(size_t)global_row * N + k0 + segment + column];
        }
#pragma unroll
        for (int column = 0; column < TILE; ++column) {
            const int super_column = segment + column;
            float value = values[column];
            for (int prior = 0; prior < SUPER; ++prior) {
                if (prior < segment) {
                    value = fmaf(
                        -panel[global_row * SUPER + prior],
                        matrix[(size_t)(k0 + super_column) * N + k0 + prior],
                        value);
                }
            }
#pragma unroll
            for (int prior = 0; prior < TILE; ++prior) {
                if (prior < column) {
                    value = fmaf(
                        -values[prior],
                        matrix[(size_t)(k0 + super_column) * N
                               + k0 + segment + prior], value);
                }
            }
            values[column] = value /
                matrix[(size_t)(k0 + super_column) * N + k0 + super_column];
        }
#pragma unroll
        for (int column = 0; column < TILE; ++column) {
            const int super_column = segment + column;
            const float value = values[column];
            matrix[(size_t)global_row * N + k0 + super_column] = value;
            panel[global_row * SUPER + super_column] = value;
            residual[global_row * SUPER + super_column] = value - round_tf32(value);
        }
    }
}

__device__ __noinline__ void update_trailing128(
    float* matrix, const float* panel, const float* residual,
    int super_step, int local_block) {
    const int warp = threadIdx.x >> 5;
    const int remaining_tiles = NS - super_step - 1;
    const int pair_count = remaining_tiles * (remaining_tiles + 1) / 2;
    const int group_warp = local_block * WARPS + warp;
    const int group_warps = GROUP_BLOCKS * WARPS;
    const int quarter_count = pair_count * 64;
    for (int task = group_warp; task < quarter_count; task += group_warps) {
        const int pair = task >> 6;
        const int quarter = task & 63;
        const int local_bi =
            (int)((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
        const int local_bj = pair - local_bi * (local_bi + 1) / 2;
        const int bi = super_step + 1 + local_bi;
        const int bj = super_step + 1 + local_bj;
        const int warp_row = quarter >> 3;
        const int warp_column = quarter & 7;
        const int panel_row_a = bi * SUPER + warp_row * 16;
        const int panel_row_b = bj * SUPER + warp_column * 16;
        const float* lhs0_ptr = panel + panel_row_a * SUPER;
        const float* lhs1_ptr = residual + panel_row_a * SUPER;
        const float* rhs0_ptr = panel + panel_row_b * SUPER;
        const float* rhs1_ptr = residual + panel_row_b * SUPER;
        float* target = matrix
            + (size_t)panel_row_a * N + bj * SUPER + warp_column * 16;
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
        for (int depth = 0; depth < SUPER; depth += 8) {
            wmma::load_matrix_sync(lhs0, lhs0_ptr + depth, SUPER);
            wmma::load_matrix_sync(lhs1, lhs1_ptr + depth, SUPER);
            wmma::load_matrix_sync(rhs0, rhs0_ptr + depth, SUPER);
            wmma::load_matrix_sync(rhs1, rhs1_ptr + depth, SUPER);
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

__global__ void atomic_super128_cholesky_512(
    float* matrices, float* panel_workspace, float* residual_workspace,
    int* arrivals, int* epochs, unsigned long long* phase) {
    const int thread = threadIdx.x;
    const int matrix_id = static_cast<int>(blockIdx.x) / GROUP_BLOCKS;
    const int local_block = static_cast<int>(blockIdx.x) % GROUP_BLOCKS;
    float* matrix = matrices + static_cast<size_t>(matrix_id) * N * N;
    float* panel = panel_workspace + static_cast<size_t>(matrix_id) * N * SUPER;
    float* residual = residual_workspace + static_cast<size_t>(matrix_id) * N * SUPER;
    extern __shared__ float diagonal[];
    int local_epoch = atomicAdd(epochs + matrix_id, 0);
    unsigned long long total_start = 0;
    unsigned long long phase_start = 0;
    unsigned long long barrier_start = 0;
    unsigned long long load_ns = 0;
    unsigned long long diagonal_ns = 0;
    unsigned long long barrier_ns = 0;
    unsigned long long panel_ns = 0;
    unsigned long long trailing_ns = 0;
    unsigned long long store_ns = 0;
    if (thread == 0) total_start = global_nanoseconds();

    for (int step = 0; step < NS; ++step) {
        const int k0 = step * SUPER;
        if (local_block == 0) {
            if (thread == 0) phase_start = global_nanoseconds();
            for (int linear = thread; linear < SUPER * SUPER; linear += blockDim.x) {
                const int row = linear / SUPER;
                const int column = linear - row * SUPER;
                diagonal[row * DIAG_STRIDE + column] =
                    matrix[(size_t)(k0 + row) * N + k0 + column];
            }
            __syncthreads();
            if (thread == 0) {
                const unsigned long long now = global_nanoseconds();
                load_ns += now - phase_start;
                phase_start = now;
            }
            factor_diagonal128(matrix, diagonal, k0);
            if (thread == 0) diagonal_ns += global_nanoseconds() - phase_start;
        }
        if (thread == 0) barrier_start = global_nanoseconds();
        matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
        if (thread == 0) {
            barrier_ns += global_nanoseconds() - barrier_start;
            phase_start = global_nanoseconds();
        }
        solve_panel128(matrix, panel, residual, k0, local_block);
        if (thread == 0) panel_ns += global_nanoseconds() - phase_start;
        if (thread == 0) barrier_start = global_nanoseconds();
        matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
        if (thread == 0) {
            barrier_ns += global_nanoseconds() - barrier_start;
            phase_start = global_nanoseconds();
        }
        update_trailing128(matrix, panel, residual, step, local_block);
        if (thread == 0) trailing_ns += global_nanoseconds() - phase_start;
        if (thread == 0) barrier_start = global_nanoseconds();
        matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
        if (thread == 0) barrier_ns += global_nanoseconds() - barrier_start;
    }

    if (thread == 0) phase_start = global_nanoseconds();
    const int row0 = local_block * TILE;
    for (int linear = thread; linear < TILE * N; linear += blockDim.x) {
        const int row = row0 + linear / N;
        const int column = linear % N;
        if (column > row) matrix[(size_t)row * N + column] = 0.0f;
    }
    if (thread == 0) store_ns = global_nanoseconds() - phase_start;
    if (thread == 0) barrier_start = global_nanoseconds();
    matrix_barrier(arrivals, epochs, matrix_id, local_epoch);
    if (thread == 0) {
        barrier_ns += global_nanoseconds() - barrier_start;
        const unsigned long long end = global_nanoseconds();
        unsigned long long* row = phase
            + ((size_t)matrix_id * GROUP_BLOCKS + local_block) * PHASES;
        row[0] = load_ns;
        row[1] = diagonal_ns;
        row[2] = barrier_ns;
        row[3] = panel_ns;
        row[4] = trailing_ns;
        row[5] = store_ns;
        row[6] = end - total_start;
    }
}

void configure_super128() {
    static bool configured = false;
    if (!configured) {
        cudaError_t status = cudaFuncSetAttribute(
            atomic_super128_cholesky_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES);
        TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
        configured = true;
    }
}

int64_t chol512_capacity() {
    configure_super128();
    int active = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, atomic_super128_cholesky_512, THREADS, SHARED_BYTES);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    int device = 0;
    status = cudaGetDevice(&device);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    cudaDeviceProp properties;
    status = cudaGetDeviceProperties(&properties, device);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    return (int64_t)active * properties.multiProcessorCount;
}

void chol512_launch(
    torch::Tensor matrix, torch::Tensor panel, torch::Tensor residual,
    torch::Tensor barrier, torch::Tensor phase) {
    TORCH_CHECK(matrix.is_cuda() && matrix.scalar_type() == torch::kFloat32 &&
                matrix.is_contiguous() && matrix.dim() == 3 &&
                matrix.size(0) == MATRICES && matrix.size(1) == N && matrix.size(2) == N,
                "matrix must be CUDA float32 16x512x512 contiguous");
    TORCH_CHECK(panel.is_cuda() && panel.scalar_type() == torch::kFloat32 &&
                panel.is_contiguous() && panel.numel() == MATRICES * N * SUPER,
                "panel workspace mismatch");
    TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kFloat32 &&
                residual.is_contiguous() && residual.numel() == MATRICES * N * SUPER,
                "residual workspace mismatch");
    TORCH_CHECK(barrier.is_cuda() && barrier.scalar_type() == torch::kInt32 &&
                barrier.is_contiguous() && barrier.numel() == MATRICES * 2,
                "barrier state mismatch");
    TORCH_CHECK(phase.is_cuda() && phase.scalar_type() == torch::kInt64 &&
                phase.is_contiguous() && phase.numel() == TOTAL_BLOCKS * PHASES,
                "phase workspace mismatch");
    const int64_t capacity = chol512_capacity();
    TORCH_CHECK(capacity >= TOTAL_BLOCKS,
                "rank128 atomic barrier unsafe: resident capacity ", capacity,
                " < required ", TOTAL_BLOCKS);
    float* matrix_ptr = matrix.data_ptr<float>();
    float* panel_ptr = panel.data_ptr<float>();
    float* residual_ptr = residual.data_ptr<float>();
    int* arrivals = barrier.data_ptr<int>();
    int* epochs = arrivals + MATRICES;
    unsigned long long* phase_ptr =
        reinterpret_cast<unsigned long long*>(phase.data_ptr<int64_t>());
    atomic_super128_cholesky_512<<<TOTAL_BLOCKS, THREADS, SHARED_BYTES>>>(
        matrix_ptr, panel_ptr, residual_ptr, arrivals, epochs, phase_ptr);
    cudaError_t status = cudaPeekAtLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""


if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _SUPER512 = load_inline(
            name="chol512_exp049_v4_super128_atomic_tf32x3",
            cpp_sources=(
                "void chol512_launch(torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor); int64_t chol512_capacity();"
            ),
            cuda_sources=_SUPER512_SOURCE,
            functions=["chol512_launch", "chol512_capacity"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _SUPER512_ERROR = repr(exc)


def _capture_phase_once() -> None:
    global _SUPER512_PROFILE_PENDING
    global _SUPER512_PHASE_LOAD_NS_HITS, _SUPER512_PHASE_DIAG_NS_HITS
    global _SUPER512_PHASE_BARRIER_NS_HITS, _SUPER512_PHASE_PANEL_NS_HITS
    global _SUPER512_PHASE_TRAILING_NS_HITS, _SUPER512_PHASE_STORE_NS_HITS
    global _SUPER512_PHASE_TOTAL_NS_HITS
    if not _SUPER512_PROFILE_PENDING:
        return
    torch.cuda.synchronize()
    rows = _SUPER512_PHASE.cpu().tolist()
    # Critical phase per matrix is the slowest of its 16 CTA owners; report the
    # median critical matrix so phase counters remain robust to one tail CTA.
    matrix_maxima = []
    for matrix_rows in rows:
        matrix_maxima.append([
            max(block_row[index] for block_row in matrix_rows)
            for index in range(7)
        ])
    medians = [
        int(statistics.median(row[index] for row in matrix_maxima))
        for index in range(7)
    ]
    (
        _SUPER512_PHASE_LOAD_NS_HITS,
        _SUPER512_PHASE_DIAG_NS_HITS,
        _SUPER512_PHASE_BARRIER_NS_HITS,
        _SUPER512_PHASE_PANEL_NS_HITS,
        _SUPER512_PHASE_TRAILING_NS_HITS,
        _SUPER512_PHASE_STORE_NS_HITS,
        _SUPER512_PHASE_TOTAL_NS_HITS,
    ) = medians
    _SUPER512_PROFILE_PENDING = False


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _SUPER512_HITS, _SUPER512_READY_HITS, _SUPER512_OCCUPANCY_HITS
    global _SUPER512_FALLBACKS, _SUPER512_PHASE, _SUPER512_PANEL
    global _SUPER512_RESIDUAL, _SUPER512_BARRIER
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (16, 512, 512)
        and data.is_contiguous()
    )
    if target and _SUPER512 is not None:
        if _SUPER512_PHASE is None:
            _SUPER512_PHASE = torch.empty(
                (16, 16, 7), dtype=torch.int64, device=data.device)
            _SUPER512_PANEL = torch.empty(
                (16, 512, 128), dtype=torch.float32, device=data.device)
            _SUPER512_RESIDUAL = torch.empty_like(_SUPER512_PANEL)
            _SUPER512_BARRIER = torch.zeros(32, dtype=torch.int32, device=data.device)
            _SUPER512_OCCUPANCY_HITS = int(_SUPER512.chol512_capacity())
        output = data.clone()
        _SUPER512.chol512_launch(
            output, _SUPER512_PANEL, _SUPER512_RESIDUAL,
            _SUPER512_BARRIER, _SUPER512_PHASE,
        )
        _SUPER512_HITS += 1
        _SUPER512_READY_HITS += 1
        _capture_phase_once()
        return output
    if target:
        _SUPER512_FALLBACKS += 1
    return _ranked.custom_kernel(data)
