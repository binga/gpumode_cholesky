"""Experiment 049 V2: one persistent 512-thread CTA per 512x512 matrix."""

import statistics

import torch
import submission as _ranked


_CTA512_HITS = 0
_CTA512_READY_HITS = 0
_CTA512_FALLBACKS = 0
_CTA512_ERROR = None
_CTA512 = None
_CTA512_PHASE = None
_CTA512_PROFILE_PENDING = True
_CTA512_PHASE_LOAD_NS_HITS = 0
_CTA512_PHASE_DIAG_NS_HITS = 0
_CTA512_PHASE_PANEL_NS_HITS = 0
_CTA512_PHASE_TRAILING_NS_HITS = 0
_CTA512_PHASE_STORE_NS_HITS = 0
_CTA512_PHASE_TOTAL_NS_HITS = 0


_CTA512_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

constexpr int N = 512;
constexpr int TILE = 32;
constexpr int NT = N / TILE;
constexpr int MATRICES = 16;
constexpr int THREADS = 512;
constexpr int WARPS = THREADS / 32;
constexpr int PANEL_FLOATS = N * TILE;
constexpr int SHARED_BYTES = 2 * PANEL_FLOATS * sizeof(float);
static_assert(SHARED_BYTES == 131072, "unexpected panel shared-memory size");

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

__global__ void persistent_cta_cholesky_512(
    float* matrices, unsigned long long* phase) {
    const int thread = threadIdx.x;
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int matrix_id = static_cast<int>(blockIdx.x);
    float* matrix = matrices + static_cast<size_t>(matrix_id) * N * N;
    extern __shared__ float dynamic_shared[];
    float* panel = dynamic_shared;
    float* residual = dynamic_shared + PANEL_FLOATS;
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];

    unsigned long long total_start = 0;
    unsigned long long phase_start = 0;
    unsigned long long load_ns = 0;
    unsigned long long diagonal_ns = 0;
    unsigned long long panel_ns = 0;
    unsigned long long trailing_ns = 0;
    unsigned long long store_ns = 0;
    if (thread == 0) total_start = global_nanoseconds();

    for (int kt = 0; kt < NT; ++kt) {
        const int k0 = kt * TILE;

        // Warp 0 loads and factors the current FP32 diagonal tile.  Keeping
        // the 32 rows in registers exposes the serial pivot-chain cost.
        if (warp == 0) {
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
                const float value = column <= lane ? values[column] : 0.0f;
                panel[lane * TILE + column] = value;
                matrix[(size_t)(k0 + lane) * N + k0 + column] = value;
            }
        }
        __syncthreads();
        if (thread == 0) {
            const unsigned long long now = global_nanoseconds();
            diagonal_ns += now - phase_start;
            phase_start = now;
        }

        // One thread owns one complete below-diagonal row.  The solved panel
        // and its TF32 residual remain resident for the trailing MMA phase.
        const int global_row = k0 + TILE + thread;
        if (global_row < N) {
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
                        value = fmaf(-values[p], panel[column * TILE + p], value);
                    }
                }
                values[column] = value / panel[column * TILE + column];
            }
#pragma unroll
            for (int column = 0; column < 32; ++column) {
                const float value = values[column];
                panel[global_row * TILE + column] = value;
                residual[global_row * TILE + column] = value - round_tf32(value);
                matrix[(size_t)global_row * N + k0 + column] = value;
            }
        }
        __syncthreads();
        if (thread == 0) {
            const unsigned long long now = global_nanoseconds();
            panel_ns += now - phase_start;
            phase_start = now;
        }

        const int remaining_tiles = NT - kt - 1;
        const int pair_count = remaining_tiles * (remaining_tiles + 1) / 2;
        const int quarter_count = pair_count * 4;
        for (int task = warp; task < quarter_count; task += WARPS) {
            const int pair = task >> 2;
            const int quarter = task & 3;
            const int local_bi =
                (int)((sqrtf(8.0f * pair + 1.0f) - 1.0f) * 0.5f);
            const int local_bj = pair - local_bi * (local_bi + 1) / 2;
            const int bi = kt + 1 + local_bi;
            const int bj = kt + 1 + local_bj;
            const int warp_row = quarter >> 1;
            const int warp_column = quarter & 1;
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
        __syncthreads();
        if (thread == 0) {
            const unsigned long long now = global_nanoseconds();
            trailing_ns += now - phase_start;
        }
    }

    if (thread == 0) phase_start = global_nanoseconds();
    for (int linear = thread; linear < N * N; linear += blockDim.x) {
        const int row = linear / N;
        const int column = linear - row * N;
        if (column > row) matrix[linear] = 0.0f;
    }
    __syncthreads();
    if (thread == 0) {
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
                "chol512 requires CUDA float32");
    TORCH_CHECK(matrix.is_contiguous() && matrix.dim() == 3 &&
                matrix.size(0) == MATRICES && matrix.size(1) == N &&
                matrix.size(2) == N, "chol512 requires 16x512x512 contiguous");
    TORCH_CHECK(phase.is_cuda() && phase.scalar_type() == torch::kInt64 &&
                phase.is_contiguous() && phase.numel() == MATRICES * 6,
                "phase must be CUDA int64[16,6]");
    static bool configured = false;
    if (!configured) {
        cudaError_t status = cudaFuncSetAttribute(
            persistent_cta_cholesky_512,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES);
        TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
        configured = true;
    }
    float* ptr = matrix.data_ptr<float>();
    unsigned long long* phase_ptr =
        reinterpret_cast<unsigned long long*>(phase.data_ptr<int64_t>());
    persistent_cta_cholesky_512<<<MATRICES, THREADS, SHARED_BYTES>>>(ptr, phase_ptr);
    cudaError_t status = cudaPeekAtLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""


if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CTA512 = load_inline(
            name="chol512_exp049_v2_persistent_cta_tf32x3",
            cpp_sources="void chol512_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CTA512_SOURCE,
            functions=["chol512_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CTA512_ERROR = repr(exc)


def _capture_phase_once() -> None:
    global _CTA512_PROFILE_PENDING
    global _CTA512_PHASE_LOAD_NS_HITS, _CTA512_PHASE_DIAG_NS_HITS
    global _CTA512_PHASE_PANEL_NS_HITS, _CTA512_PHASE_TRAILING_NS_HITS
    global _CTA512_PHASE_STORE_NS_HITS, _CTA512_PHASE_TOTAL_NS_HITS
    if not _CTA512_PROFILE_PENDING:
        return
    torch.cuda.synchronize()
    rows = _CTA512_PHASE.cpu().tolist()
    medians = [int(statistics.median(row[index] for row in rows)) for index in range(6)]
    (
        _CTA512_PHASE_LOAD_NS_HITS,
        _CTA512_PHASE_DIAG_NS_HITS,
        _CTA512_PHASE_PANEL_NS_HITS,
        _CTA512_PHASE_TRAILING_NS_HITS,
        _CTA512_PHASE_STORE_NS_HITS,
        _CTA512_PHASE_TOTAL_NS_HITS,
    ) = medians
    _CTA512_PROFILE_PENDING = False


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    global _CTA512_HITS, _CTA512_READY_HITS, _CTA512_FALLBACKS, _CTA512_PHASE
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and tuple(data.shape) == (16, 512, 512)
        and data.is_contiguous()
    )
    if target and _CTA512 is not None:
        if _CTA512_PHASE is None:
            _CTA512_PHASE = torch.empty((16, 6), dtype=torch.int64, device=data.device)
        output = data.clone()
        _CTA512.chol512_launch(output, _CTA512_PHASE)
        _CTA512_HITS += 1
        _CTA512_READY_HITS += 1
        _capture_phase_once()
        return output
    if target:
        _CTA512_FALLBACKS += 1
    return _ranked.custom_kernel(data)

