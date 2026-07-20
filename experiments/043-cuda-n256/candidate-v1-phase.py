"""Experiment 043 V1 device-clock packed-blocked phase instrumentation."""

import torch
import submission as _ranked


_CUDA256_HITS = 0
_CUDA256_ERROR = None
_CUDA256 = None

_CUDA256_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int N = 256;
constexpr int BK = 16;
constexpr int THREADS = 512;
constexpr int TRI_VALUES = N * (N + 1) / 2;
constexpr int SHARED_BYTES = TRI_VALUES * sizeof(float);

__device__ __forceinline__ int lower_index(int row, int column) {
    return ((row * (row + 1)) >> 1) + column;
}

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ __launch_bounds__(THREADS, 1)
void cholesky256_profile(
        const float* input,
        float* output,
        unsigned long long* timings) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t base = (size_t)blockIdx.x * N * N;
    extern __shared__ float tile[];
    __shared__ float reciprocal;
    __shared__ float inverse_diag[BK];
    unsigned long long launch_start = 0;
    unsigned long long mark = 0;
    unsigned long long staging_ns = 0;
    unsigned long long diagonal_ns = 0;
    unsigned long long panel_ns = 0;
    unsigned long long trailing_ns = 0;
    if (tid == 0) launch_start = global_nanoseconds();
    __syncthreads();

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 8;
        const int column = linear & 255;
        if (column <= row) tile[lower_index(row, column)] = input[base + linear];
    }
    __syncthreads();
    if (tid == 0) {
        mark = global_nanoseconds();
        staging_ns = mark - launch_start;
    }
    __syncthreads();

    #pragma unroll 1
    for (int block = 0; block < N; block += BK) {
        const int block_end = block + BK;
        #pragma unroll
        for (int local = 0; local < BK; ++local) {
            const int pivot = block + local;
            if (tid == 0) {
                const int diagonal = lower_index(pivot, pivot);
                reciprocal = rsqrtf(tile[diagonal]);
                inverse_diag[local] = reciprocal;
                tile[diagonal] *= reciprocal;
            }
            __syncthreads();
            const int panel_row = pivot + 1 + tid;
            if (panel_row < block_end) tile[lower_index(panel_row, pivot)] *= reciprocal;
            __syncthreads();
            for (int linear = tid; linear < BK * BK; linear += THREADS) {
                const int row = block + (linear >> 4);
                const int column = block + (linear & 15);
                if (row > pivot && column > pivot && column <= row) {
                    const int offset = lower_index(row, column);
                    tile[offset] = fmaf(
                        -tile[lower_index(row, pivot)],
                        tile[lower_index(column, pivot)],
                        tile[offset]);
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const unsigned long long now = global_nanoseconds();
            diagonal_ns += now - mark;
            mark = now;
        }
        __syncthreads();

        const int row = block_end + tid;
        if (row < N) {
            const int row_base = (row * (row + 1)) >> 1;
            #pragma unroll
            for (int local = 0; local < BK; ++local) {
                const int column = block + local;
                float value = tile[row_base + column];
                const int column_base = (column * (column + 1)) >> 1;
                #pragma unroll
                for (int prior = 0; prior < local; ++prior) {
                    value = fmaf(
                        -tile[row_base + block + prior],
                        tile[column_base + block + prior],
                        value);
                }
                tile[row_base + column] = value * inverse_diag[local];
            }
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned long long now = global_nanoseconds();
            panel_ns += now - mark;
            mark = now;
        }
        __syncthreads();

        for (int trailing_row = block_end + warp;
             trailing_row < N;
             trailing_row += THREADS / 32) {
            const int row_base = (trailing_row * (trailing_row + 1)) >> 1;
            for (int column = block_end + lane;
                 column <= trailing_row;
                 column += 32) {
                const int column_base = (column * (column + 1)) >> 1;
                float update = 0.0f;
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    update = fmaf(
                        tile[row_base + block + k],
                        tile[column_base + block + k],
                        update);
                }
                tile[row_base + column] -= update;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned long long now = global_nanoseconds();
            trailing_ns += now - mark;
            mark = now;
        }
        __syncthreads();
    }

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 8;
        const int column = linear & 255;
        output[base + linear] =
            column <= row ? tile[lower_index(row, column)] : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned long long launch_end = global_nanoseconds();
        timings[(size_t)blockIdx.x * 7 + 0] = launch_start;
        timings[(size_t)blockIdx.x * 7 + 1] = launch_end;
        timings[(size_t)blockIdx.x * 7 + 2] = staging_ns;
        timings[(size_t)blockIdx.x * 7 + 3] = diagonal_ns;
        timings[(size_t)blockIdx.x * 7 + 4] = panel_ns;
        timings[(size_t)blockIdx.x * 7 + 5] = trailing_ns;
        timings[(size_t)blockIdx.x * 7 + 6] = launch_end - mark;
    }
}

void chol256_profile_launch(
        torch::Tensor input,
        torch::Tensor output,
        torch::Tensor timings) {
    static bool configured = false;
    if (!configured) {
        cudaError_t attr = cudaFuncSetAttribute(
            cholesky256_profile,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHARED_BYTES);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    cholesky256_profile<<<dim3((int)input.size(0)), dim3(THREADS), SHARED_BYTES>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reinterpret_cast<unsigned long long*>(timings.data_ptr<int64_t>()));
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA256 = load_inline(
            name="chol256_exp043_v1_phase",
            cpp_sources=(
                "void chol256_profile_launch("
                "torch::Tensor, torch::Tensor, torch::Tensor);"
            ),
            cuda_sources=_CUDA256_SOURCE,
            functions=["chol256_profile_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA256_ERROR = repr(exc)


def _run(data):
    out = torch.empty_like(data)
    timings = torch.empty((64, 7), dtype=torch.int64, device=data.device)
    _CUDA256.chol256_profile_launch(data, out, timings)
    return out, timings


def custom_kernel(data):
    global _CUDA256_HITS
    if (
        _CUDA256 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.is_contiguous()
        and data.shape == (64, 256, 256)
    ):
        out, _ = _run(data)
        _CUDA256_HITS += 1
        return out
    return _ranked.custom_kernel(data)


def phase_probe(data):
    return _run(data)
