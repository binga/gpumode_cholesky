"""Experiment 042 V5 device-clock blocked-phase instrumentation."""

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

__device__ __forceinline__ unsigned long long global_nanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void cholesky128_block16(
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
        const int row = linear >> 7;
        const int column = linear & 127;
        tile[row * TILE_STRIDE + column] = input[base + linear];
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
        if (tid == 0) {
            const unsigned long long now = global_nanoseconds();
            diagonal_ns += now - mark;
            mark = now;
        }
        __syncthreads();

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
        if (tid == 0) {
            const unsigned long long now = global_nanoseconds();
            panel_ns += now - mark;
            mark = now;
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
        if (tid == 0) {
            const unsigned long long now = global_nanoseconds();
            trailing_ns += now - mark;
            mark = now;
        }
        __syncthreads();
    }

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 7;
        const int column = linear & 127;
        output[base + linear] =
            column <= row ? tile[row * TILE_STRIDE + column] : 0.0f;
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

void chol128_launch(
        torch::Tensor input,
        torch::Tensor output,
        torch::Tensor timings) {
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

        _CUDA128 = load_inline(
            name="chol128_exp042_v5_phase",
            cpp_sources=(
                "void chol128_launch(torch::Tensor, torch::Tensor, torch::Tensor);"
            ),
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
        timings = torch.empty((256, 7), dtype=torch.int64, device=data.device)
        _CUDA128.chol128_launch(data, out, timings)
        _CUDA128_HITS += 1
        return out
    return _ranked.custom_kernel(data)


def phase_probe(data):
    out = torch.empty_like(data)
    timings = torch.empty((256, 7), dtype=torch.int64, device=data.device)
    _CUDA128.chol128_launch(data, out, timings)
    return out, timings
