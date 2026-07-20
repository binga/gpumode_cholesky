"""Experiment 041 V2: rank-4 register-row CUDA Cholesky for 1024x64."""

import torch
import submission as _ranked


_CUDA64_HITS = 0
_CUDA64_ERROR = None
_CUDA64 = None

_CUDA64_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int N = 64;

__global__ void cholesky64_rank4(const float* input, float* output) {
    const int lane = threadIdx.x;
    const int row0 = lane;
    const int row1 = lane + 32;
    const size_t base = (size_t)blockIdx.x * N * N;
    __shared__ float tile[64][65];
    __shared__ float pivot0[64];
    __shared__ float pivot1[64];
    __shared__ float pivot2[64];
    __shared__ float pivot3[64];

    for (int linear = lane; linear < N * N; linear += 32) {
        tile[linear >> 6][linear & 63] = input[base + linear];
    }
    __syncwarp();

    float values0[64];
    float values1[64];
    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        values0[column] = tile[row0][column];
        values1[column] = tile[row1][column];
    }

    #pragma unroll
    for (int iteration = 0; iteration < 16; ++iteration) {
        const int p0 = 4 * iteration;
        const int p1 = p0 + 1;
        const int p2 = p0 + 2;
        const int p3 = p0 + 3;
        const int owner0 = p0 & 31;
        const int owner1 = p1 & 31;
        const int owner2 = p2 & 31;
        const int owner3 = p3 & 31;

        float m00 = lane == owner0
            ? (p0 < 32 ? values0[p0] : values1[p0]) : 0.0f;
        float m10 = lane == owner1
            ? (p1 < 32 ? values0[p0] : values1[p0]) : 0.0f;
        float m20 = lane == owner2
            ? (p2 < 32 ? values0[p0] : values1[p0]) : 0.0f;
        float m30 = lane == owner3
            ? (p3 < 32 ? values0[p0] : values1[p0]) : 0.0f;
        float m11 = lane == owner1
            ? (p1 < 32 ? values0[p1] : values1[p1]) : 0.0f;
        float m21 = lane == owner2
            ? (p2 < 32 ? values0[p1] : values1[p1]) : 0.0f;
        float m31 = lane == owner3
            ? (p3 < 32 ? values0[p1] : values1[p1]) : 0.0f;
        float m22 = lane == owner2
            ? (p2 < 32 ? values0[p2] : values1[p2]) : 0.0f;
        float m32 = lane == owner3
            ? (p3 < 32 ? values0[p2] : values1[p2]) : 0.0f;
        float m33 = lane == owner3
            ? (p3 < 32 ? values0[p3] : values1[p3]) : 0.0f;
        m00 = __shfl_sync(0xffffffffu, m00, owner0);
        m10 = __shfl_sync(0xffffffffu, m10, owner1);
        m20 = __shfl_sync(0xffffffffu, m20, owner2);
        m30 = __shfl_sync(0xffffffffu, m30, owner3);
        m11 = __shfl_sync(0xffffffffu, m11, owner1);
        m21 = __shfl_sync(0xffffffffu, m21, owner2);
        m31 = __shfl_sync(0xffffffffu, m31, owner3);
        m22 = __shfl_sync(0xffffffffu, m22, owner2);
        m32 = __shfl_sync(0xffffffffu, m32, owner3);
        m33 = __shfl_sync(0xffffffffu, m33, owner3);

        const float inverse0 = rsqrtf(m00);
        const float l10 = m10 * inverse0;
        const float l20 = m20 * inverse0;
        const float l30 = m30 * inverse0;
        const float d1 = m11 - l10 * l10;
        const float inverse1 = rsqrtf(d1);
        const float l21 = (m21 - l20 * l10) * inverse1;
        const float l31 = (m31 - l30 * l10) * inverse1;
        const float d2 = m22 - l20 * l20 - l21 * l21;
        const float inverse2 = rsqrtf(d2);
        const float l32 = (m32 - l30 * l20 - l31 * l21) * inverse2;
        const float d3 = m33 - l30 * l30 - l31 * l31 - l32 * l32;
        const float inverse3 = rsqrtf(d3);

        float a00 = row0 >= p0 ? values0[p0] * inverse0 : 0.0f;
        float a01 = row0 >= p1
            ? (values0[p1] - l10 * a00) * inverse1 : 0.0f;
        float a02 = row0 >= p2
            ? (values0[p2] - l20 * a00 - l21 * a01) * inverse2 : 0.0f;
        float a03 = row0 >= p3
            ? (values0[p3] - l30 * a00 - l31 * a01 - l32 * a02)
                * inverse3 : 0.0f;
        float a10 = row1 >= p0 ? values1[p0] * inverse0 : 0.0f;
        float a11 = row1 >= p1
            ? (values1[p1] - l10 * a10) * inverse1 : 0.0f;
        float a12 = row1 >= p2
            ? (values1[p2] - l20 * a10 - l21 * a11) * inverse2 : 0.0f;
        float a13 = row1 >= p3
            ? (values1[p3] - l30 * a10 - l31 * a11 - l32 * a12)
                * inverse3 : 0.0f;
        if (row0 >= p0) values0[p0] = a00;
        if (row0 >= p1) values0[p1] = a01;
        if (row0 >= p2) values0[p2] = a02;
        if (row0 >= p3) values0[p3] = a03;
        if (row1 >= p0) values1[p0] = a10;
        if (row1 >= p1) values1[p1] = a11;
        if (row1 >= p2) values1[p2] = a12;
        if (row1 >= p3) values1[p3] = a13;
        pivot0[row0] = a00;
        pivot1[row0] = a01;
        pivot2[row0] = a02;
        pivot3[row0] = a03;
        pivot0[row1] = a10;
        pivot1[row1] = a11;
        pivot2[row1] = a12;
        pivot3[row1] = a13;
        __syncwarp();

        if (row0 > p3) {
            #pragma unroll
            for (int column = 0; column < 64; ++column) {
                if (column > p3 && column <= row0) {
                    float value = fmaf(-a00, pivot0[column], values0[column]);
                    value = fmaf(-a01, pivot1[column], value);
                    value = fmaf(-a02, pivot2[column], value);
                    values0[column] = fmaf(-a03, pivot3[column], value);
                }
            }
        }
        if (row1 > p3) {
            #pragma unroll
            for (int column = 0; column < 64; ++column) {
                if (column > p3 && column <= row1) {
                    float value = fmaf(-a10, pivot0[column], values1[column]);
                    value = fmaf(-a11, pivot1[column], value);
                    value = fmaf(-a12, pivot2[column], value);
                    values1[column] = fmaf(-a13, pivot3[column], value);
                }
            }
        }
        __syncwarp();
    }

    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        tile[row0][column] = column <= row0 ? values0[column] : 0.0f;
        tile[row1][column] = column <= row1 ? values1[column] : 0.0f;
    }
    __syncwarp();
    for (int linear = lane; linear < N * N; linear += 32) {
        output[base + linear] = tile[linear >> 6][linear & 63];
    }
}

void chol64_launch(torch::Tensor input, torch::Tensor output) {
    const int batch = (int)input.size(0);
    cholesky64_rank4<<<dim3(batch), dim3(32)>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA64 = load_inline(
            name="chol64_exp041_v2",
            cpp_sources="void chol64_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA64_SOURCE,
            functions=["chol64_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA64_ERROR = repr(exc)


def custom_kernel(data):
    global _CUDA64_HITS
    if (
        _CUDA64 is not None
        and data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (1024, 64, 64)
        and data.is_contiguous()
    ):
        output = torch.empty_like(data)
        _CUDA64.chol64_launch(data, output)
        _CUDA64_HITS += 1
        return output
    return _ranked.custom_kernel(data)
