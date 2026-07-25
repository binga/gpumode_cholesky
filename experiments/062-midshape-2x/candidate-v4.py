#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission — experiments 016a+016b+017 integrated.

On top of the exact exp-015 ranked winner (#881981): (016b) rank-2 one-warp
n=32 kernel (1.591x); (017) rank-4 pivot micro in the split32 pipeline plus
first-touch eager mode for 640x512/60x1024 (no copy-in/clone-out) and
mirror-zero panel stores replacing the clear pass (paired 1.05-1.26x on the
six split32 shapes); (016a) large single-matrix left-looking paths: 8192 off
pure cuSOLVER onto TF32 (1.138x) and recursive GEMM triangular inversion at
16384/32768 (1.055x/1.028x). Rejected with evidence this round:
2x2048/2x4096 split32 (0.76-0.78x), FP8-shadow fixed-scale stack (<=1.0x),
TILE=256 trailing.

Prior module docstring — experiment 015 final candidate.

Integrates two measured frontiers on top of the exact exp-014 ranked winner
(#880770): (1) a two-level blocked tensor-core factorization (rank-2 1-warp
diagonal potrf+inverse micro kernel, tf32x3 panel dots, tf32/tf32x3 rank-128
trailing Schur tiles, per-shape CUDA-graph replay) for 64x256, 16x512,
640x512, 4x1024, 60x1024, 8x2048 — paired 1.31x/1.15x/1.69x/1.40x/1.94x/1.59x;
(2) a graph-replayed exact cuSOLVER factorization for 1024x64 (1.08x).
Rejected on measurement: fused one-CTA whole-matrix potrf (r1), rank-32
single-level trailing (r3), TILE=256 trailing (r6 compile budget), 2x2048
(0.65x), 1x4096/2x4096 superpanels (0.18-0.97x, candidate B).

Two-level blocked tensor-core factorization for seven mid shapes: a
Gauss-Jordan-fused 1-warp diagonal potrf+inverse micro kernel (BK=32), panel
and narrow in-panel updates per micro step, one rank-128 trailing Schur
update per outer panel, all launches replayed as a per-shape CUDA graph.
Built on the exact exp-014 ranked winner (#880770); everything below this
paragraph is the unchanged exp-014 module documentation.

Prior module docstring — experiment 012 ranked winner.

Builds on exp 006 (`#878015`) by fusing its TF32 trailing Schur product and
subtraction into an in-place `addmm_` on the trailing view. This removes the full
temporary product and subtraction launch while preserving identical TF32/FP32
numerics. Ranked `#878108`: 17/17, public geomean 1542.914 us (secret 1545.128
us), improving the prior ~1559 us. Experiment 009 adds three exact-shape paths
that were independently measured on the same B200 as their shipped control.
Ranked `#878273`: public 1500.704 us, secret 1501.440 us.
Experiment 012 replaces only the 1x16384 and 1x32768 paths with left-looking
frontiers. Ranked `#878893`: public 1459.321 us, secret 1448.377 us.

Shape dispatcher:
  * n == 32                         -> custom CUDA rank-2 warp kernel, one warp
    per matrix (experiment 039, 2.28x paired at 4096x32). Rows remain in
    registers; a shared pivot-column exchange replaces Triton's full-tile
    predication. Falls back to the shipped Triton kernel if compilation fails.
  * batch == 1024 and n == 64       -> custom CUDA two-warp rank-2 kernel, one
    register row per thread (experiment 041 V3, 1.65x beyond the first 2.27x
    winner). Padded shared staging coalesces the one-launch input/output path.
  * batch == 256 and n == 128       -> custom CUDA blocked-16 factorization,
    one eight-warp CTA per matrix (experiment 042 V5, 2.03x paired). Diagonal
    blocks, register panel solves, and rank-16 trailing dots stay in one launch.
  * batch == 16 and n == 512        -> static-buffer captured vendor batched
    factorization (1.291x paired speedup, exact numerics). The buffer refresh
    remains fast when the official harness rotates among input allocations.
  * batch == 8 and n == 2048        -> Triton blocked factorization with FP32
    diagonal/panel work and grouped lower TF32 Schur updates (1.619x paired).
  * batch == 1 and n == 16384       -> left-looking TF32 factorization that
    updates only the active diagonal and panel (1.166x paired frontier).
  * batch == 1 and n == 32768       -> left-looking factorization with native
    Blackwell FP8 panel products and FP32 accumulation (1.386x paired frontier).
  * other batch == 1 and n >= 16384 -> blocked right-looking Cholesky with a
    fused in-place TF32 tensor-core trailing update (experiment 008).
    8192 (only ~1.07x in exp 006) stays on cuSOLVER.
  * 2 <= batch <= 4 and n >= 1024   -> per-matrix factorization in a sequential
    loop (experiment 004, region trimmed by exp 005). `torch.linalg` routes
    batch>=2 to `cusolverDnSpotrfBatched`, which is tuned for many-small matrices
    and is ~1.2-4x too slow for few-large ones; factorizing each matrix on its own
    with the fast single-matrix blocked `potrf` is much faster. batch>=8 (e.g.
    8×2048) stays on batched cuSOLVER (faster on popcorn).
  * everything else                 -> batched cuSOLVER via cholesky_ex (best for
    batch=1 mid-n and high-batch small/mid-n, incl. the saturated 640×512).
"""

import torch

from task import input_t, output_t

# ---------------------------------------------------------------------------
# Experiment 039: cuSOLVER-free CUDA rank-2 Cholesky for n == 32.
#
# One warp owns one matrix and one lane owns one row. Rows stay in registers;
# only the two current pivot columns cross lanes through padded shared memory.
# Pairing pivots fuses two trailing rank-1 updates. The launch uses CUDA's
# default execution queue and introduces no auxiliary/concurrent queue API.
# ---------------------------------------------------------------------------
_CUDA32_HITS = 0
_CUDA32_ERROR = None
_CUDA32 = None

_CUDA32_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void chol32_register_rank2(const float* __restrict__ src,
                                      float* __restrict__ dst) {
    const int lane = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * 1024;
    __shared__ float staging[32][33];
    __shared__ float pivot0[32];
    __shared__ float pivot1[32];
    float row_values[32];

    #pragma unroll
    for (int item = 0; item < 32; ++item) {
        const int linear = item * 32 + lane;
        staging[linear >> 5][linear & 31] = src[base + linear];
    }
    __syncwarp();
    #pragma unroll
    for (int column = 0; column < 32; ++column) {
        row_values[column] = staging[lane][column];
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
        staging[lane][column] = column <= lane ? row_values[column] : 0.0f;
    }
    __syncwarp();
    #pragma unroll
    for (int item = 0; item < 32; ++item) {
        const int linear = item * 32 + lane;
        dst[base + linear] = staging[linear >> 5][linear & 31];
    }
}

void chol32_launch(torch::Tensor src, torch::Tensor dst) {
    chol32_register_rank2<<<(int)src.size(0), 32>>>(
        src.data_ptr<float>(), dst.data_ptr<float>());
}
"""

# Loaded together with CUDA64 and CUDA128 below to remove two fixed compiler
# startup costs. The CUDA32 kernel source and -O3 code generation are unchanged.


def _cuda_cholesky32(data: torch.Tensor) -> torch.Tensor:
    global _CUDA32_HITS
    out = torch.empty_like(data)
    _CUDA32.chol32_launch(data, out)
    _CUDA32_HITS += 1
    return out


# ---------------------------------------------------------------------------
# Experiment 041: cuSOLVER-free CUDA rank-2 Cholesky for 1024x64.
#
# Two warps own one matrix and every thread owns one register-resident row.
# A four-rendezvous rank-2 handoff exposes twice the row parallelism while
# padded shared staging coalesces input/output. The kernel writes the required
# representation in one launch and replaces the prior 17-operation graph.
# ---------------------------------------------------------------------------
_CUDA64_HITS = 0
_CUDA64_ERROR = None
_CUDA64 = None

_CUDA64_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int N64 = 64;

__global__ void cholesky64_rank2(const float* input, float* output) {
    const int row = threadIdx.x;
    const size_t base = (size_t)blockIdx.x * N64 * N64;
    __shared__ float tile[64][65];
    __shared__ float pivot0[64];
    __shared__ float pivot1[64];
    __shared__ float reciprocal0;
    __shared__ float reciprocal1;

    for (int linear = row; linear < N64 * N64; linear += 64) {
        tile[linear >> 6][linear & 63] = input[base + linear];
    }
    __syncthreads();

    float values[64];
    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        values[column] = tile[row][column];
    }

    #pragma unroll
    for (int iteration = 0; iteration < 32; ++iteration) {
        const int k = 2 * iteration;
        const int q = k + 1;
        if (row == k) reciprocal0 = rsqrtf(values[k]);
        __syncthreads();
        if (row >= k) values[k] *= reciprocal0;
        pivot0[row] = values[k];
        if (row == q) {
            values[q] = fmaf(-values[k], values[k], values[q]);
            reciprocal1 = rsqrtf(values[q]);
        }
        __syncthreads();

        if (row >= q) {
            if (row != q) {
                values[q] = fmaf(-values[k], pivot0[q], values[q]);
            }
            values[q] *= reciprocal1;
        }
        pivot1[row] = values[q];
        __syncthreads();

        if (row > q) {
            const float scale0 = values[k];
            const float scale1 = values[q];
            #pragma unroll
            for (int column = 0; column < 64; ++column) {
                if (column > q && column <= row) {
                    float value = fmaf(
                        -scale0, pivot0[column], values[column]);
                    values[column] = fmaf(
                        -scale1, pivot1[column], value);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int column = 0; column < 64; ++column) {
        tile[row][column] = column <= row ? values[column] : 0.0f;
    }
    __syncthreads();
    for (int linear = row; linear < N64 * N64; linear += 64) {
        output[base + linear] = tile[linear >> 6][linear & 63];
    }
}

void chol64_launch(torch::Tensor input, torch::Tensor output) {
    const int batch = (int)input.size(0);
    cholesky64_rank2<<<dim3(batch), dim3(64)>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

# Loaded together with CUDA32 and CUDA128 below. N64 is a source-only rename
# that resolves the combined translation unit's constant-name collision.


def _cuda_cholesky64(data: torch.Tensor) -> torch.Tensor:
    global _CUDA64_HITS
    out = torch.empty_like(data)
    _CUDA64.chol64_launch(data, out)
    _CUDA64_HITS += 1
    return out


# ---------------------------------------------------------------------------
# Experiment 042: cuSOLVER-free blocked-16 CUDA Cholesky for 256x128.
#
# One eight-warp CTA owns each matrix in a padded shared tile. Sixteen-wide
# diagonal blocks expose independent row solves, and each coarse trailing
# update computes 16-term FP32 dots. This replaces the prior 18-operation
# split32 graph, including its copies and host-visible finiteness gate.
# ---------------------------------------------------------------------------
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

__global__ void cholesky128_block16(const float* input, float* output) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t base = (size_t)blockIdx.x * N * N;
    extern __shared__ float tile[];
    __shared__ float reciprocal;
    __shared__ float inverse_diag[BK];

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 7;
        const int column = linear & 127;
        tile[row * TILE_STRIDE + column] = input[base + linear];
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
    }

    for (int linear = tid; linear < N * N; linear += THREADS) {
        const int row = linear >> 7;
        const int column = linear & 127;
        output[base + linear] =
            column <= row ? tile[row * TILE_STRIDE + column] : 0.0f;
    }
}

// --- Experiment 044 diagonal micro (compiled into this module so the
// submission keeps three nvcc invocations; a fourth extension pushed the
// official runner's six-minute compile budget over the limit).
constexpr int MICRO_BK = 32;
constexpr int MICRO_WARPS = 4;
constexpr int MICRO_THREADS = MICRO_WARPS * 32;
constexpr unsigned MICRO_FULL = 0xffffffffu;

// Rank-4 warp-synchronous 32x32 diagonal factorization with coalesced
// shared staging. Chosen from a six-variant probe: 10.26us/launch against
// 11.25us rank-1, 11.27us rank-2, 12.31us uncoalesced rank-1 and Triton
// `_micro_potrf_gj32`'s 13.56us, on a 3.5us launch floor.
__global__ __launch_bounds__(MICRO_THREADS)
void micro_potrf32_rank4(const float* __restrict__ src, float* __restrict__ work,
                 float* __restrict__ inv, int batch, int n, int k, int first) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int matrix = blockIdx.x * MICRO_WARPS + warp;
    if (matrix >= batch) return;

    __shared__ float staging[MICRO_WARPS][MICRO_BK][MICRO_BK + 1];
    __shared__ float pivot_s[MICRO_WARPS][4][MICRO_BK];
    float (*tile)[MICRO_BK + 1] = staging[warp];
    float (*pivots)[MICRO_BK] = pivot_s[warp];

    const size_t block_base = (size_t)matrix * n * n + (size_t)k * n + k;
    const float* input = (first ? src : work) + block_base;
    float* output = work + block_base;

    #pragma unroll
    for (int row = 0; row < MICRO_BK; ++row) {
        tile[row][lane] = input[(size_t)row * n + lane];
    }
    __syncwarp();

    float values[MICRO_BK];
    #pragma unroll
    for (int c = 0; c < MICRO_BK; ++c) values[c] = tile[lane][c];

    float reciprocal_row = 0.0f;
    #pragma unroll
    for (int iteration = 0; iteration < MICRO_BK / 4; ++iteration) {
        const int p0 = 4 * iteration;
        const int p1 = p0 + 1;
        const int p2 = p0 + 2;
        const int p3 = p0 + 3;

        const float rec0 = rsqrtf(__shfl_sync(MICRO_FULL, values[p0], p0));
        const float s0 = (lane >= p0) ? values[p0] * rec0 : 0.0f;
        values[p0] = s0;
        pivots[0][lane] = s0;
        const float c01 = __shfl_sync(MICRO_FULL, s0, p1);
        const float c02 = __shfl_sync(MICRO_FULL, s0, p2);
        const float c03 = __shfl_sync(MICRO_FULL, s0, p3);
        values[p1] = (lane >= p1) ? fmaf(-s0, c01, values[p1]) : values[p1];
        values[p2] = (lane >= p2) ? fmaf(-s0, c02, values[p2]) : values[p2];
        values[p3] = (lane >= p3) ? fmaf(-s0, c03, values[p3]) : values[p3];

        const float rec1 = rsqrtf(__shfl_sync(MICRO_FULL, values[p1], p1));
        const float s1 = (lane >= p1) ? values[p1] * rec1 : 0.0f;
        values[p1] = s1;
        pivots[1][lane] = s1;
        const float c12 = __shfl_sync(MICRO_FULL, s1, p2);
        const float c13 = __shfl_sync(MICRO_FULL, s1, p3);
        values[p2] = (lane >= p2) ? fmaf(-s1, c12, values[p2]) : values[p2];
        values[p3] = (lane >= p3) ? fmaf(-s1, c13, values[p3]) : values[p3];

        const float rec2 = rsqrtf(__shfl_sync(MICRO_FULL, values[p2], p2));
        const float s2 = (lane >= p2) ? values[p2] * rec2 : 0.0f;
        values[p2] = s2;
        pivots[2][lane] = s2;
        const float c23 = __shfl_sync(MICRO_FULL, s2, p3);
        values[p3] = (lane >= p3) ? fmaf(-s2, c23, values[p3]) : values[p3];

        const float rec3 = rsqrtf(__shfl_sync(MICRO_FULL, values[p3], p3));
        const float s3 = (lane >= p3) ? values[p3] * rec3 : 0.0f;
        values[p3] = s3;
        pivots[3][lane] = s3;

        if (lane == p0) reciprocal_row = rec0;
        if (lane == p1) reciprocal_row = rec1;
        if (lane == p2) reciprocal_row = rec2;
        if (lane == p3) reciprocal_row = rec3;
        __syncwarp();

        #pragma unroll
        for (int c = 0; c < MICRO_BK; ++c) {
            if (c > p3) {
                float value = values[c];
                if (c <= lane) {
                    value = fmaf(-s0, pivots[0][c], value);
                    value = fmaf(-s1, pivots[1][c], value);
                    value = fmaf(-s2, pivots[2][c], value);
                    value = fmaf(-s3, pivots[3][c], value);
                }
                values[c] = value;
            }
        }
        __syncwarp();
    }

    #pragma unroll
    for (int c = 0; c < MICRO_BK; ++c) {
        tile[lane][c] = (c <= lane) ? values[c] : 0.0f;
    }
    pivots[0][lane] = reciprocal_row;
    __syncwarp();
    #pragma unroll
    for (int row = 0; row < MICRO_BK; ++row) {
        output[(size_t)row * n + lane] = tile[row][lane];
    }

    float inverse[MICRO_BK];
    #pragma unroll
    for (int r = 0; r < MICRO_BK; ++r) {
        float accumulator = (r == lane) ? 1.0f : 0.0f;
        #pragma unroll
        for (int p = 0; p < MICRO_BK; ++p) {
            if (p < r) accumulator = fmaf(-tile[r][p], inverse[p], accumulator);
        }
        inverse[r] = (r >= lane) ? accumulator * pivots[0][r] : 0.0f;
    }
    float* inverse_out = inv + (size_t)matrix * MICRO_BK * MICRO_BK + lane;
    #pragma unroll
    for (int r = 0; r < MICRO_BK; ++r) inverse_out[r * MICRO_BK] = inverse[r];
}

void micro32_launch(
    torch::Tensor src,
    torch::Tensor work,
    torch::Tensor inv,
    int64_t n,
    int64_t k,
    int64_t first) {
    const int batch = (int)work.size(0);
    const int blocks = (batch + MICRO_WARPS - 1) / MICRO_WARPS;
    micro_potrf32_rank4<<<dim3(blocks), dim3(MICRO_THREADS)>>>(
        src.data_ptr<float>(), work.data_ptr<float>(),
        inv.data_ptr<float>(), batch, (int)n, (int)k, (int)first);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
void chol128_launch(torch::Tensor input, torch::Tensor output) {
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
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA128 = load_inline(
            name="chol3264128_exp055_combined_o3",
            cpp_sources=(
                "void chol32_launch(torch::Tensor, torch::Tensor);\n"
                "void chol64_launch(torch::Tensor, torch::Tensor);\n"
                "void chol128_launch(torch::Tensor, torch::Tensor);\n"
                "void micro32_launch(torch::Tensor, torch::Tensor, "
                "torch::Tensor, int64_t, int64_t, int64_t);"
            ),
            cuda_sources=(
                _CUDA32_SOURCE + "\n" + _CUDA64_SOURCE + "\n" +
                _CUDA128_SOURCE
            ),
            functions=["chol32_launch", "chol64_launch", "chol128_launch",
                       "micro32_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        _CUDA32 = _CUDA128
        _CUDA64 = _CUDA128
    except Exception as exc:
        _CUDA128_ERROR = repr(exc)
        _CUDA32_ERROR = _CUDA128_ERROR
        _CUDA64_ERROR = _CUDA128_ERROR


def _cuda_cholesky128(data: torch.Tensor) -> torch.Tensor:
    global _CUDA128_HITS
    out = torch.empty_like(data)
    _CUDA128.chol128_launch(data, out)
    _CUDA128_HITS += 1
    return out


# ---------------------------------------------------------------------------
# Experiment 043: cuSOLVER-free packed-lower CUDA Cholesky for 64x256.
#
# One CTA owns each matrix, but the rank-16 trailing Schur tiles use warp-level
# TF32 tensor-core MMA instead of scalar shared-memory dot products. Lower
# 16x16 tiles are packed contiguously in shared memory (139,264 bytes), which
# is both WMMA-loadable and within the B200 per-block budget. Diagonal and panel
# arithmetic remain FP32.
# ---------------------------------------------------------------------------
_CUDA256_HITS = 0
_CUDA256_ERROR = None
_CUDA256 = None

_CUDA256_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

constexpr int N256 = 256;
constexpr int BK256 = 16;
constexpr int THREADS256 = 256;
constexpr int WARPS256 = THREADS256 / 32;
constexpr int TILE_STRIDE256 = 20;
constexpr int TILE_VALUES256 = BK256 * TILE_STRIDE256;
constexpr int TILES_PER_DIM256 = N256 / BK256;
constexpr int TRI_TILES256 = TILES_PER_DIM256 * (TILES_PER_DIM256 + 1) / 2;
constexpr int SHARED_VALUES256 =
    TRI_TILES256 * TILE_VALUES256;
constexpr int SHARED_BYTES256 = SHARED_VALUES256 * sizeof(float);

__device__ __forceinline__ int tile_index256(int row, int column) {
    const int tile_row = row >> 4;
    const int tile_column = column >> 4;
    const int tile = ((tile_row * (tile_row + 1)) >> 1) + tile_column;
    return tile * TILE_VALUES256 + (row & 15) * TILE_STRIDE256 + (column & 15);
}

__device__ __noinline__ void accurate_trailing256(
        float* tile,
        int block,
        int first_tile,
        int remaining_tiles) {
    const int tid = threadIdx.x;
    const int first_row = first_tile << 4;
    const int remaining = remaining_tiles << 4;
    const int pair_count = remaining * (remaining + 1) / 2;
    for (int pair = tid; pair < pair_count; pair += THREADS256) {
        const int relative_row = (int)(
            (sqrtf(8.0f * (float)pair + 1.0f) - 1.0f) * 0.5f);
        const int relative_column =
            pair - relative_row * (relative_row + 1) / 2;
        const int row = first_row + relative_row;
        const int column = first_row + relative_column;
        const int output_index = tile_index256(row, column);
        float value = tile[output_index];
        #pragma unroll 1
        for (int depth = 0; depth < BK256; ++depth) {
            value = fmaf(
                -tile[tile_index256(row, block + depth)],
                tile[tile_index256(column, block + depth)],
                value);
        }
        tile[output_index] = value;
    }
    __syncthreads();
}

__device__ __noinline__ void factor256_accurate(
        float* tile,
        float* reciprocal0,
        float* reciprocal1,
        float* inverse_diag,
        float* pivot0,
        float* pivot1) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll 1
    for (int block = 0; block < N256; block += BK256) {
        const int block_end = block + BK256;
        const int diagonal_row = lane;
        float diagonal_values[BK256];
        if (warp == 0) {
            #pragma unroll 1
            for (int column = 0; column < BK256; ++column) {
                diagonal_values[column] =
                    diagonal_row < BK256 && column <= diagonal_row
                    ? tile[tile_index256(
                          block + diagonal_row, block + column)]
                    : 0.0f;
            }

            #pragma unroll 1
            for (int iteration = 0; iteration < BK256 / 2; ++iteration) {
                const int k = 2 * iteration;
                const int q = k + 1;
                if (diagonal_row == k) {
                    *reciprocal0 = rsqrtf(diagonal_values[k]);
                    inverse_diag[k] = *reciprocal0;
                }
                __syncwarp();

                if (diagonal_row < BK256 && diagonal_row >= k) {
                    diagonal_values[k] *= *reciprocal0;
                    pivot0[diagonal_row] = diagonal_values[k];
                }
                if (diagonal_row == q) {
                    diagonal_values[q] = fmaf(
                        -diagonal_values[k], diagonal_values[k],
                        diagonal_values[q]);
                    *reciprocal1 = rsqrtf(diagonal_values[q]);
                    inverse_diag[q] = *reciprocal1;
                }
                __syncwarp();

                if (diagonal_row < BK256 && diagonal_row >= q) {
                    if (diagonal_row != q) {
                        diagonal_values[q] = fmaf(
                            -diagonal_values[k], pivot0[q],
                            diagonal_values[q]);
                    }
                    diagonal_values[q] *= *reciprocal1;
                    pivot1[diagonal_row] = diagonal_values[q];
                }
                __syncwarp();

                if (diagonal_row < BK256 && diagonal_row > q) {
                    const float scale0 = diagonal_values[k];
                    const float scale1 = diagonal_values[q];
                    #pragma unroll 1
                    for (int column = q + 1; column < BK256; ++column) {
                        if (column <= diagonal_row) {
                            float value = fmaf(
                                -scale0, pivot0[column],
                                diagonal_values[column]);
                            diagonal_values[column] = fmaf(
                                -scale1, pivot1[column], value);
                        }
                    }
                }
                __syncwarp();
            }

            if (diagonal_row < BK256) {
                #pragma unroll 1
                for (int column = 0; column <= diagonal_row; ++column) {
                    tile[tile_index256(
                        block + diagonal_row, block + column)] =
                        diagonal_values[column];
                }
            }
        }
        __syncthreads();

        const int row = block_end + tid;
        if (row < N256) {
            const int row_block_base = tile_index256(row, block);
            #pragma unroll 1
            for (int local = 0; local < BK256; ++local) {
                const int column = block + local;
                float value = tile[row_block_base + local];
                const int column_block_base = tile_index256(column, block);
                #pragma unroll 1
                for (int prior = 0; prior < local; ++prior) {
                    value = fmaf(
                        -tile[row_block_base + prior],
                        tile[column_block_base + prior], value);
                }
                tile[row_block_base + local] = value * inverse_diag[local];
            }
        }
        __syncthreads();

        const int first_tile = block_end >> 4;
        const int remaining_tiles = TILES_PER_DIM256 - first_tile;
        accurate_trailing256(tile, block, first_tile, remaining_tiles);
    }
}

__global__ __launch_bounds__(THREADS256, 1)
void cholesky256_wmma16(const float* input, float* output) {
    namespace wmma = nvcuda::wmma;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t base = (size_t)blockIdx.x * N256 * N256;
    extern __shared__ float tile[];
    __shared__ float reciprocal0;
    __shared__ float reciprocal1;
    __shared__ float inverse_diag[BK256];
    __shared__ float pivot0[BK256];
    __shared__ float pivot1[BK256];
    __shared__ int accurate_required;

    int staging_tile = 0;
    #pragma unroll 1
    for (int tile_row = 0; tile_row < TILES_PER_DIM256; ++tile_row) {
        for (int tile_column = 0; tile_column <= tile_row;
             ++tile_column, ++staging_tile) {
            if (tid < BK256 * BK256) {
                const int local_row = tid >> 4;
                const int local_column = tid & 15;
                const int row = (tile_row << 4) + local_row;
                const int column = (tile_column << 4) + local_column;
                tile[staging_tile * TILE_VALUES256
                     + local_row * TILE_STRIDE256 + local_column] =
                    (tile_row != tile_column || local_column <= local_row)
                    ? input[base + (size_t)row * N256 + column]
                    : 0.0f;
            }
        }
    }
    __syncthreads();

    float reference_diagonal = 0.0f;
    if (warp == 0) {
        for (int diagonal = lane; diagonal < N256; diagonal += 32) {
            reference_diagonal = fmaxf(
                reference_diagonal,
                tile[tile_index256(diagonal, diagonal)]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            reference_diagonal = fmaxf(
                reference_diagonal,
                __shfl_down_sync(0xffffffffu, reference_diagonal, offset));
        }
        reference_diagonal = __shfl_sync(
            0xffffffffu, reference_diagonal, 0);
        if (lane == 0) accurate_required = 0;
        __syncwarp();
    }

    {
        #pragma unroll 1
        for (int block = 0; block < N256; block += BK256) {
            const int block_end = block + BK256;
            const int diagonal_row = lane;
            float diagonal_values[BK256];
            if (warp == 0) {
                #pragma unroll
                for (int column = 0; column < BK256; ++column) {
                    diagonal_values[column] =
                        diagonal_row < BK256 && column <= diagonal_row
                        ? tile[tile_index256(
                              block + diagonal_row, block + column)]
                        : 0.0f;
                }

                #pragma unroll
                for (int iteration = 0; iteration < BK256 / 2; ++iteration) {
                    const int k = 2 * iteration;
                    const int q = k + 1;
                    if (diagonal_row == k) {
                        if (!(diagonal_values[k]
                                > reference_diagonal * 1.0e-2f)) {
                            accurate_required = 1;
                        }
                        reciprocal0 = rsqrtf(diagonal_values[k]);
                        inverse_diag[k] = reciprocal0;
                    }
                    __syncwarp();

                    if (diagonal_row < BK256 && diagonal_row >= k) {
                        diagonal_values[k] *= reciprocal0;
                        pivot0[diagonal_row] = diagonal_values[k];
                    }
                    if (diagonal_row == q) {
                        diagonal_values[q] = fmaf(
                            -diagonal_values[k], diagonal_values[k],
                            diagonal_values[q]);
                        if (!(diagonal_values[q]
                                > reference_diagonal * 1.0e-2f)) {
                            accurate_required = 1;
                        }
                        reciprocal1 = rsqrtf(diagonal_values[q]);
                        inverse_diag[q] = reciprocal1;
                    }
                    __syncwarp();

                    if (diagonal_row < BK256 && diagonal_row >= q) {
                        if (diagonal_row != q) {
                            diagonal_values[q] = fmaf(
                                -diagonal_values[k], pivot0[q],
                                diagonal_values[q]);
                        }
                        diagonal_values[q] *= reciprocal1;
                        pivot1[diagonal_row] = diagonal_values[q];
                    }
                    __syncwarp();

                    if (diagonal_row < BK256 && diagonal_row > q) {
                        const float scale0 = diagonal_values[k];
                        const float scale1 = diagonal_values[q];
                        #pragma unroll
                        for (int column = q + 1; column < BK256; ++column) {
                            if (column <= diagonal_row) {
                                float value = fmaf(
                                    -scale0, pivot0[column],
                                    diagonal_values[column]);
                                diagonal_values[column] = fmaf(
                                    -scale1, pivot1[column], value);
                            }
                        }
                    }
                    __syncwarp();
                }

                if (diagonal_row < BK256) {
                    #pragma unroll
                    for (int column = 0; column <= diagonal_row; ++column) {
                        tile[tile_index256(
                            block + diagonal_row, block + column)] =
                            diagonal_values[column];
                    }
                }
            }
            __syncthreads();

            const int row = block_end + tid;
            if (row < N256) {
                const int row_block_base = tile_index256(row, block);
                #pragma unroll
                for (int local = 0; local < BK256; ++local) {
                    const int column = block + local;
                    float value = tile[row_block_base + local];
                    const int column_block_base = tile_index256(column, block);
                    #pragma unroll
                    for (int prior = 0; prior < local; ++prior) {
                        value = fmaf(
                            -tile[row_block_base + prior],
                            tile[column_block_base + prior],
                            value);
                    }
                    tile[row_block_base + local] =
                        value * inverse_diag[local];
                }
            }
            __syncthreads();

            const int first_tile = block_end >> 4;
            const int remaining_tiles = TILES_PER_DIM256 - first_tile;
            const int pair_count =
                remaining_tiles * (remaining_tiles + 1) / 2;
            for (int pair = warp; pair < pair_count; pair += WARPS256) {
                const int relative_row = (int)(
                    (sqrtf(8.0f * (float)pair + 1.0f) - 1.0f) * 0.5f);
                const int relative_column =
                    pair - relative_row * (relative_row + 1) / 2;
                const int tile_row = first_tile + relative_row;
                const int tile_column = first_tile + relative_column;
                float* c_ptr = tile + tile_index256(
                    tile_row << 4, tile_column << 4);
                const float* a_ptr = tile + tile_index256(
                    tile_row << 4, block);
                const float* b_ptr = tile + tile_index256(
                    tile_column << 4, block);
                wmma::fragment<
                    wmma::accumulator, 16, 16, 8, float> c_fragment;
                wmma::load_matrix_sync(
                    c_fragment, c_ptr,
                    TILE_STRIDE256, wmma::mem_row_major);
                #pragma unroll
                for (int k = 0; k < BK256; k += 8) {
                    wmma::fragment<
                        wmma::matrix_a, 16, 16, 8,
                        wmma::precision::tf32,
                        wmma::row_major> a_fragment;
                    wmma::fragment<
                        wmma::matrix_b, 16, 16, 8,
                        wmma::precision::tf32,
                        wmma::col_major> b_fragment;
                    wmma::load_matrix_sync(
                        a_fragment, a_ptr + k, TILE_STRIDE256);
                    wmma::load_matrix_sync(
                        b_fragment, b_ptr + k, TILE_STRIDE256);
                    #pragma unroll
                    for (int element = 0;
                         element < a_fragment.num_elements; ++element) {
                        a_fragment.x[element] = -a_fragment.x[element];
                    }
                    wmma::mma_sync(
                        c_fragment, a_fragment, b_fragment, c_fragment);
                }
                wmma::store_matrix_sync(
                    c_ptr, c_fragment,
                    TILE_STRIDE256, wmma::mem_row_major);
            }
            __syncthreads();
        }
    }

    if (accurate_required) {
        int restaging_tile = 0;
        #pragma unroll 1
        for (int tile_row = 0; tile_row < TILES_PER_DIM256; ++tile_row) {
            for (int tile_column = 0; tile_column <= tile_row;
                 ++tile_column, ++restaging_tile) {
                if (tid < BK256 * BK256) {
                    const int local_row = tid >> 4;
                    const int local_column = tid & 15;
                    const int row = (tile_row << 4) + local_row;
                    const int column = (tile_column << 4) + local_column;
                    tile[restaging_tile * TILE_VALUES256
                         + local_row * TILE_STRIDE256 + local_column] =
                        (tile_row != tile_column || local_column <= local_row)
                        ? input[base + (size_t)row * N256 + column]
                        : 0.0f;
                }
            }
        }
        __syncthreads();
        factor256_accurate(
            tile,
            &reciprocal0,
            &reciprocal1,
            inverse_diag,
            pivot0,
            pivot1);
    }

    int output_tile = 0;
    #pragma unroll 1
    for (int tile_row = 0; tile_row < TILES_PER_DIM256; ++tile_row) {
        for (int tile_column = 0; tile_column <= tile_row;
             ++tile_column, ++output_tile) {
            if (tid < BK256 * BK256) {
                const int local_row = tid >> 4;
                const int local_column = tid & 15;
                const int row = (tile_row << 4) + local_row;
                const int column = (tile_column << 4) + local_column;
                if (tile_row == tile_column) {
                    output[base + (size_t)row * N256 + column] =
                        local_column <= local_row
                        ? tile[output_tile * TILE_VALUES256
                               + local_row * TILE_STRIDE256 + local_column]
                        : 0.0f;
                } else {
                    output[base + (size_t)row * N256 + column] =
                        tile[output_tile * TILE_VALUES256
                             + local_row * TILE_STRIDE256 + local_column];
                    const int upper_row = (tile_column << 4) + local_row;
                    const int upper_column = (tile_row << 4) + local_column;
                    output[base + (size_t)upper_row * N256 + upper_column] = 0.0f;
                }
            }
        }
    }
}

void chol256_launch(torch::Tensor input, torch::Tensor output) {
    const int batch = (int)input.size(0);
    static bool configured = false;
    if (!configured) {
        cudaError_t attr = cudaFuncSetAttribute(
            cholesky256_wmma16,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHARED_BYTES256);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    cholesky256_wmma16<<<dim3(batch), dim3(THREADS256), SHARED_BYTES256>>>(
        input.data_ptr<float>(), output.data_ptr<float>());
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""

def _load_cuda256() -> None:
    global _CUDA256, _CUDA256_ERROR
    if _CUDA256 is not None or _CUDA256_ERROR is not None:
        return
    if not torch.cuda.is_available():
        return
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA256 = load_inline(
            name="chol256_exp043_v35_scalar_accurate",
            cpp_sources="void chol256_launch(torch::Tensor, torch::Tensor);",
            cuda_sources=_CUDA256_SOURCE,
            functions=["chol256_launch"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        )
    except Exception as exc:
        _CUDA256_ERROR = repr(exc)


def _cuda_cholesky256(data: torch.Tensor) -> torch.Tensor:
    global _CUDA256_HITS
    out = torch.empty_like(data)
    _CUDA256.chol256_launch(data, out)
    _CUDA256_HITS += 1
    return out

_MICRO32_HITS = 0
# The diagonal micro ships inside the experiment-042 extension (one nvcc
# invocation for both kernels) so the submission still compiles in three.
_MICRO32_ERROR = _CUDA128_ERROR
_MICRO32 = _CUDA128

# Experiment 045: shapes whose split32 schedule hands its two Schur updates
# (`_panel_inner32*` and `_trailing_nb`) to cuBLAS batched GEMM instead of
# Triton. Measured on B200 at 640x512, the Triton trailing kernel runs at
# 53 TFLOP/s while the same product through cuBLAS reaches 221 TFLOP/s. Both
# updates accumulate in place on a strided view (ldc = n), so they need no
# clone, no copy-back and no final `tril_`.
# Empty: measured 0.566x (fp32 SIMT) / 0.897x (tf32) at 640x512. cuBLAS wins
# the trailing product (285 TFLOP/s vs Triton's 53) but loses the inner update
# (26 TFLOP/s -- K=32, N<=96 is too skinny to fill a tensor-core tile), and the
# first-touch `out=` form materialises the accumulator (2 x 180us). See
# notes.md; the trailing-only split remains open.
_BMM_SCHUR_SHAPES = set()
_BMM_SCHUR_HITS = 0


# Experiment 046: shapes whose split32 trailing Schur update goes to cuBLAS.
# Measured on B200: Triton's `_trailing_nb` reaches 53-66 TFLOP/s while the
# same product through a strided in-place `baddbmm_` reaches 235-256. Only the
# trailing update moves. Exp 045 measured the panel inner update at 26 TFLOP/s
# through cuBLAS (K=32, N<=96 cannot fill a tensor-core tile), and exp 046
# showed that a block-inverse design which would fatten it is 0.69x overall,
# because the diagonal blocks it leaves behind carry 30% of the flops at
# ~30 TFLOP/s. The first-touch block keeps the Triton kernel: cuBLAS cannot
# read `src` and write `work` in one pass, and `baddbmm(src, ..., out=work)`
# materialises the accumulator first (measured 180us at 640x512).
# 60x1024 is excluded: measured 0.9320x with an unstable 0.63% MAD and a 0.9%
# order spread (ratios 0.89-1.02), against 1.0257x at 640x512 and 1.0386x at
# 8x2048. At batch 60 the strided in-place accumulate does not hold the
# throughput the isolated GEMM probe predicted.
_BMM_TRAILING_SHAPES = {
    (640, 512),
    (8, 2048),
}
_BMM_TRAILING_HITS = 0


# Shapes whose split32 schedule uses the CUDA diagonal micro-factorization.
# Only the eager-mode split32 shapes are enrolled. The kernel uses a plain
# <<<grid, block>>> launch with no queue argument, which is correct in eager
# mode but is not capturable into the CUDA graphs the remaining split32 shapes
# replay -- measured 0.38-0.52x there, through the finiteness fallback. Naming
# the current work queue explicitly would make capture work but is rejected by
# popcorn's source policy, so those shapes keep the Triton diagonal micro.
_MICRO32_SHAPES = {
    (640, 512),
    (60, 1024),
}


# Experiment 047: shapes whose below-diagonal panel solve is done by one
# resident-tile kernel per 128-wide block instead of the seven launches
# (4x micro + 4x apply + 3x inner) the shipped schedule emits. Maps the shape
# to (TILE_R, num_warps) for `_panel_fused128`.
#
# Motivation is a traffic bound, not a throughput estimate: at 640x512
# `_panel_inner32_subtile64` moves 275 MB per call in 36.0us = 7.6 TB/s, which
# is B200 HBM peak, so it cannot be made faster as written. It moves that much
# because the block-column tile is re-read from global on every launch of the
# block. Loading the tile once and storing it once takes total panel traffic
# from ~3.5 GB to ~503 MB at nb=128.
# (TILE_R, num_warps, merge_diag_step). `merge_diag_step` collapses the
# per-sub-step `_panel_apply32` + `_panel_inner32_subtile64` pair inside the
# diagonal block into one `_diag_block_step` launch; it trades two cheap
# launches for one register-heavy CTA-per-matrix launch and is only a win
# where the launch count dominates.
# Measured paired vs ranked #890659 (variant-05/06):
#   640x512  merge=False 1.0566x   merge=True 0.9973x
#   60x1024  merge=False 0.9147x   merge=True 1.2044x
#   8x2048   0.9070x -- excluded. Its shipped schedule is NB=256 (exp 032,
#     1.031x) and the fused panel requires uniform 128-wide panels, so
#     enrolling it doubles the panel and trailing launch count.
# At batch 640 the merged step's 128x128 register tiles cost more than the
# launches they remove; at batch 60 there is no occupancy to lose and the
# eight-panel schedule emits twice as many of them.
_FUSED_PANEL_SHAPES = {
    (640, 512): (128, 8, False),
    (60, 1024): (128, 8, True),
}
_FUSED_PANEL_HITS = 0


# ---------------------------------------------------------------------------
# Triton kernel for n == 32 (adopted experiment 002).
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _chol_batched_kernel(
        A_ptr,
        L_ptr,
        stride_ab,
        stride_ai,
        stride_aj,
        stride_lb,
        stride_li,
        stride_lj,
        N: tl.constexpr,
    ):
        """One program (CTA) factorizes one N x N SPD matrix (right-looking)."""
        pid = tl.program_id(0)
        rows = tl.arange(0, N)
        cols = tl.arange(0, N)
        a_ptrs = (
            A_ptr
            + pid * stride_ab
            + rows[:, None] * stride_ai
            + cols[None, :] * stride_aj
        )
        a = tl.load(a_ptrs)

        for k in range(N):
            akk = tl.sum(
                tl.where((rows[:, None] == k) & (cols[None, :] == k), a, 0.0)
            )
            inv = 1.0 / tl.sqrt(akk)
            col_k = (cols[None, :] == k) & (rows[:, None] >= k)
            a = tl.where(col_k, a * inv, a)
            lk = tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)
            trail = (rows[:, None] > k) & (cols[None, :] > k)
            a = tl.where(trail, a - lk[:, None] * lk[None, :], a)

        a = tl.where(cols[None, :] > rows[:, None], 0.0, a)
        l_ptrs = (
            L_ptr
            + pid * stride_lb
            + rows[:, None] * stride_li
            + cols[None, :] * stride_lj
        )
        tl.store(l_ptrs, a)

    @triton.jit
    def _chol32_rank2_kernel(
        A_ptr,
        L_ptr,
        N: tl.constexpr,
    ):
        """One warp factorizes one 32x32 SPD matrix with rank-2 steps: the
        serial dependency chain is 16 iterations instead of 32."""
        pid = tl.program_id(0).to(tl.int64)
        r = tl.arange(0, N)
        c = tl.arange(0, N)
        a = tl.load(A_ptr + pid * N * N + r[:, None] * N + c[None, :])
        for it in range(0, N // 2):
            p = 2 * it
            q = p + 1
            colp = tl.sum(tl.where(c[None, :] == p, a, 0.0), axis=1)
            colq = tl.sum(tl.where(c[None, :] == q, a, 0.0), axis=1)
            dpp = tl.sum(tl.where(r == p, colp, 0.0), axis=0)
            aqq = tl.sum(tl.where(r == q, colq, 0.0), axis=0)
            inv1 = 1.0 / tl.sqrt(dpp)
            lp = tl.where(r >= p, colp * inv1, 0.0)
            l21 = tl.sum(tl.where(r == q, lp, 0.0), axis=0)
            dqq = aqq - l21 * l21
            inv2 = 1.0 / tl.sqrt(dqq)
            lq = tl.where(r >= q, (colq - l21 * lp) * inv2, 0.0)
            trail = (r[:, None] > q) & (c[None, :] > q)
            a = tl.where(
                c[None, :] == p,
                lp[:, None],
                tl.where(
                    c[None, :] == q,
                    lq[:, None],
                    tl.where(
                        trail,
                        a - lp[:, None] * lp[None, :] - lq[:, None] * lq[None, :],
                        a,
                    ),
                ),
            )
        a = tl.where(c[None, :] <= r[:, None], a, 0.0)
        tl.store(L_ptr + pid * N * N + r[:, None] * N + c[None, :], a)

    def _triton_cholesky32_rank2(data: torch.Tensor) -> torch.Tensor:
        batch, n, _ = data.shape
        data = data.contiguous()
        out = torch.empty_like(data)
        _chol32_rank2_kernel[(batch,)](data, out, N=n, num_warps=1)
        return out

    _NUM_WARPS = {32: 1}

    def _triton_cholesky32(data: torch.Tensor) -> torch.Tensor:
        batch, n, _ = data.shape
        data = data.contiguous()
        out = torch.empty_like(data)
        _chol_batched_kernel[(batch,)](
            data,
            out,
            data.stride(0),
            data.stride(1),
            data.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            N=n,
            num_warps=_NUM_WARPS.get(n, 4),
        )
        return out


    _BK_8X2048 = 64
    _UPDATE_TILE_8X2048 = 128

    @triton.jit
    def _diag_factor_8x2048(
        a_ptr,
        n: tl.constexpr,
        k,
        BK_CONST: tl.constexpr,
    ):
        batch = tl.program_id(0)
        rows = tl.arange(0, BK_CONST)
        cols = tl.arange(0, BK_CONST)
        base = batch * n * n
        ptrs = a_ptr + base + (k + rows[:, None]) * n + k + cols[None, :]
        tile = tl.load(ptrs)

        for p in range(0, BK_CONST):
            diag_mask = (rows[:, None] == p) & (cols[None, :] == p)
            diagonal = tl.sum(tl.where(diag_mask, tile, 0.0))
            inv_sqrt = 1.0 / tl.sqrt(diagonal)
            column_mask = (cols[None, :] == p) & (rows[:, None] >= p)
            tile = tl.where(column_mask, tile * inv_sqrt, tile)
            column = tl.sum(
                tl.where(cols[None, :] == p, tile, 0.0), axis=1
            )
            trailing = (rows[:, None] > p) & (cols[None, :] > p)
            tile = tl.where(
                trailing,
                tile - column[:, None] * column[None, :],
                tile,
            )

        tl.store(ptrs, tile, mask=cols[None, :] <= rows[:, None])

    @triton.jit
    def _panel_solve_8x2048(
        a_ptr,
        n: tl.constexpr,
        k,
        remaining,
        BK_CONST: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        batch = tl.program_id(1)
        rows = row_tile * BK_CONST + tl.arange(0, BK_CONST)
        cols = tl.arange(0, BK_CONST)
        base = batch * n * n
        row_mask = rows < remaining

        diag_ptrs = (
            a_ptr
            + base
            + (k + cols[:, None]) * n
            + k
            + cols[None, :]
        )
        diagonal = tl.load(diag_ptrs)
        panel_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + rows[:, None]) * n
            + k
            + cols[None, :]
        )
        panel = tl.load(panel_ptrs, mask=row_mask[:, None], other=0.0)

        for p in range(0, BK_CONST):
            diag_column = tl.sum(
                tl.where(cols[None, :] == p, diagonal, 0.0), axis=1
            )
            diag_pp = tl.sum(
                tl.where(cols == p, diag_column, 0.0), axis=0
            )
            value = tl.sum(
                tl.where(cols[None, :] == p, panel, 0.0), axis=1
            ) / diag_pp
            panel = tl.where(cols[None, :] == p, value[:, None], panel)
            panel = tl.where(
                cols[None, :] > p,
                panel - value[:, None] * diag_column[None, :],
                panel,
            )

        tl.store(panel_ptrs, panel, mask=row_mask[:, None])

    @triton.jit
    def _lower_schur_8x2048(
        a_ptr,
        n: tl.constexpr,
        k,
        remaining,
        BK_CONST: tl.constexpr,
        TILE: tl.constexpr,
    ):
        triangular_id = tl.program_id(0)
        batch = tl.program_id(1)
        block_row = (
            (tl.sqrt(8.0 * triangular_id + 1.0) - 1.0) * 0.5
        ).to(tl.int32)
        block_col = triangular_id - block_row * (block_row + 1) // 2

        rows = block_row * TILE + tl.arange(0, TILE)
        cols = block_col * TILE + tl.arange(0, TILE)
        depth = tl.arange(0, BK_CONST)
        base = batch * n * n
        lhs_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + rows[:, None]) * n
            + k
            + depth[None, :]
        )
        rhs_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + cols[None, :]) * n
            + k
            + depth[:, None]
        )
        lhs = tl.load(lhs_ptrs, mask=rows[:, None] < remaining, other=0.0)
        rhs = tl.load(rhs_ptrs, mask=cols[None, :] < remaining, other=0.0)
        product = tl.dot(lhs, rhs, input_precision="tf32", out_dtype=tl.float32)

        out_ptrs = (
            a_ptr
            + base
            + (k + BK_CONST + rows[:, None]) * n
            + k
            + BK_CONST
            + cols[None, :]
        )
        valid = (rows[:, None] < remaining) & (cols[None, :] < remaining)
        valid = valid & (
            (block_row != block_col) | (cols[None, :] <= rows[:, None])
        )
        old = tl.load(out_ptrs, mask=valid, other=0.0)
        tl.store(out_ptrs, old - product, mask=valid)

    @triton.jit
    def _clear_upper_8x2048(
        a_ptr,
        total: tl.constexpr,
        n: tl.constexpr,
        BLOCK: tl.constexpr,
        GRID: tl.constexpr,
    ):
        first = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        for step in range(0, total, GRID * BLOCK):
            offsets = first + step
            valid = offsets < total
            matrix_offset = offsets % (n * n)
            row = matrix_offset // n
            col = matrix_offset - row * n
            tl.store(a_ptr + offsets, 0.0, mask=valid & (col > row))

    @triton.jit
    def _dual_tiled_amax_e4m3_32768(
        lhs_ptr,
        rhs_ptr,
        lhs_partial_ptr,
        rhs_partial_ptr,
        lhs_rows,
        lhs_columns,
        rhs_rows,
        rhs_columns,
        lhs_stride_row,
        lhs_stride_column,
        rhs_stride_row,
        rhs_stride_column,
        lhs_tiles,
        rhs_tiles,
        lhs_programs,
        rhs_programs,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)

        lhs_row = pid // lhs_tiles
        lhs_tile = pid - lhs_row * lhs_tiles
        lhs_cols = lhs_tile * BLOCK + offsets
        lhs_valid = (pid < lhs_programs) & (lhs_cols < lhs_columns)
        lhs = tl.load(
            lhs_ptr
            + lhs_row * lhs_stride_row
            + lhs_cols * lhs_stride_column,
            mask=lhs_valid,
            other=0.0,
        )
        lhs_max = tl.max(tl.abs(lhs), axis=0)
        tl.store(lhs_partial_ptr + pid, lhs_max, mask=pid < lhs_programs)

        rhs_row = pid // rhs_tiles
        rhs_tile = pid - rhs_row * rhs_tiles
        rhs_cols = rhs_tile * BLOCK + offsets
        rhs_valid = (pid < rhs_programs) & (rhs_cols < rhs_columns)
        rhs = tl.load(
            rhs_ptr
            + rhs_row * rhs_stride_row
            + rhs_cols * rhs_stride_column,
            mask=rhs_valid,
            other=0.0,
        )
        rhs_max = tl.max(tl.abs(rhs), axis=0)
        tl.store(rhs_partial_ptr + pid, rhs_max, mask=pid < rhs_programs)

    @triton.jit
    def _dual_scale_cast_e4m3_32768(
        lhs_ptr,
        rhs_ptr,
        quantized_lhs_ptr,
        quantized_rhs_ptr,
        scale_lhs_ptr,
        scale_rhs_ptr,
        lhs_elements,
        rhs_elements,
        lhs_columns,
        rhs_columns,
        lhs_stride_row,
        lhs_stride_column,
        rhs_stride_row,
        rhs_stride_column,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        lhs_mask = offsets < lhs_elements
        lhs_rows = offsets // lhs_columns
        lhs_cols = offsets - lhs_rows * lhs_columns
        lhs = tl.load(
            lhs_ptr
            + lhs_rows * lhs_stride_row
            + lhs_cols * lhs_stride_column,
            mask=lhs_mask,
            other=0.0,
        )
        scale_lhs = tl.load(scale_lhs_ptr)
        tl.store(
            quantized_lhs_ptr + offsets,
            lhs * scale_lhs,
            mask=lhs_mask,
        )

        rhs_mask = offsets < rhs_elements
        rhs_rows = offsets // rhs_columns
        rhs_cols = offsets - rhs_rows * rhs_columns
        rhs = tl.load(
            rhs_ptr
            + rhs_rows * rhs_stride_row
            + rhs_cols * rhs_stride_column,
            mask=rhs_mask,
            other=0.0,
        )
        scale_rhs = tl.load(scale_rhs_ptr)
        tl.store(
            quantized_rhs_ptr + offsets,
            rhs * scale_rhs,
            mask=rhs_mask,
        )

    @triton.jit
    def _mx_quant_e4m3_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        stride_xm,
        stride_xk,
        columns,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Experiment 034: single-pass MXFP8 quantization. Each program casts
        a (BLOCK_M, BLOCK_K) fp32 tile to e4m3 values plus one shared e8m0
        scale (biased-exponent byte) per 32-element K-block, per the OCP
        microscaling spec: scale = 2^(floor(log2(amax)) - 8) with a saturating
        element cast. Replaces the exp-014 per-tensor amax reduction + host
        scale + scale/cast pass pair. The grid must tile the operand exactly.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk
        )
        grouped = tl.reshape(x, (BLOCK_M, BLOCK_K // 32, 32))
        amax = tl.max(tl.abs(grouped), axis=2)
        # floor(log2(amax)) from the fp32 exponent bits; e4m3 emax is 8, so
        # the biased shared-exponent byte is exp_bits - 8 (amax == 0 -> 0).
        exp_bits = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
        sbyte = tl.maximum(exp_bits - 8, 0)
        inv_scale = tl.exp2((127 - sbyte).to(tl.float32))
        q = grouped * inv_scale[:, :, None]
        tl.store(
            q_ptr + rows[:, None] * columns + cols[None, :],
            tl.reshape(q, (BLOCK_M, BLOCK_K)).to(tl.float8e4nv),
        )
        scale_cols = pid_k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)
        tl.store(
            s_ptr + rows[:, None] * (columns // 32) + scale_cols[None, :],
            sbyte.to(tl.uint8),
        )

    @triton.jit
    def _mx_quant_e4m3_blocked_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        stride_xm,
        stride_xk,
        columns,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Experiment 034 V2: same single-pass MXFP8 quantization as
        `_mx_quant_e4m3_kernel`, but the e8m0 scale bytes are stored directly
        in the 128x4 *blocked* (swizzled) layout `torch._scaled_mm` requires
        for MX operands, so no separate permute/contiguous pass is needed.

        Within one (128 rows x 4 scale-col) tile the byte order is
        `(row % 32) * 16 + (row // 32 % 4) * 4 + scale_col % 4`, tiles laid out
        row-major over (rows/128, columns/128). With BLOCK_M=32 / BLOCK_K=128
        each program owns exactly one (32 rows x 4 scale-col) quarter-tile, so
        the row-block and intra-tile `a` index are program constants.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xk
        )
        grouped = tl.reshape(x, (BLOCK_M, BLOCK_K // 32, 32))
        amax = tl.max(tl.abs(grouped), axis=2)
        exp_bits = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
        sbyte = tl.maximum(exp_bits - 8, 0)
        inv_scale = tl.exp2((127 - sbyte).to(tl.float32))
        q = grouped * inv_scale[:, :, None]
        tl.store(
            q_ptr + rows[:, None] * columns + cols[None, :],
            tl.reshape(q, (BLOCK_M, BLOCK_K)).to(tl.float8e4nv),
        )
        tile = (pid_m // 4) * (columns // 128) + pid_k
        b = tl.arange(0, BLOCK_M)
        c_in = tl.arange(0, BLOCK_K // 32)
        tl.store(
            s_ptr
            + tile * 512
            + b[:, None] * 16
            + (pid_m % 4) * 4
            + c_in[None, :],
            sbyte.to(tl.uint8),
        )

    @triton.jit
    def _mxfp8_panel_update_kernel(
        q_lhs_ptr,
        s_lhs_ptr,
        q_rhs_ptr,
        s_rhs_ptr,
        out_ptr,
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Experiment 034: out (M, N fp32, contiguous) -= lhs @ rhs^T where
        lhs (M, K) and rhs (N, K) are contiguous MXFP8 operands (e4m3 values,
        per-32 e8m0 scales). tl.dot_scaled lowers to the Blackwell
        block-scaled tensor-core MMA (tcgen05.mma kind::mxf8f6f4) on sm_100,
        applying both scale vectors inside the instruction. The subtraction
        into the panel is fused in the epilogue, saving the separate product
        materialization + sub_ passes. Exact tiling required."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_s = tl.arange(0, BLOCK_K // 32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            lhs = tl.load(
                q_lhs_ptr + rows[:, None] * K + k0 + offs_k[None, :]
            )
            lhs_scale = tl.load(
                s_lhs_ptr
                + rows[:, None] * (K // 32)
                + k0 // 32
                + offs_s[None, :]
            )
            rhs = tl.load(
                q_rhs_ptr + cols[:, None] * K + k0 + offs_k[None, :]
            )
            rhs_scale = tl.load(
                s_rhs_ptr
                + cols[:, None] * (K // 32)
                + k0 // 32
                + offs_s[None, :]
            )
            acc = tl.dot_scaled(
                lhs,
                lhs_scale,
                "e4m3",
                tl.trans(rhs),
                rhs_scale,
                "e4m3",
                acc,
            )
        out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]
        tl.store(out_ptrs, tl.load(out_ptrs) - acc)

    @triton.jit
    def _clear_upper_tiles(
        out_ptr,
        n: tl.constexpr,
        TILE: tl.constexpr,
    ):
        """Zero the strict upper triangle, one TILE x TILE tile per CTA over
        the upper-triangular tile grid only (no div/mod per element)."""
        tri = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        br = ((tl.sqrt(8.0 * tri + 1.0) - 1.0) * 0.5).to(tl.int32)
        bc = tri - br * (br + 1) // 2
        # (br, bc) enumerates lower tiles; mirror to upper: row tile bc,
        # col tile br.
        rows = bc * TILE + tl.arange(0, TILE)
        cols = br * TILE + tl.arange(0, TILE)
        ptrs = out_ptr + b * n * n + rows[:, None] * n + cols[None, :]
        mask = cols[None, :] > rows[:, None]
        tl.store(ptrs, tl.zeros((TILE, TILE), dtype=tl.float32), mask=mask)

    @triton.jit
    def _micro_potrf_gj32(
        out_ptr,
        inv_ptr,
        src_ptr,
        n: tl.constexpr,
        k,
        FIRST: tl.constexpr,
        RECIPROCAL_SOLVE: tl.constexpr,
    ):
        """Factor the 32x32 diagonal block at (k, k) and build its triangular
        inverse in the same 32-step serial loop (row p of L is final after
        step p, so X[p,:] = (I[p,:] - L[p,:p] @ X[:p,:]) / l_pp can be formed
        immediately). One warp per matrix keeps every reduction warp-local."""
        b = tl.program_id(0).to(tl.int64)
        r = tl.arange(0, 32)
        c = tl.arange(0, 32)
        off = (k + r)[:, None] * n + (k + c)[None, :]
        ptr = out_ptr + b * n * n + off
        if FIRST:
            a = tl.load(src_ptr + b * n * n + off)
        else:
            a = tl.load(ptr)
        x = tl.where(r[:, None] == c[None, :], 1.0, 0.0)
        # Rank-4 right-looking factorization: four columns per serial step
        # (exp 017). The 4x4 pivot block reduces to a pure scalar chain fed
        # by ten ILP-parallel extractions; the trailing update is one fused
        # 4-way outer-product write; the inverse advances four rows per step
        # with scalar corrections inside the pivot block.
        for it in range(0, 8):
            p0 = 4 * it
            p1 = p0 + 1
            p2 = p0 + 2
            p3 = p0 + 3
            # Raw pivot columns and the ten pivot-block scalars (all
            # independent -> issue together).
            c0 = tl.sum(tl.where(c[None, :] == p0, a, 0.0), axis=1)
            c1 = tl.sum(tl.where(c[None, :] == p1, a, 0.0), axis=1)
            c2 = tl.sum(tl.where(c[None, :] == p2, a, 0.0), axis=1)
            c3 = tl.sum(tl.where(c[None, :] == p3, a, 0.0), axis=1)
            m00 = tl.sum(tl.where(r == p0, c0, 0.0), axis=0)
            m01 = tl.sum(tl.where(r == p1, c0, 0.0), axis=0)
            m02 = tl.sum(tl.where(r == p2, c0, 0.0), axis=0)
            m03 = tl.sum(tl.where(r == p3, c0, 0.0), axis=0)
            m11 = tl.sum(tl.where(r == p1, c1, 0.0), axis=0)
            m12 = tl.sum(tl.where(r == p2, c1, 0.0), axis=0)
            m13 = tl.sum(tl.where(r == p3, c1, 0.0), axis=0)
            m22 = tl.sum(tl.where(r == p2, c2, 0.0), axis=0)
            m23 = tl.sum(tl.where(r == p3, c2, 0.0), axis=0)
            m33 = tl.sum(tl.where(r == p3, c3, 0.0), axis=0)
            # Scalar Cholesky of the 4x4 pivot block. tl.rsqrt replaces the
            # sqrt.approx + div.full pair on the serial scalar chain (exp 029).
            inv0 = tl.rsqrt(m00)
            s01 = m01 * inv0
            s02 = m02 * inv0
            s03 = m03 * inv0
            d1 = m11 - s01 * s01
            inv1 = tl.rsqrt(d1)
            s12 = (m12 - s01 * s02) * inv1
            s13 = (m13 - s01 * s03) * inv1
            d2 = m22 - s02 * s02 - s12 * s12
            inv2 = tl.rsqrt(d2)
            s23 = (m23 - s02 * s03 - s12 * s13) * inv2
            d3 = m33 - s03 * s03 - s13 * s13 - s23 * s23
            inv3 = tl.rsqrt(d3)
            # Finalized pivot columns.
            l0 = tl.where(r >= p0, c0 * inv0, 0.0)
            l1 = tl.where(r >= p1, (c1 - s01 * l0) * inv1, 0.0)
            l2 = tl.where(r >= p2, (c2 - s02 * l0 - s12 * l1) * inv2, 0.0)
            l3 = tl.where(
                r >= p3, (c3 - s03 * l0 - s13 * l1 - s23 * l2) * inv3, 0.0
            )
            trail = (r[:, None] > p3) & (c[None, :] > p3)
            a = tl.where(
                c[None, :] == p0,
                l0[:, None],
                tl.where(
                    c[None, :] == p1,
                    l1[:, None],
                    tl.where(
                        c[None, :] == p2,
                        l2[:, None],
                        tl.where(
                            c[None, :] == p3,
                            l3[:, None],
                            tl.where(
                                trail,
                                a
                                - l0[:, None] * l0[None, :]
                                - l1[:, None] * l1[None, :]
                                - l2[:, None] * l2[None, :]
                                - l3[:, None] * l3[None, :],
                                a,
                            ),
                        ),
                    ),
                ),
            )
            # Inverse rows p0..p3. All four contributions reduce against X
            # rows < p0 (independent); the in-block terms use the pivot
            # scalars already in registers.
            row0 = tl.sum(tl.where(r[:, None] == p0, a, 0.0), axis=0)
            row1 = tl.sum(tl.where(r[:, None] == p1, a, 0.0), axis=0)
            row2 = tl.sum(tl.where(r[:, None] == p2, a, 0.0), axis=0)
            row3 = tl.sum(tl.where(r[:, None] == p3, a, 0.0), axis=0)
            rm0 = tl.where(c < p0, row0, 0.0)
            rm1 = tl.where(c < p0, row1, 0.0)
            rm2 = tl.where(c < p0, row2, 0.0)
            rm3 = tl.where(c < p0, row3, 0.0)
            g0 = tl.sum(rm0[:, None] * x, axis=0)
            g1 = tl.sum(rm1[:, None] * x, axis=0)
            g2 = tl.sum(rm2[:, None] * x, axis=0)
            g3 = tl.sum(rm3[:, None] * x, axis=0)
            e0 = tl.where(c == p0, 1.0, 0.0)
            e1 = tl.where(c == p1, 1.0, 0.0)
            e2 = tl.where(c == p2, 1.0, 0.0)
            e3 = tl.where(c == p3, 1.0, 0.0)
            if RECIPROCAL_SOLVE:
                x0 = (e0 - g0) * inv0
                x1 = (e1 - g1 - s01 * x0) * inv1
                x2 = (e2 - g2 - s02 * x0 - s12 * x1) * inv2
                x3 = (e3 - g3 - s03 * x0 - s13 * x1 - s23 * x2) * inv3
            else:
                lpp0 = m00 * inv0
                lpp1 = d1 * inv1
                lpp2 = d2 * inv2
                lpp3 = d3 * inv3
                x0 = (e0 - g0) / lpp0
                x1 = (e1 - g1 - s01 * x0) / lpp1
                x2 = (e2 - g2 - s02 * x0 - s12 * x1) / lpp2
                x3 = (e3 - g3 - s03 * x0 - s13 * x1 - s23 * x2) / lpp3
            x = tl.where(
                r[:, None] == p0,
                x0[None, :],
                tl.where(
                    r[:, None] == p1,
                    x1[None, :],
                    tl.where(
                        r[:, None] == p2,
                        x2[None, :],
                        tl.where(r[:, None] == p3, x3[None, :], x),
                    ),
                ),
            )
        a = tl.where(c[None, :] <= r[:, None], a, 0.0)
        tl.store(ptr, a)
        tl.store(inv_ptr + b * 1024 + r[:, None] * 32 + c[None, :], x)

    @triton.jit
    def _panel_apply32(
        out_ptr,
        inv_ptr,
        src_ptr,
        n: tl.constexpr,
        k,
        remaining,
        PREC: tl.constexpr,
        TILE_R: tl.constexpr,
        FIRST: tl.constexpr,
    ):
        """L[i, k-block] = A[i, k-block] @ Dinv^T for all rows below the
        diagonal block (full panel column of the factor)."""
        rt = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        rows = rt * TILE_R + tl.arange(0, TILE_R)
        c = tl.arange(0, 32)
        base = b * n * n
        mask = rows < remaining
        p_off = (k + 32 + rows)[:, None] * n + (k + c)[None, :]
        p_ptrs = out_ptr + base + p_off
        if FIRST:
            p = tl.load(src_ptr + base + p_off, mask=mask[:, None], other=0.0)
        else:
            p = tl.load(p_ptrs, mask=mask[:, None], other=0.0)
        dinv = tl.load(inv_ptr + b * 1024 + c[:, None] * 32 + c[None, :])
        lik = tl.dot(
            p, tl.trans(dinv), input_precision=PREC, out_dtype=tl.float32
        )
        tl.store(p_ptrs, lik, mask=mask[:, None])
        # Zero-fill the mirrored upper tile so no separate clear pass is
        # needed: block rows k..k+32, columns = this CTA's panel rows.
        m_ptrs = (
            out_ptr + base + (k + c)[:, None] * n + (k + 32 + rows)[None, :]
        )
        tl.store(
            m_ptrs,
            tl.zeros((32, TILE_R), dtype=tl.float32),
            mask=mask[None, :],
        )

    @triton.jit
    def _panel_inner32(
        out_ptr,
        src_ptr,
        n: tl.constexpr,
        k,
        width,
        remaining,
        PREC: tl.constexpr,
        TILE_R: tl.constexpr,
        FIRST: tl.constexpr,
    ):
        """Narrow rank-32 update of the remaining panel columns only:
        T[rows, k+32 : k+32+width] -= L[rows, k-blk] @ L[cols, k-blk]^T."""
        rt = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        rows = rt * TILE_R + tl.arange(0, TILE_R)
        cw = tl.arange(0, 128)
        c = tl.arange(0, 32)
        base = b * n * n
        rmask = rows < remaining
        li = tl.load(
            out_ptr + base + (k + 32 + rows)[:, None] * n + (k + c)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        wmask = cw < width
        lj = tl.load(
            out_ptr + base + (k + 32 + cw)[:, None] * n + (k + c)[None, :],
            mask=wmask[:, None],
            other=0.0,
        )
        prod = tl.dot(
            li, tl.trans(lj), input_precision=PREC, out_dtype=tl.float32
        )
        t_off = (k + 32 + rows)[:, None] * n + (k + 32 + cw)[None, :]
        t_ptrs = out_ptr + base + t_off
        valid = rmask[:, None] & wmask[None, :]
        if FIRST:
            t = tl.load(src_ptr + base + t_off, mask=valid, other=0.0)
        else:
            t = tl.load(t_ptrs, mask=valid, other=0.0)
        tl.store(t_ptrs, t - prod, mask=valid)

    @triton.jit
    def _panel_inner32_subtile64(
        out_ptr,
        src_ptr,
        n: tl.constexpr,
        k,
        width,
        remaining,
        PREC: tl.constexpr,
        NTILES_C: tl.constexpr,
        FIRST: tl.constexpr,
    ):
        """The same panel update with a 64x64 output tile.

        The shipped 128x128 specialization reaches the 255-register ceiling
        and spills its epilogue. Splitting both output axes reduces the live
        accumulator surface by 4x; NTILES_C maps the one-dimensional launch
        grid back to independent row/column tiles.
        """
        pid = tl.program_id(0)
        rt = pid // NTILES_C
        ct = pid - rt * NTILES_C
        b = tl.program_id(1).to(tl.int64)
        rows = rt * 64 + tl.arange(0, 64)
        cw = ct * 64 + tl.arange(0, 64)
        c = tl.arange(0, 32)
        base = b * n * n
        rmask = rows < remaining
        li = tl.load(
            out_ptr + base + (k + 32 + rows)[:, None] * n + (k + c)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        wmask = cw < width
        lj = tl.load(
            out_ptr + base + (k + 32 + cw)[:, None] * n + (k + c)[None, :],
            mask=wmask[:, None],
            other=0.0,
        )
        prod = tl.dot(
            li, tl.trans(lj), input_precision=PREC, out_dtype=tl.float32
        )
        t_off = (k + 32 + rows)[:, None] * n + (k + 32 + cw)[None, :]
        t_ptrs = out_ptr + base + t_off
        valid = rmask[:, None] & wmask[None, :]
        if FIRST:
            t = tl.load(src_ptr + base + t_off, mask=valid, other=0.0)
        else:
            t = tl.load(t_ptrs, mask=valid, other=0.0)
        tl.store(t_ptrs, t - prod, mask=valid)

    @triton.jit
    def _diag_block_step(
        out_ptr,
        dinv_ptr,
        src_ptr,
        n: tl.constexpr,
        k,
        nrows,
        TILE: tl.constexpr,
        PREC: tl.constexpr,
        FIRST: tl.constexpr,
    ):
        """One CTA per matrix: apply + inner update for the rows of the
        128-wide diagonal block that lie below the 32x32 pivot at `k`.

        With `_panel_fused128` taking every row below the block, `nrows` is at
        most 96, so the two launches the shipped schedule spends here
        (`_panel_apply32` + `_panel_inner32_subtile64`) move almost no data and
        are pure fixed cost -- measured 8.7us and 13.0us per call at 640x512.
        Doing both in one kernel halves that. The inner update is L @ L^T of
        the very tile the apply just produced, so it needs no second global
        read either."""
        b = tl.program_id(0).to(tl.int64)
        base = b * n * n
        r = tl.arange(0, TILE)
        c = tl.arange(0, 32)
        rmask = r < nrows
        off = (k + 32 + r)[:, None] * n + (k + c)[None, :]
        p = out_ptr + base + off
        if FIRST:
            a = tl.load(src_ptr + base + off, mask=rmask[:, None], other=0.0)
        else:
            a = tl.load(p, mask=rmask[:, None], other=0.0)
        dinv = tl.load(dinv_ptr + b * 1024 + c[:, None] * 32 + c[None, :])
        lik = tl.dot(
            a, tl.trans(dinv), input_precision=PREC, out_dtype=tl.float32
        )
        tl.store(p, lik, mask=rmask[:, None])
        m = out_ptr + base + (k + c)[:, None] * n + (k + 32 + r)[None, :]
        tl.store(
            m, tl.zeros((32, TILE), dtype=tl.float32), mask=rmask[None, :]
        )
        t_off = (k + 32 + r)[:, None] * n + (k + 32 + r)[None, :]
        tp = out_ptr + base + t_off
        valid = rmask[:, None] & rmask[None, :]
        if FIRST:
            t = tl.load(src_ptr + base + t_off, mask=valid, other=0.0)
        else:
            t = tl.load(tp, mask=valid, other=0.0)
        prod = tl.dot(
            lik, tl.trans(lik), input_precision=PREC, out_dtype=tl.float32
        )
        tl.store(tp, t - prod, mask=valid)

    @triton.jit
    def _fdot(a, ptr, TRANSLOAD: tl.constexpr, PREC: tl.constexpr):
        """a @ B^T where `ptr` addresses B (TRANSLOAD=0, via `tl.trans`) or
        already addresses B^T with swapped index expressions (TRANSLOAD=1).
        `tl.trans` on a freshly loaded tile costs a shared-memory round trip;
        the operand is 32x32 and L1-resident either way, so the transposed
        addressing is free."""
        if TRANSLOAD:
            return tl.dot(
                a, tl.load(ptr), input_precision=PREC, out_dtype=tl.float32
            )
        return tl.dot(
            a,
            tl.trans(tl.load(ptr)),
            input_precision=PREC,
            out_dtype=tl.float32,
        )

    @triton.jit
    def _panel_fused128(
        out_ptr,
        dinv_ptr,
        src_ptr,
        n: tl.constexpr,
        j,
        nrows,
        dinv_stride,
        TILE_R: tl.constexpr,
        PREC: tl.constexpr,
        FIRST: tl.constexpr,
        TRANSLOAD: tl.constexpr,
        MIRROR: tl.constexpr,
    ):
        """Fused 128-wide panel solve for every row below the diagonal block.

        One CTA owns a TILE_R x 128 tile of the block column, loads it once,
        runs all four 32-wide sub-steps against the diagonal inverses already
        published in `dinv`, and stores once. The shipped schedule re-reads
        the same tile from global on every one of the seven launches that make
        up one 128-wide block (4x micro + 4x apply + 3x inner); at 640x512
        that is ~3.5 GB of panel traffic against a one-load/one-store minimum
        of ~503 MB.

        No cross-CTA synchronisation is needed: the diagonal block is fully
        factored before this kernel launches and row tiles are independent of
        each other.
        """
        rt = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        base = b * n * n
        rr = rt * TILE_R + tl.arange(0, TILE_R)
        rmask = rr < nrows
        c = tl.arange(0, 32)
        off = (j + 128 + rr)[:, None] * n + (j + c)[None, :]
        p0 = out_ptr + base + off
        if FIRST:
            q0 = src_ptr + base + off
            t0 = tl.load(q0, mask=rmask[:, None], other=0.0)
            t1 = tl.load(q0 + 32, mask=rmask[:, None], other=0.0)
            t2 = tl.load(q0 + 64, mask=rmask[:, None], other=0.0)
            t3 = tl.load(q0 + 96, mask=rmask[:, None], other=0.0)
        else:
            t0 = tl.load(p0, mask=rmask[:, None], other=0.0)
            t1 = tl.load(p0 + 32, mask=rmask[:, None], other=0.0)
            t2 = tl.load(p0 + 64, mask=rmask[:, None], other=0.0)
            t3 = tl.load(p0 + 96, mask=rmask[:, None], other=0.0)
        # Both addressings visit the same 32x32 tiles; TRANSLOAD only swaps
        # which axis is the fast one, so the (u, s) block offsets are shared.
        if TRANSLOAD:
            dbase = dinv_ptr + b * 1024 + c[None, :] * 32 + c[:, None]
            dblk = out_ptr + base + (j + c)[None, :] * n + (j + c)[:, None]
        else:
            dbase = dinv_ptr + b * 1024 + c[:, None] * 32 + c[None, :]
            dblk = out_ptr + base + (j + c)[:, None] * n + (j + c)[None, :]
        # Sub-step 0: solve against dinv_0, then push into columns 32..128.
        t0 = _fdot(t0, dbase, TRANSLOAD, PREC)
        t1 -= _fdot(t0, dblk + 32 * n, TRANSLOAD, PREC)
        t2 -= _fdot(t0, dblk + 64 * n, TRANSLOAD, PREC)
        t3 -= _fdot(t0, dblk + 96 * n, TRANSLOAD, PREC)
        # Sub-step 1.
        t1 = _fdot(t1, dbase + dinv_stride, TRANSLOAD, PREC)
        t2 -= _fdot(t1, dblk + 64 * n + 32, TRANSLOAD, PREC)
        t3 -= _fdot(t1, dblk + 96 * n + 32, TRANSLOAD, PREC)
        # Sub-step 2.
        t2 = _fdot(t2, dbase + 2 * dinv_stride, TRANSLOAD, PREC)
        t3 -= _fdot(t2, dblk + 96 * n + 64, TRANSLOAD, PREC)
        # Sub-step 3.
        t3 = _fdot(t3, dbase + 3 * dinv_stride, TRANSLOAD, PREC)
        tl.store(p0, t0, mask=rmask[:, None])
        tl.store(p0 + 32, t1, mask=rmask[:, None])
        tl.store(p0 + 64, t2, mask=rmask[:, None])
        tl.store(p0 + 96, t3, mask=rmask[:, None])
        if MIRROR:
            # Zero the mirrored upper tile, exactly as `_panel_apply32` does,
            # so the eager first-touch path needs no separate clear pass.
            z = tl.zeros((32, TILE_R), dtype=tl.float32)
            m0 = (
                out_ptr + base + (j + c)[:, None] * n
                + (j + 128 + rr)[None, :]
            )
            tl.store(m0, z, mask=rmask[None, :])
            tl.store(m0 + 32 * n, z, mask=rmask[None, :])
            tl.store(m0 + 64 * n, z, mask=rmask[None, :])
            tl.store(m0 + 96 * n, z, mask=rmask[None, :])

    @triton.jit
    def _trailing_nb(
        out_ptr,
        src_ptr,
        n: tl.constexpr,
        j,
        remaining,
        NB: tl.constexpr,
        PREC: tl.constexpr,
        FP16_TRAILING: tl.constexpr,
        TILE: tl.constexpr,
        FIRST: tl.constexpr,
    ):
        """Rank-NB Schur update of the lower-triangular trailing tiles, run
        once per NB-wide panel (depth NB keeps tl.dot tensor-core efficient
        and cuts trailing read-modify-write traffic by NB/32 vs rank-32)."""
        tri = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        br = ((tl.sqrt(8.0 * tri + 1.0) - 1.0) * 0.5).to(tl.int32)
        bc = tri - br * (br + 1) // 2
        rows = br * TILE + tl.arange(0, TILE)
        cols = bc * TILE + tl.arange(0, TILE)
        d = tl.arange(0, NB)
        base = b * n * n
        li = tl.load(
            out_ptr + base + (j + NB + rows)[:, None] * n + (j + d)[None, :],
            mask=rows[:, None] < remaining,
            other=0.0,
        )
        lj = tl.load(
            out_ptr + base + (j + NB + cols)[:, None] * n + (j + d)[None, :],
            mask=cols[:, None] < remaining,
            other=0.0,
        )
        if FP16_TRAILING:
            prod = tl.dot(
                li.to(tl.float16),
                tl.trans(lj.to(tl.float16)),
                out_dtype=tl.float32,
            )
        else:
            prod = tl.dot(
                li, tl.trans(lj), input_precision=PREC, out_dtype=tl.float32
            )
        t_off = (j + NB + rows)[:, None] * n + (j + NB + cols)[None, :]
        t_ptrs = out_ptr + base + t_off
        valid = (rows[:, None] < remaining) & (cols[None, :] < remaining)
        valid = valid & ((br != bc) | (cols[None, :] <= rows[:, None]))
        if FIRST:
            t = tl.load(src_ptr + base + t_off, mask=valid, other=0.0)
        else:
            t = tl.load(t_ptrs, mask=valid, other=0.0)
        tl.store(t_ptrs, t - prod, mask=valid)

    # (batch, n) -> (panel_prec, trailing_prec) for the two-level blocked
    # path. tf32x3 keeps tensor cores with near-FP32 accuracy where the
    # n-scaled tolerance is tight; plain tf32 is enough from n=1024 up.
    # (batch, n) -> (panel_prec, trailing_prec, trailing_tile, mode,
    # fp16_trailing). The last value is a compile-time signal: the lone
    # measured regression keeps its ranked TF32 trailing update.
    # "eager" = first-touch launches reading the live input, no graph, no
    # copy-in/clone-out — a win only where per-launch GPU time far exceeds
    # enqueue time (the bandwidth-bound high-batch shapes).
    _SPLIT32_SHAPES = {
        # exp 030: 256x128 moves off graph-replayed vendor factorization onto
        # the split32 chain (10 kernel launches, paired 1.10x). 1024x64 was
        # measured a wash (0.998x) and keeps its ranked vendor route. tf32x3
        # both levels: the n-scaled tolerance is tightest at small n.
        (256, 128): ("tf32x3", "tf32x3", 128, "graph", True),
        (64, 256): ("tf32x3", "tf32x3", 128, "graph", True),
        (16, 512): ("tf32x3", "tf32x3", 128, "graph", True),
        (640, 512): ("tf32x3", "tf32", 128, "eager", True),
        # exp 033 (lever L4): plain tf32 (1-pass) panels replace tf32x3 (3-pass)
        # on the large-n split32 shapes. The reconstruction gate is 20*n*eps*|A|,
        # which grows with n, so tf32's lower per-dot accuracy is safe here:
        # paired 1.057-1.072x with the worst family residual 8.13/20 (>=2.4x
        # headroom). At smaller n the same change either fails (256x128 dense) or
        # eats the tolerance (64x256 rowscale 19/20), so those keep tf32x3.
        (4, 1024): ("tf32", "tf32", 128, "graph", True),
        (60, 1024): ("tf32", "tf32", 128, "eager", False),
        (8, 2048): ("tf32", "tf32", 128, "graph", True),
    }
    _SPLIT32_TILE = 128
    _SPLIT32_NB = 128

    # Experiment 032 (lever L2): per-shape, non-uniform panel-width schedules.
    #
    # Until now every split32 shape factored with one uniform panel width of
    # _SPLIT32_NB = 128, from the first panel to the last. The trailing block
    # shrinks monotonically as the panel walks the diagonal, so a fixed width
    # is necessarily mistuned at one end: late panels pay a 128-wide panel
    # factor whose rank-128 trailing update no longer has enough trailing rows
    # to amortize it.
    #
    # Each entry maps (batch, n) -> a tuple of panel widths that must sum to n.
    # CONSTRAINT: every width must be a power of two >= 32, because
    # _trailing_nb does `d = tl.arange(0, NB)` and Triton requires a
    # power-of-two arange bound. This is why the schedules below are staircases
    # like (128, 128, 128, 64, 32, 32) rather than gau.nernst's qr_v2
    # (96, 96, 64, 32, 32, 192) -- expressing non-power-of-two widths would
    # need a padded+masked load in _trailing_nb, which wastes MMA lanes and is
    # a separate experiment.
    #
    # A shape absent from this map keeps the uniform _SPLIT32_NB schedule, so
    # this table is strict opt-in: an absent shape emits the exact launch
    # sequence of ranked #883174.
    #
    # Experiment 032 result: of the seven split32 shapes, panel width is only a
    # live axis at 8x2048. Every candidate was measured paired same-process vs
    # #883174 on a B200 (drift <0.9%):
    #   - Tail taper (variant A, e.g. (128,)*15+(64,32,32)) regressed EVERY
    #     shape (256x128 0.925x, 640x512 0.981x, 8x2048 0.998x): each extra
    #     panel pays the ~16us serial-tile-loop launch floor (S27/S29) while its
    #     tapered trailing corner processes almost no data.
    #   - Wide uniform NB=256 (variant W) spilled _trailing_nb's [TILE x NB]
    #     tile: catastrophic on the eager-mode shapes (60x1024 0.286x, 640x512
    #     0.837x) and net-negative on the small graph shapes -- EXCEPT 8x2048,
    #     the one shape with both the most panels (16->8, half the launches) and
    #     enough per-panel tensor-core compute to hide the spill: 1.031x.
    #   - NB=512 on 8x2048 (variants X/X2) overshot: the spill grows faster than
    #     the launch saving (0.972x / 0.983x). NB=256 is the sweet spot.
    # Net: enroll 8x2048 only; the other six keep uniform-128.
    _SPLIT32_NB_SCHEDULE = {
        (8, 2048): (256,) * 8,
    }

    def _nb_schedule(batch, n):
        """Panel-width schedule for one shape. Falls back to the uniform
        _SPLIT32_NB schedule used by ranked #883174.

        Experiment 047: a fused shape must use uniform 128-wide panels. The
        fused panel solves one 128-wide block column against the diagonal
        block above it; a wider panel would need an extra rank-128 Schur
        update between its two halves, which is exactly nb=128 again."""
        if (batch, n) in _FUSED_PANEL_SHAPES:
            return (128,) * (n // 128)
        sched = _SPLIT32_NB_SCHEDULE.get((batch, n))
        if sched is None:
            nb = _SPLIT32_NB
            full, rem = divmod(n, nb)
            sched = (nb,) * full + ((rem,) if rem else ())
        return sched

    def _validate_nb_schedules():
        """Free gate: every declared schedule must sum to n and use only
        power-of-two widths >= 32. Runs at import so a malformed schedule
        fails before any GPU time is spent."""
        for (batch, n), sched in _SPLIT32_NB_SCHEDULE.items():
            total = sum(sched)
            if total != n:
                raise ValueError(
                    f"nb schedule for {(batch, n)} sums to {total}, expected {n}"
                )
            for nb in sched:
                if nb < 32 or (nb & (nb - 1)) != 0:
                    raise ValueError(
                        f"nb schedule for {(batch, n)} has width {nb}; "
                        "widths must be powers of two >= 32 "
                        "(tl.arange bound in _trailing_nb)"
                    )

    _validate_nb_schedules()
    # Experiment 021 final: retain the three stable transfer winners alongside
    # experiment 020's two ranked routes. The 60x1024 transfer was positive in
    # the isolated probe but regressed in the full grid, so it stays on the
    # exact #882927 128x128 panel-inner specialization.
    _PANEL_INNER_SUBTILE64_SHAPES = {
        (256, 128),
        (64, 256),
        (16, 512),
        (640, 512),
        (4, 1024),
        (8, 2048),
    }

    def _split32_launch(
        work,
        dinv,
        panel_prec,
        trailing_prec,
        trailing_tile,
        fp16_trailing,
        src=None,
    ):
        """Launch the full two-level blocked factorization writing into
        `work`. With src=None the factorization runs in place on `work`
        (graph mode: the caller copies the input in first). With src set,
        the first-touch launches read directly from `src` and everything is
        written to `work`, so no copy-in or clone-out pass is needed (eager
        mode for the bandwidth-bound shapes). The mirrored zero-fill in the
        panel kernel plus the zeroed diagonal-block upper make a separate
        clear pass unnecessary in both modes."""
        global _MICRO32_HITS, _BMM_SCHUR_HITS, _BMM_TRAILING_HITS
        global _FUSED_PANEL_HITS
        batch, n, _ = work.shape
        bmm_trailing = (batch, n) in _BMM_TRAILING_SHAPES
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        if bmm_trailing:
            torch.backends.cuda.matmul.allow_tf32 = True
        cuda_micro = _MICRO32 is not None and (batch, n) in _MICRO32_SHAPES
        bmm_schur = (batch, n) in _BMM_SCHUR_SHAPES
        tile = _SPLIT32_TILE
        fused_cfg = _FUSED_PANEL_SHAPES.get((batch, n))
        ft = src is not None
        if not ft:
            src = work
        j = 0
        for nb in _nb_schedule(batch, n):
            panel_end = min(j + nb, n)
            fused = fused_cfg is not None and panel_end - j == 128
            for k in range(j, panel_end, 32):
                slot = (k - j) // 32 if fused else 0
                if cuda_micro:
                    _MICRO32_HITS += 1
                    _MICRO32.micro32_launch(
                        src, work, dinv[slot], n, k,
                        1 if (ft and k == 0) else 0,
                    )
                else:
                    _micro_potrf_gj32[(batch,)](
                        work,
                        dinv[slot],
                        src,
                        n=n,
                        k=k,
                        FIRST=ft and k == 0,
                        RECIPROCAL_SOLVE=fp16_trailing,
                        num_warps=1,
                    )
                # With the fused panel the per-sub-step apply/inner launches
                # are restricted to the diagonal block; every row below it is
                # handled once, after the block is complete.
                remaining = (panel_end if fused else n) - k - 32
                if remaining <= 0:
                    if fused:
                        continue
                    break
                if fused and fused_cfg[2]:
                    _diag_block_step[(batch,)](
                        work,
                        dinv[slot],
                        src,
                        n=n,
                        k=k,
                        nrows=remaining,
                        TILE=128,
                        PREC=panel_prec,
                        FIRST=ft and k == 0,
                        num_warps=8,
                    )
                    continue
                _panel_apply32[(triton.cdiv(remaining, tile), batch)](
                    work,
                    dinv[slot],
                    src,
                    n=n,
                    k=k,
                    remaining=remaining,
                    PREC=panel_prec,
                    TILE_R=tile,
                    FIRST=ft and k == 0,
                    num_warps=4,
                )
                width = panel_end - (k + 32)
                if width > 0 and bmm_schur:
                    _BMM_SCHUR_HITS += 1
                    below = k + 32
                    factor = work[:, below:, k:below]
                    update = factor[:, :width, :].transpose(1, 2)
                    target = work[:, below:, below:panel_end]
                    if ft and k == 0:
                        torch.baddbmm(
                            src[:, below:, below:panel_end], factor, update,
                            beta=1.0, alpha=-1.0, out=target,
                        )
                    else:
                        target.baddbmm_(factor, update, beta=1.0, alpha=-1.0)
                elif width > 0:
                    if (batch, n) in _PANEL_INNER_SUBTILE64_SHAPES:
                        ntiles_c = triton.cdiv(width, 64)
                        _panel_inner32_subtile64[
                            (
                                triton.cdiv(remaining, 64) * ntiles_c,
                                batch,
                            )
                        ](
                            work,
                            src,
                            n=n,
                            k=k,
                            width=width,
                            remaining=remaining,
                            PREC=panel_prec,
                            NTILES_C=ntiles_c,
                            FIRST=ft and k == 0,
                            num_warps=4,
                        )
                    else:
                        _panel_inner32[(triton.cdiv(remaining, tile), batch)](
                            work,
                            src,
                            n=n,
                            k=k,
                            width=width,
                            remaining=remaining,
                            PREC=panel_prec,
                            TILE_R=tile,
                            FIRST=ft and k == 0,
                            num_warps=4,
                        )
            rem_out = n - panel_end
            if fused and rem_out > 0:
                _FUSED_PANEL_HITS += 1
                ftile, fwarps = fused_cfg[0], fused_cfg[1]
                _panel_fused128[(triton.cdiv(rem_out, ftile), batch)](
                    work,
                    dinv,
                    src,
                    n=n,
                    j=j,
                    nrows=rem_out,
                    dinv_stride=batch * 1024,
                    TILE_R=ftile,
                    PREC=panel_prec,
                    FIRST=ft and j == 0,
                    TRANSLOAD=1,
                    MIRROR=1,
                    num_warps=fwarps,
                )
            if rem_out > 0 and bmm_trailing and not (ft and j == 0):
                _BMM_TRAILING_HITS += 1
                block = work[:, panel_end:, j:panel_end]
                work[:, panel_end:, panel_end:].baddbmm_(
                    block, block.transpose(1, 2), beta=1.0, alpha=-1.0
                )
            elif rem_out > 0 and bmm_schur:
                _BMM_SCHUR_HITS += 1
                block = work[:, panel_end:, j:panel_end]
                target = work[:, panel_end:, panel_end:]
                if ft and j == 0:
                    torch.baddbmm(
                        src[:, panel_end:, panel_end:], block,
                        block.transpose(1, 2), beta=1.0, alpha=-1.0,
                        out=target,
                    )
                else:
                    target.baddbmm_(
                        block, block.transpose(1, 2), beta=1.0, alpha=-1.0
                    )
            elif rem_out > 0:
                tr = triton.cdiv(rem_out, trailing_tile)
                _trailing_nb[(tr * (tr + 1) // 2, batch)](
                    work,
                    src,
                    n=n,
                    j=j,
                    remaining=rem_out,
                    NB=nb,
                    PREC=trailing_prec,
                    FP16_TRAILING=fp16_trailing,
                    TILE=trailing_tile,
                    FIRST=ft and j == 0,
                    num_warps=8,
                    num_stages=3,
                )
            j = panel_end
        if bmm_trailing:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    _SPLIT32_GRAPHS = {}
    _SPLIT32_DINV = {}

    def _dinv_slots(batch, n):
        """Number of 32x32 diagonal inverses that must stay live at once.
        The fused panel consumes all four inverses of a 128-wide block after
        the block is finished, so they cannot share one buffer."""
        return 4 if (batch, n) in _FUSED_PANEL_SHAPES else 1

    def _split32_factor(data: torch.Tensor) -> torch.Tensor:
        batch, n, _ = data.shape
        panel_prec, trailing_prec, trailing_tile, mode, fp16_trailing = (
            _SPLIT32_SHAPES[(batch, n)]
        )
        slots = _dinv_slots(batch, n)
        data = data.contiguous()

        if mode == "eager":
            out = torch.empty_like(data)
            dinv = _SPLIT32_DINV.get((batch, slots))
            if dinv is None:
                dinv = torch.empty(
                    slots, batch, 32, 32, device=data.device,
                    dtype=torch.float32,
                )
                _SPLIT32_DINV[(batch, slots)] = dinv
            _split32_launch(
                out,
                dinv,
                panel_prec,
                trailing_prec,
                trailing_tile,
                fp16_trailing,
                src=data,
            )
            return out

        key = (batch, n)
        entry = _SPLIT32_GRAPHS.get(key)
        if entry is None:
            try:
                work = torch.empty_like(data)
                dinv = torch.empty(
                    slots, batch, 32, 32, device=data.device,
                    dtype=torch.float32,
                )
                for _ in range(2):
                    work.copy_(data)
                    _split32_launch(
                        work,
                        dinv,
                        panel_prec,
                        trailing_prec,
                        trailing_tile,
                        fp16_trailing,
                    )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=_shared_graph_pool()):
                    _split32_launch(
                        work,
                        dinv,
                        panel_prec,
                        trailing_prec,
                        trailing_tile,
                        fp16_trailing,
                    )
                # Keep BOTH buffers alive: the graph nodes hold raw device
                # pointers into them, so dropping either is a use-after-free
                # on every subsequent replay.
                entry = (graph, work, dinv)
                _SPLIT32_GRAPHS[key] = entry
            except Exception:
                _SPLIT32_GRAPHS[key] = False
                raise
        if entry is False:
            work = data.clone()
            dinv = torch.empty(
                slots, batch, 32, 32, device=data.device, dtype=torch.float32
            )
            _split32_launch(
                work,
                dinv,
                panel_prec,
                trailing_prec,
                trailing_tile,
                fp16_trailing,
            )
            return work
        graph, work, _dinv = entry
        work.copy_(data)
        graph.replay()
        return work.clone()


    def _triton_cholesky_8x2048(data: torch.Tensor) -> torch.Tensor:
        out = data.contiguous().clone()
        batch, n, _ = out.shape
        for k in range(0, n, _BK_8X2048):
            _diag_factor_8x2048[(batch,)](
                out,
                n=n,
                k=k,
                BK_CONST=_BK_8X2048,
                num_warps=8,
            )
            remaining = n - k - _BK_8X2048
            if remaining <= 0:
                break
            panel_tiles = triton.cdiv(remaining, _BK_8X2048)
            _panel_solve_8x2048[(panel_tiles, batch)](
                out,
                n=n,
                k=k,
                remaining=remaining,
                BK_CONST=_BK_8X2048,
                num_warps=8,
            )
            update_tiles = triton.cdiv(remaining, _UPDATE_TILE_8X2048)
            triangular_tiles = update_tiles * (update_tiles + 1) // 2
            _lower_schur_8x2048[(triangular_tiles, batch)](
                out,
                n=n,
                k=k,
                remaining=remaining,
                BK_CONST=_BK_8X2048,
                TILE=_UPDATE_TILE_8X2048,
                num_warps=8,
                num_stages=3,
            )

        total = batch * n * n
        clear_grid = 4096
        _clear_upper_8x2048[(clear_grid,)](
            out,
            total=total,
            n=n,
            BLOCK=256,
            GRID=clear_grid,
            num_warps=8,
        )
        return out


# ---------------------------------------------------------------------------
# Exact graph-replay paths for two overhead-bound ranked shapes.
# ---------------------------------------------------------------------------
_GRAPH_POOL = None


def _shared_graph_pool():
    """All CUDA graph captures in this module share one memory pool. With
    separate private pools, a capture that follows an earlier capture in the
    same process produced deterministically corrupted replays for the earlier
    pattern (measured: 256x128 after the 1024x64 capture, relative residual
    1.42); one shared pool is the documented multi-capture arrangement."""
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


_GRAPH_16X512 = None
_GRAPH_INPUT_16X512 = None
_GRAPH_OUTPUT_16X512 = None
_GRAPH_ERROR_16X512 = None

_GRAPH_256X128 = None
_GRAPH_ERROR_256X128 = None


def _graph_cholesky_16x512(data: torch.Tensor) -> torch.Tensor:
    global _GRAPH_16X512, _GRAPH_INPUT_16X512, _GRAPH_OUTPUT_16X512
    global _GRAPH_ERROR_16X512

    if _GRAPH_16X512 is None and _GRAPH_ERROR_16X512 is None:
        try:
            static_input = torch.empty_like(data)
            static_input.copy_(data)
            for _ in range(3):
                torch.linalg.cholesky_ex(
                    static_input, check_errors=False
                ).L
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=_shared_graph_pool()):
                static_output = torch.linalg.cholesky_ex(
                    static_input, check_errors=False
                ).L
            graph.replay()
            _GRAPH_INPUT_16X512 = static_input
            _GRAPH_OUTPUT_16X512 = static_output
            _GRAPH_16X512 = graph
            return static_output.clone()
        except Exception as exc:  # pragma: no cover
            _GRAPH_ERROR_16X512 = repr(exc)
            return torch.linalg.cholesky_ex(data, check_errors=False).L

    if _GRAPH_16X512 is None:
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    _GRAPH_INPUT_16X512.copy_(data)
    _GRAPH_16X512.replay()
    return _GRAPH_OUTPUT_16X512.clone()


def _graph_cholesky_256x128(data: torch.Tensor) -> torch.Tensor:
    # Experiment 015: converted from make_graphed_callables to the same
    # manual static-buffer capture pattern as the 16x512 path. The callable
    # version produced corrupted replays once another manual graph (the new
    # 1024x64 path) had been captured earlier in the process; the manual
    # pattern is measured clean in that ordering with identical numerics.
    global _GRAPH_256X128, _GRAPH_ERROR_256X128
    if _GRAPH_256X128 is None and _GRAPH_ERROR_256X128 is None:
        try:
            static_input = torch.empty_like(data.contiguous())
            static_input.copy_(data)
            for _ in range(3):
                torch.linalg.cholesky_ex(static_input, check_errors=False).L
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=_shared_graph_pool()):
                static_output = torch.linalg.cholesky_ex(
                    static_input, check_errors=False
                ).L
            graph.replay()
            torch.cuda.synchronize()
            _GRAPH_256X128 = (graph, static_input, static_output)
        except Exception as exc:  # pragma: no cover
            _GRAPH_ERROR_256X128 = repr(exc)
            _GRAPH_256X128 = False

    if _GRAPH_256X128 is False or _GRAPH_256X128 is None:
        return torch.linalg.cholesky_ex(data, check_errors=False).L
    graph, static_input, static_output = _GRAPH_256X128
    static_input.copy_(data)
    graph.replay()
    return static_output.clone()


# ---------------------------------------------------------------------------
# Large single-matrix left-looking paths (experiment 012).
# ---------------------------------------------------------------------------
_FUSED_CTA_HITS = 0
_FUSED_CTA_FALLBACKS = 0
_FUSED_CTA_ERROR = None

_GRAPH_SP_HITS = 0
_GRAPH_SP_FALLBACKS = 0
_GRAPH_SP_ERROR = None

_SP_STATE = {}


def _graph_cholesky_1024x64(data):
    """Graph-replayed exact cuSOLVER factorization for (1024, 64): identical
    numerics to the shipped default, minus the per-call launch train."""
    global _GRAPH_SP_HITS, _GRAPH_SP_FALLBACKS, _GRAPH_SP_ERROR

    key = (1024, 64)
    state = _SP_STATE.get(key)
    if state is None:
        try:
            static_in = torch.empty_like(data.contiguous())
            static_in.copy_(data)
            for _ in range(3):
                torch.linalg.cholesky_ex(static_in, check_errors=False).L
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=_shared_graph_pool()):
                static_out = torch.linalg.cholesky_ex(
                    static_in, check_errors=False
                ).L
            graph.replay()
            torch.cuda.synchronize()
            state = (graph, static_in, static_out)
            _SP_STATE[key] = state
        except Exception as exc:  # pragma: no cover
            _GRAPH_SP_ERROR = repr(exc)
            _SP_STATE[key] = False
            _GRAPH_SP_FALLBACKS += 1
            return None

    if state is False:
        _GRAPH_SP_FALLBACKS += 1
        return None

    graph, static_in, static_out = state
    static_in.copy_(data)
    graph.replay()
    _GRAPH_SP_HITS += 1
    return static_out.clone()

_LEFT_16384_HITS = 0
_LEFT_32768_HITS = 0
_LEFT_32768_ERROR = None
_LEFT_LARGE_FALLBACKS = 0
_FUSED_E4M3_QUANT_HITS = 0
_FUSED_E4M3_AMAX_HITS = 0
_FUSED_E4M3_QUANT_ERROR = None


def _clear_upper_large(matrix: torch.Tensor) -> torch.Tensor:
    if not _HAVE_TRITON:
        return torch.tril(matrix)
    grid = 4096
    _clear_upper_8x2048[(grid,)](
        matrix,
        total=matrix.numel(),
        n=matrix.shape[0],
        BLOCK=256,
        GRID=grid,
        num_warps=8,
    )
    return matrix


def _left_looking_cholesky_16384(mat: torch.Tensor) -> torch.Tensor:
    global _LEFT_16384_HITS

    nb = 2048
    n = mat.shape[0]
    a = mat.clone()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = a[k : k + kb, k : k + kb]
            if k:
                left = a[k : k + kb, :k]
                diagonal.addmm_(
                    left,
                    left.transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            diagonal_factor = torch.linalg.cholesky_ex(
                diagonal, check_errors=False
            ).L
            a[k : k + kb, k : k + kb] = diagonal_factor
            j = k + kb
            if j >= n:
                break
            panel = a[j:, k : k + kb]
            if k:
                panel.addmm_(
                    a[j:, :k],
                    a[k : k + kb, :k].transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            solved = torch.linalg.solve_triangular(
                diagonal_factor.transpose(-1, -2),
                panel,
                upper=True,
                left=False,
            )
            a[j:, k : k + kb] = solved
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _LEFT_16384_HITS += 1
    return _clear_upper_large(a)


def _scaled_mm_fp8_32768(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    scale_lhs: torch.Tensor,
    scale_rhs: torch.Tensor,
) -> torch.Tensor:
    try:
        result = torch._scaled_mm(
            lhs,
            rhs,
            scale_a=scale_lhs,
            scale_b=scale_rhs,
            out_dtype=torch.float32,
            use_fast_accum=True,
        )
    except TypeError:
        result = torch._scaled_mm(
            lhs,
            rhs,
            scale_a=scale_lhs,
            scale_b=scale_rhs,
            out_dtype=torch.float32,
        )
    return result[0] if isinstance(result, tuple) else result


def _fp8_product_32768(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    global _FUSED_E4M3_QUANT_HITS, _FUSED_E4M3_AMAX_HITS
    global _FUSED_E4M3_QUANT_ERROR

    max_value = torch.finfo(torch.float8_e4m3fn).max
    reduction_block = 1024
    lhs_tiles = triton.cdiv(lhs.shape[1], reduction_block)
    rhs_tiles = triton.cdiv(rhs.shape[1], reduction_block)
    lhs_programs = lhs.shape[0] * lhs_tiles
    rhs_programs = rhs.shape[0] * rhs_tiles
    lhs_partial = torch.empty(
        lhs_programs, device=lhs.device, dtype=torch.float32
    )
    rhs_partial = torch.empty(
        rhs_programs, device=rhs.device, dtype=torch.float32
    )
    reduction_grid = (max(lhs_programs, rhs_programs),)
    _dual_tiled_amax_e4m3_32768[reduction_grid](
        lhs,
        rhs,
        lhs_partial,
        rhs_partial,
        lhs.shape[0],
        lhs.shape[1],
        rhs.shape[0],
        rhs.shape[1],
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        lhs_tiles,
        rhs_tiles,
        lhs_programs,
        rhs_programs,
        BLOCK=reduction_block,
        num_warps=8,
    )
    _FUSED_E4M3_AMAX_HITS += 1
    scale_lhs = (max_value / lhs_partial.amax().clamp_min(2.0**-24)).float()
    scale_rhs = (max_value / rhs_partial.amax().clamp_min(2.0**-24)).float()
    quantized_lhs = torch.empty(
        lhs.shape,
        device=lhs.device,
        dtype=torch.float8_e4m3fn,
    )
    quantized_rhs = torch.empty(
        rhs.shape,
        device=rhs.device,
        dtype=torch.float8_e4m3fn,
    )
    block = 1024
    grid = (
        triton.cdiv(max(lhs.numel(), rhs.numel()), block),
    )
    try:
        _dual_scale_cast_e4m3_32768[grid](
            lhs,
            rhs,
            quantized_lhs,
            quantized_rhs,
            scale_lhs,
            scale_rhs,
            lhs.numel(),
            rhs.numel(),
            lhs.shape[1],
            rhs.shape[1],
            lhs.stride(0),
            lhs.stride(1),
            rhs.stride(0),
            rhs.stride(1),
            BLOCK=block,
            num_warps=8,
        )
        _FUSED_E4M3_QUANT_HITS += 1
        _FUSED_E4M3_QUANT_ERROR = None
    except Exception as exc:
        _FUSED_E4M3_QUANT_ERROR = repr(exc)
        raise
    return _scaled_mm_fp8_32768(
        quantized_lhs,
        quantized_rhs,
        scale_lhs.reciprocal(),
        scale_rhs.reciprocal(),
    )


# ---------------------------------------------------------------------------
# Experiment 034: MXFP8 block-scaled panel products (Blackwell tcgen05).
# ---------------------------------------------------------------------------
_MXFP8_HITS = 0
_MXFP8_ERROR = None
_MXFP8_PTX = None
_MXFP8_BACKEND = "scaled_mm_mx"

_MX_QUANT_BLOCK_M = 32
_MX_QUANT_BLOCK_K = 128
_MX_GEMM_BLOCK_M = 128
_MX_GEMM_BLOCK_N = 128
_MX_GEMM_BLOCK_K = 128
_MX_GEMM_WARPS = 8
_MX_GEMM_STAGES = 3


def _mx_quant_e4m3(x: torch.Tensor):
    """One fused pass: fp32 (rows, columns) view -> contiguous e4m3 values +
    per-32-element e8m0 scale bytes. No global amax, no host round-trip."""
    rows, columns = x.shape
    q = torch.empty(rows, columns, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(rows, columns // 32, dtype=torch.uint8, device=x.device)
    _mx_quant_e4m3_kernel[
        (rows // _MX_QUANT_BLOCK_M, columns // _MX_QUANT_BLOCK_K)
    ](
        x,
        q,
        s,
        x.stride(0),
        x.stride(1),
        columns,
        BLOCK_M=_MX_QUANT_BLOCK_M,
        BLOCK_K=_MX_QUANT_BLOCK_K,
    )
    return q, s


def _mx_quant_e4m3_blocked(x: torch.Tensor):
    """One fused pass: fp32 (rows, columns) view -> contiguous e4m3 values +
    e8m0 scale bytes already in the 128x4 blocked layout `torch._scaled_mm`
    wants. Requires rows % 128 == 0 and columns % 128 == 0."""
    rows, columns = x.shape
    q = torch.empty(rows, columns, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(
        rows * (columns // 32), dtype=torch.uint8, device=x.device
    )
    _mx_quant_e4m3_blocked_kernel[
        (rows // _MX_QUANT_BLOCK_M, columns // _MX_QUANT_BLOCK_K)
    ](
        x,
        q,
        s,
        x.stride(0),
        x.stride(1),
        columns,
        BLOCK_M=_MX_QUANT_BLOCK_M,
        BLOCK_K=_MX_QUANT_BLOCK_K,
    )
    return q, s.view(torch.float8_e8m0fnu)


def _mxfp8_panel_update(
    out: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor
) -> None:
    """out -= lhs @ rhs^T on MXFP8 block-scaled tensor cores (experiment 034
    V2). Both operands are quantized in one fused pass each, emitting e8m0
    scales straight into the blocked layout, then multiplied by cuBLAS's
    tuned block-scaled MX GEMM via `torch._scaled_mm` (V1's hand-written
    `tl.dot_scaled` kernel measured 0.65x this path). lhs (M, K) and rhs
    (N, K) may be strided factor views; out must be contiguous (M, N). All
    sizes in the 32768 left-looking schedule are multiples of nb=4096, so
    exact tiling always holds; anything else raises and the caller's existing
    fallback chain takes over."""
    global _MXFP8_HITS
    m_rows, k_cols = lhs.shape
    n_rows = rhs.shape[0]
    if (
        m_rows % 128
        or n_rows % 128
        or k_cols % 128
        or m_rows % _MX_QUANT_BLOCK_M
        or n_rows % _MX_QUANT_BLOCK_M
        or k_cols % _MX_QUANT_BLOCK_K
    ):
        raise RuntimeError("mxfp8 tiling mismatch")
    q_lhs, s_lhs = _mx_quant_e4m3_blocked(lhs)
    q_rhs, s_rhs = _mx_quant_e4m3_blocked(rhs)
    out.sub_(
        torch._scaled_mm(
            q_lhs,
            q_rhs.t(),
            scale_a=s_lhs,
            scale_b=s_rhs,
            out_dtype=torch.float32,
        )
    )
    _MXFP8_HITS += 1


def _left_looking_cholesky_32768(mat: torch.Tensor) -> torch.Tensor:
    global _LEFT_32768_HITS

    nb = 4096
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = mat[k : k + kb, k : k + kb].clone()
            if k:
                previous_row = factor[k : k + kb, :k]
                diagonal.addmm_(
                    previous_row,
                    previous_row.transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            diagonal_factor = torch.linalg.cholesky_ex(
                diagonal, check_errors=False
            ).L
            factor[k : k + kb, k : k + kb] = diagonal_factor
            j = k + kb
            if j >= n:
                break
            panel = mat[j:, k : k + kb].clone()
            if k:
                panel.sub_(
                    _fp8_product_32768(
                        factor[j:, :k],
                        factor[k : k + kb, :k].transpose(-1, -2),
                    )
                )
            identity = torch.eye(
                kb, device=mat.device, dtype=mat.dtype
            )
            inverse_transpose = torch.linalg.solve_triangular(
                diagonal_factor.transpose(-1, -2),
                identity,
                upper=True,
            )
            factor[j:, k : k + kb] = panel @ inverse_transpose
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _LEFT_32768_HITS += 1
    return factor


# ---------------------------------------------------------------------------
# Small-batch / large-n path (experiment 004, region trimmed by exp 005).
# ---------------------------------------------------------------------------
def _loop_cholesky(data: torch.Tensor) -> torch.Tensor:
    """Sequential per-matrix single-matrix potrf, then stack. Avoids the slow
    batched cuSOLVER path for few-but-large matrices."""
    batch = data.shape[0]
    return torch.stack(
        [
            torch.linalg.cholesky_ex(data[i], check_errors=False).L
            for i in range(batch)
        ]
    )


# ---------------------------------------------------------------------------
# Large single-matrix path (experiments 006 + 008): blocked right-looking
# Cholesky with a fused in-place TF32 trailing update. Diagonal block + panel
# solve stay FP32.
# ---------------------------------------------------------------------------
def _blocked_cholesky_tf32(mat: torch.Tensor, nb: int) -> torch.Tensor:
    """Right-looking blocked Cholesky of a single (n, n) FP32 SPD matrix.

    The trailing Schur update (the O(n^3) cost) runs on tensor cores in TF32;
    the diagonal block factorization and the panel triangular solve stay FP32.
    Returns an FP32 lower-triangular factor. Default-queue only.
    """
    a = mat.clone()
    n = a.shape[0]
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            a11 = a[k : k + kb, k : k + kb]
            l11 = torch.linalg.cholesky_ex(a11, check_errors=False).L
            a[k : k + kb, k : k + kb] = l11
            j = k + kb
            if j >= n:
                break
            a21 = a[j:, k : k + kb]
            # Solve L21 @ L11^T = A21 for the panel factor (FP32 TRSM).
            l21 = torch.linalg.solve_triangular(
                l11.transpose(-1, -2), a21, upper=True, left=False
            )
            a[j:, k : k + kb] = l21
            # Fused trailing Schur update on TF32 tensor cores (FP32 accumulate).
            # Writing directly into the strided trailing view avoids materializing
            # a full product followed by a separate subtraction kernel.
            a[j:, j:].addmm_(
                l21, l21.transpose(-1, -2), beta=1.0, alpha=-1.0
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return torch.tril(a)

# ---------------------------------------------------------------------------
# Experiment 016a: generalized large single-matrix left-looking path.
# ---------------------------------------------------------------------------
import math as _math

_LARGE_FP8_HITS = 0
_LARGE_FP8_FALLBACKS = 0
_LARGE_FP8_ERROR = None

_LARGE_CFG = {
    8192: dict(nb=2048, panel_mode="tf32", diag_mode="tf32", rec_inv=False, shadow=False),
    16384: dict(nb=2048, panel_mode="tf32", diag_mode="tf32", rec_inv=True, shadow=False),
    # exp 034: MXFP8 block-scaled panel products (single-pass per-32-block
    # quantization + tcgen05 block-scaled MMA) replace the exp-014 per-tensor
    # fp8 pipeline. Requires Triton; _left_looking_large raises without it and
    # custom_kernel's existing fallback chain (exp-013 fp8 path) takes over.
    # exp 061: `path` selects which 32768-only driver custom_kernel calls.
    # "exp058" is the ranked blocked-inverse loop; "exp061_v2" adds the merged
    # MXFP8 block-column update on top of V1's Triton block moves.
    # No other n reads this key, and `_left_looking_large` is untouched.
    32768: dict(
        nb=4096,
        panel_mode="mxfp8",
        diag_mode="tf32",
        rec_inv=True,
        shadow=False,
        path="exp061_v2",
    ),
}


def _tri_inv_recursive(lower: torch.Tensor, base: int = 512) -> torch.Tensor:
    """Explicit inverse of a lower-triangular factor by recursive 2x2
    blocking: inv([[A,0],[B,C]]) = [[Ai,0],[-Ci@B@Ai, Ci]]. The combines are
    plain GEMMs (TF32 tensor cores under the caller's allow_tf32), replacing
    the launch- and TRSM-bound solve_triangular against identity."""
    n = lower.shape[0]
    if n <= base:
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    m = n // 2
    inv11 = _tri_inv_recursive(lower[:m, :m], base)
    inv22 = _tri_inv_recursive(lower[m:, m:], base)
    out = torch.zeros_like(lower)
    out[:m, :m] = inv11
    out[m:, m:] = inv22
    out[m:, :m] = -(inv22 @ (lower[m:, :m] @ inv11))
    return out


def _shadow_product(
    shadow: torch.Tensor,
    r0: int,
    r1: int,
    k: int,
    t0: int,
    t1: int,
    decode: torch.Tensor,
) -> torch.Tensor:
    """shadow[r0:r1, :k] @ shadow[t0:t1, :k]^T from the persistent FP8 copy
    of the factor: no per-panel amax, no re-quantization of the frontier."""
    lhs = shadow[r0:r1, :k].contiguous()
    rhs = shadow[t0:t1, :k].t().contiguous()
    return _scaled_mm_fp8_32768(lhs, rhs, decode, decode)


def _left_looking_large(
    mat: torch.Tensor,
    nb: int,
    panel_mode: str,
    diag_mode: str,
    rec_inv: bool,
    shadow: bool,
) -> torch.Tensor:
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    shadow_buf = None
    decode = None
    scale_val = None
    if shadow:
        diag_in = mat.diagonal()
        dmax = float(diag_in.max().item())
        dmin = float(diag_in.min().item())
        # Fixed-scale quantization is only sound when the diagonal dynamic
        # range is modest (|L_ij| <= sqrt(max_ii A_ii), small entries must
        # not underflow). Ill-conditioned families take the shipped path.
        if not (dmin > 0.0 and dmax > 0.0) or dmax / dmin > 1.0e4:
            raise RuntimeError("large-path dynamic-range guard")
        scale_val = 448.0 / _math.sqrt(dmax)
        decode = torch.full(
            (), 1.0 / scale_val, device=mat.device, dtype=torch.float32
        )
        shadow_buf = torch.empty(
            n, n, device=mat.device, dtype=torch.float8_e4m3fn
        )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = mat[k : k + kb, k : k + kb].clone()
            if k:
                if diag_mode == "fp8":
                    diagonal.sub_(
                        _shadow_product(
                            shadow_buf, k, k + kb, k, k, k + kb, decode
                        )
                    )
                else:
                    row = factor[k : k + kb, :k]
                    diagonal.addmm_(
                        row, row.transpose(-1, -2), beta=1.0, alpha=-1.0
                    )
            lkk = torch.linalg.cholesky_ex(diagonal, check_errors=False).L
            factor[k : k + kb, k : k + kb] = lkk
            j = k + kb
            if j >= n:
                break
            panel = mat[j:, k : k + kb].clone()
            if k:
                if panel_mode == "fp8_shadow":
                    panel.sub_(
                        _shadow_product(shadow_buf, j, n, k, k, k + kb, decode)
                    )
                elif panel_mode == "fp8":
                    panel.sub_(
                        _fp8_product_32768(
                            factor[j:, :k],
                            factor[k : k + kb, :k].transpose(-1, -2),
                        )
                    )
                elif panel_mode == "mxfp8":
                    _mxfp8_panel_update(
                        panel, factor[j:, :k], factor[k : k + kb, :k]
                    )
                else:
                    panel.addmm_(
                        factor[j:, :k],
                        factor[k : k + kb, :k].transpose(-1, -2),
                        beta=1.0,
                        alpha=-1.0,
                    )
            if rec_inv:
                inverse = _tri_inv_recursive(lkk)
                factor[j:, k : k + kb] = panel @ inverse.transpose(-1, -2)
            else:
                factor[j:, k : k + kb] = torch.linalg.solve_triangular(
                    lkk.transpose(-1, -2), panel, upper=True, left=False
                )
            if shadow:
                block = factor[k:n, k : k + kb]
                shadow_buf[k:n, k : k + kb].copy_(
                    (block * scale_val).to(torch.float8_e4m3fn)
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


# ---------------------------------------------------------------------------
# Experiments 057 V2 + 058 V4: two large-shape frontiers.
#
# 1x16384 removes every triangular-solve leaf from the 2048-wide inverse tree:
# reciprocal scalar leaves grow breadth-first through batched GEMMs. Its
# diagonal and below-diagonal regions are updated together as one block column.
#
# 1x32768 batches the independent 256x256 triangular-inverse leaves, grows the
# 4096-wide inverse breadth-first, and applies each panel with FP16 inputs plus
# FP32 accumulation/output. Its ranked MXFP8 updates and width remain unchanged.
#
# Both are enrolled inside the incumbent's existing large-shape safety chain.
# ---------------------------------------------------------------------------

_EXP057_V2_HITS = 0
_EXP057_V2_INVERSE_CALLS = 0
_EXP057_V4_TRITON_LEAF_HITS = 0
_EXP058_V1_HITS = 0


if _HAVE_TRITON:

    @triton.jit
    def _exp057_tri_inv_leaf32_kernel(
        lower_ptr,
        inverse_ptr,
        n: tl.constexpr,
        base: tl.constexpr,
    ):
        # One program solves one column of one 32x32 diagonal block.
        pid = tl.program_id(0)
        block = pid // base
        column = pid % base
        rows = tl.arange(0, base)
        row0 = block * base
        values = tl.zeros((base,), dtype=tl.float32)
        for row in tl.static_range(0, base):
            diagonal = tl.load(
                lower_ptr + (row0 + row) * n + row0 + row
            )
            coefficients = tl.load(
                lower_ptr + (row0 + row) * n + row0 + rows,
                mask=rows < row,
                other=0.0,
            )
            rhs = tl.where(column == row, 1.0, 0.0)
            solved = (
                rhs - tl.sum(coefficients * values, axis=0)
            ) / diagonal
            values = tl.where(rows == row, solved, values)
        tl.store(
            inverse_ptr + (row0 + rows) * n + row0 + column,
            values,
            mask=rows >= column,
        )

_EXP058_V1_INVERSE_CALLS = 0
_EXP058_V4_FP16_SOLVE_HITS = 0


def _trsm_free_inverse_16384(lower: torch.Tensor) -> torch.Tensor:
    global _EXP057_V2_INVERSE_CALLS, _EXP057_V4_TRITON_LEAF_HITS
    _EXP057_V2_INVERSE_CALLS += 1
    n = lower.shape[0]
    if not _HAVE_TRITON or n % 32 or (n & (n - 1)):
        raise RuntimeError("exp057 Triton base-32 inverse precondition failed")
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    count = n // 32
    _exp057_tri_inv_leaf32_kernel[(count * 32,)](
        lower,
        inverse,
        n=n,
        base=32,
        num_warps=1,
    )
    _EXP057_V4_TRITON_LEAF_HITS += 1
    size = 32
    while size < n:
        step = 2 * size
        shape = (n // step, size, size)
        stride = (step * n + step, n, 1)
        inv11 = inverse.as_strided(shape, stride, 0)
        inv22 = inverse.as_strided(shape, stride, size * n + size)
        low21 = lower.as_strided(shape, stride, size * n)
        inverse.as_strided(shape, stride, size * n).copy_(
            torch.bmm(inv22, torch.bmm(low21, inv11)).neg_()
        )
        size = step
    return inverse


def _factor_1x16384_trsm_free(mat: torch.Tensor) -> torch.Tensor:
    n = 16384
    nb = 2048
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            j = k + nb
            block = mat[k:, k:j].contiguous()
            if k:
                block.addmm_(
                    factor[k:, :k],
                    factor[k:j, :k].transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            lkk = torch.linalg.cholesky_ex(
                block[:nb],
                check_errors=False,
            ).L
            factor[k:j, k:j] = lkk
            if j >= n:
                break
            inverse = _trsm_free_inverse_16384(lkk)
            factor[j:, k:j] = block[nb:] @ inverse.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


# ---------------------------------------------------------------------------
# Experiment 061 (1x16384 only): overhead-and-precision rework of the exp-057
# trsm-free path. Two measured facts drive it (probe-01/probe-02, B200):
#
#   * cuSOLVER's diagonal potrf is serial-latency-bound at ~0.33us per row, so
#     its total cost is ~c*n whatever the block width. Every attempt to rebuild
#     it out of PyTorch ops measured 1.6-3.8x SLOWER than one cuSOLVER call, so
#     the diagonal is left exactly as shipped.
#   * Everything else is copy traffic and TF32 GEMM. Those are addressable:
#       - one reused block-column scratch and one reused inverse buffer instead
#         of a fresh allocation + fill per step;
#       - `torch.mm(..., out=<factor slice>)` instead of materializing the
#         product and copying it into the factor;
#       - no block-column copy at all on the first step, which has no update;
#       - a persistent FP16 shadow of the factor so the left-looking GEMM and
#         the inverse apply run on FP16 tensor cores with FP32 accumulation.
#
# FP16 and TF32 carry the same 11-bit effective mantissa, so the shadow trades
# no precision for ~1.7x measured GEMM throughput (736.9 -> 1262.7 TFLOP/s);
# only the exponent range narrows, and the shipped isfinite guard in
# `custom_kernel` already routes any overflow to the exact fallback chain.
# Measured residual is unchanged at 0.211 of the 20.0 budget.
# ---------------------------------------------------------------------------
_EXP061_16384_HITS = 0
_EXP061_16384_INVERSE_CALLS = 0


def _exp061_leaf_inverse(
    lower: torch.Tensor, inverse: torch.Tensor
) -> torch.Tensor:
    """exp-057's trsm-free triangular inverse writing into a caller-owned
    buffer. Every region that is ever non-zero is fully overwritten on each
    call (the base-32 leaves fill their lower triangles, the tree fills whole
    off-diagonal blocks), so the buffer only has to be zeroed once by the
    caller instead of once per block column. The `neg` is folded into the
    second product's alpha rather than run as its own pass."""
    global _EXP061_16384_INVERSE_CALLS
    _EXP061_16384_INVERSE_CALLS += 1
    n = lower.shape[0]
    if not _HAVE_TRITON or n % 32 or (n & (n - 1)):
        raise RuntimeError("exp061 Triton base-32 inverse precondition failed")
    lower = lower.contiguous()
    count = n // 32
    _exp057_tri_inv_leaf32_kernel[(count * 32,)](
        lower,
        inverse,
        n=n,
        base=32,
        num_warps=1,
    )
    size = 32
    while size < n:
        step = 2 * size
        shape = (n // step, size, size)
        stride = (step * n + step, n, 1)
        inv11 = inverse.as_strided(shape, stride, 0)
        inv22 = inverse.as_strided(shape, stride, size * n + size)
        low21 = lower.as_strided(shape, stride, size * n)
        dest = inverse.as_strided(shape, stride, size * n)
        torch.baddbmm(
            dest,
            inv22,
            torch.bmm(low21, inv11),
            beta=0.0,
            alpha=-1.0,
            out=dest,
        )
        size = step
    return inverse


def _exp061_factor_1x16384(mat: torch.Tensor) -> torch.Tensor:
    n = 16384
    nb = 2048
    factor = torch.zeros_like(mat)
    scratch = torch.empty(n - nb, nb, device=mat.device, dtype=mat.dtype)
    inverse = torch.zeros(nb, nb, device=mat.device, dtype=mat.dtype)
    shadow = torch.empty(n, n, device=mat.device, dtype=torch.float16)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            j = k + nb
            if k:
                block = scratch[: n - k]
                # FP16 tensor-core left-looking update, FP32 accumulate. The
                # product lands straight in the scratch and the original block
                # column is subtracted from it, so no separate copy is needed.
                torch.mm(
                    shadow[k:, :k],
                    shadow[k:j, :k].transpose(-1, -2),
                    out_dtype=torch.float32,
                    out=block,
                )
                torch.sub(mat[k:, k:j], block, out=block)
            else:
                block = mat[:, :nb]
            lkk = torch.linalg.cholesky_ex(
                block[:nb],
                check_errors=False,
            ).L
            factor[k:j, k:j] = lkk
            if j >= n:
                break
            _exp061_leaf_inverse(lkk, inverse)
            torch.mm(
                block[nb:].to(torch.float16),
                inverse.transpose(-1, -2).to(torch.float16),
                out_dtype=torch.float32,
                out=factor[j:, k:j],
            )
            shadow[k:, k:j].copy_(factor[k:, k:j])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


def _blocked_tri_inv_32768(
    lower: torch.Tensor,
    base: int = 256,
) -> torch.Tensor:
    global _EXP058_V1_INVERSE_CALLS
    _EXP058_V1_INVERSE_CALLS += 1
    n = lower.shape[0]
    if n <= base or n % base or (n & (n - 1)):
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        return torch.linalg.solve_triangular(
            lower,
            identity,
            upper=False,
        )
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    count = n // base
    leaf_shape = (count, base, base)
    leaf_stride = (base * n + base, n, 1)
    blocks = lower.as_strided(leaf_shape, leaf_stride).contiguous()
    identity = torch.eye(base, device=lower.device, dtype=lower.dtype)
    inverse.as_strided(leaf_shape, leaf_stride).copy_(
        torch.linalg.solve_triangular(
            blocks,
            identity.expand(leaf_shape).contiguous(),
            upper=False,
        )
    )
    size = base
    while size < n:
        step = 2 * size
        shape = (n // step, size, size)
        stride = (step * n + step, n, 1)
        inv11 = inverse.as_strided(shape, stride, 0)
        inv22 = inverse.as_strided(shape, stride, size * n + size)
        low21 = lower.as_strided(shape, stride, size * n)
        inverse.as_strided(shape, stride, size * n).copy_(
            torch.bmm(inv22, torch.bmm(low21, inv11)).neg_()
        )
        size = step
    return inverse


def _fp16_solve_32768(
    panel: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    global _EXP058_V4_FP16_SOLVE_HITS
    solved = torch.mm(
        panel.to(torch.float16),
        inverse.transpose(-1, -2).to(torch.float16),
        out_dtype=torch.float32,
    )
    _EXP058_V4_FP16_SOLVE_HITS += 1
    return solved


def _factor_1x32768_blocked_inverse(
    mat: torch.Tensor,
) -> torch.Tensor:
    nb = 4096
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            diagonal = mat[k : k + kb, k : k + kb].contiguous()
            if k:
                previous_row = factor[k : k + kb, :k]
                diagonal.addmm_(
                    previous_row,
                    previous_row.transpose(-1, -2),
                    beta=1.0,
                    alpha=-1.0,
                )
            diagonal_factor = torch.linalg.cholesky_ex(
                diagonal,
                check_errors=False,
            ).L
            factor[k : k + kb, k : k + kb] = diagonal_factor
            j = k + kb
            if j >= n:
                break
            panel = mat[j:, k : k + kb].contiguous()
            if k:
                _mxfp8_panel_update(
                    panel,
                    factor[j:, :k],
                    factor[k : k + kb, :k],
                )
            inverse = _blocked_tri_inv_32768(diagonal_factor)
            factor[j:, k : k + kb] = _fp16_solve_32768(
                panel,
                inverse,
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor


# ---------------------------------------------------------------------------
# Experiment 061 V1: remove the block-move overhead from the 1x32768 path.
#
# The B200 kernel profile of the ranked 32768 path (experiments/061-32768-
# overhead/baseline-shapediag.json) shows 5003us -- 16.3% of the whole shape --
# inside `at::native::elementwise_kernel<128, 2, ...>` over 107 launches. That
# is PyTorch's *generic* (non-vectorized, OffsetCalculator) elementwise path,
# taken because every block move in the loop has a strided operand: the
# `mat[...] .contiguous()` clones read a 4096-wide window of a 32768-wide row,
# and the `factor[...] = ...` stores write one. Measured throughput across
# those 107 launches is ~2.0 TB/s against ~7 TB/s of achievable HBM bandwidth.
#
# V1 keeps the arithmetic of the ranked path bit-for-bit (same TF32 diagonal
# SYRK, same MXFP8 panel product, same recursive blocked inverse, same FP16
# solve-apply, same order of operations) and only replaces those moves with a
# single Triton kernel that knows the stride explicitly, so the loads and
# stores vectorize. Three moves collapse into one pass each:
#
#   * the diagonal clone and the panel clone become strided->contiguous gathers;
#   * the panel gather also subtracts the MXFP8 product and emits FP16 directly,
#     folding the old `sub_` and the old `.to(torch.float16)` into the gather;
#   * the two factor stores become contiguous->strided scatters (and the
#     solve-apply skips its scatter entirely when `torch.mm` accepts a strided
#     `out=`, letting cuBLAS write the panel through `ldc` = 32768).
#
# Workspaces are allocated once per call instead of per block step.
# ---------------------------------------------------------------------------

_EXP061_V1_HITS = 0
_EXP061_V2_HITS = 0
_EXP061_MOVE_HITS = 0
_EXP061_MX_PRODUCT_HITS = 0
_EXP061_MX_COLUMN_HITS = 0
_EXP061_MM_OUT_HITS = 0
_EXP061_MM_OUT_SUPPORTED = True

_EXP061_STEP_HITS = 0
_EXP061_ERROR = None

_EXP061_MOVE_BLOCK_M = 16
_EXP061_MOVE_BLOCK_N = 512
_EXP061_MOVE_SQUARE = 64

if _HAVE_TRITON:

    @triton.jit
    def _exp061_block_move_kernel(
        src_ptr,
        prod_ptr,
        out_ptr,
        rows,
        cols,
        stride_src_m,
        stride_src_n,
        stride_out_m,
        stride_out_n,
        HAS_PROD: tl.constexpr,
        OUT_FP16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[i, j] = src[i, j] - prod[i, j], optionally cast to FP16.

        Both operands carry explicit 2D strides, so a 4096-column window of a
        32768-wide matrix -- or the column-major factor `torch.linalg.
        cholesky_ex` hands back -- is moved with vectorized loads instead of
        PyTorch's generic OffsetCalculator elementwise kernel. `prod`, when
        present, is always the contiguous (rows, cols) MXFP8 product.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rm[:, None] < rows) & (rn[None, :] < cols)
        # 32768x32768 fp32 is 2^30 elements, so offsets are computed in 64-bit
        # to keep the address arithmetic exact at every block width.
        row64 = rm[:, None].to(tl.int64)
        col64 = rn[None, :].to(tl.int64)
        value = tl.load(
            src_ptr + row64 * stride_src_m + col64 * stride_src_n,
            mask=mask,
            other=0.0,
        )
        if HAS_PROD:
            value = value - tl.load(
                prod_ptr + row64 * cols + col64,
                mask=mask,
                other=0.0,
            )
        if OUT_FP16:
            value = value.to(tl.float16)
        tl.store(
            out_ptr + row64 * stride_out_m + col64 * stride_out_n,
            value,
            mask=mask,
        )


def _exp061_move(src, out, prod=None, out_fp16=False):
    """Move `src` (minus `prod`) into `out`, honouring both operands' strides.

    A wide row-major window uses tall-thin tiles for maximal vector width; a
    transposing move -- `torch.linalg.cholesky_ex` returns its factor in
    column-major layout -- uses square tiles so both sides stay coalesced.
    """
    global _EXP061_MOVE_HITS
    rows, cols = src.shape
    if src.stride(1) == 1 and out.stride(1) == 1:
        block_m = _EXP061_MOVE_BLOCK_M
        block_n = _EXP061_MOVE_BLOCK_N
    else:
        block_m = _EXP061_MOVE_SQUARE
        block_n = _EXP061_MOVE_SQUARE
    _exp061_block_move_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    ](
        src,
        prod if prod is not None else src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_PROD=prod is not None,
        OUT_FP16=out_fp16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _EXP061_MOVE_HITS += 1


def _exp061_mx_product(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """lhs @ rhs^T on MXFP8 block-scaled tensor cores, returned instead of
    subtracted in place -- the subtraction is folded into the panel gather."""
    global _EXP061_MX_PRODUCT_HITS
    m_rows, k_cols = lhs.shape
    n_rows = rhs.shape[0]
    if (
        m_rows % 128
        or n_rows % 128
        or k_cols % 128
        or m_rows % _MX_QUANT_BLOCK_M
        or n_rows % _MX_QUANT_BLOCK_M
        or k_cols % _MX_QUANT_BLOCK_K
    ):
        raise RuntimeError("exp061 mxfp8 tiling mismatch")
    q_lhs, s_lhs = _mx_quant_e4m3_blocked(lhs)
    q_rhs, s_rhs = _mx_quant_e4m3_blocked(rhs)
    product = torch._scaled_mm(
        q_lhs,
        q_rhs.t(),
        scale_a=s_lhs,
        scale_b=s_rhs,
        out_dtype=torch.float32,
    )
    _EXP061_MX_PRODUCT_HITS += 1
    return product


def _exp061_mx_column_product(left: torch.Tensor, nb: int) -> torch.Tensor:
    """`left @ left[:nb].T` on MXFP8 tensor cores, quantizing the frontier once.

    Experiment 061 V2 folds the diagonal block's SYRK update into the panel's
    left-looking update. At block column k both consume the same frontier
    `factor[k:, :k]`, and the right operand `factor[k:k+nb, :k]` is literally
    its first nb rows, so a single quantization of `left` serves both operands.
    `_mx_quant_e4m3_blocked` emits e8m0 scales in row-tile-major order
    (`tile = (pid_m // 4) * (columns // 128) + pid_k`), so every tile belonging
    to the first nb rows lands inside the first `nb * cols / 32` bytes and the
    right operand's scale buffer is an exact prefix slice of the left one's.

    This retires the TF32 SYRK -- 4733us of the baseline profile, running at
    ~813 TFLOP/s -- in exchange for nb extra rows on a GEMM measured at
    ~2950 TFLOP/s, and it costs no extra quantization at all: `(n - k) * k`
    elements instead of the previous `(n - j) * k + nb * k`, the same count.

    The diagonal block therefore inherits MXFP8 accuracy instead of TF32, so
    the reconstruction residual is the gate on this variant.
    """
    global _EXP061_MX_COLUMN_HITS
    rows, cols = left.shape
    if (
        rows % 128
        or cols % 128
        or nb % 128
        or rows % _MX_QUANT_BLOCK_M
        or cols % _MX_QUANT_BLOCK_K
    ):
        raise RuntimeError("exp061 mxfp8 column tiling mismatch")
    quantized, scales = _mx_quant_e4m3_blocked(left)
    product = torch._scaled_mm(
        quantized,
        quantized[:nb].t(),
        scale_a=scales,
        scale_b=scales[: nb * cols // 32],
        out_dtype=torch.float32,
    )
    _EXP061_MX_COLUMN_HITS += 1
    return product


def _exp061_factor_1x32768(mat: torch.Tensor) -> torch.Tensor:
    global _EXP061_V2_HITS, _EXP061_MM_OUT_SUPPORTED, _EXP061_MM_OUT_HITS
    global _EXP061_STEP_HITS, _EXP061_ERROR
    if not _HAVE_TRITON:
        raise RuntimeError("exp061 requires Triton")
    nb = 4096
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    diagonal = torch.empty(nb, nb, device=mat.device, dtype=torch.float32)
    panel_half = torch.empty(
        n - nb, nb, device=mat.device, dtype=torch.float16
    )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = nb
            j = k + kb
            _EXP061_STEP_HITS += 1
            # One MXFP8 block-column update covers the diagonal block and the
            # panel below it, so the TF32 SYRK disappears and the frontier is
            # quantized once for both operands.
            column = (
                _exp061_mx_column_product(factor[k:, :k], kb) if k else None
            )
            _exp061_move(
                mat[k:j, k:j],
                diagonal,
                prod=None if column is None else column[:kb],
            )
            diagonal_factor = torch.linalg.cholesky_ex(
                diagonal,
                check_errors=False,
            ).L
            _exp061_move(diagonal_factor, factor[k:j, k:j])
            if j >= n:
                break
            rows = n - j
            half_panel = panel_half[:rows]
            _exp061_move(
                mat[j:, k:j],
                half_panel,
                prod=None if column is None else column[kb:],
                out_fp16=True,
            )
            column = None
            inverse = _blocked_tri_inv_32768(diagonal_factor)
            half_inverse = inverse.transpose(-1, -2).to(torch.float16)
            target = factor[j:, k:j]
            wrote = False
            if _EXP061_MM_OUT_SUPPORTED:
                try:
                    torch.mm(
                        half_panel,
                        half_inverse,
                        out_dtype=torch.float32,
                        out=target,
                    )
                    _EXP061_MM_OUT_HITS += 1
                    wrote = True
                except (TypeError, RuntimeError):
                    _EXP061_MM_OUT_SUPPORTED = False
            if not wrote:
                _exp061_move(
                    torch.mm(
                        half_panel,
                        half_inverse,
                        out_dtype=torch.float32,
                    ),
                    target,
                )
    except Exception as exc:  # surfaced through _EXP061_ERROR for diagnosis
        _EXP061_ERROR = repr(exc)
        raise
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _EXP061_ERROR = None
    _EXP061_V2_HITS += 1
    return factor


# ---------------------------------------------------------------------------



def custom_kernel(data: input_t) -> output_t:
    global _LEFT_32768_ERROR, _LEFT_LARGE_FALLBACKS
    global _LARGE_FP8_HITS, _LARGE_FP8_FALLBACKS, _LARGE_FP8_ERROR
    global _FUSED_CTA_HITS, _FUSED_CTA_FALLBACKS, _FUSED_CTA_ERROR
    global _EXP057_V2_HITS, _EXP058_V1_HITS, _EXP061_16384_HITS

    batch, n, _ = data.shape
    is_f32_cuda = data.is_cuda and data.dtype == torch.float32

    if is_f32_cuda and _CUDA32 is not None and n == 32 and data.is_contiguous():
        return _cuda_cholesky32(data)

    if is_f32_cuda and _HAVE_TRITON and n == 32:
        return _triton_cholesky32_rank2(data)

    if (
        is_f32_cuda
        and _CUDA64 is not None
        and batch == 1024
        and n == 64
        and data.is_contiguous()
    ):
        return _cuda_cholesky64(data)

    if (
        is_f32_cuda
        and _CUDA128 is not None
        and batch == 256
        and n == 128
        and data.is_contiguous()
    ):
        return _cuda_cholesky128(data)

    if (
        is_f32_cuda
        and batch == 64
        and n == 256
        and data.is_contiguous()
    ):
        _load_cuda256()
        if _CUDA256 is not None:
            return _cuda_cholesky256(data)

    # Experiment 015 round 4: two-level blocked tensor-core potrf with
    # per-shape graph replay for the mid shapes. On any numerical failure
    # (non-finite diagonal on ill-conditioned families) fall through to the
    # previously shipped dispatch below, which is the exact ranked behavior.
    if is_f32_cuda and _HAVE_TRITON and (batch, n) in _SPLIT32_SHAPES:
        try:
            l = _split32_factor(data)
            if torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all().item():
                _FUSED_CTA_HITS += 1
                return l
            _FUSED_CTA_FALLBACKS += 1
        except Exception as exc:
            _FUSED_CTA_ERROR = repr(exc)
            _FUSED_CTA_FALLBACKS += 1

    if is_f32_cuda and batch == 1024 and n == 64:
        l = _graph_cholesky_1024x64(data)
        if l is not None:
            return l

    if is_f32_cuda and batch == 256 and n == 128:
        return _graph_cholesky_256x128(data)

    if is_f32_cuda and batch == 16 and n == 512:
        return _graph_cholesky_16x512(data)

    if is_f32_cuda and _HAVE_TRITON and batch == 8 and n == 2048:
        l = _triton_cholesky_8x2048(data)
        if torch.isfinite(l.diagonal(dim1=-2, dim2=-1)).all().item():
            return l
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    if is_f32_cuda and batch == 1 and n in _LARGE_CFG:
        try:
            if n == 16384:
                l = _exp061_factor_1x16384(data[0])
            elif n == 32768:
                if (
                    _HAVE_TRITON
                    and _LARGE_CFG[32768].get("path") == "exp061_v2"
                ):
                    l = _exp061_factor_1x32768(data[0])
                else:
                    l = _factor_1x32768_blocked_inverse(data[0])
            else:
                l = _left_looking_large(data[0], **_LARGE_CFG[n])
            if torch.isfinite(l.diagonal()).all().item():
                if n == 16384:
                    _EXP061_16384_HITS += 1
                elif n == 32768:
                    _EXP058_V1_HITS += 1
                _LARGE_FP8_HITS += 1
                return l.unsqueeze(0)
            _LARGE_FP8_FALLBACKS += 1
        except Exception as exc:
            _LARGE_FP8_ERROR = repr(exc)
            _LARGE_FP8_FALLBACKS += 1

    if is_f32_cuda and batch == 1 and n == 16384:
        try:
            l = _left_looking_cholesky_16384(data[0])
            if torch.isfinite(l.diagonal()).all().item():
                return l.unsqueeze(0)
        except Exception:
            pass
        _LEFT_LARGE_FALLBACKS += 1
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    if is_f32_cuda and batch == 1 and n == 32768:
        try:
            l = _left_looking_cholesky_32768(data[0])
            if torch.isfinite(l.diagonal()).all().item():
                _LEFT_32768_ERROR = None
                return l.unsqueeze(0)
        except Exception as exc:
            _LEFT_32768_ERROR = repr(exc)
        _LEFT_LARGE_FALLBACKS += 1
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    # Large single matrices: blocked Cholesky with a TF32 tensor-core trailing
    # update beats cuSOLVER's all-FP32 potrf (exp 006), with the product and
    # subtraction fused in-place by exp 008. Only the measured-win
    # region (batch==1, n>=16384); 8192 was only ~1.07x so it stays on cuSOLVER.
    if is_f32_cuda and batch == 1 and n >= 16384:
        nb = 4096 if n >= 32768 else 2048
        l = _blocked_cholesky_tf32(data[0], nb)
        # Numerical safety net: TF32 error can drive a late diagonal block
        # indefinite on ill-conditioned inputs (spectrum/lowrank), yielding
        # NaN/Inf. The ranked shapes are well-conditioned dense (huge margin,
        # never trips this), but fall back to exact FP32 cuSOLVER otherwise so
        # correctness holds across every family. isfinite is ~memory-bound and
        # negligible vs the O(n^3) factorization.
        if torch.isfinite(l).all().item():
            return l.unsqueeze(0)
        return torch.linalg.cholesky_ex(data, check_errors=False).L

    # Few-but-large matrices: avoid cusolverDnSpotrfBatched (see module docstring).
    # exp 005: upper bound trimmed 8->4 so 8x2048 stays on batched cuSOLVER.
    if is_f32_cuda and 2 <= batch <= 4 and n >= 1024:
        return _loop_cholesky(data)

    # Default: batched cuSOLVER. Correct for every input family.
    return torch.linalg.cholesky_ex(data, check_errors=False).L


# ---------------------------------------------------------------------------
# Experiment 062 round 3 -- fix the block's GEMM phases.
#
# Round 2 settled where the time goes. The register pivot chain is *fast*:
# 63.3 ns/pivot (2.025us per 32 pivots), against exp-050's best-ever 134
# ns/pivot and the vendor's ~330 ns/row, with zero register spilling. But the
# whole 128x128 block still cost 88.2us and only 4 x 2.025 = 8.1us of that is
# the chain. The other ~80us is the block's parallel phases, whose inner loops
# issued FIVE shared loads per FOUR fused multiply-adds -- a badly tiled GEMM.
#
# Round 3 keeps the chain verbatim and rewrites everything around it:
#   * 4x4 register tiling, so each thread keeps 16 accumulators live and the
#     inner step costs 2 vector loads per 16 FMAs (~256 FMAs per shared-load
#     instruction, against ~26 before).
#   * a staged transpose `P[t][x] = S[x][kk+t]`, which turns the trailing
#     update's strided column operand into contiguous `float4` reads.
#   * row stride 132 (a multiple of 4) so `float4` loads are legal on S and M.
#   * a triangular inverse with the 32 diagonal reciprocals computed in
#     parallel across lanes instead of 32 serial divisions.
#   * `clock64()` phase accounting, so round 4 is never guesswork.
# ---------------------------------------------------------------------------

_EXP062 = None
_EXP062_COMBINED = None
_EXP062_ERROR = None
_EXP062_HITS = 0
_EXP062_FALLBACKS = 0

_EXP062_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define E62_LD    132          // multiple of 4: float4 loads on S and M
#define E62_TB    128
#define E62_QLD   33
#define E62_PLD   132
#define E62_NW    8
#define E62_NT    256

#define E62_M_OFF   (E62_TB * E62_LD)
#define E62_QI_OFF  (2 * E62_TB * E62_LD)
#define E62_QT_OFF  (E62_QI_OFF + 32 * E62_QLD)
#define E62_P_OFF   (E62_QT_OFF + 32 * E62_LD)
#define E62_T_OFF   (E62_P_OFF + 32 * E62_PLD)
#define E62_SMEM_F  (E62_T_OFF + 256)
#define E62_SMEM_B  (E62_SMEM_F * 4)

#define E62_PROF 8

// --------------------------------------------------------------------------
// Register-resident Cholesky chain -- unchanged from round 2 (63.3 ns/pivot).
// Lane i owns row i of the 32x32 tile, so the only cross-lane traffic per
// pivot is a broadcast of the pivot column.
// --------------------------------------------------------------------------
__device__ __forceinline__ void e62_chain32_reg(float* __restrict__ Sb, int lane)
{
    float a[32];
    #pragma unroll
    for (int t = 0; t < 32; ++t) a[t] = Sb[lane * E62_LD + t];
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        const float akk = __shfl_sync(0xffffffffu, a[k], k);
        const float dk  = rsqrtf(akk);
        const float lik = a[k] * dk;
        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            const float Lt = __shfl_sync(0xffffffffu, lik, t);
            if (lane > k && t > k) a[t] -= lik * Lt;
        }
        if (lane >= k) a[k] = lik;
    }
    #pragma unroll
    for (int t = 0; t < 32; ++t) Sb[lane * E62_LD + t] = (t <= lane) ? a[t] : 0.0f;
}

// --------------------------------------------------------------------------
// Column-parallel triangular inverse. Lane j owns column j and solves
// L x = e_j. Every L[i][p] read is one shared broadcast and every x_p is
// lane-local, so there is no cross-lane traffic at all. The 32 diagonal
// reciprocals are computed once, in parallel across lanes, instead of as 32
// serial divisions on the dependent path.
// --------------------------------------------------------------------------
__device__ __forceinline__ void e62_tri_inv32(const float* __restrict__ Sb,
                                              float* __restrict__ Qi,
                                              float* __restrict__ Tmp, int lane)
{
    // Two-level blocked inverse of the 32x32 lower factor L = [[A,0],[B,C]]:
    //   inv(L) = [[Ai, 0], [-Ci*B*Ai, Ci]]
    // Lanes 0-15 solve for Ai and lanes 16-31 solve for Ci *concurrently*, so
    // the dependent substitution chain is 16 steps instead of 32; the coupling
    // block is then two 16x16x16 products with full warp parallelism.
    const int base = (lane < 16) ? 0 : 16;
    const int col  = lane & 15;
    const float rdiag = __frcp_rn(Sb[(base + col) * E62_LD + base + col]);
    float x[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float s0 = (i == col) ? 1.0f : 0.0f;
        float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        const float* Lr = Sb + (base + i) * E62_LD + base;
        #pragma unroll
        for (int p = 0; p < 16; p += 4) {
            if (p + 0 < i) s0 -= Lr[p + 0] * x[p + 0];
            if (p + 1 < i) s1 -= Lr[p + 1] * x[p + 1];
            if (p + 2 < i) s2 -= Lr[p + 2] * x[p + 2];
            if (p + 3 < i) s3 -= Lr[p + 3] * x[p + 3];
        }
        const float ri = __shfl_sync(0xffffffffu, rdiag, base + i);
        x[i] = (i >= col) ? ((s0 + s1) + (s2 + s3)) * ri : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        Qi[(base + i) * E62_QLD + base + col] = x[i];
        if (base == 0) Qi[i * E62_QLD + 16 + col] = 0.0f;
    }
    __syncwarp();
    #pragma unroll
    for (int e = 0; e < 8; ++e) {                 // Tmp = B * Ai
        const int idx = e * 32 + lane;
        const int i = idx >> 4, jc = idx & 15;
        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p)
            acc += Sb[(16 + i) * E62_LD + p] * Qi[p * E62_QLD + jc];
        Tmp[i * 16 + jc] = acc;
    }
    __syncwarp();
    #pragma unroll
    for (int e = 0; e < 8; ++e) {                 // M21 = -Ci * Tmp
        const int idx = e * 32 + lane;
        const int i = idx >> 4, jc = idx & 15;
        float acc = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p)
            acc += Qi[(16 + i) * E62_QLD + 16 + p] * Tmp[p * 16 + jc];
        Qi[(16 + i) * E62_QLD + jc] = -acc;
    }
}

// --------------------------------------------------------------------------

__global__ __launch_bounds__(E62_NT, 1)
void e62_diag128(float* __restrict__ A, float* __restrict__ Dinv,
                 long long* __restrict__ Prof, const int n, const int j)
{
    extern __shared__ float sm[];
    float* S  = sm;
    float* M  = sm + E62_M_OFF;
    float* Qi = sm + E62_QI_OFF;
    float* Qt = sm + E62_QT_OFF;
    float* P  = sm + E62_P_OFF;
    float* Tp = sm + E62_T_OFF;

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int tr   = tid >> 3;      // 0..31, row-tile index
    const int tc   = tid & 7;       // 0..7,  col-tile index

    long long t0 = 0;
    long long ph[E62_PROF];
    #pragma unroll
    for (int i = 0; i < E62_PROF; ++i) ph[i] = 0;
    const bool prof = (Prof != nullptr) && (tid == 0);
    if (prof) t0 = clock64();

    float* Ab = A + (size_t)blockIdx.x * (size_t)n * (size_t)n
                  + (size_t)j * (size_t)n + (size_t)j;

    // float4 traffic: 128 floats per row is exactly 32 vectors, so each row
    // is one fully coalesced warp transaction instead of four scalar ones.
    const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = warp; r < E62_TB; r += E62_NW) {
        const float4* srow = (const float4*)(Ab + (size_t)r * (size_t)n);
        float4* drow = (float4*)(S + r * E62_LD);
        float4* mrow = (float4*)(M + r * E62_LD);
        drow[lane] = srow[lane];
        mrow[lane] = zero4;
    }
    __syncthreads();
    for (int r = tid; r < E62_TB; r += E62_NT) M[r * E62_LD + r] = 1.0f;
    __syncthreads();
    if (prof) { ph[0] = clock64() - t0; t0 = clock64(); }

    for (int kk = 0; kk < E62_TB; kk += 32) {
        const int lwid  = kk + 32;
        const int nrow  = E62_TB - lwid;      // rows below the pivot block
        float* Sb = S + kk * E62_LD + kk;

        // ---- 1. pivot chain (warp 0) ----
        if (warp == 0) e62_chain32_reg(Sb, lane);
        __syncwarp();
        if (prof && tid == 0) { ph[1] += clock64() - t0; t0 = clock64(); }

        // ---- 2. triangular inverse of the pivot block (warp 0) ----
        if (warp == 0) e62_tri_inv32(Sb, Qi, Tp, lane);
        __syncthreads();
        if (prof) { ph[2] += clock64() - t0; t0 = clock64(); }

        // ---- 3. panel solve: S[r][kk:kk+32] <- S[r][kk:kk+32] * inv(L11)^T
        for (int r0 = lwid + warp * 4; r0 < E62_TB; r0 += E62_NW * 4) {
            const float* q  = Qi + lane * E62_QLD;
            const float* s0 = S + r0 * E62_LD + kk;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            #pragma unroll 8
            for (int u = 0; u < 32; ++u) {
                const float qv = q[u];
                a0 += s0[u] * qv;
                a1 += s0[E62_LD + u] * qv;
                a2 += s0[2 * E62_LD + u] * qv;
                a3 += s0[3 * E62_LD + u] * qv;
            }
            __syncwarp();
            S[r0 * E62_LD + kk + lane]       = a0;
            S[(r0 + 1) * E62_LD + kk + lane] = a1;
            S[(r0 + 2) * E62_LD + kk + lane] = a2;
            S[(r0 + 3) * E62_LD + kk + lane] = a3;
        }
        __syncthreads();
        if (prof) { ph[3] += clock64() - t0; t0 = clock64(); }

        // ---- 4. stage P[t][x] = S[x][kk+t] so the trailing update reads its
        //         second operand as contiguous float4 instead of a stride-132
        //         column walk.
        for (int t = warp; t < 32; t += E62_NW)
            for (int x = lane; x < nrow; x += 32)
                P[t * E62_PLD + x] = S[(lwid + x) * E62_LD + kk + t];

        // ---- 5. inverse row block: Qt <- Qi * M[kk:kk+32, 0:kk] ----
        if (kk > 0) {
            for (int idx = tid; idx < 32 * (kk >> 2); idx += E62_NT) {
                const int i = idx / (kk >> 2);
                const int c = (idx % (kk >> 2)) << 2;
                const float* q = Qi + i * E62_QLD;
                float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
                #pragma unroll 8
                for (int u = 0; u < 32; ++u) {
                    const float qv = q[u];
                    const float4 mv =
                        *(const float4*)(M + (kk + u) * E62_LD + c);
                    b0 += qv * mv.x; b1 += qv * mv.y;
                    b2 += qv * mv.z; b3 += qv * mv.w;
                }
                float4* dst = (float4*)(Qt + i * E62_LD + c);
                *dst = make_float4(b0, b1, b2, b3);
            }
        }
        __syncthreads();
        if (prof) { ph[4] += clock64() - t0; t0 = clock64(); }

        for (int i = warp; i < 32; i += E62_NW) {
            float* mrow = M + (kk + i) * E62_LD;
            for (int c = lane; c < kk; c += 32) mrow[c] = Qt[i * E62_LD + c];
            mrow[kk + lane] = Qi[i * E62_QLD + lane];
        }
        __syncthreads();
        if (prof) { ph[5] += clock64() - t0; t0 = clock64(); }

        if (nrow <= 0) continue;

        // ---- 6. trailing update, 4x4 register tiles ----
        //         S[r][c] -= sum_t P[t][r] * P[t][c]
        {
            const int nt = nrow >> 2;               // 4x4 tiles per side
            const int ntiles = nt * nt;
            for (int tile = tid; tile < ntiles; tile += E62_NT) {
                const int ti = tile / nt, tj = tile - ti * nt;
                const int rr = ti << 2, cc = tj << 2;
                float acc[4][4];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) acc[i][k2] = 0.f;
                #pragma unroll 8
                for (int t = 0; t < 32; ++t) {
                    const float4 av = *(const float4*)(P + t * E62_PLD + rr);
                    const float4 bv = *(const float4*)(P + t * E62_PLD + cc);
                    acc[0][0] += av.x * bv.x; acc[0][1] += av.x * bv.y;
                    acc[0][2] += av.x * bv.z; acc[0][3] += av.x * bv.w;
                    acc[1][0] += av.y * bv.x; acc[1][1] += av.y * bv.y;
                    acc[1][2] += av.y * bv.z; acc[1][3] += av.y * bv.w;
                    acc[2][0] += av.z * bv.x; acc[2][1] += av.z * bv.y;
                    acc[2][2] += av.z * bv.z; acc[2][3] += av.z * bv.w;
                    acc[3][0] += av.w * bv.x; acc[3][1] += av.w * bv.y;
                    acc[3][2] += av.w * bv.z; acc[3][3] += av.w * bv.w;
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4* d = (float4*)(S + (lwid + rr + i) * E62_LD
                                            + lwid + cc);
                    float4 v = *d;
                    v.x -= acc[i][0]; v.y -= acc[i][1];
                    v.z -= acc[i][2]; v.w -= acc[i][3];
                    *d = v;
                }
            }
        }

        // ---- 7. inverse update: M[r][c] -= sum_t P[t][r] * M[kk+t][c] ----
        {
            const int nt = nrow >> 2;
            const int nc = lwid >> 2;
            const int ntiles = nt * nc;
            for (int tile = tid; tile < ntiles; tile += E62_NT) {
                const int ti = tile / nc, tj = tile - ti * nc;
                const int rr = ti << 2, cc = tj << 2;
                float acc[4][4];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int k2 = 0; k2 < 4; ++k2) acc[i][k2] = 0.f;
                #pragma unroll 8
                for (int t = 0; t < 32; ++t) {
                    const float4 av = *(const float4*)(P + t * E62_PLD + rr);
                    const float4 bv =
                        *(const float4*)(M + (kk + t) * E62_LD + cc);
                    acc[0][0] += av.x * bv.x; acc[0][1] += av.x * bv.y;
                    acc[0][2] += av.x * bv.z; acc[0][3] += av.x * bv.w;
                    acc[1][0] += av.y * bv.x; acc[1][1] += av.y * bv.y;
                    acc[1][2] += av.y * bv.z; acc[1][3] += av.y * bv.w;
                    acc[2][0] += av.z * bv.x; acc[2][1] += av.z * bv.y;
                    acc[2][2] += av.z * bv.z; acc[2][3] += av.z * bv.w;
                    acc[3][0] += av.w * bv.x; acc[3][1] += av.w * bv.y;
                    acc[3][2] += av.w * bv.z; acc[3][3] += av.w * bv.w;
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4* d = (float4*)(M + (lwid + rr + i) * E62_LD + cc);
                    float4 v = *d;
                    v.x -= acc[i][0]; v.y -= acc[i][1];
                    v.z -= acc[i][2]; v.w -= acc[i][3];
                    *d = v;
                }
            }
        }
        __syncthreads();
        if (prof) { ph[6] += clock64() - t0; t0 = clock64(); }
    }

    float* Db = Dinv + (size_t)blockIdx.x * (size_t)(E62_TB * E62_TB);
    for (int r = warp; r < E62_TB; r += E62_NW) {
        const int c0 = lane << 2;
        float4 sv = *(const float4*)(S + r * E62_LD + c0);
        float4 mv = *(const float4*)(M + r * E62_LD + c0);
        if (c0 + 3 > r) {           // clip the diagonal-crossing vector
            if (c0 + 0 > r) { sv.x = 0.f; mv.x = 0.f; }
            if (c0 + 1 > r) { sv.y = 0.f; mv.y = 0.f; }
            if (c0 + 2 > r) { sv.z = 0.f; mv.z = 0.f; }
            if (c0 + 3 > r) { sv.w = 0.f; mv.w = 0.f; }
        }
        *(float4*)(Ab + (size_t)r * (size_t)n + c0) = sv;
        *(float4*)(Db + r * E62_TB + c0) = mv;
    }
    if (prof) {
        ph[7] = clock64() - t0;
        long long* out = Prof + (size_t)blockIdx.x * E62_PROF;
        #pragma unroll
        for (int i = 0; i < E62_PROF; ++i) out[i] = ph[i];
    }
}

void e62_diag128_launch(torch::Tensor A, torch::Tensor Dinv,
                        int64_t n, int64_t j, torch::Tensor Prof)
{
    static bool configured = false;
    if (!configured) {
        cudaError_t attr = cudaFuncSetAttribute(
            e62_diag128, cudaFuncAttributeMaxDynamicSharedMemorySize,
            E62_SMEM_B);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    const int batch = (int)A.size(0);
    long long* prof =
        Prof.numel() > 0 ? (long long*)Prof.data_ptr<int64_t>() : nullptr;
    e62_diag128<<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
        A.data_ptr<float>(), Dinv.data_ptr<float>(), prof, (int)n, (int)j);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""


def _load_exp062():
    global _EXP062, _EXP062_ERROR
    if _EXP062 is not None or _EXP062_ERROR is not None:
        return
    if _EXP062_COMBINED is not None:
        _EXP062 = _EXP062_COMBINED
        return
    try:
        from torch.utils.cpp_extension import load_inline

        _EXP062 = load_inline(
            name="chol_exp062_midbatch_v4",
            cpp_sources=(
                "void e62_diag128_launch(torch::Tensor, torch::Tensor, "
                "int64_t, int64_t, torch::Tensor);"
            ),
            cuda_sources=_EXP062_SOURCE,
            functions=["e62_diag128_launch"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-v"],
            verbose=True,
        )
    except Exception as exc:  # pragma: no cover
        _EXP062_ERROR = repr(exc)


_EXP062_BUF = {}
_EXP062_NOPROF = None


def _exp062_buffers(batch, n, device):
    global _EXP062_NOPROF
    if _EXP062_NOPROF is None:
        _EXP062_NOPROF = torch.empty(0, device=device, dtype=torch.int64)
    key = (batch, n)
    buf = _EXP062_BUF.get(key)
    if buf is None:
        dinv = torch.empty(batch, 128, 128, device=device, dtype=torch.float32)
        pan = torch.empty(batch * n * 128, device=device, dtype=torch.float32)
        buf = (dinv, pan)
        _EXP062_BUF[key] = buf
    return buf


def _exp062_factor(data, nb_outer=1024, prof=None):
    batch, n, _ = data.shape
    work = data.clone()
    dinv, pan = _exp062_buffers(batch, n, data.device)
    prof = _EXP062_NOPROF if prof is None else prof
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for j0 in range(0, n, nb_outer):
            jend = min(j0 + nb_outer, n)
            for jj in range(j0, jend, 128):
                if jj > j0:
                    left = work[:, jj:, j0:jj]
                    top = work[:, jj:jj + 128, j0:jj]
                    work[:, jj:, jj:jj + 128].baddbmm_(
                        left, top.transpose(1, 2), beta=1.0, alpha=-1.0
                    )
                _EXP062.e62_diag128_launch(work, dinv, n, jj, prof)
                rows = n - jj - 128
                if rows > 0:
                    src = work[:, jj + 128:, jj:jj + 128]
                    dst = pan[:batch * rows * 128].view(batch, rows, 128)
                    torch.bmm(src, dinv.transpose(1, 2), out=dst)
                    src.copy_(dst)
            if jend < n:
                blk = work[:, jend:, j0:jend]
                work[:, jend:, jend:].baddbmm_(
                    blk, blk.transpose(1, 2), beta=1.0, alpha=-1.0
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return work.tril_()


def _e62_time(fn, arg, iters=10, warmup=3):
    for _ in range(warmup):
        fn(arg)
    torch.cuda.synchronize()
    durations = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn(arg)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3)
    durations.sort()
    return sum(durations) / len(durations)


def _e62_residual(a, l):
    recon = l @ l.transpose(-2, -1)
    return float((recon - a).abs().max().item()), float(a.abs().max().item())


def _shipped(x):
    return _loop_cholesky(x)


_PHASE_NAMES = ("load", "chain", "triinv", "panel", "stageP+Qt", "commit",
                "trailing+inv", "store")


def mid_probe():
    import sys

    sys.path.insert(0, "/root/reference")
    from reference import generate_input

    _load_exp062()
    rows = []
    if _EXP062 is None:
        return [{"name": "load_inline", "us": 0.0, "ok": False,
                 "error": _EXP062_ERROR}]

    dev = torch.device("cuda")
    a = generate_input(batch=2, n=2048, cond=2, seed=44048)
    blk = a[:, :128, :128].clone()
    work = a.clone()
    dinv = torch.empty(2, 128, 128, device=dev, dtype=torch.float32)
    noprof = torch.empty(0, device=dev, dtype=torch.int64)
    profbuf = torch.zeros(2 * 8, device=dev, dtype=torch.int64)

    def _restore(_):
        work[:, :128, :128].copy_(blk)

    us_copy = _e62_time(_restore, None, iters=20, warmup=5)

    def _blockrun(_):
        work[:, :128, :128].copy_(blk)
        _EXP062.e62_diag128_launch(work, dinv, 2048, 0, noprof)

    us = _e62_time(_blockrun, None, iters=20, warmup=5)
    net = us - us_copy
    rows.append({"name": "diag128_block", "us": round(net, 3),
                 "ns_per_row": round(net * 1000.0 / 128.0, 1), "ok": True})

    work[:, :128, :128].copy_(blk)
    _EXP062.e62_diag128_launch(work, dinv, 2048, 0, profbuf)
    torch.cuda.synchronize()
    l11 = work[:, :128, :128].tril()
    err, scale = _e62_residual(a[:, :128, :128], l11)
    inv_err = float(
        (dinv[0] @ l11[0] - torch.eye(128, device=dev)).abs().max().item()
    )
    rows.append({"name": "diag128_err", "us": 0.0, "abs_err": round(err, 7),
                 "inv_err": round(inv_err, 8),
                 "ok": err < 1e-3 and inv_err < 1e-3})
    cyc = profbuf[:8].tolist()
    total = max(sum(cyc), 1)
    for name, c in zip(_PHASE_NAMES, cyc):
        rows.append({"name": f"phase_{name}", "us": round(net * c / total, 3),
                     "cycles": c, "pct": round(100.0 * c / total, 1),
                     "ok": True})
    del work, blk, a
    torch.cuda.empty_cache()

    for (batch, n, seed) in ((2, 2048, 44048), (2, 4096, 514096)):
        a = generate_input(batch=batch, n=n, cond=2, seed=seed)
        base = _e62_time(_shipped, a, iters=8, warmup=3)
        rows.append({"name": f"shipped_{batch}x{n}", "us": round(base, 1),
                     "ok": True})
        for nbo in (512, 1024, 2048):
            if nbo > n:
                continue
            try:
                us = _e62_time(lambda x: _exp062_factor(x, nbo), a,
                               iters=8, warmup=3)
                l = _exp062_factor(a, nbo)
                torch.cuda.synchronize()
                err, scale = _e62_residual(a, l)
                rows.append({
                    "name": f"e062_{batch}x{n}_nbo{nbo}", "us": round(us, 1),
                    "speedup": round(base / us, 4),
                    "abs_err": round(err, 6),
                    "ok": bool(torch.isfinite(l).all().item()),
                })
                del l
            except Exception as exc:
                rows.append({"name": f"e062_{batch}x{n}_nbo{nbo}", "us": 0.0,
                             "ok": False, "error": repr(exc)[:240]})
        del a
        torch.cuda.empty_cache()
    return rows


# Enrolled shapes -> outer trailing-update block width.
_EXP062_SHAPES = {(2, 2048): 1024, (2, 4096): 1024}
