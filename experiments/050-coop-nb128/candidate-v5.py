"""Experiment 050 V5: single cooperative NB=128 tensor-core Cholesky, 4x1024.

Architecture (goal-exp050): one cooperative launch per call replaces the
101-launch dependent chain. Per outer step (NB=128, 8 steps at n=1024):

  D. one CTA per matrix factors the 128x128 diagonal block entirely in
     shared memory, fp32: four 32x32 register rank-2 micro factorizations
     (the exp 039/048 mechanism) + explicit 32x32 lower-triangular inverses
     (the shipped `_micro_potrf_gj32` potrf+inverse contract) + fp32 SIMT
     sub-panel/sub-trailing inside the block.
  P. panel TRSM by 32-row strips: block forward substitution against the
     four 32x32 inverses, all products on tf32 WMMA with round-to-nearest
     `__float_to_tf32` staging (matching Triton's tf32 dot semantics; the
     exp 048 V2 lowrank NaN came from 31 stored-back rank-32 tf32 updates
     plus truncating loads -- this design has neither).
  T. rank-128 trailing SYRK on tf32 WMMA, 64x64 tiles across the full grid,
     fp32 accumulate, one C read-modify-write per outer step (same rounding
     structure as the shipped `_trailing_nb`, which passes all six families
     at this shape).

Barriers: 3 grid.sync per outer step (~24 total) versus 96 in exp 048 V2 and
101 kernel launches at ~13us turnaround in the shipped path.
"""

import torch
import submission as _ranked

_COOP1024_HITS = 0
_COOP1024_FALLBACKS = 0
_COOP1024_ERROR = None
_COOP = None
_SCRATCH = {}

_COOP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int NB = 128;        // outer block width
constexpr int MT = 32;         // micro tile width
constexpr int SUB = NB / MT;   // sub-steps per diagonal block
constexpr int THREADS = 256;   // 8 warps per CTA
constexpr int LDS = NB + 4;    // padded shared leading dim (132, mult of 4)
constexpr int LDT = MT + 4;    // padded 32x32 staging leading dim (36)
constexpr int LDD = NB + 5;    // Phase-D leading dim (133; gcd(5,32)=1 so
                               // column accesses are bank-conflict-free)
constexpr int SMEM_FLOATS = NB * LDD;            // 128*133 = 17024
constexpr int LDH = 136;       // fp16 trailing staging leading dim (mult 8)
// Phase D needs 128*133 fp32 = 68096 B; Phase T stages two 128x128 fp16
// tiles at 128*136*2 = 34816 B each = 69632 B. Take the max.
constexpr int SMEM_BYTES = 2 * NB * LDH * 2;     // 69632

__device__ __forceinline__ long long _globaltimer() {
    long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// fp32 rank-32 sub-trailing inside the diagonal block, size-specialized so
// the index divisions compile to multiply-shift.
template <int RR>
__device__ __forceinline__ void _subtrail32(float (*ds)[NB + 5], int o2,
                                            int o, int tid) {
    for (int idx = tid; idx < RR * (RR + 1) / 2; idx += 256) {
        int r = (int)((sqrtf(8.0f * idx + 1.0f) - 1.0f) * 0.5f);
        while ((r + 1) * (r + 2) / 2 <= idx) ++r;
        while (r * (r + 1) / 2 > idx) --r;
        const int c = idx - r * (r + 1) / 2;
        {
            float v = ds[o2 + r][o2 + c];
            #pragma unroll
            for (int p = 0; p < 32; ++p)
                v = fmaf(-ds[o2 + r][o + p], ds[o2 + c][o + p], v);
            ds[o2 + r][o2 + c] = v;
        }
    }
}

template <int N, int MATS>
__global__ void coop_cholesky(float* mats, float* dinvg, long long* prof) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float smem[];
    __shared__ float pivots[8][32];   // double-buffered rank-4 pivot columns
    __shared__ float recs[32];        // 1/L[p][p] reciprocals per pivot row
    __shared__ float dinv_sh[32][33];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const bool prof_lead = prof != nullptr && blockIdx.x == 0 && tid == 0;
    long long tprev = prof_lead ? _globaltimer() : 0;

    for (int kt = 0; kt < N / NB; ++kt) {
        const int k0 = kt * NB;

        // ---- Phase D: factor the 128x128 diagonal block, fp32 ----------
        if (blockIdx.x < MATS) {
            const int m = blockIdx.x;
            float* M = mats + (size_t)m * N * N;
            float (*ds)[LDD] = reinterpret_cast<float (*)[LDD]>(smem);
            long long td = prof_lead ? _globaltimer() : 0;
            for (int idx = tid; idx < NB * NB / 4; idx += THREADS) {
                const int row = idx / (NB / 4);
                const int c4 = 4 * (idx % (NB / 4));
                const float4 v = *reinterpret_cast<const float4*>(
                    &M[(size_t)(k0 + row) * N + k0 + c4]);
                ds[row][c4] = v.x;
                ds[row][c4 + 1] = v.y;
                ds[row][c4 + 2] = v.z;
                ds[row][c4 + 3] = v.w;
            }
            __syncthreads();
            if (prof_lead) {
                const long long t = _globaltimer();
                prof[4] += t - td;
                td = t;
            }
            for (int s = 0; s < SUB; ++s) {
                const int o = s * MT;
                // (1) warps 0-3: column-split rank-4 factorization of the
                //     32x32 sub-diagonal. Warp w owns columns [8w, 8w+8);
                //     lane = row. The owning warp runs the serial rank-4
                //     pivot chain on its own registers, publishes the four
                //     scaled pivot columns through a double-buffered shared
                //     stage, and every warp applies the 4-FMA update to its
                //     own eight columns: one __syncthreads per stage instead
                //     of a 128-FMA single-warp serial update.
                {
                    float colv[8];
                    if (warp < 4) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            colv[j] = ds[o + lane][o + 8 * warp + j];
                    }
                    #pragma unroll
                    for (int it = 0; it < 8; ++it) {
                        const int p0 = 4 * it;
                        const int p1 = p0 + 1;
                        const int p2 = p0 + 2;
                        const int p3 = p0 + 3;
                        const int ow = it >> 1;
                        const int jb = p0 - 8 * ow;
                        float* pv0 = pivots[(it & 1) ? 4 : 0];
                        float* pv1 = pv0 + 32;
                        float* pv2 = pv1 + 32;
                        float* pv3 = pv2 + 32;
                        if (warp == ow) {
                            const float rec0 = rsqrtf(
                                __shfl_sync(0xffffffffu, colv[jb + 0], p0));
                            const float s0 =
                                (lane >= p0) ? colv[jb + 0] * rec0 : 0.0f;
                            colv[jb + 0] = s0;
                            pv0[lane] = s0;
                            if (lane == p0) recs[p0] = rec0;
                            const float c01 =
                                __shfl_sync(0xffffffffu, s0, p1);
                            const float c02 =
                                __shfl_sync(0xffffffffu, s0, p2);
                            const float c03 =
                                __shfl_sync(0xffffffffu, s0, p3);
                            colv[jb + 1] = (lane >= p1)
                                ? fmaf(-s0, c01, colv[jb + 1]) : colv[jb + 1];
                            colv[jb + 2] = (lane >= p2)
                                ? fmaf(-s0, c02, colv[jb + 2]) : colv[jb + 2];
                            colv[jb + 3] = (lane >= p3)
                                ? fmaf(-s0, c03, colv[jb + 3]) : colv[jb + 3];
                            const float rec1 = rsqrtf(
                                __shfl_sync(0xffffffffu, colv[jb + 1], p1));
                            const float s1 =
                                (lane >= p1) ? colv[jb + 1] * rec1 : 0.0f;
                            colv[jb + 1] = s1;
                            pv1[lane] = s1;
                            if (lane == p1) recs[p1] = rec1;
                            const float c12 =
                                __shfl_sync(0xffffffffu, s1, p2);
                            const float c13 =
                                __shfl_sync(0xffffffffu, s1, p3);
                            colv[jb + 2] = (lane >= p2)
                                ? fmaf(-s1, c12, colv[jb + 2]) : colv[jb + 2];
                            colv[jb + 3] = (lane >= p3)
                                ? fmaf(-s1, c13, colv[jb + 3]) : colv[jb + 3];
                            const float rec2 = rsqrtf(
                                __shfl_sync(0xffffffffu, colv[jb + 2], p2));
                            const float s2 =
                                (lane >= p2) ? colv[jb + 2] * rec2 : 0.0f;
                            colv[jb + 2] = s2;
                            pv2[lane] = s2;
                            if (lane == p2) recs[p2] = rec2;
                            const float c23 =
                                __shfl_sync(0xffffffffu, s2, p3);
                            colv[jb + 3] = (lane >= p3)
                                ? fmaf(-s2, c23, colv[jb + 3]) : colv[jb + 3];
                            const float rec3 = rsqrtf(
                                __shfl_sync(0xffffffffu, colv[jb + 3], p3));
                            const float s3 =
                                (lane >= p3) ? colv[jb + 3] * rec3 : 0.0f;
                            colv[jb + 3] = s3;
                            pv3[lane] = s3;
                            if (lane == p3) recs[p3] = rec3;
                        }
                        __syncthreads();
                        if (warp < 4) {
                            const float t0 = pv0[lane];
                            const float t1 = pv1[lane];
                            const float t2 = pv2[lane];
                            const float t3 = pv3[lane];
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const int c = 8 * warp + j;
                                if (c > p3 && c <= lane) {
                                    float v = colv[j];
                                    v = fmaf(-t0, pv0[c], v);
                                    v = fmaf(-t1, pv1[c], v);
                                    v = fmaf(-t2, pv2[c], v);
                                    v = fmaf(-t3, pv3[c], v);
                                    colv[j] = v;
                                }
                            }
                        }
                    }
                    if (warp < 4) {
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int c = 8 * warp + j;
                            ds[o + lane][o + c] =
                                (c <= lane) ? colv[j] : 0.0f;
                        }
                    }
                }
                __syncthreads();
                const int rr = NB - MT - o;
                const int o2 = o + MT;
                // Concurrent region: warp 0 computes the exact fp32 32x32
                // inverse (rank-1 elimination order: the dependence chain is
                // 32 multiplies, not 32x32 FMAs); warps 1-3 solve the
                // in-block sub-panel rows by register forward substitution
                // (no Dinv dependency); both read only the factored L tile.
                if (warp == 0) {
                    float x[32];
                    #pragma unroll
                    for (int r = 0; r < 32; ++r)
                        x[r] = (r == lane) ? 1.0f : 0.0f;
                    #pragma unroll
                    for (int p = 0; p < 32; ++p) {
                        const float xp =
                            (p >= lane) ? x[p] * recs[p] : 0.0f;
                        x[p] = xp;
                        #pragma unroll
                        for (int r2 = p + 1; r2 < 32; ++r2)
                            x[r2] = fmaf(-ds[o + r2][o + p], xp, x[r2]);
                        dinv_sh[p][lane] = xp;
                    }
                } else if (warp <= 3 && rr > 0) {
                    const int r = (warp - 1) * 32 + lane;
                    if (r < rr) {
                        float xv[32];
                        #pragma unroll
                        for (int c = 0; c < 32; ++c)
                            xv[c] = ds[o2 + r][o + c];
                        #pragma unroll
                        for (int p = 0; p < 32; ++p) {
                            const float xp = xv[p] * recs[p];
                            xv[p] = xp;
                            #pragma unroll
                            for (int c = p + 1; c < 32; ++c)
                                xv[c] = fmaf(-ds[o + c][o + p], xp, xv[c]);
                        }
                        #pragma unroll
                        for (int c = 0; c < 32; ++c)
                            ds[o2 + r][o + c] = xv[c];
                    }
                }
                __syncthreads();
                if (prof_lead) {
                    const long long t = _globaltimer();
                    prof[5] += t - td;
                    td = t;
                }
                if (rr > 0) {
                    if (prof_lead) {
                        const long long t = _globaltimer();
                        prof[6] += t - td;
                        td = t;
                    }
                    // (4) fp32 rank-32 sub-trailing, size-specialized.
                    if (rr == 96) _subtrail32<96>(ds, o2, o, tid);
                    else if (rr == 64) _subtrail32<64>(ds, o2, o, tid);
                    else _subtrail32<32>(ds, o2, o, tid);
                }
                for (int idx = tid; idx < MT * MT; idx += THREADS)
                    dinvg[((size_t)m * SUB + s) * (MT * MT) + idx] =
                        dinv_sh[idx >> 5][idx & 31];
                __syncthreads();
                if (prof_lead) {
                    const long long t = _globaltimer();
                    prof[7] += t - td;
                    td = t;
                }
            }
            for (int idx = tid; idx < NB * NB / 4; idx += THREADS) {
                const int row = idx / (NB / 4);
                const int c4 = 4 * (idx % (NB / 4));
                const float4 v = make_float4(
                    ds[row][c4], ds[row][c4 + 1],
                    ds[row][c4 + 2], ds[row][c4 + 3]);
                *reinterpret_cast<float4*>(
                    &M[(size_t)(k0 + row) * N + k0 + c4]) = v;
            }
            __syncthreads();
            if (prof_lead) prof[4] += _globaltimer() - td;
        }
        grid.sync();
        if (prof_lead) {
            const long long t = _globaltimer();
            prof[0] += t - tprev;
            tprev = t;
        }

        const int r = N - k0 - NB;
        if (r > 0) {
            // ---- Phase P: panel TRSM by 32-row strips, tf32 WMMA -------
            {
                const int S = r / MT;
                const int total = MATS * S;
                float (*xs)[LDS] = reinterpret_cast<float (*)[LDS]>(smem);
                float (*xt)[LDS] =
                    reinterpret_cast<float (*)[LDS]>(smem + MT * LDS);
                float* base = smem + 2 * MT * LDS;
                float (*lb)[LDT] = reinterpret_cast<float (*)[LDT]>(base);
                float (*db)[LDT] =
                    reinterpret_cast<float (*)[LDT]>(base + MT * LDT);
                float (*ts)[LDT] =
                    reinterpret_cast<float (*)[LDT]>(base + 2 * MT * LDT);
                for (int item = blockIdx.x; item < total;
                     item += gridDim.x) {
                    const int m = item / S;
                    const int si = item % S;
                    const int row0 = k0 + NB + si * MT;
                    float* M = mats + (size_t)m * N * N;
                    __syncthreads();
                    for (int idx = tid; idx < MT * NB; idx += THREADS)
                        xs[idx / NB][idx % NB] =
                            M[(size_t)(row0 + idx / NB) * N + k0 + idx % NB];
                    __syncthreads();
                    const int wr = (warp >> 1) & 1;
                    const int wc = warp & 1;
                    for (int cb = 0; cb < SUB; ++cb) {
                        wmma::fragment<wmma::accumulator, 16, 16, 8, float>
                            acc;
                        wmma::fill_fragment(acc, 0.0f);
                        for (int j = 0; j < cb; ++j) {
                            __syncthreads();
                            for (int idx = tid; idx < MT * MT;
                                 idx += THREADS)
                                lb[idx >> 5][idx & 31] = wmma::__float_to_tf32(
                                    M[(size_t)(k0 + cb * MT + (idx >> 5)) * N
                                      + k0 + j * MT + (idx & 31)]);
                            __syncthreads();
                            if (warp < 4) {
                                #pragma unroll
                                for (int kk = 0; kk < MT; kk += 8) {
                                    wmma::fragment<
                                        wmma::matrix_a, 16, 16, 8,
                                        wmma::precision::tf32,
                                        wmma::row_major> af;
                                    wmma::fragment<
                                        wmma::matrix_b, 16, 16, 8,
                                        wmma::precision::tf32,
                                        wmma::col_major> bf;
                                    wmma::load_matrix_sync(
                                        af, &xt[wr * 16][j * MT + kk], LDS);
                                    wmma::load_matrix_sync(
                                        bf, &lb[wc * 16][kk], LDT);
                                    wmma::mma_sync(acc, af, bf, acc);
                                }
                            }
                        }
                        __syncthreads();
                        if (warp < 4)
                            wmma::store_matrix_sync(
                                &ts[wr * 16][wc * 16], acc, LDT,
                                wmma::mem_row_major);
                        __syncthreads();
                        for (int idx = tid; idx < MT * MT; idx += THREADS) {
                            const int rw = idx >> 5, cw = idx & 31;
                            lb[rw][cw] = wmma::__float_to_tf32(
                                xs[rw][cb * MT + cw] - ts[rw][cw]);
                            db[rw][cw] = wmma::__float_to_tf32(
                                dinvg[((size_t)m * SUB + cb) * (MT * MT)
                                      + idx]);
                        }
                        __syncthreads();
                        if (warp < 4) {
                            wmma::fragment<wmma::accumulator, 16, 16, 8,
                                           float> sol;
                            wmma::fill_fragment(sol, 0.0f);
                            #pragma unroll
                            for (int kk = 0; kk < MT; kk += 8) {
                                wmma::fragment<
                                    wmma::matrix_a, 16, 16, 8,
                                    wmma::precision::tf32,
                                    wmma::row_major> af;
                                wmma::fragment<
                                    wmma::matrix_b, 16, 16, 8,
                                    wmma::precision::tf32,
                                    wmma::col_major> bf;
                                wmma::load_matrix_sync(
                                    af, &lb[wr * 16][kk], LDT);
                                wmma::load_matrix_sync(
                                    bf, &db[wc * 16][kk], LDT);
                                wmma::mma_sync(sol, af, bf, sol);
                            }
                            wmma::store_matrix_sync(
                                &xs[wr * 16][cb * MT + wc * 16], sol, LDS,
                                wmma::mem_row_major);
                        }
                        __syncthreads();
                        for (int idx = tid; idx < MT * MT; idx += THREADS)
                            xt[idx >> 5][cb * MT + (idx & 31)] =
                                wmma::__float_to_tf32(
                                    xs[idx >> 5][cb * MT + (idx & 31)]);
                        __syncthreads();
                    }
                    for (int idx = tid; idx < MT * NB; idx += THREADS)
                        M[(size_t)(row0 + idx / NB) * N + k0 + idx % NB] =
                            xs[idx / NB][idx % NB];
                }
            }
            grid.sync();
            if (prof_lead) {
                const long long t = _globaltimer();
                prof[1] += t - tprev;
                tprev = t;
            }

            // ---- Phase T: rank-128 trailing SYRK, 128x128 fp16 tiles ---
            // fp16 products with fp32 accumulation -- the shipped
            // FP16_TRAILING structure at this shape (one rounding of each
            // panel element per outer step). 128-tiles halve staging traffic
            // versus 64-tiles and keep every CTA on one tile per round.
            {
                const int t128 = r / NB;
                const long pairs = (long)t128 * (t128 + 1) / 2;
                const long total = (long)MATS * pairs;
                __half (*ah)[LDH] = reinterpret_cast<__half (*)[LDH]>(smem);
                __half (*bh)[LDH] = reinterpret_cast<__half (*)[LDH]>(
                    reinterpret_cast<char*>(smem) + NB * LDH * 2);
                for (long item = blockIdx.x; item < total;
                     item += gridDim.x) {
                    const int m = (int)(item / pairs);
                    const int p = (int)(item % pairs);
                    int prow = (int)((sqrtf(8.0f * p + 1.0f) - 1.0f) * 0.5f);
                    while ((prow + 1) * (prow + 2) / 2 <= p) ++prow;
                    while (prow * (prow + 1) / 2 > p) --prow;
                    const int pcol = p - prow * (prow + 1) / 2;
                    float* M = mats + (size_t)m * N * N;
                    const int bi = k0 + NB + prow * NB;
                    const int bj = k0 + NB + pcol * NB;
                    __syncthreads();
                    for (int idx = tid; idx < NB * NB; idx += THREADS) {
                        const int rw = idx / NB, cw = idx % NB;
                        ah[rw][cw] = __float2half_rn(
                            M[(size_t)(bi + rw) * N + k0 + cw]);
                        bh[rw][cw] = __float2half_rn(
                            M[(size_t)(bj + rw) * N + k0 + cw]);
                    }
                    __syncthreads();
                    // 8 warps x (16 rows x 128 cols) each.
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                        acc[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        wmma::fill_fragment(acc[j], 0.0f);
                    #pragma unroll
                    for (int kk = 0; kk < NB; kk += 16) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
                                       wmma::row_major> af;
                        wmma::load_matrix_sync(af, &ah[warp * 16][kk], LDH);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                           __half, wmma::col_major> bf;
                            wmma::load_matrix_sync(
                                bf, &bh[j * 16][kk], LDH);
                            wmma::mma_sync(acc[j], af, bf, acc[j]);
                        }
                    }
                    float* crow = M + (size_t)(bi + warp * 16) * N + bj;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        wmma::fragment<wmma::accumulator, 16, 16, 16, float>
                            orig;
                        wmma::load_matrix_sync(orig, crow + j * 16, N,
                                               wmma::mem_row_major);
                        #pragma unroll
                        for (int e = 0; e < orig.num_elements; ++e)
                            orig.x[e] -= acc[j].x[e];
                        wmma::store_matrix_sync(crow + j * 16, orig, N,
                                                wmma::mem_row_major);
                    }
                }
            }
        }
        grid.sync();
        if (prof_lead) {
            const long long t = _globaltimer();
            prof[2] += t - tprev;
            tprev = t;
        }
    }

    // ---- strict upper cleanup ------------------------------------------
    const size_t total_e = (size_t)MATS * N * N;
    const size_t stride = (size_t)gridDim.x * THREADS;
    for (size_t idx = (size_t)blockIdx.x * THREADS + tid; idx < total_e;
         idx += stride) {
        const size_t within = idx % ((size_t)N * N);
        const int rw = (int)(within / N);
        const int cw = (int)(within % N);
        if (cw > rw) mats[idx] = 0.0f;
    }
    if (prof_lead) prof[3] += _globaltimer() - tprev;
}

template <int N, int MATS>
void launch_coop(torch::Tensor mats, torch::Tensor scratch, long long* prof) {
    TORCH_CHECK(mats.dim() == 3 && mats.size(0) == MATS &&
                mats.size(1) == N && mats.size(2) == N,
                "coop_cholesky shape mismatch");
    static int grid_size = -1;
    if (grid_size < 0) {
        cudaFuncSetAttribute(
            (void*)coop_cholesky<N, MATS>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (void*)coop_cholesky<N, MATS>, THREADS, SMEM_BYTES);
        TORCH_CHECK(per_sm >= 1, "coop_cholesky: zero occupancy");
        grid_size = props.multiProcessorCount;
    }
    float* mp = mats.data_ptr<float>();
    float* sp = scratch.data_ptr<float>();
    void* args[] = {&mp, &sp, &prof};
    cudaError_t status = cudaLaunchCooperativeKernel(
        (void*)coop_cholesky<N, MATS>, dim3(grid_size), dim3(THREADS), args,
        SMEM_BYTES, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void coop_1024x4(torch::Tensor mats, torch::Tensor scratch) {
    launch_coop<1024, 4>(mats, scratch, nullptr);
}

void coop_1024x4_profile(torch::Tensor mats, torch::Tensor scratch,
                         torch::Tensor prof) {
    launch_coop<1024, 4>(mats, scratch,
                         (long long*)prof.data_ptr<int64_t>());
}
"""

if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline

        _COOP = load_inline(
            name="coop_chol_exp050_v5",
            cpp_sources=(
                "void coop_1024x4(torch::Tensor, torch::Tensor);\n"
                "void coop_1024x4_profile(torch::Tensor, torch::Tensor,"
                " torch::Tensor);"
            ),
            cuda_sources=_COOP_SOURCE,
            functions=["coop_1024x4", "coop_1024x4_profile"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        _COOP1024_ERROR = repr(exc)


def _scratch_for(mats, sub=4, mt=32):
    key = (mats.shape[0], mats.device.index)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(
            mats.shape[0] * sub * mt * mt,
            device=mats.device,
            dtype=torch.float32,
        )
        _SCRATCH[key] = buf
    return buf


def custom_kernel(data):
    global _COOP1024_HITS, _COOP1024_FALLBACKS
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (4, 1024, 1024)
        and data.is_contiguous()
    )
def mid_probe():
    """Phase-level device timing of the cooperative kernel (midprobe mode).

    Run with --submission pointing at THIS file; the dense inputs never take
    the fallback, so the self-import never recurses.
    """
    import statistics

    rows = []
    if _COOP is None:
        return [{"name": "compile", "us": 0.0, "ok": False,
                 "error": (_COOP1024_ERROR or "no cuda")[:400]}]
    dev = torch.device("cuda")
    torch.manual_seed(20500721)
    a = torch.randn(4, 1024, 1024, device=dev)
    spd = torch.baddbmm(
        torch.eye(1024, device=dev).expand(4, 1024, 1024) * 1024,
        a, a.transpose(1, 2), beta=1.0, alpha=1.0 / 1024,
    ).contiguous()
    del a

    out = custom_kernel(spd)
    recon = torch.bmm(out, out.transpose(1, 2))
    err = (recon - spd).abs().max().item()
    scale = spd.abs().max().item()
    rows.append({
        "name": "correctness 4x1024 dense", "us": 0.0,
        "rel": err / max(scale, 1e-30),
        "hits": _COOP1024_HITS, "fallbacks": _COOP1024_FALLBACKS,
        "ok": bool(err / max(scale, 1e-30) < 1e-4
                   and _COOP1024_HITS > 0 and _COOP1024_FALLBACKS == 0),
    })
    del out, recon

    def _time(fn, iters=30, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters

    rows.append({
        "name": "custom_kernel end-to-end",
        "us": _time(lambda: custom_kernel(spd)),
        "ok": True,
    })
    work = spd.clone()
    scratch = _scratch_for(work)
    rows.append({
        "name": "bare coop kernel (in-place, garbage after iter 1)",
        "us": _time(lambda: _COOP.coop_1024x4(work, scratch)),
        "ok": True,
    })

    phases = []
    for _ in range(7):
        work = spd.clone()
        prof = torch.zeros(8, dtype=torch.int64, device=dev)
        _COOP.coop_1024x4_profile(work, scratch, prof)
        torch.cuda.synchronize()
        phases.append([v / 1000.0 for v in prof.tolist()])
    labels = ["phase_diag", "phase_panel", "phase_trailing", "phase_cleanup",
              "d_load_store", "d_micro_inv", "d_subpanel",
              "d_subtrail_dinvg"]
    for i, label in enumerate(labels):
        rows.append({
            "name": label,
            "us": statistics.median(p[i] for p in phases),
            "ok": True,
        })
    return rows


def custom_kernel(data):
    global _COOP1024_HITS, _COOP1024_FALLBACKS
    target = (
        data.is_cuda
        and data.dtype == torch.float32
        and data.shape == (4, 1024, 1024)
        and data.is_contiguous()
    )
    if target and _COOP is not None:
        # Fast cooperative path with the shipped finite-diagonal retry
        # contract: the full diagonal must be finite (a shorter check is
        # invalid -- finite/Inf == 0 absorbs an overflowed pivot into a zero
        # column), otherwise fall through to the exact ranked dispatch.
        output = data.clone()
        _COOP.coop_1024x4(output, _scratch_for(output))
        if torch.isfinite(output.diagonal(dim1=-2, dim2=-1)).all().item():
            _COOP1024_HITS += 1
            return output
        _COOP1024_FALLBACKS += 1
    elif target:
        _COOP1024_FALLBACKS += 1
    return _ranked.custom_kernel(data)
