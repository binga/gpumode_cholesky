

# ---------------------------------------------------------------------------
# Experiment 063 round 1 -- collapse the block kernel's two serial phases.
#
# exp 062 left `e62_diag128` at 375 ns/row (48-50us per 128x128 block) with
# 61% of that in two phases that run on ONE warp while the other seven idle:
#
#     chain   14.9us (31%)     32x32 register Cholesky, 4 times
#     triinv  14.6us (30%)     32x32 triangular inverse, 4 times
#
# Both are replaced by a single fused Gauss-Jordan that produces L and inv(L)
# in one pass (round 1 of exp 062 proved this is numerically fine: inverse
# error 2.4e-07). Two implementation changes make the fused version cheap
# where round 1's was not:
#
#   1. 4x8 register tiling instead of one-row-per-lane. Lane (ri, cj) owns
#      rows 4ri..4ri+3 and columns 8cj..8cj+7 of both the working tile and the
#      inverse. The per-pivot cross-lane traffic drops from 32 `shfl` (which
#      issue at quarter rate) to THREE `LDS.128` broadcasts of the pivot
#      column plus two of the pivot row of the inverse.
#   2. Partial unrolling. The pivot index only has to be a compile-time
#      constant modulo 8 (`k & 3` picks a row register, `k & 7` picks a column
#      register), so the outer loop over the four groups of eight pivots stays
#      a real loop. Round 1 unrolled all 32 pivots x 32 columns into ~6k
#      instructions, which does not fit the instruction cache; this version is
#      one eighth of that.
#
# Variant 0 is the shipped exp-062 kernel, compiled from the same source in
# the same extension, so `mid_probe` measures both under identical conditions.
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
// Variant 2 stages a double-buffered 128-float pivot column plus a
// double-buffered 32-float inverse row in the scratch area.
#define E62_SMEM_F  (E62_T_OFF + 384)
#define E62_SMEM_B  (E62_SMEM_F * 4)

#define E62_PROF 8

// Pure compiler barrier, no instructions emitted. Rounds 1 and 2 both staged
// values through shared memory inside a *loop*, which gives the optimizer far
// more scope to hoist or cache a load than exp-062's straight-line staging
// does. `__syncwarp()` / `__syncthreads()` order the hardware; this stops the
// compiler from moving a shared load across them.
#define E62_CBAR() asm volatile("" ::: "memory")

// --------------------------------------------------------------------------
// Variant 0 -- shipped exp-062 chain (63.3 ns/pivot isolated) + separate
// two-level triangular inverse. Kept verbatim as the in-source control.
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

__device__ __forceinline__ void e62_tri_inv32(const float* __restrict__ Sb,
                                              float* __restrict__ Qi,
                                              float* __restrict__ Tmp, int lane)
{
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
// Variant 1 -- fused Cholesky + inverse, 4x8 register tiles, one warp.
//
//   lane = ri * 4 + cj,  ri in 0..7 (rows 4ri..4ri+3),
//                        cj in 0..3 (cols 8cj..8cj+7)
//   r[u][v] = A[4ri+u][8cj+v]      -> becomes L
//   m[u][v] = M[4ri+u][8cj+v]      -> becomes inv(L), seeded with I
//
// Per pivot k the warp needs the whole column L[:,k] (for both the row and
// the column operand of the rank-1 update) and the whole row M[k,:]. Both are
// staged through 32-float shared scratch buffers, so each lane reads three
// float4 for the column and two for the inverse row -- five shared
// instructions instead of thirty-two shuffles.
//
// The column that is finished at pivot k must survive the rank-1 update; the
// lanes that own it zero their copy of the pivot element (`colv[kv] = 0`) and
// write the finished L column back into the register tile afterwards. Columns
// finished at earlier pivots are protected automatically, because L[k'][k] is
// zero for k' < k and that zero is what the staging buffer holds.
// --------------------------------------------------------------------------
__device__ __forceinline__ void e62_chain32_fused(float* __restrict__ Sb,
                                                  float* __restrict__ Qi,
                                                  float* Scr,
                                                  int lane)
{
    const int ri = lane >> 2;
    const int cj = lane & 3;
    const int i0 = ri << 2;
    const int j0 = cj << 3;
    float* Lk = Scr;            // column k of L
    float* Mk = Scr + 32;       // row k of inv(L)

    float r[4][8];
    float m[4][8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float4* s = (const float4*)(Sb + (i0 + u) * E62_LD + j0);
        const float4 x0 = s[0];
        const float4 x1 = s[1];
        r[u][0] = x0.x; r[u][1] = x0.y; r[u][2] = x0.z; r[u][3] = x0.w;
        r[u][4] = x1.x; r[u][5] = x1.y; r[u][6] = x1.z; r[u][7] = x1.w;
        #pragma unroll
        for (int v = 0; v < 8; ++v)
            m[u][v] = ((i0 + u) == (j0 + v)) ? 1.0f : 0.0f;
    }

    for (int kb = 0; kb < 4; ++kb) {          // deliberately NOT unrolled
        const bool colown = (cj == kb);
        #pragma unroll
        for (int kv = 0; kv < 8; ++kv) {
            const int k  = (kb << 3) + kv;
            const int ku = kv & 3;                  // k & 3   (compile time)
            const int kr = (kb << 1) + (kv >> 2);   // k >> 2
            const float akk =
                __shfl_sync(0xffffffffu, r[ku][kv], (kr << 2) + kb);
            const float d = rsqrtf(akk);

            if (colown) {
                float4 lv;
                lv.x = (i0 + 0 >= k) ? r[0][kv] * d : 0.0f;
                lv.y = (i0 + 1 >= k) ? r[1][kv] * d : 0.0f;
                lv.z = (i0 + 2 >= k) ? r[2][kv] * d : 0.0f;
                lv.w = (i0 + 3 >= k) ? r[3][kv] * d : 0.0f;
                *(float4*)(Lk + i0) = lv;
            }
            if (ri == kr) {
                float4 y0, y1;
                y0.x = m[ku][0] * d; y0.y = m[ku][1] * d;
                y0.z = m[ku][2] * d; y0.w = m[ku][3] * d;
                y1.x = m[ku][4] * d; y1.y = m[ku][5] * d;
                y1.z = m[ku][6] * d; y1.w = m[ku][7] * d;
                m[ku][0] = y0.x; m[ku][1] = y0.y;
                m[ku][2] = y0.z; m[ku][3] = y0.w;
                m[ku][4] = y1.x; m[ku][5] = y1.y;
                m[ku][6] = y1.z; m[ku][7] = y1.w;
                *(float4*)(Mk + j0)     = y0;
                *(float4*)(Mk + j0 + 4) = y1;
            }
            E62_CBAR();
            __syncwarp();
            E62_CBAR();

            const float4 rw = *(const float4*)(Lk + i0);
            const float4 c0 = *(const float4*)(Lk + j0);
            const float4 c1 = *(const float4*)(Lk + j0 + 4);
            const float4 g0 = *(const float4*)(Mk + j0);
            const float4 g1 = *(const float4*)(Mk + j0 + 4);

            const float rowv[4] = {rw.x, rw.y, rw.z, rw.w};
            float colv[8] = {c0.x, c0.y, c0.z, c0.w,
                             c1.x, c1.y, c1.z, c1.w};
            const float mrow[8] = {g0.x, g0.y, g0.z, g0.w,
                                   g1.x, g1.y, g1.z, g1.w};
            if (colown) colv[kv] = 0.0f;
            float rowm[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u)
                rowm[u] = ((i0 + u) == k) ? 0.0f : rowv[u];

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    r[u][v] -= rowv[u] * colv[v];
                    m[u][v] -= rowm[u] * mrow[v];
                }
            }
            if (colown) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) r[u][kv] = rowv[u];
            }
            E62_CBAR();
            __syncwarp();
            E62_CBAR();
        }
    }

    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float4 x0 = make_float4(r[u][0], r[u][1], r[u][2], r[u][3]);
        const float4 x1 = make_float4(r[u][4], r[u][5], r[u][6], r[u][7]);
        *(float4*)(Sb + (i0 + u) * E62_LD + j0)     = x0;
        *(float4*)(Sb + (i0 + u) * E62_LD + j0 + 4) = x1;
        #pragma unroll
        for (int v = 0; v < 8; ++v)
            Qi[(i0 + u) * E62_QLD + j0 + v] = m[u][v];
    }
}

// --------------------------------------------------------------------------
// Variant 2 -- 256-thread panel factorization.
//
// Variants 0 and 1 both leave 61% of the block on ONE warp, and a single warp
// cannot hide shared-memory latency: the measured chain sits at 119-146
// ns/pivot against a ~63 ns/pivot instruction-issue estimate, because every
// pivot's staging store -> barrier -> load is exposed end to end.
//
// This variant hands the serial phase to all eight warps instead. The whole
// 128x32 column panel is factored together, which subsumes THREE phases at
// once -- the 32x32 pivot chain, its triangular inverse, and the panel solve
// that applied that inverse to the rows below -- because a right-looking
// rank-1 update over 128 rows produces L21 directly.
//
//   thread -> tr = tid >> 3 (0..31, rows 4tr..4tr+3)
//             tc = tid & 7  (0..7,  panel columns 4tc..4tc+3)
//   t[4][4]  = S[4tr+u][kk+4tc+v]
//   mt[4][4] = inv(L11)[4tr+u-kk][4tc+v], carried only by the eight row
//              groups that lie inside the pivot block
//
// One `__syncthreads()` per pivot, not two: the staging buffers are double
// buffered, so pivot k+1 writes the buffer pivot k did not read. A thread can
// only reach pivot k+2's store after barrier k+1, which every thread's pivot-k
// read precedes.
//
// The pivot column is staged RAW and scaled after the barrier, so the
// reciprocal square root does not have to be known before the staging store --
// that is what removes the second barrier.
// --------------------------------------------------------------------------
// `WITHINV == 0` factors the panel only and leaves the 32x32 inverse to the
// shipped `e62_tri_inv32`. That splits the round-2 failure in half: if the
// L-only build is exact, the defect is in the fused inverse, not in the
// column-protection scheme the two share.
template <int WITHINV>
__device__ __forceinline__ void e62_panel32(float* S,
                                            float* Qi,
                                            float* Scr,
                                            int tid, int kk)
{
    const int tr = tid >> 3;
    const int tc = tid & 7;
    const int r0 = tr << 2;
    const int c0 = kk + (tc << 2);
    const int ib = kk >> 2;
    const bool inv_thread = (WITHINV != 0) && (tr >= ib) && (tr < ib + 8);
    const int mrow0 = r0 - kk;

    float t[4][4];
    float mt[4][4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float4 x = *(const float4*)(S + (r0 + u) * E62_LD + c0);
        t[u][0] = x.x; t[u][1] = x.y; t[u][2] = x.z; t[u][3] = x.w;
        #pragma unroll
        for (int v = 0; v < 4; ++v)
            mt[u][v] = (inv_thread && (mrow0 + u) == ((tc << 2) + v))
                       ? 1.0f : 0.0f;
    }

    for (int kq = 0; kq < 8; ++kq) {          // deliberately NOT unrolled
        const bool colown = (tc == kq);
        const bool rowown = (tr == ib + kq);
        #pragma unroll
        for (int kv = 0; kv < 4; ++kv) {
            const int kl = (kq << 2) + kv;
            const int k  = kk + kl;
            float* Lc = Scr + ((kl & 1) << 7);
            float* Mr = Scr + 256 + ((kl & 1) << 5);

            if (colown) {
                float4 lv;
                lv.x = (r0 + 0 >= k) ? t[0][kv] : 0.0f;
                lv.y = (r0 + 1 >= k) ? t[1][kv] : 0.0f;
                lv.z = (r0 + 2 >= k) ? t[2][kv] : 0.0f;
                lv.w = (r0 + 3 >= k) ? t[3][kv] : 0.0f;
                *(float4*)(Lc + r0) = lv;
            }
            if (WITHINV && rowown) {
                *(float4*)(Mr + (tc << 2)) =
                    make_float4(mt[kv][0], mt[kv][1], mt[kv][2], mt[kv][3]);
            }
            E62_CBAR();
            __syncthreads();
            E62_CBAR();

            const float d  = rsqrtf(Lc[k]);
            const float d2 = d * d;
            const float4 rw = *(const float4*)(Lc + r0);
            const float4 cw = *(const float4*)(Lc + c0);
            const float rowv[4] = {rw.x, rw.y, rw.z, rw.w};
            float colv[4] = {cw.x, cw.y, cw.z, cw.w};
            if (colown) colv[kv] = 0.0f;

            float rr[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) rr[u] = rowv[u] * d2;
            #pragma unroll
            for (int u = 0; u < 4; ++u)
                #pragma unroll
                for (int v = 0; v < 4; ++v)
                    t[u][v] -= rr[u] * colv[v];
            if (colown) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) t[u][kv] = rowv[u] * d;
            }

            if (inv_thread) {
                const float4 mw = *(const float4*)(Mr + (tc << 2));
                const float mrv[4] = {mw.x * d, mw.y * d, mw.z * d, mw.w * d};
                float rm[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    rm[u] = ((r0 + u) == k) ? 0.0f : rowv[u] * d;
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    #pragma unroll
                    for (int v = 0; v < 4; ++v)
                        mt[u][v] -= rm[u] * mrv[v];
                if (rowown) {
                    #pragma unroll
                    for (int v = 0; v < 4; ++v) mt[kv][v] = mrv[v];
                }
            }
        }
    }

    #pragma unroll
    for (int u = 0; u < 4; ++u)
        *(float4*)(S + (r0 + u) * E62_LD + c0) =
            make_float4(t[u][0], t[u][1], t[u][2], t[u][3]);
    if (inv_thread) {
        #pragma unroll
        for (int u = 0; u < 4; ++u)
            #pragma unroll
            for (int v = 0; v < 4; ++v)
                Qi[(mrow0 + u) * E62_QLD + (tc << 2) + v] = mt[u][v];
    }
}

// --------------------------------------------------------------------------

template <int VAR>
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

    long long t0 = 0;
    long long ph[E62_PROF];
    #pragma unroll
    for (int i = 0; i < E62_PROF; ++i) ph[i] = 0;
    const bool prof = (Prof != nullptr) && (tid == 0);
    if (prof) t0 = clock64();

    float* Ab = A + (size_t)blockIdx.x * (size_t)n * (size_t)n
                  + (size_t)j * (size_t)n + (size_t)j;

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
        const int nrow  = E62_TB - lwid;
        float* Sb = S + kk * E62_LD + kk;

        // ---- 1+2(+3). pivot chain and its inverse ----
        if (VAR == 0) {
            if (warp == 0) e62_chain32_reg(Sb, lane);
            __syncwarp();
            if (prof) { ph[1] += clock64() - t0; t0 = clock64(); }
            if (warp == 0) e62_tri_inv32(Sb, Qi, Tp, lane);
            __syncthreads();
            if (prof) { ph[2] += clock64() - t0; t0 = clock64(); }
        } else if (VAR == 1) {
            if (warp == 0) e62_chain32_fused(Sb, Qi, Tp, lane);
            __syncthreads();
            if (prof) { ph[1] += clock64() - t0; t0 = clock64(); }
        } else if (VAR == 2) {
            e62_panel32<1>(S, Qi, Tp, tid, kk);
            __syncthreads();
            if (prof) { ph[1] += clock64() - t0; t0 = clock64(); }
        } else {
            // L-only panel factorization, then the shipped triangular inverse.
            e62_panel32<0>(S, Qi, Tp, tid, kk);
            __syncthreads();
            if (prof) { ph[1] += clock64() - t0; t0 = clock64(); }
            if (warp == 0) e62_tri_inv32(Sb, Qi, Tp, lane);
            __syncthreads();
            if (prof) { ph[2] += clock64() - t0; t0 = clock64(); }
        }

        // ---- 3. panel solve: S[r][kk:kk+32] <- S[r][kk:kk+32] * inv(L11)^T
        //         Variants 2 and 3 already produced L21 in the panel phase.
        if (VAR < 2) {
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
        }
        if (prof) { ph[3] += clock64() - t0; t0 = clock64(); }

        // ---- 4. stage P[t][x] = S[x][kk+t] ----
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
        {
            const int nt = nrow >> 2;
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
        if (c0 + 3 > r) {
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

#define E62_DEFAULT_VAR 0

static void e62_configure()
{
    static bool configured = false;
    if (!configured) {
        cudaError_t a0 = cudaFuncSetAttribute(
            (const void*)e62_diag128<0>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, E62_SMEM_B);
        TORCH_CHECK(a0 == cudaSuccess, cudaGetErrorString(a0));
        cudaError_t a1 = cudaFuncSetAttribute(
            (const void*)e62_diag128<1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, E62_SMEM_B);
        TORCH_CHECK(a1 == cudaSuccess, cudaGetErrorString(a1));
        cudaError_t a2 = cudaFuncSetAttribute(
            (const void*)e62_diag128<2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, E62_SMEM_B);
        TORCH_CHECK(a2 == cudaSuccess, cudaGetErrorString(a2));
        cudaError_t a3 = cudaFuncSetAttribute(
            (const void*)e62_diag128<3>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, E62_SMEM_B);
        TORCH_CHECK(a3 == cudaSuccess, cudaGetErrorString(a3));
        configured = true;
    }
}

static void e62_run(torch::Tensor A, torch::Tensor Dinv, int64_t n, int64_t j,
                    torch::Tensor Prof, int variant)
{
    e62_configure();
    const int batch = (int)A.size(0);
    long long* prof =
        Prof.numel() > 0 ? (long long*)Prof.data_ptr<int64_t>() : nullptr;
    if (variant == 0) {
        e62_diag128<0><<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
            A.data_ptr<float>(), Dinv.data_ptr<float>(), prof, (int)n, (int)j);
    } else if (variant == 1) {
        e62_diag128<1><<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
            A.data_ptr<float>(), Dinv.data_ptr<float>(), prof, (int)n, (int)j);
    } else if (variant == 2) {
        e62_diag128<2><<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
            A.data_ptr<float>(), Dinv.data_ptr<float>(), prof, (int)n, (int)j);
    } else {
        e62_diag128<3><<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
            A.data_ptr<float>(), Dinv.data_ptr<float>(), prof, (int)n, (int)j);
    }
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

// Shipped entry point. The signature is byte-identical to exp 062's, so the
// merged-extension declaration never has to change.
void e62_diag128_launch(torch::Tensor A, torch::Tensor Dinv,
                        int64_t n, int64_t j, torch::Tensor Prof)
{
    e62_run(A, Dinv, n, j, Prof, E62_DEFAULT_VAR);
}

// Probe-only entry point: selects the kernel variant explicitly.
void e62_diag128_launch_var(torch::Tensor A, torch::Tensor Dinv,
                            int64_t n, int64_t j, torch::Tensor Prof,
                            int64_t variant)
{
    e62_run(A, Dinv, n, j, Prof, (int)variant);
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
            name="chol_exp063_diag128_v7",
            cpp_sources=(
                "void e62_diag128_launch(torch::Tensor, torch::Tensor, "
                "int64_t, int64_t, torch::Tensor);\n"
                "void e62_diag128_launch_var(torch::Tensor, torch::Tensor, "
                "int64_t, int64_t, torch::Tensor, int64_t);"
            ),
            cuda_sources=_EXP062_SOURCE,
            functions=["e62_diag128_launch", "e62_diag128_launch_var"],
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


def _exp062_factor(data, nb_outer=1024, prof=None, variant=None):
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
                if variant is None:
                    _EXP062.e62_diag128_launch(work, dinv, n, jj, prof)
                else:
                    _EXP062.e62_diag128_launch_var(work, dinv, n, jj, prof,
                                                   variant)
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
    # The exact shipped dispatch. In the probe layout `custom_kernel` has no
    # exp-062 branch, so this is the incumbent route for every shape -- and for
    # 2 <= batch <= 4, n >= 1024 it is `_loop_cholesky`, never the batched
    # vendor call (exp 062's harness note).
    return custom_kernel(x)


_PHASE_NAMES = ("load", "chain", "triinv", "panel", "stageP+Qt", "commit",
                "trailing+inv", "store")

_E62_VARIANTS = (0, 1, 2, 3)
_E62_SHAPE_VARIANTS = (0, 2, 3)


def mid_probe():
    import sys

    sys.path.insert(0, "/root/reference")
    from reference import generate_input

    rows = [{"name": "combined_ext", "us": 0.0,
             "ok": _CUDA128 is not None,
             "error": str(_CUDA128_ERROR)[:600]}]
    _load_exp062()
    if _EXP062 is None:
        rows.append({"name": "load_inline", "us": 0.0, "ok": False,
                     "error": str(_EXP062_ERROR)[:600]})
        return rows

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

    for var in _E62_VARIANTS:
        def _blockrun(_, v=var):
            work[:, :128, :128].copy_(blk)
            _EXP062.e62_diag128_launch_var(work, dinv, 2048, 0, noprof, v)

        us = _e62_time(_blockrun, None, iters=20, warmup=5)
        net = us - us_copy
        rows.append({"name": f"v{var}_diag128_block", "us": round(net, 3),
                     "ns_per_row": round(net * 1000.0 / 128.0, 1), "ok": True})

        work[:, :128, :128].copy_(blk)
        profbuf.zero_()
        _EXP062.e62_diag128_launch_var(work, dinv, 2048, 0, profbuf, var)
        torch.cuda.synchronize()
        l11 = work[:, :128, :128].tril()
        err, scale = _e62_residual(a[:, :128, :128], l11)
        inv_err = float(
            (dinv[0] @ l11[0] - torch.eye(128, device=dev)).abs().max().item()
        )
        row = {"name": f"v{var}_diag128_err", "us": 0.0,
               "abs_err": round(err, 7),
               "inv_err": round(inv_err, 8),
               "ok": err < 1e-3 and inv_err < 1e-3}
        if not row["ok"]:
            # Localise the failure instead of paying another Modal run for it:
            # which rows and columns of L first diverge from the reference.
            ref = torch.linalg.cholesky(a[:, :128, :128].double())[0].float()
            de = (l11[0] - ref).abs()
            bad = (de > 1e-4).nonzero()
            row["bad_count"] = int(bad.shape[0])
            row["first_bad"] = bad[:8].tolist()
            row["row_err"] = [round(v, 6) for v in
                              de.amax(dim=1)[::8].tolist()]
            row["col_err"] = [round(v, 6) for v in
                              de.amax(dim=0)[::8].tolist()]
        rows.append(row)
        cyc = profbuf[:8].tolist()
        total = max(sum(cyc), 1)
        for name, c in zip(_PHASE_NAMES, cyc):
            rows.append({"name": f"v{var}_phase_{name}",
                         "us": round(net * c / total, 3),
                         "cycles": c, "pct": round(100.0 * c / total, 1),
                         "ok": True})

    del work, blk, a
    torch.cuda.empty_cache()

    for (batch, n, seed) in ((2, 2048, 44048), (2, 4096, 514096),
                             (8, 2048, 782048), (4, 1024, 441024),
                             (16, 512, 165120)):
        a = generate_input(batch=batch, n=n, cond=2, seed=seed)
        base = _e62_time(_shipped, a, iters=8, warmup=3)
        rows.append({"name": f"shipped_{batch}x{n}", "us": round(base, 1),
                     "ok": True})
        for var in _E62_SHAPE_VARIANTS:
            for nbo in (1024,):
                if nbo > n:
                    nbo = n
                try:
                    us = _e62_time(
                        lambda x, v=var, q=nbo: _exp062_factor(x, q,
                                                               variant=v),
                        a, iters=8, warmup=3)
                    l = _exp062_factor(a, nbo, variant=var)
                    torch.cuda.synchronize()
                    err, scale = _e62_residual(a, l)
                    rows.append({
                        "name": f"v{var}_{batch}x{n}_nbo{nbo}",
                        "us": round(us, 1),
                        "speedup": round(base / us, 4),
                        "abs_err": round(err, 6),
                        "ok": bool(torch.isfinite(l).all().item()),
                    })
                    del l
                except Exception as exc:
                    rows.append({"name": f"v{var}_{batch}x{n}_nbo{nbo}",
                                 "us": 0.0, "ok": False,
                                 "error": repr(exc)[:240]})
        del a
        torch.cuda.empty_cache()
    return rows


# Enrolled shapes -> outer trailing-update block width.
_EXP062_SHAPES = {(2, 2048): 1024, (2, 4096): 1024}
