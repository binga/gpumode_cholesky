

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
