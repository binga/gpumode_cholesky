

# ---------------------------------------------------------------------------
# Experiment 062 round 2 -- diagnose and repair the resident diagonal kernel.
#
# Round 1 measured `e62_diag128` at 153.77us per 128x128 block (1201 ns/row),
# against a budget of <=21us/block (164 ns/row). It is numerically excellent
# (block residual 0.0, inverse error 2.4e-07), so the algorithm is right and
# only the pivot-chain implementation is wrong.
#
# Prime suspect: the round-1 chain fused L and inv(L) into ONE fully unrolled
# 32x32 Gauss-Jordan carrying `float a[32]` AND `float m[32]` live across 1024
# unrolled bodies with two shuffles each. If ptxas gives up and spills those to
# local memory, every pivot round-trips through DRAM.
#
# Two repairs are measured against it:
#   v2  register chain (a[32] only, 32 shuffles/pivot) + a separate
#       *column-parallel* triangular inverse: lane j solves L x = e_j, so the
#       inverse costs zero cross-lane traffic and ~500 shared reads total.
#   v3  the same split, but the chain itself lives in shared memory, so there
#       is no large register array to spill at all.
#
# Both are compiled with `-Xptxas -v` so the register/spill counts are printed.
# ---------------------------------------------------------------------------

_EXP062 = None
_EXP062_ERROR = None
_EXP062_HITS = 0
_EXP062_FALLBACKS = 0

_EXP062_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define E62_LD    129
#define E62_TB    128
#define E62_QLD   33
#define E62_NW    8
#define E62_NT    256

#define E62_M_OFF   (E62_TB * E62_LD)
#define E62_QI_OFF  (2 * E62_TB * E62_LD)
#define E62_QT_OFF  (E62_QI_OFF + 32 * E62_QLD)
#define E62_SMEM_F  (E62_QT_OFF + 32 * E62_LD)
#define E62_SMEM_B  (E62_SMEM_F * 4)

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

// Column-parallel triangular inverse of the 32x32 lower block `Sb` into `Qi`.
// Lane j owns column j and solves L x = e_j by forward substitution. Every
// L[i][p] read is the same address for all 32 lanes (a shared broadcast) and
// every x_p is lane-local, so there is no cross-lane traffic whatsoever.
__device__ __forceinline__ void e62_tri_inv32(const float* __restrict__ Sb,
                                              float* __restrict__ Qi,
                                              int lane)
{
    float x[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float acc = (i == lane) ? 1.0f : 0.0f;
        #pragma unroll
        for (int p = 0; p < 32; ++p) {
            if (p < i) acc -= Sb[i * E62_LD + p] * x[p];
        }
        const float dii = Sb[i * E62_LD + i];
        x[i] = (i >= lane) ? acc / dii : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) Qi[i * E62_QLD + lane] = x[i];
}

// ---------------------------------------------------------------------------
// v2 -- register-resident Cholesky chain, inverse computed separately.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// v3 -- the same chain kept in shared memory (no large register array).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void e62_chain32_shm(float* __restrict__ Sb, int lane)
{
    for (int k = 0; k < 32; ++k) {
        const float akk = Sb[k * E62_LD + k];
        const float dk  = rsqrtf(akk);
        const float lik = Sb[lane * E62_LD + k] * dk;
        __syncwarp();
        if (lane >= k) Sb[lane * E62_LD + k] = lik;
        __syncwarp();
        #pragma unroll 8
        for (int t = 1; t < 32; ++t) {
            if (t > k) {
                const float v = Sb[t * E62_LD + k];
                if (t <= lane) Sb[lane * E62_LD + t] -= lik * v;
            }
        }
        __syncwarp();
    }
    #pragma unroll 4
    for (int t = 0; t < 32; ++t)
        if (t > lane) Sb[lane * E62_LD + t] = 0.0f;
}

// ---------------------------------------------------------------------------
// The block kernel. CHAIN==2 uses the register chain, CHAIN==3 the shared one.
// ---------------------------------------------------------------------------
template <int CHAIN>
__global__ __launch_bounds__(E62_NT, 1)
void e62_diag128_t(float* __restrict__ A, float* __restrict__ Dinv,
                   const int n, const int j)
{
    extern __shared__ float sm[];
    float* S  = sm;
    float* M  = sm + E62_M_OFF;
    float* Qi = sm + E62_QI_OFF;
    float* Qt = sm + E62_QT_OFF;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    float* Ab = A + (size_t)blockIdx.x * (size_t)n * (size_t)n
                  + (size_t)j * (size_t)n + (size_t)j;

    for (int r = warp; r < E62_TB; r += E62_NW) {
        const float* srow = Ab + (size_t)r * (size_t)n;
        float* drow = S + r * E62_LD;
        float* mrow = M + r * E62_LD;
        for (int c = lane; c < E62_TB; c += 32) {
            drow[c] = srow[c];
            mrow[c] = (c == r) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();

    for (int kk = 0; kk < E62_TB; kk += 32) {
        const int lwid = kk + 32;
        float* Sb = S + kk * E62_LD + kk;

        if (warp == 0) {
            if (CHAIN == 2) e62_chain32_reg(Sb, lane);
            else            e62_chain32_shm(Sb, lane);
        }
        __syncwarp();
        if (warp == 0) e62_tri_inv32(Sb, Qi, lane);
        __syncthreads();

        for (int r0 = lwid + warp * 4; r0 < E62_TB; r0 += E62_NW * 4) {
            const float* q = Qi + lane * E62_QLD;
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

        if (kk > 0) {
            for (int i = warp; i < 32; i += E62_NW) {
                const float* q = Qi + i * E62_QLD;
                for (int c = lane; c < kk; c += 32) {
                    float acc = 0.f;
                    #pragma unroll 8
                    for (int u = 0; u < 32; ++u)
                        acc += q[u] * M[(kk + u) * E62_LD + c];
                    Qt[i * E62_LD + c] = acc;
                }
            }
        }
        __syncthreads();

        for (int i = warp; i < 32; i += E62_NW) {
            float* mrow = M + (kk + i) * E62_LD;
            for (int c = lane; c < kk; c += 32) mrow[c] = Qt[i * E62_LD + c];
            mrow[kk + lane] = Qi[i * E62_QLD + lane];
        }
        __syncthreads();

        if (lwid >= E62_TB) continue;

        for (int r0 = lwid + warp * 4; r0 < E62_TB; r0 += E62_NW * 4) {
            const float* p0 = S + r0 * E62_LD + kk;
            const float* p1 = p0 + E62_LD;
            const float* p2 = p0 + 2 * E62_LD;
            const float* p3 = p0 + 3 * E62_LD;

            for (int c = lwid + lane; c < E62_TB; c += 32) {
                const float* sc = S + c * E62_LD + kk;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                #pragma unroll 8
                for (int t = 0; t < 32; ++t) {
                    const float v = sc[t];
                    a0 += p0[t] * v;
                    a1 += p1[t] * v;
                    a2 += p2[t] * v;
                    a3 += p3[t] * v;
                }
                S[r0 * E62_LD + c]       -= a0;
                S[(r0 + 1) * E62_LD + c] -= a1;
                S[(r0 + 2) * E62_LD + c] -= a2;
                S[(r0 + 3) * E62_LD + c] -= a3;
            }

            for (int c = lane; c < lwid; c += 32) {
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                #pragma unroll 8
                for (int t = 0; t < 32; ++t) {
                    const float v = M[(kk + t) * E62_LD + c];
                    a0 += p0[t] * v;
                    a1 += p1[t] * v;
                    a2 += p2[t] * v;
                    a3 += p3[t] * v;
                }
                M[r0 * E62_LD + c]       -= a0;
                M[(r0 + 1) * E62_LD + c] -= a1;
                M[(r0 + 2) * E62_LD + c] -= a2;
                M[(r0 + 3) * E62_LD + c] -= a3;
            }
        }
        __syncthreads();
    }

    float* Db = Dinv + (size_t)blockIdx.x * (size_t)(E62_TB * E62_TB);
    for (int r = warp; r < E62_TB; r += E62_NW) {
        const float* srow = S + r * E62_LD;
        const float* mrow = M + r * E62_LD;
        float* arow = Ab + (size_t)r * (size_t)n;
        float* drow = Db + r * E62_TB;
        for (int c = lane; c < E62_TB; c += 32) {
            arow[c] = (c <= r) ? srow[c] : 0.0f;
            drow[c] = (c <= r) ? mrow[c] : 0.0f;
        }
    }
}

// Isolated chain micro-benchmark: `reps` back-to-back 32x32 factorizations of
// the SAME resident tile, so the cost per pivot is measured with no block
// setup, no panel and no trailing work in the way.
template <int CHAIN>
__global__ __launch_bounds__(32, 1)
void e62_chainbench(float* __restrict__ A, const int n, const int reps)
{
    extern __shared__ float sm[];
    const int lane = threadIdx.x;
    for (int r = 0; r < 32; ++r)
        sm[r * E62_LD + lane] = A[(size_t)r * n + lane];
    __syncwarp();
    float sink = 0.0f;
    for (int it = 0; it < reps; ++it) {
        if (CHAIN == 2) e62_chain32_reg(sm, lane);
        else            e62_chain32_shm(sm, lane);
        __syncwarp();
        // Consume the factor so the chain cannot be eliminated as dead code,
        // then restore a diagonally dominant tile for the next repetition.
        sink += sm[lane * E62_LD + lane];
        __syncwarp();
        for (int r = 0; r < 32; ++r)
            sm[r * E62_LD + lane] = (r == lane) ? 64.0f : 1.0f;
        __syncwarp();
    }
    if (lane == 0) A[0] = sink;
}

// ---------------------------------------------------------------------------

static void e62_configure(const void* fn, int bytes)
{
    cudaError_t attr = cudaFuncSetAttribute(
        const_cast<void*>(fn), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
}

void e62_diag128_launch(torch::Tensor A, torch::Tensor Dinv,
                        int64_t n, int64_t j, int64_t chain)
{
    static bool c2 = false, c3 = false;
    const int batch = (int)A.size(0);
    if (chain == 2) {
        if (!c2) { e62_configure((const void*)e62_diag128_t<2>, E62_SMEM_B); c2 = true; }
        e62_diag128_t<2><<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
            A.data_ptr<float>(), Dinv.data_ptr<float>(), (int)n, (int)j);
    } else {
        if (!c3) { e62_configure((const void*)e62_diag128_t<3>, E62_SMEM_B); c3 = true; }
        e62_diag128_t<3><<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
            A.data_ptr<float>(), Dinv.data_ptr<float>(), (int)n, (int)j);
    }
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void e62_chainbench_launch(torch::Tensor A, int64_t n, int64_t reps,
                           int64_t chain)
{
    const int bytes = 32 * E62_LD * 4;
    if (chain == 2)
        e62_chainbench<2><<<dim3(1), dim3(32), bytes>>>(
            A.data_ptr<float>(), (int)n, (int)reps);
    else
        e62_chainbench<3><<<dim3(1), dim3(32), bytes>>>(
            A.data_ptr<float>(), (int)n, (int)reps);
    cudaError_t status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
"""


def _load_exp062():
    global _EXP062, _EXP062_ERROR
    if _EXP062 is not None or _EXP062_ERROR is not None:
        return
    try:
        from torch.utils.cpp_extension import load_inline

        _EXP062 = load_inline(
            name="chol_exp062_midbatch_v2",
            cpp_sources=(
                "void e62_diag128_launch(torch::Tensor, torch::Tensor, "
                "int64_t, int64_t, int64_t);\n"
                "void e62_chainbench_launch(torch::Tensor, int64_t, int64_t, "
                "int64_t);"
            ),
            cuda_sources=_EXP062_SOURCE,
            functions=["e62_diag128_launch", "e62_chainbench_launch"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-v"],
            verbose=True,
        )
    except Exception as exc:  # pragma: no cover
        _EXP062_ERROR = repr(exc)


_EXP062_BUF = {}


def _exp062_buffers(batch, n, device):
    key = (batch, n)
    buf = _EXP062_BUF.get(key)
    if buf is None:
        dinv = torch.empty(batch, 128, 128, device=device, dtype=torch.float32)
        pan = torch.empty(batch * n * 128, device=device, dtype=torch.float32)
        buf = (dinv, pan)
        _EXP062_BUF[key] = buf
    return buf


def _exp062_factor(data, nb_outer=512, chain=2, tf32=True):
    batch, n, _ = data.shape
    work = data.clone()
    dinv, pan = _exp062_buffers(batch, n, data.device)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
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
                _EXP062.e62_diag128_launch(work, dinv, n, jj, chain)
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
    """The exact incumbent route for these shapes: loop the vendor call."""
    return _loop_cholesky(x)


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

    # --- isolated pivot-chain cost, both designs ---
    tile = torch.full((32, 32), 1.0, device=dev)
    tile.fill_diagonal_(64.0)
    for chain in (2, 3):
        reps = 200
        us = _e62_time(
            lambda _: _EXP062.e62_chainbench_launch(tile, 32, reps, chain),
            None, iters=10, warmup=3,
        )
        us0 = _e62_time(
            lambda _: _EXP062.e62_chainbench_launch(tile, 32, 1, chain),
            None, iters=10, warmup=3,
        )
        per_rep = (us - us0) / (reps - 1)
        rows.append({"name": f"chain32_v{chain}", "us": round(per_rep, 4),
                     "ns_per_pivot": round(per_rep * 1000.0 / 32.0, 1),
                     "ok": True})

    # --- one 128x128 block, timed with a restored source block ---
    a = generate_input(batch=2, n=2048, cond=2, seed=44048)
    blk = a[:, :128, :128].clone()
    work = a.clone()
    dinv = torch.empty(2, 128, 128, device=dev, dtype=torch.float32)

    def _restore(_):
        work[:, :128, :128].copy_(blk)

    us_copy = _e62_time(_restore, None, iters=20, warmup=5)
    for chain in (2, 3):
        def _blockrun(_, ch=chain):
            work[:, :128, :128].copy_(blk)
            _EXP062.e62_diag128_launch(work, dinv, 2048, 0, ch)

        us = _e62_time(_blockrun, None, iters=20, warmup=5)
        net = us - us_copy
        work[:, :128, :128].copy_(blk)
        _EXP062.e62_diag128_launch(work, dinv, 2048, 0, chain)
        torch.cuda.synchronize()
        l11 = work[:, :128, :128].tril()
        err, scale = _e62_residual(a[:, :128, :128], l11)
        inv_err = float(
            (dinv[0] @ l11[0] - torch.eye(128, device=dev)).abs().max().item()
        )
        rows.append({"name": f"diag128_block_v{chain}", "us": round(net, 3),
                     "ns_per_row": round(net * 1000.0 / 128.0, 1),
                     "abs_err": round(err, 7), "inv_err": round(inv_err, 8),
                     "ok": err < 1e-3 and inv_err < 1e-3})
    rows.append({"name": "restore_copy_overhead", "us": round(us_copy, 3),
                 "ok": True})
    del work, blk, a
    torch.cuda.empty_cache()

    # --- whole-shape prototypes against the SHIPPED control ---
    for (batch, n, seed) in ((2, 2048, 44048), (2, 4096, 514096)):
        a = generate_input(batch=batch, n=n, cond=2, seed=seed)
        base = _e62_time(_shipped, a, iters=8, warmup=3)
        rows.append({"name": f"shipped_{batch}x{n}", "us": round(base, 1),
                     "ok": True})
        for chain in (2, 3):
            for nbo in (512, 1024):
                try:
                    us = _e62_time(
                        lambda x: _exp062_factor(x, nbo, chain), a,
                        iters=8, warmup=3,
                    )
                    l = _exp062_factor(a, nbo, chain)
                    torch.cuda.synchronize()
                    err, scale = _e62_residual(a, l)
                    rows.append({
                        "name": f"e062_{batch}x{n}_v{chain}_nbo{nbo}",
                        "us": round(us, 1),
                        "speedup": round(base / us, 4),
                        "abs_err": round(err, 6),
                        "ok": bool(torch.isfinite(l).all().item()),
                    })
                    del l
                except Exception as exc:
                    rows.append({
                        "name": f"e062_{batch}x{n}_v{chain}_nbo{nbo}",
                        "us": 0.0, "ok": False, "error": repr(exc)[:240],
                    })
        del a
        torch.cuda.empty_cache()
    return rows
