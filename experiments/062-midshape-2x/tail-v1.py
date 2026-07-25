

# ---------------------------------------------------------------------------
# Experiment 062 -- batched blocked Cholesky for the tiny-batch mid shapes.
#
# 2x2048 and 2x4096 currently run the vendor factorization once per matrix,
# serially: 2 x 597.5us and 2 x 1390.0us, 87-91% of each shape's device time.
# That kernel is dependent-pivot-latency bound at ~0.33us per row, so its cost
# is c*n per matrix and the batch dimension buys nothing.
#
# This path replaces it with a right-looking blocked factorization whose
# diagonal blocks are factored by ONE resident CTA per matrix, so:
#   * the pivot chain runs once for the whole batch (2 CTAs in parallel), and
#   * each 128-pivot chain stays in registers/shared instead of round-tripping
#     through global memory once per pivot.
#
# `e62_diag128` factors a 128x128 diagonal block and publishes both L11 and
# its explicit inverse, so the panel below becomes a plain batched GEMM
# instead of a triangular solve.
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

// Shared layout (floats):
//   S  [128][129]  working diagonal block -> L11
//   M  [128][129]  accumulates inv(L11)
//   Qi [32][33]    current 32x32 diagonal inverse
//   Qt [32][129]   staging for the in-place inverse row-block update
#define E62_M_OFF   (E62_TB * E62_LD)
#define E62_QI_OFF  (2 * E62_TB * E62_LD)
#define E62_QT_OFF  (E62_QI_OFF + 32 * E62_QLD)
#define E62_SMEM_F  (E62_QT_OFF + 32 * E62_LD)
#define E62_SMEM_B  (E62_SMEM_F * 4)

// One CTA factors the 128x128 diagonal block at (j,j) of matrix blockIdx.x.
// Writes L11 back in place (strict upper zeroed) and inv(L11) into Dinv.
__global__ __launch_bounds__(E62_NT, 1)
void e62_diag128(float* __restrict__ A, float* __restrict__ Dinv,
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

    // ---- load the block, seed M with the identity ----
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
        const int lwid = kk + 32;   // live columns of the inverse

        // ---- warp 0: register-resident Gauss-Jordan on the 32x32 pivot
        // block. Lane i owns row i of the tile in registers, so the only
        // cross-lane traffic per pivot is a broadcast of the pivot column and
        // of the pivot row of the inverse -- the chain never touches shared.
        if (warp == 0) {
            float* Sb = S + kk * E62_LD + kk;
            float a[32];
            float m[32];
            #pragma unroll
            for (int t = 0; t < 32; ++t) {
                a[t] = Sb[lane * E62_LD + t];
                m[t] = (t == lane) ? 1.0f : 0.0f;
            }
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float akk = __shfl_sync(0xffffffffu, a[k], k);
                const float dk  = rsqrtf(akk);
                const float lik = a[k] * dk;          // L[lane][k], lane >= k
                #pragma unroll
                for (int t = 0; t < 32; ++t) {
                    const float Lt = __shfl_sync(0xffffffffu, lik, t);
                    const float Mt = __shfl_sync(0xffffffffu, m[t], k) * dk;
                    if (lane > k) {
                        if (t > k) a[t] -= lik * Lt;
                        m[t] -= lik * Mt;
                    }
                }
                if (lane == k) {
                    #pragma unroll
                    for (int t = 0; t < 32; ++t) m[t] *= dk;
                }
                if (lane >= k) a[k] = lik;
            }
            #pragma unroll
            for (int t = 0; t < 32; ++t) {
                Sb[lane * E62_LD + t]  = (t <= lane) ? a[t] : 0.0f;
                Qi[lane * E62_QLD + t] = (t <= lane) ? m[t] : 0.0f;
            }
        }
        __syncthreads();

        // ---- panel solve: S[r][kk:kk+32] <- S[r][kk:kk+32] * inv(L11)^T ----
        // Four rows per warp so the broadcast reads amortise. Row coverage is
        // exact: r0 is always <= 124, so every group is a full four rows.
        for (int r0 = lwid + warp * 4; r0 < E62_TB; r0 += E62_NW * 4) {
            const float* q = Qi + lane * E62_QLD;
            const float* s0 = S + r0 * E62_LD + kk;
            const float* s1 = s0 + E62_LD;
            const float* s2 = s0 + 2 * E62_LD;
            const float* s3 = s0 + 3 * E62_LD;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            #pragma unroll 8
            for (int u = 0; u < 32; ++u) {
                const float qv = q[u];
                a0 += s0[u] * qv;
                a1 += s1[u] * qv;
                a2 += s2[u] * qv;
                a3 += s3[u] * qv;
            }
            __syncwarp();
            S[r0 * E62_LD + kk + lane]                 = a0;
            S[(r0 + 1) * E62_LD + kk + lane]           = a1;
            S[(r0 + 2) * E62_LD + kk + lane]           = a2;
            S[(r0 + 3) * E62_LD + kk + lane]           = a3;
        }

        // ---- stage the inverse row block: Qt <- Qi * M[kk:kk+32, 0:kk] ----
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

        // ---- commit the inverse row block ----
        for (int i = warp; i < 32; i += E62_NW) {
            float* mrow = M + (kk + i) * E62_LD;
            for (int c = lane; c < kk; c += 32) mrow[c] = Qt[i * E62_LD + c];
            mrow[kk + lane] = Qi[i * E62_QLD + lane];
        }
        __syncthreads();

        if (lwid >= E62_TB) continue;

        // ---- rank-32 update of the trailing block and of the inverse ----
        for (int r0 = lwid + warp * 4; r0 < E62_TB; r0 += E62_NW * 4) {
            const float* p0 = S + r0 * E62_LD + kk;
            const float* p1 = p0 + E62_LD;
            const float* p2 = p0 + 2 * E62_LD;
            const float* p3 = p0 + 3 * E62_LD;

            // trailing: S[r][c] -= sum_t S[r][kk+t] * S[c][kk+t]
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

            // inverse: M[r][c] -= sum_t S[r][kk+t] * M[kk+t][c]
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

    // ---- publish L11 (lower) and inv(L11) ----
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

void e62_diag128_launch(torch::Tensor A, torch::Tensor Dinv,
                        int64_t n, int64_t j)
{
    static bool configured = false;
    if (!configured) {
        cudaError_t attr = cudaFuncSetAttribute(
            e62_diag128,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            E62_SMEM_B);
        TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
        configured = true;
    }
    const int batch = (int)A.size(0);
    e62_diag128<<<dim3(batch), dim3(E62_NT), E62_SMEM_B>>>(
        A.data_ptr<float>(), Dinv.data_ptr<float>(), (int)n, (int)j);
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
            name="chol_exp062_midbatch_v1",
            cpp_sources=(
                "void e62_diag128_launch(torch::Tensor, torch::Tensor, "
                "int64_t, int64_t);"
            ),
            cuda_sources=_EXP062_SOURCE,
            functions=["e62_diag128_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
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


def _exp062_factor(data, nb_outer=512, tf32=True):
    """Right-looking blocked Cholesky with resident 128x128 diagonal blocks.

    Panel width is 128 (the resident block size); the trailing rank-k update is
    deferred to `nb_outer` so the low-intensity pass sweeps the trailing
    submatrix n/nb_outer times instead of n/128 times.
    """
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
                _EXP062.e62_diag128_launch(work, dinv, n, jj)
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


# ---------------------------------------------------------------------------
# Probe harness -- `midprobe` mode calls this and prints the rows.
# ---------------------------------------------------------------------------

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


def mid_probe():
    import sys

    sys.path.insert(0, "/root/reference")
    from reference import generate_input

    _load_exp062()
    rows = []
    if _EXP062 is None:
        return [{"name": "load_inline", "us": 0.0, "ok": False,
                 "error": _EXP062_ERROR}]

    # --- cost and correctness of one resident 128x128 diagonal block ---
    a = generate_input(batch=2, n=2048, cond=2, seed=44048)
    work = a.clone()
    dinv = torch.empty(2, 128, 128, device=a.device, dtype=torch.float32)

    us = _e62_time(
        lambda _: _EXP062.e62_diag128_launch(work, dinv, 2048, 0),
        None, iters=20, warmup=5,
    )
    rows.append({"name": "diag128_block", "us": round(us, 3),
                 "ns_per_row": round(us * 1000.0 / 128.0, 1), "ok": True})

    work = a.clone()
    _EXP062.e62_diag128_launch(work, dinv, 2048, 0)
    torch.cuda.synchronize()
    l11 = work[:, :128, :128].tril()
    err, scale = _e62_residual(a[:, :128, :128], l11)
    rows.append({"name": "diag128_block_err", "us": 0.0,
                 "abs_err": round(err, 6), "scale": round(scale, 3),
                 "ok": err < 1e-2 * max(scale, 1.0)})
    eye = torch.eye(128, device=a.device)
    inv_err = float((dinv[0] @ l11[0] - eye).abs().max().item())
    rows.append({"name": "diag128_inv_err", "us": 0.0,
                 "abs_err": round(inv_err, 8), "ok": inv_err < 1e-3})
    del work, a
    torch.cuda.empty_cache()

    # --- whole-shape prototypes against the shipped vendor control ---
    for (batch, n, seed) in ((2, 2048, 44048), (2, 4096, 514096),
                             (1, 4096, 48096)):
        a = generate_input(batch=batch, n=n, cond=2, seed=seed)
        base = _e62_time(
            lambda x: torch.linalg.cholesky_ex(x, check_errors=False).L, a,
            iters=8, warmup=3,
        )
        rows.append({"name": f"control_{batch}x{n}", "us": round(base, 1),
                     "ok": True})
        for nbo in (256, 512, 1024):
            try:
                us = _e62_time(lambda x: _exp062_factor(x, nbo), a,
                               iters=8, warmup=3)
                l = _exp062_factor(a, nbo)
                torch.cuda.synchronize()
                err, scale = _e62_residual(a, l)
                rows.append({
                    "name": f"e062_{batch}x{n}_nbo{nbo}",
                    "us": round(us, 1),
                    "speedup": round(base / us, 4),
                    "abs_err": round(err, 5),
                    "scale": round(scale, 2),
                    "ok": bool(torch.isfinite(l).all().item()),
                })
                del l
            except Exception as exc:
                rows.append({"name": f"e062_{batch}x{n}_nbo{nbo}", "us": 0.0,
                             "ok": False, "error": repr(exc)[:240]})
        del a
        torch.cuda.empty_cache()
    return rows
