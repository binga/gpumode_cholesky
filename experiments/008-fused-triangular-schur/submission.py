#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission — experiment 006 (current best).

Builds on exp 005 (`#877956`) by adding a right-looking BLOCKED Cholesky for the
large single matrices (`batch == 1, n >= 16384`). The diagonal block potrf and the
panel triangular solve stay FP32 (stability), but the O(n^3) trailing Schur update
`A22 -= L21 @ L21^T` runs on B200 tensor cores via TF32, which is several times
faster than cuSOLVER's all-FP32 `potrf` at these sizes. The cholesky checker gates
only `||A - LL^T||_1 <= 20*n*eps*||A||_1`, whose tolerance grows with n; measured
residual margins stay 200-400x inside tolerance across families. Pure `torch`,
default-queue only (popcorn disqualifies any non-default CUDA-queue use).

Shape dispatcher:
  * n == 32                         -> Triton batched kernel, one warp per matrix
    (experiment 002). Beats cuSOLVER's batched-launch overhead for tiny matrices.
  * batch == 1 and n >= 16384       -> blocked right-looking Cholesky with a
    TF32 tensor-core trailing update (experiment 006). Measured on B200:
    16384 1.80x, 32768 2.94x vs batched cuSOLVER, all families correct. nb=4096
    for n>=32768 (trailing GEMM dominates -> bigger blocks win), else nb=2048.
    8192 (only ~1.07x) stays on cuSOLVER.
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

    _NUM_WARPS = {32: 1}

    def _triton_cholesky(data: torch.Tensor) -> torch.Tensor:
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
# Large single-matrix path (experiment 006): blocked right-looking Cholesky with
# a TF32 tensor-core trailing update. Diagonal block + panel solve stay FP32.
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


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    is_f32_cuda = data.is_cuda and data.dtype == torch.float32

    if is_f32_cuda and _HAVE_TRITON and n == 32:
        return _triton_cholesky(data)

    # Large single matrices: blocked Cholesky with a TF32 tensor-core trailing
    # update beats cuSOLVER's all-FP32 potrf (exp 006). Only the measured-win
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
