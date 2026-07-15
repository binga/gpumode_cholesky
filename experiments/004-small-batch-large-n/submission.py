#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission.

Batched dense Cholesky factorization. Input `A` is a `batch x n x n` float32
CUDA tensor, SPD up to FP32 roundoff. Return lower-triangular float32 `L` with
positive diagonal such that `A = L @ L.T`.

Shape dispatcher:
  * n == 32                         -> Triton batched kernel, one warp per matrix
    (experiment 002). Beats cuSOLVER's batched-launch overhead for tiny matrices.
  * 2 <= batch <= 8 and n >= 1024   -> per-matrix factorization in a sequential
    loop (experiment 004). `torch.linalg` routes batch>=2 to
    `cusolverDnSpotrfBatched`, which is tuned for many-small matrices and is
    ~1.2-4x too slow for few-large ones; factorizing each matrix on its own with
    the fast single-matrix blocked `potrf` is much faster.
  * everything else                 -> batched cuSOLVER via cholesky_ex (best for
    batch=1 large-n and high-batch small/mid-n).
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
# Small-batch / large-n path (experiment 004).
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


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    is_f32_cuda = data.is_cuda and data.dtype == torch.float32

    if is_f32_cuda and _HAVE_TRITON and n == 32:
        return _triton_cholesky(data)

    # Few-but-large matrices: avoid cusolverDnSpotrfBatched (see module docstring).
    if is_f32_cuda and 2 <= batch <= 8 and n >= 1024:
        return _loop_cholesky(data)

    # Default: batched cuSOLVER. Correct for every input family.
    return torch.linalg.cholesky_ex(data, check_errors=False).L
