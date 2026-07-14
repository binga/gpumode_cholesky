#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission.

Batched dense Cholesky factorization. Input `A` is a `batch x n x n` float32
CUDA tensor, SPD up to FP32 roundoff. Return lower-triangular float32 `L` with
positive diagonal such that `A = L @ L.T`.

`custom_kernel` is a shape dispatcher:
  * small n (32/64/128), high batch  -> custom Triton batched kernel (one CTA
    per matrix, right-looking factorization on a tile distributed across the
    block's threads). These shapes are launch/dispatch-overhead-bound under
    cuSOLVER, so a single fused launch is the win.
  * everything else                  -> cuSOLVER via torch (already strong).
"""

import torch

from task import input_t, output_t

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton always present on the B200 runner
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
        """One program (CTA) factorizes one N x N SPD matrix.

        The whole matrix lives in a single [N, N] tile, which Triton spreads
        across the block's threads (register pressure per thread = N*N/threads).
        Right-looking Cholesky: at step k, scale column k by 1/sqrt(A[k,k]),
        then rank-1 update the trailing submatrix. N is a constexpr so the loop
        is fully unrolled and specialized per matrix size.
        """
        pid = tl.program_id(0)
        rows = tl.arange(0, N)
        cols = tl.arange(0, N)
        a_ptrs = (
            A_ptr
            + pid * stride_ab
            + rows[:, None] * stride_ai
            + cols[None, :] * stride_aj
        )
        a = tl.load(a_ptrs)  # [N, N], symmetric SPD

        for k in range(N):
            akk = tl.sum(
                tl.where((rows[:, None] == k) & (cols[None, :] == k), a, 0.0)
            )
            inv = 1.0 / tl.sqrt(akk)
            # Scale column k (rows >= k): A[k,k] -> sqrt(A[k,k]); A[i,k] -> A[i,k]/sqrt.
            col_k = (cols[None, :] == k) & (rows[:, None] >= k)
            a = tl.where(col_k, a * inv, a)
            # Extract the just-scaled column k as an [N] vector.
            lk = tl.sum(tl.where(cols[None, :] == k, a, 0.0), axis=1)
            # Rank-1 update of the trailing submatrix (rows > k, cols > k).
            trail = (rows[:, None] > k) & (cols[None, :] > k)
            a = tl.where(trail, a - lk[:, None] * lk[None, :], a)

        # Zero the strict upper triangle so the output is lower-triangular.
        a = tl.where(cols[None, :] > rows[:, None], 0.0, a)
        l_ptrs = (
            L_ptr
            + pid * stride_lb
            + rows[:, None] * stride_li
            + cols[None, :] * stride_lj
        )
        tl.store(l_ptrs, a)

    # One warp per matrix (num_warps=1): the per-column reductions become cheap
    # in-warp shuffles instead of shared-memory syncs, which is what makes n=32
    # beat cuSOLVER's batched-launch overhead. Larger n spills registers under a
    # single warp and loses to cuSOLVER, so only n=32 is dispatched here.
    _NUM_WARPS = {32: 1}

    def _triton_cholesky(data: torch.Tensor) -> torch.Tensor:
        batch, n, _ = data.shape
        data = data.contiguous()
        out = torch.empty_like(data)
        grid = (batch,)
        _chol_batched_kernel[grid](
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


_CUSTOM_NS = (32,)


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    if (
        _HAVE_TRITON
        and data.is_cuda
        and data.dtype == torch.float32
        and n in _CUSTOM_NS
    ):
        return _triton_cholesky(data)
    # Default path: cuSOLVER via torch. Correct for every input family.
    return torch.linalg.cholesky_ex(data, check_errors=False).L
