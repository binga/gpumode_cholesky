#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""GPU MODE `cholesky` submission — experiment 016a candidate-v2-8192fp8.

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
  * n == 32                         -> Triton batched kernel, one warp per matrix
    (experiment 002). Beats cuSOLVER's batched-launch overhead for tiny matrices.
  * batch == 256 and n == 128       -> captured vendor batched factorization
    (1.177x paired speedup, exact numerics).
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
        n: tl.constexpr,
        k,
    ):
        """Factor the 32x32 diagonal block at (k, k) and build its triangular
        inverse in the same 32-step serial loop (row p of L is final after
        step p, so X[p,:] = (I[p,:] - L[p,:p] @ X[:p,:]) / l_pp can be formed
        immediately). One warp per matrix keeps every reduction warp-local."""
        b = tl.program_id(0).to(tl.int64)
        r = tl.arange(0, 32)
        c = tl.arange(0, 32)
        ptr = out_ptr + b * n * n + (k + r)[:, None] * n + (k + c)[None, :]
        a = tl.load(ptr)
        x = tl.where(r[:, None] == c[None, :], 1.0, 0.0)
        # Rank-2 right-looking factorization: two columns per serial step
        # halves the length of the latency-bound dependency chain, and the
        # trailing tile update is a single fused write per step.
        for it in range(0, 16):
            p = 2 * it
            q = p + 1
            # Independent extractions issue together (ILP): both columns and
            # both raw diagonal entries.
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
                        a
                        - lp[:, None] * lp[None, :]
                        - lq[:, None] * lq[None, :],
                        a,
                    ),
                ),
            )
            # Rows p and q of the factor are final; both inverse-row
            # contributions reduce against X rows < p (independent, ILP),
            # and row q gets a scalar correction for its row-p term.
            lpp = dpp * inv1
            lqq = dqq * inv2
            rowp = tl.sum(tl.where(r[:, None] == p, a, 0.0), axis=0)
            rowq = tl.sum(tl.where(r[:, None] == q, a, 0.0), axis=0)
            rmp = tl.where(c < p, rowp, 0.0)
            rmq = tl.where(c < p, rowq, 0.0)
            contrib_p = tl.sum(rmp[:, None] * x, axis=0)
            contrib_q = tl.sum(rmq[:, None] * x, axis=0)
            lqp = tl.sum(tl.where(c == p, rowq, 0.0), axis=0)
            eqp = tl.where(c == p, 1.0, 0.0)
            eqq = tl.where(c == q, 1.0, 0.0)
            xp = (eqp - contrib_p) / lpp
            xq = (eqq - contrib_q - lqp * xp) / lqq
            x = tl.where(
                r[:, None] == p,
                xp[None, :],
                tl.where(r[:, None] == q, xq[None, :], x),
            )
        a = tl.where(c[None, :] <= r[:, None], a, 0.0)
        tl.store(ptr, a)
        tl.store(inv_ptr + b * 1024 + r[:, None] * 32 + c[None, :], x)

    @triton.jit
    def _panel_apply32(
        out_ptr,
        inv_ptr,
        n: tl.constexpr,
        k,
        remaining,
        PREC: tl.constexpr,
        TILE_R: tl.constexpr,
    ):
        """L[i, k-block] = A[i, k-block] @ Dinv^T for all rows below the
        diagonal block (full panel column of the factor)."""
        rt = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        rows = rt * TILE_R + tl.arange(0, TILE_R)
        c = tl.arange(0, 32)
        base = b * n * n
        mask = rows < remaining
        p_ptrs = (
            out_ptr + base + (k + 32 + rows)[:, None] * n + (k + c)[None, :]
        )
        p = tl.load(p_ptrs, mask=mask[:, None], other=0.0)
        dinv = tl.load(inv_ptr + b * 1024 + c[:, None] * 32 + c[None, :])
        lik = tl.dot(
            p, tl.trans(dinv), input_precision=PREC, out_dtype=tl.float32
        )
        tl.store(p_ptrs, lik, mask=mask[:, None])

    @triton.jit
    def _panel_inner32(
        out_ptr,
        n: tl.constexpr,
        k,
        width,
        remaining,
        PREC: tl.constexpr,
        TILE_R: tl.constexpr,
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
        t_ptrs = (
            out_ptr
            + base
            + (k + 32 + rows)[:, None] * n
            + (k + 32 + cw)[None, :]
        )
        valid = rmask[:, None] & wmask[None, :]
        t = tl.load(t_ptrs, mask=valid, other=0.0)
        tl.store(t_ptrs, t - prod, mask=valid)

    @triton.jit
    def _trailing_nb(
        out_ptr,
        n: tl.constexpr,
        j,
        remaining,
        NB: tl.constexpr,
        PREC: tl.constexpr,
        TILE: tl.constexpr,
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
        prod = tl.dot(
            li, tl.trans(lj), input_precision=PREC, out_dtype=tl.float32
        )
        t_ptrs = (
            out_ptr
            + base
            + (j + NB + rows)[:, None] * n
            + (j + NB + cols)[None, :]
        )
        valid = (rows[:, None] < remaining) & (cols[None, :] < remaining)
        valid = valid & ((br != bc) | (cols[None, :] <= rows[:, None]))
        t = tl.load(t_ptrs, mask=valid, other=0.0)
        tl.store(t_ptrs, t - prod, mask=valid)

    # (batch, n) -> (panel_prec, trailing_prec) for the two-level blocked
    # path. tf32x3 keeps tensor cores with near-FP32 accuracy where the
    # n-scaled tolerance is tight; plain tf32 is enough from n=1024 up.
    # (batch, n) -> (panel_prec, trailing_prec, trailing_tile)
    _SPLIT32_SHAPES = {
        (64, 256): ("tf32x3", "tf32x3", 128),
        (16, 512): ("tf32x3", "tf32x3", 128),
        (640, 512): ("tf32x3", "tf32", 128),
        (4, 1024): ("tf32x3", "tf32", 128),
        (60, 1024): ("tf32x3", "tf32", 128),
        (8, 2048): ("tf32x3", "tf32", 128),
    }
    _SPLIT32_TILE = 128
    _SPLIT32_NB = 128

    def _split32_launch(work, dinv, panel_prec, trailing_prec, trailing_tile):
        """Launch the full two-level blocked factorization on `work`."""
        batch, n, _ = work.shape
        tile = _SPLIT32_TILE
        nb = _SPLIT32_NB
        for j in range(0, n, nb):
            panel_end = min(j + nb, n)
            for k in range(j, panel_end, 32):
                _micro_potrf_gj32[(batch,)](work, dinv, n=n, k=k, num_warps=1)
                remaining = n - k - 32
                if remaining <= 0:
                    break
                _panel_apply32[(triton.cdiv(remaining, tile), batch)](
                    work,
                    dinv,
                    n=n,
                    k=k,
                    remaining=remaining,
                    PREC=panel_prec,
                    TILE_R=tile,
                    num_warps=4,
                )
                width = panel_end - (k + 32)
                if width > 0:
                    _panel_inner32[(triton.cdiv(remaining, tile), batch)](
                        work,
                        n=n,
                        k=k,
                        width=width,
                        remaining=remaining,
                        PREC=panel_prec,
                        TILE_R=tile,
                        num_warps=4,
                    )
            rem_out = n - panel_end
            if rem_out > 0:
                tr = triton.cdiv(rem_out, trailing_tile)
                _trailing_nb[(tr * (tr + 1) // 2, batch)](
                    work,
                    n=n,
                    j=j,
                    remaining=rem_out,
                    NB=nb,
                    PREC=trailing_prec,
                    TILE=trailing_tile,
                    num_warps=8,
                    num_stages=3,
                )
        ct = triton.cdiv(n, tile)
        _clear_upper_tiles[(ct * (ct + 1) // 2, batch)](
            work,
            n=n,
            TILE=tile,
            num_warps=8,
        )

    _SPLIT32_GRAPHS = {}

    def _split32_factor(data: torch.Tensor) -> torch.Tensor:
        batch, n, _ = data.shape
        panel_prec, trailing_prec, trailing_tile = _SPLIT32_SHAPES[(batch, n)]
        data = data.contiguous()
        key = (batch, n)
        entry = _SPLIT32_GRAPHS.get(key)
        if entry is None:
            try:
                work = torch.empty_like(data)
                dinv = torch.empty(
                    batch, 32, 32, device=data.device, dtype=torch.float32
                )
                for _ in range(2):
                    work.copy_(data)
                    _split32_launch(work, dinv, panel_prec, trailing_prec, trailing_tile)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=_shared_graph_pool()):
                    _split32_launch(work, dinv, panel_prec, trailing_prec, trailing_tile)
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
                batch, 32, 32, device=data.device, dtype=torch.float32
            )
            _split32_launch(work, dinv, panel_prec, trailing_prec, trailing_tile)
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
# Experiment 016a: generalized large single-matrix left-looking path.
# ---------------------------------------------------------------------------
import math as _math

_LARGE_FP8_HITS = 0
_LARGE_FP8_FALLBACKS = 0
_LARGE_FP8_ERROR = None

_LARGE_CFG = {
    8192: dict(nb=2048, panel_mode="fp8", diag_mode="tf32", rec_inv=False, shadow=False),
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


def custom_kernel(data: input_t) -> output_t:
    global _LEFT_32768_ERROR, _LEFT_LARGE_FALLBACKS, _LARGE_FP8_HITS, _LARGE_FP8_FALLBACKS, _LARGE_FP8_ERROR
    global _FUSED_CTA_HITS, _FUSED_CTA_FALLBACKS, _FUSED_CTA_ERROR

    batch, n, _ = data.shape
    is_f32_cuda = data.is_cuda and data.dtype == torch.float32

    if is_f32_cuda and _HAVE_TRITON and n == 32:
        return _triton_cholesky32(data)

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
            l = _left_looking_large(data[0], **_LARGE_CFG[n])
            if torch.isfinite(l.diagonal()).all().item():
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
