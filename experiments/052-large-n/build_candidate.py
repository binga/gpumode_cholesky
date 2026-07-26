"""Generate experiment 052 candidates from the exact ranked incumbent.

Every candidate is the byte-exact `baseline-890798.py` plus a small number of
exact-substring edits, so the diff the main thread has to integrate is always
mechanically recoverable.  Scope: the `batch == 1 and n >= 8192` dispatch only.

Usage:  python3 build_candidate.py v1 v2 ...
"""

import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE / "baseline-890798.py"
BASELINE_SHA = "fd3072b5160ea31b92464de4aa2ce06ebdc9b70994c6279b494e7107994244c1"

# --- anchor: end of the exp-016a large-n section, start of custom_kernel -----
ANCHOR = "\n# ---------------------------------------------------------------------------\n\n\n\ndef custom_kernel(data: input_t) -> output_t:"

DISPATCH_OLD = """    if is_f32_cuda and batch == 1 and n in _LARGE_CFG:
        try:
            l = _left_looking_large(data[0], **_LARGE_CFG[n])"""

DISPATCH_NEW = """    if is_f32_cuda and batch == 1 and n in _LARGE_CFG_V2:
        try:
            l = _large_v2(data[0], **_LARGE_CFG_V2[n])"""

HEADER = '''

# ---------------------------------------------------------------------------
# Experiment 052: large single-matrix left-looking path, v2.
#
# Measured against the ranked incumbent (results/inc-890798-shapediag.json):
#
#   1x8192   2.53ms cuSOLVER potrf, 1.36ms `trsm_right_kernel` (96 launches),
#            0.51ms elementwise, only 0.61ms in tensor-core GEMM.
#   1x16384  5.05ms potrf, 4.50ms `kernel_trsm_l_mul32` (56 launches),
#            1.47ms elementwise, 0.24ms of `torch.eye` fills.
#   1x32768  11.19ms potrf, 9.00ms `kernel_trsm_l_mul32` (112 launches),
#            5.26ms elementwise, 0.82ms of fills.
#
# The triangular solves and the scaffolding around them, not the O(n^3) GEMM,
# are the cost.  cuBLAS `kernel_trsm_l_mul32` costs ~77us per launch almost
# independently of the operand size, and `torch.linalg.solve_triangular` on a
# small batch issues one such launch per matrix, so the shipped
# `_tri_inv_recursive` pays ~620us for every nb x nb inverse it forms.
#
# No new cuSOLVER call site: the per-block `cholesky_ex` on the nb x nb
# diagonal is exactly the one the ranked path already issues, at the same block
# sizes and the same call count.  Everything else is torch tensor-core ops.
# ---------------------------------------------------------------------------

_LARGE_CFG_V2 = {CFG}
'''

# --- v1: blocked inverse (batched leaves), merged block-column update -------
BODY_V1 = '''

def _blocked_tri_inv(lower: torch.Tensor, base: int = 256) -> torch.Tensor:
    """Explicit inverse of a lower-triangular factor, breadth-first.

    Level 0 inverts the `base`-sized diagonal blocks -- mutually independent --
    with a single batched `solve_triangular`.  Each later level doubles the
    block size with inv([[A,0],[B,C]]) = [[Ai,0],[-Ci B Ai, Ci]], evaluated for
    every pair at that level at once as two strided batched GEMMs read straight
    off views of `lower` / `inverse`.
    """
    n = lower.shape[0]
    if n <= base or n % base or (n & (n - 1)):
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    count = n // base
    leaf_shape = (count, base, base)
    leaf_stride = (base * n + base, n, 1)
    blocks = lower.as_strided(leaf_shape, leaf_stride).contiguous()
    identity = torch.eye(base, device=lower.device, dtype=lower.dtype)
    inverse.as_strided(leaf_shape, leaf_stride).copy_(
        torch.linalg.solve_triangular(
            blocks, identity.expand(leaf_shape).contiguous(), upper=False
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


def _large_v2(mat: torch.Tensor, nb: int, update: str, inv_base: int) -> torch.Tensor:
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            j = k + kb
            if update == "mxfp8":
                block = mat[k : k + kb, k : k + kb].contiguous()
                if k:
                    row = factor[k : k + kb, :k]
                    block.addmm_(
                        row, row.transpose(-1, -2), beta=1.0, alpha=-1.0
                    )
            else:
                block = mat[k:, k : k + kb].contiguous()
                if k:
                    block.addmm_(
                        factor[k:, :k],
                        factor[k : k + kb, :k].transpose(-1, -2),
                        beta=1.0,
                        alpha=-1.0,
                    )
            lkk = torch.linalg.cholesky_ex(block[:kb], check_errors=False).L
            factor[k : k + kb, k : k + kb] = lkk
            if j >= n:
                break
            if update == "mxfp8":
                panel = mat[j:, k : k + kb].contiguous()
                if k:
                    _mxfp8_panel_update(
                        panel, factor[j:, :k], factor[k : k + kb, :k]
                    )
            else:
                panel = block[kb:]
            inverse = _blocked_tri_inv(lkk, inv_base)
            factor[j:, k : k + kb] = panel @ inverse.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor

'''

# --- v2: trsm-free inverse + FP16-resident left-looking update --------------
BODY_V2 = '''

def _blocked_tri_inv(lower: torch.Tensor, base: int = 1) -> torch.Tensor:
    """Explicit inverse of a lower-triangular factor, breadth-first.

    Every level doubles the block size using
    inv([[A,0],[B,C]]) = [[Ai,0],[-Ci B Ai, Ci]], evaluated for every pair at
    that level at once as two strided batched GEMMs read straight off views of
    `lower` / `inverse`.  With `base == 1` the recursion starts from the
    reciprocal of the diagonal and NO triangular solve is issued at all: the
    whole inverse is 1 elementwise kernel + 3*log2(n) small batched GEMM /
    copy launches.  That matters because cuBLAS `kernel_trsm_l_mul32` costs
    ~77us per launch regardless of operand size and
    `torch.linalg.solve_triangular` does not batch it below 16 matrices, so
    the shipped leaf-solve recursion pays ~620us per nb x nb inverse.

    `base > 1` keeps a batched `solve_triangular` at the leaves, which is
    faster only where cuBLAS actually picks `batch_trsm_left_kernel`
    (>= 16 leaves).
    """
    n = lower.shape[0]
    if n & (n - 1) or base < 1 or n % base:
        identity = torch.eye(n, device=lower.device, dtype=lower.dtype)
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    lower = lower.contiguous()
    inverse = torch.zeros_like(lower)
    if base == 1:
        inverse.diagonal().copy_(lower.diagonal().reciprocal())
    else:
        count = n // base
        leaf_shape = (count, base, base)
        leaf_stride = (base * n + base, n, 1)
        identity = torch.eye(base, device=lower.device, dtype=lower.dtype)
        inverse.as_strided(leaf_shape, leaf_stride).copy_(
            torch.linalg.solve_triangular(
                lower.as_strided(leaf_shape, leaf_stride).contiguous(),
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


def _large_v2(mat: torch.Tensor, nb: int, update: str, inv_base: int) -> torch.Tensor:
    """Left-looking blocked Cholesky with an FP16-resident factor shadow.

    `shadow` holds L in FP16 and is written once per block column, never
    re-quantised, so the left-looking update
    `A[k:, k:k+kb] -= L[k:, :k] @ L[k:k+kb, :k]^T` -- the O(n^3) term -- runs
    on FP16 tensor cores at ~2.2x the TF32 rate.  FP16 and TF32 carry the SAME
    10-bit significand, so this does not lose accuracy relative to the shipped
    TF32 update; only the product's own output rounding is new, worth
    eps_h/(eps32*n) = 0.50 / 0.25 / 0.13 of a scaled-residual unit at
    n = 8192 / 16384 / 32768 against a budget of 20.

    The panel apply `L21 = A21 @ inv(L11)^T` stays in FP32/TF32: the explicit
    inverse of an ill-conditioned diagonal block is the one quantity here with
    a real FP16 overflow risk, and it is only O(nb*n^2/2) work.
    """
    n = mat.shape[0]
    factor = torch.zeros_like(mat)
    shadow = torch.empty(n, n, device=mat.device, dtype=torch.float16)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            j = k + kb
            # One contiguous staging buffer for the whole block column: the
            # diagonal block and the panel below it are consecutive rows.
            block = mat[k:, k : k + kb].contiguous()
            if k:
                left = shadow[k : k + kb, :k].transpose(-1, -2)
                if update == "mxfp8":
                    # 32768 keeps the shipped MXFP8 panel product; only the
                    # diagonal block's update moves to FP16.
                    block[:kb].sub_(torch.mm(shadow[k : k + kb, :k], left))
                else:
                    block.sub_(torch.mm(shadow[k:, :k], left))
            lkk = torch.linalg.cholesky_ex(block[:kb], check_errors=False).L
            factor[k : k + kb, k : k + kb] = lkk
            shadow[k : k + kb, k : k + kb] = lkk
            if j >= n:
                break
            if update == "mxfp8" and k:
                _mxfp8_panel_update(
                    block[kb:], factor[j:, :k], factor[k : k + kb, :k]
                )
            inverse = _blocked_tri_inv(lkk, inv_base)
            solved = block[kb:] @ inverse.transpose(-1, -2)
            factor[j:, k : k + kb] = solved
            shadow[j:, k : k + kb] = solved
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return factor

'''

VARIANTS = {
    "v1": {
        "body": BODY_V1,
        "cfg": """{
    8192: dict(nb=2048, update="tf32", inv_base=256),
    16384: dict(nb=2048, update="tf32", inv_base=256),
    32768: dict(nb=4096, update="mxfp8", inv_base=256),
}""",
    },
    "v2": {
        "body": BODY_V2,
        "cfg": """{
    8192: dict(nb=2048, update="fp16", inv_base=1),
    16384: dict(nb=2048, update="fp16", inv_base=1),
    32768: dict(nb=4096, update="mxfp8", inv_base=256),
}""",
    },
    # v2b: v2's machinery, TF32 left-looking update instead of FP16, so the
    # FP16 term can be isolated from the trsm-free inverse term.
    "v2b": {
        "body": BODY_V2,
        "cfg": """{
    8192: dict(nb=2048, update="fp16", inv_base=1),
    16384: dict(nb=2048, update="fp16", inv_base=1),
    32768: dict(nb=4096, update="mxfp8", inv_base=1),
}""",
    },
}


def build(name):
    source = BASELINE.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != BASELINE_SHA:
        raise SystemExit(f"baseline hash mismatch: {digest}")
    spec = VARIANTS[name]
    block = HEADER.replace("{CFG}", spec["cfg"]) + spec["body"]
    if source.count(ANCHOR) != 1:
        raise SystemExit("anchor not unique")
    source = source.replace(ANCHOR, block + ANCHOR, 1)
    if source.count(DISPATCH_OLD) != 1:
        raise SystemExit("dispatch anchor not unique")
    source = source.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
    out = HERE / f"candidate-{name}.py"
    out.write_text(source)
    print(f"wrote {out} ({len(source)} bytes)")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["v1"]:
        build(arg)
