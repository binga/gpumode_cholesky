"""Free gate: CPU float64/float32 correctness of the two functions experiment
052 inserts, extracted verbatim from candidate-v1.py so the tested text is the
shipped text.  Validates the as_strided block views, the level combines and the
full factorization against torch.linalg.cholesky.
"""

import pathlib
import re

import torch

SRC = (pathlib.Path(__file__).resolve().parent / "candidate-v1.py").read_text()

start = SRC.index("def _blocked_tri_inv(")
end = SRC.index("# ---", SRC.index("    return factor", SRC.index("def _large_v2(")))
namespace = {"torch": torch}
exec(SRC[start:end], namespace)  # noqa: S102 - exact candidate text
_blocked_tri_inv = namespace["_blocked_tri_inv"]
_large_v2 = namespace["_large_v2"]

torch.manual_seed(0)
failures = []

# --- triangular inverse ----------------------------------------------------
for n, base in [(256, 256), (512, 128), (1024, 256), (2048, 256), (2048, 64)]:
    a = torch.randn(n, n, dtype=torch.float64)
    spd = a @ a.T + n * torch.eye(n, dtype=torch.float64)
    lower = torch.linalg.cholesky(spd)
    got = _blocked_tri_inv(lower, base)
    err = (got @ lower - torch.eye(n, dtype=torch.float64)).abs().max().item()
    ok = err < 1e-8
    print(f"tri_inv n={n:5d} base={base:4d} max|L^-1 L - I|={err:.3e} ok={ok}")
    if not ok:
        failures.append(("tri_inv", n, base, err))
    # strictly upper part of the inverse must be exactly zero
    up = torch.triu(got, diagonal=1).abs().max().item()
    if up != 0.0:
        failures.append(("tri_inv-upper", n, base, up))

# --- full factorization ----------------------------------------------------
for n, nb, base in [(1024, 256, 64), (2048, 512, 128), (2048, 256, 64)]:
    a = torch.randn(n, n)
    spd = (a @ a.T) / n + torch.eye(n)
    got = _large_v2(spd, nb=nb, update="tf32", inv_base=base)
    ref = torch.linalg.cholesky(spd)
    err = (got - ref).abs().max().item()
    recon = (got @ got.T - spd).abs().sum(dim=0).max().item()
    scale = spd.abs().sum(dim=0).max().item()
    allowed = 20.0 * n * torch.finfo(torch.float32).eps * scale
    upper = torch.triu(got, diagonal=1).abs().max().item()
    ok = recon < allowed and upper == 0.0
    print(
        f"large_v2 n={n:5d} nb={nb:4d} base={base:4d} max|L-Lref|={err:.3e} "
        f"residual/allowed={recon / allowed:.3f} upper={upper} ok={ok}"
    )
    if not ok:
        failures.append(("large_v2", n, nb, recon / allowed))

print("FAILURES:", failures if failures else "none")
raise SystemExit(1 if failures else 0)
