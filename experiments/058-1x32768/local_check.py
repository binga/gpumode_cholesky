"""CPU free gate for the exact blocked-inverse helper in candidate-v1."""

import pathlib

import torch


SOURCE = (pathlib.Path(__file__).parent / "candidate-v1.py").read_text()
START = SOURCE.index("def _blocked_tri_inv_32768(")
END = SOURCE.index(
    "\ndef _left_looking_32768_blocked_inverse(", START
)
namespace = {"torch": torch}
exec(SOURCE[START:END], namespace)
blocked_inverse = namespace["_blocked_tri_inv_32768"]

torch.manual_seed(58032768)
failures = []
for n, base in ((64, 16), (128, 16), (256, 32), (512, 64)):
    raw = torch.randn(n, n, dtype=torch.float64)
    spd = raw @ raw.T + n * torch.eye(n, dtype=torch.float64)
    lower = torch.linalg.cholesky(spd)
    inverse = blocked_inverse(lower, base)
    residual = (
        inverse @ lower - torch.eye(n, dtype=torch.float64)
    ).abs().max().item()
    upper = torch.triu(inverse, diagonal=1).abs().max().item()
    passed = residual < 1.0e-10 and upper == 0.0
    print(
        f"n={n} base={base} max_residual={residual:.3e} "
        f"upper={upper:.3e} passed={passed}"
    )
    if not passed:
        failures.append((n, base, residual, upper))

print("failures:", failures if failures else "none")
raise SystemExit(1 if failures else 0)
