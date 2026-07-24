"""Free numerical/property gates for experiment 057 candidate helpers."""

import ast
import pathlib

import torch


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = (HERE / "candidate-v1-blocked-inverse.py").read_text()
TREE = ast.parse(SOURCE)
BLOCKED = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "_blocked_tri_inv"
)
namespace = {"torch": torch, "_EXP057_V1_INVERSE_CALLS": 0}
exec(compile(ast.Module(body=[BLOCKED], type_ignores=[]), "<v1-helper>", "exec"), namespace)
blocked_tri_inv = namespace["_blocked_tri_inv"]

torch.manual_seed(5701)
failures = []
for n, base in ((128, 32), (256, 64), (512, 128), (1024, 256)):
    raw = torch.randn(n, n, dtype=torch.float64)
    lower = torch.linalg.cholesky(raw @ raw.T + n * torch.eye(n))
    inverse = blocked_tri_inv(lower, base)
    residual = (
        inverse @ lower - torch.eye(n, dtype=torch.float64)
    ).abs().max().item()
    upper = torch.triu(inverse, diagonal=1).abs().max().item()
    ok = residual < 1.0e-8 and upper == 0.0
    print(
        f"n={n} base={base} inverse_residual={residual:.3e} "
        f"upper={upper:.1e} ok={ok}"
    )
    if not ok:
        failures.append((n, base, residual, upper))

source_contracts = {
    "exact target shape": "tuple(data.shape) == (1, 16384, 16384)" in SOURCE,
    "non-target delegation": "return _ranked.custom_kernel(data)" in SOURCE,
    "fallback counter": "_EXP057_V1_FALLBACKS += 1" in SOURCE,
    "positive hit counter": "_EXP057_V1_HITS += 1" in SOURCE,
    "no other shape literal": all(
        token not in SOURCE
        for token in ("(1, 8192, 8192)", "(1, 32768, 32768)")
    ),
}
for name, ok in source_contracts.items():
    print(f"{name}: {ok}")
    if not ok:
        failures.append(("source-contract", name))

print("FAILURES:", failures if failures else "none")
raise SystemExit(1 if failures else 0)
