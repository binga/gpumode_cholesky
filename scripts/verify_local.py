"""Zero-cost, device-agnostic property check for the cholesky submission.

Runs the *real* reference checker (`check_implementation` from the vendored
harness) against `custom_kernel` on small shapes. Uses CUDA if available, else
CPU. On a CPU-only machine this validates STRUCTURE and LOGIC (lower-triangular,
positive diagonal, reconstruction residual) but NOT GPU cuSOLVER numerics --
for real B200 numerics use `scripts/modal_verify.py`.

Usage:
    python scripts/verify_local.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))

from reference import check_implementation, generate_input  # noqa: E402
from submission import custom_kernel  # noqa: E402

# Small subset of the task.yml `tests:` grid -- cheap enough for CPU.
SPECS = [
    {"batch": 16, "n": 32, "cond": 2, "seed": 53124},
    {"batch": 16, "n": 64, "cond": 2, "seed": 53125},
    {"batch": 16, "n": 128, "cond": 2, "seed": 3321},
    {"batch": 8, "n": 128, "cond": 5, "seed": 1200, "case": "spectrum"},
    {"batch": 8, "n": 128, "cond": 5, "seed": 1201, "case": "diagonal"},
    {"batch": 4, "n": 256, "cond": 4, "seed": 32524, "case": "lowrank"},
    {"batch": 4, "n": 256, "cond": 4, "seed": 32525, "case": "rowscale"},
    {"batch": 4, "n": 256, "cond": 1, "seed": 32526, "case": "tridiagonal"},
    # Degenerate edges.
    {"batch": 1, "n": 1, "cond": 2, "seed": 7},
    {"batch": 1, "n": 32, "cond": 2, "seed": 8},
]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}")
    failures = 0
    for spec in SPECS:
        data = generate_input(**spec)
        output = custom_kernel(data.clone())
        good, message = check_implementation(data, output)
        label = f"batch={spec['batch']} n={spec['n']} case={spec.get('case', 'dense')}"
        print(f"[{'PASS' if good else 'FAIL'}] {label}: {message}")
        failures += 0 if good else 1
    print(f"\n{len(SPECS) - failures}/{len(SPECS)} specs passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
