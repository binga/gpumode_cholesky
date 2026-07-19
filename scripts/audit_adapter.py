"""Normalize the repository's paired Modal B200 harness for audit-kernel.

The frozen baseline is ``audit/baseline-submission.py`` (ranked #888352).
The candidate is always the repository-root ``submission.py``.  The adapter
measures only the three user-approved campaign shapes and writes the normalized
schema consumed by ``skills/audit-kernel/scripts/audit_kernel.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKLOAD_IDS = {
    (2, 2048): "2x2048-fp32-dense",
    (1, 4096): "1x4096-fp32-dense",
    (2, 4096): "2x4096-fp32-dense",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_paired(raw_path: Path) -> dict:
    command = [
        "uv",
        "run",
        "--with",
        "modal",
        "python",
        "scripts/modal_verify.py",
        "pairedgrid",
        "--submission",
        "audit/baseline-submission.py",
        "--candidate",
        "submission.py",
        "--shapes",
        "2048,4096",
        "--json",
        str(raw_path),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return json.loads(raw_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--mode", choices=("quick", "system", "kernel", "full"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".pairedgrid.json")
    paired = _run_paired(raw_path)

    rows = []
    checks = []
    for row in paired["shapes"]:
        key = (int(row["batch"]), int(row["n"]))
        workload_id = WORKLOAD_IDS.get(key)
        if workload_id is None:
            continue
        passed = bool(row["ok"] and row["candidate_ok"] and not row["new_fallbacks"])
        checks.append(
            {
                "name": workload_id,
                "value": row["candidate_msg"],
                "passed": passed,
            }
        )
        rows.append(
            {
                "id": workload_id,
                "latency_us": float(row["candidate_us"]),
                "cv": float(row["ratio_mad"]),
                "metrics": {},
            }
        )

    if set(r["id"] for r in rows) != set(WORKLOAD_IDS.values()):
        print("paired harness did not return all contract workloads", file=sys.stderr)
        return 5

    normalized = {
        "schema_version": 1,
        "run_id": raw_path.stem,
        "candidate_id": _sha256(ROOT / "submission.py"),
        "environment": {
            "gpu": "NVIDIA B200",
            "driver": "Modal managed",
            "cuda": "13.0",
            "framework": "torch 2.13.0+cu130",
        },
        "correctness": {
            "passed": all(c["passed"] for c in checks),
            "checks": checks,
        },
        "workloads": rows,
        "artifacts": {
            "pairedgrid_json": str(raw_path),
        },
    }
    output.write_text(json.dumps(normalized, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
