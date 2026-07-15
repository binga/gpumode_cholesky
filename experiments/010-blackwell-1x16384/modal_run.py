"""Launch the bounded experiment-010 runner on a Modal B200."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install("torch", "numpy", "ninja")
    .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
    .add_local_file(str(HERE / "baseline-exp009.py"), "/root/exp010/baseline-exp009.py", copy=True)
    .add_local_file(str(HERE / "candidates.py"), "/root/exp010/candidates.py", copy=True)
    .add_local_file(str(HERE / "gpu_runner.py"), "/root/exp010/gpu_runner.py", copy=True)
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("paired", "families", "profile"))
    parser.add_argument("variants")
    parser.add_argument("--json")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    app = modal.App.lookup("gpumode-cholesky-exp010", create_if_missing=True)
    payload = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "/root/exp010/gpu_runner.py",
            args.mode,
            args.variants,
            image=IMAGE,
            gpu="B200",
            app=app,
            timeout=args.timeout,
            workdir="/root/exp010",
        )
        try:
            for line in sandbox.stdout:
                line = line.rstrip("\n")
                print(line)
                if line.startswith("RESULT_JSON:"):
                    payload = json.loads(line[len("RESULT_JSON:") :])
            sandbox.wait()
            error = sandbox.stderr.read()
            if error and error.strip():
                print(error, file=sys.stderr)
        finally:
            sandbox.terminate()

    if payload is None:
        return 1
    if args.json:
        path = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
