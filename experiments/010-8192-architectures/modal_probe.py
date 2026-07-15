"""Launch the bounded experiment-010 probe on one Modal B200."""

import argparse
import json
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent

IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install("torch", "numpy", "ninja")
    .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
    .add_local_file(str(ROOT / "submission.py"), "/root/baseline.py", copy=True)
    .add_local_file(str(EXP / "candidates.py"), "/root/candidates.py", copy=True)
    .add_local_file(str(EXP / "probe_runner.py"), "/root/probe_runner.py", copy=True)
    .add_local_file(str(EXP / "profile_runner.py"), "/root/profile_runner.py", copy=True)
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--runner", choices=["probe_runner", "profile_runner"], default="probe_runner"
    )
    args = parser.parse_args()

    app = modal.App.lookup("gpumode-cholesky-exp010", create_if_missing=True)
    payload = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python", "-u", f"/root/{args.runner}.py",
            image=IMAGE,
            gpu="B200",
            app=app,
            timeout=args.timeout,
            workdir="/root",
        )
        try:
            for line in sandbox.stdout:
                line = line.rstrip("\n")
                print(line)
                if line.startswith("RESULT_JSON:"):
                    payload = json.loads(line[len("RESULT_JSON:"):])
            sandbox.wait()
            err = sandbox.stderr.read()
            if err and err.strip():
                print(err, file=sys.stderr)
        finally:
            sandbox.terminate()

    if payload is None:
        return 1
    output = Path(args.json)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
