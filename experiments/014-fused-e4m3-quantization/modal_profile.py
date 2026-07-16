"""Launch the bounded experiment-013 paired probe on a Modal B200."""

import argparse
import json
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "014-fused-e4m3-quantization"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", default="candidate-recursive-fp8.py"
    )
    parser.add_argument("--json", required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    candidate = EXP / args.candidate
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
        )
        .entrypoint([])
        .pip_install("torch", "numpy", "ninja")
        .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
        .add_local_file(
            str(EXP / "baseline-exp012.py"),
            "/root/baseline_exp012.py",
            copy=True,
        )
        .add_local_file(
            str(candidate), "/root/candidate_exp014.py", copy=True
        )
        .add_local_file(
            str(EXP / "profile_runner.py"),
            "/root/profile_runner.py",
            copy=True,
        )
    )

    app = modal.App.lookup(
        "gpumode-cholesky-exp014", create_if_missing=True
    )
    payload = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "/root/profile_runner.py",
            image=image,
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
                    payload = json.loads(line[len("RESULT_JSON:") :])
            sandbox.wait()
            stderr = sandbox.stderr.read()
            if stderr and stderr.strip():
                print(stderr, file=sys.stderr)
        finally:
            sandbox.terminate()

    if payload is None:
        raise RuntimeError("Modal runner emitted no RESULT_JSON payload")
    output = Path(args.json)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
