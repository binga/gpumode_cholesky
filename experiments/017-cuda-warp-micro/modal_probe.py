"""Launch the experiment-015 paired probe on a Modal B200.

Uploads the exact ranked baseline (root submission.py, #880770), one candidate,
the vendored reference checker, and probe_runner.py, then runs the paired
probe inside a sandbox. Authorized by program.md's standing Modal
authorization (bounded benchmark package to the owner's account).
"""

import argparse
import json
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "017-cuda-warp-micro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="candidate-a-fused-cta.py")
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline file (defaults to root submission.py = #880770)",
    )
    parser.add_argument("--json", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--shapes",
        default=None,
        help="comma-separated BxN filters, e.g. 64x256,640x512",
    )
    parser.add_argument("--nofam", action="store_true", help="skip family sweep")
    parser.add_argument(
        "--profile", action="store_true", help="per-kernel torch profiler pass"
    )
    parser.add_argument(
        "--fullgrid", action="store_true", help="all 15 ranked shapes, no families"
    )
    parser.add_argument(
        "--blocking", action="store_true", help="CUDA_LAUNCH_BLOCKING=1"
    )
    args = parser.parse_args()

    baseline = Path(args.baseline) if args.baseline else ROOT / "submission.py"
    candidate = EXP / args.candidate
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
        )
        .entrypoint([])
        .pip_install("torch", "numpy", "ninja")
        .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
        .add_local_file(str(baseline), "/root/baseline_submission.py", copy=True)
        .add_local_file(str(candidate), "/root/candidate_submission.py", copy=True)
        .add_local_file(str(EXP / "probe_runner.py"), "/root/probe_runner.py", copy=True)
    )

    runner_args = ["python", "-u", "/root/probe_runner.py"]
    if args.shapes:
        runner_args.append(args.shapes)
    if args.nofam:
        runner_args.append("nofam")
    if args.profile:
        runner_args.append("profile")
    if args.fullgrid:
        runner_args.append("fullgrid")
    if args.blocking:
        runner_args.append("blocking")

    app = modal.App.lookup("gpumode-cholesky-exp017", create_if_missing=True)
    payload = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            *runner_args,
            image=image,
            gpu="B200",
            app=app,
            timeout=args.timeout,
            workdir="/root",
        )
        try:
            for line in sandbox.stdout:
                line = line.rstrip("\n")
                if line.startswith("RESULT_JSON:"):
                    payload = json.loads(line[len("RESULT_JSON:") :])
                else:
                    print(line)
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
