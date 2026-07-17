"""Launch the experiment-022 standalone rank-4 probe on a Modal B200.

Uploads the exact ranked baseline (exp-021, #882958), one candidate,
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
EXP = ROOT / "experiments" / "022-rank4-n32"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="candidate-rank4.py")
    parser.add_argument(
        "--baseline",
        default=None,
        help="baseline file (defaults to frozen exp-021 ranked snapshot)",
    )
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--artifacts",
        default=None,
        help="optional output zip for candidate Triton compiled artifacts",
    )
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

    baseline = Path(args.baseline) if args.baseline else EXP / "baseline-exp021.py"
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

    app = modal.App.lookup("gpumode-cholesky-exp022", create_if_missing=True)
    payload = None
    artifact_parts = []
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
                elif line.startswith("ARTIFACT_B64_PART:"):
                    _, index, encoded = line.split(":", 2)
                    artifact_parts.append((int(index), encoded))
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
    if args.artifacts:
        if not artifact_parts:
            raise RuntimeError("Modal runner emitted no ARTIFACT_B64 payload")
        import base64

        artifact_bytes = base64.b64decode(
            "".join(part for _, part in sorted(artifact_parts))
        )
        artifact_output = Path(args.artifacts)
        if not artifact_output.is_absolute():
            artifact_output = ROOT / artifact_output
        artifact_output.parent.mkdir(parents=True, exist_ok=True)
        artifact_output.write_bytes(artifact_bytes)
        print(f"wrote {artifact_output}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
