"""Run the experiment-020 Nsight Compute comparison on Modal B200."""

import argparse
import base64
import json
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "020-panel-inner-subtile"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
        )
        .entrypoint([])
        .pip_install("torch", "numpy", "ninja")
        .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
        .add_local_file(
            str(EXP / "baseline-exp019.py"),
            "/root/baseline_submission.py",
            copy=True,
        )
        .add_local_file(
            str(EXP / "candidate-subtile64.py"),
            "/root/candidate_submission.py",
            copy=True,
        )
        .add_local_file(str(EXP / "ncu_workload.py"), "/root/ncu_workload.py", copy=True)
        .add_local_file(str(EXP / "ncu_remote.py"), "/root/ncu_remote.py", copy=True)
    )
    app = modal.App.lookup("gpumode-cholesky-exp020", create_if_missing=True)
    parts = []
    meta = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "/root/ncu_remote.py",
            image=image,
            gpu="B200",
            app=app,
            timeout=args.timeout,
            workdir="/root",
        )
        try:
            for line in sandbox.stdout:
                line = line.rstrip("\n")
                if line.startswith("NCU_META:"):
                    meta = json.loads(line[len("NCU_META:") :])
                elif line.startswith("NCU_B64_PART:"):
                    _, index, encoded = line.split(":", 2)
                    parts.append((int(index), encoded))
                else:
                    print(line)
            sandbox.wait()
            stderr = sandbox.stderr.read()
            if stderr and stderr.strip():
                print(stderr, file=sys.stderr)
        finally:
            sandbox.terminate()
    if meta is None or not parts:
        raise RuntimeError("remote NCU collector returned incomplete output")
    output = ROOT / args.output
    output.write_bytes(base64.b64decode("".join(x for _, x in sorted(parts))))
    meta_output = ROOT / args.meta
    meta_output.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {meta_output}")


if __name__ == "__main__":
    main()
