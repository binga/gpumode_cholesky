"""Run the exact experiment-059 source in a fresh Modal B200 sandbox."""

import argparse
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    source = Path(args.submission).resolve()
    output = Path(args.json).resolve()
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04",
            add_python="3.11",
        )
        .entrypoint([])
        .pip_install("torch", "numpy", "ninja")
        .env(
            {
                "TORCH_EXTENSIONS_DIR":
                    "/tmp/torch_extensions_exp059_f8d67dce"
            }
        )
        .add_local_dir(
            str(ROOT / "reference"),
            "/root/reference",
            copy=True,
        )
        .add_local_file(
            str(source),
            "/root/submission.py",
            copy=True,
        )
        .add_local_file(
            str(EXPERIMENT / "cold_import_runner.py"),
            "/root/cold_import_runner.py",
            copy=True,
        )
    )
    app = modal.App.lookup(
        "gpumode-cholesky-cold-import",
        create_if_missing=True,
    )
    payload = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "/root/cold_import_runner.py",
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
                    payload = json.loads(line.removeprefix("RESULT_JSON:"))
            sandbox.wait()
            stderr = sandbox.stderr.read()
            if stderr and stderr.strip():
                print(stderr)
        finally:
            sandbox.terminate()

    if payload is None:
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
