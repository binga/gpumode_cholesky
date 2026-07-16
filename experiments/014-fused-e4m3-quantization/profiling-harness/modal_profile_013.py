"""Launch the experiment-013 paired runner on one Modal B200 sandbox.

This driver never accepts an arbitrary baseline: the experiment-012 snapshot is
hard-coded and checked against the ranked commit's SHA-256 before upload and
again inside the sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[3]
HARNESS = Path(__file__).resolve().parent
CONFIG_PATH = HARNESS / "profile-config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
BASELINE_PATH = ROOT / CONFIG["baseline"]["source_path"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("target", "families", "full-grid", "all"))
    parser.add_argument("--candidate", default=str(ROOT / "submission.py"))
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate).resolve()
    output_path = Path(args.json).resolve()
    expected = CONFIG["baseline"]["source_sha256"]
    observed = _sha256(BASELINE_PATH)
    if observed != expected:
        print(
            f"refusing stale baseline: expected {expected}, observed {observed}",
            file=sys.stderr,
        )
        return 2
    if not candidate_path.is_file():
        print(f"candidate does not exist: {candidate_path}", file=sys.stderr)
        return 2

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
        )
        .entrypoint([])
        .pip_install("torch", "numpy", "ninja")
        .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
        .add_local_file(str(BASELINE_PATH), "/root/baseline.py", copy=True)
        .add_local_file(str(candidate_path), "/root/candidate.py", copy=True)
        .add_local_file(str(CONFIG_PATH), "/root/profile-config.json", copy=True)
        .add_local_file(
            str(HARNESS / "gpu_profile_013.py"),
            "/root/gpu_profile_013.py",
            copy=True,
        )
    )

    app = modal.App.lookup("gpumode-cholesky-exp014", create_if_missing=True)
    payload = None
    payload_chunks = []
    expected_payload_size = None
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "/root/gpu_profile_013.py",
            args.mode,
            image=image,
            gpu=args.gpu,
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
                elif line.startswith("RESULT_JSON_BEGIN:"):
                    expected_payload_size = int(
                        line[len("RESULT_JSON_BEGIN:") :]
                    )
                    payload_chunks = []
                elif line.startswith("RESULT_JSON_CHUNK:"):
                    payload_chunks.append(line[len("RESULT_JSON_CHUNK:") :])
                elif line == "RESULT_JSON_END":
                    serialized = "".join(payload_chunks)
                    if (
                        expected_payload_size is not None
                        and len(serialized) != expected_payload_size
                    ):
                        raise RuntimeError(
                            "incomplete RESULT_JSON transport: "
                            f"expected {expected_payload_size}, got {len(serialized)}"
                        )
                    payload = json.loads(serialized)
            sandbox.wait()
            stderr = sandbox.stderr.read()
            if stderr and stderr.strip():
                print(stderr, file=sys.stderr)
        finally:
            sandbox.terminate()

    if payload is None:
        print("no RESULT_JSON emitted", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output_path}")
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
