"""Verify / benchmark the cholesky submission on a real B200 via a Modal sandbox.

Spins up a Modal Sandbox on a B200 GPU, bakes in the vendored reference harness
+ the submission, and runs `_gpu_runner.py` inside it. This gives real Blackwell
cuSOLVER numerics and timings BEFORE spending popcorn leaderboard quota.

Prereqs: `modal` installed locally and authenticated (`~/.modal.toml`).

Usage:
    python scripts/modal_verify.py            # correctness on the test grid
    python scripts/modal_verify.py benchmark  # per-shape timings on the 15-shape grid
    python scripts/modal_verify.py benchmark --gpu B200 --json results/baseline-benchmark.json

Note: B200 sandbox time is billed per second. `verify` is cheap (small shapes);
`benchmark` runs the full grid incl. a 32768x32768 matrix and costs more.
"""

import argparse
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

# CUDA *devel* base so `nvcc` is available for torch.utils.cpp_extension.load_inline
# (the plain pip-torch image has no nvcc). 13.0.0 matches the popcorn runner's
# torch 2.12.0+cu130. pip `torch` currently resolves to the same cu130 wheel, so
# nvcc 13.0 and torch's CUDA agree. The Blackwell (sm_100) kernels ship in that
# wheel. `.entrypoint([])` clears the nvidia image's default entrypoint script.
def _build_image(submission_path, candidate_path=None):
    # /root/candidate.py always exists so `pairedgrid` can load it; when no
    # --candidate is given it is the same file as the submission, which is
    # exactly the null-calibration configuration.
    candidate_path = candidate_path or submission_path
    return (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install("torch", "numpy", "ninja")
    .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
    .add_local_file(str(submission_path), "/root/submission.py", copy=True)
    .add_local_file(str(candidate_path), "/root/candidate.py", copy=True)
    .add_local_file(
        str(
            ROOT
            / "experiments"
            / "009-combined-shape-frontiers"
            / "baseline-exp008.py"
        ),
        "/root/baseline_exp008.py",
        copy=True,
    )
    .add_local_file(
        str(
            ROOT
            / "experiments"
            / "012-large-left-looking-frontiers"
            / "baseline-exp009.py"
        ),
        "/root/baseline_exp009.py",
        copy=True,
    )
    .add_local_file(
        str(
            ROOT
            / "experiments"
            / "013-1x32768-no-cusolver"
            / "baseline-exp012.py"
        ),
        "/root/baseline_exp012.py",
        copy=True,
    )
    .add_local_file(
        str(
            ROOT
            / "experiments"
            / "028-dual-matrix-persistent"
            / "baseline-exp021.py"
        ),
        "/root/baseline_exp028.py",
        copy=True,
    )
    .add_local_file(
        str(
            ROOT
            / "experiments"
            / "033-fp16x3-panels"
            / "baseline-l4.py"
        ),
        "/root/baseline_sched.py",
        copy=True,
    )
    .add_local_file(
        str(
            ROOT
            / "experiments"
            / "034-mxfp8-32768"
            / "baseline.py"
        ),
        "/root/baseline_exp034.py",
        copy=True,
    )
    .add_local_file(str(ROOT / "scripts" / "_gpu_runner.py"), "/root/_gpu_runner.py", copy=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="verify",
        choices=[
            "coldimport",
            "verify",
            "benchmark",
            "probe",
            "precprobe",
            "emuprobe",
            "fusionprobe",
            "frontierprobe",
            "largefrontierprobe",
            "nocusolverprobe",
            "dualprobe",
            "schedprobe",
            "dotprobe",
            "mxprobe",
            "pairedgrid",
            "familygrid",
            "shapediag",
            "midprobe",
            "microprobe",
            "memoprobe",
            "officialbench",
            "asmprobe",
            "coopprobe",
            "coopphase",
            "n128phase",
            "n256phase",
        ],
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="path to a second submission.py uploaded as /root/candidate.py "
        "(pairedgrid mode; pass the same file as --submission for the null "
        "calibration run)",
    )
    parser.add_argument(
        "--submission",
        default=None,
        help="path to the submission.py to upload as /root/submission.py "
        "(defaults to root submission.py; use the exp-013 candidate for probes)",
    )
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--json", default=None, help="write RESULT_JSON payload to this path")
    parser.add_argument(
        "--shapes",
        default=None,
        help="comma-separated n values to restrict the benchmark grid (cost saver), e.g. 32,64,128",
    )
    parser.add_argument(
        "--emu",
        action="store_true",
        help="enable cuBLAS BF16x9 FP32 emulation (CUBLAS_FP32_EMULATED_BF16X9_MATH=1) in the sandbox",
    )
    args = parser.parse_args()

    submission_path = (
        Path(args.submission).resolve()
        if args.submission
        else ROOT / "submission.py"
    )
    candidate_path = Path(args.candidate).resolve() if args.candidate else None
    image = _build_image(submission_path, candidate_path)
    print(f"submission -> {submission_path}", file=sys.stderr)
    if candidate_path:
        print(f"candidate  -> {candidate_path}", file=sys.stderr)

    app = modal.App.lookup("gpumode-cholesky-verify", create_if_missing=True)
    result_payload = None

    # Run the GPU runner as the sandbox's ENTRYPOINT command (not via .exec()).
    # Its stdout/stderr stream over Modal's standard control channel, avoiding
    # the newer per-task command-router direct connection (which is blocked in
    # some restricted-egress environments).
    runner_args = ["python", "-u", "/root/_gpu_runner.py", args.mode]
    if args.shapes:
        runner_args.append(args.shapes)
    if args.emu:
        runner_args.append("emu")

    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            *runner_args,
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
                    result_payload = json.loads(line[len("RESULT_JSON:"):])
            sandbox.wait()
            err = sandbox.stderr.read()
            if err and err.strip():
                print("--- stderr ---", file=sys.stderr)
                print(err, file=sys.stderr)
        finally:
            sandbox.terminate()

    if result_payload is None:
        print("No RESULT_JSON emitted -- the run likely errored.", file=sys.stderr)
        return 1

    if args.json:
        out = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result_payload, indent=2))
        print(f"\nwrote {out}")

    ok = result_payload.get("passed", True) and (
        result_payload.get("mode") != "benchmark" or result_payload.get("geomean_us") is not None
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
