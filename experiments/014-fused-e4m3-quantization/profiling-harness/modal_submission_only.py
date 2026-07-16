"""Run experiment 014 while exporting only the candidate source to Modal.

The sandbox clones the public repository at the exact ranked commit to obtain
the already-public baseline and reference checker. The only local workspace
file added to the image is the candidate submission passed with --candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[3]
PUBLIC_REPOSITORY = "https://github.com/binga/gpumode_cholesky"
RANKED_COMMIT = "141d015aa54dee65109722f9a59742588f20926d"
BASELINE_SHA256 = "112ee017f96f4dafb95a173cf51bb59190c2ded1e7702a64992e6795985759dd"


REMOTE_RUNNER = r'''
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path("/root/repo")
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT))
from reference import check_implementation, generate_input

MODE = sys.argv[1]
EXPECTED_CANDIDATE_SHA = sys.argv[2]
EXPECTED_BASELINE_SHA = sys.argv[3]
BASELINE_PATH = ROOT / "experiments/012-large-left-looking-frontiers/submission.py"
CANDIDATE_PATH = ROOT / "candidate.py"
L2_FLUSH = None

FAMILY_SPECS = [
    {"batch": 1, "n": 32768, "cond": 2, "seed": 68368, "case": "dense"},
    {"batch": 1, "n": 32768, "cond": 5, "seed": 68371, "case": "spectrum"},
    {"batch": 1, "n": 32768, "cond": 4, "seed": 68369, "case": "lowrank"},
    {"batch": 1, "n": 32768, "cond": 4, "seed": 68372, "case": "rowscale"},
    {"batch": 1, "n": 32768, "cond": 5, "seed": 68373, "case": "diagonal"},
    {"batch": 1, "n": 32768, "cond": 1, "seed": 68370, "case": "tridiagonal"},
]

FULL_GRID = [
    {"batch": 4096, "n": 32, "cond": 2, "seed": 41032},
    {"batch": 1024, "n": 64, "cond": 2, "seed": 41064},
    {"batch": 256, "n": 128, "cond": 2, "seed": 41128},
    {"batch": 64, "n": 256, "cond": 2, "seed": 41256},
    {"batch": 16, "n": 512, "cond": 2, "seed": 41512},
    {"batch": 640, "n": 512, "cond": 2, "seed": 510512},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 42024},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 511024},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 44048},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
    {"batch": 1, "n": 4096, "cond": 2, "seed": 48096},
    {"batch": 2, "n": 4096, "cond": 2, "seed": 514096},
    {"batch": 1, "n": 8192, "cond": 2, "seed": 48192},
    {"batch": 1, "n": 16384, "cond": 2, "seed": 48284},
    {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
]

CONTRACTS = {
    "baseline": {
        "positive": ["_LEFT_32768_HITS"],
        "zero": ["_LEFT_LARGE_FALLBACKS"],
        "none": ["_LEFT_32768_ERROR"],
        "truthy": ["_HAVE_TRITON"],
    },
    "candidate": {
        "positive": [
            "_LEFT_32768_HITS",
            "_FUSED_E4M3_AMAX_HITS",
            "_FUSED_E4M3_QUANT_HITS",
        ],
        "zero": ["_LEFT_LARGE_FALLBACKS"],
        "none": ["_LEFT_32768_ERROR", "_FUSED_E4M3_QUANT_ERROR"],
        "truthy": ["_HAVE_TRITON"],
    },
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "custom_kernel", None)):
        raise RuntimeError(f"{path} has no custom_kernel")
    return module


def snapshot(module, contract):
    names = contract["positive"] + contract["zero"] + contract["none"] + contract["truthy"]
    return {name: getattr(module, name, "__MISSING__") for name in names}


def evaluate_contract(before, after, contract):
    failures = []
    deltas = {}
    for name in contract["positive"]:
        left, right = before.get(name), after.get(name)
        delta = right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
        deltas[name] = delta
        if delta is None or delta <= 0:
            failures.append(f"{name} delta must be positive, got {delta!r}")
    for name in contract["zero"]:
        left, right = before.get(name), after.get(name)
        delta = right - left if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
        deltas[name] = delta
        if delta != 0:
            failures.append(f"{name} delta must be zero, got {delta!r}")
    for name in contract["none"]:
        if after.get(name, "__MISSING__") is not None:
            failures.append(f"{name} must be None, got {after.get(name)!r}")
    for name in contract["truthy"]:
        if not after.get(name, False):
            failures.append(f"{name} must be truthy, got {after.get(name)!r}")
    return {"passed": not failures, "failures": failures, "deltas": deltas, "before": before, "after": after}


def official_check(data, output):
    passed, message = check_implementation(data, output)
    match = re.search(r"scaled_reconstruction_residual=([0-9.eE+-]+)", message)
    scaled = float(match.group(1)) if match else None
    diagonal = output.diagonal(dim1=-2, dim2=-1)
    return {
        "passed": bool(passed),
        "message": message,
        "finite": bool(torch.isfinite(output).all().item()),
        "positive_diagonal": bool((diagonal > 0).all().item()),
        "exactly_lower_triangular": bool((torch.triu(output, diagonal=1) == 0).all().item()),
        "scaled_reconstruction_residual": scaled,
        "tolerance_fraction": scaled / 20.0 if scaled is not None else None,
    }


def l2_flush():
    global L2_FLUSH
    if L2_FLUSH is None:
        L2_FLUSH = torch.empty(int(256e6 // 4), dtype=torch.float32, device="cuda")
    L2_FLUSH.zero_()


def timed_call(fn, data):
    l2_flush()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn(data)
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end) * 1e3


def summary(values):
    ordered = sorted(values)
    return {
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "best_us": ordered[0],
        "worst_us": ordered[-1],
        "samples_us": values,
    }


def run_families(modules):
    rows = []
    for spec in FAMILY_SPECS:
        print(f"family {spec['case']}", flush=True)
        data = generate_input(**spec)
        torch.cuda.synchronize()
        row = {"spec": spec}
        for name, module in modules.items():
            before = snapshot(module, CONTRACTS[name])
            output = module.custom_kernel(data)
            torch.cuda.synchronize()
            check = official_check(data, output)
            after = snapshot(module, CONTRACTS[name])
            contract = evaluate_contract(before, after, CONTRACTS[name])
            row[name] = {"check": check, "backend_contract": contract}
            del output
        row["passed"] = all(row[name]["check"]["passed"] and row[name]["backend_contract"]["passed"] for name in modules)
        rows.append(row)
        del data
        torch.cuda.empty_cache()
    return {"mode": "families", "families": rows, "passed": all(row["passed"] for row in rows), "timing_evidence": False}


def grid_rounds(n):
    if n <= 256:
        return 3, 20
    if n <= 2048:
        return 2, 10
    if n <= 8192:
        return 1, 6
    return 1, 4


def run_full_grid(modules):
    rows = []
    for spec in FULL_GRID:
        print(f"grid batch={spec['batch']} n={spec['n']}", flush=True)
        data = generate_input(**spec)
        torch.cuda.synchronize()
        warmup, rounds = grid_rounds(spec["n"])
        before = {name: snapshot(module, CONTRACTS[name]) for name, module in modules.items()}
        for index in range(warmup):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            for name in order:
                retained = modules[name].custom_kernel(data)
        torch.cuda.synchronize()
        durations = {"baseline": [], "candidate": []}
        retained = {}
        samples = []
        for index in range(rounds):
            order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
            sample = {"round": index, "order": list(order)}
            for name in order:
                output, elapsed = timed_call(modules[name].custom_kernel, data)
                retained[name] = output
                durations[name].append(elapsed)
                sample[name + "_us"] = elapsed
            sample["speedup"] = sample["baseline_us"] / sample["candidate_us"]
            samples.append(sample)
        checks = {name: official_check(data, retained[name]) for name in modules}
        after = {name: snapshot(module, CONTRACTS[name]) for name, module in modules.items()}
        is_target = spec["batch"] == 1 and spec["n"] == 32768
        if is_target:
            contracts = {name: evaluate_contract(before[name], after[name], CONTRACTS[name]) for name in modules}
        else:
            contracts = {name: {"passed": True, "skipped": True, "reason": "outside changed dispatch"} for name in modules}
        base = summary(durations["baseline"])
        candidate = summary(durations["candidate"])
        ratio = candidate["mean_us"] / base["mean_us"]
        off_target_ok = is_target or ratio <= 1.03
        passed = all(check["passed"] for check in checks.values()) and all(contract["passed"] for contract in contracts.values()) and off_target_ok
        rows.append({
            "spec": spec,
            "baseline": base,
            "candidate": candidate,
            "paired_samples": samples,
            "checks": checks,
            "backend_contract": contracts,
            "candidate_over_baseline": ratio,
            "off_target_gate_passed": off_target_ok,
            "outputs_retained_through_validation": True,
            "passed": passed,
        })
        del data, retained
        torch.cuda.empty_cache()
    base_geo = math.exp(statistics.fmean(math.log(row["baseline"]["mean_us"]) for row in rows))
    candidate_geo = math.exp(statistics.fmean(math.log(row["candidate"]["mean_us"]) for row in rows))
    improved = candidate_geo < base_geo
    return {
        "mode": "full-grid",
        "shapes": rows,
        "baseline_geomean_us": base_geo,
        "candidate_geomean_us": candidate_geo,
        "aggregate_speedup": base_geo / candidate_geo,
        "aggregate_improved": improved,
        "passed": all(row["passed"] for row in rows) and improved,
    }


def main():
    baseline_sha = sha256(BASELINE_PATH)
    candidate_sha = sha256(CANDIDATE_PATH)
    envelope = {
        "schema_version": 1,
        "experiment": "014-fused-e4m3-quantization-submission-only",
        "mode": MODE,
        "source_export": {
            "local_files_uploaded": ["candidate.py"],
            "public_repository": "https://github.com/binga/gpumode_cholesky",
            "public_commit": "141d015aa54dee65109722f9a59742588f20926d",
        },
        "baseline": {"expected_sha256": EXPECTED_BASELINE_SHA, "observed_sha256": baseline_sha},
        "candidate": {"expected_sha256": EXPECTED_CANDIDATE_SHA, "observed_sha256": candidate_sha},
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
    }
    if baseline_sha != EXPECTED_BASELINE_SHA or candidate_sha != EXPECTED_CANDIDATE_SHA:
        envelope.update({"passed": False, "error": "source lock failed"})
        print("RESULT_JSON:" + json.dumps(envelope), flush=True)
        return 2
    try:
        modules = {
            "baseline": load_module("baseline_141d015", BASELINE_PATH),
            "candidate": load_module("candidate_014", CANDIDATE_PATH),
        }
        result = run_families(modules) if MODE == "families" else run_full_grid(modules)
        envelope["result"] = result
        envelope["passed"] = bool(result["passed"])
    except Exception as exc:
        envelope.update({"passed": False, "runtime_error": repr(exc)})
    print("RESULT_JSON:" + json.dumps(envelope), flush=True)
    return 0 if envelope["passed"] else 1


raise SystemExit(main())
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("families", "full-grid"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    candidate = Path(args.candidate).resolve()
    output = Path(args.json).resolve()
    if not candidate.is_file():
        print(f"candidate does not exist: {candidate}", file=sys.stderr)
        return 2
    candidate_sha = _sha256(candidate)

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
        )
        .entrypoint([])
        .apt_install("git")
        .pip_install("torch", "numpy", "ninja")
        .run_commands(
            f"git clone --filter=blob:none {PUBLIC_REPOSITORY} /root/repo",
            f"cd /root/repo && git checkout --detach {RANKED_COMMIT}",
        )
        .add_local_file(str(candidate), "/root/repo/candidate.py", copy=True)
    )

    app = modal.App.lookup("gpumode-cholesky-exp014-submission-only", create_if_missing=True)
    payload = None
    transcript = []
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            "python",
            "-u",
            "-c",
            REMOTE_RUNNER,
            args.mode,
            candidate_sha,
            BASELINE_SHA256,
            image=image,
            gpu=args.gpu,
            app=app,
            timeout=args.timeout,
            workdir="/root/repo",
        )
        try:
            for line in sandbox.stdout:
                line = line.rstrip("\n")
                transcript.append(line)
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
        print("no RESULT_JSON emitted", file=sys.stderr)
        return 1
    payload["transcript"] = transcript
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
