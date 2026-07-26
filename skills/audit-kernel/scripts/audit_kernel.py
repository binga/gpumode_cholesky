#!/usr/bin/env python3
"""Deterministic contract validation, execution, comparison, and gap classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


EXIT_OK = 0
EXIT_CORRECTNESS = 2
EXIT_REGRESSION = 3
EXIT_MEASUREMENT = 4
EXIT_INFRA = 5
EXIT_CONTRACT = 6

DEFAULT_THRESHOLDS = {
    "short_kernel_us": 10.0,
    "host_gpu_active_max": 0.70,
    "host_cuda_api_fraction_min": 0.20,
    "host_short_kernel_fraction_min": 0.50,
    "dependency_gpu_active_max": 0.70,
    "dependency_max_concurrency": 1,
    "dependency_min_launches": 10,
    "memory_throughput_min": 0.75,
    "memory_stall_min": 0.30,
    "compute_throughput_min": 0.75,
    "tensor_utilization_min": 0.75,
    "barrier_stall_min": 0.20,
    "occupancy_low_max": 0.30,
    "synchronization_fraction_min": 0.10,
    "memcpy_fraction_min": 0.10,
}


class AuditError(Exception):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AuditError(f"file not found: {path}", EXIT_MEASUREMENT) from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path}: {exc}", EXIT_MEASUREMENT) from exc
    if not isinstance(value, dict):
        raise AuditError(f"top-level JSON value must be an object: {path}", EXIT_MEASUREMENT)
    return value


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_hash(path: Path) -> str:
    if path.is_file():
        return file_hash(path)
    if path.is_dir():
        entries = []
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            entries.append((str(child.relative_to(path)), file_hash(child)))
        return canonical_hash(entries)
    raise AuditError(f"protected path not found: {path}", EXIT_CONTRACT)


def validate_contract(contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if not isinstance(contract.get("name"), str) or not contract.get("name"):
        errors.append("name must be a non-empty string")
    workloads = contract.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        errors.append("workloads must be a non-empty array")
    else:
        ids: set[str] = set()
        total_weight = 0.0
        for index, item in enumerate(workloads):
            if not isinstance(item, dict):
                errors.append(f"workloads[{index}] must be an object")
                continue
            workload_id = item.get("id")
            if not isinstance(workload_id, str) or not workload_id:
                errors.append(f"workloads[{index}].id must be a non-empty string")
            elif workload_id in ids:
                errors.append(f"duplicate workload id: {workload_id}")
            else:
                ids.add(workload_id)
            weight = item.get("weight")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                errors.append(f"workloads[{index}].weight must be positive")
            else:
                total_weight += float(weight)
        if workloads and total_weight <= 0:
            errors.append("workload weights must sum to a positive value")

    measurement = contract.get("measurement", {})
    max_cv = measurement.get("maximum_cv", 0.03) if isinstance(measurement, dict) else None
    if not isinstance(max_cv, (int, float)) or isinstance(max_cv, bool) or not 0 <= max_cv < 1:
        errors.append("measurement.maximum_cv must be in [0, 1)")

    acceptance = contract.get("acceptance")
    if not isinstance(acceptance, dict):
        errors.append("acceptance must be an object")
    else:
        for key in ("minimum_improvement_fraction", "maximum_case_regression_fraction"):
            value = acceptance.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                errors.append(f"acceptance.{key} must be non-negative")

    adapter = contract.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict) or not isinstance(adapter.get("argv"), list):
            errors.append("adapter.argv must be an argument array")
        elif not adapter["argv"] or not all(isinstance(v, str) and v for v in adapter["argv"]):
            errors.append("adapter.argv entries must be non-empty strings")
        timeout = adapter.get("timeout_seconds", 1800) if isinstance(adapter, dict) else None
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            errors.append("adapter.timeout_seconds must be a positive integer")

    expected = contract.get("integrity", {}).get("expected_sha256", {})
    if not isinstance(expected, dict):
        errors.append("integrity.expected_sha256 must be an object")
    else:
        for key, digest in expected.items():
            if not isinstance(key, str) or not isinstance(digest, str) or len(digest) != 64:
                errors.append("integrity.expected_sha256 values must be SHA-256 hex strings")
    return errors


def contract_or_raise(path: Path) -> dict[str, Any]:
    try:
        contract = load_json(path)
    except AuditError as exc:
        raise AuditError(str(exc), EXIT_CONTRACT) from exc
    errors = validate_contract(contract)
    if errors:
        raise AuditError("invalid contract:\n- " + "\n- ".join(errors), EXIT_CONTRACT)
    return contract


def measurement_errors(measurement: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if measurement.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    correctness = measurement.get("correctness")
    if not isinstance(correctness, dict) or not isinstance(correctness.get("passed"), bool):
        errors.append("correctness.passed must be present and boolean")
    rows = measurement.get("workloads")
    if not isinstance(rows, list):
        errors.append("workloads must be an array")
        return errors
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"workloads[{index}] must be an object")
            continue
        workload_id = row.get("id")
        latency = row.get("latency_us")
        if not isinstance(workload_id, str) or not workload_id:
            errors.append(f"workloads[{index}].id must be a non-empty string")
            continue
        if workload_id in by_id:
            errors.append(f"duplicate measurement workload id: {workload_id}")
        by_id[workload_id] = row
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency <= 0:
            errors.append(f"workload {workload_id} latency_us must be positive and finite")
        cv = row.get("cv")
        if cv is not None and (not isinstance(cv, (int, float)) or isinstance(cv, bool) or not math.isfinite(cv) or cv < 0):
            errors.append(f"workload {workload_id} cv must be non-negative and finite")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"workload {workload_id} metrics must be an object")
    required = {item["id"] for item in contract["workloads"]}
    missing = sorted(required - set(by_id))
    if missing:
        errors.append("missing required workloads: " + ", ".join(missing))
    return errors


def measurement_or_raise(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    measurement = load_json(path)
    errors = measurement_errors(measurement, contract)
    if errors:
        raise AuditError("invalid measurement:\n- " + "\n- ".join(errors), EXIT_MEASUREMENT)
    return measurement


def verify_integrity(contract: dict[str, Any], root: Path) -> dict[str, str]:
    integrity = contract.get("integrity", {})
    protected = integrity.get("protected_paths", [])
    expected = integrity.get("expected_sha256", {})
    hashes: dict[str, str] = {}
    for relative in protected:
        if not isinstance(relative, str) or not relative:
            raise AuditError("integrity.protected_paths entries must be strings", EXIT_CONTRACT)
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise AuditError(f"protected path escapes contract directory: {relative}", EXIT_CONTRACT) from exc
        hashes[relative] = path_hash(path)
        if relative in expected and hashes[relative] != expected[relative]:
            raise AuditError(f"protected path hash mismatch: {relative}", EXIT_CONTRACT)
    return hashes


def rows_by_id(measurement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in measurement["workloads"]}


def weighted_geomean(measurement: dict[str, Any], contract: dict[str, Any]) -> float:
    rows = rows_by_id(measurement)
    weights = {item["id"]: float(item["weight"]) for item in contract["workloads"]}
    total = sum(weights.values())
    return math.exp(sum(weights[key] * math.log(float(rows[key]["latency_us"])) for key in weights) / total)


def threshold_config(contract: dict[str, Any]) -> dict[str, float]:
    values = dict(DEFAULT_THRESHOLDS)
    overrides = contract.get("classification", {})
    if isinstance(overrides, dict):
        for key in values:
            value = overrides.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key] = float(value)
    return values


def present(metrics: dict[str, Any], *keys: str) -> list[str]:
    return [key for key in keys if isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)]


def classify_workload(row: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    t = threshold_config(contract)
    gaps: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(label: str, evidence: dict[str, Any], opportunity: str) -> None:
        if evidence:
            used.update(evidence)
            gaps.append({
                "label": label,
                "confidence": round(min(1.0, 0.55 + 0.15 * len(evidence)), 2),
                "evidence": evidence,
                "opportunity": opportunity,
            })

    host: dict[str, Any] = {}
    if metrics.get("gpu_active_fraction", 1.0) < t["host_gpu_active_max"]:
        host["gpu_active_fraction"] = metrics["gpu_active_fraction"]
    if metrics.get("cuda_api_time_fraction", 0.0) >= t["host_cuda_api_fraction_min"]:
        host["cuda_api_time_fraction"] = metrics["cuda_api_time_fraction"]
    if metrics.get("short_kernel_fraction", 0.0) >= t["host_short_kernel_fraction_min"]:
        host["short_kernel_fraction"] = metrics["short_kernel_fraction"]
    add("host_launch", host, "Reduce dispatch overhead with fusion, batching, graphs, or persistent scheduling.")

    dependency: dict[str, Any] = {}
    if (
        metrics.get("gpu_active_fraction", 1.0) < t["dependency_gpu_active_max"]
        and metrics.get("maximum_kernel_concurrency", math.inf) <= t["dependency_max_concurrency"]
        and metrics.get("kernel_launch_count", 0) >= t["dependency_min_launches"]
    ):
        dependency = {key: metrics[key] for key in present(metrics, "gpu_active_fraction", "maximum_kernel_concurrency", "kernel_launch_count")}
    if metrics.get("barrier_stall_fraction", 0.0) >= t["barrier_stall_min"]:
        dependency["barrier_stall_fraction"] = metrics["barrier_stall_fraction"]
    add("dependency_synchronization", dependency, "Expose independent work, reduce barriers, or overlap dependent stages.")

    memory: dict[str, Any] = {}
    if metrics.get("dram_throughput_fraction", 0.0) >= t["memory_throughput_min"]:
        memory["dram_throughput_fraction"] = metrics["dram_throughput_fraction"]
    if metrics.get("memory_dependency_stall_fraction", 0.0) >= t["memory_stall_min"]:
        memory["memory_dependency_stall_fraction"] = metrics["memory_dependency_stall_fraction"]
    add("memory", memory, "Improve locality, coalescing, tiling, reuse, layout, or fusion.")

    compute: dict[str, Any] = {}
    if metrics.get("compute_throughput_fraction", 0.0) >= t["compute_throughput_min"]:
        compute["compute_throughput_fraction"] = metrics["compute_throughput_fraction"]
    if metrics.get("tensor_core_utilization_fraction", 0.0) >= t["tensor_utilization_min"]:
        compute["tensor_core_utilization_fraction"] = metrics["tensor_core_utilization_fraction"]
    add("compute", compute, "Reduce operations or use an appropriate lower/mixed precision and Tensor Core path.")

    resources: dict[str, Any] = {}
    if metrics.get("achieved_occupancy", 1.0) < t["occupancy_low_max"]:
        resources["achieved_occupancy"] = metrics["achieved_occupancy"]
    if metrics.get("local_memory_bytes", 0) > 0:
        resources["local_memory_bytes"] = metrics["local_memory_bytes"]
    add("resource_pressure", resources, "Rebalance tile size, registers, shared memory, warps, or pipeline stages.")

    sync: dict[str, Any] = {}
    if metrics.get("synchronization_time_fraction", 0.0) >= t["synchronization_fraction_min"]:
        sync["synchronization_time_fraction"] = metrics["synchronization_time_fraction"]
    add("host_synchronization", sync, "Remove unnecessary host/device synchronization from the critical path.")

    transfer: dict[str, Any] = {}
    if metrics.get("memcpy_time_fraction", 0.0) >= t["memcpy_fraction_min"]:
        transfer["memcpy_time_fraction"] = metrics["memcpy_time_fraction"]
    add("transfer", transfer, "Keep data resident, reuse buffers, or overlap required transfers.")

    expected = {
        "gpu_active_fraction", "cuda_api_time_fraction", "short_kernel_fraction",
        "maximum_kernel_concurrency", "kernel_launch_count", "barrier_stall_fraction",
        "dram_throughput_fraction", "memory_dependency_stall_fraction",
        "compute_throughput_fraction", "tensor_core_utilization_fraction",
        "achieved_occupancy", "local_memory_bytes", "synchronization_time_fraction",
        "memcpy_time_fraction",
    }
    missing = sorted(expected - set(present(metrics, *expected)))
    gaps.sort(key=lambda item: (-item["confidence"], item["label"]))
    return {"id": row["id"], "gaps": gaps, "coverage": {"present": sorted(set(metrics) & expected), "missing": missing}}


def evaluate(contract: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not baseline["correctness"]["passed"]:
        return {"verdict": "invalid_baseline", "reason": "baseline correctness did not pass"}, EXIT_MEASUREMENT
    if not candidate["correctness"]["passed"]:
        return {"verdict": "correctness_failed", "correctness": candidate["correctness"]}, EXIT_CORRECTNESS
    acceptance = contract["acceptance"]
    if acceptance.get("require_same_environment", True) and baseline.get("environment") != candidate.get("environment"):
        return {"verdict": "invalid_environment", "reason": "baseline and candidate environments differ"}, EXIT_MEASUREMENT

    maximum_cv = float(contract.get("measurement", {}).get("maximum_cv", 0.03))
    noisy_baseline = [row["id"] for row in baseline["workloads"] if row.get("cv", 0.0) > maximum_cv]
    noisy_candidate = [row["id"] for row in candidate["workloads"] if row.get("cv", 0.0) > maximum_cv]
    if noisy_baseline or noisy_candidate:
        return {
            "verdict": "noisy_measurement",
            "noisy_baseline_workloads": noisy_baseline,
            "noisy_candidate_workloads": noisy_candidate,
            "maximum_cv": maximum_cv,
        }, EXIT_MEASUREMENT

    base_rows = rows_by_id(baseline)
    cand_rows = rows_by_id(candidate)
    details = []
    regressions = []
    max_regression = float(acceptance["maximum_case_regression_fraction"])
    for item in contract["workloads"]:
        key = item["id"]
        before = float(base_rows[key]["latency_us"])
        after = float(cand_rows[key]["latency_us"])
        change = after / before - 1.0
        row = {"id": key, "weight": item["weight"], "baseline_us": before, "candidate_us": after, "change_fraction": change}
        details.append(row)
        if change > max_regression:
            regressions.append(row)

    before_objective = weighted_geomean(baseline, contract)
    after_objective = weighted_geomean(candidate, contract)
    improvement = 1.0 - after_objective / before_objective
    minimum = float(acceptance["minimum_improvement_fraction"])
    accepted = improvement >= minimum and not regressions
    result = {
        "schema_version": 1,
        "verdict": "accepted" if accepted else "rejected",
        "correctness": candidate["correctness"],
        "objective": {
            "name": "weighted_geomean_latency_us",
            "baseline": before_objective,
            "candidate": after_objective,
            "improvement_fraction": improvement,
            "minimum_improvement_fraction": minimum,
        },
        "workloads": details,
        "regressions": regressions,
        "analysis": [classify_workload(row, contract) for row in candidate["workloads"]],
        "artifacts": candidate.get("artifacts", {}),
    }
    return result, EXIT_OK if accepted else EXIT_REGRESSION


def inspect(contract: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if not candidate["correctness"]["passed"]:
        return {"verdict": "correctness_failed", "correctness": candidate["correctness"]}, EXIT_CORRECTNESS
    return {
        "schema_version": 1,
        "verdict": "inspected",
        "correctness": candidate["correctness"],
        "objective_us": weighted_geomean(candidate, contract),
        "analysis": [classify_workload(row, contract) for row in candidate["workloads"]],
        "artifacts": candidate.get("artifacts", {}),
    }, EXIT_OK


def write_result(result: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(encoded, end="")


def expand_adapter(argv: list[str], contract_path: Path, mode: str, output: Path) -> list[str]:
    replacements = {"{contract}": str(contract_path), "{mode}": mode, "{output}": str(output)}
    expanded = []
    for arg in argv:
        for token, value in replacements.items():
            arg = arg.replace(token, value)
        if "{" in arg or "}" in arg:
            raise AuditError(f"unsupported adapter placeholder in argument: {arg}", EXIT_CONTRACT)
        expanded.append(arg)
    return expanded


def run_adapter(contract_path: Path, contract: dict[str, Any], mode: str, measurement_path: Path) -> None:
    adapter = contract.get("adapter")
    if not isinstance(adapter, dict):
        raise AuditError("contract has no adapter", EXIT_CONTRACT)
    argv = expand_adapter(adapter["argv"], contract_path, mode, measurement_path)
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["AUDIT_KERNEL_MODE"] = mode
    env["AUDIT_KERNEL_OUTPUT"] = str(measurement_path)
    try:
        completed = subprocess.run(
            argv,
            cwd=contract_path.parent,
            env=env,
            timeout=adapter.get("timeout_seconds", 1800),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AuditError(f"adapter failed to run: {exc}", EXIT_INFRA) from exc
    if completed.returncode != 0:
        raise AuditError(f"adapter exited with status {completed.returncode}", EXIT_INFRA)
    if not measurement_path.exists():
        raise AuditError(f"adapter did not write measurement: {measurement_path}", EXIT_INFRA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-contract")
    validate.add_argument("--contract", required=True, type=Path)

    validate_measurement = sub.add_parser("validate-measurement")
    validate_measurement.add_argument("--contract", required=True, type=Path)
    validate_measurement.add_argument("--measurement", required=True, type=Path)

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--contract", required=True, type=Path)
    evaluate_parser.add_argument("--baseline", required=True, type=Path)
    evaluate_parser.add_argument("--candidate", required=True, type=Path)
    evaluate_parser.add_argument("--output", type=Path)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--contract", required=True, type=Path)
    inspect_parser.add_argument("--measurement", required=True, type=Path)
    inspect_parser.add_argument("--output", type=Path)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--contract", required=True, type=Path)
    run_parser.add_argument("--mode", choices=("quick", "system", "kernel", "full"), default="quick")
    run_parser.add_argument("--measurement", required=True, type=Path)
    run_parser.add_argument("--baseline", type=Path)
    run_parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        contract_path = args.contract.resolve()
        contract = contract_or_raise(contract_path)
        integrity = verify_integrity(contract, contract_path.parent)

        if args.command == "validate-contract":
            write_result({"valid": True, "contract_sha256": canonical_hash(contract), "protected_sha256": integrity}, None)
            return EXIT_OK
        if args.command == "validate-measurement":
            measurement_or_raise(args.measurement.resolve(), contract)
            write_result({"valid": True}, None)
            return EXIT_OK
        if args.command == "run":
            measurement_path = args.measurement.resolve()
            run_adapter(contract_path, contract, args.mode, measurement_path)
            candidate = measurement_or_raise(measurement_path, contract)
            candidate.setdefault("audit", {}).update({
                "contract_sha256": canonical_hash(contract),
                "protected_sha256": integrity,
                "mode": args.mode,
            })
            measurement_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
            if args.baseline:
                baseline = measurement_or_raise(args.baseline.resolve(), contract)
                result, code = evaluate(contract, baseline, candidate)
            else:
                result, code = inspect(contract, candidate)
            write_result(result, args.output.resolve() if args.output else None)
            return code
        if args.command == "evaluate":
            baseline = measurement_or_raise(args.baseline.resolve(), contract)
            candidate = measurement_or_raise(args.candidate.resolve(), contract)
            result, code = evaluate(contract, baseline, candidate)
            write_result(result, args.output.resolve() if args.output else None)
            return code
        if args.command == "inspect":
            candidate = measurement_or_raise(args.measurement.resolve(), contract)
            result, code = inspect(contract, candidate)
            write_result(result, args.output.resolve() if args.output else None)
            return code
        raise AssertionError(args.command)
    except AuditError as exc:
        print(json.dumps({"error": str(exc), "exit_code": exc.exit_code}, indent=2), file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
