"""Paired B200 profiling runner for experiment 014.

Runs inside a Modal sandbox. It imports the exact ranked source and a candidate
as separate modules in one process. RESULT_JSON is the only machine interface.
Timing evidence is admissible only when the source lock, correctness checks,
backend contract, and profiler engagement proof all pass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root")

from reference import check_implementation, generate_input  # noqa: E402


ROOT = Path("/root")
CONFIG = json.loads((ROOT / "profile-config.json").read_text())
_L2_FLUSH: torch.Tensor | None = None
_STATUS_NAMES = {
    name
    for contract in CONFIG["backend_contract"].values()
    for rule in ("positive_delta", "zero_delta", "exact_delta", "none_after", "truthy_after")
    for name in (
        contract.get(rule, {}).keys()
        if rule == "exact_delta"
        else contract.get(rule, [])
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "custom_kernel", None)):
        raise RuntimeError(f"{path} does not export custom_kernel")
    return module


def _device_metadata() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": props.name,
        "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory,
    }


def _snapshot(module) -> dict[str, Any]:
    result = {}
    for name in sorted(_STATUS_NAMES):
        value = getattr(module, name, "__MISSING__")
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
        else:
            result[name] = repr(value)
    status_fn = getattr(module, "_backend_status_32768", None)
    if callable(status_fn):
        result["_backend_status_32768"] = status_fn()
    return result


def _numeric_delta(before: dict[str, Any], after: dict[str, Any], name: str):
    left, right = before.get(name), after.get(name)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return right - left
    return None


def _evaluate_contract(
    before: dict[str, Any], after: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    deltas = {}
    for name in contract.get("positive_delta", []):
        delta = _numeric_delta(before, after, name)
        deltas[name] = delta
        if delta is None or delta <= 0:
            failures.append(f"{name} delta must be positive, got {delta!r}")
    for name in contract.get("zero_delta", []):
        delta = _numeric_delta(before, after, name)
        deltas[name] = delta
        if delta != 0:
            failures.append(f"{name} delta must be zero, got {delta!r}")
    for name, expected in contract.get("exact_delta", {}).items():
        delta = _numeric_delta(before, after, name)
        deltas[name] = delta
        if delta != expected:
            failures.append(
                f"{name} delta must be {expected!r}, got {delta!r}"
            )
    for name in contract.get("none_after", []):
        if after.get(name, "__MISSING__") is not None:
            failures.append(f"{name} must be None, got {after.get(name)!r}")
    for name in contract.get("truthy_after", []):
        if not after.get(name, False):
            failures.append(f"{name} must be truthy, got {after.get(name)!r}")
    return {
        "passed": not failures,
        "failures": failures,
        "deltas": deltas,
        "before": before,
        "after": after,
    }


def _l2_flush() -> None:
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(
            int(256e6 // 4), dtype=torch.float32, device="cuda"
        )
    _L2_FLUSH.zero_()


def _official_check(data: torch.Tensor, output: torch.Tensor) -> dict[str, Any]:
    passed, message = check_implementation(data, output)
    match = re.search(r"scaled_reconstruction_residual=([0-9.eE+-]+)", message)
    scaled = float(match.group(1)) if match else None
    finite = bool(torch.isfinite(output).all().item())
    diagonal = output.diagonal(dim1=-2, dim2=-1)
    positive_diagonal = bool((diagonal > 0).all().item())
    upper_zero = bool((torch.triu(output, diagonal=1) == 0).all().item())
    return {
        "passed": bool(passed),
        "message": message,
        "finite": finite,
        "positive_diagonal": positive_diagonal,
        "exactly_lower_triangular": upper_zero,
        "scaled_reconstruction_residual": scaled,
        "tolerance_fraction": scaled / 20.0 if scaled is not None else None,
    }


def _summarize(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "best_us": ordered[0],
        "worst_us": ordered[-1],
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples_us": values,
    }


def _timed_call(fn: Callable[[torch.Tensor], torch.Tensor], data: torch.Tensor):
    _l2_flush()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn(data)
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end) * 1e3


def _paired_shape(
    baseline,
    candidate,
    specs: list[dict[str, Any]],
    warmup_rounds: int,
    paired_rounds: int,
) -> dict[str, Any]:
    data_list = [generate_input(**spec) for spec in specs]
    torch.cuda.synchronize()

    modules = {"baseline": baseline, "candidate": candidate}
    status_before = {name: _snapshot(module) for name, module in modules.items()}
    for round_index in range(warmup_rounds):
        for backend in (("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")):
            for data in data_list:
                retained = modules[backend].custom_kernel(data)
    torch.cuda.synchronize()
    # Do not carry the final warmup allocation into the timed retention set.
    # Keeping it live forces the first reversed-order backend to expand the CUDA
    # allocator while three full outputs coexist, producing an allocation-only
    # outlier despite ample warmup. Timed outputs below remain retained through
    # validation exactly as required.
    del retained

    durations = {"baseline": [], "candidate": []}
    retained_outputs: dict[tuple[str, int], torch.Tensor] = {}
    sample_rows = []
    for round_index in range(paired_rounds):
        input_index = round_index % len(data_list)
        data = data_list[input_index]
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        row = {"round": round_index, "input_index": input_index, "order": list(order)}
        for backend in order:
            output, elapsed_us = _timed_call(modules[backend].custom_kernel, data)
            retained_outputs[(backend, input_index)] = output
            durations[backend].append(elapsed_us)
            row[f"{backend}_us"] = elapsed_us
        row["speedup"] = row["baseline_us"] / row["candidate_us"]
        sample_rows.append(row)

    # Timed outputs remain strongly referenced until every rotated input/backend
    # combination has been validated. Missing combinations are run untimed.
    checks = {"baseline": [], "candidate": []}
    for backend, module in modules.items():
        for input_index, data in enumerate(data_list):
            key = (backend, input_index)
            if key not in retained_outputs:
                retained_outputs[key] = module.custom_kernel(data)
            torch.cuda.synchronize()
            checks[backend].append(_official_check(data, retained_outputs[key]))

    status_after = {name: _snapshot(module) for name, module in modules.items()}
    enforce_backend_contract = any(
        spec["batch"] == 1 and spec["n"] == 32768 for spec in specs
    )
    if enforce_backend_contract:
        contract = {
            name: _evaluate_contract(
                status_before[name],
                status_after[name],
                CONFIG["backend_contract"][name],
            )
            for name in modules
        }
    else:
        contract = {
            name: {
                "passed": True,
                "skipped": True,
                "reason": "shape is outside the changed 1x32768 dispatch",
                "before": status_before[name],
                "after": status_after[name],
            }
            for name in modules
        }
    baseline_summary = _summarize(durations["baseline"])
    candidate_summary = _summarize(durations["candidate"])
    speedup = baseline_summary["mean_us"] / candidate_summary["mean_us"]
    return {
        "rotating_inputs": len(data_list),
        "specs": specs,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "paired_samples": sample_rows,
        "mean_speedup": speedup,
        "checks": checks,
        "backend_contract": contract,
        "outputs_retained_through_validation": True,
        "passed": all(
            check["passed"]
            for backend_checks in checks.values()
            for check in backend_checks
        )
        and all(item["passed"] for item in contract.values()),
    }


def _component_profile(module, data: torch.Tensor) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    before = _snapshot(module)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        with torch.profiler.record_function("custom_kernel_1x32768"):
            output = module.custom_kernel(data)
    torch.cuda.synchronize()
    check = _official_check(data, output)
    after = _snapshot(module)
    rows = []
    for event in prof.key_averages():
        # PyTorch 2.13 generalized the CUDA-only profiler field names to
        # "device". Keep the 2.12 names first so the exact Popcorn-era wheel and
        # the current Modal wheel both emit the same experiment schema.
        self_device_time = getattr(
            event,
            "self_cuda_time_total",
            getattr(event, "self_device_time_total", 0.0),
        )
        device_time = getattr(
            event,
            "cuda_time_total",
            getattr(event, "device_time_total", 0.0),
        )
        rows.append(
            {
                "operator": event.key,
                "calls": event.count,
                "self_cuda_time_us": float(self_device_time),
                "cuda_time_us": float(device_time),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "cpu_time_us": float(event.cpu_time_total),
                "input_shapes": event.input_shapes,
            }
        )
    rows.sort(key=lambda row: row["self_cuda_time_us"], reverse=True)
    rows = rows[: CONFIG["component_profile"]["row_limit"]]
    observed = [row["operator"] for row in rows]
    required = CONFIG["component_profile"]["required_candidate_operator_any_of"]
    engagement = [token for token in required if any(token in op for op in observed)]
    contract = _evaluate_contract(
        before, after, CONFIG["backend_contract"]["candidate"]
    )
    return {
        "check": check,
        "operator_rows": rows,
        "engagement_tokens_observed": engagement,
        "backend_contract": contract,
        "passed": check["passed"] and contract["passed"] and bool(engagement),
        "timing_evidence": False,
        "note": "Profiler measurements identify components only; paired CUDA-event results decide latency.",
    }


def run_target(baseline, candidate) -> dict[str, Any]:
    target = CONFIG["target"]
    specs = [
        {
            "batch": target["batch"],
            "n": target["n"],
            "cond": target["cond"],
            "seed": seed,
            "case": target["case"],
        }
        for seed in target["rotating_seeds"]
    ]
    paired = _paired_shape(
        baseline,
        candidate,
        specs,
        target["warmup_rounds"],
        target["paired_rounds"],
    )
    speedup = paired["mean_speedup"]
    if not paired["passed"]:
        classification = "REJECTED"
    elif speedup >= target["required_speedup"]:
        classification = "WINNER"
    elif speedup > 1.0:
        classification = "FRONTIER"
    else:
        classification = "REJECTED"
    component = None
    if CONFIG["component_profile"]["enabled"]:
        component = _component_profile(candidate, generate_input(**specs[0]))
    passed = paired["passed"] and component is not None and component["passed"]
    return {
        "mode": "target",
        "paired": paired,
        "component_profile": component,
        "classification": classification if passed else "REJECTED",
        "passed": passed,
        "promotion_target_met": passed and classification == "WINNER",
    }


def run_families(baseline, candidate) -> dict[str, Any]:
    rows = []
    for spec in CONFIG["family_specs"]:
        data = generate_input(**spec)
        torch.cuda.synchronize()
        row = {"spec": spec}
        expected_fallback = spec.get("case", "dense") in set(
            CONFIG.get("family_expected_safety_fallback_cases", [])
        )
        for name, module in (("baseline", baseline), ("candidate", candidate)):
            before = _snapshot(module)
            output = module.custom_kernel(data)
            torch.cuda.synchronize()
            # Keep the output alive until both official and explicit properties
            # have been checked. Family runs are correctness-only, never timing.
            check = _official_check(data, output)
            after = _snapshot(module)
            family_contract = deepcopy(CONFIG["backend_contract"][name])
            if expected_fallback:
                family_contract["zero_delta"] = [
                    status
                    for status in family_contract.get("zero_delta", [])
                    if status != "_LEFT_LARGE_FALLBACKS"
                ]
                family_contract.setdefault("exact_delta", {})[
                    "_LEFT_LARGE_FALLBACKS"
                ] = 1
            row[name] = {
                "check": check,
                "backend_contract": _evaluate_contract(
                    before, after, family_contract
                ),
            }
        row["expected_safety_fallback"] = expected_fallback
        row["passed"] = all(
            row[name]["check"]["passed"]
            and row[name]["backend_contract"]["passed"]
            for name in ("baseline", "candidate")
        )
        rows.append(row)
        del data
        torch.cuda.empty_cache()
    return {
        "mode": "families",
        "families": rows,
        "passed": all(row["passed"] for row in rows),
        "timing_evidence": False,
    }


def _grid_rounds(n: int) -> tuple[int, int]:
    if n <= 256:
        return 5, 20
    if n <= 2048:
        return 5, 10
    if n <= 8192:
        return 4, 8
    if n <= 16384:
        return 4, 6
    return 4, 6


def _emit_result(envelope: dict[str, Any]) -> None:
    """Emit JSON without exceeding Modal's per-stdout-line size limit."""
    serialized = json.dumps(envelope, separators=(",", ":"))
    chunk_size = 8 * 1024
    if len(serialized) <= chunk_size:
        print("RESULT_JSON:" + serialized, flush=True)
        return
    print(f"RESULT_JSON_BEGIN:{len(serialized)}", flush=True)
    for offset in range(0, len(serialized), chunk_size):
        print(
            "RESULT_JSON_CHUNK:" + serialized[offset : offset + chunk_size],
            flush=True,
        )
    print("RESULT_JSON_END", flush=True)


def run_full_grid(baseline, candidate) -> dict[str, Any]:
    rows = []
    for spec in CONFIG["full_grid"]:
        warmup, rounds = _grid_rounds(spec["n"])
        paired = _paired_shape(baseline, candidate, [spec], warmup, rounds)
        ratio = paired["candidate"]["mean_us"] / paired["baseline"]["mean_us"]
        is_target = spec["batch"] == 1 and spec["n"] == 32768
        off_target_ok = is_target or ratio <= CONFIG["promotion_gates"]["off_target_mean_ratio_limit"]
        rows.append(
            {
                "spec": spec,
                "paired": paired,
                "candidate_over_baseline": ratio,
                "off_target_gate_passed": off_target_ok,
                "passed": paired["passed"] and off_target_ok,
            }
        )
        torch.cuda.empty_cache()
    baseline_geo = math.exp(
        statistics.fmean(math.log(row["paired"]["baseline"]["mean_us"]) for row in rows)
    )
    candidate_geo = math.exp(
        statistics.fmean(math.log(row["paired"]["candidate"]["mean_us"]) for row in rows)
    )
    aggregate_improved = candidate_geo < baseline_geo
    return {
        "mode": "full-grid",
        "shapes": rows,
        "baseline_geomean_us": baseline_geo,
        "candidate_geomean_us": candidate_geo,
        "aggregate_speedup": baseline_geo / candidate_geo,
        "aggregate_improved": aggregate_improved,
        "passed": all(row["passed"] for row in rows) and aggregate_improved,
    }


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "target"
    baseline_path = ROOT / "baseline.py"
    candidate_path = ROOT / "candidate.py"
    expected = CONFIG["baseline"]["source_sha256"]
    baseline_sha = _sha256(baseline_path)
    candidate_sha = _sha256(candidate_path)
    source_lock = baseline_sha == expected
    envelope = {
        "schema_version": CONFIG["schema_version"],
        "experiment": CONFIG["experiment"],
        "mode": mode,
        "baseline": {
            **CONFIG["baseline"],
            "observed_sha256": baseline_sha,
            "source_lock_passed": source_lock,
        },
        "candidate": {"source_sha256": candidate_sha},
        "environment": _device_metadata(),
        "module_load": {"baseline": False, "candidate": False},
    }
    if not source_lock:
        envelope.update({"passed": False, "error": "exact ranked baseline source lock failed"})
        _emit_result(envelope)
        return 2
    try:
        baseline = _load("baseline_141d015", baseline_path)
        envelope["module_load"]["baseline"] = True
        candidate = _load("candidate_014", candidate_path)
        envelope["module_load"]["candidate"] = True
        if mode == "target":
            result = run_target(baseline, candidate)
        elif mode == "families":
            result = run_families(baseline, candidate)
        elif mode == "full-grid":
            result = run_full_grid(baseline, candidate)
        elif mode == "all":
            target = run_target(baseline, candidate)
            families = run_families(baseline, candidate) if target["promotion_target_met"] else None
            full_grid = (
                run_full_grid(baseline, candidate)
                if families is not None and families["passed"]
                else None
            )
            result = {
                "mode": "all",
                "target": target,
                "families": families,
                "full_grid": full_grid,
                "passed": bool(
                    target["promotion_target_met"]
                    and families is not None
                    and families["passed"]
                    and full_grid is not None
                    and full_grid["passed"]
                ),
            }
        else:
            raise ValueError(f"unknown mode {mode!r}")
        envelope["result"] = result
        envelope["passed"] = bool(result["passed"])
    except Exception as exc:
        envelope.update({"passed": False, "runtime_error": repr(exc)})
    _emit_result(envelope)
    return 0 if envelope["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
