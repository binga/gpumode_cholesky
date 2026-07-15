"""Runs experiment 010 inside the Modal B200 sandbox."""

from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root/exp010")

import torch

import candidates
from reference import check_implementation, generate_input


TARGET = {"batch": 1, "n": 16384, "cond": 2, "seed": 48284, "case": "dense"}
FAMILIES = [
    {"batch": 1, "n": 16384, "cond": 2, "seed": 68284, "case": "dense"},
    {"batch": 1, "n": 16384, "cond": 5, "seed": 68285, "case": "spectrum"},
    {"batch": 1, "n": 16384, "cond": 5, "seed": 68288, "case": "diagonal"},
    {"batch": 1, "n": 16384, "cond": 4, "seed": 68286, "case": "lowrank"},
    {"batch": 1, "n": 16384, "cond": 4, "seed": 68287, "case": "rowscale"},
    {"batch": 1, "n": 16384, "cond": 1, "seed": 68289, "case": "tridiagonal"},
]

_L2 = None


def _flush_l2():
    global _L2
    if _L2 is None:
        _L2 = torch.empty(int(256e6 // 4), dtype=torch.float32, device="cuda")
    _L2.zero_()


def _stats(values):
    ordered = sorted(values)
    return {
        "mean_us": sum(ordered) / len(ordered),
        "best_us": ordered[0],
        "worst_us": ordered[-1],
        "samples_us": ordered,
    }


def _tol_fraction(data, output):
    n = data.shape[-1]
    eps = torch.finfo(torch.float32).eps
    scale = torch.linalg.matrix_norm(data, ord=1, dim=(-2, -1)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        residual = torch.linalg.matrix_norm(
            output @ output.transpose(-1, -2) - data,
            ord=1,
            dim=(-2, -1),
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    allowed = 20.0 * n * eps * scale
    return (residual / allowed).amax().item()


def _event_call(fn, data):
    _flush_l2()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    output = fn(data)
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end) * 1e3


def run_paired(names):
    # The official input-byte rule yields one allocation for this 1 GiB shape.
    # Keep the list-based ownership contract anyway and retain its output.
    data_list = [generate_input(**TARGET)]
    baseline = candidates.BASELINE.custom_kernel
    results = []
    for name in names:
        fn = lambda x, n=name: candidates.candidate_call(n, x)
        row = {"variant": name, "rotating_inputs": len(data_list)}
        try:
            baseline_outputs = [baseline(data) for data in data_list]
            candidate_outputs = [fn(data) for data in data_list]
            torch.cuda.synchronize()
            baseline_checks = [
                check_implementation(data, out)
                for data, out in zip(data_list, baseline_outputs, strict=True)
            ]
            candidate_checks = [
                check_implementation(data, out)
                for data, out in zip(data_list, candidate_outputs, strict=True)
            ]
            row["baseline_passed"] = all(ok for ok, _ in baseline_checks)
            row["candidate_passed"] = all(ok for ok, _ in candidate_checks)
            row["baseline_message"] = baseline_checks[0][1]
            row["candidate_message"] = candidate_checks[0][1]
            row["candidate_tol_fraction"] = _tol_fraction(
                data_list[0], candidate_outputs[0]
            )
            # Warm both paths, then alternate order per round in the same process.
            for _ in range(2):
                baseline(data_list[0])
                fn(data_list[0])
            torch.cuda.synchronize()
            baseline_us = []
            candidate_us = []
            retained_baseline = None
            retained_candidate = None
            for round_index in range(6):
                order = (("baseline", baseline), ("candidate", fn))
                if round_index & 1:
                    order = tuple(reversed(order))
                for label, call in order:
                    output, elapsed = _event_call(call, data_list[0])
                    if label == "baseline":
                        baseline_us.append(elapsed)
                        retained_baseline = output
                    else:
                        candidate_us.append(elapsed)
                        retained_candidate = output
            # Validate the retained output objects after all timed calls.
            row["retained_baseline_passed"] = check_implementation(
                data_list[0], retained_baseline
            )[0]
            row["retained_candidate_passed"] = check_implementation(
                data_list[0], retained_candidate
            )[0]
            row["baseline"] = _stats(baseline_us)
            row["candidate"] = _stats(candidate_us)
            row["speedup"] = row["baseline"]["mean_us"] / row["candidate"]["mean_us"]
            row["strict_2x"] = row["candidate"]["mean_us"] <= 0.5 * row["baseline"]["mean_us"]
        except Exception as exc:
            row["candidate_passed"] = False
            row["error"] = repr(exc)
        row["backend"] = candidates.backend_status()
        results.append(row)
        print(json.dumps(row), flush=True)
    return {
        "mode": "paired",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "target": TARGET,
        "results": results,
    }


def run_families(name):
    fn = lambda x: candidates.candidate_call(name, x)
    rows = []
    for spec in FAMILIES:
        row = {"variant": name, "spec": spec}
        try:
            data = generate_input(**spec)
            output = fn(data)
            torch.cuda.synchronize()
            good, message = check_implementation(data, output)
            ratio = _tol_fraction(data, output) if good else None
            row.update(
                passed=bool(good),
                message=message,
                tol_fraction=ratio,
                margin_x=(1.0 / ratio if ratio and ratio > 0 else None),
            )
        except Exception as exc:
            row.update(passed=False, error=repr(exc))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {
        "mode": "families",
        "variant": name,
        "passed": all(row["passed"] for row in rows),
        "backend": candidates.backend_status(),
        "results": rows,
    }


def _profile_shipped_once(data):
    marks = {}

    def mark(label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        marks[label] = event

    _flush_l2()
    torch.cuda.synchronize()
    mark("start")
    a = data[0].clone()
    mark("clone")
    previous = torch.backends.cuda.matmul.allow_tf32
    step_bounds = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for step, k in enumerate(range(0, 16384, 2048)):
            a11 = a[k : k + 2048, k : k + 2048]
            l11 = torch.linalg.cholesky_ex(a11, check_errors=False).L
            mark(f"s{step}_diag")
            a[k : k + 2048, k : k + 2048] = l11
            mark(f"s{step}_diag_store")
            j = k + 2048
            if j >= 16384:
                step_bounds.append((step, False))
                break
            a21 = a[j:, k : k + 2048]
            l21 = torch.linalg.solve_triangular(
                l11.transpose(0, 1), a21, upper=True, left=False
            )
            mark(f"s{step}_trsm")
            a[j:, k : k + 2048] = l21
            mark(f"s{step}_panel_store")
            a[j:, j:].addmm_(
                l21, l21.transpose(0, 1), beta=1.0, alpha=-1.0
            )
            mark(f"s{step}_update")
            step_bounds.append((step, True))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    output = torch.tril(a).unsqueeze(0)
    mark("tril")
    finite = torch.isfinite(output).all()
    mark("finite")
    torch.cuda.synchronize()

    order = ["start", "clone"]
    components = {"clone": marks["start"].elapsed_time(marks["clone"]) * 1e3}
    previous_label = "clone"
    for step, has_update in step_bounds:
        for suffix, bucket in (
            ("diag", "diag"),
            ("diag_store", "diag_store"),
        ):
            label = f"s{step}_{suffix}"
            components[f"s{step}_{bucket}"] = marks[previous_label].elapsed_time(marks[label]) * 1e3
            previous_label = label
        if has_update:
            for suffix, bucket in (
                ("trsm", "trsm"),
                ("panel_store", "panel_store"),
                ("update", "update"),
            ):
                label = f"s{step}_{suffix}"
                components[f"s{step}_{bucket}"] = marks[previous_label].elapsed_time(marks[label]) * 1e3
                previous_label = label
    components["tril"] = marks[previous_label].elapsed_time(marks["tril"]) * 1e3
    components["finite"] = marks["tril"].elapsed_time(marks["finite"]) * 1e3
    components["total"] = marks["start"].elapsed_time(marks["finite"]) * 1e3
    components["finite_value"] = bool(finite.item())
    return output, components


def run_profile():
    data = generate_input(**TARGET)
    # Exclude one-time library initialization/JIT effects from component means.
    _profile_shipped_once(data)
    rows = []
    for _ in range(3):
        output, components = _profile_shipped_once(data)
        good, message = check_implementation(data, output)
        components["passed"] = bool(good)
        components["message"] = message
        rows.append(components)
        print(json.dumps(components), flush=True)
    aggregate = {}
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, float)]
    for key in numeric_keys:
        values = [row[key] for row in rows]
        aggregate[key] = sum(values) / len(values)
    categories = {
        "clone": aggregate["clone"],
        "diag": sum(value for key, value in aggregate.items() if key.endswith("_diag")),
        "diag_store": sum(value for key, value in aggregate.items() if key.endswith("_diag_store")),
        "trsm": sum(value for key, value in aggregate.items() if key.endswith("_trsm")),
        "panel_store": sum(value for key, value in aggregate.items() if key.endswith("_panel_store")),
        "update": sum(value for key, value in aggregate.items() if key.endswith("_update")),
        "tril": aggregate["tril"],
        "finite": aggregate["finite"],
        "total": aggregate["total"],
    }
    return {
        "mode": "profile",
        "target": TARGET,
        "passed": all(row["passed"] for row in rows),
        "runs": rows,
        "mean_components_us": aggregate,
        "mean_categories_us": categories,
    }


def main():
    mode = sys.argv[1]
    names = sys.argv[2].split(",")
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    if mode == "paired":
        result = run_paired(names)
    elif mode == "families" and len(names) == 1:
        result = run_families(names[0])
    elif mode == "profile":
        result = run_profile()
    else:
        raise SystemExit("usage: gpu_runner.py paired v1,v2 | families v1 | profile baseline")
    print("RESULT_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
