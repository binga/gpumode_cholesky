"""Run experiment-013 candidate/baseline probes inside a Modal B200 sandbox."""

import importlib.util
import json
import sys

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root")

import torch

from reference import check_implementation, generate_input


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate_module = _load("candidate_exp014", "/root/candidate_exp014.py")
baseline_module = _load("baseline_exp012", "/root/baseline_exp012.py")


_L2_FLUSH = None


def _flush_l2():
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(
            int(256e6 // 4), dtype=torch.float32, device="cuda"
        )
    _L2_FLUSH.zero_()


def _time_rotating(fn, inputs, warmup=1, iters=3):
    for _ in range(warmup):
        warmup_outputs = [fn(data) for data in inputs]
    torch.cuda.synchronize()
    del warmup_outputs
    durations = []
    for _ in range(iters):
        _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        timed_outputs = [fn(data) for data in inputs]
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3 / len(inputs))
        del timed_outputs
    durations.sort()
    return {
        "mean_us": sum(durations) / len(durations),
        "best_us": durations[0],
        "rotating_inputs": len(inputs),
        "warmup": warmup,
        "iters": iters,
    }


def _backend_status(module):
    names = [
        "_RECURSIVE_32768_HITS",
        "_RECURSIVE_32768_BASE_TILES",
        "_RECURSIVE_32768_FP8_UPDATES",
        "_RECURSIVE_32768_TF32_UPDATES",
        "_RECURSIVE_32768_ERROR",
        "_LEFT_32768_HITS",
        "_LEFT_32768_ERROR",
        "_LEFT_LARGE_FALLBACKS",
    ]
    return {name: getattr(module, name, None) for name in names}


def run_pair():
    specs = [
        {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
        {"batch": 1, "n": 32768, "cond": 2, "seed": 48410},
    ]
    inputs = [generate_input(**spec) for spec in specs]
    torch.cuda.synchronize()

    candidate_outputs = [
        candidate_module.custom_kernel(data.clone()) for data in inputs
    ]
    baseline_outputs = [
        baseline_module.custom_kernel(data.clone()) for data in inputs
    ]
    torch.cuda.synchronize()
    candidate_checks = [
        check_implementation(data, output)
        for data, output in zip(inputs, candidate_outputs, strict=True)
    ]
    baseline_checks = [
        check_implementation(data, output)
        for data, output in zip(inputs, baseline_outputs, strict=True)
    ]

    candidate_time = _time_rotating(
        candidate_module.custom_kernel, inputs
    )
    baseline_time = _time_rotating(
        baseline_module.custom_kernel, inputs
    )
    speedup = baseline_time["mean_us"] / candidate_time["mean_us"]
    candidate_status = _backend_status(candidate_module)
    baseline_status = _backend_status(baseline_module)
    backend_ok = (
        candidate_status["_RECURSIVE_32768_HITS"] > 0
        and candidate_status["_RECURSIVE_32768_BASE_TILES"] > 0
        and candidate_status["_RECURSIVE_32768_FP8_UPDATES"] > 0
        and candidate_status["_RECURSIVE_32768_TF32_UPDATES"] > 0
        and candidate_status["_RECURSIVE_32768_ERROR"] is None
        and candidate_status["_LEFT_LARGE_FALLBACKS"] == 0
    )
    passed = (
        backend_ok
        and all(good for good, _ in candidate_checks)
        and all(good for good, _ in baseline_checks)
    )
    return {
        "mode": "pair",
        "passed": passed,
        "candidate_checks": [
            {"passed": good, "message": message}
            for good, message in candidate_checks
        ],
        "baseline_checks": [
            {"passed": good, "message": message}
            for good, message in baseline_checks
        ],
        "candidate": candidate_time,
        "baseline": baseline_time,
        "speedup": speedup,
        "classification": (
            "WINNER" if passed and speedup >= 2.0
            else "FRONTIER" if passed and speedup > 1.0
            else "REJECTED"
        ),
        "candidate_backend": candidate_status,
        "baseline_backend": baseline_status,
    }


if __name__ == "__main__":
    result = run_pair()
    print(json.dumps(result, indent=2), flush=True)
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    raise SystemExit(0 if result["passed"] else 1)
