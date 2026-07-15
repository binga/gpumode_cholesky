"""Paired B200 probe for experiment 010."""

import importlib.util
import json
import math
import sys

import torch

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root")

from candidates import (  # noqa: E402
    _CUDA_LOAD_ERROR,
    _CUDA_MOD,
    batched_potrf,
    blocked_batched_diag_syrk,
    blocked_syrk_fp32,
    blocked_syrk_tf32,
    expert_potrf,
    graph_blocked_syrk_tf32,
    legacy_potrf,
    triton_lower_blocked,
)
from reference import check_implementation, generate_input  # noqa: E402


def _load_baseline():
    spec = importlib.util.spec_from_file_location("exp009_baseline", "/root/baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.custom_kernel


_L2 = None


def _flush_l2():
    global _L2
    if _L2 is None:
        _L2 = torch.empty(int(256e6 // 4), device="cuda", dtype=torch.float32)
    _L2.zero_()


def _tolerance_fraction(data, output):
    n = data.shape[-1]
    eps = torch.finfo(torch.float32).eps
    scale = torch.linalg.matrix_norm(data, ord=1, dim=(-2, -1)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reconstructed = output @ output.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    residual = torch.linalg.matrix_norm(
        reconstructed - data, ord=1, dim=(-2, -1)
    )
    allowed = 20.0 * n * eps * scale
    return float((residual / allowed).amax().item())


def _check_retained(fn, inputs):
    outputs = [fn(x) for x in inputs]
    torch.cuda.synchronize()
    checks = []
    for data, output in zip(inputs, outputs, strict=True):
        good, message = check_implementation(data, output)
        checks.append(
            {
                "passed": bool(good),
                "message": message,
                "tol_frac": _tolerance_fraction(data, output),
            }
        )
    return checks


def _time_retained(fn, inputs, warmup=2, iters=7):
    outputs = None
    for _ in range(warmup):
        outputs = [fn(x) for x in inputs]
    torch.cuda.synchronize()
    durations = []
    for _ in range(iters):
        _flush_l2()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        outputs = [fn(x) for x in inputs]
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3 / len(inputs))
    durations.sort()
    return {
        "mean_us": sum(durations) / len(durations),
        "best_us": durations[0],
        "samples_us": durations,
        "rotating_inputs": len(inputs),
    }


def main():
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(f"compiled_active={_CUDA_MOD is not None}", flush=True)
    if _CUDA_MOD is None:
        print(_CUDA_LOAD_ERROR, flush=True)
        raise SystemExit(2)

    baseline = _load_baseline()
    inputs = [
        generate_input(batch=1, n=8192, cond=2, seed=48192 + 42 * i)
        for i in range(4)
    ]
    variants = [
        ("baseline_exp009", baseline),
        ("v7_direct_batched_potrf", batched_potrf),
        (
            "v7_two_level_batched_diag_lower_syrk_nb4096",
            lambda x: blocked_batched_diag_syrk(x, 4096),
        ),
    ]

    rows = []
    for name, fn in variants:
        print(f"checking {name}", flush=True)
        try:
            checks = _check_retained(fn, inputs)
            timing = _time_retained(fn, inputs)
            row = {
                "variant": name,
                "activated": True,
                "passed": all(x["passed"] for x in checks),
                "checks": checks,
                **timing,
            }
        except Exception as exc:
            row = {
                "variant": name,
                "activated": False,
                "passed": False,
                "error": repr(exc),
            }
        rows.append(row)
        print(json.dumps(row), flush=True)

    base = rows[0].get("mean_us")
    for row in rows:
        if base and row.get("mean_us"):
            row["speedup_vs_baseline"] = base / row["mean_us"]
            row["meets_2x"] = row["mean_us"] <= 0.5 * base
    payload = {
        "mode": "exp010_stage3",
        "compiled_active": _CUDA_MOD is not None,
        "target_shape": [1, 8192, 8192],
        "paired_baseline_mean_us": base,
        "strict_target_us": 0.5 * base if base else None,
        "variants": rows,
    }
    print("RESULT_JSON:" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
