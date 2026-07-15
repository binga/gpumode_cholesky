"""Device component profile of the shipped exact-shape path."""

import importlib.util
import json
import sys

import torch

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root")

from reference import check_implementation, generate_input  # noqa: E402


def _load_baseline():
    spec = importlib.util.spec_from_file_location("exp009_baseline", "/root/baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.custom_kernel


def _time(fn, data, warmup=2, iters=7):
    for _ in range(warmup):
        output = fn(data)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn(data)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    samples.sort()
    return {
        "mean_us": sum(samples) / len(samples),
        "best_us": samples[0],
        "samples_us": samples,
    }


def main():
    baseline = _load_baseline()
    data = generate_input(batch=1, n=8192, cond=2, seed=48192)
    variants = [
        ("clone_only", lambda x: x.clone()),
        ("shipped_3d", baseline),
        (
            "torch_2d_single",
            lambda x: torch.linalg.cholesky_ex(x[0], check_errors=False).L.unsqueeze(0),
        ),
        (
            "torch_3d_direct",
            lambda x: torch.linalg.cholesky_ex(x, check_errors=False).L,
        ),
    ]
    timing = {name: _time(fn, data) for name, fn in variants}

    output = baseline(data)
    torch.cuda.synchronize()
    good, message = check_implementation(data, output)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        output = baseline(data)
        torch.cuda.synchronize()
    events = []
    for event in prof.key_averages(group_by_input_shape=True):
        device_us = getattr(event, "self_device_time_total", 0.0)
        if not device_us:
            device_us = getattr(event, "self_cuda_time_total", 0.0)
        events.append(
            {
                "key": event.key,
                "calls": event.count,
                "self_device_us": float(device_us),
                "self_cpu_us": float(event.self_cpu_time_total),
                "input_shapes": event.input_shapes,
            }
        )
    events.sort(key=lambda x: x["self_device_us"], reverse=True)
    payload = {
        "mode": "exp010_profile",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "passed": bool(good),
        "message": message,
        "timing": timing,
        "top_device_events": events[:30],
    }
    print("RESULT_JSON:" + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
