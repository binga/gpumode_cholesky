"""Runs INSIDE the Modal B200 sandbox. Not meant to be run locally.

Reuses the real reference harness (`generate_input`, `check_implementation`)
and the submission's `custom_kernel` to verify correctness and/or benchmark on
an actual B200 GPU. Emits a JSON line prefixed with `RESULT_JSON:` for the
driver to parse.
"""

import json
import sys
import time

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root")

import torch  # noqa: E402

from reference import check_implementation, generate_input  # noqa: E402
from submission import custom_kernel  # noqa: E402

TEST_SPECS = [
    {"batch": 16, "n": 32, "cond": 2, "seed": 53124},
    {"batch": 16, "n": 64, "cond": 2, "seed": 53125},
    {"batch": 16, "n": 128, "cond": 2, "seed": 3321},
    {"batch": 8, "n": 256, "cond": 2, "seed": 94010},
    {"batch": 4, "n": 512, "cond": 2, "seed": 32523},
    {"batch": 2, "n": 1024, "cond": 2, "seed": 4327},
    {"batch": 1, "n": 2048, "cond": 2, "seed": 224466},
    {"batch": 8, "n": 128, "cond": 5, "seed": 1200, "case": "spectrum"},
    {"batch": 8, "n": 128, "cond": 5, "seed": 1201, "case": "diagonal"},
    {"batch": 4, "n": 256, "cond": 4, "seed": 32524, "case": "lowrank"},
    {"batch": 4, "n": 512, "cond": 4, "seed": 32525, "case": "rowscale"},
    {"batch": 4, "n": 512, "cond": 1, "seed": 32526, "case": "tridiagonal"},
    {"batch": 2, "n": 1024, "cond": 4, "seed": 4330, "case": "lowrank"},
    # n=32 across all families + high batch: this is the shape our custom Triton
    # kernel handles, so exercise it hard before any ranked submission.
    {"batch": 4096, "n": 32, "cond": 2, "seed": 41032},
    {"batch": 256, "n": 32, "cond": 5, "seed": 90001, "case": "spectrum"},
    {"batch": 256, "n": 32, "cond": 5, "seed": 90002, "case": "diagonal"},
    {"batch": 256, "n": 32, "cond": 4, "seed": 90003, "case": "lowrank"},
    {"batch": 256, "n": 32, "cond": 4, "seed": 90004, "case": "rowscale"},
    {"batch": 256, "n": 32, "cond": 1, "seed": 90005, "case": "tridiagonal"},
    # exp 004: small-batch/large-n region (streamed for n<4096, loop for n>=4096)
    # across families, to prove the per-matrix paths are numerically clean.
    {"batch": 2, "n": 1024, "cond": 5, "seed": 94001, "case": "spectrum"},
    {"batch": 2, "n": 1024, "cond": 5, "seed": 94002, "case": "diagonal"},
    {"batch": 4, "n": 1024, "cond": 4, "seed": 94003, "case": "rowscale"},
    {"batch": 4, "n": 1024, "cond": 1, "seed": 94004, "case": "tridiagonal"},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 94005},
    {"batch": 2, "n": 4096, "cond": 2, "seed": 94006},
    {"batch": 2, "n": 4096, "cond": 4, "seed": 94007, "case": "lowrank"},
]

# The 15-shape ranked benchmark grid from task.yml.
BENCH_SPECS = [
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


def _spec_label(spec):
    return f"batch={spec['batch']} n={spec['n']} case={spec.get('case', 'dense')}"


def run_verify():
    failures = 0
    for spec in TEST_SPECS:
        data = generate_input(**spec)
        torch.cuda.synchronize()
        output = custom_kernel(data.clone())
        torch.cuda.synchronize()
        good, message = check_implementation(data, output)
        print(f"[{'PASS' if good else 'FAIL'}] {_spec_label(spec)}: {message}", flush=True)
        failures += 0 if good else 1
    passed = failures == 0
    print(f"\n{len(TEST_SPECS) - failures}/{len(TEST_SPECS)} specs passed", flush=True)
    return {"mode": "verify", "passed": passed, "failures": failures}


# A big buffer used to evict the L2 cache between timed iterations, mirroring
# popcorn's official timing (which clears L2 before each run). Allocated lazily
# and reused. B200 has a large L2, so use ~256MB.
_L2_FLUSH = None


def _l2_flush():
    global _L2_FLUSH
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(int(256e6 // 4), dtype=torch.float32, device="cuda")
    _L2_FLUSH.zero_()


def _time_shape(spec, warmup=None, iters=None, l2_clear=True):
    data = generate_input(**spec)
    torch.cuda.synchronize()
    out = custom_kernel(data.clone())
    torch.cuda.synchronize()
    good, message = check_implementation(data, out)
    if not good:
        return {"spec": _spec_label(spec), "passed": False, "error": message}
    # Adaptive iteration count: cheap shapes get more iters for a stable mean,
    # expensive shapes (large n) get fewer to save B200 wall-clock/cost.
    n = spec["n"]
    if iters is None:
        iters = 50 if n <= 256 else (20 if n <= 2048 else 8)
    if warmup is None:
        warmup = 5 if n <= 2048 else 2
    for _ in range(warmup):
        custom_kernel(data)
    torch.cuda.synchronize()
    durations = []
    for _ in range(iters):
        if l2_clear:
            _l2_flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        custom_kernel(data)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3)  # ms -> us
    durations.sort()
    mean_us = sum(durations) / len(durations)
    return {
        "spec": _spec_label(spec),
        "batch": spec["batch"],
        "n": spec["n"],
        "passed": True,
        "mean_us": mean_us,
        "best_us": durations[0],
    }


def run_benchmark(filter_ns=None):
    specs = BENCH_SPECS
    if filter_ns:
        specs = [s for s in BENCH_SPECS if s["n"] in filter_ns]
    results = []
    for spec in specs:
        r = _time_shape(spec)
        if r.get("passed"):
            print(f"{r['spec']:<40} mean={r['mean_us']:.1f}us best={r['best_us']:.1f}us", flush=True)
        else:
            print(f"{r['spec']:<40} FAILED: {r.get('error')}", flush=True)
        results.append(r)
    timed = [r["mean_us"] for r in results if r.get("passed")]
    geom = None
    if timed:
        log_sum = sum(__import__("math").log(t) for t in timed)
        geom = __import__("math").exp(log_sum / len(timed))
        print(f"\ngeomean(mean_us) over {len(timed)} shapes = {geom:.1f}us", flush=True)
    return {"mode": "benchmark", "geomean_us": geom, "shapes": results}


# ---------------------------------------------------------------------------
# probe mode: compare 3 ways of factorizing small-batch/large-n shapes to test
# the exp-004 hypothesis (batched cuSOLVER path is bad for few-large matrices).
# ---------------------------------------------------------------------------
def _batched_call(data):
    return torch.linalg.cholesky_ex(data, check_errors=False).L


def _loop_call(data):
    batch = data.shape[0]
    return torch.stack(
        [torch.linalg.cholesky_ex(data[i], check_errors=False).L for i in range(batch)]
    )


def _streamed_call(data):
    batch = data.shape[0]
    outs = [None] * batch
    streams = [torch.cuda.Stream() for _ in range(batch)]
    for i in range(batch):
        with torch.cuda.stream(streams[i]):
            outs[i] = torch.linalg.cholesky_ex(data[i], check_errors=False).L
    torch.cuda.synchronize()
    return torch.stack(outs)


def _time_callable(data, fn, warmup, iters, l2_clear=True):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(data)
    torch.cuda.synchronize()
    durations = []
    for _ in range(iters):
        if l2_clear:
            _l2_flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn(data)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3)
    durations.sort()
    return sum(durations) / len(durations), durations[0]


PROBE_SPECS = [
    {"batch": 2, "n": 4096, "cond": 2, "seed": 514096},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 44048},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 42024},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 511024},
    {"batch": 1, "n": 4096, "cond": 2, "seed": 48096},
]

_APPROACHES = [("batched", _batched_call), ("loop", _loop_call), ("streamed", _streamed_call)]


def run_probe(filter_ns=None):
    specs = PROBE_SPECS
    if filter_ns:
        specs = [s for s in PROBE_SPECS if s["n"] in filter_ns]
    results = []
    for spec in specs:
        data = generate_input(**spec)
        n = spec["n"]
        iters = 20 if n <= 2048 else 8
        warmup = 5 if n <= 2048 else 2
        row = {"spec": _spec_label(spec), "batch": spec["batch"], "n": n, "times_us": {}}
        for name, fn in _APPROACHES:
            out = fn(data.clone())
            torch.cuda.synchronize()
            good, message = check_implementation(data, out)
            if not good:
                row["times_us"][name] = None
                row.setdefault("errors", {})[name] = message
                continue
            mean_us, best_us = _time_callable(data, fn, warmup, iters)
            row["times_us"][name] = mean_us
        base = row["times_us"].get("batched")
        best_name = min(
            (k for k, v in row["times_us"].items() if v is not None),
            key=lambda k: row["times_us"][k],
            default=None,
        )
        row["best_approach"] = best_name
        summary = "  ".join(
            f"{k}={v:.1f}us" if v is not None else f"{k}=FAIL"
            for k, v in row["times_us"].items()
        )
        speedup = ""
        if base and best_name and row["times_us"][best_name]:
            speedup = f"  (best={best_name}, {base / row['times_us'][best_name]:.2f}x vs batched)"
        print(f"{row['spec']:<40} {summary}{speedup}", flush=True)
        results.append(row)
    return {"mode": "probe", "shapes": results}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    filter_ns = None
    if len(sys.argv) > 2 and sys.argv[2].strip():
        filter_ns = {int(x) for x in sys.argv[2].split(",") if x.strip()}
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}", flush=True)
    import submission as _sub
    print(f"custom_cuda_loaded={getattr(_sub, '_CUDA_MOD', None) is not None}", flush=True)
    _err = getattr(_sub, "_CUDA_LOAD_ERROR", None)
    if _err:
        print("CUDA_LOAD_ERROR:\n" + _err, flush=True)
    if mode == "benchmark":
        result = run_benchmark(filter_ns)
    elif mode == "probe":
        result = run_probe(filter_ns)
    else:
        result = run_verify()
    print("RESULT_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
