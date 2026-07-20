"""Runs INSIDE the Modal B200 sandbox. Not meant to be run locally.

Reuses the real reference harness (`generate_input`, `check_implementation`)
and the submission's `custom_kernel` to verify correctness and/or benchmark on
an actual B200 GPU. Emits a JSON line prefixed with `RESULT_JSON:` for the
driver to parse.
"""

import importlib.util
import json
import math as _math
import os
import sys
import time

# BF16x9 FP32 emulation (exp 007): cuBLAS 12.9+/CUDA 13 emulates a true FP32 GEMM
# as 9 BF16 products on Blackwell BF16 tensor cores (~3-4x native FP32, >=FP32
# accuracy). The env var must be set BEFORE the cuBLAS handle is created (first
# CUDA matmul), so we set it here at the very top, before `import torch`, when the
# runner is invoked with the `emu` token. Pass A (no token) = genuine native math
# controls; Pass B (`emu`) = the same blocked_fp32 variant runs as fused BF16x9.
_EMU = "emu" in sys.argv
if _EMU:
    # CUBLAS_EMULATE_SINGLE_PRECISION=1 is the master switch that actually engages
    # FP32 emulation through PyTorch's default cuBLAS path (the BF16X9 var alone
    # did NOT engage -- measured). CUBLAS_FP32_EMULATED_BF16X9_MATH=1 pins the
    # algorithm to BF16x9. Do NOT prefer cuBLASLt: the emulated matmul is faster on
    # the default heuristic; forcing cuBLASLt was measurably slower on the B200.
    os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    os.environ["CUBLAS_FP32_EMULATED_BF16X9_MATH"] = "1"

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
    # exp 009 exact-shape integrations, across every input family.
    {"batch": 256, "n": 128, "cond": 2, "seed": 95128},
    {"batch": 256, "n": 128, "cond": 5, "seed": 95129, "case": "spectrum"},
    {"batch": 256, "n": 128, "cond": 5, "seed": 95130, "case": "diagonal"},
    {"batch": 256, "n": 128, "cond": 4, "seed": 95131, "case": "lowrank"},
    {"batch": 256, "n": 128, "cond": 4, "seed": 95132, "case": "rowscale"},
    {"batch": 256, "n": 128, "cond": 1, "seed": 95133, "case": "tridiagonal"},
    {"batch": 16, "n": 512, "cond": 2, "seed": 95512},
    {"batch": 16, "n": 512, "cond": 5, "seed": 95513, "case": "spectrum"},
    {"batch": 16, "n": 512, "cond": 5, "seed": 95514, "case": "diagonal"},
    {"batch": 16, "n": 512, "cond": 4, "seed": 95515, "case": "lowrank"},
    {"batch": 16, "n": 512, "cond": 4, "seed": 95516, "case": "rowscale"},
    {"batch": 16, "n": 512, "cond": 1, "seed": 95517, "case": "tridiagonal"},
    {"batch": 8, "n": 2048, "cond": 5, "seed": 97049, "case": "spectrum"},
    {"batch": 8, "n": 2048, "cond": 5, "seed": 97050, "case": "diagonal"},
    {"batch": 8, "n": 2048, "cond": 4, "seed": 97051, "case": "lowrank"},
    {"batch": 8, "n": 2048, "cond": 4, "seed": 97052, "case": "rowscale"},
    {"batch": 8, "n": 2048, "cond": 1, "seed": 97053, "case": "tridiagonal"},
    # exp 006: large single matrices. 16384/32768 route to the blocked TF32
    # tensor-core path; 8192 stays on cuSOLVER. Cover all families at 16384
    # (hardest conditioning via spectrum/lowrank/rowscale) and a cheaper
    # cross-family set at 32768 (dense/lowrank/tridiagonal; spectrum's giant QR
    # is too costly at 32768, and 16384 spectrum + the size-growing tolerance
    # give confidence). 8192 across a couple families for completeness.
    {"batch": 1, "n": 8192, "cond": 2, "seed": 68192},
    {"batch": 1, "n": 8192, "cond": 5, "seed": 68193, "case": "spectrum"},
    {"batch": 1, "n": 16384, "cond": 2, "seed": 68284},
    {"batch": 1, "n": 16384, "cond": 5, "seed": 68285, "case": "spectrum"},
    {"batch": 1, "n": 16384, "cond": 4, "seed": 68286, "case": "lowrank"},
    {"batch": 1, "n": 16384, "cond": 4, "seed": 68287, "case": "rowscale"},
    {"batch": 1, "n": 16384, "cond": 5, "seed": 68288, "case": "diagonal"},
    {"batch": 1, "n": 16384, "cond": 1, "seed": 68289, "case": "tridiagonal"},
    {"batch": 1, "n": 32768, "cond": 2, "seed": 68368},
    {"batch": 1, "n": 32768, "cond": 5, "seed": 68371, "case": "spectrum"},
    {"batch": 1, "n": 32768, "cond": 4, "seed": 68369, "case": "lowrank"},
    {"batch": 1, "n": 32768, "cond": 4, "seed": 68372, "case": "rowscale"},
    {"batch": 1, "n": 32768, "cond": 5, "seed": 68373, "case": "diagonal"},
    {"batch": 1, "n": 32768, "cond": 1, "seed": 68370, "case": "tridiagonal"},
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


def run_verify(filter_ns=None):
    specs = TEST_SPECS
    if filter_ns:
        specs = [s for s in TEST_SPECS if s["n"] in filter_ns]
    failures = 0
    for spec in specs:
        print(f"... running {_spec_label(spec)}", flush=True)
        data = generate_input(**spec)
        torch.cuda.synchronize()
        output = custom_kernel(data.clone())
        torch.cuda.synchronize()
        good, message = check_implementation(data, output)
        print(f"[{'PASS' if good else 'FAIL'}] {_spec_label(spec)}: {message}", flush=True)
        failures += 0 if good else 1
    passed = failures == 0
    print(f"\n{len(specs) - failures}/{len(specs)} specs passed", flush=True)
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


def _load_exp008_baseline():
    spec = importlib.util.spec_from_file_location(
        "baseline_exp008", "/root/baseline_exp008.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.custom_kernel


def _load_exp009_baseline():
    spec = importlib.util.spec_from_file_location(
        "baseline_exp009", "/root/baseline_exp009.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.custom_kernel


def _load_exp012_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "baseline_exp012", "/root/baseline_exp012.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exp028_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "baseline_exp028", "/root/baseline_exp028.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sched_baseline_module():
    """Behavioral #883174 (empty _SPLIT32_NB_SCHEDULE scaffold) for the exp-032
    paired panel-width probe."""
    spec = importlib.util.spec_from_file_location(
        "baseline_sched", "/root/baseline_sched.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _time_callable_rotating(fn, data_list, warmup, iters):
    for _ in range(warmup):
        outputs = [fn(data) for data in data_list]
    torch.cuda.synchronize()
    durations = []
    for _ in range(iters):
        _l2_flush()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        outputs = [fn(data) for data in data_list]
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3 / len(data_list))
    durations.sort()
    return {
        "mean_us": sum(durations) / len(durations),
        "best_us": durations[0],
        "rotating_inputs": len(data_list),
    }


def run_frontierprobe():
    baseline = _load_exp008_baseline()
    target_specs = [
        {"batch": 256, "n": 128, "cond": 2, "seed": 41128},
        {"batch": 16, "n": 512, "cond": 2, "seed": 41512},
        {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
    ]
    results = []
    for shape_spec in target_specs:
        bytes_per_input = (
            shape_spec["batch"] * shape_spec["n"] * shape_spec["n"] * 4
        )
        count = max(1, min(16, int(256e6 // bytes_per_input)))
        data_list = []
        args = dict(shape_spec)
        for _ in range(count):
            data_list.append(generate_input(**args))
            args["seed"] += 42

        candidate_outputs = [custom_kernel(data.clone()) for data in data_list]
        baseline_outputs = [baseline(data.clone()) for data in data_list]
        torch.cuda.synchronize()
        candidate_checks = [
            check_implementation(data, output)
            for data, output in zip(data_list, candidate_outputs, strict=True)
        ]
        baseline_checks = [
            check_implementation(data, output)
            for data, output in zip(data_list, baseline_outputs, strict=True)
        ]
        candidate_ok = all(good for good, _ in candidate_checks)
        baseline_ok = all(good for good, _ in baseline_checks)
        candidate_message = "; ".join(
            message for good, message in candidate_checks if not good
        ) or candidate_checks[0][1]
        baseline_message = "; ".join(
            message for good, message in baseline_checks if not good
        ) or baseline_checks[0][1]
        candidate_time = _time_callable_rotating(
            custom_kernel, data_list, warmup=3, iters=20
        )
        baseline_time = _time_callable_rotating(
            baseline, data_list, warmup=3, iters=20
        )
        speedup = baseline_time["mean_us"] / candidate_time["mean_us"]
        row = {
            "spec": _spec_label(shape_spec),
            "candidate_passed": candidate_ok,
            "candidate_message": candidate_message,
            "baseline_passed": baseline_ok,
            "baseline_message": baseline_message,
            "candidate": candidate_time,
            "baseline": baseline_time,
            "speedup": speedup,
        }
        results.append(row)
        print(
            f"{row['spec']} candidate={candidate_time['mean_us']:.3f}us "
            f"baseline={baseline_time['mean_us']:.3f}us "
            f"speedup={speedup:.4f}x",
            flush=True,
        )
    passed = all(
        row["candidate_passed"]
        and row["baseline_passed"]
        and row["speedup"] > 1.0
        for row in results
    )
    return {"mode": "frontierprobe", "passed": passed, "shapes": results}


def run_largefrontierprobe():
    import submission as candidate_module

    baseline = _load_exp009_baseline()
    target_specs = [
        {"batch": 1, "n": 16384, "cond": 2, "seed": 48284},
        {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
    ]
    results = []
    for shape_spec in target_specs:
        data_list = []
        args = dict(shape_spec)
        for _ in range(2):
            data_list.append(generate_input(**args))
            args["seed"] += 42

        candidate_outputs = [custom_kernel(data.clone()) for data in data_list]
        baseline_outputs = [baseline(data.clone()) for data in data_list]
        torch.cuda.synchronize()
        candidate_checks = [
            check_implementation(data, output)
            for data, output in zip(data_list, candidate_outputs, strict=True)
        ]
        baseline_checks = [
            check_implementation(data, output)
            for data, output in zip(data_list, baseline_outputs, strict=True)
        ]
        candidate_ok = all(good for good, _ in candidate_checks)
        baseline_ok = all(good for good, _ in baseline_checks)
        candidate_message = "; ".join(
            message for good, message in candidate_checks if not good
        ) or candidate_checks[0][1]
        baseline_message = "; ".join(
            message for good, message in baseline_checks if not good
        ) or baseline_checks[0][1]
        iters = 6 if shape_spec["n"] == 16384 else 4
        candidate_time = _time_callable_rotating(
            custom_kernel, data_list, warmup=1, iters=iters
        )
        baseline_time = _time_callable_rotating(
            baseline, data_list, warmup=1, iters=iters
        )
        speedup = baseline_time["mean_us"] / candidate_time["mean_us"]
        row = {
            "spec": _spec_label(shape_spec),
            "candidate_passed": candidate_ok,
            "candidate_message": candidate_message,
            "baseline_passed": baseline_ok,
            "baseline_message": baseline_message,
            "candidate": candidate_time,
            "baseline": baseline_time,
            "speedup": speedup,
        }
        results.append(row)
        print(
            f"{row['spec']} candidate={candidate_time['mean_us']:.3f}us "
            f"baseline={baseline_time['mean_us']:.3f}us "
            f"speedup={speedup:.4f}x",
            flush=True,
        )

    backend_status = {
        "left_16384_hits": getattr(candidate_module, "_LEFT_16384_HITS", 0),
        "left_32768_hits": getattr(candidate_module, "_LEFT_32768_HITS", 0),
        "left_32768_error": getattr(candidate_module, "_LEFT_32768_ERROR", None),
        "fallbacks": getattr(candidate_module, "_LEFT_LARGE_FALLBACKS", 0),
    }
    backend_ok = (
        backend_status["left_16384_hits"] > 0
        and backend_status["left_32768_hits"] > 0
        and backend_status["left_32768_error"] is None
        and backend_status["fallbacks"] == 0
    )
    passed = backend_ok and all(
        row["candidate_passed"]
        and row["baseline_passed"]
        and row["speedup"] > 1.0
        for row in results
    )
    return {
        "mode": "largefrontierprobe",
        "passed": passed,
        "backend_status": backend_status,
        "shapes": results,
    }


# ---------------------------------------------------------------------------
# nocusolverprobe (experiment 013): paired same-process comparison of the
# cuSOLVER-free 1x32768 left-looking path against the exp-012 ranked path
# (baseline_exp012.py). Develop cheaply on 8192/16384 proxies (both are generic
# in n), then confirm on the 50ms/iter 32768 probe. Also micro-benchmarks the
# cuSOLVER-free diagonal potrf against cuSOLVER on a single 4096 block.
# ---------------------------------------------------------------------------
def _recon_margin(data, output):
    """Reconstruction tolerance fraction (residual/allowed), TF32 off."""
    n = data.shape[-1]
    eps = torch.finfo(torch.float32).eps
    scale = torch.linalg.matrix_norm(data, ord=1, dim=(-2, -1)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        recon = output @ output.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    residual = torch.linalg.matrix_norm(recon - data, ord=1, dim=(-2, -1))
    allowed = 20.0 * max(n, 1) * eps * scale
    return (residual / allowed).amax().item()


def _diag_potrf_microbench():
    """Compare cuSOLVER potrf vs the cuSOLVER-free Triton blocked potrf on a
    single 4096x4096 dense SPD block (cheap ~1ms/iter)."""
    import submission as cand
    block = generate_input(batch=1, n=4096, cond=2, seed=71337)[0].contiguous()

    def _cusolver(x):
        return torch.linalg.cholesky_ex(x, check_errors=False).L

    rows = []

    def _run(name, fn):
        out = fn(block.clone())
        torch.cuda.synchronize()
        margin = _recon_margin(block, out)
        for _ in range(2):
            fn(block.clone())
        torch.cuda.synchronize()
        durs = []
        for _ in range(6):
            _l2_flush()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            fn(block.clone())
            e.record()
            torch.cuda.synchronize()
            durs.append(s.elapsed_time(e) * 1e3)
        durs.sort()
        mean_us = sum(durs) / len(durs)
        rows.append({"name": name, "mean_us": mean_us, "tol_frac": margin})
        print(
            f"  potrf4096 {name:<18} mean={mean_us:>9.1f}us "
            f"tol_frac={margin:.3e} margin={1.0/margin if margin>0 else float('inf'):.1f}x",
            flush=True,
        )

    _run("cusolver", _cusolver)
    for bk in (64,):
        cand._DIAG_POTRF_BK = bk
        cand._DIAG_POTRF_TILE = bk if bk <= 128 else 128

        def _tri(x, _bk=bk):
            return cand._triton_blocked_potrf(x)

        try:
            _run(f"triton_bk{bk}", _tri)
        except Exception as exc:  # pragma: no cover
            print(f"  potrf4096 triton_bk{bk:<11} FAILED: {exc!r}", flush=True)
    for use_tf32 in (True, False):
        tag = "tf32" if use_tf32 else "fp32"

        def _cub(x, _t=use_tf32):
            return cand._blocked_cublas_potrf(x, 32, _t)

        try:
            _run(f"cublas32_{tag}", _cub)
        except Exception as exc:  # pragma: no cover
            print(f"  potrf4096 cublas32_{tag:<9} FAILED: {exc!r}", flush=True)
    return rows


def run_nocusolverprobe(filter_ns=None):
    import submission as cand

    baseline_mod = _load_exp012_baseline_module()
    baseline_fn = baseline_mod._left_looking_cholesky_32768
    candidate_fn = cand._left_looking_cholesky_32768

    ns = sorted(filter_ns) if filter_ns else [8192, 16384]

    print("diagonal potrf micro-benchmark (single 4096 block):", flush=True)
    micro = _diag_potrf_microbench()

    # Choose the fastest passing cuSOLVER-free bk for the full-path comparison.
    tri_rows = [r for r in micro if r["name"].startswith("triton") and r["tol_frac"] < 1.0]
    if tri_rows:
        best = min(tri_rows, key=lambda r: r["mean_us"])
        best_bk = int(best["name"].replace("triton_bk", ""))
    else:
        best_bk = 128
    # Two candidate configurations, spanning the speed/accuracy tradeoff of a
    # cuSOLVER-free diagonal: (a) the fastest free diagonal (Triton bk64) with an
    # FP8 panel solve, (b) the most accurate free diagonal (cuBLAS bk32 FP32) with
    # a TF32 panel solve. Both are compared against the exp-012 cuSOLVER path.
    configs = [
        {"tag": "fast_triton64_fp8panel", "diag": "triton", "bk": 64, "panel_fp8": True},
        {"tag": "accurate_cublas32fp32_tf32panel", "diag": "cublas32_fp32", "bk": 64, "panel_fp8": False},
    ]

    results = []
    for cfg in configs:
        cand._DIAG_METHOD = cfg["diag"]
        cand._DIAG_POTRF_BK = cfg["bk"]
        cand._DIAG_POTRF_TILE = cfg["bk"] if cfg["bk"] <= 128 else 128
        cand._PANEL_SOLVE_FP8 = cfg["panel_fp8"]
        print(f"\nconfig={cfg['tag']} (diag={cfg['diag']}, panel_fp8={cfg['panel_fp8']})", flush=True)
        for n in ns:
            data_list = []
            seed = 48368
            for _ in range(2):
                data_list.append(generate_input(batch=1, n=n, cond=2, seed=seed)[0].contiguous())
                seed += 42

            cand_out = [candidate_fn(m) for m in data_list]
            base_out = [baseline_fn(m) for m in data_list]
            torch.cuda.synchronize()
            cand_checks = [
                check_implementation(m.unsqueeze(0), o.unsqueeze(0))
                for m, o in zip(data_list, cand_out, strict=True)
            ]
            base_checks = [
                check_implementation(m.unsqueeze(0), o.unsqueeze(0))
                for m, o in zip(data_list, base_out, strict=True)
            ]
            cand_ok = all(g for g, _ in cand_checks)
            base_ok = all(g for g, _ in base_checks)
            cand_margin = max(_recon_margin(m.unsqueeze(0), o.unsqueeze(0)) for m, o in zip(data_list, cand_out))
            base_margin = max(_recon_margin(m.unsqueeze(0), o.unsqueeze(0)) for m, o in zip(data_list, base_out))

            iters = 4 if n < 32768 else 3
            cand_t = _time_callable_rotating(candidate_fn, data_list, warmup=1, iters=iters)
            base_t = _time_callable_rotating(baseline_fn, data_list, warmup=1, iters=iters)
            speedup = base_t["mean_us"] / cand_t["mean_us"]
            row = {
                "config": cfg["tag"],
                "n": n,
                "candidate_passed": cand_ok,
                "baseline_passed": base_ok,
                "candidate_tol_frac": cand_margin,
                "baseline_tol_frac": base_margin,
                "candidate": cand_t,
                "baseline": base_t,
                "speedup": speedup,
                "candidate_message": cand_checks[0][1],
                "baseline_message": base_checks[0][1],
            }
            results.append(row)
            print(
                f"  n={n:<6} candidate={cand_t['mean_us']:.1f}us baseline={base_t['mean_us']:.1f}us "
                f"speedup={speedup:.4f}x  cand_margin={1.0/cand_margin if cand_margin>0 else float('inf'):.1f}x "
                f"(tol_frac={cand_margin:.3e}) cand_ok={cand_ok} base_ok={base_ok}",
                flush=True,
            )
    cand._DIAG_METHOD = "triton"

    # Backend proof: route a real dense (1, n, n) input through custom_kernel so
    # the dispatch + safety-net counters are exercised. Only at 32768 (the ranked
    # shape) unless a smaller n was requested.
    dispatch_n = 32768 if 32768 in ns else max(ns)
    fallbacks_before = getattr(cand, "_LEFT_LARGE_FALLBACKS", 0)
    hits_before = getattr(cand, "_NOCUSOLVER_32768_HITS", 0)
    if dispatch_n == 32768:
        dense = generate_input(batch=1, n=32768, cond=2, seed=48368)
        disp_out = cand.custom_kernel(dense)
        torch.cuda.synchronize()
        disp_ok, disp_msg = check_implementation(dense, disp_out)
        owned = disp_out.data_ptr() != dense.data_ptr()
    else:
        disp_ok, disp_msg, owned = None, "dispatch-check skipped (no 32768)", None

    backend_status = {
        "nocusolver_32768_hits": getattr(cand, "_NOCUSOLVER_32768_HITS", 0),
        "nocusolver_potrf_calls": getattr(cand, "_NOCUSOLVER_POTRF_CALLS", 0),
        "left_32768_error": getattr(cand, "_LEFT_32768_ERROR", None),
        "fallbacks": getattr(cand, "_LEFT_LARGE_FALLBACKS", 0),
        "fallbacks_during_dispatch": getattr(cand, "_LEFT_LARGE_FALLBACKS", 0) - fallbacks_before,
        "hits_during_dispatch": getattr(cand, "_NOCUSOLVER_32768_HITS", 0) - hits_before,
        "dispatch_n": dispatch_n,
        "dispatch_passed": disp_ok,
        "dispatch_message": disp_msg,
        "dispatch_owned_output": owned,
        "best_bk": best_bk,
    }
    print(f"backend_status={json.dumps(backend_status)}", flush=True)

    backend_ok = (
        backend_status["nocusolver_32768_hits"] > 0
        and backend_status["fallbacks"] == 0
        and backend_status["left_32768_error"] is None
    )
    passed = backend_ok and all(
        r["candidate_passed"] and r["baseline_passed"] and r["speedup"] > 1.0
        for r in results
    )
    return {
        "mode": "nocusolverprobe",
        "passed": passed,
        "backend_status": backend_status,
        "diag_microbench": micro,
        "shapes": results,
    }


# ---------------------------------------------------------------------------
# dualprobe (experiment 028): exact ranked baseline versus the cuSOLVER-free
# persistent two-matrix candidate. Retains rotating outputs, checks the official
# reconstruction gate, and rejects timing whenever candidate fallback metadata
# is nonzero.
# ---------------------------------------------------------------------------
def run_dualprobe(filter_ns=None):
    import submission as cand

    baseline = _load_exp028_baseline_module().custom_kernel
    specs = [
        {"batch": 2, "n": 2048, "cond": 2, "seed": 44048},
        {"batch": 2, "n": 4096, "cond": 2, "seed": 514096},
    ]
    if filter_ns:
        specs = [spec for spec in specs if spec["n"] in filter_ns]

    results = []
    for shape_spec in specs:
        bytes_per_input = shape_spec["batch"] * shape_spec["n"] ** 2 * 4
        count = max(1, min(8, int(256e6 // bytes_per_input)))
        data_list = []
        args = dict(shape_spec)
        for _ in range(count):
            data_list.append(generate_input(**args))
            args["seed"] += 42

        hits_before = getattr(cand, "_DUAL_PERSIST_HITS", 0)
        fallbacks_before = getattr(cand, "_DUAL_PERSIST_FALLBACKS", 0)
        candidate_outputs = [cand.custom_kernel(data.clone()) for data in data_list]
        torch.cuda.synchronize()
        hits_after_correctness = getattr(cand, "_DUAL_PERSIST_HITS", 0)
        fallbacks_after_correctness = getattr(cand, "_DUAL_PERSIST_FALLBACKS", 0)
        error_after_correctness = getattr(cand, "_DUAL_PERSIST_ERROR", None)

        baseline_outputs = [baseline(data.clone()) for data in data_list]
        torch.cuda.synchronize()
        candidate_checks = [
            check_implementation(data, output)
            for data, output in zip(data_list, candidate_outputs, strict=True)
        ]
        baseline_checks = [
            check_implementation(data, output)
            for data, output in zip(data_list, baseline_outputs, strict=True)
        ]
        candidate_ok = all(good for good, _ in candidate_checks)
        baseline_ok = all(good for good, _ in baseline_checks)
        backend_ok = (
            hits_after_correctness - hits_before == count
            and fallbacks_after_correctness == fallbacks_before
            and error_after_correctness is None
        )

        candidate_time = None
        baseline_time = _time_callable_rotating(
            baseline, data_list, warmup=2, iters=12 if shape_spec["n"] == 2048 else 8
        )
        speedup = None
        if backend_ok:
            candidate_time = _time_callable_rotating(
                cand.custom_kernel,
                data_list,
                warmup=2,
                iters=12 if shape_spec["n"] == 2048 else 8,
            )
            speedup = baseline_time["mean_us"] / candidate_time["mean_us"]

        row = {
            "spec": _spec_label(shape_spec),
            "candidate_passed": candidate_ok,
            "candidate_checks": [message for _, message in candidate_checks],
            "baseline_passed": baseline_ok,
            "baseline_checks": [message for _, message in baseline_checks],
            "backend_ok": backend_ok,
            "backend": {
                "hits_during_correctness": hits_after_correctness - hits_before,
                "fallbacks_during_correctness": (
                    fallbacks_after_correctness - fallbacks_before
                ),
                "error": error_after_correctness,
            },
            "candidate": candidate_time,
            "baseline": baseline_time,
            "speedup": speedup,
        }
        results.append(row)
        print(
            f"{row['spec']} backend_ok={backend_ok} candidate_ok={candidate_ok} "
            f"candidate={(candidate_time or {}).get('mean_us')}us "
            f"baseline={baseline_time['mean_us']:.3f}us speedup={speedup}",
            flush=True,
        )

    final_backend = {
        "hits": getattr(cand, "_DUAL_PERSIST_HITS", 0),
        "fallbacks": getattr(cand, "_DUAL_PERSIST_FALLBACKS", 0),
        "error": getattr(cand, "_DUAL_PERSIST_ERROR", None),
    }
    valid = all(
        row["backend_ok"]
        and row["candidate_passed"]
        and row["baseline_passed"]
        and row["speedup"] is not None
        for row in results
    )
    aggregate_speedup = None
    if valid and results:
        aggregate_speedup = __import__("math").prod(
            row["speedup"] for row in results
        ) ** (1.0 / len(results))
    passed = valid and aggregate_speedup is not None and aggregate_speedup > 1.0
    return {
        "mode": "dualprobe",
        "passed": passed,
        "aggregate_speedup": aggregate_speedup,
        "backend_status": final_backend,
        "shapes": results,
    }


# ---------------------------------------------------------------------------
# schedprobe mode (exp 032, lever L2): paired same-process comparison of the
# behavioral #883174 baseline (uniform _SPLIT32_NB=128) against a candidate that
# enrolls per-shape panel-width schedules in `_SPLIT32_NB_SCHEDULE`. Times each
# split32 shape with the official-style harness (rotating inputs to 256MiB, L2
# clear, retained outputs, correctness re-check), brackets baseline drift with a
# second baseline pass, and runs a five-family correctness sweep for every
# enrolled shape. Only enrolled shapes count toward the aggregate; unenrolled
# split32 shapes are a neutrality sanity check (must stay ~1.00x).
# ---------------------------------------------------------------------------
SCHED_SPECS = [
    {"batch": 256, "n": 128, "cond": 2, "seed": 41128},
    {"batch": 64, "n": 256, "cond": 2, "seed": 41256},
    {"batch": 16, "n": 512, "cond": 2, "seed": 41512},
    {"batch": 640, "n": 512, "cond": 2, "seed": 510512},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 42024},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 511024},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
]

# (family, cond) matching the reference/TEST_SPECS conventions.
SCHED_FAMILIES = [
    ("spectrum", 5),
    ("diagonal", 5),
    ("lowrank", 4),
    ("rowscale", 4),
    ("tridiagonal", 1),
]


# ---------------------------------------------------------------------------
# dotprobe mode (exp 033, lever L4 kill-gate): does a manually-emulated
# fp16x3 (three-fp16-MMA fp32) dot beat Triton's native tf32x3 on the B200 at
# the panel tile shapes the split32 chain actually uses? fp16 and tf32 share a
# 10-bit mantissa, so fp16x3 ~= tf32x3 in accuracy; the only question is raw
# tensor-core throughput. If fp16x3 does NOT beat tf32x3 here, L4 is dead before
# any kernel rewrite. Saturates the SMs with BIGM/128 independent output tiles in
# a single launch, so this measures tensor-core throughput, not launch latency.
# ---------------------------------------------------------------------------
def run_microprobe(filter_ns=None):
    """Experiment 036: is `_micro_potrf_gj32` fixed-cost bound (register
    spill / occupancy) rather than serial-step bound?

    Exp 029 concluded "step latency is the floor" but never varied num_warps.
    The shapediag profile implies cost ~= 12.7us fixed + 0.10us/step, so a
    large per-launch fixed term dominates. This compiles the SHIPPED kernel at
    several num_warps, reports register/spill/shared stats from the compiled
    handle, and times a realistic serial chain of 32 dependent launches at
    batch=4 -- the exact structure the 4x1024 path runs.
    """
    import triton  # noqa: F401

    import submission as sub

    mod = sub.custom_kernel.__globals__
    micro = None
    for name in ("_micro_potrf_gj32",):
        micro = mod.get(name)
    result = {
        "mode": "microprobe",
        "device": torch.cuda.get_device_name(0),
        "found_kernel": micro is not None,
        "variants": [],
    }
    if micro is None:
        # The kernel lives in a closure, not module globals; reach it through
        # the split32 launcher's own globals.
        result["note"] = "kernel not in module globals"
        result["passed"] = False
        return result

    batch, n = 4, 1024
    for warps in (1, 2, 4, 8):
        try:
            work = torch.zeros(batch, n, n, device="cuda", dtype=torch.float32)
            eye = torch.eye(n, device="cuda", dtype=torch.float32)
            work.copy_((eye * 4.0).expand(batch, n, n).contiguous())
            dinv = torch.empty(batch, 32, 32, device="cuda", dtype=torch.float32)

            def one(k):
                return micro[(batch,)](
                    work, dinv, work, n=n, k=k,
                    FIRST=False, RECIPROCAL_SOLVE=True, num_warps=warps,
                )

            handle = one(0)
            torch.cuda.synchronize()
            stats = {
                "num_warps": warps,
                "n_regs": getattr(handle, "n_regs", None),
                "n_spills": getattr(handle, "n_spills", None),
                "shared": getattr(handle, "metadata", None)
                and getattr(handle.metadata, "shared", None),
            }
            # realistic serial chain: 32 dependent launches
            for _ in range(3):
                for k in range(0, n, 32):
                    one(k)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            iters = 10
            start.record()
            for _ in range(iters):
                for k in range(0, n, 32):
                    one(k)
            end.record()
            torch.cuda.synchronize()
            chain_us = start.elapsed_time(end) * 1000.0 / iters
            stats["chain32_us"] = round(chain_us, 1)
            stats["per_call_us"] = round(chain_us / 32.0, 3)
            result["variants"].append(stats)
            print(
                f"num_warps={warps}: regs={stats['n_regs']} spills={stats['n_spills']} "
                f"chain32={stats['chain32_us']}us per_call={stats['per_call_us']}us",
                flush=True,
            )
            del work, dinv, eye
            torch.cuda.empty_cache()
        except Exception as exc:
            result["variants"].append({"num_warps": warps, "error": repr(exc)})
            print(f"num_warps={warps}: ERROR {exc!r}", flush=True)
    result["passed"] = any("chain32_us" in v for v in result["variants"])
    return result


def run_asmprobe(filter_ns=None):
    """Experiment 037 GATE: does the `tl.where` cascade in `_micro_potrf_gj32`
    survive compilation, or does the compiler fold it?

    The exp-036 hypothesis was that Triton rebuilds the full 32x32 tile through
    a 5-deep nested `tl.where` every iteration (~10k predicated ops/iteration),
    setting the ~12.7us fixed floor. That is only true if the selects are
    RUNTIME selects. `for it in range(0, 8)` has constant bounds, so if Triton
    unrolls it, `p0 = 4*it` becomes a compile-time constant, every
    `c == p0` predicate is statically known per element, and the whole cascade
    folds into register renaming at zero cost.

    This dumps PTX + SASS for the shipped kernel and counts the instruction
    mix. If selp/predicated-select counts are low relative to FFMA, the
    hypothesis is REFUTED and a CUDA rewrite targeting the cascade is
    pointless.
    """
    import collections
    import re
    import subprocess
    import tempfile

    import submission as sub

    micro = sub.custom_kernel.__globals__.get("_micro_potrf_gj32")
    result = {
        "mode": "asmprobe",
        "device": torch.cuda.get_device_name(0),
        "found_kernel": micro is not None,
    }
    if micro is None:
        result["passed"] = False
        result["note"] = "kernel not reachable from module globals"
        return result

    batch, n = 4, 1024
    work = torch.zeros(batch, n, n, device="cuda", dtype=torch.float32)
    eye = torch.eye(n, device="cuda", dtype=torch.float32)
    work.copy_((eye * 4.0).expand(batch, n, n).contiguous())
    dinv = torch.empty(batch, 32, 32, device="cuda", dtype=torch.float32)
    handle = micro[(batch,)](
        work, dinv, work, n=n, k=0,
        FIRST=False, RECIPROCAL_SOLVE=True, num_warps=1,
    )
    torch.cuda.synchronize()

    asm = getattr(handle, "asm", {}) or {}
    result["asm_keys"] = sorted(asm.keys())
    result["n_regs"] = getattr(handle, "n_regs", None)
    result["n_spills"] = getattr(handle, "n_spills", None)

    ptx = asm.get("ptx", "")
    result["ptx_lines"] = len(ptx.splitlines())
    ptx_ops = collections.Counter()
    labels = set()
    branch_targets = []
    for line in ptx.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", ".", "{", "}")):
            continue
        if line.endswith(":"):
            labels.add(line[:-1])
            continue
        # Predicated instructions start with @%p / @!%p -- strip the guard so
        # they are COUNTED, not skipped (skipping them hides every loop
        # backedge, which is exactly the structure question here).
        if line.startswith("@"):
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            line = parts[1]
        tok = line.split()
        if not tok:
            continue
        op = tok[0].split(".")[0]
        ptx_ops[op] += 1
        if op == "bra":
            branch_targets.append(line)
    result["ptx_labels"] = len(labels)
    result["ptx_branch_lines"] = branch_targets[:8]
    result["ptx_total_insts"] = sum(ptx_ops.values())
    result["ptx_top"] = ptx_ops.most_common(18)
    # selp = predicated select; setp = predicate compute. These are what a
    # surviving runtime `tl.where` cascade would emit.
    result["ptx_selp"] = ptx_ops.get("selp", 0)
    result["ptx_setp"] = ptx_ops.get("setp", 0)
    result["ptx_fma"] = ptx_ops.get("fma", 0)
    result["ptx_branches"] = ptx_ops.get("bra", 0)
    # Cross-lane / shared-memory traffic: the alternative bottleneck. With one
    # warp there is no occupancy to hide any of this latency.
    result["ptx_shfl"] = ptx_ops.get("shfl", 0)
    result["ptx_bar"] = ptx_ops.get("bar", 0)
    result["ptx_rsqrt"] = ptx_ops.get("rsqrt", 0)
    result["ptx_ldmatrix"] = ptx_ops.get("ldmatrix", 0)
    result["ptx_stmatrix"] = ptx_ops.get("stmatrix", 0)
    # 8 rank-4 steps x 4 rsqrt = 32 expected if the loop is fully unrolled;
    # 4 means the PTX holds ONE loop body and the trip count is dynamic.
    result["loop_unrolled"] = result["ptx_rsqrt"] >= 32

    # SASS via nvdisasm on the cubin (the devel image has it).
    sass_ops = collections.Counter()
    sass_total = 0
    cubin = asm.get("cubin")
    if cubin:
        try:
            with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as fh:
                fh.write(cubin)
                path = fh.name
            out = subprocess.run(
                ["nvdisasm", "-c", path], capture_output=True, text=True, timeout=120
            )
            sass = out.stdout
            result["sass_lines"] = len(sass.splitlines())
            for line in sass.splitlines():
                m = re.search(r"\*/\s+(@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)", line)
                if m:
                    sass_ops[m.group(2).split(".")[0]] += 1
                    sass_total += 1
            result["sass_total_insts"] = sass_total
            result["sass_top"] = sass_ops.most_common(18)
            result["sass_sel"] = sass_ops.get("SEL", 0) + sass_ops.get("FSEL", 0)
            result["sass_isetp"] = sass_ops.get("ISETP", 0)
            result["sass_ffma"] = sass_ops.get("FFMA", 0)
            result["sass_branches"] = sass_ops.get("BRA", 0) + sass_ops.get("BSSY", 0)
            result["sass_shfl"] = sass_ops.get("SHFL", 0)
            result["sass_bar"] = sass_ops.get("BAR", 0) + sass_ops.get("BARRIER", 0)
            result["sass_lds"] = sass_ops.get("LDS", 0) + sass_ops.get("STS", 0)
            result["sass_mufu"] = sass_ops.get("MUFU", 0)
            # Latency-weighted critical-path proxy. Rough Blackwell issue-to-
            # use latencies; only the RATIO between classes matters here.
            result["latency_proxy_cycles"] = (
                sass_ops.get("SHFL", 0) * 30
                + (sass_ops.get("LDS", 0) + sass_ops.get("STS", 0)) * 30
                + (sass_ops.get("BAR", 0) + sass_ops.get("BARRIER", 0)) * 25
                + sass_ops.get("MUFU", 0) * 15
                + (sass_ops.get("FFMA", 0) + sass_ops.get("FADD", 0)
                   + sass_ops.get("FSEL", 0)) * 4
            )
        except Exception as exc:
            result["sass_error"] = repr(exc)
    else:
        result["sass_error"] = "no cubin in asm dict"

    # Verdict. 8 iterations; a surviving cascade means thousands of selects.
    sel = result.get("sass_sel", 0) or result["ptx_selp"]
    ffma = result.get("sass_ffma", 0) or result["ptx_fma"]
    result["select_to_fma_ratio"] = round(sel / max(ffma, 1), 4)
    result["cascade_survives"] = bool(sel > 500)
    print(f"regs={result['n_regs']} spills={result['n_spills']}", flush=True)
    print(
        f"PTX: {result['ptx_total_insts']} insts  selp={result['ptx_selp']} "
        f"setp={result['ptx_setp']} fma={result['ptx_fma']} bra={result['ptx_branches']}",
        flush=True,
    )
    print(f"PTX top: {result['ptx_top']}", flush=True)
    if sass_total:
        print(
            f"SASS: {sass_total} insts  SEL={result['sass_sel']} "
            f"ISETP={result['sass_isetp']} FFMA={result['sass_ffma']} "
            f"BRA={result['sass_branches']}",
            flush=True,
        )
        print(f"SASS top: {result['sass_top']}", flush=True)
    else:
        print(f"SASS unavailable: {result.get('sass_error')}", flush=True)
    print(
        f"\nVERDICT: select/FMA = {result['select_to_fma_ratio']}  "
        f"cascade_survives={result['cascade_survives']}",
        flush=True,
    )

    # ---- irreducible-floor probe -------------------------------------------
    # How much of the ~13.3us is unavoidable for ANY kernel of this shape?
    # `floor` does the same global traffic (load 32x32, store 32x32 + 32x32)
    # and the same 32-deep serial rsqrt dependency chain, but ZERO cross-lane
    # reductions. Anything a CUDA rewrite achieves is bounded below by this.
    import triton
    import triton.language as tl

    @triton.jit
    def _floor_kernel(out_ptr, inv_ptr, src_ptr, n: tl.constexpr, k):
        b = tl.program_id(0).to(tl.int64)
        r = tl.arange(0, 32)
        c = tl.arange(0, 32)
        off = (k + r)[:, None] * n + (k + c)[None, :]
        a = tl.load(src_ptr + b * n * n + off)
        # 32-deep serial scalar chain (same depth as the real factorization),
        # entirely lane-local -- no shuffles, no shared memory, no barriers.
        acc = tl.sum(tl.sum(a, axis=1), axis=0) * 0.0 + 4.0
        for _ in range(0, 32):
            acc = tl.rsqrt(acc) + 4.0
        tl.store(out_ptr + b * n * n + off, a * acc)
        tl.store(inv_ptr + b * 1024 + r[:, None] * 32 + c[None, :], a)

    # Disambiguate the floor: is the ~13us the kernel's WORK, or pure
    # per-launch cost in a 32-deep dependent chain? `empty` does nothing at
    # all; `ldst` does only the global traffic. If `empty` also costs ~13us,
    # the target is the launch structure, not any kernel body.
    @triton.jit
    def _empty_kernel(out_ptr, inv_ptr, src_ptr, n: tl.constexpr, k):
        # Touch one scalar so the body is non-empty (a bare `pass` fails to
        # capture `tl` in the JIT scope) but does no global traffic.
        b = tl.program_id(0)
        if b < 0:
            tl.store(out_ptr, 0.0)

    @triton.jit
    def _ldst_kernel(out_ptr, inv_ptr, src_ptr, n: tl.constexpr, k):
        b = tl.program_id(0).to(tl.int64)
        r = tl.arange(0, 32)
        c = tl.arange(0, 32)
        off = (k + r)[:, None] * n + (k + c)[None, :]
        a = tl.load(src_ptr + b * n * n + off)
        tl.store(out_ptr + b * n * n + off, a)
        tl.store(inv_ptr + b * 1024 + r[:, None] * 32 + c[None, :], a)

    timings = {}
    for label, fn in (
        ("shipped", None),
        ("floor", _floor_kernel),
        ("ldst", _ldst_kernel),
        ("empty", _empty_kernel),
    ):
        try:
            if fn is None:
                def one(kk):
                    return micro[(batch,)](
                        work, dinv, work, n=n, k=kk,
                        FIRST=False, RECIPROCAL_SOLVE=True, num_warps=1,
                    )
            else:
                def one(kk):
                    return fn[(batch,)](work, dinv, work, n=n, k=kk, num_warps=1)
            for _ in range(3):
                for kk in range(0, n, 32):
                    one(kk)
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            iters = 10
            ev0.record()
            for _ in range(iters):
                for kk in range(0, n, 32):
                    one(kk)
            ev1.record()
            torch.cuda.synchronize()
            per_call = ev0.elapsed_time(ev1) * 1000.0 / iters / 32.0
            timings[label] = round(per_call, 3)
            print(f"{label}: {per_call:.3f}us/call", flush=True)
        except Exception as exc:
            timings[label] = repr(exc)
            print(f"{label}: ERROR {exc!r}", flush=True)
    result["floor_us"] = timings
    if isinstance(timings.get("floor"), float) and isinstance(
        timings.get("shipped"), float
    ):
        result["headroom_over_floor"] = round(
            timings["shipped"] / max(timings["floor"], 1e-9), 2
        )
        print(
            f"\nshipped/floor = {result['headroom_over_floor']}x "
            f"({timings['shipped']}us vs {timings['floor']}us)",
            flush=True,
        )

    result["passed"] = True
    del work, dinv, eye
    torch.cuda.empty_cache()
    return result


def run_coopprobe(filter_ns=None):
    """Experiment 040 gate: measure cooperative full-grid barrier cost.

    The proposed 1x4096 factorization needs one resident CTA per SM and about
    three hardware grid rendezvous per outer tile. This isolates that cost
    before the much larger correctness kernel is built. The launch uses the
    CUDA default execution queue; no auxiliary/concurrent queue is created.
    """
    from torch.utils.cpp_extension import load_inline

    cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cooperative_barrier_floor(int phases,
                                          unsigned long long* ticks) {
    cg::grid_group grid = cg::this_grid();
    const unsigned long long begin = clock64();
    for (int phase = 0; phase < phases; ++phase) grid.sync();
    ticks[blockIdx.x] = clock64() - begin;
}

torch::Tensor coop_launch(int64_t blocks64, int64_t phases64) {
    const int blocks = (int)blocks64;
    int phases = (int)phases64;
    torch::Tensor ticks = torch::empty(
        {blocks}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    unsigned long long* ptr =
        reinterpret_cast<unsigned long long*>(ticks.data_ptr<int64_t>());
    void* args[] = {&phases, &ptr};
    cudaError_t status = cudaLaunchCooperativeKernel(
        (void*)cooperative_barrier_floor, dim3(blocks), dim3(32), args, 0);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    return ticks;
}
"""
    module = load_inline(
        name="coop_floor_exp040",
        cpp_sources="torch::Tensor coop_launch(int64_t, int64_t);",
        cuda_sources=cuda_source,
        functions=["coop_launch"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    properties = torch.cuda.get_device_properties(0)
    sms = int(properties.multi_processor_count)
    rows = []
    for blocks in (sms, 2 * sms):
        for phases in (0, 64, 96, 128, 192, 256):
            try:
                for _ in range(3):
                    module.coop_launch(blocks, phases)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                iterations = 30
                start.record()
                ticks = None
                for _ in range(iterations):
                    ticks = module.coop_launch(blocks, phases)
                end.record()
                torch.cuda.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000.0 / iterations
                max_ticks = int(ticks.max().item()) if ticks is not None else 0
                row = {
                    "blocks": blocks,
                    "phases": phases,
                    "mean_us": elapsed_us,
                    "max_ticks": max_ticks,
                }
                rows.append(row)
                print(
                    f"blocks={blocks} phases={phases}: {elapsed_us:.3f}us "
                    f"max_ticks={max_ticks}",
                    flush=True,
                )
            except Exception as exc:
                rows.append({"blocks": blocks, "phases": phases, "error": repr(exc)})
                print(f"blocks={blocks} phases={phases}: ERROR {exc!r}", flush=True)
                break
    return {
        "mode": "coopprobe",
        "device": torch.cuda.get_device_name(0),
        "sms": sms,
        "rows": rows,
        "passed": any(r.get("phases") == 192 and "mean_us" in r for r in rows),
    }


def run_coopphase(filter_ns=None):
    """Read device-clock constituent timings from the instrumented V1 kernel."""
    import statistics
    import candidate as cand

    module = getattr(cand, "_COOP4096", None)
    if module is None or not hasattr(module, "chol4096_profile"):
        return {
            "mode": "coopphase",
            "passed": False,
            "error": repr(getattr(cand, "_COOP4096_ERROR", "profile unavailable")),
        }
    pristine = generate_input(batch=1, n=4096, cond=2, seed=74040)
    samples = []
    output = None
    for _ in range(7):
        output = pristine.clone()
        raw = [int(v) for v in module.chol4096_profile(output)]
        samples.append([nanoseconds / 1000.0 for nanoseconds in raw[:4]])
    labels = ["diagonal_us", "panel_us", "trailing_us", "clear_us"]
    medians = {
        label: statistics.median(row[index] for row in samples)
        for index, label in enumerate(labels)
    }
    medians["accounted_us"] = sum(medians.values())
    good, message = check_implementation(pristine, output)
    for label in labels:
        print(f"{label}={medians[label]:.3f}", flush=True)
    print(f"accounted_us={medians['accounted_us']:.3f} ok={good} {message}", flush=True)
    return {
        "mode": "coopphase",
        "device": torch.cuda.get_device_name(0),
        "samples_us": samples,
        "medians_us": medians,
        "correct": bool(good),
        "correctness_message": message,
        "passed": bool(good),
    }


def run_dotprobe(filter_ns=None):
    import statistics

    import triton
    import triton.language as tl

    @triton.jit
    def _dot_kernel(
        a_ptr, b_ptr, c_ptr, M, K: tl.constexpr, N: tl.constexpr,
        MODE: tl.constexpr, BM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        rk = tl.arange(0, K)
        rn = tl.arange(0, N)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        if MODE == "tf32":
            acc = tl.dot(a, b, input_precision="tf32", out_dtype=tl.float32)
        elif MODE == "tf32x3":
            acc = tl.dot(a, b, input_precision="tf32x3", out_dtype=tl.float32)
        elif MODE == "fp16":
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32)
        else:  # fp16x3: three-fp16-MMA emulated fp32
            a_hi = a.to(tl.float16)
            a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
            b_hi = b.to(tl.float16)
            b_lo = (b - b_hi.to(tl.float32)).to(tl.float16)
            acc = tl.dot(a_hi, b_hi, out_dtype=tl.float32)
            acc += tl.dot(a_hi, b_lo, out_dtype=tl.float32)
            acc += tl.dot(a_lo, b_hi, out_dtype=tl.float32)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)

    # (K, N) tile shapes the split32 panel/trailing dots use.
    tile_shapes = [
        (32, 128), (32, 256), (128, 128), (256, 128), (128, 256),
    ]
    BM = 128
    BIGM = 128 * 148 * 4  # ~4x the SM count in 128-row tiles: saturate the GPU
    modes = ["tf32", "tf32x3", "fp16x3", "fp16"]

    def _time(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            _l2_flush()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        ts.sort()
        return sum(ts) / len(ts)

    results = []
    for K, N in tile_shapes:
        a = torch.randn(BIGM, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)
        c = torch.empty(BIGM, N, device="cuda", dtype=torch.float32)
        grid = (triton.cdiv(BIGM, BM),)
        ref = (a @ b)  # fp32 reference
        row = {"K": K, "N": N, "us": {}, "max_relerr": {}}
        for mode in modes:
            fn = lambda m=mode: _dot_kernel[grid](a, b, c, BIGM, K, N, m, BM)
            fn(); torch.cuda.synchronize()
            relerr = ((c - ref).abs().max() / ref.abs().max()).item()
            row["us"][mode] = _time(fn)
            row["max_relerr"][mode] = relerr
        base = row["us"]["tf32x3"]
        row["fp16x3_vs_tf32x3"] = base / row["us"]["fp16x3"]
        results.append(row)
        print(
            f"K={K:<4} N={N:<4} "
            + "  ".join(f"{m}={row['us'][m]:.1f}us(relerr {row['max_relerr'][m]:.1e})" for m in modes)
            + f"  ==> fp16x3_vs_tf32x3={row['fp16x3_vs_tf32x3']:.3f}x",
            flush=True,
        )
        del a, b, c, ref
        torch.cuda.empty_cache()

    speedups = [r["fp16x3_vs_tf32x3"] for r in results]
    return {
        "mode": "dotprobe",
        "passed": True,
        "fp16x3_vs_tf32x3_geomean": statistics.geometric_mean(speedups),
        "fp16x3_vs_tf32x3_min": min(speedups),
        "fp16x3_vs_tf32x3_max": max(speedups),
        "shapes": results,
    }


def run_schedprobe(filter_ns=None):
    import math

    import submission as cand

    baseline = _load_sched_baseline_module()

    def _cfg(mod, key):
        """The full per-shape config that determines the emitted split32 path:
        panel/trailing precision + tile + mode + fp16 flag, plus the panel-width
        schedule. A shape is 'changed' iff this differs between candidate and
        baseline -- covers both the L2 schedule table and the L4 panel_prec."""
        shapes = getattr(mod, "_SPLIT32_SHAPES", {}) or {}
        sched = getattr(mod, "_SPLIT32_NB_SCHEDULE", {}) or {}
        return (shapes.get(key), sched.get(key))

    specs = SCHED_SPECS
    if filter_ns:
        specs = [s for s in SCHED_SPECS if s["n"] in filter_ns]

    results = []
    for spec in specs:
        key = (spec["batch"], spec["n"])
        cand_cfg = _cfg(cand, key)
        base_cfg = _cfg(baseline, key)
        changed = cand_cfg != base_cfg
        sched = cand_cfg[0] if changed else None  # non-None marks 'changed' below
        bytes_per = spec["batch"] * spec["n"] ** 2 * 4
        count = max(1, min(30, int(256e6 // bytes_per)))
        args = dict(spec)
        data_list = []
        for _ in range(count):
            data_list.append(generate_input(**args))
            args["seed"] += 42
        pristine = [d.clone() for d in data_list]

        cand_outs = [cand.custom_kernel(d.clone()) for d in data_list]
        base_outs = [baseline.custom_kernel(d.clone()) for d in data_list]
        torch.cuda.synchronize()
        cand_ok = all(
            check_implementation(d, o)[0]
            for d, o in zip(pristine, cand_outs, strict=True)
        )
        base_ok = all(
            check_implementation(d, o)[0]
            for d, o in zip(pristine, base_outs, strict=True)
        )

        fam_rows = []
        if changed:
            for fam, condv in SCHED_FAMILIES:
                fdata = generate_input(
                    batch=spec["batch"],
                    n=spec["n"],
                    cond=condv,
                    seed=spec["seed"] + 777,
                    case=fam,
                )
                fpristine = fdata.clone()
                fout = cand.custom_kernel(fdata)
                torch.cuda.synchronize()
                fok, fmsg = check_implementation(fpristine, fout)
                fam_rows.append({"family": fam, "ok": bool(fok), "msg": fmsg})
                del fdata, fpristine, fout

        # Time baseline, candidate, baseline again to bracket run-to-run drift.
        n = spec["n"]
        iters = 20 if n <= 512 else (15 if n <= 1024 else 12)
        warmup = 4
        base_t1 = _time_callable_rotating(
            baseline.custom_kernel, data_list, warmup, iters
        )
        cand_t = _time_callable_rotating(
            cand.custom_kernel, data_list, warmup, iters
        )
        base_t2 = _time_callable_rotating(
            baseline.custom_kernel, data_list, 2, iters
        )
        base_mean = (base_t1["mean_us"] + base_t2["mean_us"]) / 2.0
        drift = abs(base_t1["mean_us"] - base_t2["mean_us"]) / base_mean
        speedup = base_mean / cand_t["mean_us"]

        fam_ok = all(r["ok"] for r in fam_rows) if fam_rows else None
        row = {
            "shape": _spec_label(spec),
            "batch": spec["batch"],
            "n": n,
            "changed": changed,
            "cand_cfg": [list(cand_cfg[0]) if cand_cfg[0] else None,
                         list(cand_cfg[1]) if cand_cfg[1] else None],
            "base_cfg": [list(base_cfg[0]) if base_cfg[0] else None,
                         list(base_cfg[1]) if base_cfg[1] else None],
            "rotating_inputs": count,
            "baseline_us": base_mean,
            "baseline_us_pass1": base_t1["mean_us"],
            "baseline_us_pass2": base_t2["mean_us"],
            "baseline_drift": drift,
            "candidate_us": cand_t["mean_us"],
            "candidate_best_us": cand_t["best_us"],
            "speedup": speedup,
            "candidate_ok": cand_ok,
            "baseline_ok": base_ok,
            "families": fam_rows,
            "families_ok": fam_ok,
        }
        results.append(row)
        print(
            f"{row['shape']:<12} changed={changed} cfg={row['cand_cfg']} "
            f"base={base_mean:.2f}us cand={cand_t['mean_us']:.2f}us "
            f"speedup={speedup:.4f}x drift={drift * 100:.2f}% "
            f"cand_ok={cand_ok} fam_ok={fam_ok}",
            flush=True,
        )
        del data_list, pristine, cand_outs, base_outs
        torch.cuda.empty_cache()

    changed_rows = [r for r in results if r["changed"]]
    agg = None
    if changed_rows:
        agg = math.prod(r["speedup"] for r in changed_rows) ** (
            1.0 / len(changed_rows)
        )
    all_ok = all(
        r["candidate_ok"] and (r["families_ok"] in (True, None)) for r in results
    )
    return {
        "mode": "schedprobe",
        "passed": bool(all_ok),
        "changed_geomean_speedup": agg,
        "num_changed": len(changed_rows),
        "shapes": results,
    }


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


def _make_chunked_call(chunk):
    """Split the batch into sub-batches of `chunk` and call batched cuSOLVER on
    each, then concatenate. Diagnostic for whether a better-occupancy code path
    is hit at a smaller batch (pure default stream, shippable if it wins)."""

    def _call(data):
        batch = data.shape[0]
        parts = [
            torch.linalg.cholesky_ex(data[i : i + chunk], check_errors=False).L
            for i in range(0, batch, chunk)
        ]
        return torch.cat(parts)

    return _call


PROBE_SPECS = [
    {"batch": 640, "n": 512, "cond": 2, "seed": 510512},
    {"batch": 2, "n": 4096, "cond": 2, "seed": 514096},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 44048},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 42024},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 511024},
    {"batch": 1, "n": 4096, "cond": 2, "seed": 48096},
]

_APPROACHES = [
    ("batched", _batched_call),
    ("loop", _loop_call),
    ("streamed", _streamed_call),
    ("chunk64", _make_chunked_call(64)),
    ("chunk128", _make_chunked_call(128)),
]


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


# ---------------------------------------------------------------------------
# precprobe mode: test the exp-006 hypothesis for large single matrices --
# a right-looking BLOCKED Cholesky whose diagonal block stays FP32 but whose
# O(n^3) trailing Schur update runs on tensor cores (TF32 or FP16, FP32
# accumulate) can beat cuSOLVER's all-FP32 potrf on n >= 8192 while still
# passing the reconstruction gate. Pure torch, DEFAULT STREAM ONLY.
# ---------------------------------------------------------------------------
def _bf16x9_syrk(L21):
    """Genuine BF16x9 numerics for `L21 @ L21^T`, computed in pure torch as a
    manual 3-way BF16 split with FP32 accumulation.

    An FP32 value is represented exactly by three BF16 values
    (b0 + b1 + b2, each successive term ~2^-8 smaller). Rounding the operands to
    BF16 and doing the products/sums in FP32 reproduces exactly what the cuBLAS
    BF16x9 tensor-core path does (bf16 operands, fp32 accumulate). Used as a
    ground-truth accuracy proxy: run with emulation OFF so the fp32 GEMMs here are
    genuinely native FP32 (not themselves emulated). Slow (9 GEMMs) -- accuracy
    probe only, never for the shipped speed path.
    """
    b0 = L21.bfloat16()
    r1 = L21 - b0.float()
    b1 = r1.bfloat16()
    r2 = r1 - b1.float()
    b2 = r2.bfloat16()
    parts = [b0.float(), b1.float(), b2.float()]
    upd = None
    for i in range(3):
        for j in range(3):
            term = parts[i] @ parts[j].transpose(-1, -2)
            upd = term if upd is None else upd + term
    return upd


def _blocked_cholesky(mat, nb, trailing):
    """Right-looking blocked Cholesky on a single (n, n) FP32 matrix.

    trailing selects the precision of the O(n^3) trailing Schur update
    `A22 -= L21 @ L21^T`. The diagonal block potrf and the panel triangular solve
    stay FP32 for stability. Returns FP32 lower-triangular. Options:
      * "tf32"     -- allow_tf32=True, TF32 GEMM plus separate subtraction
                       (exp 006 ship path; Stage-A control).
      * "tf32_addmm" -- fused in-place TF32 addmm into the trailing view.
      * "fp16"/"bf16" -- cast operands to fp16/bf16, GEMM, back to fp32.
      * "fp32"     -- allow_tf32=False, plain FP32 GEMM. With the cuBLAS BF16x9
                      env var set (Pass B) this GEMM becomes fused BF16x9-emulated
                      FP32; without it (Pass A) it is native FP32. Same source ->
                      timing delta between the two passes proves BF16x9 engaged.
      * "bf16x9"   -- manual 3-way BF16 split (genuine BF16x9 numerics, slow).
    """
    A = mat.clone()
    n = A.shape[0]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = trailing in ("tf32", "tf32_addmm")
        for k in range(0, n, nb):
            kb = min(nb, n - k)
            A11 = A[k : k + kb, k : k + kb]
            L11 = torch.linalg.cholesky_ex(A11, check_errors=False).L
            A[k : k + kb, k : k + kb] = L11
            j = k + kb
            if j >= n:
                break
            A21 = A[j:, k : k + kb]
            # Solve L21 @ L11^T = A21 for the panel (FP32 TRSM).
            L21 = torch.linalg.solve_triangular(
                L11.transpose(-1, -2), A21, upper=True, left=False
            )
            A[j:, k : k + kb] = L21
            # Trailing Schur update on tensor cores.
            if trailing == "tf32_addmm":
                A[j:, j:].addmm_(
                    L21, L21.transpose(-1, -2), beta=1.0, alpha=-1.0
                )
                continue
            if trailing in ("tf32", "fp32"):
                upd = L21 @ L21.transpose(-1, -2)
            elif trailing == "bf16x9":
                upd = _bf16x9_syrk(L21)
            else:
                dt = torch.float16 if trailing == "fp16" else torch.bfloat16
                lh = L21.to(dt)
                upd = (lh @ lh.transpose(-1, -2)).float()
            A[j:, j:] -= upd
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return torch.tril(A)


def _make_blocked_call(nb, trailing):
    def _call(data):
        return _blocked_cholesky(data[0], nb, trailing).unsqueeze(0)

    return _call


def _recon_ratio(data, output):
    """Fraction of the reconstruction tolerance used (worst over batch).

    Mirrors reference.check_implementation: residual = ||L L^T - A||_1,
    allowed = 20 * n * eps * ||A||_1. Returns residual/allowed (so <1 passes),
    computed with TF32 disabled just like the real checker.
    """
    n = data.shape[-1]
    eps = torch.finfo(torch.float32).eps
    scale = torch.linalg.matrix_norm(data, ord=1, dim=(-2, -1)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        recon = output @ output.transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    residual = torch.linalg.matrix_norm(recon - data, ord=1, dim=(-2, -1))
    allowed = 20.0 * max(n, 1) * eps * scale
    return (residual / allowed).amax().item()


# Precision x block-size variants for the large-n probe. batched is the control.
# exp 007: the `blocked_fp32_nb*` variants are the STAR -- a plain-FP32 blocked
# Cholesky whose trailing GEMM becomes fused BF16x9-emulated FP32 when the cuBLAS
# env var is set (run the runner with the `emu` token, Pass B). Comparing Pass B
# (emu on) vs Pass A (emu off) on the SAME `blocked_fp32_*` variant proves BF16x9
# engaged and gives its speedup vs native FP32 blocked; comparing against
# `blocked_tf32_*` / `batched` gives the ship decision. `blocked_bf16x9split_*`
# gives a genuine (native-GEMM) accuracy proxy -- run it with emu OFF.
def _precprobe_variants(spec):
    n = spec["n"]
    case = spec.get("case", "dense")
    if case != "dense":
        # Ill-conditioned families (spectrum/lowrank/...): accuracy is the concern,
        # not speed (speed is conditioning-independent). Run a minimal set focused
        # on residual margins at the ship block size.
        return [
            ("batched", _batched_call),
            ("blocked_tf32_nb2048", _make_blocked_call(2048, "tf32")),
            ("blocked_fp32_nb2048", _make_blocked_call(2048, "fp32")),
            ("blocked_bf16x9split_nb2048", _make_blocked_call(2048, "bf16x9")),
        ]
    if n >= 32768:
        # Very expensive (~200ms/iter): lean sweep at the sizes that matter.
        return [
            ("batched", _batched_call),
            ("blocked_tf32_nb4096", _make_blocked_call(4096, "tf32")),
            ("blocked_fp32_nb2048", _make_blocked_call(2048, "fp32")),
            ("blocked_fp32_nb4096", _make_blocked_call(4096, "fp32")),
        ]
    variants = [("batched", _batched_call)]
    for nb in (1024, 2048):
        variants.append((f"blocked_tf32_nb{nb}", _make_blocked_call(nb, "tf32")))
    for nb in (1024, 2048, 4096):
        variants.append((f"blocked_fp32_nb{nb}", _make_blocked_call(nb, "fp32")))
    # Genuine BF16x9 numerics (manual split, native FP32 GEMMs) -- accuracy proxy.
    variants.append(("blocked_bf16x9split_nb2048", _make_blocked_call(2048, "bf16x9")))
    return variants


def _engagement_bench(m=8192, iters=5):
    """Time a standalone large FP32 A @ B (tf32 disabled). With the BF16x9 env var
    set this GEMM is emulated (should be several x faster on B200); without it, it
    is native FP32. The ratio across the two passes proves engagement -- do not
    trust the blocked speedups blindly."""
    a = torch.randn((m, m), device="cuda", dtype=torch.float32)
    b = torch.randn((m, m), device="cuda", dtype=torch.float32)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.synchronize()
        for _ in range(2):
            _ = a @ b
        torch.cuda.synchronize()
        durs = []
        for _ in range(iters):
            _l2_flush()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            _ = a @ b
            end.record()
            torch.cuda.synchronize()
            durs.append(start.elapsed_time(end) * 1e3)
        durs.sort()
        return sum(durs) / len(durs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


# 1x32768 is HELD OUT of the default sweep (it is ~221ms/iter and expensive);
# probe it explicitly with `precprobe 32768` only once the approach works.
PRECPROBE_SPECS = [
    {"batch": 1, "n": 8192, "cond": 2, "seed": 48192},
    {"batch": 1, "n": 8192, "cond": 5, "seed": 68193, "case": "spectrum"},
    {"batch": 1, "n": 8192, "cond": 4, "seed": 68194, "case": "lowrank"},
    {"batch": 1, "n": 16384, "cond": 2, "seed": 48284},
    {"batch": 1, "n": 16384, "cond": 5, "seed": 68285, "case": "spectrum"},
    {"batch": 1, "n": 16384, "cond": 4, "seed": 68286, "case": "lowrank"},
    {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
]


def run_precprobe(filter_ns=None):
    specs = PRECPROBE_SPECS
    if filter_ns:
        specs = [s for s in PRECPROBE_SPECS if s["n"] in filter_ns]
    else:
        specs = [s for s in PRECPROBE_SPECS if s["n"] != 32768]
    emu = os.environ.get("CUBLAS_FP32_EMULATED_BF16X9_MATH") == "1"
    eng = _engagement_bench()
    print(
        f"BF16x9_EMU={'ON' if emu else 'OFF'}  "
        f"standalone_fp32_matmul_8192(tf32off)={eng:.1f}us  "
        f"(emu-on should be several x faster -> proves BF16x9 engaged)",
        flush=True,
    )
    results = []
    for spec in specs:
        data = generate_input(**spec)
        n = spec["n"]
        variants = _precprobe_variants(spec)
        iters = 8 if n <= 8192 else (5 if n <= 16384 else 3)
        warmup = 2 if n <= 16384 else 1
        base_us = None
        rows = []
        for name, fn in variants:
            out = fn(data.clone())
            torch.cuda.synchronize()
            good, message = check_implementation(data, out)
            ratio = _recon_ratio(data, out)
            mean_us, best_us = _time_callable(data, fn, warmup, iters)
            if name == "batched":
                base_us = mean_us
            rows.append(
                {
                    "variant": name,
                    "passed": bool(good),
                    "mean_us": mean_us,
                    "best_us": best_us,
                    "tol_frac": ratio,
                    "margin_x": (1.0 / ratio) if ratio > 0 else float("inf"),
                    "message": message,
                }
            )
        for r in rows:
            r["speedup_vs_batched"] = (base_us / r["mean_us"]) if base_us else None
            flag = "PASS" if r["passed"] else "FAIL"
            sp = r["speedup_vs_batched"]
            print(
                f"n={n:<6} {r['variant']:<22} {flag} mean={r['mean_us']:>9.1f}us "
                f"speedup={ (f'{sp:.2f}x' if sp else '   -  '):>7} "
                f"tol_frac={r['tol_frac']:.3e} margin={r['margin_x']:.1f}x",
                flush=True,
            )
        results.append({"spec": _spec_label(spec), "n": n, "variants": rows})
    return {
        "mode": "precprobe",
        "bf16x9_emu": emu,
        "engagement_fp32_matmul_8192_us": eng,
        "shapes": results,
    }


# exp 008 Stage A: paired same-process comparison so B200 timing drift cannot
# masquerade as an improvement. Both variants use identical blocking, panel
# factorization, panel solve, result construction, and checker; only the Schur
# update expression differs.
FUSIONPROBE_SPECS = [
    {"batch": 1, "n": 16384, "cond": 2, "seed": 48284},
    {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
]


def run_fusionprobe(filter_ns=None):
    specs = FUSIONPROBE_SPECS
    if filter_ns:
        specs = [s for s in specs if s["n"] in filter_ns]
    else:
        specs = [s for s in specs if s["n"] == 16384]
    results = []
    for spec in specs:
        data = generate_input(**spec)
        n = spec["n"]
        nb = 4096 if n >= 32768 else 2048
        variants = [
            ("separate_tf32", _make_blocked_call(nb, "tf32")),
            ("fused_addmm_tf32", _make_blocked_call(nb, "tf32_addmm")),
        ]
        rows = []
        for name, fn in variants:
            out = fn(data.clone())
            torch.cuda.synchronize()
            good, message = check_implementation(data, out)
            ratio = _recon_ratio(data, out)
            mean_us, best_us = _time_callable(
                data, fn, warmup=2 if n <= 16384 else 1,
                iters=7 if n <= 16384 else 4,
            )
            rows.append(
                {
                    "variant": name,
                    "passed": bool(good),
                    "mean_us": mean_us,
                    "best_us": best_us,
                    "tol_frac": ratio,
                    "margin_x": (1.0 / ratio) if ratio > 0 else float("inf"),
                    "message": message,
                }
            )
        control = rows[0]["mean_us"]
        for row in rows:
            row["speedup_vs_separate"] = control / row["mean_us"]
            print(
                f"n={n:<6} {row['variant']:<20} "
                f"{'PASS' if row['passed'] else 'FAIL'} "
                f"mean={row['mean_us']:>9.1f}us best={row['best_us']:>9.1f}us "
                f"speedup={row['speedup_vs_separate']:.3f}x "
                f"tol_frac={row['tol_frac']:.3e}",
                flush=True,
            )
        results.append({"spec": _spec_label(spec), "n": n, "nb": nb, "variants": rows})
    return {"mode": "fusionprobe", "shapes": results}


def _time_matmul(m, iters=5):
    a = torch.randn((m, m), device="cuda", dtype=torch.float32)
    b = torch.randn((m, m), device="cuda", dtype=torch.float32)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.synchronize()
        for _ in range(2):
            _ = a @ b
        torch.cuda.synchronize()
        durs = []
        for _ in range(iters):
            _l2_flush()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            _ = a @ b
            e.record()
            torch.cuda.synchronize()
            durs.append(s.elapsed_time(e) * 1e3)
        durs.sort()
        return sum(durs) / len(durs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def run_emuprobe(filter_ns=None):
    """Find which config (if any) makes PyTorch route FP32 matmul through cuBLAS
    BF16x9 emulation on the B200. Times a standalone FP32 A@B (tf32 off) under
    each preferred BLAS backend; engagement shows as a several-x speedup."""
    emu_env = {
        k: os.environ.get(k)
        for k in (
            "CUBLAS_FP32_EMULATED_BF16X9_MATH",
            "CUBLAS_EMULATE_SINGLE_PRECISION",
            "TORCH_BLAS_PREFER_CUBLASLT",
        )
    }
    print(f"emu_env={emu_env}", flush=True)
    try:
        print(f"fp32_precision={torch.backends.cuda.matmul.fp32_precision}", flush=True)
    except Exception as exc:
        print(f"fp32_precision attr unavailable: {exc}", flush=True)
    ms = sorted(filter_ns) if filter_ns else [8192, 16384]
    results = []
    for lib in ("default", "cublas", "cublaslt"):
        try:
            torch.backends.cuda.preferred_blas_library(lib)
        except Exception as exc:
            print(f"preferred_blas_library({lib}) failed: {exc}", flush=True)
            continue
        row = {"blas": lib, "times_us": {}}
        for m in ms:
            t = _time_matmul(m)
            row["times_us"][str(m)] = t
            print(f"blas={lib:<9} m={m:<6} fp32_matmul(tf32off)={t:.1f}us", flush=True)
        results.append(row)
    return {"mode": "emuprobe", "emu_env": emu_env, "results": results}


# ---------------------------------------------------------------------------
# mxprobe (experiment 034): MXFP8 block-scaled panel products for 1x32768.
# One sandbox run produces: Triton/torch versions, PTX backend proof
# (tcgen05 block-scaled MMA or bust), a micro numeric sanity check, a
# component micro-bench vs the exp-014 per-tensor fp8 pipeline (with a GEMM
# config sweep and a torch._scaled_mm MX availability check), the six-family
# checker gate at 32768, and a paired end-to-end timing vs the exact ranked
# baseline (/root/baseline_exp034.py).
# ---------------------------------------------------------------------------
def _load_exp034_baseline_module():
    spec = importlib.util.spec_from_file_location(
        "baseline_exp034", "/root/baseline_exp034.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_mxprobe(filter_ns=None):
    import triton as _triton
    import submission as cand

    ns = sorted(filter_ns) if filter_ns else [32768]
    result = {
        "mode": "mxprobe",
        "torch": torch.__version__,
        "triton": getattr(_triton, "__version__", "?"),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "has_dot_scaled": hasattr(_triton.language, "dot_scaled"),
        "target_ns": ns,
    }
    passed = True

    # --- 1. micro numerics + backend proof on an exact-tile toy case -------
    torch.manual_seed(1234)
    lhs = torch.randn(512, 512, device="cuda")
    rhs = torch.randn(256, 512, device="cuda")
    out = torch.zeros(512, 256, device="cuda")
    micro = {}
    try:
        cand._mxfp8_panel_update(out, lhs, rhs)
        torch.cuda.synchronize()
        exact = lhs @ rhs.T
        micro["rel_err"] = float(
            ((out + exact).norm() / exact.norm()).item()
        )
        micro["finite"] = bool(torch.isfinite(out).all().item())
        micro["ok"] = micro["finite"] and micro["rel_err"] < 0.10
        del exact
    except Exception as exc:
        micro = {"ok": False, "error": repr(exc)}
    passed = passed and micro.get("ok", False)
    result["micro"] = micro
    del lhs, rhs, out

    ptx = getattr(cand, "_MXFP8_PTX", None) or ""
    tc_ops = sorted(
        {
            tok.rstrip(",;")
            for line in ptx.splitlines()
            if "tcgen05" in line
            for tok in line.split()
            if "tcgen05" in tok
        }
    )
    result["ptx_bytes"] = len(ptx)
    result["ptx_tcgen05_ops"] = tc_ops[:20]
    result["ptx_kind_mxf8f6f4"] = "mxf8f6f4" in ptx
    result["ptx_block_scale"] = "block_scale" in ptx
    backend = getattr(cand, "_MXFP8_BACKEND", "triton_dot_scaled")
    result["backend"] = backend
    if backend == "scaled_mm_mx":
        # V2 dispatches cuBLAS rather than a Triton kernel, so there is no PTX
        # to inspect. Prove the block-scaled MX GEMM engaged by capturing the
        # device kernel names the panel update actually launches.
        names = []
        try:
            from torch.profiler import ProfilerActivity, profile

            lhs_p = torch.randn(512, 512, device="cuda")
            rhs_p = torch.randn(256, 512, device="cuda")
            out_p = torch.zeros(512, 256, device="cuda")
            cand._mxfp8_panel_update(out_p, lhs_p, rhs_p)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                cand._mxfp8_panel_update(out_p, lhs_p, rhs_p)
                torch.cuda.synchronize()
            names = sorted(
                {
                    e.name
                    for e in prof.events()
                    if getattr(e, "device_type", None) is not None
                    and "cuda" in str(getattr(e, "device_type", "")).lower()
                }
            )
            del lhs_p, rhs_p, out_p
        except Exception as exc:
            names = [f"profiler-error: {exc!r}"]
        result["scaled_mm_kernels"] = names[:30]
        blob = " ".join(names).lower()
        result["backend_proof"] = any(
            tok in blob
            for tok in ("mxf8", "block_scale", "blockscaled", "blockwise", "mx_")
        )
    else:
        result["backend_proof"] = bool(tc_ops) and result["ptx_kind_mxf8f6f4"]
    if not result["backend_proof"]:
        # Timing an emulated path would be invalid-as-MXFP8 evidence; keep
        # running to gather diagnostics but fail the probe.
        passed = False

    def _time_op(fn, warmup=3, iters=15):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / iters

    # --- 2. component micro-bench vs exp-014 per-tensor fp8 + config sweep -
    # Representative mid-loop panel product of the 32768/nb=4096 schedule:
    # k=16384 -> lhs (12288, 16384), rhs (4096, 16384), panel (12288, 4096).
    base = _load_exp034_baseline_module()
    m_r, k_r, n_r = 12288, 16384, 4096
    factor_lhs = torch.randn(m_r, k_r, device="cuda") * 0.05
    factor_rhs = torch.randn(n_r, k_r, device="cuda") * 0.05
    panel = torch.randn(m_r, n_r, device="cuda")

    def _cand_call():
        cand._mxfp8_panel_update(panel, factor_lhs, factor_rhs)

    def _base_call():
        panel.sub_(
            base._fp8_product_32768(
                factor_lhs, factor_rhs.transpose(-1, -2)
            )
        )

    component = {}
    try:
        component["baseline_fp8_us"] = _time_op(_base_call)
    except Exception as exc:
        component["baseline_error"] = repr(exc)
    sweep = []
    default_cfg = (
        cand._MX_GEMM_BLOCK_M,
        cand._MX_GEMM_BLOCK_N,
        cand._MX_GEMM_BLOCK_K,
        cand._MX_GEMM_WARPS,
        cand._MX_GEMM_STAGES,
    )
    # V2 dispatches cuBLAS, so the Triton tile globals do nothing: one entry.
    sweep_cfgs = (
        [default_cfg]
        if backend == "scaled_mm_mx"
        else [
            default_cfg,
            (128, 256, 128, 8, 3),
            (256, 128, 128, 8, 3),
            (128, 128, 256, 8, 3),
            (128, 128, 128, 4, 4),
        ]
    )
    for bm, bn, bk, w, s in sweep_cfgs:
        if (bm, bn, bk, w, s) in [tuple(r["cfg"]) for r in sweep]:
            continue
        cand._MX_GEMM_BLOCK_M = bm
        cand._MX_GEMM_BLOCK_N = bn
        cand._MX_GEMM_BLOCK_K = bk
        cand._MX_GEMM_WARPS = w
        cand._MX_GEMM_STAGES = s
        try:
            us = _time_op(_cand_call)
            sweep.append({"cfg": [bm, bn, bk, w, s], "us": us})
            print(f"sweep cfg={bm},{bn},{bk},w{w},s{s} -> {us:.1f}us", flush=True)
        except Exception as exc:
            sweep.append({"cfg": [bm, bn, bk, w, s], "error": repr(exc)})
            print(f"sweep cfg={bm},{bn},{bk},w{w},s{s} -> ERROR {exc!r}", flush=True)
    timed = [r for r in sweep if "us" in r]
    if timed:
        best = min(timed, key=lambda r: r["us"])
        component["mxfp8_best_us"] = best["us"]
        component["mxfp8_best_cfg"] = best["cfg"]
        bm, bn, bk, w, s = best["cfg"]
        cand._MX_GEMM_BLOCK_M = bm
        cand._MX_GEMM_BLOCK_N = bn
        cand._MX_GEMM_BLOCK_K = bk
        cand._MX_GEMM_WARPS = w
        cand._MX_GEMM_STAGES = s
        if "baseline_fp8_us" in component:
            component["component_speedup"] = (
                component["baseline_fp8_us"] / best["us"]
            )
    else:
        passed = False
    component["sweep"] = sweep
    result["component"] = component

    # --- V2 availability: torch._scaled_mm with MX (1x32) e8m0 scales ------
    v2 = {}
    try:
        def _to_blocked(x):
            rows, cols = x.shape
            b = x.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
            b = b.reshape(-1, 4, 32, 4).transpose(1, 2)
            return b.reshape(-1).contiguous()

        q_l, s_l = cand._mx_quant_e4m3(factor_lhs)
        q_r, s_r = cand._mx_quant_e4m3(factor_rhs)
        sa = _to_blocked(s_l).view(torch.float8_e8m0fnu)
        sb = _to_blocked(s_r).view(torch.float8_e8m0fnu)

        def _v2_call():
            return torch._scaled_mm(
                q_l, q_r.t(), scale_a=sa, scale_b=sb,
                out_dtype=torch.float32,
            )

        prod = _v2_call()
        exact_r = factor_lhs @ factor_rhs.transpose(-1, -2)
        v2["rel_err"] = float(
            ((prod - exact_r).norm() / exact_r.norm()).item()
        )
        v2["gemm_only_us"] = _time_op(_v2_call)
        v2["available"] = v2["rel_err"] < 0.10
        del q_l, s_l, q_r, s_r, sa, sb, prod, exact_r
    except Exception as exc:
        v2 = {"available": False, "error": repr(exc)}
    result["v2_scaled_mm_mx"] = v2
    del factor_lhs, factor_rhs, panel
    torch.cuda.empty_cache()

    # --- 3. six-family checker gate at each target n -----------------------
    fam_rows = []
    for n in ns:
        for spec in [s for s in TEST_SPECS if s["n"] == n and s["batch"] == 1]:
            hits0 = cand._MXFP8_HITS
            falls0 = cand._LARGE_FP8_FALLBACKS
            data = generate_input(**spec)
            pristine = data.clone()
            out_t = cand.custom_kernel(data)
            torch.cuda.synchronize()
            ok, msg = check_implementation(pristine, out_t)
            row = {
                "spec": _spec_label(spec),
                "case": spec.get("case", "dense"),
                "ok": bool(ok),
                "msg": msg,
                "mx_hits": cand._MXFP8_HITS - hits0,
                "fallbacks": cand._LARGE_FP8_FALLBACKS - falls0,
                "mx_error": getattr(cand, "_MXFP8_ERROR", None),
                "large_error": getattr(cand, "_LARGE_FP8_ERROR", None),
            }
            fam_rows.append(row)
            # `fallbacks == 0` was the wrong gate: `_left_looking_large`'s
            # finiteness handoff fires on ill-conditioned families at 32768 in
            # the BASELINE too, so it flagged clean candidates as failures (exp
            # 034). Correctness is the checker; fallbacks are recorded as data
            # and compared against the baseline by the pairedgrid gate.
            passed = passed and ok and row["mx_hits"] > 0
            print(
                f"family {row['spec']:<16} {row['case']:<12} ok={ok} "
                f"mx_hits={row['mx_hits']} fallbacks={row['fallbacks']} {msg}",
                flush=True,
            )
            del data, pristine, out_t
            torch.cuda.empty_cache()
    result["families"] = fam_rows

    # --- 4. paired end-to-end timing per target n (dense ranked spec) ------
    paired = []
    for n in ns:
        spec = next(s for s in BENCH_SPECS if s["n"] == n)
        args = dict(spec)
        data_list = []
        for _ in range(2):
            data_list.append(generate_input(**args))
            args["seed"] += 42
        hits0 = cand._MXFP8_HITS
        base_t1 = _time_callable_rotating(base.custom_kernel, data_list, 2, 8)
        cand_t = _time_callable_rotating(cand.custom_kernel, data_list, 2, 8)
        base_t2 = _time_callable_rotating(base.custom_kernel, data_list, 1, 8)
        base_mean = (base_t1["mean_us"] + base_t2["mean_us"]) / 2.0
        drift = abs(base_t1["mean_us"] - base_t2["mean_us"]) / base_mean
        row = {
            "n": n,
            "baseline_us": base_mean,
            "baseline_us_pass1": base_t1["mean_us"],
            "baseline_us_pass2": base_t2["mean_us"],
            "baseline_drift": drift,
            "candidate_us": cand_t["mean_us"],
            "candidate_best_us": cand_t["best_us"],
            "speedup": base_mean / cand_t["mean_us"],
            "mx_hits_timing": cand._MXFP8_HITS - hits0,
        }
        paired.append(row)
        print(
            f"paired n={n} base={base_mean:.1f}us cand={cand_t['mean_us']:.1f}us "
            f"speedup={row['speedup']:.4f}x drift={drift * 100:.2f}% "
            f"mx_hits={row['mx_hits_timing']}",
            flush=True,
        )
        if row["mx_hits_timing"] <= 0:
            passed = False
        del data_list
        torch.cuda.empty_cache()
    result["paired"] = paired

    result["passed"] = bool(passed)
    return result


# ---------------------------------------------------------------------------
# pairedgrid (experiment 035): paired same-process 15-shape grid.
#
# The full-grid `benchmark` mode times one submission per sandbox, so a
# candidate-vs-baseline comparison crosses process and machine boundaries. Exp
# 034 measured the cost of that: byte-identical `60x1024` produced
# 1565.6/1389.4/1989.7us across three grid runs (43% spread on unchanged code),
# which is far too coarse to promote a ~0.6% geomean change. This mode loads
# BOTH sources in one process and interleaves them A-B-B-A per shape so that
# linear drift cancels within each pair, then reports per-shape ratios with a
# bootstrap CI instead of a ratio of independently-measured geomeans.
# ---------------------------------------------------------------------------
def _load_submission_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counter_deltas(before, after):
    return {k: after.get(k, 0) - v for k, v in before.items() if after.get(k, 0) != v}


def _counters(mod):
    """Hit/fallback counters exposed by a submission module, by convention
    `_<NAME>_HITS` / `_<NAME>_FALLBACKS`. Collected generically so baseline and
    candidate can expose different sets."""
    out = {}
    for key, value in vars(mod).items():
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        if key.endswith("_HITS") or key.endswith("_FALLBACKS"):
            out[key] = value
    return out


def _module_errors(mod):
    return {
        key: repr(value)
        for key, value in vars(mod).items()
        if key.endswith("_ERROR") and value is not None
    }


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mad(values):
    """Median absolute deviation - a tail-robust spread, unlike stdev."""
    med = _median(values)
    return _median([abs(v - med) for v in values])


def _bootstrap_ci(values, statistic, iters=4000, seed=20350719, alpha=0.05):
    import random as _random

    rng = _random.Random(seed)
    count = len(values)
    if count < 2:
        point = statistic(values)
        return point, point
    reps = []
    for _ in range(iters):
        reps.append(statistic([values[rng.randrange(count)] for _ in range(count)]))
    reps.sort()
    lo_idx = int((alpha / 2.0) * iters)
    hi_idx = min(iters - 1, int((1.0 - alpha / 2.0) * iters))
    return reps[lo_idx], reps[hi_idx]


def _geomean(values):
    import math as _math

    return _math.exp(sum(_math.log(v) for v in values) / len(values))


def _pairedgrid_budget(n):
    """(rotating_inputs, iters_per_block, repeats). Cheap shapes get more of
    both; n>=16384 keeps one 4GB input to leave headroom for two modules'
    workspaces."""
    if n <= 128:
        return 2, 20, 9
    if n <= 512:
        return 2, 15, 9
    if n <= 2048:
        return 2, 10, 7
    if n <= 8192:
        return 2, 5, 7
    return 1, 3, 5


def _timed_block(fn, data_list, iters):
    """Median us/call over `iters` L2-flushed timed regions. Median (not mean)
    so a single tail event does not shift the block."""
    durations = []
    for _ in range(iters):
        _l2_flush()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for data in data_list:
            fn(data)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1e3 / len(data_list))
    return _median(durations)


def _pairedgrid_shape(spec, base_mod, cand_mod):
    n = spec["n"]
    n_inputs, iters, repeats = _pairedgrid_budget(n)

    args = dict(spec)
    data_list = []
    for _ in range(n_inputs):
        data_list.append(generate_input(**args))
        args["seed"] += 42

    # --- correctness on both modules, before any timing --------------------
    base_before, cand_before = _counters(base_mod), _counters(cand_mod)
    pristine = data_list[0].clone()
    base_out = base_mod.custom_kernel(data_list[0].clone())
    cand_out = cand_mod.custom_kernel(data_list[0].clone())
    torch.cuda.synchronize()
    base_ok, base_msg = check_implementation(pristine, base_out)
    cand_ok, cand_msg = check_implementation(pristine, cand_out)
    base_delta = _counter_deltas(base_before, _counters(base_mod))
    cand_delta = _counter_deltas(cand_before, _counters(cand_mod))
    del pristine, base_out, cand_out

    # The exp-034 gate demanded `fallbacks == 0`, which fires on
    # `_left_looking_large`'s pre-existing finiteness handoff at 32768 - present
    # in baseline and candidate alike - and reported `passed: false` on a clean
    # result. Gate on checker pass + no fallback the baseline does not also take.
    new_fallbacks = {}
    for key, value in cand_delta.items():
        if not key.endswith("_FALLBACKS"):
            continue
        base_value = base_delta.get(key, 0)
        if value > base_value:
            new_fallbacks[key] = [base_value, value]

    # --- warm BOTH modules before any timed block --------------------------
    # First-call cost (JIT compile, autotune, graph capture) differs by orders
    # of magnitude and would otherwise land entirely on whichever runs first.
    warmup = 3 if n <= 2048 else 2
    for _ in range(warmup):
        for data in data_list:
            base_mod.custom_kernel(data)
        for data in data_list:
            cand_mod.custom_kernel(data)
    torch.cuda.synchronize()

    # --- A B B A interleaving; discard the first repeat ---------------------
    ratios, base_us, cand_us, order_spread = [], [], [], []
    for repeat in range(repeats + 1):
        a1 = _timed_block(base_mod.custom_kernel, data_list, iters)
        b1 = _timed_block(cand_mod.custom_kernel, data_list, iters)
        b2 = _timed_block(cand_mod.custom_kernel, data_list, iters)
        a2 = _timed_block(base_mod.custom_kernel, data_list, iters)
        if repeat == 0:
            continue
        a_mean = 0.5 * (a1 + a2)
        b_mean = 0.5 * (b1 + b2)
        ratios.append(a_mean / b_mean)
        base_us.append(a_mean)
        cand_us.append(b_mean)
        # A-vs-A within one block pair: the harness's own floor, independent of
        # any candidate difference.
        order_spread.append(abs(a1 - a2) / a_mean)

    half = len(ratios) // 2
    ratio_med = _median(ratios)
    ci_lo, ci_hi = _bootstrap_ci(ratios, _median)
    row = {
        "n": n,
        "batch": spec["batch"],
        "spec": _spec_label(spec),
        "ratio_median": ratio_med,
        "ratio_ci95": [ci_lo, ci_hi],
        "ratio_mad": _mad(ratios),
        "ratios": [round(r, 5) for r in ratios],
        "baseline_us": _median(base_us),
        "candidate_us": _median(cand_us),
        "order_spread_median": _median(order_spread),
        "ratio_first_half": _median(ratios[:half]) if half else None,
        "ratio_second_half": _median(ratios[half:]) if half else None,
        "repeats": len(ratios),
        "iters_per_block": iters,
        "rotating_inputs": n_inputs,
        "baseline_ok": bool(base_ok),
        "candidate_ok": bool(cand_ok),
        "baseline_msg": base_msg,
        "candidate_msg": cand_msg,
        "baseline_counters": base_delta,
        "candidate_counters": cand_delta,
        "new_fallbacks": new_fallbacks,
    }
    row["ok"] = bool(base_ok and cand_ok and not new_fallbacks)

    del data_list
    torch.cuda.empty_cache()
    return row


def run_shapediag(filter_ns=None):
    """Experiment 036 diagnosis: where does one shape's wall clock go?

    Profiles the shipped path for each requested shape and reports the CUDA
    kernel breakdown (per-kernel total time, call count, share of device
    time), plus total device time vs measured wall time. The gap between
    them is idle/launch/dependency stall -- the number that decides whether
    a shape is launch-bound or compute-bound.
    """
    from torch.profiler import ProfilerActivity, profile

    import submission as sub

    specs = BENCH_SPECS
    if filter_ns:
        specs = [s for s in specs if s["n"] in filter_ns]
    out = {"mode": "shapediag", "device": torch.cuda.get_device_name(0), "shapes": []}
    for spec in specs:
        data = generate_input(**{k: v for k, v in spec.items() if k != "case"},
                              case=spec.get("case", "dense"))
        for _ in range(5):
            sub.custom_kernel(data)
        torch.cuda.synchronize()

        # wall time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 20
        start.record()
        for _ in range(iters):
            sub.custom_kernel(data)
        end.record()
        torch.cuda.synchronize()
        wall_us = start.elapsed_time(end) * 1000.0 / iters

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                sub.custom_kernel(data)
            torch.cuda.synchronize()
        agg = {}
        for e in prof.key_averages():
            dev = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
            if dev <= 0.0:
                continue
            name = e.key
            slot = agg.setdefault(name, {"us": 0.0, "calls": 0})
            slot["us"] += dev / 3.0
            slot["calls"] += int(e.count) // 3
        total_dev = sum(v["us"] for v in agg.values())
        kernels = sorted(agg.items(), key=lambda kv: -kv[1]["us"])
        rows = [
            {
                "kernel": k[:70],
                "us": round(v["us"], 2),
                "calls": v["calls"],
                "us_per_call": round(v["us"] / max(v["calls"], 1), 3),
                "pct_device": round(100.0 * v["us"] / max(total_dev, 1e-9), 1),
            }
            for k, v in kernels[:18]
        ]
        total_calls = sum(v["calls"] for v in agg.values())
        entry = {
            "spec": f"batch={spec['batch']} n={spec['n']}",
            "batch": spec["batch"],
            "n": spec["n"],
            "wall_us": round(wall_us, 1),
            "device_us": round(total_dev, 1),
            "idle_us": round(wall_us - total_dev, 1),
            "idle_pct": round(100.0 * (wall_us - total_dev) / max(wall_us, 1e-9), 1),
            "kernel_launches": total_calls,
            "us_per_launch_wall": round(wall_us / max(total_calls, 1), 3),
            "kernels": rows,
        }
        out["shapes"].append(entry)
        print(
            f"{entry['spec']}: wall={entry['wall_us']}us device={entry['device_us']}us "
            f"idle={entry['idle_us']}us ({entry['idle_pct']}%) "
            f"launches={total_calls} wall/launch={entry['us_per_launch_wall']}us",
            flush=True,
        )
        for r in rows:
            print(
                f"    {r['pct_device']:>5}% {r['us']:>9.2f}us {r['calls']:>4} calls "
                f"{r['us_per_call']:>8.3f}us/call  {r['kernel']}",
                flush=True,
            )
        del data
        torch.cuda.empty_cache()
    out["passed"] = True
    return out


def run_pairedgrid(filter_ns=None):
    specs = BENCH_SPECS
    if filter_ns:
        specs = [s for s in BENCH_SPECS if s["n"] in filter_ns]

    base_mod = _load_submission_module("sub_base", "/root/submission.py")
    cand_mod = _load_submission_module("sub_cand", "/root/candidate.py")
    identical = (
        open("/root/submission.py", "rb").read()
        == open("/root/candidate.py", "rb").read()
    )
    print(
        f"pairedgrid: baseline=/root/submission.py candidate=/root/candidate.py "
        f"byte_identical={identical} shapes={len(specs)}",
        flush=True,
    )

    rows = []
    for spec in specs:
        row = _pairedgrid_shape(spec, base_mod, cand_mod)
        rows.append(row)
        ci = row["ratio_ci95"]
        print(
            f"{row['spec']:<34} ratio={row['ratio_median']:.4f} "
            f"CI=[{ci[0]:.4f},{ci[1]:.4f}] mad={row['ratio_mad'] * 100:.2f}% "
            f"AvA={row['order_spread_median'] * 100:.2f}% "
            f"base={row['baseline_us']:.1f}us cand={row['candidate_us']:.1f}us "
            f"ok={row['ok']}",
            flush=True,
        )
        if not row["ok"]:
            print(
                f"    baseline_ok={row['baseline_ok']} candidate_ok={row['candidate_ok']} "
                f"new_fallbacks={row['new_fallbacks']}",
                flush=True,
            )

    medians = [r["ratio_median"] for r in rows]
    geo = _geomean(medians)
    # Aggregate uncertainty must be PROPAGATED from each shape's own CI, not
    # bootstrapped over shapes. The 15 shapes are the complete fixed scoring
    # population, not a random sample; resampling them lets a draw omit the
    # only shape that moved, which inflates the interval until it swallows a
    # real win. (Exp 035: V2 measured 1.0061 with a resample CI of
    # [0.9990,1.0186] -> "not significant", vs a propagated [1.0057,1.0066]
    # -> significant. The 1x32768 shape's own CI was [1.0902,1.0910].)
    # log(geomean) = mean(log r_i), so var = sum(var_i) / n^2.
    log_ses = []
    for r in rows:
        lo, hi = r["ratio_ci95"]
        log_ses.append((_math.log(hi) - _math.log(lo)) / (2.0 * 1.96))
    geo_log_se = _math.sqrt(sum(s * s for s in log_ses)) / len(rows)
    geo_lo = _math.exp(_math.log(geo) - 1.96 * geo_log_se)
    geo_hi = _math.exp(_math.log(geo) + 1.96 * geo_log_se)
    # The harness's own noise floor: worst per-shape deviation from parity, and
    # the widest A-vs-A spread seen. On a byte-identical pair these ARE the
    # resolution limit; any candidate delta smaller than this is unmeasurable.
    worst_dev = max(abs(m - 1.0) for m in medians)
    worst_ava = max(r["order_spread_median"] for r in rows)
    result = {
        "mode": "pairedgrid",
        "byte_identical": identical,
        "shapes": rows,
        "geomean_ratio": geo,
        "geomean_ci95": [geo_lo, geo_hi],
        "geomean_ci_excludes_1": bool(geo_lo > 1.0 or geo_hi < 1.0),
        "worst_shape_deviation": worst_dev,
        "worst_order_spread": worst_ava,
        "all_shapes_ok": all(r["ok"] for r in rows),
        "passed": all(r["ok"] for r in rows),
    }
    print(
        f"\ngeomean(ratio) = {geo:.4f} CI95=[{geo_lo:.4f},{geo_hi:.4f}] "
        f"excludes_1.0={result['geomean_ci_excludes_1']}",
        flush=True,
    )
    print(
        f"noise floor: worst per-shape deviation {worst_dev * 100:.2f}%, "
        f"worst A-vs-A spread {worst_ava * 100:.2f}%",
        flush=True,
    )
    return result


def main():
    # argv may contain the mode, an optional comma-separated shapes filter, and
    # an optional `emu` token (handled at import time). Ignore `emu` here.
    args = [a for a in sys.argv[1:] if a.strip() and a != "emu"]
    mode = args[0] if args else "verify"
    filter_ns = None
    if len(args) > 1 and args[1].strip():
        filter_ns = {int(x) for x in args[1].split(",") if x.strip()}
    print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"BF16x9_EMU={'ON' if os.environ.get('CUBLAS_FP32_EMULATED_BF16X9_MATH') == '1' else 'OFF'}",
        flush=True,
    )
    import submission as _sub
    print(f"custom_cuda_loaded={getattr(_sub, '_CUDA_MOD', None) is not None}", flush=True)
    _err = getattr(_sub, "_CUDA_LOAD_ERROR", None)
    if _err:
        print("CUDA_LOAD_ERROR:\n" + _err, flush=True)
    if mode == "benchmark":
        result = run_benchmark(filter_ns)
    elif mode == "probe":
        result = run_probe(filter_ns)
    elif mode == "precprobe":
        result = run_precprobe(filter_ns)
    elif mode == "emuprobe":
        result = run_emuprobe(filter_ns)
    elif mode == "fusionprobe":
        result = run_fusionprobe(filter_ns)
    elif mode == "frontierprobe":
        result = run_frontierprobe()
    elif mode == "largefrontierprobe":
        result = run_largefrontierprobe()
    elif mode == "nocusolverprobe":
        result = run_nocusolverprobe(filter_ns)
    elif mode == "dualprobe":
        result = run_dualprobe(filter_ns)
    elif mode == "schedprobe":
        result = run_schedprobe(filter_ns)
    elif mode == "dotprobe":
        result = run_dotprobe(filter_ns)
    elif mode == "mxprobe":
        result = run_mxprobe(filter_ns)
    elif mode == "asmprobe":
        result = run_asmprobe(filter_ns)
    elif mode == "coopprobe":
        result = run_coopprobe(filter_ns)
    elif mode == "coopphase":
        result = run_coopphase(filter_ns)
    elif mode == "microprobe":
        result = run_microprobe(filter_ns)
    elif mode == "shapediag":
        result = run_shapediag(filter_ns)
    elif mode == "pairedgrid":
        result = run_pairedgrid(filter_ns)
    else:
        result = run_verify(filter_ns)
    print("RESULT_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
