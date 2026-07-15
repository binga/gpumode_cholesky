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
    {"batch": 1, "n": 32768, "cond": 4, "seed": 68369, "case": "lowrank"},
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
def _blocked_cholesky(mat, nb, trailing):
    """Right-looking blocked Cholesky on a single (n, n) FP32 matrix.

    trailing in {"tf32", "fp16", "bf16"} selects the precision of the O(n^3)
    trailing Schur update `A22 -= L21 @ L21^T`. The diagonal block potrf and the
    panel triangular solve stay FP32 for stability. Returns FP32 lower-triangular.
    """
    A = mat.clone()
    n = A.shape[0]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = trailing == "tf32"
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
            if trailing == "tf32":
                upd = L21 @ L21.transpose(-1, -2)
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
def _precprobe_variants(n):
    # For the very expensive 1x32768 case, run a lean TF32 nb-sweep (plus one
    # fp16 reference) to keep B200 cost down while picking the best block size.
    if n >= 32768:
        return [
            ("batched", _batched_call),
            ("blocked_tf32_nb1024", _make_blocked_call(1024, "tf32")),
            ("blocked_tf32_nb2048", _make_blocked_call(2048, "tf32")),
            ("blocked_tf32_nb4096", _make_blocked_call(4096, "tf32")),
            ("blocked_fp16_nb2048", _make_blocked_call(2048, "fp16")),
        ]
    variants = [("batched", _batched_call)]
    for nb in (512, 1024, 2048):
        variants.append((f"blocked_tf32_nb{nb}", _make_blocked_call(nb, "tf32")))
    for nb in (512, 1024, 2048):
        variants.append((f"blocked_fp16_nb{nb}", _make_blocked_call(nb, "fp16")))
    variants.append(("blocked_bf16_nb1024", _make_blocked_call(1024, "bf16")))
    return variants


# 1x32768 is HELD OUT of the default sweep (it is ~221ms/iter and expensive);
# probe it explicitly with `precprobe 32768` only once the approach works.
PRECPROBE_SPECS = [
    {"batch": 1, "n": 8192, "cond": 2, "seed": 48192},
    {"batch": 1, "n": 16384, "cond": 2, "seed": 48284},
    {"batch": 1, "n": 32768, "cond": 2, "seed": 48368},
]


def run_precprobe(filter_ns=None):
    specs = PRECPROBE_SPECS
    if filter_ns:
        specs = [s for s in PRECPROBE_SPECS if s["n"] in filter_ns]
    else:
        specs = [s for s in PRECPROBE_SPECS if s["n"] != 32768]
    results = []
    for spec in specs:
        data = generate_input(**spec)
        n = spec["n"]
        variants = _precprobe_variants(n)
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
    return {"mode": "precprobe", "shapes": results}


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
    elif mode == "precprobe":
        result = run_precprobe(filter_ns)
    else:
        result = run_verify(filter_ns)
    print("RESULT_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
