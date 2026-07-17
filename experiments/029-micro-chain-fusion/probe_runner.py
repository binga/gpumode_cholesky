"""Runs inside the Modal B200 sandbox for experiment 029.

Paired same-process probe: times the exact ranked exp-021 baseline (#882958)
against an exp-029 candidate on the target shapes, mirroring the official
harness (rotating inputs to a 256MiB target, L2 clear between iterations,
retained outputs, correctness re-check after timing). Also runs a six-family
correctness sweep for every changed shape and reports backend counters to
prove the fast path executed. Emits `RESULT_JSON:` for the driver.
"""

import importlib.util
import base64
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import zipfile

if "blocking" in sys.argv:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.insert(0, "/root/reference")

import torch  # noqa: E402

from reference import check_implementation, generate_input  # noqa: E402

TARGET_SPECS = [
    {"batch": 64, "n": 256, "cond": 2, "seed": 41256},
    {"batch": 16, "n": 512, "cond": 2, "seed": 41512},
    {"batch": 640, "n": 512, "cond": 2, "seed": 510512},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 42024},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 511024},
    {"batch": 8, "n": 2048, "cond": 2, "seed": 512048},
]

FULL_GRID_SPECS = [
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

FAMILIES = ["dense", "spectrum", "diagonal", "lowrank", "rowscale", "tridiagonal"]

COUNTER_NAMES = [
    "_FUSED_CTA_HITS",
    "_FUSED_CTA_FALLBACKS",
    "_FUSED_CTA_ERROR",
    "_GRAPH_SP_HITS",
    "_GRAPH_SP_FALLBACKS",
    "_GRAPH_SP_ERROR",
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _counters(mod):
    return {c: getattr(mod, c, None) for c in COUNTER_NAMES}


def _gen(spec):
    args = {k: v for k, v in spec.items() if k in ("batch", "n", "cond", "seed", "case")}
    return generate_input(**args)


def _rotating_inputs(spec):
    bytes_per = spec["batch"] * spec["n"] * spec["n"] * 4
    count = max(1, min(50, (256 * 1024 * 1024) // bytes_per))
    data_list = []
    args = dict(spec)
    for _ in range(count):
        data_list.append(_gen(args))
        args["seed"] = args.get("seed", 0) + 42
    return data_list


_L2 = None


def _clear_l2():
    global _L2
    if _L2 is None:
        _L2 = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda")
    _L2.zero_()


def _time_fn(fn, data_list, warmup=3, iters=15):
    outs = None
    for _ in range(warmup):
        outs = [fn(d) for d in data_list]
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        _clear_l2()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outs = [fn(d) for d in data_list]
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / len(data_list))
    return times, outs


def _time_pair(base_fn, cand_fn, data_list, warmup=3, iters=15):
    """Alternate baseline/candidate order every round to control drift."""
    outputs = {"baseline": None, "candidate": None}
    functions = {"baseline": base_fn, "candidate": cand_fn}
    for iteration in range(warmup):
        order = ("baseline", "candidate") if iteration % 2 == 0 else (
            "candidate",
            "baseline",
        )
        for name in order:
            outputs[name] = [functions[name](data) for data in data_list]
    torch.cuda.synchronize()
    times = {"baseline": [], "candidate": []}
    paired_rounds = []
    for iteration in range(iters):
        order = ("baseline", "candidate") if iteration % 2 == 0 else (
            "candidate",
            "baseline",
        )
        row = {"round": iteration, "order": list(order)}
        for name in order:
            torch.cuda.synchronize()
            _clear_l2()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[name] = [functions[name](data) for data in data_list]
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) * 1e3 / len(data_list)
            times[name].append(elapsed)
            row[f"{name}_us"] = elapsed
        row["speedup"] = row["baseline_us"] / row["candidate_us"]
        paired_rounds.append(row)
    return times, outputs, paired_rounds


def _residual(data, out):
    ok, msg = check_implementation(data, out)
    return ok, msg


def _probe_shape(spec, base_fn, cand_fn, base_mod, cand_mod):
    label = f"{spec['batch']}x{spec['n']}"
    data_list = _rotating_inputs(spec)
    pristine = [d.clone() for d in data_list]

    pre_base = _counters(base_mod)
    pre_cand = _counters(cand_mod)
    times, outputs, paired_rounds = _time_pair(base_fn, cand_fn, data_list)
    base_times = times["baseline"]
    cand_times = times["candidate"]
    base_outs = outputs["baseline"]
    cand_outs = outputs["candidate"]
    post_cand = _counters(cand_mod)

    checks = []
    for d, o in zip(pristine, cand_outs):
        ok, msg = _residual(d, o)
        checks.append({"ok": bool(ok), "msg": msg})
        if not ok:
            break
    base_mean = statistics.mean(base_times)
    cand_mean = statistics.mean(cand_times)
    result = {
        "shape": label,
        "rotating_inputs": len(data_list),
        "baseline": {
            "mean_us": base_mean,
            "best_us": min(base_times),
            "median_us": statistics.median(base_times),
        },
        "candidate": {
            "mean_us": cand_mean,
            "best_us": min(cand_times),
            "median_us": statistics.median(cand_times),
        },
        "speedup_mean": base_mean / cand_mean,
        "paired_speedup_mean": statistics.mean(
            row["speedup"] for row in paired_rounds
        ),
        "paired_rounds": paired_rounds,
        "candidate_checks": checks,
        "candidate_all_ok": all(c["ok"] for c in checks),
        "counters_before": pre_cand,
        "counters_after": post_cand,
    }
    del base_outs, cand_outs, data_list, pristine
    torch.cuda.empty_cache()
    return result


def _family_sweep(spec, cand_fn, cand_mod):
    rows = []
    for fam in FAMILIES:
        args = dict(spec)
        args["seed"] = args.get("seed", 0) + hash(fam) % 1000 + 7
        if fam != "dense":
            args["case"] = fam
            args["cond"] = 5 if fam in ("spectrum", "diagonal") else 4
            if fam == "tridiagonal":
                args["cond"] = 1
        data = _gen(args)
        pristine = data.clone()
        pre = _counters(cand_mod)
        out = cand_fn(data)
        torch.cuda.synchronize()
        post = _counters(cand_mod)
        ok, msg = _residual(pristine, out)
        rows.append(
            {
                "family": fam,
                "ok": bool(ok),
                "msg": msg,
                "hits_delta": (post.get("_FUSED_CTA_HITS") or 0)
                - (pre.get("_FUSED_CTA_HITS") or 0),
                "fallbacks_delta": (post.get("_FUSED_CTA_FALLBACKS") or 0)
                - (pre.get("_FUSED_CTA_FALLBACKS") or 0),
            }
        )
        del data, pristine, out
    torch.cuda.empty_cache()
    return rows


def _profile_shape(spec, cand_fn):
    data = _gen(spec)
    for _ in range(3):
        cand_fn(data)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            cand_fn(data)
        torch.cuda.synchronize()
    rows = []
    for avg in sorted(
        prof.key_averages(), key=lambda a: -a.self_device_time_total
    )[:15]:
        if avg.self_device_time_total > 0:
            rows.append(
                {
                    "kernel": avg.key[:100],
                    "self_us": avg.self_device_time_total / 5.0,
                    "count_per_call": avg.count / 5.0,
                }
            )
    del data
    torch.cuda.empty_cache()
    return rows


def _compiled_artifact_bundle(mod):
    """Package the exact candidate specializations launched by this run."""
    handles = getattr(mod, "_TRITON_COMPILED", {})
    stream = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (key, compiled) in enumerate(sorted(handles.items(), key=str)):
            label = f"{index:02d}-" + "-".join(str(part) for part in key)
            entry = {
                "key": [str(part) for part in key],
                "compiled_type": type(compiled).__name__,
                "metadata": repr(getattr(compiled, "metadata", None)),
                "stages": {},
            }
            for stage, value in getattr(compiled, "asm", {}).items():
                if str(stage) in ("source", "llir"):
                    continue
                payload = value if isinstance(value, bytes) else str(value).encode()
                suffix = "cubin" if isinstance(value, bytes) else str(stage)
                name = f"{label}/{stage}.{suffix}"
                archive.writestr(name, payload)
                entry["stages"][str(stage)] = {
                    "path": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                if str(stage) == "cubin" and isinstance(value, bytes):
                    with tempfile.NamedTemporaryFile(suffix=".cubin") as cubin_file:
                        cubin_file.write(value)
                        cubin_file.flush()
                        for tool, arguments, suffix_name in (
                            ("cuobjdump", ["--dump-resource-usage"], "resources.txt"),
                        ):
                            completed = subprocess.run(
                                [tool, *arguments, cubin_file.name],
                                capture_output=True,
                                check=False,
                            )
                            tool_payload = completed.stdout + completed.stderr
                            tool_name = f"{label}/{suffix_name}"
                            archive.writestr(tool_name, tool_payload)
                            entry["stages"][suffix_name] = {
                                "path": tool_name,
                                "bytes": len(tool_payload),
                                "sha256": hashlib.sha256(tool_payload).hexdigest(),
                                "returncode": completed.returncode,
                            }
            manifest[label] = entry
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    return base64.b64encode(stream.getvalue()).decode(), manifest


def main():
    shapes_filter = None
    skip_families = False
    do_profile = False
    full_grid = False
    for arg in sys.argv[1:]:
        if arg == "nofam":
            skip_families = True
        elif arg == "profile":
            do_profile = True
        elif arg == "blocking":
            pass
        elif arg == "fullgrid":
            full_grid = True
            skip_families = True
        else:
            shapes_filter = {
                tuple(int(x) for x in part.split("x"))
                for part in arg.split(",")
            }

    base_mod = _load("baseline_submission", "/root/baseline_submission.py")
    cand_mod = _load("candidate_submission", "/root/candidate_submission.py")

    def source_sha256(path):
        with open(path, "rb") as source_file:
            return hashlib.sha256(source_file.read()).hexdigest()

    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = None
    try:
        driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        driver_version = None

    payload = {
        "schema_version": 1,
        "experiment": "029-micro-chain-fusion",
        "mode": "paired-probe",
        "baseline_sha256": source_sha256("/root/baseline_submission.py"),
        "candidate_sha256": source_sha256("/root/candidate_submission.py"),
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "driver": driver_version,
            "triton": triton_version,
            "ncu_available": bool(__import__("shutil").which("ncu")),
        },
        "shapes": [],
        "families": {},
    }

    source = FULL_GRID_SPECS if full_grid else TARGET_SPECS
    specs = [
        s
        for s in source
        if shapes_filter is None or (s["batch"], s["n"]) in shapes_filter
    ]
    for spec in specs:
        label = f"{spec['batch']}x{spec['n']}"
        print(f"probing {label} ...", flush=True)
        res = _probe_shape(
            spec, base_mod.custom_kernel, cand_mod.custom_kernel, base_mod, cand_mod
        )
        payload["shapes"].append(res)
        print(
            f"  base {res['baseline']['mean_us']:.1f}us cand "
            f"{res['candidate']['mean_us']:.1f}us speedup "
            f"{res['speedup_mean']:.3f}x ok={res['candidate_all_ok']}",
            flush=True,
        )
        if do_profile:
            prof_rows = _profile_shape(spec, cand_mod.custom_kernel)
            payload.setdefault("profiles", {})[label] = prof_rows
            for r in prof_rows[:8]:
                print(
                    f"  prof {r['self_us']:>9.1f}us x{r['count_per_call']:<6.1f}"
                    f" {r['kernel']}",
                    flush=True,
                )
        if not skip_families:
            fams = _family_sweep(spec, cand_mod.custom_kernel, cand_mod)
            payload["families"][label] = fams
            bad = [f for f in fams if not f["ok"]]
            print(
                f"  families ok={len(fams)-len(bad)}/{len(fams)}"
                + (f" FAIL={[f['family'] for f in bad]}" if bad else ""),
                flush=True,
            )

    payload["final_counters"] = _counters(cand_mod)
    payload["passed"] = all(s["candidate_all_ok"] for s in payload["shapes"]) and all(
        f["ok"] for rows in payload["families"].values() for f in rows
    )
    speedups = [s["speedup_mean"] for s in payload["shapes"]]
    if speedups:
        payload["geomean_target_speedup"] = math.exp(
            sum(math.log(s) for s in speedups) / len(speedups)
        )
    if full_grid and len(payload["shapes"]) == len(FULL_GRID_SPECS):
        for side in ("baseline", "candidate"):
            payload[f"{side}_geomean_us"] = math.exp(
                sum(math.log(s[side]["mean_us"]) for s in payload["shapes"])
                / len(payload["shapes"])
            )
        payload["aggregate_speedup"] = (
            payload["baseline_geomean_us"] / payload["candidate_geomean_us"]
        )
        print(
            f"full grid geomean: baseline "
            f"{payload['baseline_geomean_us']:.1f}us -> candidate "
            f"{payload['candidate_geomean_us']:.1f}us "
            f"({payload['aggregate_speedup']:.4f}x)",
            flush=True,
        )
    artifact_b64, manifest = _compiled_artifact_bundle(cand_mod)
    payload["compiled_artifacts"] = manifest
    print("RESULT_JSON:" + json.dumps(payload), flush=True)
    chunk_size = 8192
    for index, offset in enumerate(range(0, len(artifact_b64), chunk_size)):
        print(
            f"ARTIFACT_B64_PART:{index}:"
            + artifact_b64[offset : offset + chunk_size],
            flush=True,
        )


if __name__ == "__main__":
    main()
