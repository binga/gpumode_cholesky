"""Nsight Compute latency decomposition for one shape, on a Modal B200.

Answers, per kernel, WHAT the time is made of -- measured, not modelled:

  memory   dram__throughput.avg.pct_of_peak_sustained_elapsed
           -> how close to the HBM roofline. High = data-movement bound.
  compute  sm__throughput.avg.pct_of_peak_sustained_elapsed
           -> how close to the SM issue roofline.
  tensor   sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
           -> how much of the time the MMA pipes are actually busy.
  occ      sm__warps_active.avg.pct_of_peak_sustained_active
           -> achieved occupancy. Low occ + low memory + low tensor is the
              signature of a LATENCY-bound kernel: nothing is saturated and
              there aren't enough warps to hide anything.
  stall_*  smsp__average_warp_latency_issue_stalled_* (per-warp cycles)
           -> WHY warps aren't issuing:
              long_scoreboard = waiting on global/shared memory
              wait            = waiting on a dependent instruction's result
              barrier / membar = waiting on other warps
              imc_miss        = constant-cache miss
              not_selected    = ready, but another warp was picked (healthy)

Read it as: a kernel whose top stall is `wait` with low occupancy cannot be
fixed by precision or bandwidth work -- only by shortening the dependency
chain or adding parallelism.

Usage:
    uv run --with modal python scripts/ncu_profile.py --batch 4 --n 1024
    uv run --with modal python scripts/ncu_profile.py --batch 640 --n 512 \
        --json results/ncu-640x512.json

Counter access: profiling counters often require elevated permission. If the
run reports ERR_NVGPUCTRPERM the sandbox denies them; that is a platform
limit, not a bug here, and it is reported rather than worked around.
"""

import argparse
import csv
import io
import json
import statistics
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes.sum",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio",
    "smsp__average_warp_latency_issue_stalled_wait.ratio",
    "smsp__average_warp_latency_issue_stalled_barrier.ratio",
    "smsp__average_warp_latency_issue_stalled_imc_miss.ratio",
    "smsp__average_warp_latency_issue_stalled_not_selected.ratio",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
])

SHORT = {
    "gpu__time_duration.sum": "us",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "mem%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm%",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "tensor%",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occ%",
    "dram__bytes.sum": "bytes",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio": "st_mem",
    "smsp__average_warp_latency_issue_stalled_wait.ratio": "st_wait",
    "smsp__average_warp_latency_issue_stalled_barrier.ratio": "st_barrier",
    "smsp__average_warp_latency_issue_stalled_imc_miss.ratio": "st_imc",
    "smsp__average_warp_latency_issue_stalled_not_selected.ratio": "st_notsel",
    "launch__grid_size": "grid",
    "launch__block_size": "block",
    "launch__registers_per_thread": "regs",
}


def _build_image(submission_path):
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
        )
        .entrypoint([])
        .apt_install("cuda-nsight-compute-13-0")
        .pip_install("torch", "numpy", "ninja")
        .add_local_dir(str(ROOT / "reference"), "/root/reference", copy=True)
        .add_local_file(str(submission_path), "/root/submission.py", copy=True)
        .add_local_file(
            str(ROOT / "scripts" / "_ncu_runner.py"), "/root/_ncu_runner.py", copy=True
        )
    )


def parse_ncu_csv(text):
    """ncu --csv emits one row per (kernel invocation, metric)."""
    start = text.find('"ID","Process ID"')
    if start < 0:
        start = text.find('"ID"')
    if start < 0:
        return {}
    reader = csv.DictReader(io.StringIO(text[start:]))
    per_kernel = {}
    for row in reader:
        name = (row.get("Kernel Name") or "").strip()
        metric = (row.get("Metric Name") or "").strip()
        raw = (row.get("Metric Value") or "").replace(",", "").strip()
        if not name or not metric:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        key = SHORT.get(metric, metric)
        per_kernel.setdefault(name, {}).setdefault(key, []).append(value)
    return per_kernel


def summarize(per_kernel):
    rows = []
    for name, metrics in per_kernel.items():
        row = {"kernel": name, "calls": len(metrics.get("us", []))}
        for key, values in metrics.items():
            row[key] = statistics.median(values)
        row["total_us"] = row.get("us", 0.0) * row["calls"] / 1000.0
        rows.append(row)
    rows.sort(key=lambda r: -r.get("total_us", 0.0))
    return rows


def verdict(r):
    """One-word cause, from the counters rather than from a work model."""
    mem, sm = r.get("mem%", 0.0), r.get("sm%", 0.0)
    tensor, occ = r.get("tensor%", 0.0), r.get("occ%", 0.0)
    if mem >= 60:
        return "DRAM-bound"
    if tensor >= 50:
        return "tensor-bound"
    if sm >= 60:
        return "issue-bound"
    stalls = {k: v for k, v in r.items() if k.startswith("st_") and k != "st_notsel"}
    top = max(stalls, key=stalls.get) if stalls else "?"
    if occ < 20:
        return f"LATENCY-bound (occ {occ:.0f}%, top stall {top})"
    return f"latency/other (top stall {top})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--submission", default=None)
    ap.add_argument("--gpu", default="B200")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sub = Path(args.submission).resolve() if args.submission else ROOT / "submission.py"
    print(f"submission -> {sub}", file=sys.stderr)
    print(f"shape      -> {args.batch}x{args.n}", file=sys.stderr)

    cmd = [
        "ncu", "--csv", "--target-processes", "all",
        "--metrics", METRICS,
        "--kernel-name-base", "demangled",
        "python", "-u", "/root/_ncu_runner.py",
    ]
    app = modal.App.lookup("gpumode-cholesky-ncu", create_if_missing=True)
    out_lines = []
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            *cmd,
            image=_build_image(sub),
            gpu=args.gpu,
            app=app,
            timeout=args.timeout,
            workdir="/root",
            secrets=[
                modal.Secret.from_dict({
                    "NCU_BATCH": str(args.batch),
                    "NCU_N": str(args.n),
                    "NCU_ITERS": str(args.iters),
                })
            ],
        )
        try:
            for line in sandbox.stdout:
                out_lines.append(line.rstrip("\n"))
            sandbox.wait()
            err = sandbox.stderr.read()
        finally:
            sandbox.terminate()

    text = "\n".join(out_lines)
    if "ERR_NVGPUCTRPERM" in text or "ERR_NVGPUCTRPERM" in (err or ""):
        print("\nERROR: the sandbox denies GPU performance counters "
              "(ERR_NVGPUCTRPERM). ncu cannot collect metrics here.",
              file=sys.stderr)
        print(err or "", file=sys.stderr)
        return 2

    per_kernel = parse_ncu_csv(text)
    if not per_kernel:
        print("\nNo ncu CSV parsed. Raw tail:", file=sys.stderr)
        print(text[-3000:], file=sys.stderr)
        print(err or "", file=sys.stderr)
        return 1

    rows = summarize(per_kernel)
    print(f"\n{'kernel':<42} {'us/call':>8} {'mem%':>6} {'sm%':>6} "
          f"{'tensor%':>8} {'occ%':>6} {'regs':>5}  cause")
    for r in rows:
        print(f"{r['kernel'][:42]:<42} {r.get('us',0)/1000:>8.2f} "
              f"{r.get('mem%',0):>6.1f} {r.get('sm%',0):>6.1f} "
              f"{r.get('tensor%',0):>8.1f} {r.get('occ%',0):>6.1f} "
              f"{r.get('regs',0):>5.0f}  {verdict(r)}")

    print(f"\n{'kernel':<42} {'st_mem':>8} {'st_wait':>8} {'st_barrier':>10} "
          f"{'st_imc':>8} {'st_notsel':>10}")
    for r in rows:
        print(f"{r['kernel'][:42]:<42} {r.get('st_mem',0):>8.2f} "
              f"{r.get('st_wait',0):>8.2f} {r.get('st_barrier',0):>10.2f} "
              f"{r.get('st_imc',0):>8.2f} {r.get('st_notsel',0):>10.2f}")

    if args.json:
        out = ROOT / args.json if not Path(args.json).is_absolute() else Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"batch": args.batch, "n": args.n, "kernels": rows}, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
