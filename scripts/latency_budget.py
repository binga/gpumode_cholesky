"""Decompose measured per-kernel latency into its constituent floors.

Consumes a `shapediag` JSON and, for every kernel, computes the two hardware
floors its work implies and the residual that neither explains:

    measured = max(dram_floor, math_floor) + residual

    dram_floor  : compulsory bytes / HBM bandwidth      -> data-movement bound
    math_floor  : algorithmic FLOPs / the relevant peak -> math bound
    residual    : everything else = exposed instruction latency, low
                  occupancy, serialization, launch ramp

The point of the split is that the three buckets have *different* cures.
A dram-bound kernel wants fewer passes or a smaller dtype. A math-bound
kernel wants a faster pipe (tf32 -> fp8) or fewer flops. A residual-bound
kernel wants more parallelism or a shorter dependent-instruction chain --
and is completely deaf to precision work.

Work models are analytic, from the algorithm each kernel implements. They
are lower bounds: real kernels re-read tiles. So `residual` is an UPPER
bound on exposed latency, and a kernel is only confidently residual-bound
when the residual dominates by a wide margin. Kernels whose work model is
unknown are reported as `?` rather than guessed.

Peaks (B200): HBM3e ~7.7 TB/s measured-attainable; tf32 tensor ~450 TFLOP/s;
fp32 FMA (non-tensor, what cuSOLVER's panel code and our micro kernels run
on) ~80 TFLOP/s. Adjust BW/PEAK below if better numbers are measured.
"""

import json
import sys
from pathlib import Path

BW = 7.7e12          # B/s
PEAK_TF32 = 4.5e14   # FLOP/s, tensor core
PEAK_FP32 = 8.0e13   # FLOP/s, plain FMA pipe

F = 4  # bytes per fp32


def _potrf_flops(n):
    return n ** 3 / 3.0


def kernel_work(name, batch, n, calls):
    """(bytes, flops, peak) per CALL, or None when the model is unknown.

    NB=128 blocked split32 path: each outer step factors a 128-wide panel,
    driven by 32-wide micro steps. Per-call work is the whole batch since
    one launch covers all matrices.
    """
    nb = 128
    steps = max(n // nb, 1)

    if name == "_micro_potrf_gj32":
        # One 32x32 diagonal-block factorization per matrix, per call.
        return batch * 32 * 32 * F * 2, batch * _potrf_flops(32), PEAK_FP32

    if name == "_chol32_rank2_kernel":
        # Whole 32x32 factorization, one call, entire batch.
        return batch * 32 * 32 * F * 2, batch * _potrf_flops(32), PEAK_FP32

    if name in ("_panel_apply32", "_panel_inner32", "_panel_inner32_subtile64"):
        # Triangular solve / rank-32 update of the panel below the diagonal
        # block. Averaged over the run: mean panel height ~ n/2.
        rows = max(n // 2, 32)
        return batch * rows * 32 * F * 2, 2.0 * batch * rows * 32 * 32, PEAK_TF32

    if name == "_trailing_nb":
        # Trailing SYRK: A(rows x nb) @ A^T, averaged rows ~ n/2.
        rows = max(n // 2, nb)
        return batch * rows * rows * F * 2, 2.0 * batch * rows * rows * nb, PEAK_TF32

    if name.startswith("void kernel<getrf_wo_pivot_params_"):
        # cuSOLVER unpivoted factorization of one nb-block column; over the
        # whole run these calls sum to the full n^3/3, so amortize per call.
        per = _potrf_flops(n) / max(calls, 1)
        return batch * n * n * F * 2 / max(calls, 1), batch * per, PEAK_FP32

    if "trsm" in name:
        # Triangular solve against the factored diagonal block.
        rows = max(n // 2, 32)
        return batch * rows * nb * F * 2, 1.0 * batch * rows * nb * nb, PEAK_TF32

    if "gemm" in name or name.startswith("nvjet_"):
        # Schur-complement GEMM. Square-ish tile of side ~n/2 by nb.
        rows = max(n // 2, nb)
        return batch * rows * rows * F * 2, 2.0 * batch * rows * rows * nb, PEAK_TF32

    if "triu_tril" in name or "elementwise" in name or "CatArrayBatch" in name:
        # Pure streaming: touch the matrix once in, once out.
        return batch * n * n * F * 2 / max(calls, 1), 0.0, PEAK_FP32

    if "Memcpy" in name or "Memset" in name:
        return batch * n * n * F * 2 / max(calls, 1), 0.0, PEAK_FP32

    return None


def classify(measured_us, dram_us, math_us):
    """A floor ABOVE the measured time means the work model is wrong -- a real
    kernel cannot beat its own compulsory traffic or flop count. Say so
    instead of laundering the error as 'at-floor', which would silently
    exclude the kernel from every bucket and shrink the totals."""
    floor = max(dram_us, math_us)
    residual = measured_us - floor
    if floor > measured_us * 1.02:
        return "model-invalid", 0.0, 0.0
    if floor <= 0:
        return "residual", residual, 1.0
    frac = residual / measured_us
    if frac < 0.35:
        kind = "dram" if dram_us >= math_us else "math"
    else:
        kind = "residual"
    return kind, residual, frac


def main(path):
    data = json.loads(Path(path).read_text())
    print(f"device: {data.get('device','?')}   source: {path}")
    print(
        "\nbuckets: dram=data movement | math=arithmetic pipe | "
        "residual=exposed latency/occupancy/serialization\n"
    )
    totals = {"dram": 0.0, "math": 0.0, "residual": 0.0, "?": 0.0,
              "model-invalid": 0.0}
    launch_total = 0.0

    for s in data["shapes"]:
        batch, n = s["batch"], s["n"]
        print(f"=== {batch}x{n}  wall={s['wall_us']:.0f}us "
              f"device={s['device_us']:.0f}us idle={s['idle_us']:.0f}us "
              f"({s['idle_pct']:.1f}%)  launches={s['kernel_launches']}")
        launch_total += max(s["idle_us"], 0.0)
        print(f"    {'kernel':<44} {'meas':>9} {'dram':>8} {'math':>8} "
              f"{'resid':>9}  bound")
        for k in s["kernels"]:
            name = k["kernel"]
            work = kernel_work(name, batch, n, k["calls"])
            meas = k["us"]
            if work is None:
                totals["?"] += meas
                print(f"    {name[:44]:<44} {meas:>9.1f} {'?':>8} {'?':>8} "
                      f"{'?':>9}  ?")
                continue
            per_call_bytes, per_call_flops, peak = work
            dram = per_call_bytes / BW * 1e6 * k["calls"]
            math = per_call_flops / peak * 1e6 * k["calls"]
            kind, resid, _ = classify(meas, dram, math)
            totals[kind] += meas
            print(f"    {name[:44]:<44} {meas:>9.1f} {dram:>8.1f} {math:>8.1f} "
                  f"{resid:>9.1f}  {kind}")
        print()

    grand = sum(totals.values()) + launch_total
    print("=" * 78)
    print("GRID-WIDE DEVICE TIME BY BUCKET")
    for k in ("residual", "dram", "math", "?", "model-invalid"):
        v = totals[k]
        print(f"  {k:<10} {v:>10.0f}us  {v/grand*100:>5.1f}%")
    print(f"  {'launch':<10} {launch_total:>10.0f}us  "
          f"{launch_total/grand*100:>5.1f}%   (wall - device, i.e. gaps)")
    print(f"  {'TOTAL':<10} {grand:>10.0f}us")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/exp036-diag-all.json")
