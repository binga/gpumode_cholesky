"""Runs inside the Modal sandbox under Nsight Compute.

Executes ONE shape's factorization a few times so ncu can attach counters to
every kernel launch. Kept deliberately tiny: ncu serializes and replays each
kernel, so anything extra here multiplies profiling time.
"""

import os
import sys

sys.path.insert(0, "/root/reference")
sys.path.insert(0, "/root")

import torch  # noqa: E402

from reference import generate_input  # noqa: E402
from submission import custom_kernel  # noqa: E402


def main():
    batch = int(os.environ.get("NCU_BATCH", "4"))
    n = int(os.environ.get("NCU_N", "1024"))
    iters = int(os.environ.get("NCU_ITERS", "1"))

    data = generate_input(batch=batch, n=n, cond=4, seed=1234, case="dense")
    # Warm up OUTSIDE the profiled region is impossible (ncu profiles the whole
    # process), so the first iteration carries JIT/autotune noise. Kernels are
    # reported per-invocation, so later launches are the clean ones -- the
    # aggregator takes the median per kernel name.
    for _ in range(iters):
        out = custom_kernel(data)
    torch.cuda.synchronize()
    print(f"ncu_runner done batch={batch} n={n} out={tuple(out.shape)}", flush=True)


if __name__ == "__main__":
    main()
