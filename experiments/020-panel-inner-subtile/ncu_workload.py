"""Launch one representative panel-inner kernel for Nsight Compute."""

import importlib.util
import sys

import torch
import triton

sys.path.insert(0, "/root/reference")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    mode = sys.argv[1]
    path = (
        "/root/baseline_submission.py"
        if mode == "baseline"
        else "/root/candidate_submission.py"
    )
    module = load_module(f"{mode}_submission", path)
    batch, n = 4, 1024
    k, width, remaining = 0, 96, n - 32
    output = torch.randn((batch, n, n), device="cuda", dtype=torch.float32)
    source = output.clone()
    if mode == "baseline":
        module._panel_inner32[(triton.cdiv(remaining, 128), batch)](
            output,
            source,
            n=n,
            k=k,
            width=width,
            remaining=remaining,
            PREC="tf32x3",
            TILE_R=128,
            FIRST=False,
            num_warps=4,
        )
    else:
        ntiles_c = triton.cdiv(width, 64)
        module._panel_inner32_subtile64[
            (triton.cdiv(remaining, 64) * ntiles_c, batch)
        ](
            output,
            source,
            n=n,
            k=k,
            width=width,
            remaining=remaining,
            PREC="tf32x3",
            NTILES_C=ntiles_c,
            FIRST=False,
            num_warps=4,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
