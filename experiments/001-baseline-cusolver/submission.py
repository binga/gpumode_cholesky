#!POPCORN leaderboard cholesky
#!POPCORN gpu B200

"""Experiment 001 — cuSOLVER baseline.

Plain batched Cholesky via torch/cuSOLVER. Correct across all input families;
establishes the reference geomean (~2080us, ranked submission #876988).
"""

import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape
    return torch.linalg.cholesky_ex(data, check_errors=False).L
