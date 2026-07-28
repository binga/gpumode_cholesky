"""Experiment 068 - tcgen05 Blackwell GEMM with ThunderKittens, benchmarked on B200.

Builds a Modal image with ThunderKittens cloned in, compiles a generalized
tcgen05 + TMA GEMM (adapted from TK's educational_b200/level_06) through
`torch.utils.cpp_extension.load_inline` at sm_100a, and benchmarks it against
cuBLAS BF16 and cuBLAS TF32 (`torch.matmul`) on the shapes that matter to this
Cholesky campaign's large-n trailing update.

Run:
    uv run --with modal -- python experiments/068-tcgen05-tk/tk_gemm_probe.py

Needs full_network for Modal.
"""

import json
import sys

import modal

TK_REPO = "https://github.com/HazyResearch/ThunderKittens.git"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "numpy", "ninja")
    .run_commands(f"git clone --depth 1 {TK_REPO} /opt/tk")
    .env({"TORCH_CUDA_ARCH_LIST": "10.0a"})
)

app = modal.App("gpumode-cholesky-tk-gemm")


# ---------------------------------------------------------------------------
# tcgen05 + TMA GEMM, generalized to rectangular M x N x K from TK's
# educational_b200/level_06.cu. A: [M,K] bf16, B: [K,N] bf16, D: [M,N] bf16.
# TILE_M=TILE_N=128, TILE_K=64 (level_06 defaults). Accumulate in TMEM (tt).
# ---------------------------------------------------------------------------
CUDA_SRC = r"""
#include <torch/extension.h>
#include "kittens.cuh"
using namespace kittens;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int TILE_K = 64;
static constexpr int NUM_WARPS = 4;
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_matmul_kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int M, int N, int K
) {
    const int wg_laneid = warpgroup::laneid();
    const int grid_n = N / TILE_N;
    const int bid_m = blockIdx.x / grid_n;
    const int bid_n = blockIdx.x % grid_n;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a_tile (&a_smem) = al.allocate<a_tile>();
    b_tile (&b_smem) = al.allocate<b_tile>();
    d_tile (&d_smem) = al.allocate<d_tile>();

    __shared__ semaphore inputs_arrived;
    __shared__ semaphore inputs_finished;
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(inputs_finished, 1, 0);
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tm_alloc{};
    d_tt_t accum;
    if (wg_laneid == 0) accum = tm_alloc.allocate<d_tt_t>(0);
    warpgroup::sync(1);

    const int num_k_iters = K / TILE_K;
    int phase = 0;
    for (int iter_k = 0; iter_k < num_k_iters; iter_k++) {
        if (threadIdx.x == 0) {
            wait(inputs_finished, phase ^ 1);
            tma::expect_bytes(inputs_arrived, sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(a_smem, A_layout, {bid_m, iter_k}, inputs_arrived);
            tma::load_async(b_smem, B_layout, {iter_k, bid_n}, inputs_arrived);
        }
        wait(inputs_arrived, phase);
        phase ^= 1;
        if (wg_laneid == 0) {
            if (iter_k == 0) mm_AB (accum, a_smem, b_smem, inputs_finished);
            else             mma_AB(accum, a_smem, b_smem, inputs_finished);
        }
    }

    if (wg_laneid == 0) kittens::detail::tcgen05::commit<1>(compute_done);
    wait(compute_done, 0);

    rt_bf<TILE_M / 4, TILE_N> d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();
    warpgroup::sync(1);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);
    if (wg_laneid == 0) {
        tma::store_async(D_layout, d_smem, {bid_m, bid_n});
        tma::store_async_read_wait();
    }
}

torch::Tensor tk_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "cuda only");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "B must be bf16");
    A = A.contiguous(); B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "K mismatch");
    TORCH_CHECK(M % TILE_M == 0 && N % TILE_N == 0 && K % TILE_K == 0,
                "M,N must be multiples of 128 and K a multiple of 64");
    auto C = torch::empty({M, N}, A.options());

    a_gl A_layout{reinterpret_cast<bf16*>(A.data_ptr()), nullptr, nullptr,
                  (unsigned long)M, (unsigned long)K};
    b_gl B_layout{reinterpret_cast<bf16*>(B.data_ptr()), nullptr, nullptr,
                  (unsigned long)K, (unsigned long)N};
    d_gl D_layout{reinterpret_cast<bf16*>(C.data_ptr()), nullptr, nullptr,
                  (unsigned long)M, (unsigned long)N};

    int grid = (M / TILE_M) * (N / TILE_N);
    int smem_size = MAX_SHARED_MEMORY - 1024;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(tk_matmul_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        cfg = true;
    }
    tk_matmul_kernel<<<grid, NUM_THREADS, smem_size>>>(A_layout, B_layout, D_layout, M, N, K);
    cudaError_t st = cudaGetLastError();
    TORCH_CHECK(st == cudaSuccess, cudaGetErrorString(st));
    return C;
}
"""


@app.function(image=image, gpu="B200", timeout=1200)
def run():
    import os
    import torch
    from torch.utils.cpp_extension import load_inline

    info = {"cuda": torch.version.cuda, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0)}
    try:
        mod = load_inline(
            name="tk_gemm_ext",
            cpp_sources="torch::Tensor tk_gemm(torch::Tensor, torch::Tensor);",
            cuda_sources=CUDA_SRC,
            functions=["tk_gemm"],
            extra_include_paths=["/opt/tk/include", "/opt/tk/prototype"],
            extra_cuda_cflags=[
                "-std=c++20", "-O3", "--use_fast_math",
                "--expt-extended-lambda", "--expt-relaxed-constexpr",
                "-forward-unknown-to-host-compiler",
                "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing",
                "-DNDEBUG", "-DKITTENS_SM100",
            ],
            extra_ldflags=["-lcuda", "-lcudadevrt"],
            verbose=True,
        )
    except Exception as exc:
        import traceback
        return {"info": info, "build_ok": False,
                "error": traceback.format_exc()[-4000:]}

    dev = torch.device("cuda")

    def bench(fn, iters=30, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1e3  # us

    shapes = [
        (4096, 4096, 4096),      # TK validated square (de-risk + headline)
        (16384, 16384, 512),     # large trailing, thin K
        (8192, 8192, 2048),      # mid-large trailing
        (16128, 128, 16128),     # tall panel-apply-ish (M mult128, N=128, K mult64)
    ]

    rows = []
    for (M, N, K) in shapes:
        A = (torch.rand(M, K, device=dev) - 0.5).to(torch.bfloat16)
        B = (torch.rand(K, N, device=dev) - 0.5).to(torch.bfloat16)

        # correctness vs true fp32
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        ref = A.float() @ B.float()
        torch.backends.cuda.matmul.allow_tf32 = prev

        try:
            C = mod.tk_gemm(A, B)
            torch.cuda.synchronize()
            err = float((C.float() - ref).abs().max().item())
            rel = float(((C.float() - ref).abs().max()
                         / ref.abs().max().clamp_min(1e-6)).item())
            tk_us = bench(lambda: mod.tk_gemm(A, B))
        except Exception as exc:
            import traceback
            rows.append({"shape": [M, N, K], "tk_ok": False,
                         "error": traceback.format_exc()[-1500:]})
            continue

        # cuBLAS bf16
        bf16_us = bench(lambda: torch.matmul(A, B))
        # cuBLAS tf32 (what the shipped 1x16384 trailing uses)
        Af = A.float(); Bf = B.float()
        torch.backends.cuda.matmul.allow_tf32 = True
        tf32_us = bench(lambda: torch.matmul(Af, Bf))
        torch.backends.cuda.matmul.allow_tf32 = prev

        flops = 2.0 * M * N * K
        rows.append({
            "shape": [M, N, K],
            "tk_ok": True,
            "tk_us": round(tk_us, 2),
            "tk_tflops": round(flops / tk_us / 1e6, 1),
            "cublas_bf16_us": round(bf16_us, 2),
            "cublas_bf16_tflops": round(flops / bf16_us / 1e6, 1),
            "cublas_tf32_us": round(tf32_us, 2),
            "cublas_tf32_tflops": round(flops / tf32_us / 1e6, 1),
            "tk_vs_tf32": round(tf32_us / tk_us, 3),
            "tk_vs_bf16": round(bf16_us / tk_us, 3),
            "abs_err": round(err, 4),
            "rel_err": round(rel, 6),
        })
        del A, B, Af, Bf, ref, C
        torch.cuda.empty_cache()

    return {"info": info, "build_ok": True, "rows": rows}


@app.local_entrypoint()
def main():
    out = run.remote()
    print("RESULT_JSON:" + json.dumps(out))
    print(json.dumps(out, indent=2), file=sys.stderr)
