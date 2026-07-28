# Experiment 068 — tcgen05 Blackwell GEMM with ThunderKittens

**Goal.** User directive: "write a tcgen05 Blackwell-ready kernel" / "use
thunderkittens". This is levers.md **lever #3** (CUTLASS/TK fused tensor-core
kernel), marked high-effort and untried. Build a real tcgen05 GEMM on
ThunderKittens and measure whether it beats the shipped vendor trailing GEMM
(cuBLAS TF32 / MXFP8) on this campaign's large-n shapes.

**Prior (stated before spending).** TK's best published B200 GEMM is BF16
1285–1540 TFLOPs; the shipped trailing already runs cuBLAS TF32 / MXFP8 at
~3466 TFLOP/s (exp 064). A drop-in GEMM win is unlikely. The one live
hypothesis worth a measurement: TK BF16 > cuBLAS **TF32** on the `1×16384`
TF32-trailing path, *if* BF16 survives the loose large-n accuracy gate.

## Toolchain (proven working — reusable)

The Modal image `nvidia/cuda:13.0.0-devel-ubuntu24.04` + `torch 2.13.0+cu130`
compiles ThunderKittens 2.0 through `torch.utils.cpp_extension.load_inline`:

- clone TK to `/opt/tk`, pass `-I/opt/tk/include -I/opt/tk/prototype`
- `-gencode arch=compute_100a,code=sm_100a` (set via `TORCH_CUDA_ARCH_LIST=10.0a`)
- `-std=c++20 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr -DKITTENS_SM100`
- `extra_ldflags=["-lcuda", "-lcudadevrt"]` — TK uses the CUDA **driver** API
  (`cuGetErrorString`), which `-lcudart` alone does not resolve.
- Two mechanical fixes were needed: fully-qualify `kittens::detail::tcgen05::commit`
  (`detail` is ambiguous once `torch/extension.h` is in scope), and the driver-lib
  link flag above.

Kernel: `educational_b200/level_06.cu` (tcgen05 + TMA, single warpgroup issuer,
FP32 accumulation in TMEM `tt<float,128,128>`), generalized to rectangular
M×N×K with a torch pybind. Correct on B200 (rel_err 0.2–0.3% vs fp32).

## Result — REJECTED

`results/068-tk-gemm-probe.json`. B200, 30-iter CUDA-event timing.

| shape (M,N,K) | TK tcgen05 | cuBLAS BF16 | cuBLAS TF32 | TK/tf32 | TK/bf16 | rel_err |
|---|---:|---:|---:|---:|---:|---:|
| 4096³ | 292.9 TF | 1546.1 | 812.3 | 0.361× | 0.189× | 0.22% |
| 16384×16384×512 | 333.7 | 1329.9 | 709.6 | 0.470× | 0.251× | 0.29% |
| 8192×8192×2048 | 330.1 | 1587.8 | 874.7 | 0.377× | 0.208× | 0.30% |
| 16128×128×16128 | 230.9 | 827.7 | 413.5 | 0.558× | 0.279× | 0.22% |

The 4096³ number (**292.9 TFLOPs**) matches TK's own published figure for
level_06 (293 TFLOPs), so the kernel is faithful — it is the *naive* tcgen05
kernel, not the pipelined / warp-specialized / 2-CTA production one.

## Why this closes lever #3, not just this kernel

1. **The naive TK kernel loses 2–5× to cuBLAS.** Porting up the TK ladder
   (level_07 pipelined 731 TF → level_09 2-CTA 1285 TF → production `bf16_b200`
   1540 TF) would at best reach TK's published ceiling of ~1540 TFLOPs.
2. **TK's ceiling only *ties* cuBLAS BF16**, which this run measured at
   1546–1588 TFLOPs — and cuBLAS BF16 is already reachable from Python with
   `torch.matmul` on bf16 tensors. TK adds nothing over the vendor bf16 path for
   a drop-in trailing GEMM.
3. **Neither bf16 path beats the shipped trailing choices.** The `1×16384` path
   ships TF32 because TF32 *won on accuracy* there (exp 006: TF32 1.60× vs
   BF16x9 1.15×; exp 024: FP8 0.997×, quantization overhead erased the gain).
   The `1×32768` path already ships MXFP8 at 3466 TFLOP/s — ~2.2× the cuBLAS
   BF16 measured here. So a bf16-class GEMM is either less accurate (16384) or
   slower (32768) than what already ships.
4. **The GEMM is not the bottleneck.** This repo's load-bearing finding
   (`docs/lessons.md`, `docs/levers.md` Part 2) is that the *serial diagonal /
   panel factorization* dominates the large shapes (diagonal potrf is 59.6% of
   `1×16384`, 46.9% of `1×32768`). A faster trailing GEMM primitive — TK or
   otherwise — cannot move a geomean that is gated by the serial factor. This is
   exactly what `kernels_last30days.md` predicts inverted: tcgen05/TMEM/CTA-pairs
   are GEMM-dataflow levers, and this problem is not GEMM-bound.

## Not pursued (and why)

- **Porting level_09 / production `bf16_b200`.** Would confirm ≈cuBLAS BF16;
  cannot change the verdict (point 2/3). Not worth the B200 spend.
- **TK *fused* Cholesky megakernel** (panel + trailing + inverse in one
  persistent launch, TMEM accumulators). This is the only angle where TK could
  add value over vendor GEMMs — but it is the persistent/cooperative family,
  which is **5-for-5 negative** here (`docs/experiments.md`: exps 028/038/040/
  048/049, best 0.697×), and the serial diagonal still cannot be tensor-cored.

## Reusable takeaways

- The `load_inline` + TK recipe above works and is banked for any future
  tensor-core kernel work on this Modal image.
- For this campaign, **stop treating the trailing GEMM as a lever.** It is
  vendor-near-peak and precision-saturated. Remaining geomean lives in the
  serial factor (secret-split-blocked at the margin, exps 063/065) and in
  per-call overhead (`Ov`, lever #7 — the board's clearest untried steer).
