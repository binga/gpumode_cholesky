# Profiling evidence

## Evidence ladder

Use the least expensive evidence that answers the current question:

1. Run correctness and stable CUDA-event latency for every candidate.
2. Use Nsight Systems when launch, CPU submission, synchronization, transfer, stream overlap, or dependency serialization may dominate.
3. Use Nsight Compute only on dominant or suspicious kernels when memory, compute, occupancy, register, shared-memory, or instruction behavior is unresolved.

## Normalized system metrics

Map `nsys` or equivalent timeline data to these optional keys:

- `gpu_active_fraction`
- `cuda_api_time_fraction`
- `synchronization_time_fraction`
- `memcpy_time_fraction`
- `kernel_launch_count`
- `median_kernel_us`
- `short_kernel_fraction`
- `maximum_kernel_concurrency`
- `gpu_idle_gap_us`
- `cpu_idle_gap_us`

Scope all fractions to the same NVTX range or measured operation. Preserve the `.nsys-rep` and any SQLite/statistics export.

## Normalized kernel metrics

Map `ncu` or equivalent counter data to these optional keys:

- `dram_throughput_fraction`
- `compute_throughput_fraction`
- `tensor_core_utilization_fraction`
- `memory_dependency_stall_fraction`
- `barrier_stall_fraction`
- `achieved_occupancy`
- `registers_per_thread`
- `shared_memory_bytes_per_block`
- `local_memory_bytes`

Preserve the `.ncu-rep`, kernel name, launch configuration, and replay conditions. Counter names vary by GPU and tool version; perform version-specific mapping in the repository adapter, not in the skill.

## Classification limits

Treat threshold matches as evidence, not universal physical laws. A workload may be launch-bound and memory-bound at different layers. Do not use `nsys` to claim DRAM or Tensor Core saturation. Do not use `ncu` on one kernel to explain host gaps across an operation.

When metrics are missing, include the missing evidence in `coverage.missing` and recommend the cheapest next profiling step.

