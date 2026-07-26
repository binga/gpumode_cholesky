# Contract and adapter specification

Use JSON for the repository contract so validation does not depend on a YAML package. Start from [contract-example.json](contract-example.json).

## Contract ownership

Keep `kernel-audit.json`, the adapter, references, tests, and baseline outside the candidate-edit surface. Version them with the repository. Change them only in an explicit contract revision, then establish a new baseline.

## Adapter protocol

Set `adapter.argv` to an argument array, never a shell string. The runner replaces only:

- `{contract}` with the absolute contract path;
- `{mode}` with `quick`, `system`, `kernel`, or `full`;
- `{output}` with the requested measurement path.

Run the adapter from the contract directory. Require it to exit nonzero on infrastructure failure and write normalized measurement JSON to `{output}`. The adapter may invoke an existing benchmark harness, remote GPU runner, `nsys`, or `ncu`.

## Measurement format

Emit this minimum structure:

```json
{
  "schema_version": 1,
  "run_id": "candidate-001",
  "candidate_id": "git-or-content-hash",
  "environment": {
    "gpu": "NVIDIA B200",
    "driver": "...",
    "cuda": "...",
    "framework": "..."
  },
  "correctness": {
    "passed": true,
    "checks": [{"name": "relative_residual", "value": 0.00001, "passed": true}]
  },
  "workloads": [
    {
      "id": "64x256-fp32",
      "latency_us": 350.2,
      "cv": 0.012,
      "metrics": {
        "gpu_active_fraction": 0.86,
        "kernel_launch_count": 22,
        "median_kernel_us": 10.8,
        "short_kernel_fraction": 0.41
      }
    }
  ],
  "artifacts": {
    "nsys_report": "audit/runs/candidate/profile.nsys-rep",
    "ncu_report": "audit/runs/candidate/profile.ncu-rep"
  }
}
```

Use fractions in `[0,1]`, microseconds for time, and bytes for memory. Omit unavailable optional metrics; do not invent zero values.

## Baseline compatibility

Compare only measurements with matching workload IDs and compatible environment fingerprints. Preserve raw measurements and profiler reports. Treat the human-readable audit as a view over machine-readable artifacts, not as the source of truth.

