---
name: audit-kernel
description: Audit and compare GPU kernel implementations with a repository-owned, machine-verifiable contract. Use when Codex needs to establish a kernel baseline, validate correctness and latency measurements, analyze normalized CUDA-event, torch.profiler, Nsight Systems (nsys), or Nsight Compute (ncu) evidence, classify launch/dependency/memory/compute/resource gaps, rank optimization opportunities, or decide whether an optimization candidate should be accepted.
---

# Audit Kernel

Treat the repository contract and evaluator as the oracle. Treat source changes as candidates. Never infer a bottleneck from code inspection alone.

## Choose the operation

- **Initialize**: Draft `kernel-audit.json` and a project adapter when no contract exists. Ask the user to approve workloads, correctness gates, weights, and regression limits before freezing a baseline.
- **Baseline**: Run the approved contract and save a known-good measurement.
- **Audit**: Measure a candidate, compare it with the compatible baseline, classify evidenced gaps, and rank one-step experiments.
- **Inspect**: Classify an existing normalized measurement without changing code.

Read [references/contract.md](references/contract.md) when creating or changing a contract or adapter. Read [references/profiling.md](references/profiling.md) when collecting or interpreting profiler evidence.

## Preserve evaluator integrity

Do not modify the contract, adapter, correctness reference, benchmark, protected paths, or baseline in the same candidate optimization. If any must change, create a separately reviewed contract revision and establish a new baseline.

Fail closed when:

- correctness is absent or fails;
- a required workload is absent;
- measurement noise exceeds policy after allowed retries;
- environments differ when equality is required;
- protected-file hashes differ;
- profiler evidence required for a claim is unavailable.

Never weaken tolerances, remove workloads, discard regressions, shorten measurement below the contract, or special-case benchmark inputs to obtain an acceptance verdict.

## Execute an audit

1. Locate `kernel-audit.json`. Validate it:

   ```bash
   python3 <skill-dir>/scripts/audit_kernel.py validate-contract --contract kernel-audit.json
   ```

2. Run the repository adapter through the deterministic runner. Use `quick` on every iteration; use `system` for `nsys`, `kernel` for `ncu`, and `full` when both are needed:

   ```bash
   python3 <skill-dir>/scripts/audit_kernel.py run \
     --contract kernel-audit.json \
     --mode quick \
     --baseline audit/baseline.json \
     --measurement audit/candidate.json \
     --output audit/result.json
   ```

3. If measurements already exist, compare them without rerunning:

   ```bash
   python3 <skill-dir>/scripts/audit_kernel.py evaluate \
     --contract kernel-audit.json \
     --baseline audit/baseline.json \
     --candidate audit/candidate.json \
     --output audit/result.json
   ```

4. Report the verdict, correctness gate, aggregate latency change, per-workload regressions, evidence coverage, ranked gaps, and artifact paths. Separate facts from hypotheses.

5. Propose one falsifiable experiment for the highest-value evidenced gap. State the observed metric, intended change, predicted metric movement, affected workloads, and correctness/resource risk.

6. After a code change, rerun correctness and the entire workload matrix. Re-profile after an accepted change because the bottleneck may move.

## Interpret the evidence ladder

- Use benchmark latency to decide whether a candidate improved.
- Use `nsys` timeline evidence to diagnose host/launch overhead, gaps, synchronization, transfers, streams, and serialization.
- Use `ncu` counters to diagnose memory, compute, occupancy, register, shared-memory, and instruction-level limits inside a kernel.
- Use aggregated `torch.profiler` kernel time/count as preliminary launch evidence, not proof of CPU gaps or hardware saturation.

Emit multiple gap labels when supported. Report `insufficient_evidence` rather than forcing a classification.

## Exit codes

- `0`: valid and accepted, or valid inspection without a comparison
- `2`: correctness failure
- `3`: insufficient improvement or excessive regression
- `4`: invalid or noisy measurement
- `5`: adapter, environment, or profiler failure
- `6`: invalid contract

