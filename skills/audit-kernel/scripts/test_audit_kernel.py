#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import audit_kernel


def contract():
    return {
        "schema_version": 1,
        "name": "test-kernel",
        "workloads": [{"id": "small", "weight": 0.7}, {"id": "large", "weight": 0.3}],
        "measurement": {"maximum_cv": 0.03},
        "acceptance": {
            "minimum_improvement_fraction": 0.02,
            "maximum_case_regression_fraction": 0.05,
            "require_same_environment": True,
        },
        "integrity": {"protected_paths": [], "expected_sha256": {}},
    }


def measurement(small, large, *, passed=True, metrics=None):
    return {
        "schema_version": 1,
        "environment": {"gpu": "test-gpu", "cuda": "test-cuda"},
        "correctness": {"passed": passed, "checks": []},
        "workloads": [
            {"id": "small", "latency_us": small, "cv": 0.01, "metrics": metrics or {}},
            {"id": "large", "latency_us": large, "cv": 0.01, "metrics": {}},
        ],
    }


class AuditKernelTests(unittest.TestCase):
    def test_contract_validation(self):
        self.assertEqual(audit_kernel.validate_contract(contract()), [])

    def test_accepts_weighted_improvement(self):
        result, code = audit_kernel.evaluate(contract(), measurement(100, 200), measurement(90, 180))
        self.assertEqual(code, audit_kernel.EXIT_OK)
        self.assertEqual(result["verdict"], "accepted")
        self.assertAlmostEqual(result["objective"]["improvement_fraction"], 0.1)

    def test_rejects_case_regression(self):
        result, code = audit_kernel.evaluate(contract(), measurement(100, 200), measurement(80, 212))
        self.assertEqual(code, audit_kernel.EXIT_REGRESSION)
        self.assertEqual(result["verdict"], "rejected")
        self.assertEqual([row["id"] for row in result["regressions"]], ["large"])

    def test_correctness_fails_closed(self):
        result, code = audit_kernel.evaluate(contract(), measurement(100, 200), measurement(80, 180, passed=False))
        self.assertEqual(code, audit_kernel.EXIT_CORRECTNESS)
        self.assertEqual(result["verdict"], "correctness_failed")

    def test_classifies_system_and_kernel_evidence(self):
        metrics = {
            "gpu_active_fraction": 0.55,
            "cuda_api_time_fraction": 0.25,
            "short_kernel_fraction": 0.7,
            "maximum_kernel_concurrency": 1,
            "kernel_launch_count": 30,
            "dram_throughput_fraction": 0.82,
            "achieved_occupancy": 0.2,
        }
        analysis = audit_kernel.classify_workload(measurement(100, 200, metrics=metrics)["workloads"][0], contract())
        labels = {gap["label"] for gap in analysis["gaps"]}
        self.assertTrue({"host_launch", "dependency_synchronization", "memory", "resource_pressure"} <= labels)

    def test_cli_validate_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path = root / "kernel-audit.json"
            measurement_path = root / "measurement.json"
            contract_path.write_text(json.dumps(contract()))
            measurement_path.write_text(json.dumps(measurement(100, 200)))
            self.assertEqual(audit_kernel.measurement_errors(audit_kernel.load_json(measurement_path), audit_kernel.contract_or_raise(contract_path)), [])


if __name__ == "__main__":
    unittest.main()
