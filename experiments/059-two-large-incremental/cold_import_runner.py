"""Measure a clean submission import and report extension readiness."""

import json
import sys
import time

sys.path.insert(0, "/root/reference")

runner_start = time.perf_counter()
torch_start = time.perf_counter()
import torch

torch_seconds = time.perf_counter() - torch_start
submission_start = time.perf_counter()
import submission

submission_seconds = time.perf_counter() - submission_start
ready = {
    name: getattr(submission, name, None) is not None
    for name in ("_CUDA32", "_CUDA64", "_CUDA128")
}
errors = {
    name: value
    for name in ("_CUDA32_ERROR", "_CUDA64_ERROR", "_CUDA128_ERROR")
    if (value := getattr(submission, name, None))
}
result = {
    "mode": "coldimport",
    "passed": all(ready.values()) and not errors,
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "torch_import_seconds": torch_seconds,
    "submission_import_seconds": submission_seconds,
    "runner_init_seconds": time.perf_counter() - runner_start,
    "readiness": ready,
    "load_errors": errors,
}
print("RESULT_JSON:" + json.dumps(result), flush=True)
