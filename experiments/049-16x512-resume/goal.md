# Experiment 049 resume: measure the pending `16x512` fused-panel overlay

- Exact control: ranked `#890798`, commit `f90ef909`, SHA-256 `fd3072b5...4c1`.
- Shape control: `389.408us`; strict per-shape target: `194.704us`.
- Candidate: enroll only `(batch=16, n=512)` in the already-ranked
  `_panel_fused128` path using `(tile_r=128, warps=8, merged_diag=False)`.
- First gate: paired same-process B200 target measurement with positive
  `_FUSED512_HITS`, no fallback/error, retained outputs, and official checker.
- Promotion: only a robust improvement proceeds to six families and a full
  grid. A standalone source and clean-build proof are required before Popcorn.

This resumes the previously blocked V5 without reopening the four measured and
rejected persistent architectures.
