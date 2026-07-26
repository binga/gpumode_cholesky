# Experiments

Each optimization attempt is a numbered folder `NNN-slug/`. The repo-root
`submission.py` always holds the **current best** (the last *adopted* experiment).
Each experiment folder is a self-contained snapshot so results stay reproducible
and auditable.

## Convention

`experiments/NNN-slug/` contains:
- `submission.py` — the exact submission used for this experiment.
- `notes.md` — hypothesis, what changed, Modal verify/benchmark results (geomean +
  per-shape), correctness status across all families, popcorn submission id (if
  submitted), verdict (**adopted** / **rejected**), and approx Modal spend.
- `benchmark.json` — the Modal per-shape benchmark for this experiment (optional).

## Workflow (one commit per experiment)

1. Create the next `experiments/NNN-slug/` folder.
2. Iterate the inner loop (CPU check → Modal verify → Modal benchmark).
3. Write `notes.md` with results and verdict.
4. If adopted: copy the winning `submission.py` to repo root.
5. If it beats the last ranked result on Modal: `popcorn submit --mode test` then
   `--mode leaderboard`; record the id.
6. **`git commit`** the experiment folder (+ root `submission.py`/`journal.md`/`README.md`
   if changed) with a message like `exp NNN: <slug> — <one-line result>`.

## See also

- `docs/workflow.md` — the outer and inner loop drawn out, with gate ordering.
- `docs/experiment-matrix.md` — every experiment × the strategies it used, with
  the latency it moved. **Add a row there when an experiment closes.**
- `docs/lever-ladder.md` — the standing backlog of levers to pick from, ported
  from the QR project, with measured ROI estimates.

## Log

The per-experiment log and the strategy matrix now live in
**`docs/experiments.md`** — one row per experiment carrying both the levers it
used and the latency it moved. Add the row there when an experiment closes.
