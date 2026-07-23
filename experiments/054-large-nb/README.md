# Experiment 054 — large left-looking panel width

Status: **exhausted; no promotion**.

The frozen ranked incumbent uses `nb=2048` for `batch=1,n=16384`. B200
profiling showed POTRF plus triangular inversion/solve consumed 65.4% of the
route, so this experiment changed only panel width to rebalance those calls.

| Variant | Width | Baseline | Candidate | Ratio | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| V1 | 1024 | 15115.6us | 15520.6us | `0.973682x` | rejected |
| V2 | 4096 | 15074.0us | 15497.7us | `0.972580x` | rejected |

Both candidates were correct, reached their intended backend, introduced no
fallback, and had confidence intervals wholly below 1.0. Widths on both sides
of 2048 regress by about 2.7%, making 2048 a measured local optimum. The
directional prerequisite for testing 512 or 8192 was therefore absent.

No family grid, full grid, clean build, or Popcorn submission was warranted.
