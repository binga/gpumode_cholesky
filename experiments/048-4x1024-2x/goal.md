# Experiment 048 — `4x1024` 2x latency target

Control: exact ranked `#890659`, source SHA-256
`59558b501fb32d403667fd85a338ece7bb196f352a93685f7934bab8526d5e52`,
public score `806.036509999us`, secret score `806.396199899us`.

The target is a paired same-process B200 candidate latency at most 50% of the
exact control, with the official checker unchanged, positive intended-backend
proof, zero new fallback, and no off-target dispatch.  New paths must contain
no cuSOLVER factorization and no auxiliary CUDA queue.  Preserve every correct
frontier, but classify only a correct candidate at or above 2.00x as a winner.

Start with a fresh constituent profile (wall/device idle, launches, diagonal,
panel, trailing update, copies/gates, HBM and arithmetic floors).  Measure a
bounded ladder of six serious variants, one after another.  Root owns Popcorn
test and leaderboard submissions; this experiment does not run either gate.
