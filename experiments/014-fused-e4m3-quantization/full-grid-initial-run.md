# Initial full-grid transport/warmup failure

The first experiment-014 full-grid attempt on 2026-07-17 completed the remote
runner but did not produce a local JSON artifact. The runner emitted its entire
result as one `RESULT_JSON:` line; the Modal stdout transport truncated that
line at about 64 KiB, and the local driver raised
`JSONDecodeError: Unterminated string`.

The visible prefix also exposed a measurement defect: identical off-target code
had one-time candidate outliers after only one to three warmups. For example,
`4096x32` settled near 71–73 us for both modules but had a 559 us candidate
sample, and `1x16384` settled near 16.1 ms but began with 69.7 and 21.1 ms
candidate samples. Those initialization samples made the arithmetic-mean
off-target gate fail despite identical source in those dispatch regions.

The corrective harness change does not relax a threshold or discard timed
samples. It raises per-module per-shape warmups to at least four before timing,
keeps every arithmetic-mean sample, retains the 1.03x off-target limit, and
chunks the JSON transport into 32 KiB lines with an exact reconstructed-length
check. The run is repeated from scratch; this incomplete console prefix is not
used as promotion evidence.

The second attempt proved that the transport ceiling was lower than 32 KiB for
at least one stdout chunk, so the local exact-length check correctly rejected
the reconstructed 60,069-character payload versus 68,261 expected. That run
also isolated the outlier mechanism: `_paired_shape` accidentally retained the
last warmup output in addition to the latest timed output for each backend. The
first reversed-order candidate invocation therefore temporarily required one
extra full output allocation; after that allocation, both modules returned to
identical off-target steady-state latency. The final correction uses 8 KiB
transport chunks and releases only the warmup reference before timing. It does
not release any timed output before validation or remove any timed sample.
