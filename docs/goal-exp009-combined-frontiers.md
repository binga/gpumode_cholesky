# Goal — experiment 009 combined shape frontiers

Integrate the independently measured exact-shape frontiers for `256x128`,
`16x512`, and `8x2048` into the ranked experiment-008 submission without
regressing any other shape or input family.

Promotion gates:

1. CPU property checks pass 10/10 and source-policy scans are clean.
2. Paired Modal B200 evidence confirms each target path is correct and faster
   than the experiment-008 path in the same sandbox.
3. Modal verification passes every input family and the full 15-shape benchmark
   shows a lower geometric mean with no material off-target regression.
4. Popcorn test mode passes 17/17.
5. Only then submit once in leaderboard mode and adopt only if the completed
   result improves ranked latency.
