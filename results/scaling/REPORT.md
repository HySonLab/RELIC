# Supplementary size-generalization pilot

One graph per topology and size, three trained model seeds. No exact optimum, no MIS experiments, and no confidence-interval claim. Single-thread CPU timings are descriptive; these are different instances from the main evaluation. The models were not retrained for this pilot.

| Nodes | Method | Mean normalized energy | Mean seconds |
|---|---|---:|---:|---:|
| 100 | relic_greedy | -0.7664 | 0.0528 |
| 100 | greedy_contraction | -0.7712 | 0.0023 |
| 250 | relic_greedy | -0.7656 | 0.2135 |
| 250 | greedy_contraction | -0.7701 | 0.0123 |
| 500 | relic_greedy | -0.7502 | 0.7141 |
| 500 | greedy_contraction | -0.7599 | 0.0500 |
| 1000 | relic_greedy | -0.7376 | 2.5795 |
| 1000 | greedy_contraction | -0.7538 | 0.2065 |

This pilot demonstrates executable inference at larger sizes, not reproduction of the paper's 4,000-node quality or GPU speedup claims. Greedy contraction is deterministic and repeated across model seeds only for a paired table layout. Raw data and reconstructed spins are included alongside this report.
