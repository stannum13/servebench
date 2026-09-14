# Servebench Experiment State

## Current state

- Phase: backend smoke verified; GPU baseline pending
- Latest run: `results/smoke/requests.jsonl` (mock backend, 8/8 successful)
- Smoke observation: p95 TTFT 76.83 ms, p95 inter-token latency 2.40 ms; not GPU evidence
- Baseline saturation run: pending NVIDIA GPU environment
- Current bottleneck: pending measurement
- Active hypothesis: none until the baseline identifies a quantitative bottleneck
- Decision: no keep/revert decision yet

Each future loop records one baseline or candidate, one bottleneck, one hypothesis, one changed
variable, repeated confidence intervals, and a keep/revert/inconclusive decision.
