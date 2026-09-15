# Servebench Experiment State

## Current state

- Phase: backend smoke verified; GPU baseline pending
- Latest run: `results/smoke/requests.jsonl` (mock backend, 8/8 successful)
- Smoke observation (2026-09-15): p95 TTFT 27.90 ms, p95 inter-token latency 2.91 ms;
  p95 router admission 0.54 ms and p95 post-header TTFT 21.88 ms. Eight exact
  256-token prompts produced 128 timed token IDs each; not GPU evidence.
- Baseline saturation run: pending NVIDIA GPU environment
- Current bottleneck: pending measurement
- Active hypothesis: none until the baseline identifies a quantitative bottleneck
- Decision: no keep/revert decision yet

Each future loop records one baseline or candidate, one bottleneck, one hypothesis, one changed
variable, repeated confidence intervals, and a keep/revert/inconclusive decision.

## mock-saturation — 2026-09-14

- Bottleneck: mock service transition near concurrency 4; client-side queueing fell as concurrency rose
- Hypothesis: the experiment pipeline can locate a transition and preserve run-scoped telemetry
- Variable changed: concurrency (1, 2, 4; three repeats each)
- Result: 108/108 requests succeeded; suggested refinement concurrency is 3
- Decision: inconclusive for GPU performance because the backend is deterministic CPU mock

## mock-scheduler — 2026-09-14

- Bottleneck: no GPU/KV bottleneck exists in the mock backend
- Hypothesis: the controlled policy selector produces paired FIFO/SLO artifacts and restores safely
- Variable changed: scheduling policy (three paired repeats at concurrency 4)
- Result: comparison artifact generated; mock evidence is explicitly excluded from optimization claims
- Decision: inconclusive
