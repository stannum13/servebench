# Servebench Experiment State

## Current state

- Phase: backend smoke verified; GPU baseline pending
- Latest run: `results/smoke/requests.jsonl` (mock backend, 8/8 successful)
- Smoke observation (2026-09-22): p95 TTFT 31.44 ms, p95 stream-event ITL 2.92 ms,
  p95 TPOT 2.11 ms, p95 router admission 0.37 ms, and p95 post-header TTFT 23.36 ms.
  Eight 256-token prompts produced 128 token IDs each; not GPU evidence.
- Baseline saturation run: pending NVIDIA GPU environment
- Current bottleneck: pending measurement
- Active hypothesis: none until the baseline identifies a quantitative bottleneck
- Decision: no keep/revert decision yet
- Validation hardening (2026-09-22): comparisons now constrain both throughput and completion
  rate, stream-event ITL is separated from TPOT, paired scheduler order alternates by repeat, and
  engine variants record an explicit vLLM restart/cache boundary plus full run provenance.
- External blocker: this Apple Silicon development host has no NVIDIA GPU, so the real saturation,
  KV-cache, DCGM, and scheduler evidence sequence in `README.md` remains pending on a GPU host.

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
## mock-saturation — 2026-09-22T16:16:41+00:00

- Bottleneck: saturation not observed through concurrency 4
- Hypothesis: queue, KV, and GPU telemetry at the transition identify the limiting resource
- Variable changed: concurrency
- Decision: inconclusive
## mock-scheduler — 2026-09-22T16:17:06+00:00

- Bottleneck: one or more policies produced no successful first token
- Hypothesis: SLO admission reduces p95 TTFT with at least 95% FIFO throughput
- Variable changed: scheduling policy
- Decision: inconclusive
## mock-scheduler — 2026-09-22T16:32:45+00:00

- Bottleneck: mixed-workload p95 TTFT at saturation
- Hypothesis: SLO admission reduces p95 TTFT while preserving 95% of FIFO throughput
- Variable changed: scheduling policy
- Measurement: p95 TTFT delta: -8.50 ms, 95% CI [-29.04, 3.07] ms; throughput ratio: 0.987, 95% CI [0.866, 1.100]; completion ratio: 1.000, 95% CI [1.000, 1.000]
- Decision: inconclusive
## mock-scheduler — 2026-09-22T16:42:14+00:00

- Bottleneck: mixed-workload p95 TTFT at saturation
- Hypothesis: SLO admission reduces p95 TTFT while preserving 95% of FIFO throughput and not reducing completion rate
- Variable changed: scheduling policy
- Measurement: p95 TTFT delta: 0.05 ms, 95% CI [-3.14, 1.98] ms; throughput ratio: 0.993, 95% CI [0.959, 1.035]; completion ratio: 1.000, 95% CI [1.000, 1.000]
- Decision: inconclusive
