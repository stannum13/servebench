# Servebench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, instrumented vLLM inference benchmark with an SLO-aware admission controller, transition-focused experiment sweeps, machine-readable results, and an evidence-based report workflow.

**Architecture:** A FastAPI router exposes an OpenAI-compatible chat/completions proxy, samples vLLM/Prometheus state, and applies either FIFO or an understandable SLO policy. An asyncio load generator produces deterministic workload classes and writes request-level JSONL plus summary JSON; an experiment runner repeats controlled comparisons and produces confidence intervals, Parquet, plots, and Markdown reports. Docker Compose connects these services to vLLM, Prometheus, Grafana, and NVIDIA DCGM, while a mock backend keeps development and smoke tests GPU-independent.

**Tech Stack:** Python 3.12, FastAPI, httpx, prometheus-client, Pydantic, PyYAML, pandas/pyarrow, matplotlib, pytest, Docker Compose, vLLM, Prometheus, Grafana, NVIDIA DCGM exporter.

## Global Constraints

- Use vLLM as the primary backend.
- Default to an ungated ~7–8B open model and make `MODEL` configurable.
- Workloads are deterministic for a supplied seed.
- Every measured run writes machine-readable JSONL and Parquet-compatible data.
- Sweeps locate transitions instead of executing a giant Cartesian grid.
- Optimization claims require repeated measurements and confidence intervals.
- The scheduling policy observes queue depth, prompt length, and KV pressure and can admit, delay, reject, or route to another worker.
- Compare the SLO-aware policy with FIFO under a defined mixed workload and throughput constraint.
- Maintain `BENCH_STATE.md` with baseline, bottleneck, hypothesis, one-variable change, confidence interval, and keep/revert decision.

---

### Task 1: Project skeleton and typed configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `configs/default.yaml`
- Create: `src/servebench/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> BenchConfig`, with nested router, workload, experiment, and backend settings.

- [x] Write tests proving defaults select `Qwen/Qwen2.5-7B-Instruct`, environment variables override the model and backend URL, invalid workload weights are rejected, and configured seeds are stable.
- [x] Run `pytest tests/test_config.py -q` and confirm imports or assertions fail because configuration is absent.
- [x] Implement the Pydantic models and YAML/environment loading needed by those tests.
- [x] Re-run `pytest tests/test_config.py -q` and confirm all configuration tests pass.
- [x] Commit the skeleton and configuration as `feat: add typed benchmark configuration`.

### Task 2: Deterministic workload generation

**Files:**
- Create: `src/servebench/workloads.py`
- Create: `tests/test_workloads.py`

**Interfaces:**
- Consumes: workload settings from `BenchConfig`.
- Produces: `RequestSpec`, `WorkloadGenerator.generate(kind, count, seed)`, and `arrival_schedule(kind, count, rate, seed)`.

- [x] Write tests that assert exact reproducibility, approximate prompt sizes for short and long-prefill requests, common-prefix identity for shared-prefix requests, a visible burst in bursty arrivals, and weighted representation in mixed requests.
- [x] Run `pytest tests/test_workloads.py -q` and confirm failure because the generator does not exist.
- [x] Implement token-shaped synthetic prompts without requiring a tokenizer, deterministic RNG isolation, Poisson arrivals, and a fixed concurrency spike.
- [x] Re-run `pytest tests/test_workloads.py -q` and confirm all workload tests pass.
- [x] Commit as `feat: add deterministic workload classes`.

### Task 3: Request metrics and result storage

**Files:**
- Create: `src/servebench/results.py`
- Create: `src/servebench/stats.py`
- Create: `tests/test_results.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Produces: `RequestMeasurement`, `JsonlWriter`, `summarize_measurements`, `bootstrap_ci`, and `write_parquet`.

- [x] Write tests for TTFT, inter-token latency, end-to-end latency, token throughput, percentile interpolation, deterministic bootstrap confidence intervals, failure accounting, JSONL round trips, and Parquet output.
- [x] Run the result/stat tests and confirm the missing implementation fails.
- [x] Implement request records with run/workload/policy metadata, streaming event timestamps, percentile summaries, bootstrap intervals, atomic JSONL append, and pandas Parquet conversion.
- [x] Re-run the result/stat tests and confirm they pass.
- [x] Commit as `feat: add benchmark measurement pipeline`.

### Task 4: SLO-aware router policy

**Files:**
- Create: `src/router/policy.py`
- Create: `tests/router/test_policy.py`

**Interfaces:**
- Produces: `Decision(action, worker, delay_seconds, reason)`, `RouterSnapshot`, `RequestFeatures`, `FifoPolicy.decide`, and `SloPolicy.decide`.

- [x] Write table-driven tests showing immediate admission below thresholds, short delays near saturation, overload rejection beyond queue/KV limits, long-prompt protection under KV pressure, and least-loaded alternate-worker routing.
- [x] Run `pytest tests/router/test_policy.py -q` and confirm the absent policy fails.
- [x] Implement a deterministic threshold policy with configurable TTFT SLO, maximum delay, queue limit, prompt-length cutoff, KV soft/hard limits, and worker selection.
- [x] Re-run the policy tests and confirm they pass.
- [x] Commit as `feat: add slo-aware admission policy`.

### Task 5: OpenAI-compatible router service and observability

**Files:**
- Create: `src/router/app.py`
- Create: `src/router/backend.py`
- Create: `src/router/state.py`
- Create: `src/metrics/prometheus.py`
- Create: `tests/router/test_app.py`

**Interfaces:**
- Consumes: policies from Task 4 and one or more vLLM OpenAI base URLs.
- Produces: `/v1/chat/completions`, `/v1/completions`, `/healthz`, `/readyz`, and `/metrics`.

- [x] Write ASGI tests for forwarding, streaming SSE preservation, queue accounting, 429 overload responses with retry hints, delayed admission, alternate routing, health endpoints, and Prometheus labels.
- [x] Run `pytest tests/router/test_app.py -q` and confirm service imports fail.
- [x] Implement bounded admission state, async backend streaming with disconnect cleanup, periodic vLLM metric sampling, KV/preemption/prefix-cache parsing, and low-cardinality Prometheus metrics.
- [x] Re-run router tests and confirm they pass.
- [x] Commit as `feat: add observable inference router`.

### Task 6: Async load generator and mock backend

**Files:**
- Create: `src/loadgen/client.py`
- Create: `src/loadgen/cli.py`
- Create: `src/mock_backend/app.py`
- Create: `tests/loadgen/test_client.py`
- Create: `tests/test_mock_backend.py`

**Interfaces:**
- Consumes: `RequestSpec` and router OpenAI-compatible streaming endpoints.
- Produces: `run_load`, a `servebench-loadgen` CLI, and a deterministic development backend.

- [x] Write tests that feed fragmented SSE, assert TTFT/token timestamps and usage parsing, enforce request timeouts, preserve failures, and verify bounded concurrency.
- [x] Run the load-generator tests and confirm failure due to missing client.
- [x] Implement the asyncio scheduler, SSE parser, request measurement, progress logging, run metadata, JSONL output, and deterministic mock streaming backend.
- [x] Re-run load-generator and mock-backend tests and confirm they pass.
- [x] Commit as `feat: add streaming load generator`.

### Task 7: Experiment runner, sweeps, and autonomous state loop

**Files:**
- Create: `src/experiments/runner.py`
- Create: `src/experiments/sweep.py`
- Create: `src/experiments/state.py`
- Create: `experiments/baseline.yaml`
- Create: `experiments/scheduler.yaml`
- Create: `tests/experiments/test_runner.py`
- Create: `tests/experiments/test_sweep.py`

**Interfaces:**
- Produces: `servebench-experiment`, transition-aware concurrency selection, repeated paired policy comparisons, throughput-constraint evaluation, and append-only `BENCH_STATE.md` updates.

- [x] Write tests for geometric concurrency growth followed by transition refinement, one-variable experiment validation, minimum repeat enforcement, paired bootstrap comparisons, keep/revert decisions, and state-log rendering.
- [x] Run experiment tests and confirm missing modules fail.
- [x] Implement subprocess/service orchestration, repeat manifests, saturation detection, independent engine-variable variants, FIFO/SLO comparisons, and confidence-aware decisions.
- [x] Re-run experiment tests and confirm they pass.
- [x] Commit as `feat: add transition-focused experiment runner`.

### Task 8: Reports and figures

**Files:**
- Create: `src/servebench/report.py`
- Create: `tests/test_report.py`
- Create: `REPORT.md`
- Create: `figures/.gitkeep`
- Create: `results/.gitkeep`

**Interfaces:**
- Consumes: summary JSON and request Parquet files from Tasks 3 and 7.
- Produces: saturation, latency decomposition, and FIFO-vs-SLO figures plus a regenerated `REPORT.md`.

- [x] Write tests using a tiny fixture dataset to verify all three figures, latency/throughput/cost tables, confidence intervals, bottleneck language, and refusal to claim improvement with fewer than three repeats.
- [x] Run `pytest tests/test_report.py -q` and confirm missing report code fails.
- [x] Implement headless matplotlib plots and evidence-gated Markdown generation.
- [x] Re-run report tests and confirm they pass.
- [x] Commit as `feat: generate evidence-based benchmark report`.

### Task 9: Container deployment and dashboards

**Files:**
- Create: `docker-compose.yml`
- Create: `docker-compose.smoke.yml`
- Create: `docker/Dockerfile`
- Create: `configs/prometheus.yml`
- Create: `configs/grafana/provisioning/datasources/prometheus.yml`
- Create: `configs/grafana/provisioning/dashboards/default.yml`
- Create: `dashboards/servebench.json`
- Create: `Makefile`
- Create: `tests/test_deployment_contract.py`

**Interfaces:**
- Produces: `make serve`, `make smoke`, `make loadtest`, `make sweep`, and `make report`.

- [x] Write structural tests that parse Compose, Prometheus, Grafana, dashboard, and Make configuration and require every contract service, GPU reservation, health check, persistent result mount, and command target.
- [x] Run deployment-contract tests and confirm configuration is absent.
- [x] Implement the image, real GPU Compose stack, mock smoke override, scrape configuration, provisioned dashboard panels, and Make commands.
- [x] Re-run deployment-contract tests; then run `docker compose config` when Docker is available.
- [x] Commit as `feat: add one-command observability stack`.

### Task 10: Operator documentation and end-to-end verification

**Files:**
- Create: `README.md`
- Create: `BENCH_STATE.md`
- Modify: `.gitignore`
- Modify: `REPORT.md`

**Interfaces:**
- Documents: architecture, model selection, prerequisites, experiment protocol, metric definitions, commands, result schema, graph interpretation, multi-GPU routing, and limitations.

- [x] Add a documentation test requiring the architecture diagram, three key graph references, all Make commands, configurable `MODEL`, raw-result locations, and explicit no-GPU smoke instructions.
- [x] Run the documentation test and confirm it fails until the operator guide is complete.
- [x] Write the README, initialize the state ledger without optimization claims, and ensure the report distinguishes example/smoke data from GPU evidence.
- [x] Run the full unit suite, lint/type checks, mock smoke stack, load test, report generation, Compose validation, and verify generated JSONL/Parquet/figures.
- [x] Record unavailable GPU/Docker checks honestly, with exact commands for reproduction, and commit as `docs: add servebench operator guide`.

## Self-Review

- Spec coverage: all required services, workload classes, metrics, sweeps, scheduler decisions, repeated measurements, confidence intervals, state loop, repository paths, commands, results, report tradeoffs, architecture, and graphs map to Tasks 1–10.
- Placeholder scan: the plan contains no deferred implementation steps; the initial evidence ledger may describe pending GPU measurements but cannot claim an optimization.
- Type consistency: configuration feeds workload/router/experiment components; `RequestSpec` becomes `RequestMeasurement`; JSONL/Parquet feed summaries and reports; policy inputs and outputs remain stable across service and tests.
