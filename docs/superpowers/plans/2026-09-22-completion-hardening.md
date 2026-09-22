# Servebench Completion Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining code-side validity gaps so a real NVIDIA run can produce defensible overload and scheduler conclusions without further implementation work.

**Architecture:** Extend the existing request schema and aggregate summaries rather than adding another data pipeline. Keep paired policy decisions in `experiments.runner`, add explicit streaming-output and per-token latency semantics in `servebench.results`/`servebench.stats`, and record cache isolation/provenance in engine artifacts. Exercise the full mock Compose path as the final contract test while keeping all mock results evidence-gated.

**Tech Stack:** Python 3.11+, FastAPI, httpx SSE, Pydantic, pytest, Prometheus, Docker Compose, vLLM 0.18.

## Global Constraints

- Do not use the brainstorming plugin.
- Do not provision paid GPU or cloud resources.
- Write a failing regression test before every behavior change.
- Require at least three paired repeats before an optimization decision.
- Never allow rejection, timeout, missing telemetry, or mock data to support a positive GPU optimization claim.
- Keep `MODEL` and quantized checkpoints configurable.

---

### Task 1: Rejection-proof policy comparison

**Files:**
- Modify: `src/experiments/runner.py`
- Modify: `src/servebench/report.py`
- Test: `tests/experiments/test_runner.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: per-repeat `p95_ttft_ms`, `requests_per_second`, `requests`, `successful`, `rejections`, `timeouts`, and `failures`.
- Produces: `comparison.json` with paired confidence intervals plus offered, accepted, rejection, timeout, and completion-rate evidence; a positive decision requires the configured throughput floor and no degraded completion rate.

- [x] **Step 1: Write failing tests** demonstrating that a low-TTFT policy with a worse completion ratio cannot be kept and that the report renders acceptance/rejection evidence.
- [x] **Step 2: Run focused tests** with `.venv/bin/pytest -q tests/experiments/test_runner.py tests/test_report.py`; expect the new assertions to fail.
- [x] **Step 3: Implement the minimal comparison fields and gates** using paired bootstrap intervals and configured `throughput_floor_ratio`.
- [x] **Step 4: Re-run the focused tests** and require all to pass.

### Task 2: Explicit ITL and TPOT semantics

**Files:**
- Modify: `src/servebench/results.py`
- Modify: `src/servebench/stats.py`
- Modify: `src/servebench/report.py`
- Test: `tests/test_results.py`
- Test: `tests/test_stats.py`
- Test: `tests/loadgen/test_client.py`

**Interfaces:**
- Consumes: `first_token_at`, `completed_at`, output-token count, per-SSE-event token IDs, and event timestamps.
- Produces: stream-event ITL and request-level TPOT as distinct metrics; no artificial zero-duration ITL entries for several token IDs delivered by one SSE event.

- [x] **Step 1: Write failing tests** for a multi-token SSE event and for TPOT `(E2E - TTFT) / (output_tokens - 1)`.
- [x] **Step 2: Run the focused tests** and verify the semantic mismatch is exposed.
- [x] **Step 3: Store one timestamp per streamed output event, preserve the total token count, and add `tpot_ms` plus p50/p95/p99 aggregate output.**
- [x] **Step 4: Update report labels** so ITL says stream-event ITL and TPOT says per output token.
- [x] **Step 5: Re-run focused tests** and require all to pass.

### Task 3: Cache isolation and run provenance

**Files:**
- Modify: `src/experiments/engine.py`
- Modify: `src/experiments/runner.py`
- Modify: `src/mock_backend/app.py`
- Test: `tests/experiments/test_engine.py`
- Test: `tests/experiments/test_runner.py`

**Interfaces:**
- Consumes: suite config, model identity, engine settings, paired baseline/candidate names, seed, and process restart boundaries.
- Produces: run manifests that record model/settings/cache-isolation mode and decision artifacts linking the exact baseline and candidate suite IDs.

- [x] **Step 1: Write failing tests** requiring provenance fields and explicit cold-start/cache-state metadata for baseline and engine candidates.
- [x] **Step 2: Run focused tests** and confirm missing fields fail.
- [x] **Step 3: Add provenance to every summary and decision artifact.** Treat vLLM restart as the cache reset boundary for engine variants; document that ordinary repeated scheduler runs intentionally preserve cache state and use paired seeds/order.
- [x] **Step 4: Re-run focused tests** and require all to pass.

### Task 4: End-to-end completion contract and documentation

**Files:**
- Modify: `tests/test_deployment_contract.py`
- Modify: `README.md`
- Modify: `BENCH_STATE.md`
- Modify: `REPORT.md` only through `make report`

**Interfaces:**
- Consumes: Compose services, Make targets, machine-readable outputs, report evidence gates, and the public results API.
- Produces: a reproducible non-GPU acceptance path and a precise GPU-run checklist.

- [x] **Step 1: Add deployment contract assertions** for rejection-proof comparison fields, TPOT output, provenance, and the three required graph paths.
- [x] **Step 2: Run the contract test** and verify missing deliverables fail.
- [x] **Step 3: Update operator documentation and state ledger** with the exact real-GPU command order and remaining external blocker.
- [x] **Step 4: Run `.venv/bin/pytest -q`, `.venv/bin/ruff check src tests`, `docker compose config --quiet`, `make report`, and a rebuilt `make smoke`.**
- [ ] **Step 5: Request read-only code review, resolve critical/important findings, commit, push to GitHub, and stop all local containers.**

## Self-Review

- Spec coverage: scheduler validity, latency semantics, cache isolation, provenance, report outputs, and deployment verification are each assigned to a task.
- External gap: the plan deliberately cannot produce NVIDIA measurements on this host; it leaves one documented execution step, not an implementation gap.
- Placeholder scan: no TBD/TODO steps remain.
- Type consistency: comparison fields remain JSON-serializable dictionaries; request-derived latency metrics remain nullable floats; provenance remains plain JSON/YAML-compatible values.
