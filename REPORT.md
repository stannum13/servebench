# Servebench Report

## Evidence status

GPU baseline measurements are pending. The repository includes a mock smoke path, but mock timings
are not vLLM or GPU evidence. Therefore, no optimization claim is made.

## Required analysis after the first GPU run

The generated report will connect p95 TTFT to queue time, KV-cache utilization, prefix-cache hit
rate, preemptions, GPU utilization/memory/power, and throughput. It will compare the SLO policy with
FIFO using at least three paired measurements and a 95% throughput floor.

## Key graphs

- `figures/saturation.png` — pending GPU sweep
- `figures/latency-decomposition.png` — pending GPU sweep
- `figures/scheduler-comparison.png` — pending repeated policy comparison

