# Servebench Report

> Generated from machine-readable measurements. Smoke/mock results are not GPU evidence.

## Latency / throughput / cost

| policy | p95 TTFT ms (95% CI) | requests/s (95% CI) | completion % | rejection % | power (W) | $ / 1M output tokens |
|---|---:|---:|---:|---:|---:|---:|
| fifo | 25.09 [13.65, 47.16] | 14.54 [13.52, 15.09] | 100.00 | 0.00 | unavailable | not configured |
| slo | 16.59 [14.95, 18.09] | 14.28 [12.97, 15.02] | 100.00 | 0.00 | unavailable | not configured |

Mock evidence only: repeated measurements exist, but no optimization claim is made.

ITL is measured between non-empty streamed output events; it is stream-event ITL,
not a fabricated per-token gap when one event contains multiple token IDs. TPOT is
the elapsed time after the first token divided by the remaining output-token count.

## Key graphs

![Saturation curve](figures/saturation.png)

![Latency decomposition](figures/latency-decomposition.png)

Client queue is loadgen semaphore wait and is outside TTFT. Router admission is measured inside the router; post-header TTFT includes backend queue, prefill, and transport. Run-scoped vLLM queue and prefill means are unavailable and unavailable, respectively. These p95 stage percentiles are not additive.

![Scheduler comparison](figures/scheduler-comparison.png)

Queue growth, KV-cache pressure, and GPU utilization must be interpreted together
to locate saturation.
