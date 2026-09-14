# Servebench Report

> Generated from machine-readable measurements. Smoke/mock results are not GPU evidence.

## Latency / throughput / cost

| policy | p95 TTFT ms (95% CI) | requests/s (95% CI) | power (W) | $ / 1M output tokens |
|---|---:|---:|---:|---:|
| fifo | 21.37 [20.32, 26.26] | 14.29 [13.47, 14.67] | unavailable | not configured |
| slo | 22.71 [20.19, 24.79] | 12.59 [12.58, 12.60] | unavailable | not configured |

Mock evidence only: repeated measurements exist, but no optimization claim is made.

## Key graphs

![Saturation curve](figures/saturation.png)

![Latency decomposition](figures/latency-decomposition.png)

![Scheduler comparison](figures/scheduler-comparison.png)

Queue growth, KV-cache pressure, and GPU utilization must be interpreted together
to locate saturation.
