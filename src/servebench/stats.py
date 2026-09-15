"""Statistics used by experiment summaries and policy comparisons."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

from .results import RequestMeasurement


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence: float


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 6)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {name: percentile(values, q) for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}


def summarize_measurements(items: list[RequestMeasurement]) -> dict[str, object]:
    successful = [item for item in items if item.status == "ok"]
    ttft = [value for item in successful if (value := item.ttft_ms) is not None]
    itl = [value for item in successful for value in item.inter_token_latencies_ms]
    e2e = [item.e2e_latency_ms for item in successful]
    queues = [item.queue_time_ms for item in successful if item.queue_time_ms is not None]
    router_admissions = [
        item.router_admission_ms for item in successful if item.router_admission_ms is not None
    ]
    post_header_ttfts = [
        item.post_header_ttft_ms for item in successful if item.post_header_ttft_ms is not None
    ]
    if items:
        last_completion = max(item.completed_at for item in items)
        first_arrival = min(item.scheduled_at for item in items)
        duration = last_completion - first_arrival
    else:
        duration = 0.0
    duration = max(duration, 1e-9)
    return {
        "requests": len(items),
        "successful": len(successful),
        "failures": len(items) - len(successful),
        "timeouts": sum(item.status == "timeout" for item in items),
        "rejections": sum(item.status == "rejected" for item in items),
        "duration_seconds": duration,
        "requests_per_second": len(successful) / duration,
        "input_tokens_per_second": sum(item.prompt_tokens for item in successful) / duration,
        "output_tokens_per_second": sum(item.output_tokens for item in successful) / duration,
        "ttft_ms": _percentiles(ttft),
        "inter_token_latency_ms": _percentiles(itl),
        "e2e_latency_ms": _percentiles(e2e),
        "queue_time_ms": _percentiles(queues),
        "client_queue_time_ms": _percentiles(queues),
        "router_admission_ms": _percentiles(router_admissions),
        "post_header_ttft_ms": _percentiles(post_header_ttfts),
    }


def bootstrap_ci(
    values: list[float], confidence: float = 0.95, samples: int = 2000, seed: int = 0
) -> ConfidenceInterval:
    if not values:
        raise ValueError("values cannot be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
    tail = (1 - confidence) / 2
    low = percentile(estimates, tail)
    high = percentile(estimates, 1 - tail)
    assert low is not None and high is not None
    return ConfidenceInterval(float(mean(values)), low, high, confidence)
