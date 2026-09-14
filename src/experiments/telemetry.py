"""Run-scoped telemetry queries against the Prometheus HTTP API."""

from __future__ import annotations

import math

import httpx


class PrometheusTelemetry:
    def __init__(self, url: str, *, transport: httpx.BaseTransport | None = None) -> None:
        self.url = url.rstrip("/")
        self.client = httpx.Client(transport=transport, timeout=10)

    def _query(self, expression: str, completed_at: float) -> float | None:
        response = self.client.get(
            self.url + "/api/v1/query",
            params={"query": expression, "time": completed_at},
        )
        response.raise_for_status()
        results = response.json()["data"]["result"]
        if not results:
            return None
        return float(results[0]["value"][1])

    def collect(self, started_at: float, completed_at: float) -> dict[str, float | None]:
        window = max(1, math.ceil(completed_at - started_at))
        span = f"[{window}s]"
        expressions = {
            "kv_cache_peak": (
                f"max_over_time(vllm:kv_cache_usage_perc{span}) or "
                f"max_over_time(vllm:gpu_cache_usage_perc{span})"
            ),
            "preemptions": (
                f"increase(vllm:num_preemptions_total{span}) or "
                f"increase(vllm:num_preemptions{span})"
            ),
            "prefix_cache_hit_rate": (
                f"(increase(vllm:prefix_cache_hits_total{span}) or "
                f"increase(vllm:prefix_cache_hits{span})) / "
                f"(increase(vllm:prefix_cache_queries_total{span}) or "
                f"increase(vllm:prefix_cache_queries{span}))"
            ),
            "gpu_utilization_peak": f"max_over_time(DCGM_FI_DEV_GPU_UTIL{span})",
            "gpu_memory_peak_mib": f"max_over_time(DCGM_FI_DEV_FB_USED{span})",
            "gpu_power_average_watts": f"avg_over_time(DCGM_FI_DEV_POWER_USAGE{span})",
            "vllm_queue_peak": f"max_over_time(vllm:num_requests_waiting{span})",
        }
        return {
            name: self._query(expression, completed_at)
            for name, expression in expressions.items()
        }

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> PrometheusTelemetry:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
