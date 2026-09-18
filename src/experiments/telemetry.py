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
        if len(results) != 1:
            raise ValueError("telemetry query returned multiple series; aggregate it explicitly")
        value = float(results[0]["value"][1])
        return value if math.isfinite(value) else None

    def collect(self, started_at: float, completed_at: float) -> dict[str, float | None]:
        window_ms = max(1, math.floor((completed_at - started_at) * 1000))
        span = f"[{window_ms}ms]"
        subquery_step_ms = min(5000, window_ms)

        def histogram_mean_ms(name: str) -> str:
            numerator = f"sum(increase({name}_sum{span}))"
            denominator = f"sum(increase({name}_count{span}))"
            return f"({numerator} / {denominator}) * 1000"

        expressions = {
            "kv_cache_peak": (
                f"max(max_over_time(vllm:kv_cache_usage_perc{span}) or "
                f"max_over_time(vllm:gpu_cache_usage_perc{span}))"
            ),
            "preemptions": (
                f"sum(increase(vllm:num_preemptions_total{span}) or "
                f"increase(vllm:num_preemptions{span}))"
            ),
            "prefix_cache_hit_rate": (
                f"sum(increase(vllm:prefix_cache_hits_total{span}) or "
                f"increase(vllm:prefix_cache_hits{span})) / "
                f"sum(increase(vllm:prefix_cache_queries_total{span}) or "
                f"increase(vllm:prefix_cache_queries{span}))"
            ),
            "gpu_utilization_peak": f"max(max_over_time(DCGM_FI_DEV_GPU_UTIL{span}))",
            "gpu_memory_peak_mib": f"max(max_over_time(DCGM_FI_DEV_FB_USED{span}))",
            "gpu_power_average_watts": f"sum(avg_over_time(DCGM_FI_DEV_POWER_USAGE{span}))",
            "vllm_queue_peak": (
                "max_over_time((sum(vllm:num_requests_waiting))"
                f"[{window_ms}ms:{subquery_step_ms}ms])"
            ),
            "vllm_queue_mean_ms": histogram_mean_ms("vllm:request_queue_time_seconds"),
            "vllm_prefill_mean_ms": histogram_mean_ms("vllm:request_prefill_time_seconds"),
            "vllm_ttft_mean_ms": histogram_mean_ms("vllm:time_to_first_token_seconds"),
            "vllm_decode_mean_ms": histogram_mean_ms("vllm:request_decode_time_seconds"),
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
