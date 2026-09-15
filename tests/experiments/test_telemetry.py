import httpx
import pytest

from experiments.telemetry import PrometheusTelemetry


def test_collects_run_scoped_gpu_and_cache_metrics() -> None:
    values = iter([0.82, 7, 0.75, 93, 12000, 275, 11, 30, 55, 95, 800])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/query"
        payload = {
            "status": "success",
            "data": {"result": [{"value": [10, str(next(values))]}]},
        }
        return httpx.Response(200, json=payload)

    telemetry = PrometheusTelemetry("http://prometheus", transport=httpx.MockTransport(handler))
    result = telemetry.collect(started_at=100, completed_at=110)
    assert result["kv_cache_peak"] == 0.82
    assert result["preemptions"] == 7
    assert result["prefix_cache_hit_rate"] == 0.75
    assert result["gpu_utilization_peak"] == 93
    assert result["gpu_memory_peak_mib"] == 12000
    assert result["gpu_power_average_watts"] == 275
    assert result["vllm_queue_peak"] == 11
    assert result["vllm_queue_mean_ms"] == 30
    assert result["vllm_prefill_mean_ms"] == 55
    assert result["vllm_ttft_mean_ms"] == 95
    assert result["vllm_decode_mean_ms"] == 800


def test_run_telemetry_queries_aggregate_multi_worker_series() -> None:
    expressions = []

    def handler(request: httpx.Request) -> httpx.Response:
        expression = request.url.params["query"]
        expressions.append(expression)
        return httpx.Response(200, json={
            "status": "success", "data": {"result": [{"value": [10, "1"]}]},
        })

    transport = httpx.MockTransport(handler)
    with PrometheusTelemetry("http://prometheus", transport=transport) as telemetry:
        telemetry.collect(100, 120)
    assert any(
        "sum(increase(vllm:request_queue_time_seconds_sum" in value
        for value in expressions
    )
    assert any(
        "sum(increase(vllm:request_prefill_time_seconds_sum" in value
        for value in expressions
    )
    assert any("max(max_over_time(DCGM_FI_DEV_GPU_UTIL" in value for value in expressions)


def test_unaggregated_multiple_series_are_not_silently_first_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": "success", "data": {"result": [
                {"value": [10, "1"]}, {"value": [10, "2"]},
            ]},
        })

    transport = httpx.MockTransport(handler)
    with (
        PrometheusTelemetry("http://prometheus", transport=transport) as telemetry,
        pytest.raises(ValueError, match="multiple series"),
    ):
        telemetry.collect(100, 120)
