import httpx

from experiments.telemetry import PrometheusTelemetry


def test_collects_run_scoped_gpu_and_cache_metrics() -> None:
    values = iter([0.82, 7, 0.75, 93, 12000, 275, 11])

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
