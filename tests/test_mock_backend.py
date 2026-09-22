from fastapi.testclient import TestClient

from mock_backend.app import create_app


def test_mock_exposes_vllm_health_and_metrics_contract() -> None:
    client = TestClient(create_app(token_delay=0))
    assert client.get("/health").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "vllm:gpu_cache_usage_perc" in metrics.text
    assert "vllm:num_requests_waiting" in metrics.text
