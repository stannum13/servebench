import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from router.app import create_app
from router.backend import parse_vllm_metrics
from router.state import RouterState
from servebench.config import BenchConfig


class FakeBackend:
    async def complete(self, worker: str, path: str, body: dict[str, object]) -> tuple[int, bytes]:
        assert worker == "worker-0"
        assert path == "/v1/completions"
        return 200, json.dumps({"choices": [{"text": "ok"}]}).encode()

    async def stream(
        self, worker: str, path: str, body: dict[str, object]
    ) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"text":"a"}]}\n\n'
        yield b"data: [DONE]\n\n"


def test_forwards_non_streaming_completion() -> None:
    app = create_app(BenchConfig(), backend=FakeBackend())
    response = TestClient(app).post("/v1/completions", json={"prompt": "hello", "stream": False})
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "ok"


def test_streaming_sse_is_preserved() -> None:
    app = create_app(BenchConfig(), backend=FakeBackend())
    response = TestClient(app).post("/v1/completions", json={"prompt": "hello", "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.endswith(b"data: [DONE]\n\n")


def test_slo_returns_429_when_kv_is_exhausted() -> None:
    config = BenchConfig.model_validate({"router": {"policy": "slo"}})
    state = RouterState(["http://worker-0:8000"])
    state.update_metrics("worker-0", kv_usage=0.99)
    response = TestClient(create_app(config, backend=FakeBackend(), state=state)).post(
        "/v1/completions", json={"prompt": "hello"}
    )
    assert response.status_code == 429
    assert "retry-after" in response.headers


def test_health_and_metrics_expose_queue_and_decisions() -> None:
    client = TestClient(create_app(BenchConfig(), backend=FakeBackend()))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    client.post("/v1/completions", json={"prompt": "hello"})
    metrics = client.get("/metrics").text
    assert "servebench_router_queue_depth" in metrics
    assert 'servebench_admission_decisions_total{action="admit"}' in metrics


def test_vllm_metrics_parser_extracts_cache_and_preemptions() -> None:
    parsed = parse_vllm_metrics(
        'vllm:gpu_cache_usage_perc{model_name="x"} 0.83\n'
        "vllm:num_preemptions_total 7\n"
        "vllm:prefix_cache_hits_total 30\n"
        "vllm:prefix_cache_queries_total 40\n"
    )
    assert parsed == {"kv_usage": 0.83, "preemptions": 7, "prefix_cache_hit_rate": 0.75}
