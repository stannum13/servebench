import asyncio
import json
from collections.abc import AsyncIterator

import httpx
from fastapi.testclient import TestClient

from router.app import _prompt_tokens, create_app
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
    assert response.headers["X-Servebench-Worker"] == "worker-0"
    assert float(response.headers["X-Servebench-Admission-Ms"]) >= 0


def test_streaming_sse_is_preserved() -> None:
    app = create_app(BenchConfig(), backend=FakeBackend())
    response = TestClient(app).post("/v1/completions", json={"prompt": "hello", "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.endswith(b"data: [DONE]\n\n")
    assert response.headers["X-Servebench-Worker"] == "worker-0"


def test_slo_returns_429_when_kv_is_exhausted() -> None:
    config = BenchConfig.model_validate({"router": {"policy": "slo"}})
    state = RouterState(["http://worker-0:8000"])
    state.update_metrics("worker-0", kv_usage=0.99, backend_waiting=0)
    response = TestClient(create_app(config, backend=FakeBackend(), state=state)).post(
        "/v1/completions", json={"prompt": "hello"}
    )
    assert response.status_code == 429
    assert "retry-after" in response.headers


def test_benchmark_header_selects_slo_against_fifo_default() -> None:
    config = BenchConfig.model_validate({"router": {"allow_policy_override": True}})
    state = RouterState(["http://worker-0:8000"])
    state.update_metrics("worker-0", kv_usage=0.99, backend_waiting=0)
    response = TestClient(create_app(config, backend=FakeBackend(), state=state)).post(
        "/v1/completions",
        json={"prompt": "hello"},
        headers={"X-Servebench-Policy": "slo"},
    )
    assert response.status_code == 429


def test_health_and_metrics_expose_queue_and_decisions() -> None:
    client = TestClient(create_app(BenchConfig(), backend=FakeBackend()))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    client.post("/v1/completions", json={"prompt": "hello"})
    metrics = client.get("/metrics").text
    assert "servebench_router_queue_depth" in metrics
    assert 'servebench_admission_decisions_total{action="admit"}' in metrics


def test_metrics_distinguish_assigned_requests_from_backend_waiting() -> None:
    state = RouterState(["http://worker"])
    state.update_metrics("worker-0", kv_usage=0.5, backend_waiting=7)
    metrics = TestClient(create_app(BenchConfig(), backend=FakeBackend(), state=state)).get(
        "/metrics"
    ).text
    assert 'servebench_backend_waiting_requests{worker="worker-0"} 7.0' in metrics
    assert 'servebench_worker_pressure_telemetry_fresh{worker="worker-0"} 1.0' in metrics


def test_vllm_metrics_parser_extracts_cache_and_preemptions() -> None:
    parsed = parse_vllm_metrics(
        'vllm:gpu_cache_usage_perc{model_name="x"} 0.83\n'
        "vllm:num_preemptions_total 7\n"
        "vllm:prefix_cache_hits_total 30\n"
        "vllm:prefix_cache_queries_total 40\n"
        "vllm:num_requests_waiting 5\n"
    )
    assert parsed == {
        "kv_usage": 0.83, "preemptions": 7, "prefix_cache_hit_rate": 0.75,
        "backend_waiting": 5,
    }


def test_missing_metric_sample_does_not_reset_known_worker_pressure() -> None:
    now = [0.0]
    state = RouterState(
        ["http://worker"], clock=lambda: now[0], telemetry_ttl_seconds=5
    )
    assert state.snapshot().workers[0].pressure_fresh is False
    state.update_metrics("worker-0", kv_usage=0.81, backend_waiting=5)
    assert state.snapshot().workers[0].pressure_fresh is True
    now[0] = 4
    state.update_metrics("worker-0")
    worker = state.workers["worker-0"]
    assert worker.kv_usage == 0.81
    assert worker.backend_waiting == 5
    now[0] = 6
    assert state.snapshot().workers[0].pressure_fresh is False


def test_router_requires_bearer_token_when_configured() -> None:
    config = BenchConfig.model_validate({"router": {"api_key": "secret"}})
    client = TestClient(create_app(config, backend=FakeBackend()))
    assert client.post("/v1/completions", json={"prompt": "hello"}).status_code == 401
    response = client.post(
        "/v1/completions",
        json={"prompt": "hello"},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200


async def test_delayed_request_rechecks_kv_hard_limit() -> None:
    config = BenchConfig.model_validate({"router": {
        "policy": "slo", "max_delay_ms": 100, "kv_soft_limit": 0.8,
        "kv_hard_limit": 0.95,
    }})
    state = RouterState(["http://worker-0:8000"])
    state.update_metrics("worker-0", kv_usage=0.81, backend_waiting=0)
    app = create_app(config, backend=FakeBackend(), state=state)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pending = asyncio.create_task(client.post(
            "/v1/completions", json={"prompt": "hello", "stream": False}
        ))
        await asyncio.sleep(0.01)
        state.update_metrics("worker-0", kv_usage=0.99, backend_waiting=0)
        response = await pending
    assert response.status_code == 429


async def test_concurrent_delayed_requests_do_not_overbook_queue_limit() -> None:
    class SlowBackend(FakeBackend):
        async def complete(
            self, worker: str, path: str, body: dict[str, object]
        ) -> tuple[int, bytes]:
            await asyncio.sleep(0.08)
            return await super().complete(worker, path, body)

    config = BenchConfig.model_validate({"router": {
        "policy": "slo", "queue_limit": 1, "max_delay_ms": 100,
    }})
    state = RouterState(["http://worker-0:8000"])
    state.update_metrics("worker-0", kv_usage=0.81, backend_waiting=0)
    app = create_app(config, backend=SlowBackend(), state=state)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(*[
            client.post("/v1/completions", json={"prompt": "hello"}) for _ in range(4)
        ])
    assert [response.status_code for response in responses].count(200) == 1
    assert [response.status_code for response in responses].count(429) == 3


def test_router_counts_token_id_prompt_without_string_heuristic() -> None:
    assert _prompt_tokens({"prompt": list(range(256))}) == 256
