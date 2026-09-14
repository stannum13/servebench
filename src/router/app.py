"""OpenAI-compatible SLO-aware inference router."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from metrics.prometheus import RouterMetrics
from servebench.config import BenchConfig, load_config

from .backend import BackendPool
from .policy import FifoPolicy, RequestFeatures, SloPolicy
from .state import RouterState


class Backend(Protocol):
    async def complete(
        self, worker: str, path: str, body: dict[str, object]
    ) -> tuple[int, bytes]: ...

    def stream(self, worker: str, path: str, body: dict[str, object]) -> AsyncIterator[bytes]: ...


def _prompt_tokens(body: dict[str, object]) -> int:
    prompt = body.get("prompt", "")
    if not prompt and isinstance(body.get("messages"), list):
        messages = body["messages"]
        prompt = " ".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
    return max(1, len(str(prompt).split()))


def create_app(
    config: BenchConfig, backend: Backend | None = None, state: RouterState | None = None
) -> FastAPI:
    router_state = state or RouterState(config.backend.urls)
    transport = backend or BackendPool(router_state, config.backend.request_timeout_seconds)
    policies = {"fifo": FifoPolicy(), "slo": SloPolicy(config.router)}
    instruments = RouterMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        async def sample_metrics() -> None:
            fetch = getattr(transport, "fetch_metrics", None)
            while fetch is not None:
                for worker in router_state.workers:
                    try:
                        values = await fetch(worker)
                        router_state.update_metrics(
                            worker,
                            kv_usage=float(values.get("kv_usage", 0)),
                            prefix_cache_hit_rate=values.get("prefix_cache_hit_rate"),
                            preemptions=int(values.get("preemptions", 0)),
                        )
                    except Exception:  # metric loss must not stop inference
                        continue
                await asyncio.sleep(1)

        sampler = asyncio.create_task(sample_metrics())
        try:
            yield
        finally:
            sampler.cancel()
            with suppress(asyncio.CancelledError):
                await sampler
            close = getattr(transport, "close", None)
            if close is not None:
                await close()

    app = FastAPI(title="servebench router", lifespan=lifespan)
    app.state.router = router_state

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready() -> JSONResponse:
        status = 200 if router_state.workers else 503
        return JSONResponse({"ready": bool(router_state.workers)}, status_code=status)

    @app.get("/metrics")
    async def metrics() -> Response:
        for worker in router_state.workers.values():
            instruments.queue.labels(worker.name).set(worker.queue_depth)
            instruments.kv.labels(worker.name).set(worker.kv_usage)
            if worker.prefix_cache_hit_rate is not None:
                instruments.prefix_hits.labels(worker.name).set(worker.prefix_cache_hit_rate)
            instruments.preemptions.labels(worker.name).set(worker.preemptions)
        return Response(instruments.render(), media_type="text/plain; version=0.0.4")

    async def proxy(request: Request) -> Response:
        body: dict[str, object] = await request.json()
        policy_name = config.router.policy
        if config.router.allow_policy_override:
            policy_name = request.headers.get("X-Servebench-Policy", policy_name)
        policy = policies.get(policy_name)
        if policy is None:
            return JSONResponse({"error": {"message": "unknown policy"}}, status_code=400)
        decision = policy.decide(RequestFeatures(_prompt_tokens(body)), router_state.snapshot())
        instruments.decisions.labels(decision.action).inc()
        if decision.action == "reject" or decision.worker is None:
            return JSONResponse(
                {"error": {"message": decision.reason, "type": "overload"}},
                status_code=429,
                headers={"Retry-After": "1"},
            )
        if decision.action == "delay":
            await asyncio.sleep(decision.delay_seconds)
        worker = decision.worker
        await router_state.acquire(worker)
        if body.get("stream"):
            async def chunks() -> AsyncIterator[bytes]:
                try:
                    async for chunk in transport.stream(worker, request.url.path, body):
                        yield chunk
                finally:
                    await router_state.release(worker)
            return StreamingResponse(chunks(), media_type="text/event-stream")
        try:
            status, content = await transport.complete(worker, request.url.path, body)
            return Response(content, status_code=status, media_type="application/json")
        finally:
            await router_state.release(worker)

    app.add_api_route("/v1/completions", proxy, methods=["POST"])
    app.add_api_route("/v1/chat/completions", proxy, methods=["POST"])
    return app


app = create_app(load_config(os.getenv("CONFIG_PATH", "configs/default.yaml")))
