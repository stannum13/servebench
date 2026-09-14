"""HTTP transport for vLLM's OpenAI-compatible endpoints."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx

from .state import RouterState


def _metric(text: str, name: str) -> float | None:
    match = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_vllm_metrics(text: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    if (kv := _metric(text, "vllm:gpu_cache_usage_perc")) is not None:
        result["kv_usage"] = kv
    if (preemptions := _metric(text, "vllm:num_preemptions_total")) is not None:
        result["preemptions"] = int(preemptions)
    hits = _metric(text, "vllm:prefix_cache_hits_total")
    queries = _metric(text, "vllm:prefix_cache_queries_total")
    if hits is not None and queries:
        result["prefix_cache_hit_rate"] = hits / queries
    return result


class BackendPool:
    def __init__(self, state: RouterState, timeout: float = 300.0) -> None:
        self.state = state
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10))

    async def complete(
        self, worker: str, path: str, body: dict[str, object]
    ) -> tuple[int, bytes]:
        response = await self.client.post(self.state.workers[worker].url + path, json=body)
        return response.status_code, response.content

    async def stream(
        self, worker: str, path: str, body: dict[str, object]
    ) -> AsyncIterator[bytes]:
        async with self.client.stream(
            "POST", self.state.workers[worker].url + path, json=body
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_raw():
                yield chunk

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch_metrics(self, worker: str) -> dict[str, float | int]:
        response = await self.client.get(self.state.workers[worker].url + "/metrics")
        response.raise_for_status()
        return parse_vllm_metrics(response.text)
