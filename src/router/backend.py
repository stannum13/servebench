"""HTTP transport for vLLM's OpenAI-compatible endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from .state import RouterState


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

