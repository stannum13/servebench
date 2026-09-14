"""Concurrency-safe in-memory worker pressure state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .policy import RouterSnapshot, WorkerState


@dataclass(slots=True)
class WorkerRuntime:
    name: str
    url: str
    queue_depth: int = 0
    kv_usage: float = 0.0


class RouterState:
    def __init__(self, urls: list[str]) -> None:
        self.workers = {
            f"worker-{index}": WorkerRuntime(f"worker-{index}", url.rstrip("/"))
            for index, url in enumerate(urls)
        }
        self._lock = asyncio.Lock()

    def snapshot(self) -> RouterSnapshot:
        return RouterSnapshot(
            [
                WorkerState(item.name, item.queue_depth, item.kv_usage)
                for item in self.workers.values()
            ]
        )

    def update_metrics(self, worker: str, *, kv_usage: float) -> None:
        self.workers[worker].kv_usage = kv_usage

    async def acquire(self, worker: str) -> None:
        async with self._lock:
            self.workers[worker].queue_depth += 1

    async def release(self, worker: str) -> None:
        async with self._lock:
            self.workers[worker].queue_depth = max(0, self.workers[worker].queue_depth - 1)
