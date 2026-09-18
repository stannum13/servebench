"""Concurrency-safe in-memory worker pressure state."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .policy import Decision, RequestFeatures, RouterSnapshot, WorkerState


class AdmissionPolicy(Protocol):
    def decide(self, request: RequestFeatures, snapshot: RouterSnapshot) -> Decision: ...


@dataclass(slots=True)
class WorkerRuntime:
    name: str
    url: str
    queue_depth: int = 0
    backend_waiting: int = 0
    kv_usage: float = 0.0
    prefix_cache_hit_rate: float | None = None
    preemptions: int = 0
    pressure_updated_at: float | None = None


class RouterState:
    def __init__(
        self,
        urls: list[str],
        *,
        clock: Callable[[], float] = time.monotonic,
        telemetry_ttl_seconds: float = 5.0,
    ) -> None:
        self.workers = {
            f"worker-{index}": WorkerRuntime(f"worker-{index}", url.rstrip("/"))
            for index, url in enumerate(urls)
        }
        self._clock = clock
        self._telemetry_ttl_seconds = telemetry_ttl_seconds
        self._lock = asyncio.Lock()

    def snapshot(self) -> RouterSnapshot:
        return RouterSnapshot(
            [
                WorkerState(
                    item.name,
                    item.queue_depth,
                    item.kv_usage,
                    item.backend_waiting,
                    pressure_fresh=self.pressure_is_fresh(item.name),
                )
                for item in self.workers.values()
            ]
        )

    def pressure_is_fresh(self, worker: str) -> bool:
        updated_at = self.workers[worker].pressure_updated_at
        return (
            updated_at is not None
            and self._clock() - updated_at <= self._telemetry_ttl_seconds
        )

    def update_metrics(
        self,
        worker: str,
        *,
        kv_usage: float | None = None,
        prefix_cache_hit_rate: float | None = None,
        preemptions: int | None = None,
        backend_waiting: int | None = None,
    ) -> None:
        runtime = self.workers[worker]
        if kv_usage is not None:
            runtime.kv_usage = kv_usage
        if prefix_cache_hit_rate is not None:
            runtime.prefix_cache_hit_rate = prefix_cache_hit_rate
        if preemptions is not None:
            runtime.preemptions = preemptions
        if backend_waiting is not None:
            runtime.backend_waiting = max(0, backend_waiting)
        if kv_usage is not None and backend_waiting is not None:
            runtime.pressure_updated_at = self._clock()

    async def acquire(self, worker: str) -> None:
        async with self._lock:
            self.workers[worker].queue_depth += 1

    async def reserve(
        self, policy: AdmissionPolicy, request: RequestFeatures, *, after_delay: bool = False
    ) -> Decision:
        async with self._lock:
            decision = policy.decide(request, self.snapshot())
            if decision.worker is not None and (
                decision.action == "admit" or (after_delay and decision.action == "delay")
            ):
                self.workers[decision.worker].queue_depth += 1
                if decision.action == "delay":
                    return Decision(
                        "admit", decision.worker,
                        reason=f"admitted after rechecked delay: {decision.reason}",
                    )
            return decision

    async def release(self, worker: str) -> None:
        async with self._lock:
            self.workers[worker].queue_depth = max(0, self.workers[worker].queue_depth - 1)
