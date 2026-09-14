"""Understandable FIFO and SLO-aware admission policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from servebench.config import RouterConfig


@dataclass(frozen=True, slots=True)
class RequestFeatures:
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class WorkerState:
    name: str
    queue_depth: int
    kv_usage: float

    @property
    def pressure_score(self) -> float:
        return self.queue_depth + self.kv_usage * 100


@dataclass(frozen=True, slots=True)
class RouterSnapshot:
    workers: list[WorkerState]


@dataclass(frozen=True, slots=True)
class Decision:
    action: Literal["admit", "delay", "reject"]
    worker: str | None
    delay_seconds: float = 0.0
    reason: str = ""


class FifoPolicy:
    def decide(self, request: RequestFeatures, snapshot: RouterSnapshot) -> Decision:
        del request
        if not snapshot.workers:
            return Decision("reject", None, reason="no healthy workers")
        return Decision("admit", snapshot.workers[0].name, reason="fifo")


class SloPolicy:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig(policy="slo")

    def decide(self, request: RequestFeatures, snapshot: RouterSnapshot) -> Decision:
        if not snapshot.workers:
            return Decision("reject", None, reason="overload: no healthy workers")
        viable = [
            worker
            for worker in snapshot.workers
            if worker.queue_depth < self.config.queue_limit
            and worker.kv_usage < self.config.kv_hard_limit
        ]
        if not viable:
            return Decision("reject", None, reason="overload: queue or KV hard limit")

        selected = min(viable, key=lambda worker: (worker.pressure_score, worker.name))
        alternate = selected.name != snapshot.workers[0].name
        if selected.kv_usage >= self.config.kv_soft_limit:
            long_prompt = request.prompt_tokens >= self.config.prompt_length_cutoff
            reason = "long prompt delayed under KV pressure" if long_prompt else "KV soft limit"
            delay = min(self.config.max_delay_ms / 1000, 0.05 + selected.queue_depth * 0.005)
            return Decision("delay", selected.name, delay_seconds=delay, reason=reason)
        estimated_ttft_ms = (
            20
            + request.prompt_tokens * 0.05
            + selected.queue_depth * 40
            + selected.kv_usage * 100
        )
        if estimated_ttft_ms > self.config.ttft_slo_ms:
            delay = min(self.config.max_delay_ms / 1000, 0.025 + selected.queue_depth * 0.005)
            return Decision(
                "delay",
                selected.name,
                delay_seconds=delay,
                reason=f"TTFT estimate {estimated_ttft_ms:.0f}ms exceeds SLO",
            )
        if selected.queue_depth >= max(1, self.config.queue_limit // 8):
            delay = min(self.config.max_delay_ms / 1000, 0.025 + selected.queue_depth * 0.005)
            return Decision("delay", selected.name, delay_seconds=delay, reason="queue pressure")
        reason = "healthy alternate worker" if alternate else "within SLO pressure limits"
        return Decision("admit", selected.name, reason=reason)
