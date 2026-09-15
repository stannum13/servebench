"""Deterministic synthetic workload definitions for inference experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from .config import WorkloadConfig

WorkloadKind = Literal["short", "long-prefill", "shared-prefix", "bursty", "mixed"]


@dataclass(frozen=True, slots=True)
class RequestSpec:
    request_id: str
    kind: str
    prompt: str | list[int]
    prompt_tokens: int
    max_tokens: int = 128


def _tokens(count: int, rng: random.Random) -> str:
    return " ".join(f"tok{rng.randrange(10_000):04d}" for _ in range(count))


class WorkloadGenerator:
    def __init__(self, config: WorkloadConfig | None = None) -> None:
        self.config = config or WorkloadConfig()

    def generate(self, kind: WorkloadKind, count: int, seed: int) -> list[RequestSpec]:
        if count < 0:
            raise ValueError("count must be non-negative")
        rng = random.Random(seed)
        if kind == "mixed":
            kinds: list[str] = []
            weights = self.config.mixed_weights
            allocated = {name: math.floor(count * weight) for name, weight in weights.items()}
            remainder = count - sum(allocated.values())
            order = sorted(weights, key=lambda name: (-(count * weights[name] % 1), name))
            for name in order[:remainder]:
                allocated[name] += 1
            for name, amount in allocated.items():
                kinds.extend([name] * amount)
            rng.shuffle(kinds)
        else:
            kinds = [kind] * count

        shared_prefix = _tokens(3968, random.Random(seed ^ 0x5EED))
        requests = []
        for index, request_kind in enumerate(kinds):
            request_rng = random.Random(rng.randrange(2**63))
            if request_kind == "long-prefill":
                prompt_tokens = 4096
                prompt = _tokens(prompt_tokens, request_rng)
            elif request_kind == "shared-prefix":
                suffix = _tokens(128, request_rng)
                prompt_tokens = 4096
                prompt = f"{shared_prefix} UNIQUE_SUFFIX {index:06d} {suffix}"
            else:
                prompt_tokens = 256
                prompt = _tokens(prompt_tokens, request_rng)
            requests.append(
                RequestSpec(
                    request_id=f"{seed}-{index:06d}-{request_kind}",
                    kind=request_kind,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                )
            )
        return requests


def arrival_schedule(
    kind: WorkloadKind, count: int, rate: float, seed: int, spike_size: int = 10
) -> list[float]:
    """Return monotonic seconds from run start for open-loop request arrivals."""

    if count < 0 or rate <= 0:
        raise ValueError("count must be non-negative and rate must be positive")
    rng = random.Random(seed)
    arrivals: list[float] = []
    now = 0.0
    spike_at = count // 2
    for index in range(count):
        if kind in {"bursty", "mixed"} and spike_at <= index < min(count, spike_at + spike_size):
            arrivals.append(now)
            continue
        now += rng.expovariate(rate)
        arrivals.append(now)
    return arrivals
