"""Request-level measurement schema and machine-readable persistence."""

from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field


class RequestMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    request_id: str
    workload: str
    policy: str
    prompt_tokens: int = Field(ge=0)
    nominal_prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int = Field(ge=0)
    scheduled_at: float
    started_at: float
    first_token_at: float | None = None
    headers_received_at: float | None = None
    stream_event_timestamps: list[float] = Field(default_factory=list)
    stream_timing_valid: bool = False
    token_timestamps: list[float] = Field(default_factory=list)
    token_timing_exact: bool = False
    completed_at: float
    status: Literal["ok", "error", "timeout", "rejected"]
    http_status_code: int | None = None
    error: str | None = None
    queue_time_ms: float | None = None
    client_queue_time_ms: float | None = None
    router_admission_ms: float | None = None
    worker: str | None = None
    kv_cache_usage: float | None = None
    prefix_cache_hit_rate: float | None = None
    preemptions: int | None = None
    gpu_utilization: float | None = None
    gpu_memory_bytes: int | None = None
    gpu_power_watts: float | None = None

    @computed_field
    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 6)

    @computed_field
    @property
    def post_header_ttft_ms(self) -> float | None:
        if self.first_token_at is None or self.headers_received_at is None:
            return None
        return round((self.first_token_at - self.headers_received_at) * 1000, 6)

    @computed_field
    @property
    def inter_token_latencies_ms(self) -> list[float]:
        if self.stream_timing_valid:
            timestamps = self.stream_event_timestamps
        elif self.token_timing_exact:
            # Backward compatibility for measurements written before stream-event
            # timing was represented explicitly.
            timestamps = self.token_timestamps
        else:
            return []
        return [
            round((current - previous) * 1000, 6)
            for previous, current in pairwise(timestamps)
        ]

    @computed_field
    @property
    def tpot_ms(self) -> float | None:
        """Average post-first-token time per generated token."""
        if self.first_token_at is None or self.output_tokens <= 1:
            return None
        return round(
            (self.completed_at - self.first_token_at) * 1000 / (self.output_tokens - 1),
            6,
        )

    @computed_field
    @property
    def e2e_latency_ms(self) -> float:
        return round((self.completed_at - self.started_at) * 1000, 6)


class JsonlWriter:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, measurement: RequestMeasurement) -> None:
        payload = measurement.model_dump_json(exclude_computed_fields=True) + "\n"
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, payload.encode())
        finally:
            os.close(descriptor)


def read_jsonl(path: Path | str) -> list[RequestMeasurement]:
    with Path(path).open(encoding="utf-8") as handle:
        return [RequestMeasurement.model_validate_json(line) for line in handle if line.strip()]


def write_parquet(jsonl_path: Path | str, parquet_path: Path | str) -> None:
    lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    output = Path(parquet_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(output, index=False)
