"""Typed configuration shared by the router, load generator, and experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

WorkloadName = Literal["short", "long-prefill", "shared-prefix", "bursty"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackendConfig(StrictModel):
    urls: list[str] = Field(default_factory=lambda: ["http://vllm:8000"], min_length=1)
    request_timeout_seconds: float = Field(default=300.0, gt=0)


class RouterConfig(StrictModel):
    policy: Literal["fifo", "slo"] = "fifo"
    allow_policy_override: bool = False
    ttft_slo_ms: int = Field(default=1000, gt=0)
    max_delay_ms: int = Field(default=250, ge=0)
    queue_limit: int = Field(default=128, gt=0)
    prompt_length_cutoff: int = Field(default=2048, gt=0)
    kv_soft_limit: float = Field(default=0.80, ge=0, le=1)
    kv_hard_limit: float = Field(default=0.95, ge=0, le=1)

    @model_validator(mode="after")
    def validate_kv_limits(self) -> RouterConfig:
        if self.kv_soft_limit >= self.kv_hard_limit:
            raise ValueError("kv_soft_limit must be lower than kv_hard_limit")
        return self


class WorkloadConfig(StrictModel):
    seed: int = 20260914
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    mixed_weights: dict[WorkloadName, float] = Field(
        default_factory=lambda: {
            "short": 0.40,
            "long-prefill": 0.20,
            "shared-prefix": 0.20,
            "bursty": 0.20,
        }
    )

    @model_validator(mode="after")
    def validate_mixed_weights(self) -> WorkloadConfig:
        if not self.mixed_weights:
            raise ValueError("mixed_weights cannot be empty")
        if any(weight < 0 for weight in self.mixed_weights.values()):
            raise ValueError("mixed_weights cannot be negative")
        if abs(sum(self.mixed_weights.values()) - 1.0) > 1e-9:
            raise ValueError("mixed_weights must sum to 1")
        return self


class ExperimentConfig(StrictModel):
    repeats: int = Field(default=3, ge=3)
    warmup_requests: int = Field(default=8, ge=0)
    throughput_floor_ratio: float = Field(default=0.95, gt=0, le=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)


class BenchConfig(StrictModel):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    backend: BackendConfig = Field(default_factory=BackendConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    workloads: WorkloadConfig = Field(default_factory=WorkloadConfig)
    experiments: ExperimentConfig = Field(default_factory=ExperimentConfig)


def load_config(path: Path | str) -> BenchConfig:
    """Load YAML configuration and apply the documented environment overrides."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    if model := os.getenv("MODEL"):
        raw["model"] = model
    if backend_urls := os.getenv("VLLM_BASE_URL"):
        backend = raw.setdefault("backend", {})
        backend["urls"] = [
            url.strip().rstrip("/") for url in backend_urls.split(",") if url.strip()
        ]
    if policy := os.getenv("ROUTER_POLICY"):
        raw.setdefault("router", {})["policy"] = policy
    if override := os.getenv("ALLOW_POLICY_OVERRIDE"):
        enabled = override.lower() in {"1", "true", "yes"}
        raw.setdefault("router", {})["allow_policy_override"] = enabled
    if seed := os.getenv("SERVEBENCH_SEED"):
        raw.setdefault("workloads", {})["seed"] = int(seed)

    return BenchConfig.model_validate(raw)
