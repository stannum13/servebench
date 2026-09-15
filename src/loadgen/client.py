"""OpenAI SSE load client with request-level timing."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from servebench.results import JsonlWriter, RequestMeasurement
from servebench.workloads import RequestSpec


@dataclass(frozen=True, slots=True)
class LoadOptions:
    run_id: str
    url: str
    policy: str
    concurrency: int
    timeout_seconds: float = 300.0
    model: str = field(default_factory=lambda: os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    api_key: str | None = None


async def _request(
    spec: RequestSpec,
    options: LoadOptions,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    scheduled_at: float,
) -> RequestMeasurement:
    token_times: list[float] = []
    prompt_tokens = spec.prompt_tokens
    output_tokens = 0
    token_ids_seen = False
    first_token: float | None = None
    headers_received_at: float | None = None
    router_admission_ms: float | None = None
    worker: str | None = None
    kv_cache_usage: float | None = None
    started = time.monotonic()
    status = "ok"
    error: str | None = None
    http_status_code: int | None = None
    async with semaphore:
        started = time.monotonic()
        try:
            async with asyncio.timeout(options.timeout_seconds):
                async with client.stream(
                    "POST",
                    options.url.rstrip("/") + "/v1/completions",
                    headers={
                        "X-Servebench-Policy": options.policy,
                        **(
                            {"Authorization": f"Bearer {options.api_key}"}
                            if options.api_key
                            else {}
                        ),
                    },
                    json={
                        "model": options.model,
                        "prompt": spec.prompt,
                        "max_tokens": spec.max_tokens,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "return_token_ids": True,
                        "ignore_eos": True,
                        "temperature": 0,
                    },
                ) as response:
                    headers_received_at = time.monotonic()
                    http_status_code = response.status_code
                    worker = response.headers.get("X-Servebench-Worker")
                    if admission_header := response.headers.get("X-Servebench-Admission-Ms"):
                        router_admission_ms = float(admission_header)
                    if kv_header := response.headers.get("X-Servebench-KV-Usage"):
                        kv_cache_usage = float(kv_header)
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        payload = json.loads(line[6:])
                        usage = payload.get("usage") or {}
                        if usage.get("prompt_tokens") is not None:
                            prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            output_tokens = int(usage["completion_tokens"])
                        choices = payload.get("choices") or []
                        choice = choices[0] if choices else {}
                        prompt_ids = choice.get("prompt_token_ids")
                        if isinstance(prompt_ids, list) and not usage:
                            prompt_tokens = len(prompt_ids)
                        token_ids = choice.get("token_ids")
                        if isinstance(token_ids, list):
                            token_ids_seen = True
                            now = time.monotonic()
                            if token_ids:
                                first_token = first_token or now
                                token_times.extend([now] * len(token_ids))
                                if not usage:
                                    output_tokens += len(token_ids)
                        elif choice.get("text"):
                            first_token = first_token or time.monotonic()
        except TimeoutError:
            status, error = "timeout", "request timeout"
        except httpx.HTTPStatusError as exc:
            status = "rejected" if exc.response.status_code == 429 else "error"
            error = str(exc)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            status, error = "error", str(exc)
    completed = time.monotonic()
    return RequestMeasurement(
        run_id=options.run_id,
        request_id=spec.request_id,
        workload=spec.kind,
        policy=options.policy,
        prompt_tokens=prompt_tokens,
        nominal_prompt_tokens=spec.prompt_tokens,
        output_tokens=output_tokens,
        scheduled_at=scheduled_at,
        started_at=started,
        first_token_at=first_token,
        headers_received_at=headers_received_at,
        token_timestamps=token_times,
        token_timing_exact=token_ids_seen and len(token_times) == output_tokens,
        completed_at=completed,
        status=status,
        http_status_code=http_status_code,
        error=error,
        queue_time_ms=round((started - scheduled_at) * 1000, 6),
        client_queue_time_ms=round((started - scheduled_at) * 1000, 6),
        router_admission_ms=router_admission_ms,
        worker=worker,
        kv_cache_usage=kv_cache_usage,
    )


async def run_load(
    requests: list[RequestSpec],
    options: LoadOptions,
    output_path: Path,
    *,
    client: httpx.AsyncClient | None = None,
    arrival_offsets: list[float] | None = None,
) -> list[RequestMeasurement]:
    if options.concurrency < 1:
        raise ValueError("concurrency must be positive")
    own_client = client is None
    active_client = client or httpx.AsyncClient()
    semaphore = asyncio.Semaphore(options.concurrency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    writer = JsonlWriter(output_path)
    offsets = arrival_offsets or [0.0] * len(requests)
    if len(offsets) != len(requests) or any(value < 0 for value in offsets):
        raise ValueError("arrival_offsets must contain one non-negative value per request")
    run_started = time.monotonic()

    async def scheduled_request(spec: RequestSpec, offset: float) -> RequestMeasurement:
        scheduled_at = run_started + offset
        await asyncio.sleep(max(0, scheduled_at - time.monotonic()))
        return await _request(spec, options, active_client, semaphore, scheduled_at)

    try:
        tasks = [
            asyncio.create_task(scheduled_request(spec, offset))
            for spec, offset in zip(requests, offsets, strict=True)
        ]
        results = await asyncio.gather(*tasks)
        for result in results:
            writer.write(result)
        return results
    finally:
        if own_client:
            await active_client.aclose()
