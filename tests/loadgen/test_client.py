import asyncio
from pathlib import Path

import httpx

from loadgen.client import LoadOptions, run_load
from mock_backend.app import create_app
from servebench.workloads import WorkloadGenerator


async def test_streaming_client_records_tokens_and_latency(tmp_path: Path) -> None:
    output = tmp_path / "requests.jsonl"
    output.write_text("stale\n", encoding="utf-8")
    transport = httpx.ASGITransport(app=create_app(token_delay=0))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 2, 3),
            LoadOptions(run_id="run", url="http://test", policy="fifo", concurrency=2),
            output, client=client,
        )
    assert len(result) == 2
    assert all(item.status == "ok" for item in result)
    assert all(item.output_tokens == 128 for item in result)
    assert all(item.ttft_ms is not None for item in result)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


async def test_concurrency_is_bounded(tmp_path: Path) -> None:
    active = 0
    maximum = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(500, text="failure")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 6, 1),
            LoadOptions(run_id="run", url="http://test", policy="fifo", concurrency=2),
            tmp_path / "failed.jsonl", client=client,
        )
    assert maximum == 2
    assert all(item.status == "error" for item in result)


async def test_open_loop_offsets_are_preserved_in_measurements(tmp_path: Path) -> None:
    transport = httpx.ASGITransport(app=create_app(token_delay=0))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 2, 5),
            LoadOptions(run_id="run", url="http://test", policy="fifo", concurrency=2),
            tmp_path / "open-loop.jsonl",
            client=client,
            arrival_offsets=[0.0, 0.02],
        )
    assert round(result[1].scheduled_at - result[0].scheduled_at, 2) == 0.02


async def test_router_overload_is_recorded_as_rejection(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "1"},
            json={"error": {"type": "overload", "message": "KV hard limit"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 1, 1),
            LoadOptions(run_id="run", url="http://test", policy="slo", concurrency=1),
            tmp_path / "rejected.jsonl", client=client,
        )
    assert result[0].status == "rejected"
    assert result[0].http_status_code == 429


def test_load_options_use_configured_model(monkeypatch) -> None:
    monkeypatch.setenv("MODEL", "custom/model")
    options = LoadOptions(run_id="r", url="http://test", policy="fifo", concurrency=1)
    assert options.model == "custom/model"


async def test_streamed_token_ids_and_usage_drive_actual_token_measurements(tmp_path: Path) -> None:
    events = (
        'data: {"choices":[{"text":"hello","token_ids":[11,12],'
        '"prompt_token_ids":[1,2]}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":512,"completion_tokens":2}}\n\n'
        'data: [DONE]\n\n'
    )
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.read().decode().find('"return_token_ids":true') >= 0
        return httpx.Response(200, text=events, headers={"Content-Type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 1, 1),
            LoadOptions(run_id="run", url="http://test", policy="fifo", concurrency=1),
            tmp_path / "tokens.jsonl", client=client,
        )
    item = result[0]
    assert item.prompt_tokens == 512
    assert item.nominal_prompt_tokens == 256
    assert item.output_tokens == 2
    assert item.token_timing_exact is True
    assert len(item.token_timestamps) == 2


async def test_text_only_stream_does_not_claim_inter_token_timing(tmp_path: Path) -> None:
    events = (
        'data: {"choices":[{"text":"hello world"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":256,"completion_tokens":2}}\n\n'
        'data: [DONE]\n\n'
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text=events, headers={"Content-Type": "text/event-stream"}
        ))
    ) as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 1, 1),
            LoadOptions(run_id="run", url="http://test", policy="fifo", concurrency=1),
            tmp_path / "text-only.jsonl", client=client,
        )
    assert result[0].token_timing_exact is False
    assert result[0].inter_token_latencies_ms == []


async def test_loadgen_records_router_admission_and_post_header_ttft(tmp_path: Path) -> None:
    events = (
        'data: {"choices":[{"text":"a","token_ids":[11]}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":256,"completion_tokens":1}}\n\n'
        'data: [DONE]\n\n'
    )
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=events, headers={
            "Content-Type": "text/event-stream",
            "X-Servebench-Admission-Ms": "12.5",
            "X-Servebench-Worker": "worker-1",
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_load(
            WorkloadGenerator().generate("short", 1, 1),
            LoadOptions(run_id="run", url="http://test", policy="fifo", concurrency=1),
            tmp_path / "stages.jsonl", client=client,
        )
    item = result[0]
    assert item.router_admission_ms == 12.5
    assert item.worker == "worker-1"
    assert item.client_queue_time_ms is not None
    assert item.post_header_ttft_ms is not None
