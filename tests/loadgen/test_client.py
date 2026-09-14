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
